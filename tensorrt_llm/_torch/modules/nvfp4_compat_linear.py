# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Load-time compatibility shims for NVFP4 checkpoints on pre-Blackwell GPUs.

NVFP4 checkpoints store weights in FP4 E2M1 format with FP8 E4M3 per-group
scales (group_size=16). These kernels require Blackwell (SM100+) native FP4
tensor cores and cannot run directly on H100 (SM90) or A100 (SM80).

This module provides two LinearMethod implementations that dequantize weights
at model load time (once, not at every forward pass), storing them in a wider
dtype that the target GPU can execute natively:

  NVFP4ToFP8RowwiseLinearMethod  — target: H100 (SM90+)
    Dequants FP4 → BF16 → re-quantizes to FP8 per output channel.
    Inference uses fp8_rowwise_gemm (wgmma FP8 tensor cores, native on H100).
    Memory: 1 byte/weight (same as FP8), 2x larger than the FP4 checkpoint.

  NVFP4ToBF16LinearMethod        — target: A100 (SM80) and older
    Dequants FP4 → BF16 with group scales fully absorbed into the weights.
    Inference uses standard BF16 matmul.
    Memory: 2 bytes/weight, 4x larger than the FP4 checkpoint.

Usage (via QuantConfig):
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4_TO_FP8_ROWWISE)  # H100
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4_TO_BF16)          # A100
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tensorrt_llm._torch.modules.linear import LinearMethodBase, load_weight_shard

# FP8 E4M3FN max representable value
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# NVFP4 group size: one FP8 scale covers this many consecutive weights per row
_NVFP4_GROUP_SIZE = 16


# ---------------------------------------------------------------------------
# Core dequantisation helpers
# ---------------------------------------------------------------------------


def _fp4_e2m1_nibbles_to_bf16(nibbles: torch.Tensor) -> torch.Tensor:
    """Convert a tensor of packed FP4 E2M1 nibble values (int, range 0-15) to BF16.

    Bit layout of each nibble:  [sign | exp1 | exp0 | mant]
        sign : 1 bit
        exp  : 2 bits, biased by 1  (i.e. stored_exp = true_exp + 1)
        mant : 1 bit

    Normal  (exp > 0): value = (-1)^sign * 2^(exp-1) * (1 + mant/2)
    Subnorm (exp == 0): value = (-1)^sign * 0.5 * mant
    """
    n = nibbles.int()
    sign = (n >> 3) & 1
    exp = (n >> 1) & 0x3
    mant = n & 0x1

    sign_f = torch.where(
        sign == 1,
        torch.full_like(n, -1, dtype=torch.bfloat16),
        torch.ones_like(n, dtype=torch.bfloat16),
    )
    exp_f = (2.0 ** (exp.bfloat16() - 1)).clamp(min=0.5)
    mant_f = 1.0 + mant.bfloat16() * 0.5

    normal = sign_f * exp_f * mant_f
    subnorm = sign_f * 0.5 * mant.bfloat16()
    return torch.where(exp == 0, subnorm, normal)


def _unpack_fp4_packed_to_bf16(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor (2x FP4 per byte) to BF16.

    Args:
        packed: uint8 tensor, shape (..., in_features // 2).
                Low nibble = first weight, high nibble = second weight.

    Returns:
        BF16 tensor, shape (..., in_features).
    """
    lo = packed & 0x0F  # lower nibble — first weight of each pair
    hi = (packed >> 4) & 0x0F  # upper nibble — second weight of each pair

    lo_bf16 = _fp4_e2m1_nibbles_to_bf16(lo)
    hi_bf16 = _fp4_e2m1_nibbles_to_bf16(hi)

    # Interleave: [lo0, hi0, lo1, hi1, ...] along the last dimension
    result = torch.stack([lo_bf16, hi_bf16], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _dequant_nvfp4_chunk_to_bf16(
    packed_chunk: torch.Tensor,  # (chunk_rows, in_features // 2) uint8
    scales_chunk: torch.Tensor,  # (chunk_rows, in_features // group_size) float8_e4m3fn
    group_size: int,
) -> torch.Tensor:
    """Dequantize a row-chunk of NVFP4 weights to BF16.

    Keeps everything in BF16 to minimise peak memory.  The FP8 group scales
    are upcast to BF16 before multiplication.

    Returns: (chunk_rows, in_features) BF16
    """
    w_bf16 = _unpack_fp4_packed_to_bf16(packed_chunk)  # (chunk, in)
    s_bf16 = scales_chunk.bfloat16()  # (chunk, in//group_size)
    s_expanded = s_bf16.repeat_interleave(group_size, dim=1)  # (chunk, in)
    w_bf16 *= s_expanded  # in-place; no new alloc
    return w_bf16


# ---------------------------------------------------------------------------
# Conversion entry points
# ---------------------------------------------------------------------------


def nvfp4_to_fp8_rowwise(
    packed: torch.Tensor,  # (out, in//2) uint8
    scales: torch.Tensor,  # (out, in//group_size) float8_e4m3fn
    group_size: int = _NVFP4_GROUP_SIZE,
    chunk_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NVFP4 weights to FP8 E4M3FN with per-output-channel scales.

    Suitable for H100 (SM90+) fp8_rowwise_gemm.

    Peak memory per chunk: chunk_rows * in_features * 2 bytes (BF16 intermediate).
    The final weight is 1 byte/weight (FP8).

    Returns:
        w_fp8:        (out, in) float8_e4m3fn
        weight_scale: (out,)   float32  — per output channel
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2

    w_fp8_out = torch.empty(
        out_features, in_features, dtype=torch.float8_e4m3fn, device=packed.device
    )
    w_scale_out = torch.empty(out_features, dtype=torch.float32, device=packed.device)

    for row_start in range(0, out_features, chunk_rows):
        row_end = min(row_start + chunk_rows, out_features)

        w_bf16 = _dequant_nvfp4_chunk_to_bf16(
            packed[row_start:row_end],
            scales[row_start:row_end],
            group_size,
        )  # (chunk, in) BF16

        # Per-channel max → FP32 scale → cast to FP8
        amax = w_bf16.abs().amax(dim=1, keepdim=True).float()  # (chunk, 1)
        scale = (amax / _FP8_MAX).clamp(min=1e-12)  # (chunk, 1)

        w_fp8_out[row_start:row_end] = (
            (w_bf16 / scale.bfloat16()).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
        w_scale_out[row_start:row_end] = scale.squeeze(1)

        del w_bf16
        torch.cuda.empty_cache()

    return w_fp8_out, w_scale_out


def nvfp4_to_bf16(
    packed: torch.Tensor,  # (out, in//2) uint8
    scales: torch.Tensor,  # (out, in//group_size) float8_e4m3fn
    group_size: int = _NVFP4_GROUP_SIZE,
    chunk_rows: int = 512,
) -> torch.Tensor:
    """Convert NVFP4 weights to BF16 with group scales fully absorbed.

    Suitable for A100 (SM80) standard BF16 matmul.  No separate weight_scale
    is stored; the group-level quantisation error is absorbed into the weights.

    Peak memory per chunk: chunk_rows * in_features * 2 bytes (BF16 intermediate).
    Final weight is 2 bytes/weight (BF16).

    Returns:
        w_bf16: (out, in) bfloat16
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2

    w_bf16_out = torch.empty(out_features, in_features, dtype=torch.bfloat16, device=packed.device)

    for row_start in range(0, out_features, chunk_rows):
        row_end = min(row_start + chunk_rows, out_features)

        w_bf16_out[row_start:row_end] = _dequant_nvfp4_chunk_to_bf16(
            packed[row_start:row_end],
            scales[row_start:row_end],
            group_size,
        )
        torch.cuda.empty_cache()

    return w_bf16_out


# ---------------------------------------------------------------------------
# Internal weight-loading helpers shared by both methods
# ---------------------------------------------------------------------------


def _load_nvfp4_raw(
    weights: List[Dict],
    tp_size: int,
    tp_rank: int,
    tp_mode,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and TP-shard the raw packed FP4 weight and FP8 group scales.

    Returns:
        packed: (out, in//2) uint8
        scales: (out, in//group_size) float8_e4m3fn
    """
    assert len(weights) == 1, "Expected a single weight dict for vanilla load"
    w = weights[0]

    packed = load_weight_shard(w["weight"], tp_size, tp_rank, tp_mode, device).contiguous()
    # weight is stored as float4_e2m1x2 (uint8 alias) — view as uint8
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)

    scales = load_weight_shard(w["weight_scale"], tp_size, tp_rank, tp_mode, device).contiguous()
    # scales are stored as float8_e4m3fn in the NVFP4 checkpoint
    if scales.dtype != torch.float8_e4m3fn:
        scales = scales.view(torch.float8_e4m3fn)

    return packed, scales


def _load_nvfp4_raw_fused(
    weights: List[Dict],
    tp_size: int,
    tp_rank: int,
    tp_mode,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate FP4 packed weights and FP8 scales from multiple shards.

    Used for fused QKV and fused gate-up projections.

    Returns:
        packed: (sum_out, in//2) uint8
        scales: (sum_out, in//group_size) float8_e4m3fn
    """
    packed_list, scales_list = [], []
    for w in weights:
        p = load_weight_shard(w["weight"], tp_size, tp_rank, tp_mode, device).contiguous()
        if p.dtype != torch.uint8:
            p = p.view(torch.uint8)
        packed_list.append(p)

        s = load_weight_shard(w["weight_scale"], tp_size, tp_rank, tp_mode, device).contiguous()
        if s.dtype != torch.float8_e4m3fn:
            s = s.view(torch.float8_e4m3fn)
        scales_list.append(s)

    return torch.cat(packed_list, dim=0), torch.cat(scales_list, dim=0)


# ---------------------------------------------------------------------------
# NVFP4 → FP8 Rowwise  (H100 / SM90+)
# ---------------------------------------------------------------------------


class NVFP4ToFP8RowwiseLinearMethod(LinearMethodBase):
    """Load NVFP4 checkpoint weights as FP8 rowwise for inference on H100.

    At model load time, FP4 weights are dequantised (via BF16 intermediate)
    and re-quantised to FP8 per output channel.  Inference then uses
    fp8_rowwise_gemm with native wgmma FP8 tensor cores.

    Memory overhead vs the original checkpoint: 2x (1 byte/weight instead of
    0.5 bytes/weight).
    """

    def create_weights(
        self, module, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, **kwargs
    ):
        module.weight = Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        # Per output-channel scale consumed by fp8_rowwise_gemm
        module.weight_scale = Parameter(
            torch.empty(out_features, dtype=torch.float32),
            requires_grad=False,
        )
        # Input scale is dynamic (computed per-token at runtime)
        module.input_scale = None

        if bias:
            module.bias = Parameter(
                torch.empty(out_features, dtype=dtype),
                requires_grad=False,
            )
        else:
            module.register_parameter("bias", None)

    def apply(self, module, input: torch.Tensor, bias: Optional[torch.Tensor]):
        if input.dim() > 2:
            original_shape = input.shape
            input = input.reshape(-1, input.shape[-1])
        else:
            original_shape = None

        # Dynamic per-token activation quantisation
        x_fp8, x_scale = torch.ops.trtllm.quantize_e4m3_activation(input)

        output = torch.ops.trtllm.fp8_rowwise_gemm(
            x_fp8,
            module.weight,
            x_scale.float(),  # (M,) per-token
            module.weight_scale,  # (out,) per-channel
            input.dtype,
        )

        if original_shape is not None:
            output = output.reshape(*original_shape[:-1], output.shape[-1])

        if bias is not None:
            output = output + bias
        return output

    def load_weights_vanilla(
        self, module, weights: List[Dict], allow_partial_loading: bool = False
    ) -> None:
        device = module.weight.device
        packed, scales = _load_nvfp4_raw(
            weights, module.tp_size, module.tp_rank, module.tp_mode, device
        )

        w_fp8, w_scale = nvfp4_to_fp8_rowwise(packed, scales)
        module.weight.data.copy_(w_fp8)
        module.weight_scale.data.copy_(w_scale)

        if module.bias is not None and "bias" in weights[0]:
            module.bias.data.copy_(weights[0]["bias"].to(module.bias.dtype))

    def load_weights_fused_qkv_linear(
        self, module, weights: List[Dict], allow_partial_loading: bool = False
    ) -> None:
        device = module.weight.device
        packed, scales = _load_nvfp4_raw_fused(
            weights, module.tp_size, module.tp_rank, module.tp_mode, device
        )

        w_fp8, w_scale = nvfp4_to_fp8_rowwise(packed, scales)
        module.weight.data.copy_(w_fp8)
        module.weight_scale.data.copy_(w_scale)

    def load_weights_fused_gate_up_linear(
        self, module, weights: List[Dict], allow_partial_loading: bool = False
    ) -> None:
        device = module.weight.device
        packed, scales = _load_nvfp4_raw_fused(
            weights, module.tp_size, module.tp_rank, module.tp_mode, device
        )

        w_fp8, w_scale = nvfp4_to_fp8_rowwise(packed, scales)
        module.weight.data.copy_(w_fp8)
        module.weight_scale.data.copy_(w_scale)


# ---------------------------------------------------------------------------
# NVFP4 → BF16  (A100 / SM80 and older)
# ---------------------------------------------------------------------------


class NVFP4ToBF16LinearMethod(LinearMethodBase):
    """Load NVFP4 checkpoint weights as BF16 for inference on A100 or older.

    At model load time, FP4 weights are dequantised to BF16 with per-group
    FP8 scales fully absorbed.  Inference uses standard BF16 matmul (cuBLAS).

    Memory overhead vs the original checkpoint: 4x (2 bytes/weight instead of
    0.5 bytes/weight).
    """

    def create_weights(
        self, module, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, **kwargs
    ):
        module.weight = Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        if bias:
            module.bias = Parameter(
                torch.empty(out_features, dtype=dtype),
                requires_grad=False,
            )
        else:
            module.register_parameter("bias", None)

    def apply(self, module, input: torch.Tensor, bias: Optional[torch.Tensor]):
        return F.linear(input.to(module.weight.dtype), module.weight, bias)

    def load_weights_vanilla(
        self, module, weights: List[Dict], allow_partial_loading: bool = False
    ) -> None:
        device = module.weight.device
        packed, scales = _load_nvfp4_raw(
            weights, module.tp_size, module.tp_rank, module.tp_mode, device
        )

        module.weight.data.copy_(nvfp4_to_bf16(packed, scales))

        if module.bias is not None and "bias" in weights[0]:
            module.bias.data.copy_(weights[0]["bias"].to(module.bias.dtype))

    def load_weights_fused_qkv_linear(
        self, module, weights: List[Dict], allow_partial_loading: bool = False
    ) -> None:
        device = module.weight.device
        packed, scales = _load_nvfp4_raw_fused(
            weights, module.tp_size, module.tp_rank, module.tp_mode, device
        )

        module.weight.data.copy_(nvfp4_to_bf16(packed, scales))

    def load_weights_fused_gate_up_linear(
        self, module, weights: List[Dict], allow_partial_loading: bool = False
    ) -> None:
        device = module.weight.device
        packed, scales = _load_nvfp4_raw_fused(
            weights, module.tp_size, module.tp_rank, module.tp_mode, device
        )

        module.weight.data.copy_(nvfp4_to_bf16(packed, scales))
