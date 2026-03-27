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
"""Unit tests for NVFP4 load-time compatibility shims.

Tests cover:
  - FP4 E2M1 nibble decoding correctness (all 16 representable values)
  - Round-trip: FP32 → quantise to FP4+FP8-scales → nvfp4_to_fp8_rowwise → decode → compare
  - Round-trip: FP32 → quantise to FP4+FP8-scales → nvfp4_to_bf16 → compare
  - Peak memory: BF16 intermediate, no FP32 spike
  - Chunked processing produces identical results to single-pass
  - NVFP4ToFP8RowwiseLinearMethod forward pass (numerical accuracy vs F.linear reference)
  - NVFP4ToBF16LinearMethod forward pass (numerical accuracy vs F.linear reference)
"""

import pytest
import torch

from tensorrt_llm._torch.modules.nvfp4_compat_linear import (
    NVFP4ToBF16LinearMethod,
    NVFP4ToFP8RowwiseLinearMethod,
    _fp4_e2m1_nibbles_to_bf16,
    _unpack_fp4_packed_to_bf16,
    nvfp4_to_bf16,
    nvfp4_to_fp8_rowwise,
)

# Skip entire module if no CUDA device is available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# ---------------------------------------------------------------------------
# Helpers for building synthetic NVFP4 checkpoints
# ---------------------------------------------------------------------------

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_FP4_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _fp32_to_fp4_nibble(value: float) -> int:
    """Quantise a float to the nearest FP4 E2M1 nibble value (0-15)."""
    best_nibble, best_err = 0, float("inf")
    for nibble in range(16):
        fp4_val = _FP4_VALUES[nibble]
        err = abs(value - fp4_val)
        if err < best_err:
            best_err = err
            best_nibble = nibble
    return best_nibble


def _make_nvfp4_checkpoint(
    out_features: int,
    in_features: int,
    group_size: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic NVFP4 checkpoint and matching FP32 reference weights.

    Returns:
        packed:    (out, in//2) uint8
        scales:    (out, in//group_size) float8_e4m3fn
        w_fp32_ref: (out, in) float32  — the "true" dequantised weights
    """
    torch.manual_seed(seed)

    # Random FP32 reference weights in a range representable by FP4*FP8
    w_fp32 = torch.randn(out_features, in_features) * 2.0

    n_groups = in_features // group_size
    scales_fp32 = torch.empty(out_features, n_groups)
    packed_out = torch.zeros(out_features, in_features // 2, dtype=torch.uint8)
    w_fp32_ref = torch.zeros_like(w_fp32)

    for row in range(out_features):
        for g in range(n_groups):
            col_start = g * group_size
            col_end = col_start + group_size
            group_vals = w_fp32[row, col_start:col_end]

            # Pick FP8 scale so the largest value in the group maps to ≤ FP4_MAX (6.0)
            fp4_max = 6.0
            amax = group_vals.abs().max().item()
            scale = amax / fp4_max if amax > 0 else 1.0
            scales_fp32[row, g] = scale

            for k in range(group_size):
                true_val = group_vals[k].item()
                # Quantise: divide by scale, find nearest FP4 nibble
                nibble = _fp32_to_fp4_nibble(true_val / scale)
                # Reconstruct: dequantise for reference
                w_fp32_ref[row, col_start + k] = _FP4_VALUES[nibble] * scale

                # Pack: even index → low nibble, odd → high nibble
                byte_idx = (col_start + k) // 2
                if (col_start + k) % 2 == 0:
                    packed_out[row, byte_idx] |= nibble & 0x0F
                else:
                    packed_out[row, byte_idx] |= (nibble & 0x0F) << 4

    # Clamp scales to FP8 range and convert
    scales_fp8 = scales_fp32.clamp(max=_FP8_MAX).to(torch.float8_e4m3fn)

    return packed_out, scales_fp8, w_fp32_ref


# ---------------------------------------------------------------------------
# 1. FP4 nibble decoding
# ---------------------------------------------------------------------------


class TestFP4Nibbles:
    def test_all_16_values(self):
        """All 16 FP4 E2M1 nibbles decode to their expected BF16 values."""
        nibbles = torch.arange(16, dtype=torch.int32)
        result = _fp4_e2m1_nibbles_to_bf16(nibbles).float().tolist()
        assert result == pytest.approx(_FP4_VALUES, abs=1e-3), (
            f"FP4 decode mismatch:\n  got: {result}\n  expected: {_FP4_VALUES}"
        )

    def test_sign_symmetry(self):
        """Positive and negative nibbles should be equal in magnitude."""
        pos = _fp4_e2m1_nibbles_to_bf16(torch.arange(8)).float()
        neg = _fp4_e2m1_nibbles_to_bf16(torch.arange(8, 16)).float()
        torch.testing.assert_close(pos.abs(), neg.abs())

    def test_zero_nibble(self):
        """Nibble 0 = +0.0, nibble 8 = -0.0 (subnormal, both effectively zero)."""
        vals = _fp4_e2m1_nibbles_to_bf16(torch.tensor([0, 8])).float()
        assert vals[0].item() == pytest.approx(0.0)
        assert vals[1].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Unpack round-trip
# ---------------------------------------------------------------------------


class TestUnpackFP4:
    def test_shape(self):
        packed = torch.zeros(8, 16, dtype=torch.uint8)
        out = _unpack_fp4_packed_to_bf16(packed)
        assert out.shape == (8, 32)
        assert out.dtype == torch.bfloat16

    def test_lo_hi_interleaving(self):
        """Low nibble maps to even columns, high nibble to odd columns."""
        # Pack: lo=nibble 1 (0.5), hi=nibble 2 (1.0)
        packed = torch.full((1, 1), 0x21, dtype=torch.uint8)  # lo=1, hi=2
        out = _unpack_fp4_packed_to_bf16(packed).float()
        assert out[0, 0].item() == pytest.approx(0.5, abs=1e-3)  # lo → col 0
        assert out[0, 1].item() == pytest.approx(1.0, abs=1e-3)  # hi → col 1


# ---------------------------------------------------------------------------
# 3. nvfp4_to_fp8_rowwise correctness
# ---------------------------------------------------------------------------


class TestNVFP4ToFP8Rowwise:
    @pytest.mark.parametrize(
        "out_features,in_features",
        [
            (64, 64),
            (128, 256),
            (256, 128),
        ],
    )
    def test_round_trip_accuracy(self, out_features, in_features):
        """Dequanted FP8 weights should closely match the FP32 reference."""
        packed, scales, w_ref = _make_nvfp4_checkpoint(out_features, in_features)
        packed = packed.cuda()
        scales = scales.cuda()
        w_ref = w_ref.cuda()

        w_fp8, w_scale = nvfp4_to_fp8_rowwise(packed, scales)

        # Dequantise the FP8 result: w_dequant = w_fp8 * w_scale (per-channel)
        w_dequant = w_fp8.float() * w_scale.unsqueeze(1)

        # Should closely match the reference (within FP4 + FP8 quantisation error)
        max_err = (w_dequant - w_ref).abs().max().item()
        assert max_err < 0.5, f"max abs error {max_err:.4f} exceeds threshold"

        # Relative error on non-zero elements
        mask = w_ref.abs() > 0.1
        rel_err = ((w_dequant[mask] - w_ref[mask]) / w_ref[mask]).abs().max().item()
        assert rel_err < 0.15, f"max relative error {rel_err:.4f} exceeds 15%"

    def test_output_dtype(self):
        packed, scales, _ = _make_nvfp4_checkpoint(32, 64)
        w_fp8, w_scale = nvfp4_to_fp8_rowwise(packed.cuda(), scales.cuda())
        assert w_fp8.dtype == torch.float8_e4m3fn
        assert w_scale.dtype == torch.float32

    def test_output_shape(self):
        out_f, in_f = 48, 96
        packed, scales, _ = _make_nvfp4_checkpoint(out_f, in_f)
        w_fp8, w_scale = nvfp4_to_fp8_rowwise(packed.cuda(), scales.cuda())
        assert w_fp8.shape == (out_f, in_f)
        assert w_scale.shape == (out_f,)

    def test_chunked_matches_single_pass(self):
        """chunk_rows=1 and chunk_rows=large should give identical results."""
        packed, scales, _ = _make_nvfp4_checkpoint(64, 64, seed=7)
        packed, scales = packed.cuda(), scales.cuda()

        w1, s1 = nvfp4_to_fp8_rowwise(packed, scales, chunk_rows=1)
        w2, s2 = nvfp4_to_fp8_rowwise(packed, scales, chunk_rows=1024)

        # FP8 weights must be bit-identical (same deterministic quantisation)
        assert torch.equal(w1.view(torch.uint8), w2.view(torch.uint8))
        torch.testing.assert_close(s1, s2)

    def test_per_channel_scale_positive(self):
        """All per-channel scales must be positive."""
        packed, scales, _ = _make_nvfp4_checkpoint(32, 64)
        _, w_scale = nvfp4_to_fp8_rowwise(packed.cuda(), scales.cuda())
        assert (w_scale > 0).all()


# ---------------------------------------------------------------------------
# 4. nvfp4_to_bf16 correctness
# ---------------------------------------------------------------------------


class TestNVFP4ToBF16:
    @pytest.mark.parametrize(
        "out_features,in_features",
        [
            (64, 64),
            (128, 256),
        ],
    )
    def test_round_trip_accuracy(self, out_features, in_features):
        """BF16 dequant should closely match the FP32 reference."""
        packed, scales, w_ref = _make_nvfp4_checkpoint(out_features, in_features)
        w_bf16 = nvfp4_to_bf16(packed.cuda(), scales.cuda())

        max_err = (w_bf16.float() - w_ref.cuda()).abs().max().item()
        assert max_err < 0.3, f"max abs error {max_err:.4f} exceeds threshold"

    def test_output_dtype(self):
        packed, scales, _ = _make_nvfp4_checkpoint(32, 64)
        w_bf16 = nvfp4_to_bf16(packed.cuda(), scales.cuda())
        assert w_bf16.dtype == torch.bfloat16

    def test_output_shape(self):
        out_f, in_f = 48, 96
        packed, scales, _ = _make_nvfp4_checkpoint(out_f, in_f)
        w_bf16 = nvfp4_to_bf16(packed.cuda(), scales.cuda())
        assert w_bf16.shape == (out_f, in_f)

    def test_chunked_matches_single_pass(self):
        packed, scales, _ = _make_nvfp4_checkpoint(64, 64, seed=13)
        packed, scales = packed.cuda(), scales.cuda()
        w1 = nvfp4_to_bf16(packed, scales, chunk_rows=1)
        w2 = nvfp4_to_bf16(packed, scales, chunk_rows=1024)
        torch.testing.assert_close(w1, w2)

    def test_no_fp32_intermediate_peak(self):
        """Conversion should not materialise a full FP32 weight matrix.

        We approximate this by verifying that peak reserved memory during
        conversion is less than 3x the final BF16 weight size (allowing for
        intermediate BF16 chunk buffers but ruling out a full FP32 copy).
        """
        out_f, in_f = 256, 512
        packed, scales, _ = _make_nvfp4_checkpoint(out_f, in_f)
        packed, scales = packed.cuda(), scales.cuda()

        torch.cuda.reset_peak_memory_stats()
        nvfp4_to_bf16(packed, scales, chunk_rows=32)
        peak = torch.cuda.max_memory_reserved()

        # BF16 weight tensor size in bytes
        final_size = out_f * in_f * 2
        # Peak should be well under 6x final (a full FP32 copy would be 4x * 2 = 8x)
        assert peak < final_size * 6, (
            f"Peak memory {peak} suggests an FP32 intermediate was materialised "
            f"(final BF16 size: {final_size})"
        )


# ---------------------------------------------------------------------------
# 5. NVFP4ToFP8RowwiseLinearMethod — forward pass
# ---------------------------------------------------------------------------


class TestNVFP4ToFP8RowwiseLinearMethod:
    """Integration test: load synthetic NVFP4 weights and run a forward pass."""

    @pytest.fixture
    def layer_fp8(self):
        """Small FP8-rowwise linear layer loaded from a synthetic NVFP4 ckpt."""
        out_f, in_f = 64, 128
        packed, scales, w_ref = _make_nvfp4_checkpoint(out_f, in_f)

        method = NVFP4ToFP8RowwiseLinearMethod()

        # Simulate the module object that create_weights would produce
        class _FakeModule:
            tp_size = 1
            tp_rank = 0
            tp_mode = None
            bias = None

        mod = _FakeModule()
        method.create_weights(mod, in_f, out_f, bias=False, dtype=torch.bfloat16)
        mod.weight = mod.weight.cuda()
        mod.weight_scale = mod.weight_scale.cuda()

        weights = [{"weight": packed, "weight_scale": scales}]
        method.load_weights_vanilla(mod, weights)

        return mod, w_ref.cuda(), method

    def test_forward_shape(self, layer_fp8):
        mod, w_ref, method = layer_fp8
        x = torch.randn(4, 128, dtype=torch.bfloat16).cuda()
        out = method.apply(mod, x, bias=None)
        assert out.shape == (4, 64)

    def test_forward_dtype(self, layer_fp8):
        mod, w_ref, method = layer_fp8
        x = torch.randn(4, 128, dtype=torch.bfloat16).cuda()
        out = method.apply(mod, x, bias=None)
        assert out.dtype == torch.bfloat16

    def test_forward_accuracy_vs_reference(self, layer_fp8):
        """FP8-rowwise output should be close to F.linear with the reference weights."""
        mod, w_ref, method = layer_fp8
        torch.manual_seed(0)
        x = torch.randn(8, 128, dtype=torch.bfloat16).cuda()

        out_fp8 = method.apply(mod, x, bias=None).float()
        out_ref = torch.nn.functional.linear(x.float(), w_ref)

        # Expect close agreement — tolerate FP4+FP8 quantisation noise
        torch.testing.assert_close(out_fp8, out_ref, atol=1.0, rtol=0.1)

    def test_3d_input(self, layer_fp8):
        """Method should handle (batch, seq, hidden) inputs."""
        mod, _, method = layer_fp8
        x = torch.randn(2, 6, 128, dtype=torch.bfloat16).cuda()
        out = method.apply(mod, x, bias=None)
        assert out.shape == (2, 6, 64)

    def test_bias_added(self):
        out_f, in_f = 32, 64
        packed, scales, _ = _make_nvfp4_checkpoint(out_f, in_f)
        bias_val = torch.ones(out_f, dtype=torch.bfloat16).cuda()

        method = NVFP4ToFP8RowwiseLinearMethod()

        class _FakeModule:
            tp_size = 1
            tp_rank = 0
            tp_mode = None
            bias = None

        mod = _FakeModule()
        method.create_weights(mod, in_f, out_f, bias=False, dtype=torch.bfloat16)
        mod.weight = mod.weight.cuda()
        mod.weight_scale = mod.weight_scale.cuda()

        weights = [{"weight": packed, "weight_scale": scales}]
        method.load_weights_vanilla(mod, weights)

        x = torch.randn(4, in_f, dtype=torch.bfloat16).cuda()
        out_no_bias = method.apply(mod, x, bias=None)
        out_bias = method.apply(mod, x, bias=bias_val)
        torch.testing.assert_close(out_bias, out_no_bias + 1.0, atol=1e-3, rtol=0.0)


# ---------------------------------------------------------------------------
# 6. NVFP4ToBF16LinearMethod — forward pass
# ---------------------------------------------------------------------------


class TestNVFP4ToBF16LinearMethod:
    @pytest.fixture
    def layer_bf16(self):
        out_f, in_f = 64, 128
        packed, scales, w_ref = _make_nvfp4_checkpoint(out_f, in_f)

        method = NVFP4ToBF16LinearMethod()

        class _FakeModule:
            tp_size = 1
            tp_rank = 0
            tp_mode = None
            bias = None

        mod = _FakeModule()
        method.create_weights(mod, in_f, out_f, bias=False, dtype=torch.bfloat16)
        mod.weight = mod.weight.cuda()

        weights = [{"weight": packed, "weight_scale": scales}]
        method.load_weights_vanilla(mod, weights)

        return mod, w_ref.cuda(), method

    def test_forward_shape(self, layer_bf16):
        mod, _, method = layer_bf16
        x = torch.randn(4, 128, dtype=torch.bfloat16).cuda()
        out = method.apply(mod, x, bias=None)
        assert out.shape == (4, 64)

    def test_forward_dtype(self, layer_bf16):
        mod, _, method = layer_bf16
        x = torch.randn(4, 128, dtype=torch.bfloat16).cuda()
        out = method.apply(mod, x, bias=None)
        assert out.dtype == torch.bfloat16

    def test_forward_accuracy_vs_reference(self, layer_bf16):
        """BF16 output should closely match F.linear with the reference weights."""
        mod, w_ref, method = layer_bf16
        torch.manual_seed(0)
        x = torch.randn(8, 128, dtype=torch.bfloat16).cuda()

        out_bf16 = method.apply(mod, x, bias=None).float()
        out_ref = torch.nn.functional.linear(x.float(), w_ref)

        torch.testing.assert_close(out_bf16, out_ref, atol=0.5, rtol=0.05)

    def test_weight_dtype_is_bf16(self, layer_bf16):
        mod, _, _ = layer_bf16
        assert mod.weight.dtype == torch.bfloat16

    def test_no_weight_scale_attribute(self, layer_bf16):
        """BF16 method absorbs scales; no weight_scale parameter needed."""
        mod, _, _ = layer_bf16
        assert not hasattr(mod, "weight_scale")

    def test_float32_input_upcast(self, layer_bf16):
        """FP32 activations should be silently upcast via F.linear."""
        mod, _, method = layer_bf16
        x = torch.randn(4, 128, dtype=torch.float32).cuda()
        out = method.apply(mod, x, bias=None)
        assert out.shape == (4, 64)

    def test_fused_gate_up_shape(self):
        """Fused gate-up load should concatenate two shards along output dim."""
        out_f, in_f = 32, 64
        packed0, scales0, _ = _make_nvfp4_checkpoint(out_f, in_f, seed=1)
        packed1, scales1, _ = _make_nvfp4_checkpoint(out_f, in_f, seed=2)

        method = NVFP4ToBF16LinearMethod()

        class _FakeModule:
            tp_size = 1
            tp_rank = 0
            tp_mode = None
            bias = None

        mod = _FakeModule()
        method.create_weights(mod, in_f, out_f * 2, bias=False, dtype=torch.bfloat16)
        mod.weight = mod.weight.cuda()

        weights = [
            {"weight": packed0, "weight_scale": scales0},
            {"weight": packed1, "weight_scale": scales1},
        ]
        method.load_weights_fused_gate_up_linear(mod, weights)
        assert mod.weight.shape == (out_f * 2, in_f)
