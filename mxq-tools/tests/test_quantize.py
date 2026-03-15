"""Tests for MXQ quantization and dequantization."""

import numpy as np
import pytest
from mxq_tools.quantize import (
    quantize_block_rtn,
    quantize_block_mse,
    quantize_tensor,
    dequantize_tensor,
)
from mxq_tools.allocate import allocate_bits_greedy


class TestBlockQuantize:
    """Test per-block quantization."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_rtn_range(self, bits):
        """Quantized values should be in valid range."""
        rng = np.random.default_rng(42)
        weights = rng.standard_normal(64).astype(np.float32)
        q_ints, scale, zero = quantize_block_rtn(weights, bits)

        max_val = (1 << bits) - 1
        assert q_ints.min() >= 0
        assert q_ints.max() <= max_val

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
    def test_mse_better_than_rtn(self, bits):
        """MSE-optimal quantization should have equal or lower error than RTN."""
        rng = np.random.default_rng(42)
        weights = rng.standard_normal(64).astype(np.float32)

        q_rtn, s_rtn, z_rtn = quantize_block_rtn(weights, bits)
        q_mse, s_mse, z_mse = quantize_block_mse(weights, bits)

        # Dequantize and measure error
        dq_rtn = (q_rtn.astype(np.float32) - z_rtn) * s_rtn
        dq_mse = (q_mse.astype(np.float32) - z_mse) * s_mse

        mse_rtn = float(np.mean((weights - dq_rtn) ** 2))
        mse_mse = float(np.mean((weights - dq_mse) ** 2))

        assert mse_mse <= mse_rtn * 1.01  # allow tiny floating point tolerance

    def test_constant_block(self):
        """Constant-valued block should quantize without error."""
        weights = np.full(64, 3.14, dtype=np.float32)
        q, s, z = quantize_block_rtn(weights, 4)
        # Should not crash; constant block is a valid edge case


class TestTensorQuantize:
    """Test full tensor quantization with variable bit widths."""

    def test_roundtrip_quality(self):
        """Quantize → dequantize should produce reasonable reconstruction."""
        rng = np.random.default_rng(42)
        # Simulate a weight matrix (512 × 512)
        weights = rng.standard_normal((512, 512)).astype(np.float32) * 0.02

        n_blocks = (512 * 512) // 64
        # Mix of bit widths: mostly 4-bit, some 2-bit, some 8-bit
        bit_alloc = np.full(n_blocks, 4, dtype=np.uint8)
        bit_alloc[:n_blocks // 4] = 2
        bit_alloc[-n_blocks // 8:] = 8

        qt = quantize_tensor(weights, bit_alloc, block_size=64, method="rtn")
        dq = dequantize_tensor(qt, block_size=64)

        # Check shape preserved
        assert dq.shape == weights.shape

        # Check reconstruction quality — should be reasonable
        mse = float(np.mean((weights - dq) ** 2))
        assert mse < 1e-3  # reasonable for mixed 2/4/8-bit

    def test_all_bit_widths(self):
        """Test quantization at each individual bit width."""
        rng = np.random.default_rng(123)
        weights = rng.standard_normal((64, 64)).astype(np.float32) * 0.02

        for bits in [2, 3, 4, 5, 6, 8]:
            n_blocks = (64 * 64) // 64
            bit_alloc = np.full(n_blocks, bits, dtype=np.uint8)

            qt = quantize_tensor(weights, bit_alloc, block_size=64)
            dq = dequantize_tensor(qt, block_size=64)

            assert dq.shape == weights.shape
            mse = float(np.mean((weights - dq) ** 2))
            # Higher bits = lower error
            if bits >= 4:
                assert mse < 1e-4

    def test_mixed_precision_better_than_uniform_low(self):
        """
        Mixed-precision (avg 2.5 bits) should outperform uniform 2-bit.
        This is the core thesis of MXQ.
        """
        rng = np.random.default_rng(42)
        weights = rng.standard_normal((256, 256)).astype(np.float32) * 0.02
        n_blocks = (256 * 256) // 64

        # Uniform 2-bit
        uniform_2 = np.full(n_blocks, 2, dtype=np.uint8)
        qt_2 = quantize_tensor(weights, uniform_2, block_size=64)
        dq_2 = dequantize_tensor(qt_2, block_size=64)
        mse_2 = float(np.mean((weights - dq_2) ** 2))

        # Mixed: half at 2-bit, half at 3-bit (avg 2.5)
        mixed = np.full(n_blocks, 2, dtype=np.uint8)
        # Give 3-bit to blocks with highest weight variance
        flat = weights.reshape(-1)
        block_vars = np.array([
            float(np.var(flat[i * 64:(i + 1) * 64]))
            for i in range(n_blocks)
        ])
        important = np.argsort(block_vars)[-n_blocks // 2:]
        mixed[important] = 3

        qt_mix = quantize_tensor(weights, mixed, block_size=64)
        dq_mix = dequantize_tensor(qt_mix, block_size=64)
        mse_mix = float(np.mean((weights - dq_mix) ** 2))

        # Mixed should be better
        assert mse_mix < mse_2, f"Mixed MSE {mse_mix} should be < uniform 2-bit MSE {mse_2}"


class TestBitAllocation:
    """Test the bit allocation algorithm."""

    def test_average_bits_near_target(self):
        """Allocation should produce average bits close to target."""
        rng = np.random.default_rng(42)
        n_blocks = 1000
        importance = rng.random(n_blocks).astype(np.float32)
        names = [f"layers.{i // 100}.mlp.gate_proj" for i in range(n_blocks)]

        for target in [2.5, 3.0, 4.0]:
            bits = allocate_bits_greedy(
                importance, target, names, n_layers=10
            )
            avg = float(np.mean(bits))
            # Should be within 0.3 bits of target
            assert abs(avg - target) < 0.3, f"Target {target}, got {avg}"

    def test_important_blocks_get_more_bits(self):
        """More important blocks should get more (or equal) bits."""
        n_blocks = 100
        # First half very important, second half not
        importance = np.zeros(n_blocks, dtype=np.float32)
        importance[:50] = 10.0
        importance[50:] = 0.1
        names = [f"layers.5.mlp.gate_proj" for _ in range(n_blocks)]

        bits = allocate_bits_greedy(importance, 3.0, names, n_layers=10)

        avg_important = float(np.mean(bits[:50]))
        avg_unimportant = float(np.mean(bits[50:]))

        assert avg_important >= avg_unimportant

    def test_layer_priors(self):
        """Embedding and lm_head should get minimum bit floors."""
        n_blocks = 10
        importance = np.ones(n_blocks, dtype=np.float32)
        names = [
            "model.embed_tokens",  # should get >= 4 bit
            "layers.0.self_attn.q_proj",  # should get >= 3 bit
            "layers.0.self_attn.k_proj",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.1.mlp.gate_proj",
            "layers.1.mlp.up_proj",
            "layers.1.mlp.down_proj",
            "layers.2.mlp.gate_proj",
            "lm_head",  # should get >= 6 bit
        ]

        bits = allocate_bits_greedy(importance, 3.0, names, n_layers=3)

        assert bits[0] >= 4, f"embed_tokens got {bits[0]} bits, expected >= 4"
        assert bits[1] >= 3, f"q_proj got {bits[1]} bits, expected >= 3"
        assert bits[-1] >= 6, f"lm_head got {bits[-1]} bits, expected >= 6"
