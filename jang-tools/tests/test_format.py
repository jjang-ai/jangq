"""Tests for JANG format writer/reader — end-to-end roundtrip."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jang_tools.quantize import quantize_tensor
from jang_tools.format.writer import write_jang_model
from jang_tools.format.reader import load_jang_model, is_jang_model


class TestFormatRoundtrip:
    """Write a JANG model, read it back, verify everything matches."""

    def test_write_read_roundtrip(self):
        """Full roundtrip: quantize → write → read → verify."""
        rng = np.random.default_rng(42)

        # Create fake weight tensors
        weights = {
            "layers.0.self_attn.q_proj": rng.standard_normal((256, 256)).astype(np.float32) * 0.02,
            "layers.0.self_attn.k_proj": rng.standard_normal((256, 256)).astype(np.float32) * 0.02,
            "layers.0.mlp.gate_proj": rng.standard_normal((512, 256)).astype(np.float32) * 0.02,
        }

        # Quantize each tensor — different bit widths per tensor (tier-based)
        quantized = {}
        tensor_bits = {
            "layers.0.self_attn.q_proj": 6,   # CRITICAL
            "layers.0.self_attn.k_proj": 6,   # CRITICAL
            "layers.0.mlp.gate_proj": 2,       # COMPRESS
        }
        for name, w in weights.items():
            n_blocks = (w.size) // 64
            bits = tensor_bits[name]
            bit_alloc = np.full(n_blocks, bits, dtype=np.uint8)

            qt = quantize_tensor(w, bit_alloc, block_size=64)
            quantized[name] = qt

        # Write to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test-model-JANG-3bit"

            model_config = {"hidden_size": 256, "num_layers": 1}
            jang_config = {
                "quantization": {
                    "method": "jang-importance",
                    "target_bits": 3.0,
                    "actual_bits": 3.1,
                    "block_size": 64,
                    "scoring_method": "awq+hessian",
                    "bit_widths_used": [2, 4, 6],
                },
                "source_model": {
                    "name": "test-model",
                    "dtype": "float32",
                    "parameters": "1M",
                },
            }

            write_jang_model(
                output_dir=model_dir,
                quantized_tensors=quantized,
                model_config=model_config,
                jang_config=jang_config,
            )

            # Verify files exist
            assert is_jang_model(model_dir)
            assert (model_dir / "jang_config.json").exists()
            assert (model_dir / "config.json").exists()
            assert (model_dir / "model.jang.index.json").exists()

            # Read back
            model = load_jang_model(model_dir)
            assert model.target_bits == 3.0
            assert model.source_model == "test-model"

            # Verify weight names
            weight_names = model.weight_names
            assert len(weight_names) == 3
            assert "layers.0.self_attn.q_proj" in weight_names

            # Verify tensor data matches
            for name in weight_names:
                original = quantized[name]
                loaded = model.get_quantized_tensor(name)

                np.testing.assert_array_equal(loaded.qweight, original.qweight)
                np.testing.assert_array_equal(loaded.scales, original.scales)
                np.testing.assert_array_equal(loaded.biases, original.biases)
                assert loaded.bits == original.bits

            # Verify summary
            summary = model.summary()
            assert summary["total_weight_names"] == 3
            assert summary["total_blocks"] > 0
            assert "2-bit" in summary["histogram"]
            assert "6-bit" in summary["histogram"]
            assert "6-bit" in summary["histogram"]

    def test_not_mxq_model(self):
        """Non-JANG directory should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not is_jang_model(tmpdir)

    def test_invalid_format_rejected(self):
        """Model with wrong format field should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"format": "not-mxq", "format_version": "1.0"}
            (Path(tmpdir) / "jang_config.json").write_text(json.dumps(config))

            with pytest.raises(ValueError, match="Invalid format"):
                load_jang_model(tmpdir)
