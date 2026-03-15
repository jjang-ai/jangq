"""Tests for MXQ format writer/reader — end-to-end roundtrip."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlxq_tools.quantize import quantize_tensor
from mlxq_tools.format.writer import write_mxq_model
from mlxq_tools.format.reader import load_mxq_model, is_mxq_model


class TestFormatRoundtrip:
    """Write an MXQ model, read it back, verify everything matches."""

    def test_write_read_roundtrip(self):
        """Full roundtrip: quantize → write → read → verify."""
        rng = np.random.default_rng(42)

        # Create fake weight tensors
        weights = {
            "layers.0.self_attn.q_proj": rng.standard_normal((256, 256)).astype(np.float32) * 0.02,
            "layers.0.self_attn.k_proj": rng.standard_normal((256, 256)).astype(np.float32) * 0.02,
            "layers.0.mlp.gate_proj": rng.standard_normal((512, 256)).astype(np.float32) * 0.02,
        }

        # Quantize each tensor
        quantized = {}
        for name, w in weights.items():
            n_blocks = (w.size) // 64
            # Mix of bit widths
            bit_alloc = np.full(n_blocks, 4, dtype=np.uint8)
            bit_alloc[:n_blocks // 3] = 2
            bit_alloc[-n_blocks // 4:] = 6

            qt = quantize_tensor(w, bit_alloc, block_size=64)
            quantized[name] = qt

        # Write to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test-model-MXQ-3bit"

            model_config = {"hidden_size": 256, "num_layers": 1}
            mxq_config = {
                "quantization": {
                    "method": "mxq-importance",
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

            write_mxq_model(
                output_dir=model_dir,
                quantized_tensors=quantized,
                model_config=model_config,
                mxq_config=mxq_config,
            )

            # Verify files exist
            assert is_mxq_model(model_dir)
            assert (model_dir / "mxq_config.json").exists()
            assert (model_dir / "config.json").exists()
            assert (model_dir / "model.mxq.index.json").exists()

            # Read back
            model = load_mxq_model(model_dir)
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
                np.testing.assert_array_equal(loaded.zeros, original.zeros)
                np.testing.assert_array_equal(loaded.bit_map, original.bit_map)
                np.testing.assert_array_equal(loaded.block_offsets, original.block_offsets)

            # Verify summary
            summary = model.summary()
            assert summary["total_weight_names"] == 3
            assert summary["total_blocks"] > 0
            assert "2-bit" in summary["histogram"]
            assert "4-bit" in summary["histogram"]
            assert "6-bit" in summary["histogram"]

    def test_not_mxq_model(self):
        """Non-MXQ directory should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not is_mxq_model(tmpdir)

    def test_invalid_format_rejected(self):
        """Model with wrong format field should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"format": "not-mxq", "format_version": "1.0"}
            (Path(tmpdir) / "mxq_config.json").write_text(json.dumps(config))

            with pytest.raises(ValueError, match="Invalid format"):
                load_mxq_model(tmpdir)
