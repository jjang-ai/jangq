"""Regression coverage for serializing both halves of a fused expert stack."""

import numpy as np
from safetensors.numpy import load_file, save_file

from jang_tools.format import writer


def test_v2_writer_serializes_strided_experts_without_copying_contiguous_tensors(
    tmp_path, monkeypatch
):
    fused = np.arange(4 * 6 * 2, dtype=np.uint32).reshape(4, 6, 2)
    gate, up = fused[:, :3, :], fused[:, 3:, :]
    assert not gate.flags.c_contiguous
    assert not up.flags.c_contiguous
    scalar = np.array(27, dtype=np.int32)
    weights = {"gate.weight": gate, "up.weight": up, "scalar": scalar}

    def require_contiguous(tensors, filename, *, metadata):
        assert all(tensor.flags.c_contiguous for tensor in tensors.values())
        assert tensors["scalar"] is scalar
        return save_file(tensors, filename, metadata=metadata)

    monkeypatch.setattr(writer, "save_file", require_contiguous)
    writer.write_jang_v2_model(tmp_path, weights, {}, {})
    saved = load_file(str(next(tmp_path.glob("model-*.safetensors"))))
    for name, expected in weights.items():
        assert saved[name].dtype == expected.dtype
        assert saved[name].shape == expected.shape
        assert saved[name].tobytes() == expected.tobytes()
