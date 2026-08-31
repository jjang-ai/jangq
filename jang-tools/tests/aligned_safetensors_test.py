import json
import struct

from jang_tools.format.aligned_safetensors import (
    rewrite_aligned_safetensors,
    verify_safetensors_alignment,
)


def _write_deliberately_unaligned(path):
    tensors = {
        "half": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        "word": {"dtype": "U32", "shape": [1], "data_offsets": [2, 6]},
        "__metadata__": {"format": "mlx"},
    }
    encoded = json.dumps(tensors, separators=(",", ":")).encode()
    # Force data_start to 0 mod 4. The preceding two-byte F16 payload then
    # places the U32 tensor at an invalid absolute offset congruent to 2.
    encoded += b" " * ((-(8 + len(encoded))) % 4)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(b"abWXYZ")


def test_streaming_rewrite_preserves_payloads_and_aligns_all_dtypes(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    _write_deliberately_unaligned(source)

    assert verify_safetensors_alignment(source) == (2, 1)
    rewrite_aligned_safetensors(source, output, chunk_bytes=2)
    assert verify_safetensors_alignment(output) == (2, 0)

    with output.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        payload = handle.read()
    assert header["__metadata__"] == {"format": "mlx"}
    assert payload[0:4] == b"WXYZ"
    assert payload[4:6] == b"ab"
