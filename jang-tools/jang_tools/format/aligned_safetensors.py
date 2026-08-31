"""Alignment-safe safetensors writing and streaming rebundling.

The safetensors format permits tensor payloads to appear in any order.  The
reference serializer does not promise an order that preserves each dtype's
natural alignment, so mixed F16/BF16/U32 JANG shards can place most U32
payloads at offsets congruent to 2 modulo 4.  MLX/Metal cannot expose those
payloads as typed zero-copy buffers and must allocate resident aligned copies.

This module rewrites only the container layout.  Tensor bytes, names, shapes,
dtypes, metadata, and the model index remain unchanged.  Payloads are ordered
by decreasing element size and the JSON header is padded so every absolute
payload offset is aligned to its dtype size.  Rewrites stream in bounded
chunks; multi-gigabyte tensors are never loaded into RAM as a whole.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import BinaryIO


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "C64": 8,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C128": 16,
}


def _read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 0 < header_length < 100_000_000:
            raise ValueError(f"invalid safetensors header length {header_length}: {path}")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return 8 + header_length, header


def _copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    offset: int,
    length: int,
    buffer: bytearray,
) -> None:
    source.seek(offset)
    remaining = length
    view = memoryview(buffer)
    while remaining:
        count = source.readinto(view[: min(remaining, len(buffer))])
        if not count:
            raise ValueError("source safetensors payload is truncated")
        destination.write(view[:count])
        remaining -= count


def rewrite_aligned_safetensors(
    source_path: str | Path,
    destination_path: str | Path | None = None,
    *,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> Path:
    """Stream ``source_path`` into an alignment-safe safetensors container.

    If ``destination_path`` is omitted or resolves to the source path, an
    adjacent temporary file is atomically installed only after the complete
    rewrite succeeds.
    """

    source = Path(source_path)
    destination = Path(destination_path) if destination_path is not None else source
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    data_start, header = _read_header(source)
    metadata = header.get("__metadata__")
    tensors: list[tuple[str, dict, int, int, int]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid descriptor for {name}: {source}")
        dtype = descriptor.get("dtype")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name}: {source}")
        offsets = descriptor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid data_offsets for {name}: {source}")
        start, end = (int(value) for value in offsets)
        if start < 0 or end < start:
            raise ValueError(f"invalid payload range for {name}: {source}")
        tensors.append((name, descriptor, start, end, _DTYPE_BYTES[dtype]))

    source_size = source.stat().st_size
    for name, _, _, end, _ in tensors:
        if data_start + end > source_size:
            raise ValueError(f"payload for {name} exceeds source size: {source}")

    # A descending-alignment order makes every following start valid without
    # inserting illegal gaps into safetensors' contiguous data section.
    tensors.sort(key=lambda item: (-item[4], item[0]))
    output_header: dict[str, object] = {}
    cursor = 0
    for name, descriptor, start, end, _ in tensors:
        length = end - start
        rewritten = dict(descriptor)
        rewritten["data_offsets"] = [cursor, cursor + length]
        output_header[name] = rewritten
        cursor += length
    if metadata is not None:
        output_header["__metadata__"] = metadata

    encoded = json.dumps(
        output_header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    maximum_alignment = max((item[4] for item in tensors), default=8)
    header_alignment = max(8, maximum_alignment)
    padding = (-(8 + len(encoded))) % header_alignment
    encoded += b" " * padding

    destination.parent.mkdir(parents=True, exist_ok=True)
    same_path = source.resolve() == destination.resolve()
    temporary = destination.with_name(
        f".{destination.name}.aligned-{os.getpid()}.tmp"
    )
    buffer = bytearray(chunk_bytes)
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for _, _, start, end, _ in tensors:
                _copy_range(
                    source_handle,
                    output,
                    data_start + start,
                    end - start,
                    buffer,
                )
            output.flush()
            os.fsync(output.fileno())
        shutil.copymode(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if same_path:
        return source
    return destination


def verify_safetensors_alignment(path: str | Path) -> tuple[int, int]:
    """Return ``(tensor_count, unaligned_count)`` for absolute offsets."""

    file_path = Path(path)
    data_start, header = _read_header(file_path)
    count = 0
    unaligned = 0
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        count += 1
        size = _DTYPE_BYTES[descriptor["dtype"]]
        if (data_start + int(descriptor["data_offsets"][0])) % size:
            unaligned += 1
    return count, unaligned


def repack_bundle(source_dir: str | Path, output_dir: str | Path) -> None:
    """Create a separate aligned bundle without mutating the source bundle."""

    source = Path(source_dir)
    output = Path(output_dir)
    if source.resolve() == output.resolve():
        raise ValueError("bundle output must differ from source")
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        destination = output / path.name
        if path.is_dir():
            continue
        if path.suffix == ".safetensors":
            print(f"aligning {path.name}", flush=True)
            rewrite_aligned_safetensors(path, destination)
        else:
            shutil.copy2(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a JANG bundle into dtype-aligned safetensors shards"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repack_bundle(args.source, args.output)


if __name__ == "__main__":
    main()
