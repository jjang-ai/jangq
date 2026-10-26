"""Concurrent (16-lane) Engram gathers == a plain memory-mapped gather, and pipelined lookup == lookup.

CPU only, on a small synthetic FP8 table written as a safetensors shard.

  PYTHONPATH=jang-tools python jang_tools/dsv41/tests/test_engram_io.py
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.dsv41 import engram as E
from jang_tools.dsv41.st_io import Shards


def _write_table(path: Path, n_rows: int, dim: int, sdim: int, rng) -> tuple[np.ndarray, np.ndarray]:
    w = rng.integers(0, 0x7F, (n_rows, dim), dtype=np.uint8)  # finite positive E4M3 codes
    s = rng.integers(120, 134, (n_rows, sdim), dtype=np.uint8)  # E8M0 exponents near 1.0
    header = {
        "layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [n_rows, dim], "data_offsets": [0, w.nbytes]},
        "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [n_rows, sdim],
                                        "data_offsets": [w.nbytes, w.nbytes + s.nbytes]},
    }
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + w.tobytes() + s.tobytes())
    return w, s


def test_concurrent_gather_matches_memmap():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        _write_table(Path(d) / "engram.safetensors", 20000, 256, 8, rng)
        t = E.DiskEngramTable(Shards(d, index_name="none"), 1)
        for n in (1, 7, E._TAKE_LANES, E._TAKE_LANES + 1, 9000):  # fewer rows than lanes, exact, many
            ids = rng.integers(0, 20000, (n, 3))  # duplicates included
            inv, rr, ss = t.read(ids)
            uniq = np.unique(ids.reshape(-1))
            assert np.array_equal(rr, np.asarray(t.rows[uniq])), n
            assert np.array_equal(ss, np.asarray(t.scales[uniq])), n
            assert np.array_equal(uniq[inv], ids.reshape(-1)), n
            a = t.lookup(ids)
            b = t.finish(ids.shape, t.prefetch(ids).result())
            assert a.shape == (n, 3, 256) and mx.array_equal(a, b).item(), n


if __name__ == "__main__":
    test_concurrent_gather_matches_memmap()
    print("ok")
