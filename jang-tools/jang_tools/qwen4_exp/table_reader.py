"""Page-granular reader for the 51B n-gram table.

MLX's gather materializes each touched shard tensor (~800 MB) — one PLE call
can page the whole 95 GiB table. This reader maps safetensors data sections
with np.memmap and reads ONLY the requested rows (4 KiB pages), then hands a
small packed array to MLX. Used for the bf16 source; quantized bundles keep
the (much smaller) MLX path or stay resident.
"""

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
           "U32": np.uint32, "I64": np.int64}


class SafetensorsRowReader:
    """Row reader over one 2-D tensor stored in a safetensors file."""

    def __init__(self, path: str, tensor_name: str):
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        info = header[tensor_name]
        self.dtype_tag = info["dtype"]
        self.np_dtype = _DTYPES[self.dtype_tag]
        self.shape = info["shape"]
        start, end = info["data_offsets"]
        self.mm = np.memmap(path, dtype=self.np_dtype, mode="r",
                            offset=8 + hlen + start,
                            shape=tuple(self.shape))

    def rows(self, idx: np.ndarray) -> np.ndarray:
        out = self.mm[idx]  # fancy-index on memmap reads only touched pages
        if self.dtype_tag == "BF16":
            u = np.asarray(out, dtype=np.uint16)
            return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
        return np.asarray(out)


class FileBackedNGramTable:
    """Drop-in gather source for ShardedNGramEmbedding.set_file_backed()."""

    def __init__(self, model_dir: str, shard_key_fmt: str, n_shards: int, index_name="model.safetensors.index.json"):
        model_dir = Path(model_dir)
        wm = json.loads((model_dir / index_name).read_text())["weight_map"]
        self.readers = []
        for s in range(n_shards):
            key = shard_key_fmt.format(s)
            self.readers.append(SafetensorsRowReader(str(model_dir / wm[key]), key))
        self.per = self.readers[0].shape[0]

    def gather(self, flat_rows: np.ndarray) -> np.ndarray:
        shard_idx = flat_rows // self.per
        local = flat_rows % self.per
        head_dim = self.readers[0].shape[1]
        out = np.empty((flat_rows.shape[0], head_dim), dtype=np.float32)
        for s in np.unique(shard_idx):
            sel = np.nonzero(shard_idx == s)[0]
            out[sel] = self.readers[int(s)].rows(local[sel])
        return out
