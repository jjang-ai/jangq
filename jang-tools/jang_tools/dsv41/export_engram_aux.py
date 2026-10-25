"""Export the Engram n-gram-hash tables (token_map / multipliers / primes / offsets + scalars) that
the vmlx-SWIFT port needs, so Swift does not reimplement HF-tokenizer normalization. Reuses the proven
jang_tools.dsv41.engram setup (EngramLayout + NgramHasher) and dumps one little safetensors sidecar.

  python export_engram_aux.py <bundle> <out.safetensors>

Swift parses safetensors itself (JSON header + raw bytes); dtypes chosen for easy Swift load:
  token_map   I32 [vocab]                      (compressed token id per vocab id; < compressed_vocab)
  multipliers I64 [n_layers, max_ngram]
  primes      I64 [n_layers, max_ngram-1, n_heads]
  offsets     I64 [n_layers, cols]             (cols = (max_ngram-1)*n_heads)
  pad_id      I64 [1]
Scalars (n_heads, head_dim, max_ngram, layer_ids, compressed_vocab) go in the safetensors __metadata__.
"""
import json, struct, sys
from pathlib import Path
import numpy as np

bundle = Path(sys.argv[1]).expanduser()
out = Path(sys.argv[2]).expanduser()

cfg = json.load(open(bundle / "config.json"))["text_config"]
from jang_tools.dsv41.engram import EngramLayout, NgramHasher

layout = EngramLayout.from_config(cfg)
hasher = NgramHasher(cfg, layout, bundle / "tokenizer.json")

tensors = {
    "token_map":   hasher.token_map.astype(np.int32),
    "multipliers": hasher.multipliers.astype(np.int64),
    "primes":      layout.primes.astype(np.int64),
    "offsets":     hasher.offsets.astype(np.int64),
    "pad_id":      np.array([hasher.pad_id], np.int64),
}
meta = {
    "n_heads": str(layout.n_heads),
    "head_dim": str(layout.head_dim),
    "max_ngram_size": str(layout.max_ngram_size),
    "layer_ids": ",".join(str(x) for x in layout.layer_ids),
    "compressed_vocab_size": str(int(hasher.token_map.max()) + 1),
    "vocab_size": str(len(hasher.token_map)),
}

DT = {np.dtype(np.int32): "I32", np.dtype(np.int64): "I64"}
header, blob, off = {}, bytearray(), 0
for name, arr in tensors.items():
    arr = np.ascontiguousarray(arr)
    b = arr.tobytes()
    header[name] = {"dtype": DT[arr.dtype], "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
    blob += b
    off += len(b)
header["__metadata__"] = meta
hj = json.dumps(header).encode()
pad = (-len(hj)) % 8
hj += b" " * pad
out.write_bytes(struct.pack("<Q", len(hj)) + hj + bytes(blob))
print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
print("meta:", meta)
for k, a in tensors.items():
    print(f"  {k}: {a.dtype} {a.shape}")
