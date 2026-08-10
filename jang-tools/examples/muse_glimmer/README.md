# Muse Glimmer examples

Runtime generation examples are intentionally deferred until the vMLX
Swift/Python model implementations exist. The checked-in example for this pass
is the real artifact verifier; it does not pretend structural loading is
generation proof.

From the repository root:

```bash
PYTHONPATH=jang-tools uv run --no-project \
  --with mlx --with numpy --with safetensors --with tqdm \
  python jang-tools/scripts/verify_muse_glimmer_artifact.py \
  ~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_4M \
  --profile JANG_4M --dequant
```

Replace the path/profile with `Muse-Glimmer-30B-JANG_2L` / `JANG_2L` for the
low-memory bundle or `Muse-Glimmer-30B-JANG_6M` / `JANG_6M` for the
near-lossless bundle. Expected output is JSON with `"status": "PASS"` and two
measured dequantization rel-L1 values. This proves artifact integrity, not
coherent generation; 2L in particular must pass the real runtime gate before
publication.

See `docs/runtime/2026-08-10-muse-glimmer-quant-runtime-handoff.md` for the
source contract, method limitations, native reasoning/ATEM format, media path,
cache partial-block/suffix rules, and runtime proof matrix.
