# JANG Issues & TODO

## CRITICAL: Format v2.0 — Store MLX-native uint32 (eliminate repack)

### Problem
Current JANG format stores qweights as uint8 (packed). The loader repacks uint8→uint32 on EVERY load. This takes 5-10 minutes for large MoE models (47K tensors). Users cannot use models until repack finishes.

### Root cause (convert.py line 308)
```python
qw, scales, biases = mx.quantize(w_mx, group_size=block_size, bits=bits)
# mx.quantize returns uint32 — MLX native format. READY TO USE.
# But then we convert to uint8 for storage:
qweight = np.array(qw).view(np.uint8).reshape(-1)  # ← POINTLESS
```
The loader then does the reverse:
```python
mlx_qweight = mx.array(np.frombuffer(qweight_raw.tobytes(), dtype=np.uint32))  # ← POINTLESS
```

### Fix: Format v2.0
1. **converter**: Store `mx.quantize()` output directly as uint32 safetensors. Keep scales/biases as float16. Store `.weight` key (not `.qweight`).
2. **writer**: Write standard `.safetensors` shards with `{name}.weight` (uint32), `{name}.scales` (float16), `{name}.biases` (float16).
3. **loader**: Just `mx.load()` via mmap. No repack. Same speed as native MLX.
4. **format version**: `"format_version": "2.0"` in jang_config.json.
5. **backward compat**: Loader detects v1.x → old repack path. v2.0 → instant mmap.

### Result
- MiniMax (60GB): 5 min → seconds
- 35B: 25s → seconds
- 397B: hours → seconds
- Same speed as mlx-community quantized models
- Same file size (uint32 packed weights are same bytes as uint8 view)
- No `.mlx_cache/` hack needed

### Implementation plan
1. Update `jang_tools/format/writer.py`: store uint32 qweights directly under `{name}.weight` key
2. Update `jang_tools/convert.py`: skip uint8 conversion (line 308)
3. Update `jang_tools/format/spec.py`: add v2.0 spec
4. Update `vmlx_engine/utils/jang_loader.py`: add v2.0 fast path (just mx.load)
5. Re-convert existing models: MiniMax, Qwen3.5 35B/122B
6. Upload to HuggingFace as v2.0

---

## Fixed (from previous session)

### tree_flatten import
- Fixed in vmlx: `from mlx.utils import tree_flatten`
- **Still broken in jang-tools/loader.py** — uses `model.parameters().values()`

### VLM support
- JANG VLMs blocked by design (`is_mllm_model` returns False for JANG)
- `load_jang_vlm_model()` exists but is not wired to inference pipeline
- Will be enabled after v2.0 format lands

### MoE expert stacking
- Working correctly in both vmlx and jang-tools loaders
- `_stack_per_expert_weights` handles MiniMax w1/w2/w3 naming

### Streaming timeout
- Fixed: `_default_timeout` configurable via `--timeout` flag
- Image edits use 30-min timeout

---

## Disk Space Analysis (MacBook — 2026-03-17)

| Path | Size | Notes |
|---|---|---|
| `~/.mlxstudio/models/` | **1.1 TB** | After cleanup |
| `~/jang/models/` | **200 GB** | JANG converted models |
| `~/Library/Caches/` | 14 GB | System caches |
