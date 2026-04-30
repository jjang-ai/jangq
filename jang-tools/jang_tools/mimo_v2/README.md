# MiMo-V2.5-Pro — JANG quant pipeline

Brand-new architecture (`model_type=mimo_v2`) — 1.02T total / 42B active MoE,
hybrid SWA + global, 384 experts, FP8 [128,128] block source. Target bundles:
`JANGQ-AI/MiMo-V2.5-Pro-JANGTQ2` and `JANGQ-AI/MiMo-V2.5-Pro-JANG_2L`.

## Files

| file | purpose |
|---|---|
| `config.py` | dataclass mirror of upstream `MiMoV2Config` |
| `model.py` | MLX port of `modeling_mimo_v2.py` |
| `fp8_codec.py` | FP8 e4m3 + FP32 [128,128] block scale -> bf16 |
| `weight_loader.py` | stream FP8 source -> bf16 mlx tree |
| `runtime.py` | single-node sanity / decode |
| `decode.py` | greedy + sampling helpers |
| `dist_runtime.py` | distributed inference (sanity / decode) |
| `jang_loader.py` | load JANG (mx.quantize affine) bundle, EP-aware |
| `jangtq_loader.py` | load JANGTQ (TurboQuant) bundle, EP-aware |
| `convert.py` | source -> JANG_2L bundle |
| `convert_jangtq.py` | source -> JANGTQ2 bundle |

## Pipeline

```text
  FP8 source (1.03 TB)
       │
       ├─ runtime.py --no-cache   (sanity, no quant — runtime-before-quant rule)
       │
       ├─ convert.py             -> JANGQ-AI/MiMo-V2.5-Pro-JANG_2L (~260 GB)
       └─ convert_jangtq.py      -> JANGQ-AI/MiMo-V2.5-Pro-JANGTQ2 (~210 GB)
              │
              └─ dist_runtime.py --mode decode   (distributed validation)
```

## Distributed

Launch from `jang-tools/jang_tools/distributed/`:

```bash
# 1. Bring up TB5 bridge
./scripts/setup_tb5.sh studio        # on Mac Studio
./scripts/setup_tb5.sh macbook       # on M4 Max

# 2. Build hostfile
python -m jang_tools.distributed.discovery \
    --out ./hostfile.json \
    --peers macstudio,macstudio,10.42.0.1 \
            macbook,macbook,10.42.0.2

# 3. Probe transport (TB5 / RDMA)
./scripts/launch.sh probe --rounds 5

# 4. Sanity forward across two nodes
./scripts/launch.sh sanity --src /Volumes/EricsLLMDrive/jangq-ai/_sources/MiMo-V2.5-Pro

# 5. Distributed decode
./scripts/launch.sh decode \
    --src /Volumes/EricsLLMDrive/jangq-ai/MiMo-V2.5-Pro-JANGTQ2 \
    --prompt "The capital of France is" --max-new 32
```

## Bundle invariants

Every output JANG / JANGTQ bundle MUST contain in `config.json`:

- `mxtq_bits` AND `routed_expert_bits` (2026-04-25 invariant — vmlx-swift §418
  fallback breaks without these)
- `rope_parameters` block with `rope_theta` (float) and `rope_type`
  (transformers >= 4.50 contract — without it `load_tokenizer` falls back
  to bare `PreTrainedTokenizerFast` and drops `chat_template`)
- `auto_map` -> `modeling_mimo_v2.MiMoV2ForCausalLM`
- Per-module overrides for the 70 `o_proj` layers (stay bf16, match
  upstream `quantization_config.ignored_layers`)
- Carry `chat_template` + `generation_config` from upstream

## Port to vMLX

Once Python reference is correct on this rig, port to:

- `vmlx/swift/Sources/vMLXLMCommon/` — Swift module + JANGTQ kernels
- `vmlx/inference/` — runtime
- distributed transport stays in MLX core (jaccl / ring) — no port needed
