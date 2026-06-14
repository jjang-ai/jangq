# Runtime Quickstart

This guide is for people who already have a converted JANG or JANGTQ bundle and
want to run it from an application, local server, or framework.

## Recommended Runtime Path

Use `vmlx-engine` when you want a local OpenAI-compatible server:

```sh
python -m pip install -U "jang[mlx]" vmlx-engine openai httpx
export JANG_MODEL=/path/to/model-JANG-or-JANGTQ
./examples/runtime/serve_vmlx.sh
```

Then call it with any OpenAI-compatible client:

```sh
python examples/runtime/openai_chat.py --prompt "Explain this model in one paragraph."
```

See [`../../examples/runtime/`](../../examples/runtime/) for Python, Swift,
tool-calling, multimodal, and Responses API clients.

## Bundle Detection

Runtimes should read metadata instead of guessing from the directory name.

Required files:

- `config.json`: source architecture, tokenizer/runtime parser hints, and
  top-level quantization metadata.
- `jang_config.json`: JANG-specific format and quantization metadata.
- `model.safetensors.index.json`: shard index and tensor map.
- `tokenizer_config.json` plus tokenizer files.

JANG versus JANGTQ:

```python
import json
from pathlib import Path

model_dir = Path("/path/to/model")
cfg = json.loads((model_dir / "jang_config.json").read_text())
method = cfg.get("quantization", {}).get("method")

if method == "jangtq":
    print("JANGTQ: codebook/Hadamard expert format")
else:
    print("JANG: affine packed-weight format")
```

Do not infer bit width from the profile name alone. For standard JANG tensors,
infer per-tensor bits from `.weight` and `.scales` shapes as described in
[`PORTING.md`](PORTING.md). For JANGTQ tensors, preserve the `mxtq_bits` metadata
whether it is stored as a single integer or a per-role dictionary.

## Direct Python Loading

Use direct loading for tests, probes, and custom MLX applications:

```sh
python docs/adoption/EXAMPLES/python.py /path/to/model "Hello"
```

Text JANG bundles use `jang_tools.loader.load_jang_model`. Text JANGTQ bundles
use `jang_tools.load_jangtq.load_jangtq_model`. VL/video JANGTQ bundles use
`jang_tools.load_jangtq_vlm.load_jangtq_vlm_model`.

Direct loading is the best path for framework authors validating tensor layout,
sidecars, parser metadata, and low-level coherency.

## Server Runtime

Use the server path for applications, agents, Swift clients, web apps, and
multi-user local serving:

```sh
JANG_MODEL=/path/to/model ./examples/runtime/serve_vmlx.sh
```

Equivalent expanded shape:

```sh
vmlx-engine serve /path/to/model \
  --served-model-name default \
  --host 127.0.0.1 \
  --port 8000 \
  --continuous-batching \
  --use-paged-cache \
  --max-cache-blocks 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser auto
```

For very large bundles, tune the prompt and cache envelope instead of letting
prefill hit a Metal allocation cliff:

```sh
JANG_MAX_TOKENS=2048 \
JANG_MAX_CACHE_BLOCKS=2048 \
JANG_KV_CACHE_QUANTIZATION=q8 \
./examples/runtime/serve_vmlx.sh
```

The launcher leaves `--kv-cache-quantization` unset by default so vMLX can use
its model-aware cache auto mode. Set `JANG_KV_CACHE_QUANTIZATION=q4`, `q8`, or
`none` only when you want an explicit override.

If you need an L2 block cache on SSD:

```sh
JANG_BLOCK_DISK_CACHE_MAX_GB=20 ./examples/runtime/serve_vmlx.sh
```

## Parser And Template Contract

JANG conversion preserves chat templates and parser hints when they exist in the
source model metadata. Runtime integration should respect:

- `tokenizer_config.json.chat_template` or `chat_template.jinja`
- `generation_config.json.eos_token_id`, including list-valued EOS
- `config.json.tool_call_parser`
- `config.json.reasoning_parser`
- `config.json.enable_thinking`
- `jang_config.json.capabilities` when present

For OpenAI-compatible serving, pick the parser that matches the source model
family. Common examples:

| Family | Tool parser | Reasoning parser |
|---|---|---|
| Qwen / Qwen-MoE | `qwen` | `qwen3` |
| DeepSeek-R1 style | `deepseek` or `dsml` | `deepseek_r1` |
| MiniMax | `minimax` | model-specific |
| Nemotron | `nemotron` | `deepseek_r1` when configured |
| Gemma tool models | `gemma3` or `gemma4` | model-specific |

Do not repair coherency by forcing hidden prompt text, fake thinking tags,
sampler tricks, or parser substitutions. If parser output is wrong, inspect the
source chat template, EOS list, and runtime parser selection.

## Vision, Audio, And Video

A quantized language bundle is not automatically multimodal. Check the files and
metadata:

- Vision/VL: `preprocessor_config.json` plus the expected vision tower/projector
  tensors.
- Video: `video_preprocessor_config.json` or model-family-specific video config.
- Audio: model-family-specific audio encoder/tokenizer files.

Use `examples/runtime/openai_multimodal.py` only for bundles whose runtime path
actually exposes the modality:

```sh
python examples/runtime/openai_multimodal.py --image ./photo.jpg --prompt "Describe this image."
python examples/runtime/openai_multimodal.py --video ./clip.mp4 --prompt "What happens?"
```

If a bundle preserves vision/audio files but the runtime does not yet wire the
processor path, document that as preserved-but-not-served rather than claiming
full multimodal support.

## Runtime-Specific Gotchas

- JANGTQ routed MoE speed depends on stacked experts and fused decode paths.
  Per-expert Python loops will be slow.
- For fused `gate_up_proj` source tensors, runtimes must split gate and up on
  the documented axis before hydrating expert modules.
- Some hybrid or path-dependent cache models need architecture-specific cache
  state in addition to normal KV. Do not reuse a plain KV-only cache when the
  model has SSM, CSA/HSA, CCA, or other side state.
- Dense 2-bit JANGTQ is much riskier than sparse expert 2-bit JANGTQ. If a
  dense model shows a large residual spike, rebuild with a higher precision or
  activation-aware scales rather than hiding it at sampling time.
- Keep embedding, lm head, routed experts, attention, and projector tensors in
  the bit allocation contract declared by the bundle. Missing sidecars on
  quantized tensors should fail verification.

## Public Verification Checklist

Before publishing or recommending a bundle:

- `jang_config.json` exists and declares the correct format and quantization.
- `config.json` preserves model type, tokenizer/parser hints, EOS behavior, and
  modality declarations.
- Tokenizer files and chat template load without fallback hacks.
- Quantized tensors have required sidecars or JANGTQ codebooks.
- Tool/reasoning parser selection matches the source family.
- Text smoke test reaches stop/EOS cleanly.
- For VL/audio/video bundles, a real modality request is tested, not just a
  text prompt against the language model.
- Server clients work against `/v1/chat/completions`; agent clients that need it
  work against `/v1/responses`.
