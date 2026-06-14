# JANG Runtime Examples

These examples show how to run JANG and JANGTQ bundles through an
OpenAI-compatible local server and call them from Python, Swift, curl, or agent
clients.

The recommended public runtime path is:

1. Install `jang` for bundle inspection and direct MLX loading.
2. Install `vmlx-engine` for OpenAI-compatible serving.
3. Serve a local JANG or JANGTQ bundle.
4. Call the server through standard `/v1/chat/completions` or `/v1/responses`
   clients.

## Install

```sh
python -m pip install -U "jang[mlx]" vmlx-engine openai httpx
```

For vision, audio, or video bundles also install the runtime extras required by
the model family:

```sh
python -m pip install -U "jang[vlm]" pillow
```

## Serve A Bundle

```sh
export JANG_MODEL=/path/to/model-JANG-or-JANGTQ
./examples/runtime/serve_vmlx.sh
```

The script uses `vmlx-engine serve` with conservative defaults:

- `--continuous-batching` for concurrent clients.
- `--use-paged-cache` for bounded prompt-cache memory.
- vMLX cache auto mode by default; set `JANG_KV_CACHE_QUANTIZATION=q4`, `q8`,
  or `none` when you want to override it.
- `--enable-auto-tool-choice --tool-call-parser auto` for tool-capable chat
  models.

Override any setting through environment variables:

```sh
JANG_MODEL=/models/DeepSeek-V4-Flash-JANGTQ2 \
JANG_PORT=8001 \
JANG_REASONING_PARSER=deepseek_r1 \
JANG_TOOL_PARSER=deepseek \
JANG_KV_CACHE_QUANTIZATION=none \
./examples/runtime/serve_vmlx.sh
```

## Clients

```sh
python examples/runtime/openai_chat.py --prompt "Explain JANG in one paragraph."
python examples/runtime/openai_tools.py --prompt "What is the weather in Paris?"
python examples/runtime/openai_responses.py --prompt "Write a haiku about Metal kernels."
python examples/runtime/openai_multimodal.py --image ./photo.jpg --prompt "Describe this image."
```

Swift client:

```sh
swift examples/runtime/swift/OpenAIChat.swift \
  --base-url http://127.0.0.1:8000/v1 \
  --model default \
  --prompt "Hello from Swift"
```

## Direct Loader Examples

For lower-level MLX integration, see:

- `docs/adoption/EXAMPLES/python.py`
- `jang-tools/examples/dsv4_flash/`
- `jang-tools/examples/nemotron_omni/`
- `jang-tools/examples/mimo_v2/`

Those examples load bundles directly through `jang_tools` and are useful for
runtime authors. Application developers should usually start with the server
clients in this directory.
