# Serving a JANG model as an OpenAI-compatible HTTP server

The recommended public server path is `vmlx-engine`. It exposes standard
OpenAI-compatible endpoints for chat completions, Responses API, tool calling,
reasoning parser output, multimodal requests, cache inspection, and model lists.

## Quick start

```sh
python -m pip install -U "jang[mlx]" vmlx-engine openai httpx
export JANG_MODEL=/path/to/your-JANG-or-JANGTQ-model
./examples/runtime/serve_vmlx.sh
```

The helper script expands to a `vmlx-engine serve` command with these defaults:

```sh
vmlx-engine serve "$JANG_MODEL" \
  --served-model-name default \
  --host 127.0.0.1 \
  --port 8000 \
  --continuous-batching \
  --use-paged-cache \
  --enable-auto-tool-choice \
  --tool-call-parser auto
```

## Call it like OpenAI

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 128,
    "stream": true
  }'
```

Python client:

```sh
python examples/runtime/openai_chat.py --prompt "Hello" --stream
```

Swift client:

```sh
swift examples/runtime/swift/OpenAIChat.swift --prompt "Hello"
```

## Tool calling

Start the server with the parser that matches the model family:

```sh
JANG_TOOL_PARSER=qwen ./examples/runtime/serve_vmlx.sh
```

Then call with standard OpenAI `tools`:

```sh
python examples/runtime/openai_tools.py --prompt "What is the weather in Paris?"
```

The model must have been trained for tool use. If tool calls parse incorrectly,
check `config.json.tool_call_parser`, tokenizer chat template, and EOS settings.

## Reasoning / thinking

For reasoning models, pass the parser that matches the source format:

```sh
JANG_REASONING_PARSER=qwen3 ./examples/runtime/serve_vmlx.sh
```

The API may return reasoning as `reasoning` or `reasoning_content`, depending on
the client schema and parser. Do not add fake prompt tags to force reasoning;
fix parser/template metadata instead.

## Vision, audio, and video

Multimodal requests require a bundle and runtime loader that expose the modality.
Check `preprocessor_config.json`, video/audio processor files, and the model
card before advertising support.

Image request:

```sh
python examples/runtime/openai_multimodal.py \
  --image ./photo.jpg \
  --prompt "Describe this image."
```

Video request:

```sh
python examples/runtime/openai_multimodal.py \
  --video ./clip.mp4 \
  --prompt "Summarize the action." \
  --video-fps 2 \
  --video-max-frames 32
```

## Responses API

Agent clients can use `/v1/responses`:

```sh
python examples/runtime/openai_responses.py --prompt "What is 2 + 2?"
```

Codex-style local configuration:

```toml
model_provider = "vmlx"
model = "default"

[model_providers.vmlx]
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
```

## Health, models, and cache

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/cache/stats
curl http://127.0.0.1:8000/v1/cache/entries
```

For a deeper integration checklist, see
[`../RUNTIME_QUICKSTART.md`](../RUNTIME_QUICKSTART.md).
