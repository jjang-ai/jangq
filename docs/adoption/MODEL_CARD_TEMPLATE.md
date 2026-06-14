# JANG model card template

This is the template `jang-tools modelcard` generates for every converted model. If you are
publishing a JANG model to HuggingFace manually, copy the block below and replace each
`{{ ... }}` placeholder with the value described in the Fields Reference at the bottom.

To generate it automatically from a converted model, run:

```bash
python -m jang_tools modelcard --model /path/to/JANG-model --output README.md
```

---

## Template

```markdown
---
license: {{ license }}
base_model: {{ source_hf_repo }}
tags:
  - jang
  - mlx
  - apple-silicon
  - quantized
  # Add as applicable:
  # - vision-language
  # - video
  # - tool-use
  # - reasoning
quantization_config:
  family: {{ jang or jangtq }}
  profile: {{ e.g. JANG_4K }}
  actual_bits: {{ e.g. 4.23 }}
  block_size: {{ e.g. 64 }}
  size_gb: {{ on-disk size in GB }}
---

# {{ Model name }} — JANG {{ profile }}

Quantized from [`{{ source_hf_repo }}`](https://huggingface.co/{{ source_hf_repo }}) via
[JANG Studio](https://jangq.ai).

## Quick start

### Python (standard JANG)

```python
from jang_tools.loader import load_jang_model
from mlx_lm import generate

model, tokenizer = load_jang_model("/path/to/this-model")
messages = [{"role": "user", "content": "Hello"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### Python (JANGTQ — use this for JANGTQ family models)

```python
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm import generate

model, tokenizer = load_jangtq_model("/path/to/this-model")
messages = [{"role": "user", "content": "Hello"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

### Python (VL / vision models)

```python
from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
from mlx_vlm import generate
from PIL import Image

model, processor = load_jangtq_vlm_model("/path/to/this-model")
image = Image.open("photo.jpg")
response = generate(model, processor, image=image, prompt="Describe this image.", max_tokens=200)
print(getattr(response, "text", response))
```

### CLI

```bash
pip install 'jang[mlx]'
python -m jang_tools inference --model /path/to/this-model --prompt "Hello" --max-tokens 200
```

### OpenAI-compatible server

```bash
pip install "jang[mlx]" vmlx-engine openai httpx
vmlx-engine serve /path/to/this-model --served-model-name default --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

### Swift (via JANGKit)

```swift
import JANGKit

let model = try await JANGKit.Model.load(at: URL(fileURLWithPath: "/path/to/this-model"))
let result = try await model.generate(
    prompt: "Hello",
    config: JANGKit.SamplingConfig(maxTokens: 200)
)
print(result.text)
```

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/jjang-ai/jangq", branch: "main")
],
targets: [
    .executableTarget(name: "YourApp", dependencies: [.product(name: "JANGKit", package: "jangq")])
]
```

> `JANGKit.Model.load` auto-detects JANG vs JANGTQ via `jang_config.json` — the same API works for both families.

## Performance

- **Size on disk:** {{ size_gb }} GB
- **Average bits/weight:** {{ actual_bits }}
- **First-token latency (M3 Ultra, BF16 KV cache):** {{ ms }} ms
- **Steady-state throughput:** {{ tok/s }} tok/s

## Capabilities

- Chat template preserved: {{ yes | no }}
- Tool calling: {{ yes (parser: X) | no }}
- Reasoning / thinking: {{ yes (via `<think>...</think>`) | no }}
- Vision (image input): {{ yes | no }}
- Video input: {{ yes | no }}

## Why JANG?

JANG uses mixed-precision quantization that gives each tensor class the bits it needs —
attention stays at 6-8 bits while expert MLPs compress to 2-4 bits — so the model stays
coherent at drastically smaller sizes.

See [FORMAT.md](https://github.com/jjang-ai/jangq/blob/main/FORMAT.md) for the format
specification and [PORTING.md](https://github.com/jjang-ai/jangq/blob/main/docs/adoption/PORTING.md)
for adding support in your own framework.

## Reproducing this conversion

```bash
pip install 'jang[mlx]'
python -m jang_tools convert {{ source_hf_repo }} -o {{ output_name }} -p {{ profile }}
```

## Adopting JANG in your project

- Python: [EXAMPLES/python.py](https://github.com/jjang-ai/jangq/blob/main/docs/adoption/EXAMPLES/python.py)
- Swift: [EXAMPLES/swift.swift](https://github.com/jjang-ai/jangq/blob/main/docs/adoption/EXAMPLES/swift.swift)
- Server: [EXAMPLES/server.md](https://github.com/jjang-ai/jangq/blob/main/docs/adoption/EXAMPLES/server.md)
- Framework port: [PORTING.md](https://github.com/jjang-ai/jangq/blob/main/docs/adoption/PORTING.md)

## License

Inherits the license of the base model. JANG conversion does not add new license terms.

---

*Converted by [JANG Studio](https://jangq.ai) on {{ date }}.*
```

---

## Fields reference

| Placeholder | Source |
|---|---|
| `license` | `config.json.license` — or `apache-2.0` if absent |
| `source_hf_repo` | `jang_config.json.source_model.name` |
| `family` | `jang_config.json.quantization.method` — `"jang"` or `"jangtq"` |
| `profile` | `jang_config.json.quantization.profile` |
| `actual_bits` | `jang_config.json.quantization.actual_bits` |
| `block_size` | `jang_config.json.quantization.block_size` (default 64) |
| `size_gb` | `jang_config.json.runtime.total_weight_gb` |
| `has_chat_template` | `tokenizer_config.json.chat_template` is non-null, OR `chat_template.jinja` file exists |
| `has_tool_parser` | `config.json.tool_call_parser` is set |
| `has_reasoning` | `config.json.reasoning_parser` is set, or `config.json.enable_thinking == true` |
| `is_vl` | `preprocessor_config.json` exists in the model directory |
| `is_video_vl` | `video_preprocessor_config.json` exists in the model directory |
| `ms` | Measured first-token latency — run `python -m jang_tools benchmark --model ...` |
| `tok_s` | Measured decode throughput — run `python -m jang_tools benchmark --model ...` |
