# jang_tools.jangspec

Bundle format and builder for jang-spec — SSD-streamed MoE speculative decoding.

Refer to the package interfaces below for the bundle contract.

## CLI

```bash
# Build a bundle from a source JANG MoE model
jang spec build /path/to/Qwen3.5-35B-A3B-JANG_2S --out /path/to/out.jangspec

# Inspect a bundle
jang spec inspect /path/to/out.jangspec
```

## Modules

| File | Purpose |
|---|---|
| `format.py` | On-disk constants, struct layouts, magic numbers, alignment |
| `blob.py` | Pack/unpack ExpertBlob (one expert: gate/up/down triples) |
| `index.py` | Flat experts.jsidx writer/reader |
| `manifest.py` | jangspec.json schema + JSON I/O |
| `tier.py` | Classify tensors: hot-core vs streamed experts |
| `builder.py` | JangSpecBuilder end-to-end |
| `reader.py` | JangSpecReader for tests and Swift-parity checks |
| `cli.py` | `jang spec build/inspect` subcommands |

## Format

A `.jangspec` directory contains:

```
<name>.jangspec/
  jangspec.json                 Manifest (bundle_version, tensor lists, sizes)
  tokenizer.json                Shared tokenizer
  tokenizer_config.json
  target/
    config.json                 Copied from source
    jang_config.json            Copied from source
    hot_core.safetensors        Pinned-resident tensors (attn, router, norms, embed, lm_head, shared experts)
    experts.jsidx               Flat binary index: (layer, expert_id) -> (file, offset, nbytes)
    experts-00001.bin           Raw ExpertBlob payloads, 4 KB-aligned
    experts-00002.bin           (rolled at 4 GB per file)
    ...
```

v1 does NOT include `draft/` or `router_prior/`. Those are populated by Plans 2 and 3.
