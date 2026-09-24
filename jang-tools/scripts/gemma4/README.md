# Gemma 4 (`gemma4_unified`) bundle validation scripts

Helpers for validating MXFP4 / MXFP8 / JANG bundles built by
`jang_tools.convert_gemma4_mxfp` and `jang_tools.convert_gemma4_jang`.

Run with the project venv (`jang-tools/.venv/bin/python`).

## `g4_coherence.py` — text-decoder coherence
Builds a throwaway `gemma4_text` shim from a bundle (remaps
`language_model.model.*`→`model.*`, drops the multimodal embedders, carries the
`quantization` block incl. per-module overrides), then generates with the chat
template. Use this to confirm the quantized **text** decoder is coherent.

```
.venv/bin/python scripts/gemma4/g4_coherence.py <bundle_dir> /tmp/g4_textshim
```

> Gemma 4 12B-it is instruct-only — prompts MUST go through the chat template
> (the script does this). Raw prompts produce gibberish.

## `verify_bundle_integrity.py` — multimodal / tokenizer / audio faithfulness
Confirms all 11 multimodal embedder tensors are present, correctly shaped, free
of inf/nan, and **fp16-range-safe** vs the bf16 source; and that
`tokenizer.json` / `processor_config.json` / `generation_config.json` /
chat template / token IDs are faithful. Use this to rule the bundle bytes out
when debugging a runtime VL/audio bug.

```
.venv/bin/python scripts/gemma4/verify_bundle_integrity.py <source_dir> <bundle_dir>
```

## `validate_dequant.py` — quantization round-trip
Dequantizes a sample of quantized tensors and compares to the source bf16
(relative Frobenius error), checks the `attention_k_eq_v` layout (no `v_proj` on
the 8 full-attention layers), and confirms the norm convention is stored as-is
(NO `+1`).

```
.venv/bin/python scripts/gemma4/validate_dequant.py <source_dir> <bundle_dir> <bits>
```

Validate generation in the target runtime after conversion.
