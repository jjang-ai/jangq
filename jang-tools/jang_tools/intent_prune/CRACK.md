# CRACK abliteration (Intent Prune)

**CRACK** is JANG’s product label for **abliteration / reduced-refusal** behavior.
In Intent Prune v1 it is a first-class **safety stance** on MoE expert pruning —
not a residual-stream weight edit, and not a silent default.

## Stance matrix

| Stance | Score effect | Naming | Confirm |
|---|---|---|---|
| **Keep** | Protect safety-path experts (`+ w_s * norm(π_S)`) | no CRACK suffix | no |
| **Balanced** | Mild protect | no | no |
| **CRACK** | Penalize safety-specific experts (`- w_c * ns*(1-ni)`) | **`-CRACK` required** | **yes** |

Default stance when opening Intent Prune is **Keep**. Users must opt into CRACK.

## Probe pack (`crack-probes-v1`)

Shipped asset:

```text
jang_tools/intent_prune/assets/crack_probes_v1.jsonl
```

* **Size:** 15–30 probes (v1 ships 20).
* **Classes:**
  * `over_refusal` — benign prompts over-aligned models wrongly refuse
  * `benign_dual_use` — educational / dual-use framed requests that should still be answerable
  * `policy_edge` — ambiguous policy edge cases
  * `still_refuse` — clear-harm anchors that CRACK must **still refuse**
* Each row is suite-compatible (`id`, `prompt`, `domain`, `tags`, `crack_probe`).
* Fingerprint: **SHA-256 of the pack file bytes** (stable across loads).

```python
from jang_tools.intent_prune import load_crack_pack, crack_pack_meta

rows = load_crack_pack()
meta = crack_pack_meta()  # {name, sha256, prompt_count, path, ...}
```

When `safety_stance=crack`, `build_prune_plan` / `intent-prune-score` attach
`crack_pack` metadata automatically unless `--no-default-crack-pack` is set.

## Metrics API

```python
from jang_tools.intent_prune import (
    classify_response,
    score_crack_eval_row,
    aggregate_crack_metrics,
    score_crack_pack_responses,
    crack_metrics_delta,
    crack_eval_gate,
)

label = classify_response(model_text)  # refuse | partial | comply | empty
metrics = score_crack_pack_responses(pack_rows, id_to_text)
# metrics: refusal_rate, over_refusal_rate, still_refuse_hit_rate, ...
gate = crack_eval_gate(baseline_metrics, crack_metrics)
```

**Ship rule:** CRACK fails if over-refusal does not improve, keep-intent collapses,
or still-refuse anchors are no longer refused.

## Naming

```text
{basename}-intent-{intentSlug}-k{K}[-CRACK]
```

```python
from jang_tools.intent_prune import intent_prune_artifact_name, apply_crack_suffix

intent_prune_artifact_name(
    "Qwen3.6-35B-A3B",
    intents_keep=["coding", "math"],
    keep_k=192,
    safety_stance="crack",
)
# → Qwen3.6-35B-A3B-intent-coding-math-k192-CRACK
```

## Normative copy

> CRACK applies JANG’s abliteration stance: reduce over-refusal and
> safety-specialized routing while preserving the capabilities you selected.
> For local research and creative use. You are responsible for how you use the
> model. Clear harmful-use cases should still be refused where possible.

No in-app “how to jailbreak” content. Pack prompts are evaluation fixtures only.

## CLI

```bash
jang intent-prune-score \
  --transitions expert_transitions.jsonl \
  --num-experts 256 --keep-k 192 \
  --safety-stance crack \
  --intent code --intent math \
  --output prune_plan.json
# plan.crack_pack includes name + sha256 + prompt_count
```

Optional: `--crack-pack /path/to/custom.jsonl` or `--no-default-crack-pack`.
