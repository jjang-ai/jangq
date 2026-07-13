# Intent Prune (`jang_tools.intent_prune`)

Hybrid (MAESTRO-class) path + mass + domain scoring for MoE Intent Prune.

| Module | Role |
|---|---|
| `transitions.py` | `expert_transitions.jsonl` emission / adjacency |
| `graph.py` | Layer-prefixed graph, power iteration, path scores |
| `score.py` | `hybrid_v1` fusion, keep-K, `jang-intent-prune-plan-v1` |
| `crack.py` | CRACK pack load, fingerprint, `-CRACK` naming |
| `metrics.py` | Refusal / over-refusal / still-refuse eval metrics |
| `cli.py` | `jang intent-prune-score` |
| `assets/crack_probes_v1.jsonl` | Versioned CRACK probe pack |

See **[CRACK.md](./CRACK.md)** for abliteration stance, pack classes, metrics
ship rules, and naming.

## Quick score

```bash
jang intent-prune-score \
  --transitions expert_transitions.jsonl \
  --num-experts 256 \
  --keep-k 192 \
  --preset balanced \
  --safety-stance balanced \
  --intent code \
  --output prune_plan.json
```

Safety stances: `keep` | `balanced` | `crack`.
