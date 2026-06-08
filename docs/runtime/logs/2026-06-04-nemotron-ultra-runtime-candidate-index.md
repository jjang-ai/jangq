# Nemotron Ultra Runtime Candidate Index

log_dir: `docs/runtime/logs`
queue_json: `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json`
status: `OPEN`

## Status Counts
- MISSING: `4`

## Lanes
| lane | kind | status | compare | gate | best_tps delta | moe_ms delta | mamba_ms delta | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `moe-routed-shared-scheduling` | `speed_candidate` | `MISSING` | `None` | `None` | `missing` | `missing` | `missing` | docs/runtime/logs/candidate-moe-scheduling: missing speed compare JSON, speed gate JSON, experiment result check JSON |
| `mamba-projection-dispatch` | `speed_candidate` | `MISSING` | `None` | `None` | `missing` | `missing` | `missing` | docs/runtime/logs/candidate-mamba-dispatch: missing candidate directory, speed compare JSON, speed gate JSON, experiment result check JSON |
| `weighted-moe-ablation` | `negative_control` | `MISSING` | `None` | `None` | `missing` | `missing` | `missing` | docs/runtime/logs/candidate-disable-weighted-moe: missing candidate directory, speed compare JSON, speed gate JSON, experiment result check JSON |
| `activation-bf16-ablation` | `negative_control` | `MISSING` | `None` | `None` | `missing` | `missing` | `missing` | docs/runtime/logs/candidate-disable-activation-bf16: missing candidate directory, speed compare JSON, speed gate JSON, experiment result check JSON |

## Candidate Directories
- `moe-routed-shared-scheduling`: `docs/runtime/logs/candidate-moe-scheduling`
- `mamba-projection-dispatch`: `docs/runtime/logs/candidate-mamba-dispatch`
- `weighted-moe-ablation`: `docs/runtime/logs/candidate-disable-weighted-moe`
- `activation-bf16-ablation`: `docs/runtime/logs/candidate-disable-activation-bf16`

## Interpretation
- MISSING means the lane has not produced enough no-load proof to accept or reject.
- ACCEPTED speed_candidate lanes are the only lanes that should be promoted.
- Negative-control lanes should not be treated as speed wins even if their compare status is unusual.
