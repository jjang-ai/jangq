# Nemotron Ultra Runtime Speed Tools

These scripts support the local JANGTQ_1L runtime-speed work for:

`/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`

Run from the repo root with:

```sh
PYTHONPATH=jang-tools jang-tools/.venv/bin/python <script> ...
```

## Start Here

Cheap current-state refresh from saved logs:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py \
  --log-dir docs/runtime/logs \
  --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md
```

Current baseline status:

- best live row: `8.335 tok/s`
- manual synchronized decode: `143.237 ms/token`
- MoE bucket: `65.773 ms`
- Mamba bucket: `64.157 ms`
- attention bucket: `8.990 ms`
- final norm/lm_head: `4.317 ms`
- gate: `PARTIAL`

## No-Load Tools

These read existing JSON logs and do not load the 98G model.

| Script | Purpose | Output |
| --- | --- | --- |
| `runtime_status_report.py` | Compact current status from logs. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-status-report.md` |
| `runtime_proof_manifest.py` | First-open manifest for current status, artifacts, lanes, and commands. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.md` and optional JSON |
| `host_runtime_readiness.py` | Cheap memory pressure, disk, and high-RSS process snapshot before expensive probes. | `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.md` and optional JSON |
| `host_cleanup_runbook.py` | List high-RSS processes with PIDs and safe follow-up commands before model-load probes. | `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.md` and optional JSON |
| `speed_experiment_plan.py` | Rank next speed experiments from measured buckets. | `docs/runtime/logs/2026-06-04-nemotron-ultra-speed-experiment-plan.md` |
| `runtime_issue_ledger.py` | Convert saved proof logs into open/fixed runtime issues, next actions, and acceptance gates. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.md` and optional JSON |
| `token_speed_budget.py` | Convert target token/s values into required MoE/Mamba ms cuts. | `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.md` and optional JSON |
| `component_budget_matrix.py` | Convert component medians into projected token/s gains for 25/50/100% cuts. | `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.md` and optional JSON |
| `runtime_patch_spec.py` | Generate an implementation-facing MoE/Mamba patch spec from the saved evidence. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.md` and optional JSON |
| `runtime_shape_contract.py` | Record bundle layer counts, tensor shapes, bit roles, MTP, cache, and modality invariants. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.md` and optional JSON |
| `runtime_candidate_preflight.py` | Check manifest freshness, host readiness, shape contract, patch spec, and queue before a candidate run. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.md` and optional JSON |
| `runtime_lane_readiness_matrix.py` | Preflight every queue lane and summarize READY/WATCH/BLOCKED status. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md` and optional JSON |
| `runtime_next_runbook.py` | Emit the exact next safe runtime lane, cleanup notes, commands, and proof sequence. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.md` and optional JSON |
| `runtime_experiment_queue.py` | Generate candidate-suite command lanes and required proof artifacts. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.md` and optional JSON |
| `runtime_candidate_index.py` | Summarize candidate directories by lane, compare status, gate status, and experiment verdict. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md` and optional JSON |
| `runtime_candidate_launch_guard.py` | Emit the selected candidate/post-check commands while blocking WATCH/BLOCKED expensive runs by default. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md` and optional JSON |
| `runtime_cleanup_ready_check.py` | Combine host readiness, cleanup runbook, and launch guard into one cleanup-to-ready checklist. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md` and optional JSON |
| `runtime_moe_candidate_contract.py` | Focus the first speed lane into MoE invariants, target cuts, preconditions, and acceptance checks. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.md` and optional JSON |
| `runtime_moe_execution_ticket.py` | Consolidate guard, cleanup, candidate index, and MoE contract into a single run ticket. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md` and optional JSON |
| `runtime_moe_surface_map.py` | Map the JANG loader/kernel/proof symbols that bound the first MoE speed lane. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.md` and optional JSON |
| `runtime_moe_patch_plan.py` | Turn MoE timings and source anchors into an ordered implementation checklist. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md` and optional JSON |
| `runtime_moe_delta_contract.py` | Convert the MoE lane into exact baseline, target, and candidate pass/fail thresholds. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md` and optional JSON |
| `runtime_mamba_candidate_contract.py` | Focus the second speed lane into Mamba invariants, target cuts, preconditions, and acceptance checks. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md` and optional JSON |
| `experiment_result_check.py` | Check a completed candidate lane against queue expectations, speed deltas, coherence, handoff, and cache/parser gates. | candidate `*experiment-result-check.md` and optional JSON |
| `runtime_speed_gate.py` | FIXED/PARTIAL/BLOCKED speed gate over saved logs. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.md` and optional JSON |
| `runtime_speed_fix_acceptance.py` | Strict no-load audit for whether speed work can actually be called fixed. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md` and optional JSON |
| `compare_runtime_speed_logs.py` | Compare baseline vs candidate log dirs. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.md` and optional JSON |
| `agent_handoff_report.py` | Single speed/parser/cache/modality handoff for downstream agents. | `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.md` and optional JSON |
| `runtime_cache_parser_contract.py` | Pin parser, reasoning/tool, hybrid cache, Mamba state, MTP, and text-only gates for speed candidates. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md` and optional JSON |
| `refresh_runtime_proof_bundle.py` | Regenerate status, plan, gate, readiness runbooks, issue ledger, compare, and agent handoff. | `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md` |
| `validate_runtime_log_bundle.py` | Check candidate/baseline logs are complete before comparing. | `docs/runtime/logs/*runtime-log-bundle-validation.md` |

Use `runtime_speed_gate.py --strict` or
`refresh_runtime_proof_bundle.py --strict-gate` when partial bottlenecks should
exit nonzero.

Use `runtime_speed_gate.py --json-out <path>` for automation that should not
scrape markdown.

Use `runtime_proof_manifest.py --json-out <path>` as the first no-load index:
it points to current metrics, lanes, proof artifacts, and refresh commands.

Use `host_runtime_readiness.py --json-out <path>` before expensive model-load
probes when decode looks unusually slow or the machine has many apps open. It
records macOS memory pressure, swap counters, disk headroom, and high-RSS
processes; it does not prove or disprove the saved MoE/Mamba bottleneck by
itself.
Use `host_cleanup_runbook.py --json-out <path>` to list high-RSS process PIDs
and safe follow-up commands. It does not stop or kill processes.
Use `runtime_issue_ledger.py --json-out <path>` as the short implementation
ledger: it names the open MoE, Mamba, coherence, and host issues, preserves the
already-fixed buckets as regression gates, and points to source proof files.

Use `compare_runtime_speed_logs.py --json-out <path>` to persist the same
baseline/candidate deltas, coherence counts, thresholds, wins, and failures as
machine-readable JSON.

Use `agent_handoff_report.py --json-out <path>` when another runtime agent
needs one machine-readable object for current speed buckets, parser state,
cache/VL/MTP gates, runtime toggles, next experiments, and negative controls.
Use `runtime_cache_parser_contract.py --json-out <path>` when a speed candidate
changes scheduling, cache movement, parser streaming, or runtime request
routing. It makes hybrid prefix cache, Mamba companion state, reasoning parser,
tool parser, MTP-disabled behavior, and text-only/VL rejection explicit gates.
Use `token_speed_budget.py --targets 10,12,15 --json-out <path>` to turn target
decode rates into concrete total, MoE, Mamba, and per-layer millisecond cuts.
Use `component_budget_matrix.py --json-out <path>` to rank MoE/Mamba substeps
by projected token/s movement from 25/50/100% cuts before choosing a runtime
patch.
Use `runtime_patch_spec.py --json-out <path>` when handing the work to a
runtime implementer; it lists the MoE and Mamba patch lanes, non-goals,
candidate commands, post-checks, and required proof outputs.
Use `runtime_shape_contract.py --json-out <path>` before changing MoE/Mamba
runtime code; it records the shapes and bit roles that speed patches must
preserve.
Use `runtime_candidate_preflight.py --lane-id moe-routed-shared-scheduling
--json-out <path>` immediately before an expensive candidate suite. With
`--strict`, `WATCH` and `BLOCKED` statuses exit nonzero.
Use `runtime_lane_readiness_matrix.py --json-out <path>` to check all
speed-candidate and negative-control lanes at once.
Use `runtime_next_runbook.py --json-out <path>` after the readiness matrix to
get the exact next lane, host cleanup note, candidate command, post-check, and
proof sequence.
Use `runtime_experiment_queue.py --json-out <path>` before an expensive run to
get exact candidate-suite lanes, env toggles, required outputs, and acceptance
checks.
Use `runtime_candidate_index.py --json-out <path>` after one or more candidate
runs to see which lanes are still missing proof, rejected, blocked, or accepted.
Use `runtime_candidate_launch_guard.py --strict` immediately before an
expensive candidate run. It exits nonzero for `WATCH`/`BLOCKED` unless
`--allow-watch` is passed, but always records the exact candidate and post-check
commands.
Use `runtime_cleanup_ready_check.py --strict` after closing high-RSS apps to
verify host readiness, cleanup state, and the launch guard are all ready.
Use `runtime_moe_candidate_contract.py --json-out <path>` before implementing
the first speed lane; it records the MoE tensor/bit invariants and exact target
cuts that candidate must preserve and improve.
Use `runtime_moe_execution_ticket.py --strict` immediately before running the
MoE lane; it requires the launch guard, cleanup check, and MoE contract to be
READY, then emits the exact candidate, post-check, index, and refresh commands.
Use `runtime_moe_surface_map.py --json-out <path>` when handing the MoE speed
lane to an implementer; it records the JANG loader, routed-kernel,
grouped-NAX, probe, env-toggle, and verdict symbols that bound the work.
Use `runtime_moe_patch_plan.py --json-out <path>` after the surface map to get
the ordered MoE implementation checklist: full path scheduling first, routed
SwitchMLP kernels second, shared experts third.
Use `runtime_moe_delta_contract.py --json-out <path>` after the MoE patch plan
to pin the actual candidate thresholds: baseline ms/token, required MoE cut,
10 tok/s trajectory MoE ceiling, final acceptance ceilings, negative controls,
and exact post-check commands.
Use `runtime_mamba_candidate_contract.py --json-out <path>` before implementing
the second speed lane; it records the Mamba projection/cache-state invariants
and blocks until MoE lane evidence exists.
Use each lane's `post_check_command` after the expensive run to write an
`ACCEPTED` / `REJECTED` / `BLOCKED` no-load verdict. Acceptance requires the
candidate cache/parser contract generated by the refresh step, so a speed row
cannot skip hybrid cache, Mamba state, parser, MTP, or text-only gates.
Use `runtime_speed_fix_acceptance.py --strict` after candidate index refresh to
decide whether the speed objective is actually fixed. It requires an accepted
speed-candidate lane, target token/s, MoE/Mamba bucket ceilings, and a fixed
speed gate by default.

## Expensive Candidate Suite

This loads the real model for multiple probes:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py \
  --candidate-log-dir docs/runtime/logs/candidate-YYYYMMDD-HHMM \
  --baseline-log-dir docs/runtime/logs \
  --lane-id moe-routed-shared-scheduling
```

It writes live speed, layer decode, long coherence, Mamba component, MoE
component, projection tradeoff, log-bundle validation, candidate proof refresh,
agent handoff, baseline compare, and optional experiment result-check artifacts
into the candidate directory.

`--lane-id` should match one `runtime_experiment_queue.py` lane. When present,
the suite runs `experiment_result_check.py --strict` at the end so the candidate
directory gets a final `ACCEPTED`, `REJECTED`, or `BLOCKED` verdict.

Use `--skip-model-probes` only to smoke-test wrapper wiring. It intentionally
does not run reports or comparisons because no candidate JSON logs are created.

## Model-Loading Probes

These load the 98G bundle and should be run deliberately.

| Script | Main question |
| --- | --- |
| `live_speed_probe.py` | What is warm live decode token/s on short prompts? |
| `layer_decode_probe.py` | Which layer families dominate one decode step? |
| `long_decode_coherence_probe.py` | Does longer decode stay coherent and parser-clean? |
| `mamba_component_probe.py` | Which Mamba substeps dominate a decode-shaped hidden state? |
| `moe_component_probe.py` | Which MoE substeps dominate a decode-shaped hidden state? |
| `projection_tradeoff_probe.py` | Are temporary BF16 projection copies faster than 8-bit affine? |
| `startup_warmup_probe.py` | How much TTFT is moved into loader warmup? |
| `switchmlp_fc1_microbench.py` | Does broadcast TQ gather beat the repeat path? |
| `parser_probe.py` | How does template/parser behavior look without full runtime changes? |

## Current Interpretation

Speed is improved but not done. The fixed buckets are attention and final
`lm_head`; the remaining speed targets are MoE routed/shared scheduling and
Mamba projection/dispatch fusion. Do not chase attention first, do not
dequantize 8-bit affine projections as a speed fix, and do not mask coherence
issues with sampler or prompt tricks.
