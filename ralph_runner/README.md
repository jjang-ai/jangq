# Ralph Runner

Autonomous test harness for JANG Studio. Converts small test models through every JANG and JANGTQ profile on Mac Studio, audits each output, and flags regressions.

## Quick start

```bash
# Activate tier 1 and run next pending combo
cd <repo> && python3 -m ralph_runner.runner --tier 1
python3 -m ralph_runner.runner --next

# Check status
python3 -m ralph_runner.runner --status
```

## How it works

1. Reads `models.yaml` + `profiles.yaml` to build a (model, profile) matrix.
2. SSHes to `macstudio`, rsyncs the local `jang-tools/` source tree.
3. For each combo: downloads source model (if needed), runs `python3 -m jang_tools convert`, saves result.
4. Writes per-run JSON + logs to `results/<date>/<slug>/`.
5. State is tracked in `results/state.json` — each combo is `pending` / `running` / `green` / `failed` / `skipped`.

## Adding a new test model

Append to `models.yaml` under the appropriate tier:

```yaml
- hf_repo: org/model-id
  family: dense | moe | moe_hybrid_ssm | moe_mla | vl_image | vl_video
  archs: [model_type_as_in_config_json]
  approx_gb: 2.5
  has_chat_template: true
```

## Skipping a known-broken combo

Add `skip: "reason"` to the model entry:

```yaml
- hf_repo: org/broken-model
  skip: "fails preflight — config.json missing num_hidden_layers; upstream bug"
```

## Ralph Loop integration

This runner is designed for the Ralph Loop pattern — each `--next` invocation is one iteration.
Inspect each result before starting the next conversion.

```bash
# Reset and re-activate (after fixing a bug)
python3 -m ralph_runner.runner --reset
python3 -m ralph_runner.runner --tier 1

# Run one combo at a time
python3 -m ralph_runner.runner --next
```

## Test assets

`fixtures/` ships with:
- `test_image.png` — 64×64 RGB gradient for A11 (VL preprocessor functional)
- `test_video_frames.npy` — 16×32×32×3 numpy array for A12 (video preprocessor functional)

Both are deterministic and copyright-free; regenerate with the snippets in `audit.py` docstrings if ever lost.
