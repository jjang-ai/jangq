# Nemotron Ultra Runtime Proof Refresh

## Runtime log bundle validation

````text
# Nemotron Ultra Runtime Log Bundle Validation

log_dir: `docs/runtime/logs`
status: `FIXED`

## Found
- found live speed log: 2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json
- found layer decode log: 2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json
- found long coherence log: 2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json
- found mamba component log: 2026-06-04-nemotron-ultra-mamba-component-probe.json
- found moe component log: 2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json
- found projection tradeoff log: 2026-06-04-nemotron-ultra-projection-tradeoff-probe.json

## Failures
````

## Runtime status report

````text
# Nemotron Ultra Runtime Status

log_dir: docs/runtime/logs

## Speed
FIXED/PARTIAL: best observed warm decode row is 8.335 tok/s
source: 2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json :: think_math_default
remaining: MoE/Mamba forward overhead; not sampler or generic generation loop

## Coherence
PARTIAL: visible parser marker leakage, high repeated n-gram fraction, at least one row did not reach EOS
- factual_japan: eos=True speed=8.178 leaks=['</think>'] repeat_fraction=0.5151515151515151
- arithmetic_brief: eos=True speed=8.136 leaks=['</think>'] repeat_fraction=0.43902439024390244
- reasoning_apples: eos=False speed=8.133 leaks=['</think>'] repeat_fraction=0.06451612903225806

## Layer Split
FOUND
manual_decode_total_ms: 143.23724881978706
norm_lm_head_ms: 4.3167500407435
- MoE: total_ms=65.773 count=48 median_ms=1.269
- Mamba: total_ms=64.157 count=48 median_ms=1.241
- Attention: total_ms=8.99 count=12 median_ms=0.747

## Projection Tradeoff
FOUND
Use quantized 8-bit affine projections unless a new probe proves otherwise.
- mamba_in_proj: quantized_median_ms=0.951 bf16_median_ms=1.365 speedup=1.43x
- mamba_out_proj: quantized_median_ms=0.512 bf16_median_ms=0.672 speedup=1.31x
- shared_up: quantized_median_ms=0.351 bf16_median_ms=0.466 speedup=1.33x
- shared_down: quantized_median_ms=0.356 bf16_median_ms=0.545 speedup=1.53x

## Mamba Component
FOUND
- outer_norm: median_ms=0.171
- in_proj: median_ms=0.835
- conv: median_ms=0.216
- ssm_update: median_ms=0.190
- mamba_norm_gated: median_ms=0.178
- out_proj: median_ms=0.470
- full_mamba_mixer: median_ms=1.197
Interpretation: projection/dispatch fusion is a better first target than a Python-level conv rewrite.

## Cache / VL Gates
PARTIAL: cache and VL gates are documented, not live-proven in vMLX.
- TurboQuant KV only covers 12 attention layers.
- Full prefix hit also requires 48 Mamba companion states.
- Parser streaming state must be salted/restored.
- This artifact is text-only; media requests must reject or reroute.
````

## Host runtime readiness

````text
# Nemotron Ultra Host Runtime Readiness

status: `READY`
bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
log_dir: `docs/runtime/logs`

## Memory
- total: `128.0 GiB`
- free_like: `97.8 GiB`
- active: `23.3 GiB`
- wired: `4.3 GiB`
- compressed: `0.7 GiB`
- memory_pressure: `OK`
- swapins: `468875`
- swapouts: `737981`

## Disk
- bundle_volume_free: `1803.0 GiB` at `/Volumes/EricsLLMDrive`
- log_volume_free: `1667.0 GiB` at `/System/Volumes/Data`

## High RSS Processes
- `7.58 GiB` /Applications/Parallels Desktop.app/Contents/MacOS//Parallels VM.app/Contents/MacOS/prl_vm_app --vm-name Windows 11 --uuid {fdb1ad5c-18d7-43fb-a001-f439a8f09eed} --dir-uuid {6c93cd9f-8e88-4769-b638-ec16443e05b4} --log-dir /Users/eric/Parallels/Windows 11.pvm
- `0.89 GiB` /opt/homebrew/lib/node_modules/@openai/codex/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex --yolo resume
- `0.51 GiB` /usr/sbin/spindump
- `0.47 GiB` /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/Metadata.framework/Versions/A/Support/corespotlightd
- `0.30 GiB` /Applications/Discord.app/Contents/Frameworks/Discord Helper (Renderer).app/Contents/MacOS/Discord Helper (Renderer) --type=renderer --user-data-dir=/Users/eric/Library/Application Support/discord --standard-schemes=disclip --secure-schemes=disclip,sentry-ipc --bypasscsp-schemes=sentry-ipc --cors-schemes=sentry-ipc --fetch-schemes=disclip,sentry-ipc --streaming-schemes=disclip --app-path=/Applications/Discord.app/Contents/Resources/app.asar --no-sandbox --no-zygote --enable-blink-features=EnumerateDevices,AudioOutputDevices --autoplay-policy=no-user-gesture-required --lang=en-US --num-raster-threads=4 --enable-zero-copy --enable-gpu-memory-buffer-compositor-resources --enable-main-frame-before-activation --renderer-client-id=6 --time-ticks-at-unix-epoch=-1780706979544332 --launch-time-ticks=1292027421 --shared-files --field-trial-handle=1718379636,r,12230779718568364321,6918296377303196006,262144 --enable-features=ScreenCaptureKitPickerScreen,ScreenCaptureKitStreamPickerSonoma --disable-features=AllowAggressiveThrottlingWithWebSocket,HardwareMediaKeyHandling,IntensiveWakeUpThrottling,MacWebContentsOcclusion,MediaSessionService,ScreenAIOCREnabled,SpareRendererForSitePerProcess,TimeoutHangingVideoCaptureStarts,UseEcoQoSForBackgroundProcess,WinRetrieveSuggestionsOnlyOnDemand --variations-seed-version --enable-node-leakage-in-renderers
- `0.29 GiB` /Applications/Slack.app/Contents/Frameworks/Slack Helper (Renderer).app/Contents/MacOS/Slack Helper (Renderer) --type=renderer --user-data-dir=/Users/eric/Library/Containers/com.tinyspeck.slackmacgap/Data/Library/Application Support/Slack --standard-schemes=app,slack-webapp-dev --enable-sandbox --secure-schemes=app,slack-webapp-dev,sentry-ipc --bypasscsp-schemes=slack-webapp-dev,sentry-ipc --cors-schemes=slack-webapp-dev,sentry-ipc --fetch-schemes=slack-webapp-dev,sentry-ipc --service-worker-schemes=slack-webapp-dev --app-path=/Applications/Slack.app/Contents/Resources/app-arm64.asar --enable-sandbox --enable-blink-features=ExperimentalJSProfiler --disable-blink-features=CustomizableSelect --force-color-profile=display-p3-d65 --lang=en-US --num-raster-threads=4 --enable-zero-copy --enable-main-frame-before-activation --renderer-client-id=4 --time-ticks-at-unix-epoch=-1780706979542896 --launch-time-ticks=1266432014 --shared-files --field-trial-handle=1718379636,r,3221813994612234732,9347406774625481651,262144 --enable-features=DocumentPolicyIncludeJSCallStacksInCrashReports,PdfUseShowSaveFilePicker,ScreenCaptureKitMac,ScreenCaptureKitPickerScreen,ScreenCaptureKitStreamPickerSonoma --disable-features=AllowAggressiveThrottlingWithWebSocket,CalculateNativeWinOcclusion,DropInputEventsWhilePaintHolding,EnableWatermarkView,EnumerateDevicesRelaxedCache,HardwareMediaKeyHandling,IntensiveWakeUpThrottling,LocalNetworkAccessChecks,LogJsConsoleMessages,PostQuantumKyber,RequestInitiatorSiteLockEnfocement,ScreenAIOCREnabled,SkipEmptyDisplayHotplugEvent,SpareRendererForSitePerProcess,TimeoutHangingVideoCaptureStarts,TraceSiteInstanceGetProcessCreation,WebRtcHideLocalIpsWithMdns,WinRetrieveSuggestionsOnlyOnDemand --variations-seed-version --pseudonymization-salt-handle=1935764596,r,90380828058812681,4875636096419484633,4 --trace-process-track-uuid=3190708990060038890 --window-type=main --seatbelt-client=43
- `0.23 GiB` /Applications/Codex.app/Contents/Frameworks/Codex Framework.framework/Versions/149.0.7827.54/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer) --type=renderer --user-data-dir=/Users/eric/Library/Application Support/Codex --standard-schemes=app --secure-schemes=app,sentry-ipc --bypasscsp-schemes=sentry-ipc --fetch-schemes=app,sentry-ipc --cors-schemes=sentry-ipc --streaming-schemes=app --start-stack-profiler --lang=en-US --num-raster-threads=4 --enable-zero-copy --enable-gpu-memory-buffer-compositor-resources --enable-main-frame-before-activation --renderer-client-id=5 --time-ticks-at-unix-epoch=-1780706979544864 --launch-time-ticks=1272797851 --shared-files --metrics-shmem-handle=1752395122,r,4116435887305588721,17327766087610603451,2097152 --field-trial-handle=1718379636,r,12934019734534294067,3320144658749660357,262144 --disable-features=DropInputEventsWhilePaintHolding --variations-seed-version --pseudonymization-salt-handle=1935764596,r,9442993399792598513,1095926252554029526,4 --trace-process-track-uuid=3190708990997080739 --seatbelt-client=115
- `0.23 GiB` /Applications/Codex.app/Contents/MacOS/Codex

## Warnings
- none

## Interpretation
- Saved speed logs still point at MoE and Mamba compute/dispatch as the primary bottlenecks.
- Host pressure can add noise or stalls, so rerun expensive probes only when this report is READY or warnings are understood.
- Closing high-RSS apps may help stability, but it is not proven to fix the current 65.773 ms MoE and 64.157 ms Mamba buckets.

## Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/host_runtime_readiness.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --log-dir docs/runtime/logs`
````

## Host cleanup runbook

````text
# Nemotron Ultra Host Cleanup Runbook

status: `READY`
log_dir: `docs/runtime/logs`
min_rss_gib: `2.0`

## High RSS Processes
- pid `9423` rss `7.58 GiB` tags `vm`: /Applications/Parallels Desktop.app/Contents/MacOS//Parallels VM.app/Contents/MacOS/prl_vm_app --vm-name Windows 11 --uuid {fdb1ad5c-18d7-43fb-a001-f439a8f09eed} --dir-uuid {6c93cd9f-8e88-4769-b638-ec16443e05b4} --log-dir /Users/eric/Parallels/Windows 11.pvm

## Recommended Actions
- Use the owning app UI or service control to stop model servers before loading the 98G Nemotron bundle.
- Do not kill unknown processes blindly; confirm the PID still matches the command immediately before stopping it.
- After cleanup, rerun host_runtime_readiness.py, runtime_lane_readiness_matrix.py, and runtime_next_runbook.py.

## Follow-Up Commands
- host_readiness: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/host_runtime_readiness.py --log-dir docs/runtime/logs`
- lane_matrix: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_lane_readiness_matrix.py --log-dir docs/runtime/logs`
- next_runbook: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_next_runbook.py --log-dir docs/runtime/logs`
````

## Speed experiment plan

````text
# Nemotron Ultra Speed Experiment Plan

log_dir: `docs/runtime/logs`
layer_log: `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
moe_log: `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
mamba_log: `2026-06-04-nemotron-ultra-mamba-component-probe.json`

## Current Bottleneck
- manual synchronized decode: `143.237 ms/token`
- implied synchronized throughput: `6.981 tok/s`
- best live generator row: `8.335 tok/s`
- MoE total: `65.773 ms` across 48 layers
- Mamba total: `64.157 ms` across 48 layers
- attention total: `8.990 ms` across 12 layers
- final norm/lm_head: `4.317 ms`

## Ranked Experiments
| Rank | Experiment | Evidence | Target | Sensitivity |
| --- | --- | --- | --- | --- |
| 1 | MoE routed/shared scheduling or fused decode kernel | `E=65.773 ms`; single-layer `switch_mlp=1.130 ms`, `shared_experts=0.577 ms` | reduce fixed per-layer MoE overhead | 10% MoE cut implies `7.317 tok/s` synchronized |
| 2 | Mamba fused decode kernel / lower-overhead state update | `M=64.157 ms`; first-layer `in_proj=0.835 ms`, `out_proj=0.470 ms`, `conv=0.216 ms`, `ssm=0.190 ms` | reduce projection/dispatch overhead first | 10% Mamba cut implies `7.309 tok/s` synchronized |
| 3 | Joint MoE+Mamba scheduling path | `E+M=129.930 ms` dominates decode | attack Python/MLX dispatch and sync boundaries | 10% combined cut implies `7.678 tok/s` synchronized |
| 4 | Ahead-of-time warmup plan | cold JIT is about 33s, warmed TTFT about 1s | make startup predictable, not steady decode faster | improves TTFT, not tok/s |

## Negative Controls
- Do not chase attention first: it is the smallest bucket after BF16 retention.
- Do not dequantize 8-bit Mamba/shared projections for speed; current probe says quantized is faster.
- Do not lower router top-k as the main fix; top-k 8 did not materially improve live decode.
- Do not replace `mlx_lm.generate_step` with a Python argmax loop; manual loop was slower.
- Do not hide parser/coherence problems with prompt suffixes, forced closing tags, or sampler tricks.

## Projection Evidence
- `mamba_in_proj`: quantized `0.951 ms`, BF16 dequantized `1.365 ms`, quantized speedup `1.43x`
- `mamba_out_proj`: quantized `0.512 ms`, BF16 dequantized `0.672 ms`, quantized speedup `1.31x`
- `shared_up`: quantized `0.351 ms`, BF16 dequantized `0.466 ms`, quantized speedup `1.33x`
- `shared_down`: quantized `0.356 ms`, BF16 dequantized `0.545 ms`, quantized speedup `1.53x`

## Mamba Component Evidence
- `outer_norm`: `0.171 ms`
- `in_proj`: `0.835 ms`
- `conv`: `0.216 ms`
- `ssm_update`: `0.190 ms`
- `mamba_norm_gated`: `0.178 ms`
- `out_proj`: `0.470 ms`
- `full_mamba_mixer`: `1.197 ms`
Interpretation: the generic grouped conv and SSM update are not the largest isolated Mamba substeps in this probe; projection/dispatch fusion is the more credible first Mamba speed target.

## Next Proof Rows
- rerun layer decode after any MoE or Mamba runtime change
- rerun live speed probe after warm compile
- rerun long coherence probe; speed wins cannot regress parser/coherence
- keep cache/VL proof separate from speed proof
````

## Token speed budget

````text
# Nemotron Ultra Token Speed Budget

log_dir: `docs/runtime/logs`
layer_log: `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`

## Current Baseline
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- best_live_tps: `8.335` from `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- other_ms: `0.000`
- moe_plus_mamba: `129.930` ms (`90.71%` of manual decode)

## Target Budgets
| target tok/s | target ms/token | total cut needed | total cut % | MoE cut | Mamba cut | per MoE layer | per Mamba layer | MoE/Mamba enough |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `10.000` | `100.000` | `43.237` | `30.19%` | `21.888` (`33.28%`) | `21.350` (`33.28%`) | `0.4560` | `0.4448` | `True` |
| `12.000` | `83.333` | `59.904` | `41.82%` | `30.325` (`46.10%`) | `29.579` (`46.10%`) | `0.6318` | `0.6162` | `True` |
| `15.000` | `66.667` | `76.571` | `53.46%` | `38.762` (`58.93%`) | `37.809` (`58.93%`) | `0.8075` | `0.7877` | `True` |

## Interpretation
- Use manual synchronized decode for millisecond budgets; use live speed for user-visible baseline.
- If a target is not reachable by MoE/Mamba only, attention/lm_head/loop work also must move.
- Per-layer cuts are proportional planning numbers, not proof of an implementation strategy.
````

## Component budget matrix

````text
# Nemotron Ultra Component Budget Matrix

log_dir: `docs/runtime/logs`

## Sources
- layer: `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- moe: `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- mamba: `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- projection: `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- token_budget: `2026-06-04-nemotron-ultra-token-speed-budget.json`

## Current Baseline
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_mamba_pct: `90.710`

## Component Cut Scenarios
| family | role | component | per-layer median | projected total | family coverage | 25% cut tps | 50% cut tps | 100% cut tps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoE | `inclusive_path` | `full_moe` | `2.107` | `101.148` | `153.8%` | `8.478` | `10.792` | `23.759` |
| MoE | `substep` | `switch_mlp` | `1.130` | `54.264` | `82.5%` | `7.712` | `8.613` | `11.239` |
| MoE | `substep` | `shared_experts` | `0.577` | `27.720` | `42.1%` | `7.336` | `7.729` | `8.657` |
| MoE | `substep` | `fc1_latent_proj` | `0.231` | `11.068` | `16.8%` | `7.119` | `7.262` | `7.566` |
| MoE | `substep` | `gate` | `0.221` | `10.602` | `16.1%` | `7.113` | `7.250` | `7.539` |
| MoE | `substep` | `fc2_latent_proj` | `0.212` | `10.172` | `15.5%` | `7.108` | `7.238` | `7.515` |
| MoE | `substep` | `norm` | `0.203` | `9.732` | `14.8%` | `7.102` | `7.227` | `7.490` |
| MoE | `substep` | `score_weighted_sum` | `0.176` | `8.468` | `12.9%` | `7.086` | `7.194` | `7.420` |
| Mamba | `inclusive_path` | `full_mamba_mixer` | `1.197` | `57.480` | `89.6%` | `7.760` | `8.734` | `11.661` |
| Mamba | `substep` | `in_proj` | `0.835` | `40.062` | `62.4%` | `7.506` | `8.116` | `9.692` |
| Mamba | `substep` | `out_proj` | `0.470` | `22.564` | `35.2%` | `7.268` | `7.578` | `8.287` |
| Mamba | `substep` | `conv` | `0.216` | `10.360` | `16.1%` | `7.110` | `7.243` | `7.526` |
| Mamba | `substep` | `ssm_update` | `0.190` | `9.124` | `14.2%` | `7.094` | `7.211` | `7.456` |
| Mamba | `substep` | `mamba_norm_gated` | `0.178` | `8.544` | `13.3%` | `7.087` | `7.196` | `7.424` |
| Mamba | `substep` | `outer_norm` | `0.171` | `8.216` | `12.8%` | `7.083` | `7.188` | `7.406` |

## Target Coverage
- `10.000` tok/s needs `43.237` ms cut; single measured row/path enough: `MoE:full_moe`, `MoE:switch_mlp`, `Mamba:full_mamba_mixer`
- `12.000` tok/s needs `59.904` ms cut; single measured row/path enough: `MoE:full_moe`
- `15.000` tok/s needs `76.571` ms cut; single measured row/path enough: `MoE:full_moe`

## Projection Tradeoff
- `mamba_in_proj`: quantized `0.951` ms, BF16 `1.365` ms, speedup `1.43x`
- `mamba_out_proj`: quantized `0.512` ms, BF16 `0.672` ms, speedup `1.31x`
- `shared_up`: quantized `0.351` ms, BF16 `0.466` ms, speedup `1.33x`
- `shared_down`: quantized `0.356` ms, BF16 `0.545` ms, speedup `1.53x`

## Interpretation
- Component/path totals are projected from first measured layer medians; use them for ranking, not final proof.
- `full_*` rows are inclusive path measurements, not additive leaf substeps.
- Rows with large projected totals are plausible speed targets; small rows cannot move token/s enough alone.
- The current projection tradeoff says quantized 8-bit affine projections are faster than temporary BF16 copies.
````

## Runtime experiment queue

````text
# Nemotron Ultra Runtime Experiment Queue

baseline_log_dir: `docs/runtime/logs`
bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
candidate_root: `docs/runtime/logs`

## Current Baseline
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- moe_plus_mamba_pct_of_total: `90.71%`

## Lanes
### MoE routed/shared scheduling
- id: `moe-routed-shared-scheduling`
- kind: `speed_candidate`
- goal: Reduce MoE bucket without changing routing semantics or routed expert bit layout.
- evidence: MoE is 65.773 ms; first target needs about 21.888 ms MoE reduction for 10.0 tok/s.
- patch_surface: JANG loader/TurboQuant MoE scheduling only; no bundle expansion.
- env: none
- expected_compare_statuses: `IMPROVED`
- command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: candidate compare status is IMPROVED with no failures; MoE bucket improves and Mamba/attention/lm_head do not materially regress; long coherence leak/repeat/no_eos counts do not regress; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

### Mamba projection/dispatch fusion
- id: `mamba-projection-dispatch`
- kind: `speed_candidate`
- goal: Reduce Mamba bucket by attacking projection and dispatch overhead before conv rewrites.
- evidence: Mamba is 64.157 ms; first target needs about 21.350 ms Mamba reduction for 10.0 tok/s.
- patch_surface: JANG loader/runtime Mamba path only; keep 8-bit affine projections unless new proof reverses it.
- env: none
- expected_compare_statuses: `IMPROVED`
- command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: candidate compare status is IMPROVED with no failures; Mamba bucket improves and MoE/attention/lm_head do not materially regress; long coherence leak/repeat/no_eos counts do not regress; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

### Weighted MoE fast-path A/B
- id: `weighted-moe-ablation`
- kind: `negative_control`
- goal: Confirm the current weighted-MoE default remains beneficial after any MoE refactor.
- evidence: Weighted MoE is a small positive/noisy improvement and must not silently regress.
- patch_surface: A/B proof lane only; no code change implied.
- env: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- expected_compare_statuses: `FAIL`, `UNCHANGED`
- command: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: compare must not be treated as a speed fix; if the lane is faster, preserve evidence before changing the default; long coherence leak/repeat/no_eos counts do not regress; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

### BF16 activation retention guard
- id: `activation-bf16-ablation`
- kind: `negative_control`
- goal: Guard the large lm_head/activation dtype speed fix from accidental rollback.
- evidence: BF16 retention moved synchronized decode from about 320 ms/token to about 144 ms/token.
- patch_surface: A/B proof lane only; should be slower and marked as a negative-control regression.
- env: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- expected_compare_statuses: `FAIL`
- command: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: compare should fail or clearly regress speed versus baseline; lm_head/norm and manual decode regressions confirm BF16 retention is still required; do not promote this lane as a candidate fix; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

## Notes
- Run one candidate lane at a time; the model is about 98G and probes are expensive.
- Lane-specific env vars are already embedded in the generated command.
- Do not call a speed lane fixed until compare, gate, and long-coherence rows agree.
- Negative-control lanes are guards; expected regressions should not be promoted as fixes.
````

## Runtime shape contract

````text
# Nemotron Ultra Runtime Shape Contract

bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
log_dir: `docs/runtime/logs`
status: `READY`

## Architecture
- family: `nemotron_h`
- modality: `text`
- cache_type: `hybrid`
- hidden_size: `8192`
- vocab_size: `131072`
- tie_word_embeddings: `False`
- num_hidden_layers: `108`
- layer_counts: `{"attention": 12, "mamba": 48, "moe": 48, "total": 108}`
- attention_heads: `64`
- key_value_heads: `2`
- num_experts_per_tok: `22`
- moe_intermediate_size: `5120`
- ssm_state_size: `128`

## Quantization
- mxtq_bits: `{"mamba_projection": 8, "routed_expert": {"down_proj": 1, "up_proj": 1}, "shared_expert": 8}`
- method: `routed-tq1-control-plane-preserved`
- drops_mtp: `True`
- estimated_output_gib: `98.35`
- fp8_projection_affine_bits: `8`
- fp8_projection_group_size: `128`
- keeps_attention_bf16: `True`
- keeps_latent_moe_bf16: `True`
- keeps_router_gates_source_precision: `True`
- shard_count: `51`

## Mamba Decode Contract
- layer_index: `0`
- cache_ordinal: `0`
- hidden_shape: `[1, 1, 8192]`
- normed_shape: `[1, 1, 8192]`
- projected_shape: `[1, 1, 35072]`
- gate_shape: `[1, 1, 16384]`
- conv_dim: `18432`
- conv_input_shape: `[1, 1, 18432]`
- conv_output_shape: `[1, 1, 18432]`
- ssm_out_shape: `[1, 1, 16384]`
- intermediate_size: `16384`
- num_heads: `256`
- n_groups: `8`
- ssm_state_size: `128`

## MoE Decode Contract
- layer_index: `1`
- hidden_shape: `[1, 1, 8192]`
- scores_shape: `[1, 1, 22]`
- indices_shape: `[1, 1, 22]`
- latent_shape: `[1, 1, 2048]`
- routed_shape: `[1, 1, 22, 2048]`

## Preserve
- 48 Mamba companion cache entries plus 12 attention KV entries for hybrid prefix cache
- MTP remains dropped; speculative draft KV/SSM state is out of scope for this bundle
- routed expert up/down projections remain 1-bit according to mxtq_bits
- shared expert and Mamba projection paths remain 8-bit unless new proof reverses the projection tradeoff
- router gates and attention BF16 retention remain source-precision/preserved runtime surfaces
- text-only modality remains explicit; media requests must reject or reroute
````

## Runtime patch spec

````text
# Nemotron Ultra Runtime Patch Spec

log_dir: `docs/runtime/logs`
current_status: `PARTIAL`

## Current Speed State
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_mamba_pct: `90.710`

## Target Cuts
- `10.000` tok/s needs `43.237` ms cut; single measured row/path enough: `MoE:full_moe`, `MoE:switch_mlp`, `Mamba:full_mamba_mixer`
- `12.000` tok/s needs `59.904` ms cut; single measured row/path enough: `MoE:full_moe`
- `15.000` tok/s needs `76.571` ms cut; single measured row/path enough: `MoE:full_moe`

## Implementation Lanes
### 1. MoE routed/shared scheduling and switch_mlp path
- id: `moe-routed-shared-scheduling`
- implementation_surface: JANG loader/TurboQuant MoE scheduling only; no bundle expansion.
- why:
  - MoE bucket is 65.773 ms across 48 layers.
  - `switch_mlp` projects to 54.264 ms; 50% cut implies 8.613 tok/s synchronized.
  - `full_moe` is an inclusive path row at 101.148 ms, so path-level scheduling is the highest-leverage MoE target.
- do:
  - preserve router top-k and weighted expert semantics
  - preserve routed expert 1-bit layout and shared expert 8-bit layout
  - reduce per-layer dispatch/synchronization around routed/shared expert execution
  - measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate
- do_not:
  - do not lower router top-k as the primary speed fix
  - do not expand quantized experts to full precision
  - do not promote a change unless long-coherence counts do not regress
- proof:
  - candidate_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
  - post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
  - expected_compare_statuses: `IMPROVED`
  - required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

### 2. Mamba projection/dispatch path
- id: `mamba-projection-dispatch`
- implementation_surface: JANG loader/runtime Mamba path only; keep 8-bit affine projections unless new proof reverses it.
- why:
  - Mamba bucket is 64.157 ms across 48 layers.
  - `full_mamba_mixer` projects to 57.480 ms; 50% cut implies 8.734 tok/s synchronized.
  - `in_proj` projects to 40.062 ms; it is larger than conv/SSM update in the saved component probe.
- do:
  - attack projection/dispatch overhead before grouped conv rewrites
  - preserve 8-bit affine projection path unless a new projection tradeoff probe reverses the result
  - preserve Mamba companion cache/state order for hybrid prefix cache compatibility
  - measure M bucket plus `in_proj`, `out_proj`, `conv`, and `ssm_update` after every candidate
- do_not:
  - do not dequantize Mamba projections to BF16 as a default speed fix
  - do not treat attention KV cache work as a substitute for Mamba state proof
  - do not change cache topology without rerunning cache/block handoff checks
- proof:
  - candidate_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
  - post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
  - expected_compare_statuses: `IMPROVED`
  - required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Global Non-Goals
- do not chase attention first while attention is 8.990 ms and below the gate ceiling
- do not hide parser/coherence failures with prompts, forced tags, or sampler tweaks
- do not enable MTP/speculative decode for this MTP-dropped bundle
- do not call a speed lane fixed without compare, gate, and long-coherence proof

## Runtime Controls
- disable_weighted_moe_fastpath: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- disable_activation_bf16: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- disable_switchmlp_fastpath: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
````

## Runtime speed gate

````text
# Nemotron Ultra Runtime Speed Gate

log_dir: `docs/runtime/logs`
status: `PARTIAL`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Failures

## Current Buckets
- best_live: `8.335 tok/s` from `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
````

## Runtime speed compare

````text
# Nemotron Ultra Runtime Speed Compare

baseline_log_dir: `docs/runtime/logs`
candidate_log_dir: `docs/runtime/logs`
status: `UNCHANGED`

## Metric Deltas
- `best_tps`: baseline `8.335` tok/s, candidate `8.335` tok/s, delta `+0.000` (+0.00%), same
- `manual_decode_total_ms`: baseline `143.237` ms, candidate `143.237` ms, delta `+0.000` (+0.00%), same
- `moe_ms`: baseline `65.773` ms, candidate `65.773` ms, delta `+0.000` (+0.00%), same
- `mamba_ms`: baseline `64.157` ms, candidate `64.157` ms, delta `+0.000` (+0.00%), same
- `attention_ms`: baseline `8.990` ms, candidate `8.990` ms, delta `+0.000` (+0.00%), same
- `norm_lm_head_ms`: baseline `4.317` ms, candidate `4.317` ms, delta `+0.000` (+0.00%), same

## Coherence Counts
- `leaks`: baseline `3`, candidate `3`
- `repeats`: baseline `2`, candidate `2`
- `no_eos`: baseline `1`, candidate `1`

## Wins

## Failures
````

## Agent runtime handoff

````text
# Nemotron Ultra Agent Runtime Handoff

bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
log_dir: `docs/runtime/logs`
handoff_status: `PARTIAL`
speed_gate: `PARTIAL`

## Artifact
- profile: `JANGTQ_1L`
- format: `jangtq` `2.0`
- estimated_output_gib: `98.35`
- shard_count: `51`
- drops_mtp: `True`
- capabilities: `{"cache_type": "hybrid", "family": "nemotron_h", "modality": "text", "reasoning_parser": "deepseek_r1", "supports_thinking": true, "supports_tools": true, "think_in_template": true, "tool_parser": "nemotron"}`
- mxtq_bits: `{"mamba_projection": 8, "routed_expert": {"down_proj": 1, "up_proj": 1}, "shared_expert": 8}`

## Topology
- layers_total: `108`
- mamba/moe/attention: `48` / `48` / `12`
- cache_entries: `60` = `48` Mamba companion states + `12` attention KV entries

## Current Speed Buckets
- best_live_tps: `8.335` from `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Parser And Coherence
- parser_status: `PARTIAL` marker_leak_rows=['nt_capital_default'] truncated_reasoning_rows=['think_math_default'] tool_rows=0
- long_coherence_status: `PARTIAL` leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'] repeats=['factual_japan', 'arithmetic_brief'] no_eos=['reasoning_apples']

## Token Speed Budgets
- target `10.000` tok/s: cut `43.237` ms total; proportional MoE `21.888` ms, Mamba `21.350` ms
- target `12.000` tok/s: cut `59.904` ms total; proportional MoE `30.325` ms, Mamba `29.579` ms
- target `15.000` tok/s: cut `76.571` ms total; proportional MoE `38.762` ms, Mamba `37.809` ms

## Runtime Controls
- disable_switchmlp_fastpath: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
- legacy_disable_switchmlp_fastpath: `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`
- disable_activation_bf16: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- disable_weighted_moe_fastpath: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`

## Cache And Modality Gates
- cache_type: `hybrid`
- text_only: `True`
- kv_cache_boundary: `TurboQuant KV applies only to attention KV entries.`
- mamba_state_boundary: `Prefix hits require matching Mamba companion states.`
- vl_policy: `Reject or reroute media requests; this artifact has no VL/audio tensors or processor configs.`
- mtp_policy: `Disabled for this bundle; draft KV/SSM state is out of scope.`

## Next Experiments
- MoE routed/shared scheduling or fused decode kernel
- Mamba projection/dispatch fusion or fused decode state update
- Joint MoE+Mamba dispatch-boundary reduction
- rerun layer decode and live speed after any runtime change
- rerun long coherence; speed wins must not regress parser-visible output

## Negative Controls
- Do not chase attention first while it remains under the current ceiling.
- Do not dequantize 8-bit affine projections as a speed fix without new proof.
- Do not lower router top-k as the main fix; saved top-k probe did not materially improve decode.
- Do not hide parser/coherence failures with prompt suffixes or sampler tricks.
- Do not enable speculative/MTP decode for this MTP-dropped bundle.
````

## Runtime cache parser contract

````text
# Nemotron Ultra Cache Parser Contract

log_dir: `docs/runtime/logs`
status: `PARTIAL`
speed_acceptance_status: `PARTIAL`
candidate_status: `OPEN`

## Cache Contract
- cache_type: `hybrid`
- cache_entries: `60`
- mamba_companion_state_entries: `48`
- attention_kv_cache_entries: `12`
- mamba_layers: `48`
- attention_layers: `12`
- kv_cache_boundary: `TurboQuant KV applies only to attention KV entries.`
- mamba_state_boundary: `Prefix hits require matching Mamba companion states.`
- prefix_cache_acceptance: `["attention KV hit is insufficient without the matching 48 Mamba companion states", "cache restore must preserve layer order and cache ordinal mapping", "parser streaming state must be salted/restored with the cache key"]`

## Parser Contract
- reasoning_parser: `deepseek_r1`
- tool_parser: `nemotron`
- supports_thinking: `True`
- supports_tools: `True`
- think_in_template: `True`
- parser_probe: `{"marker_leak_rows": ["nt_capital_default"], "parser": "deepseek_r1 compatible <think> parser + Ultra XML function calls", "rows": 3, "status": "PARTIAL", "tool_rows": 0, "truncated_reasoning_rows": ["think_math_default"]}`
- long_coherence: `{"leak_rows": ["factual_japan", "arithmetic_brief", "reasoning_apples"], "no_eos_rows": ["reasoning_apples"], "repeat_rows": ["factual_japan", "arithmetic_brief"], "rows": 3, "status": "PARTIAL"}`
- acceptance: `["no new visible <think>, </think>, <tool_call>, or <tool_response> leakage", "no truncated reasoning rows versus baseline", "tool-call parser remains Nemotron XML compatible", "do not hide parser failures with prompt suffixes, forced tags, or sampler penalties"]`

## Modality And MTP
- modality: `text`
- text_only: `True`
- vl_policy: `Reject or reroute media requests; this artifact has no VL/audio tensors or processor configs.`
- audio_policy: `No audio tensors or processor configs are present; reject or reroute audio requests.`
- mtp_policy: `Disabled for this bundle; draft KV/SSM state is out of scope.`
- drops_mtp: `True`

## Partial
- parser probe is PARTIAL
- long coherence is PARTIAL
- speed fix acceptance is PARTIAL
- candidate index is OPEN

## Failures
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-parser-probe.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
````

## Runtime proof manifest preflight input

````text
# Nemotron Ultra Runtime Proof Manifest

log_dir: `docs/runtime/logs`
status: `PARTIAL`

## Current Metrics
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_plus_mamba_pct_of_total: `90.710`
- best_live_source: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Lanes
- `moe-routed-shared-scheduling` (speed_candidate): MoE routed/shared scheduling; expected=['IMPROVED']
- `mamba-projection-dispatch` (speed_candidate): Mamba projection/dispatch fusion; expected=['IMPROVED']
- `weighted-moe-ablation` (negative_control): Weighted MoE fast-path A/B; expected=['FAIL', 'UNCHANGED']
- `activation-bf16-ablation` (negative_control): BF16 activation retention guard; expected=['FAIL']

## Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- strict_gate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_gate.py --log-dir docs/runtime/logs --strict`

## Artifacts
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`: present, optional, combined no-load refresh output
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-log-bundle-validation.md`: present, optional, required log presence validation
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-status-report.md`: present, optional, human-readable current status
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`: present, optional, host memory/disk readiness snapshot
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.md`: present, optional, human-readable host readiness snapshot
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`: present, optional, host cleanup runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.md`: present, optional, human-readable host cleanup runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-speed-experiment-plan.md`: present, optional, ranked speed experiment plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`: present, optional, runtime issue ledger
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.md`: present, optional, human-readable runtime issue ledger
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`: present, optional, runtime candidate index
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md`: present, optional, human-readable runtime candidate index
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`: present, optional, runtime candidate launch guard
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md`: present, optional, human-readable runtime candidate launch guard
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json`: present, optional, runtime cleanup ready check
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md`: present, optional, human-readable runtime cleanup ready check
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`: present, optional, MoE candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.md`: present, optional, human-readable MoE candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`: present, optional, MoE execution ticket
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md`: present, optional, human-readable MoE execution ticket
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.json`: present, optional, MoE runtime source surface map
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.md`: present, optional, human-readable MoE runtime source surface map
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json`: present, optional, MoE runtime patch plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md`: present, optional, human-readable MoE runtime patch plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json`: present, optional, MoE runtime delta acceptance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md`: present, optional, human-readable MoE runtime delta acceptance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json`: present, optional, Mamba candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md`: present, optional, human-readable Mamba candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`: present, required, target token/s millisecond budgets
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.md`: present, optional, human-readable target budgets
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.json`: present, optional, component-level token/s sensitivity matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.md`: present, optional, human-readable component budget matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`: present, optional, implementation-facing runtime patch spec
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.md`: present, optional, human-readable runtime patch spec
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`: present, optional, runtime shape and bit contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.md`: present, optional, human-readable runtime shape contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.json`: present, optional, runtime candidate preflight
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.md`: present, optional, human-readable candidate preflight
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json`: present, optional, all-lane runtime readiness matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md`: present, optional, human-readable lane readiness matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`: present, optional, next runtime runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.md`: present, optional, human-readable next runtime runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json`: present, required, machine-readable experiment queue
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.md`: present, optional, human-readable experiment queue
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`: present, required, machine-readable speed gate
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.md`: present, optional, human-readable speed gate
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`: present, optional, runtime speed fix acceptance audit
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md`: present, optional, human-readable speed fix acceptance audit
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.json`: present, optional, baseline-vs-baseline compare for current logs
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.md`: present, optional, human-readable compare report
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`: present, optional, cache/parser/runtime nuance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md`: present, optional, human-readable cache/parser/runtime nuance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.json`: present, required, machine-readable downstream agent handoff
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.md`: present, optional, human-readable downstream agent handoff
- `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`: present, required, best live decode source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`: present, required, layer decode bucket source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`: present, required, long coherence source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-mamba-component-probe.json`: present, required, Mamba component source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`: present, required, MoE component source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`: present, required, projection tradeoff source
````

## Runtime candidate preflight

````text
# Nemotron Ultra Runtime Candidate Preflight

lane_id: `moe-routed-shared-scheduling`
log_dir: `docs/runtime/logs`
status: `READY`

## Fixed
- found manifest: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json
- found host readiness: docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json
- found shape contract: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json
- found patch spec: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json
- found experiment queue: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json
- lane moe-routed-shared-scheduling is present in required lane registries
- manifest status is PARTIAL
- shape contract is READY
- host readiness is READY

## Warnings
- none

## Failures
- none

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- dry_run: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96 --dry-run`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Required Outputs
- `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- `2026-06-04-nemotron-ultra-runtime-speed-compare.json`
- `2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `2026-06-04-nemotron-ultra-token-speed-budget.json`
- `2026-06-04-nemotron-ultra-agent-handoff.json`
- `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Expected Compare Statuses
- `IMPROVED`
````

## Runtime lane readiness matrix

````text
# Nemotron Ultra Runtime Lane Readiness Matrix

log_dir: `docs/runtime/logs`
status: `READY`
queue: `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json`

## Lanes
| lane | kind | status | warnings | failures | expected compare |
| --- | --- | --- | --- | --- | --- |
| `moe-routed-shared-scheduling` | `speed_candidate` | `READY` | `0` | `0` | `IMPROVED` |
| `mamba-projection-dispatch` | `speed_candidate` | `READY` | `0` | `0` | `IMPROVED` |
| `weighted-moe-ablation` | `negative_control` | `READY` | `0` | `0` | `FAIL`, `UNCHANGED` |
| `activation-bf16-ablation` | `negative_control` | `READY` | `0` | `0` | `FAIL` |

## Commands
- `moe-routed-shared-scheduling` candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `moe-routed-shared-scheduling` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- `mamba-projection-dispatch` candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `mamba-projection-dispatch` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- `weighted-moe-ablation` candidate: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `weighted-moe-ablation` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- `activation-bf16-ablation` candidate: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `activation-bf16-ablation` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Warnings And Failures
### moe-routed-shared-scheduling
- warning: none
- failure: none
### mamba-projection-dispatch
- warning: none
- failure: none
### weighted-moe-ablation
- warning: none
- failure: none
### activation-bf16-ablation
- warning: none
- failure: none

## Interpretation
- Run speed_candidate lanes for proposed runtime fixes; run negative_control lanes as guards after related changes.
- WATCH means proof wiring is usable but host/readiness warnings should be handled before loading the 98G bundle.
- BLOCKED means missing or stale proof files must be refreshed before a candidate suite.
````

## Runtime next runbook

````text
# Nemotron Ultra Runtime Next Runbook

log_dir: `docs/runtime/logs`
runbook_status: `READY`
current_runtime_status: `PARTIAL`
host_status: `READY`
shape_status: `READY`

## Next Lane
- id: `moe-routed-shared-scheduling`
- kind: `speed_candidate`
- status: `READY`
- title: MoE routed/shared scheduling

## Why This Lane
- MoE bucket is 65.773 ms across 48 layers.
- `switch_mlp` projects to 54.264 ms; 50% cut implies 8.613 tok/s synchronized.
- `full_moe` is an inclusive path row at 101.148 ms, so path-level scheduling is the highest-leverage MoE target.

## Host Cleanup
- Close or stop the high-RSS vMLX server before loading the 98G Nemotron bundle.
- Rerun host_runtime_readiness.py and runtime_lane_readiness_matrix.py after cleanup.
- Proceed with the candidate command only when the selected lane is READY, or consciously accept WATCH noise.

## Do
- preserve router top-k and weighted expert semantics
- preserve routed expert 1-bit layout and shared expert 8-bit layout
- reduce per-layer dispatch/synchronization around routed/shared expert execution
- measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate

## Do Not
- do not lower router top-k as the primary speed fix
- do not expand quantized experts to full precision
- do not promote a change unless long-coherence counts do not regress

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Proof Sequence
- Refresh no-load proof bundle.
- Run runtime_lane_readiness_matrix.py.
- Run exactly one speed_candidate lane.
- Run that lane's post_check_command.
- Accept only IMPROVED compare status with no long-coherence/cache/modality regressions.
````

## Runtime issue ledger

````text
# Nemotron Ultra Runtime Issue Ledger

log_dir: `docs/runtime/logs`
status: `OPEN`
current_runtime_status: `PARTIAL`

## Status Counts
- OPEN: `3`
- BLOCKED: `0`
- FIXED: `2`

## Target Summary
- 10.000 tok/s needs 43.237 ms total cut (MoE 21.888 ms, Mamba 21.350 ms proportional).
- 12.000 tok/s needs 59.904 ms total cut (MoE 30.325 ms, Mamba 29.579 ms proportional).
- 15.000 tok/s needs 76.571 ms total cut (MoE 38.762 ms, Mamba 37.809 ms proportional).

## Issues

### NU-SPEED-001: MoE routed/shared decode path dominates token latency
- status: `OPEN`
- severity: `critical`
- evidence:
  - MoE bucket is 65.773 ms.
  - Current next speed lane is `moe-routed-shared-scheduling`: MoE routed/shared scheduling.
  - 10.000 tok/s needs 43.237 ms total cut (MoE 21.888 ms, Mamba 21.350 ms proportional).
- next_actions:
  - Run exactly one MoE scheduling candidate after host readiness is READY or WATCH is accepted.
  - Preserve router top-k, weighted expert semantics, routed 1-bit experts, and shared 8-bit experts.
  - Compare candidate logs with compare_runtime_speed_logs.py and experiment_result_check.py.
- acceptance_checks:
  - runtime-speed compare status is IMPROVED.
  - MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.
  - long coherence leak/repeat/EOS counts do not regress.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`

### NU-SPEED-002: Mamba projection/dispatch path dominates token latency
- status: `OPEN`
- severity: `critical`
- evidence:
  - Mamba bucket is 64.157 ms.
  - Speed gate records projection/dispatch as the current Mamba target before conv rewrite.
  - 12.000 tok/s needs 59.904 ms total cut (MoE 30.325 ms, Mamba 29.579 ms proportional).
- next_actions:
  - Keep this as the second speed-candidate lane after MoE scheduling evidence is gathered.
  - Preserve projected shape [1, 1, 35072], gate shape [1, 1, 16384], SSM state size 128, groups 8.
  - Recheck mamba_component_probe.py and layer_decode_probe.py after any candidate.
- acceptance_checks:
  - Mamba bucket drops without changing cache cardinality or Mamba shape contract.
  - Attention and norm/lm_head remain below current ceilings.
  - long coherence leak/repeat/EOS counts do not regress.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`

### NU-COHERENCE-001: Long decode still leaks/repeats or misses EOS
- status: `OPEN`
- severity: `high`
- evidence:
  - MoE remains a bottleneck at 65.773 ms Mamba remains a bottleneck at 64.157 ms coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])
- next_actions:
  - Treat coherence as a regression gate for speed candidates, not a sampler/prompt masking target.
  - Use long_decode_coherence_probe.py after any accepted speed candidate.
- acceptance_checks:
  - No visible thinking marker leaks.
  - Repeat fraction stays below gate threshold.
  - Expected rows reach EOS within the probe limit.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`

### NU-HOST-001: Host RAM/process state can add noise to expensive probes
- status: `FIXED`
- severity: `medium`
- evidence:
  - host_runtime_readiness status is `READY`.
- next_actions:
  - Use host_cleanup_runbook.py before loading the 98G bundle.
  - Rerun host_runtime_readiness.py, runtime_lane_readiness_matrix.py, and runtime_next_runbook.py after cleanup.
- acceptance_checks:
  - Host readiness is READY, or WATCH is explicitly accepted before candidate timing.
  - No unrelated high-RSS model server is competing with the Nemotron candidate run.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`

### NU-FIXED-001: Already-fixed runtime buckets must stay fixed
- status: `FIXED`
- severity: `info`
- evidence:
  - attention bucket is 8.990 ms.
  - norm/lm_head bucket is 4.317 ms.
  - best live speed 8.335 tok/s clears floor 8.000
  - attention bucket 8.990 ms is below ceiling 10.000
  - norm/lm_head 4.317 ms is below ceiling 5.000
  - Mamba component evidence points to projection/dispatch before conv rewrite
- next_actions:
  - Do not prioritize attention or lm_head while MoE/Mamba remain above bottleneck threshold.
  - Keep these buckets in every compare report as regression checks.
- acceptance_checks:
  - Attention remains below 10 ms.
  - norm/lm_head remains below 5 ms.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`

## Commands
- refresh_manifest: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_proof_manifest.py --log-dir docs/runtime/logs`
- rerun_ledger: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_issue_ledger.py --log-dir docs/runtime/logs`
````

## Runtime candidate index

````text
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
````

## Runtime speed fix acceptance

````text
# Nemotron Ultra Runtime Speed Fix Acceptance

log_dir: `docs/runtime/logs`
status: `PARTIAL`
target_tps: `10.000`
max_moe_ms: `40.000`
max_mamba_ms: `40.000`
require_speed_gate_fixed: `True`

## Current
- manifest_status: `PARTIAL`
- ledger_status: `OPEN`
- speed_gate_status: `PARTIAL`
- candidate_index_status: `OPEN`
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`

## Fixed
- none

## Partial
- speed gate is PARTIAL, not FIXED
- no speed_candidate lane has ACCEPTED evidence
- best live token/s 8.335 is below target 10.000
- MoE bucket 65.773 ms exceeds acceptance ceiling 40.000
- Mamba bucket 64.157 ms exceeds acceptance ceiling 40.000
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Blockers
- none

## Accepted Speed Lanes
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`
````

## Runtime candidate launch guard

````text
# Nemotron Ultra Runtime Candidate Launch Guard

log_dir: `docs/runtime/logs`
status: `READY`
allow_watch: `False`
runbook_status: `READY`
host_status: `READY`

## Selected Lane
- id: `moe-routed-shared-scheduling`
- kind: `speed_candidate`
- status: `READY`
- candidate_index_status: `MISSING`
- title: MoE routed/shared scheduling

## Warnings
- none

## Failures
- none

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- refresh_after_cleanup: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
````

## Runtime cleanup ready check

````text
# Nemotron Ultra Runtime Cleanup Ready Check

log_dir: `docs/runtime/logs`
status: `READY`

## Fixed
- found host readiness: docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json
- found host cleanup runbook: docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json
- found candidate launch guard: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json
- host readiness is READY
- host cleanup runbook is READY
- candidate launch guard is READY

## Blockers
- none

## Failures
- none

## Manual Actions
- Stop or close the owning app for unrelated high-RSS model servers before loading the 98G Nemotron bundle.
- Confirm the PID still matches the expected process before stopping anything.
- After cleanup, rerun the refresh command and require this check plus the launch guard to be READY.

## Verify Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- strict_guard: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_candidate_launch_guard.py --log-dir docs/runtime/logs --strict`
- strict_lane_matrix: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_lane_readiness_matrix.py --log-dir docs/runtime/logs --strict`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
````

## Runtime MoE candidate contract

````text
# Nemotron Ultra MoE Candidate Contract

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`

## Current Speed
- moe_ms: `65.773`
- mamba_ms: `64.157`
- manual_decode_total_ms: `143.237`
- best_live_tps: `8.335`

## First Target
- target_tps: `10.000`
- required_total_cut_ms: `43.237`
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`

## MoE Invariants
- hidden_shape: `[1, 1, 8192]`
- indices_shape: `[1, 1, 22]`
- scores_shape: `[1, 1, 22]`
- routed_shape: `[1, 1, 22, 2048]`
- latent_shape: `[1, 1, 2048]`
- routed_expert_bits: `{'down_proj': 1, 'up_proj': 1}`
- shared_expert_bits: `8`
- keeps_latent_moe_bf16: `True`
- keeps_router_gates_source_precision: `True`
- drops_mtp: `True`

## Preconditions
- none

## Do
- preserve router top-k and weighted expert semantics
- preserve routed expert 1-bit layout and shared expert 8-bit layout
- reduce per-layer dispatch/synchronization around routed/shared expert execution
- measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate

## Do Not
- do not lower router top-k as the primary speed fix
- do not expand quantized experts to full precision
- do not promote a change unless long-coherence counts do not regress

## Acceptance Checks
- runtime-speed compare status is IMPROVED.
- MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.
- long coherence leak/repeat/EOS counts do not regress.

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Required Outputs
- `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- `2026-06-04-nemotron-ultra-runtime-speed-compare.json`
- `2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `2026-06-04-nemotron-ultra-token-speed-budget.json`
- `2026-06-04-nemotron-ultra-agent-handoff.json`
- `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
````

## Runtime MoE execution ticket

````text
# Nemotron Ultra MoE Execution Ticket

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`
candidate_status: `MISSING`
guard_status: `READY`
cleanup_status: `READY`
contract_status: `READY`
manifest_status: `PARTIAL`

## Failures
- none

## Warnings
- none

## Target
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`
- required_total_cut_ms: `43.237`
- target_tps: `10.000`

## Invariants
- drops_mtp: `True`
- hidden_shape: `[1, 1, 8192]`
- indices_shape: `[1, 1, 22]`
- keeps_latent_moe_bf16: `True`
- keeps_router_gates_source_precision: `True`
- latent_shape: `[1, 1, 2048]`
- routed_expert_bits: `{'down_proj': 1, 'up_proj': 1}`
- routed_shape: `[1, 1, 22, 2048]`
- scores_shape: `[1, 1, 22]`
- shared_expert_bits: `8`

## Execution Order
1. confirm this ticket status is READY
2. run candidate command exactly once for the selected MoE lane
3. run post_check command to write ACCEPTED/REJECTED/BLOCKED verdict
4. rerun candidate index to surface lane status
5. rerun proof refresh to update manifest, ledger, and next runbook

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- refresh_after_cleanup: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- post_candidate_index: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_candidate_index.py --log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md --json-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- post_candidate_refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`

## Acceptance Checks
- runtime-speed compare status is IMPROVED.
- MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.
- long coherence leak/repeat/EOS counts do not regress.

## Required Outputs
- `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- `2026-06-04-nemotron-ultra-runtime-speed-compare.json`
- `2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `2026-06-04-nemotron-ultra-token-speed-budget.json`
- `2026-06-04-nemotron-ultra-agent-handoff.json`
- `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Do Not
- do not run the Mamba lane until this MoE lane has accepted evidence
- do not treat an improved short live row as accepted without long coherence and experiment_result_check
- do not change MTP, VL/audio, parser, or hybrid cache assumptions for this speed lane

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json`
````

## Runtime MoE surface map

````text
# Nemotron Ultra MoE Runtime Surface Map

log_dir: `docs/runtime/logs`
source_root: `jang-tools`
lane_id: `moe-routed-shared-scheduling`
status: `READY`
ticket_status: `READY`
contract_status: `READY`

## Target
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`
- required_total_cut_ms: `43.237`
- target_tps: `10.000`

## Current Speed
- best_live_tps: `8.335`
- mamba_ms: `64.157`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`

## Component Timings
- full_moe: `2.107 ms`
- switch_mlp: `1.130 ms`
- shared_experts: `0.577 ms`
- fc1_latent_proj: `0.231 ms`
- gate: `0.221 ms`
- fc2_latent_proj: `0.212 ms`
- norm: `0.203 ms`
- score_weighted_sum: `0.176 ms`

## Surfaces
| id | status | file | role | anchors |
| --- | --- | --- | --- | --- |
| `loader-hydration` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | hydrates switch_mlp tensors into TurboQuantSwitchLinear modules | def _hydrate_jangtq_model@831, TurboQuantSwitchLinear@48, switch_mlp@499 |
| `nemotron-weighted-moe-patch` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | patches NemotronHMoE decode to call weighted SwitchMLP and shared experts | JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH@1616, Nemotron-H MoE weighted SwitchMLP decode@1656, _switchmlp_weighted_decode@1463 |
| `switchmlp-fastpath-toggle` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | controls the SwitchMLP fast path and its negative-control env toggle | JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH@1583, JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH@1582 |
| `activation-bf16-toggle` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | preserves or disables BF16 activation retention for negative-control proof | JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16@1676 |
| `routed-gather-kernel` | `READY` | `jang-tools/jang_tools/turboquant/gather_tq_kernel.py` | routed down/fc2 gather matmul kernel for selected experts | def gather_tq_matmul@695, make_gather_tq_decode_broadcast@658, make_gather_tq_decode_per_row@593, JANGTQ_GATHER_OPT@39 |
| `fused-gate-up-kernel` | `READY` | `jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py` | fused gate/up/SwiGLU routed expert path used by SwitchMLP | def fused_gate_up_swiglu_matmul@399, make_fused_gate_up_swiglu_decode@271, JANGTQ_MPP_NAX@18 |
| `grouped-nax-proof-surface` | `READY` | `jang-tools/jang_tools/turboquant/mpp_nax_kernel.py` | same-expert grouped tile helpers for possible routed scheduling work | build_sorted_group_tiles@590, gather_tq_matmul_mpp_nax_grouped_from_rot@641, fused_gate_up_swiglu_mpp_nax_grouped_from_rot@835 |
| `moe-component-proof` | `READY` | `jang-tools/examples/nemotron_ultra/moe_component_probe.py` | measures gate, switch_mlp, weighted_decode, shared_experts, weighted sum, and full_moe | moe.switch_mlp@62, weighted_decode@64, shared_experts@69, full_moe@98 |
| `candidate-verdict-proof` | `READY` | `jang-tools/examples/nemotron_ultra/experiment_result_check.py` | accepts/rejects the MoE candidate using compare, speed gate, and handoff invariants | moe-routed-shared-scheduling@144, MoE lane did not improve moe_ms@145, ACCEPTED@171 |

## Runtime Controls
- disable_weighted_moe_fastpath: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- disable_switchmlp_fastpath: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
- legacy_disable_switchmlp_fastpath: `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`
- disable_activation_bf16: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- gather_opt: `JANGTQ_GATHER_OPT`
- mpp_nax: `JANGTQ_MPP_NAX`
- mpp_nax_strict: `JANGTQ_MPP_NAX_STRICT=1`

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Non Goals
- do not edit vMLX or MLX Studio for this JANG handoff
- do not expand routed 1-bit or shared 8-bit tensors to full precision
- do not lower top-k or hide parser/coherence issues to make a speed row look better
- do not run the Mamba lane until the MoE lane has accepted evidence

## Missing
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
````

## Runtime MoE patch plan

````text
# Nemotron Ultra MoE Patch Plan

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`
speed_acceptance_status: `PARTIAL`

## Current Speed
- best_live_tps: `8.335`
- mamba_ms: `64.157`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`

## Target
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`
- required_total_cut_ms: `43.237`
- target_tps: `10.000`

## Ordered Steps
### moe-01-path-scheduling: Reduce full MoE path scheduling overhead first
- component: `full_moe` (inclusive_path)
- projected_total_ms: `101.148`
- 25pct_cut_tps: `8.478`
- 50pct_cut_tps: `10.792`
- goal: Attack the inclusive NemotronHMoE path before isolated micro-optimizations.
- surfaces:
  - `nemotron-weighted-moe-patch`: `jang-tools/jang_tools/load_jangtq.py` (JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH@1616, Nemotron-H MoE weighted SwitchMLP decode@1656, _switchmlp_weighted_decode@1463)
  - `switchmlp-fastpath-toggle`: `jang-tools/jang_tools/load_jangtq.py` (JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH@1583, JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH@1582)
  - `moe-component-proof`: `jang-tools/examples/nemotron_ultra/moe_component_probe.py` (full_moe@98, moe.switch_mlp@62, shared_experts@69, weighted_decode@64)
  - `candidate-verdict-proof`: `jang-tools/examples/nemotron_ultra/experiment_result_check.py` (ACCEPTED@171, MoE lane did not improve moe_ms@145, moe-routed-shared-scheduling@144)
- validation:
  - rerun moe_component_probe.py and require full_moe plus switch_mlp timing to move
  - rerun layer_decode_probe.py and require E bucket improvement
  - rerun long_decode_coherence_probe.py and reject marker/repeat/EOS regression
- non_goals:
  - do not lower top-k
  - do not bypass weighted expert scores
  - do not count a short live row as acceptance without experiment_result_check
### moe-02-switchmlp-routed-kernels: Optimize SwitchMLP routed gate/up/down execution
- component: `switch_mlp` (substep)
- projected_total_ms: `54.264`
- 25pct_cut_tps: `7.712`
- 50pct_cut_tps: `8.613`
- goal: Reduce routed 1-bit expert dispatch around fused gate/up and gather down kernels.
- surfaces:
  - `fused-gate-up-kernel`: `jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py` (JANGTQ_MPP_NAX@18, def fused_gate_up_swiglu_matmul@399, make_fused_gate_up_swiglu_decode@271)
  - `routed-gather-kernel`: `jang-tools/jang_tools/turboquant/gather_tq_kernel.py` (JANGTQ_GATHER_OPT@39, def gather_tq_matmul@695, make_gather_tq_decode_broadcast@658, make_gather_tq_decode_per_row@593)
  - `grouped-nax-proof-surface`: `jang-tools/jang_tools/turboquant/mpp_nax_kernel.py` (build_sorted_group_tiles@590, fused_gate_up_swiglu_mpp_nax_grouped_from_rot@835, gather_tq_matmul_mpp_nax_grouped_from_rot@641)
  - `moe-component-proof`: `jang-tools/examples/nemotron_ultra/moe_component_probe.py` (full_moe@98, moe.switch_mlp@62, shared_experts@69, weighted_decode@64)
  - `candidate-verdict-proof`: `jang-tools/examples/nemotron_ultra/experiment_result_check.py` (ACCEPTED@171, MoE lane did not improve moe_ms@145, moe-routed-shared-scheduling@144)
- validation:
  - compare fused_gate_up and gather path timings through moe_component_probe.py
  - run candidate suite with default env, then compare against weighted-MoE and activation-BF16 negative controls
  - preserve routed_expert_bits up/down=1 and indices shape [1,1,22]
- non_goals:
  - do not expand routed experts to BF16
  - do not make grouped NAX the default without candidate proof
  - do not edit vMLX or MLX Studio for this JANG lane
### moe-03-shared-experts-overlap: Measure shared expert overlap only after routed path moves
- component: `shared_experts` (substep)
- projected_total_ms: `27.720`
- 25pct_cut_tps: `7.336`
- 50pct_cut_tps: `7.729`
- goal: Treat shared experts as secondary unless routed scheduling leaves them dominant.
- surfaces:
  - `nemotron-weighted-moe-patch`: `jang-tools/jang_tools/load_jangtq.py` (JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH@1616, Nemotron-H MoE weighted SwitchMLP decode@1656, _switchmlp_weighted_decode@1463)
  - `moe-component-proof`: `jang-tools/examples/nemotron_ultra/moe_component_probe.py` (full_moe@98, moe.switch_mlp@62, shared_experts@69, weighted_decode@64)
  - `candidate-verdict-proof`: `jang-tools/examples/nemotron_ultra/experiment_result_check.py` (ACCEPTED@171, MoE lane did not improve moe_ms@145, moe-routed-shared-scheduling@144)
- validation:
  - require shared_experts timing to improve without increasing switch_mlp
  - preserve shared_expert_bits=8
  - keep speed acceptance PARTIAL unless token/s target, bucket ceilings, and candidate acceptance all pass
- non_goals:
  - do not dequantize shared experts by default
  - do not optimize shared path before routed path evidence

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- post_candidate_refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- acceptance: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py --log-dir docs/runtime/logs --strict`

## Failures
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`
````

## Runtime MoE delta contract

````text
# Nemotron Ultra MoE Delta Contract

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`

## Baseline
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`

## Target
- target_tps: `10.000`
- target_ms_per_token: `100.000`
- required_total_cut_ms: `43.237`
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- target_moe_ms_for_proportional_10tps: `43.886`
- acceptance_max_moe_ms: `40.000`
- acceptance_max_mamba_ms: `40.000`

## Acceptance Thresholds
- experiment_result_status: `ACCEPTED`
- compare_status: `IMPROVED`
- gate_status: `FIXED for final speed acceptance; PARTIAL only means candidate moved one bucket`
- best_live_tps: `>= 10.000 for final speed acceptance`
- moe_ms: `must improve versus baseline and should fall below 43.886 ms for a 10 tok/s trajectory`
- final_moe_ceiling_ms: `40.000`
- final_mamba_ceiling_ms: `40.000`
- coherence: `no regression in leak, repeat, or EOS counts`
- regression_guards:
  - Mamba ms must not materially regress
  - attention ms must stay under fixed gate ceiling
  - norm/lm_head ms must stay under fixed gate ceiling
  - parser/tool/reasoning behavior must not be hidden by prompt or sampler guards

## Ordered MoE Steps
- `moe-01-path-scheduling` component=`full_moe` projected_total_ms=`101.148` 25pct_tps=`8.478` 50pct_tps=`10.792`
- `moe-02-switchmlp-routed-kernels` component=`switch_mlp` projected_total_ms=`54.264` 25pct_tps=`7.712` 50pct_tps=`8.613`
- `moe-03-shared-experts-overlap` component=`shared_experts` projected_total_ms=`27.720` 25pct_tps=`7.336` 50pct_tps=`7.729`

## Negative Controls
- weighted-moe-ablation is diagnostic only and must not be promoted as a speed fix
- activation-bf16-ablation is diagnostic only and must not be promoted as a speed fix

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- post_candidate_index: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_candidate_index.py --log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md --json-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- post_candidate_refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- acceptance_strict: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py --log-dir docs/runtime/logs --strict`

## Failures
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
````

## Runtime Mamba candidate contract

````text
# Nemotron Ultra Mamba Candidate Contract

log_dir: `docs/runtime/logs`
lane_id: `mamba-projection-dispatch`
status: `BLOCKED`

## Current Speed
- mamba_ms: `64.157`
- moe_ms: `65.773`
- manual_decode_total_ms: `143.237`
- best_live_tps: `8.335`

## Target
- target_tps: `12.000`
- required_total_cut_ms: `59.904`
- mamba_cut_ms_proportional: `29.579`
- mamba_cut_pct_of_current_mamba: `46.105`
- mamba_per_layer_cut_ms: `0.616`

## Mamba Invariants
- hidden_shape: `[1, 1, 8192]`
- normed_shape: `[1, 1, 8192]`
- projected_shape: `[1, 1, 35072]`
- gate_shape: `[1, 1, 16384]`
- conv_dim: `18432`
- conv_input_shape: `[1, 1, 18432]`
- conv_output_shape: `[1, 1, 18432]`
- ssm_out_shape: `[1, 1, 16384]`
- intermediate_size: `16384`
- num_heads: `256`
- n_groups: `8`
- ssm_state_size: `128`
- mamba_projection_bits: `8`
- fp8_projection_affine_bits: `8`
- fp8_projection_group_size: `128`
- drops_mtp: `True`

## Preconditions
- MoE lane evidence is MISSING; run/accept MoE lane before Mamba lane

## Do
- attack projection/dispatch overhead before grouped conv rewrites
- preserve 8-bit affine projection path unless a new projection tradeoff probe reverses the result
- preserve Mamba companion cache/state order for hybrid prefix cache compatibility
- measure M bucket plus `in_proj`, `out_proj`, `conv`, and `ssm_update` after every candidate

## Do Not
- do not dequantize Mamba projections to BF16 as a default speed fix
- do not treat attention KV cache work as a substitute for Mamba state proof
- do not change cache topology without rerunning cache/block handoff checks

## Acceptance Checks
- Mamba bucket drops without changing cache cardinality or Mamba shape contract.
- Attention and norm/lm_head remain below current ceilings.
- long coherence leak/repeat/EOS counts do not regress.

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Required Outputs
- `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- `2026-06-04-nemotron-ultra-runtime-speed-compare.json`
- `2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `2026-06-04-nemotron-ultra-token-speed-budget.json`
- `2026-06-04-nemotron-ultra-agent-handoff.json`
- `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
````

## Runtime proof manifest final

````text
# Nemotron Ultra Runtime Proof Manifest

log_dir: `docs/runtime/logs`
status: `PARTIAL`

## Current Metrics
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_plus_mamba_pct_of_total: `90.710`
- best_live_source: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Lanes
- `moe-routed-shared-scheduling` (speed_candidate): MoE routed/shared scheduling; expected=['IMPROVED']
- `mamba-projection-dispatch` (speed_candidate): Mamba projection/dispatch fusion; expected=['IMPROVED']
- `weighted-moe-ablation` (negative_control): Weighted MoE fast-path A/B; expected=['FAIL', 'UNCHANGED']
- `activation-bf16-ablation` (negative_control): BF16 activation retention guard; expected=['FAIL']

## Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- strict_gate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_gate.py --log-dir docs/runtime/logs --strict`

## Artifacts
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`: present, optional, combined no-load refresh output
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-log-bundle-validation.md`: present, optional, required log presence validation
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-status-report.md`: present, optional, human-readable current status
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`: present, optional, host memory/disk readiness snapshot
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.md`: present, optional, human-readable host readiness snapshot
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`: present, optional, host cleanup runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.md`: present, optional, human-readable host cleanup runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-speed-experiment-plan.md`: present, optional, ranked speed experiment plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`: present, optional, runtime issue ledger
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.md`: present, optional, human-readable runtime issue ledger
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`: present, optional, runtime candidate index
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md`: present, optional, human-readable runtime candidate index
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`: present, optional, runtime candidate launch guard
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md`: present, optional, human-readable runtime candidate launch guard
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json`: present, optional, runtime cleanup ready check
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md`: present, optional, human-readable runtime cleanup ready check
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`: present, optional, MoE candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.md`: present, optional, human-readable MoE candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`: present, optional, MoE execution ticket
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md`: present, optional, human-readable MoE execution ticket
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.json`: present, optional, MoE runtime source surface map
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.md`: present, optional, human-readable MoE runtime source surface map
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json`: present, optional, MoE runtime patch plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md`: present, optional, human-readable MoE runtime patch plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json`: present, optional, MoE runtime delta acceptance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md`: present, optional, human-readable MoE runtime delta acceptance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json`: present, optional, Mamba candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md`: present, optional, human-readable Mamba candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`: present, required, target token/s millisecond budgets
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.md`: present, optional, human-readable target budgets
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.json`: present, optional, component-level token/s sensitivity matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.md`: present, optional, human-readable component budget matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`: present, optional, implementation-facing runtime patch spec
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.md`: present, optional, human-readable runtime patch spec
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`: present, optional, runtime shape and bit contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.md`: present, optional, human-readable runtime shape contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.json`: present, optional, runtime candidate preflight
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.md`: present, optional, human-readable candidate preflight
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json`: present, optional, all-lane runtime readiness matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md`: present, optional, human-readable lane readiness matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`: present, optional, next runtime runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.md`: present, optional, human-readable next runtime runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json`: present, required, machine-readable experiment queue
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.md`: present, optional, human-readable experiment queue
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`: present, required, machine-readable speed gate
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.md`: present, optional, human-readable speed gate
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`: present, optional, runtime speed fix acceptance audit
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md`: present, optional, human-readable speed fix acceptance audit
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.json`: present, optional, baseline-vs-baseline compare for current logs
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.md`: present, optional, human-readable compare report
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`: present, optional, cache/parser/runtime nuance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md`: present, optional, human-readable cache/parser/runtime nuance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.json`: present, required, machine-readable downstream agent handoff
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.md`: present, optional, human-readable downstream agent handoff
- `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`: present, required, best live decode source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`: present, required, layer decode bucket source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`: present, required, long coherence source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-mamba-component-probe.json`: present, required, Mamba component source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`: present, required, MoE component source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`: present, required, projection tradeoff source
````
