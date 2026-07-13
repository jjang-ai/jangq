# JANG Studio — Design Spec

**Date:** 2026-04-18
**Author:** Jinho Jang
**Status:** Draft — awaiting user review

A native macOS SwiftUI wizard that converts HuggingFace models (BF16 / FP16 / FP8) into JANG and JANGTQ formats by driving the existing `jang-tools` Python pipeline as a subprocess. Public-distribution `.app`, code-signed and notarized, with embedded Python runtime so users get a turnkey experience.

---

## Brainstorm Decisions (locked)

| # | Question | Choice |
|---|---|---|
| Q1 | How does Swift run conversion? | **A — Shell out to `jang` CLI subprocess.** Reuses every existing flag/profile, zero quantization-logic duplication. |
| Q2 | Audience & distribution? | **C — Public distribution.** Xcode project, code-signed + notarized DMG. |
| Q3 | Step 2 (architecture) UX? | **A — Auto-detect + show summary + override.** Reuses `architectures.detect_architecture()`. |
| Q4 | Wizard layout? | **B — Sidebar Nav (`NavigationSplitView`).** Steps in left rail; power users can jump back. |
| Q5 | Progress reporting? | **C — Both text and `--progress=json`.** JSON mode is opt-in via flag; text output unchanged for CLI users. |
| Q6 | How does shipped `.app` find `jang`? | **A — Bundle Python + `jang_tools` inside the `.app`.** Turnkey, ~150–200 MB, zero user setup. |
| Q7 | Repo location & name? | **A — Top-level `JANGStudio/`. Name: "JANG Studio".** Sibling to `jang-runtime`, `jang-server`, `jang-tools`. |

Cross-cutting requirement (added mid-brainstorm): The wizard must **prove** the converted output is complete (`jang_config.json`, tokenizer, chat template, capabilities, sharded safetensors index, custom modeling `.py` for MiniMax-class, VL preprocessors when applicable). This is an explicit verifier panel that gates the Done screen.

---

## Section 1 — Architecture & Repo Layout

### 1.1 Repo layout

```
jang/
├── JANGStudio/                          ← NEW top-level
│   ├── JANGStudio.xcodeproj
│   ├── JANGStudio/                      Swift sources (SwiftUI app target)
│   │   ├── App/                         JANGStudioApp, AppDelegate (if needed)
│   │   ├── Wizard/                      WizardCoordinator + 5 step views
│   │   ├── Runner/                      PythonRunner, JSONLProgress, BundleResolver
│   │   ├── Models/                      ConversionPlan, ArchitectureSummary, …
│   │   ├── Verify/                      PreflightRunner, PostConvertVerifier
│   │   └── Resources/                   Assets.xcassets, Info.plist, Entitlements
│   ├── Scripts/
│   │   ├── build-python-bundle.sh       Assembles Contents/Resources/python/
│   │   ├── codesign-runtime.sh          Deep-signs every bundled .dylib
│   │   └── notarize.sh                  xcrun notarytool wrapper
│   ├── Tests/
│   │   ├── JANGStudioTests/             XCTest — JSONL parser, plan validation
│   │   └── JANGStudioUITests/           XCUITest — golden wizard flow
│   └── README.md
└── jang-tools/jang_tools/__main__.py    ← MODIFIED: add --progress=json + --quiet-text
```

### 1.2 Runtime architecture

1. **SwiftUI app** uses `NavigationSplitView`. Sidebar lists the 5 steps with state icons (idle / active / done / blocked). Detail view renders the active step.
2. **WizardCoordinator** holds a single `ConversionPlan` `@Observable` model. Steps mutate it; the sidebar derives enable/disable from completeness.
3. **PythonRunner** (`Process` + `Pipe`) launches `Contents/Resources/python/bin/python3 -m jang_tools convert <src> -o <out> -p <profile> -m <method> [--hadamard] --progress=json --quiet-text`. Stderr is JSONL, stdout is plain text logs. Both stream into the live log view.
4. **Bundled Python** lives in `JANGStudio.app/Contents/Resources/python/` (built via `python-build-standalone` 3.11+ + `pip install jang` into a private site-packages). All `.dylib`s deep-signed. App size target ≤ 200 MB.
5. **PostConvertVerifier** runs after success — calls `python -m jang_tools validate <out>` plus the 12-row checklist from §4.2. Blocks the Done screen on red rows.

### 1.3 Single-source-of-truth principle

The Swift app holds **no** quantization logic. Every conversion decision (architecture detection, profile semantics, bit allocation, calibration, capabilities stamping) stays in `jang_tools`. The wizard cannot drift from the CLI. This is what the "no bandaid fixes" / "verify research" principles demand.

---

## Section 2 — Wizard State Machine & 5 Step Screens

### 2.1 ConversionPlan (shared model, persisted to UserDefaults under `lastPlan`)

```swift
@Observable final class ConversionPlan {
    var sourceURL: URL?                    // step 1
    var sourceDtype: SourceDtype?          // .bf16, .fp16, .fp8, .jangV2 (upgrade), .unknown
    var detected: ArchitectureSummary?     // step 2 — auto-filled
    var overrides: ArchitectureOverrides   // step 2 — user edits
    var family: Family                     // .jang or .jangtq
    var profile: String                    // step 3 — e.g. "JANG_4K", "JANGTQ3"
    var method: QuantMethod = .mse         // step 3 advanced — .mse | .rtn | .mseAll
    var hadamard: Bool = false             // step 3 advanced
    var outputURL: URL?                    // step 4
    var diskFreeBytes: Int64?              // preflight
    var ramBytes: Int64?                   // preflight
    var run: RunState = .idle              // step 4 live state
    var verifyResults: [VerifyCheck] = []  // step 5
}
```

### 2.2 Sidebar (always visible)

| # | Step | Active when | Done when |
|---|---|---|---|
| 1 | Source Model | always | `sourceURL != nil && detected != nil` |
| 2 | Architecture | step 1 done | user clicks Confirm |
| 3 | Profile | step 2 done | profile + output dir set + preflight green |
| 4 | Run | step 3 done | `run == .succeeded` |
| 5 | Verify & Finish | step 4 done | all required checks green |

Items below the active step are dimmed and unclickable until prereqs land — locks prevent skipping preflight.

### 2.3 Step 1 — Source Model

- Big drop-zone + "Choose Folder…" button (NSOpenPanel, directories only). Recents row lists last 5 source dirs.
- On selection, runs **inline detection** in a background `Task`:
  1. Confirm `config.json` exists, else show "Not a HuggingFace model" error.
  2. Sniff first shard header → infer `sourceDtype` (`bfloat16` / `float16` / `float8_e4m3fn` / JANG v2 marker for upgrade flow).
  3. Call `python -m jang_tools inspect-source --json <src>` (new tiny subcommand) returning architecture + param count + expert count + dtype as JSON. ~1s, no model load.
- Renders a summary card: model name, dtype badge (BF16/FP16/FP8), param count, file count, total GB, MoE/Dense badge.
- "Continue" enables only after detection completes successfully.

### 2.4 Step 2 — Architecture (auto-detect + override)

- Big read-only summary: detected `model_type`, attention type (full / MLA / linear / hybrid SSM), expert count, MTP layers, recommended dtype (e.g., "bfloat16 forced — 256 experts > 128 threshold").
- "Looks right →" CTA continues. Below, a collapsible **Advanced Overrides** section:
  - Force dtype: auto / float16 / bfloat16
  - Force block size: auto / 32 / 64 / 128
  - Add tensors to skip-pattern (free text, comma-separated)
  - Custom calibration set: default / browse for `.jsonl`
- Overrides serialize into `ConversionPlan.overrides` and pass through as extra CLI flags (or a generated `--overrides-json <path>` file for things the CLI doesn't already accept).

### 2.5 Step 3 — Profile

Two-tab segmented control: **JANG** (always enabled) / **JANGTQ** (enabled only when `detected.model_type ∈ {qwen3_5_moe, minimax_m2}`; otherwise tab is greyed with tooltip *"JANGTQ supports Qwen 3.6 and MiniMax only (v1) — your model is X"*). GLM JANGTQ is deferred to v1.1 — see Open Item #4.

**JANG tab** — grid of profile cards grouped by bit tier:
- 2-bit: 2S / 2M / 2L / 1L
- 3-bit: 3S / 3M / 3L / 3K
- 4-bit: 4S / 4M / 4L / **4K (default badge)**
- 5/6-bit: 5K / 6K / 6M

Each card shows: name, estimated output size (computed from param count × bits + overhead), critical/important/compress bit triple, recommended use case (1-line). Selected card glows blue.

**JANGTQ tab** — three cards: JANGTQ2 / JANGTQ3 / JANGTQ4 with same size estimates, plus a footer: *"Routes to `convert_qwen35_jangtq.py` or `convert_minimax_jangtq.py` based on detected arch."*

Below the cards:
- **Output folder** picker (default = sibling dir `<source>-<profile>`)
- **Method** segmented (mse / rtn / mse-all, mse default)
- **Hadamard rotation** toggle (auto-enabled for ≥3-bit profiles per the `feedback_no_bandaid_fixes` finding; warns red if user turns it on at 2-bit per `project_hadamard_rotation`)

**Pre-flight panel** at the bottom (live; full check list in §4.1). "Start Conversion" stays disabled until preflight is fully green.

### 2.6 Step 4 — Run

- Top: macro progress bar (`[N/5] phase` from JSONL `phase` events).
- Middle: fine progress bar + tensor name (from JSONL `tensor` + `done/total`).
- Below: live log scroll view (monospace, auto-scroll with pinning, color-coded WARN/ERROR lines, copyable). Filter chips: All / Stdout / Stderr / Errors only.
- Right rail: ETA (computed from tensor throughput), elapsed time, peak RAM (sampled via `host_statistics64`).
- **Cancel** button → SIGTERM the subprocess, give it 3s for clean exit, else SIGKILL. Marks run `.cancelled`. Output dir is left as-is (user might want to inspect partial state); a banner offers "Delete partial output".
- On non-zero exit: log view freezes, "Retry" / "Open Output" / "Copy Diagnostics" buttons appear.

### 2.7 Step 5 — Verify & Finish

- Auto-runs the 12-row checklist (§4.2).
- Each row: green check / red X / grey "n/a" + a "Why?" disclosure.
- Any required-row red blocks the "Finish" button; offers "Re-run conversion" + "Open Output in Finder".
- All green → success card: "Model ready: `<output path>`", buttons: **Reveal in Finder**, **Copy Path**, **Open in MLX Studio** (deep link if installed), **Test load** (runs `python -m jang_tools inspect <out>` and shows the bit histogram), **Convert another model** (resets to step 1, keeps last paths in Recents).

---

## Section 3 — PythonRunner, JSONL Progress Protocol, Cancellation

### 3.1 jang-tools changes (Python side, minimal)

Two new global flags in `jang_tools/__main__.py`, plumbed into every long-running subcommand (`convert`, `convert_qwen35_jangtq`, `convert_minimax_jangtq`, the GLM converter, `validate`, `inspect`, the new `inspect-source`):

| Flag | Effect |
|---|---|
| `--progress=json` | Emit one JSONL event per progress tick to **stderr**. Default off. |
| `--quiet-text` | Suppress the existing `[N/5] …` and tqdm prints on stdout (keeps real warnings/errors). Default off. Orthogonal to JSON mode. |

A small `jang_tools/progress.py` module:

```python
class ProgressEmitter:
    def __init__(self, json_to_stderr: bool, quiet_text: bool): ...
    def phase(self, n: int, total: int, name: str) -> None
    def tick(self, done: int, total: int, label: str = "") -> None
    def event(self, level: str, message: str, **fields) -> None  # warn/info/error
```

Each existing `print("  [N/5] …")` becomes `progress.phase(...)` (still prints human line unless `--quiet-text`). The `tqdm(...)` wrapper at `convert.py:439` becomes a thin shim that also calls `progress.tick(...)`.

### 3.2 JSONL schema (one object per line, stderr)

```jsonl
{"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1776538000.123}
{"v":1,"type":"info","msg":"Detected qwen3_5_moe — 256 experts, MLA","ts":...}
{"v":1,"type":"phase","n":4,"total":5,"name":"quantize","ts":...}
{"v":1,"type":"tick","done":1234,"total":2630,"label":"layer.5.gate_proj","ts":...}
{"v":1,"type":"warn","msg":"No chat template found; model may loop","ts":...}
{"v":1,"type":"phase","n":5,"total":5,"name":"write","ts":...}
{"v":1,"type":"done","ok":true,"output":"/path/to/out","elapsed_s":712.4,"ts":...}
```

- `v:1` is the protocol version. Swift refuses to parse and shows "Update JANG Studio" if a future Python emits `v:2` with breaking changes.
- All numeric fields are JSON numbers; `ts` is unix seconds with subsecond precision for ETA math.
- `done` event always written exactly once at end (success or failure with `ok:false, error:"..."`).

### 3.3 Swift PythonRunner

```swift
actor PythonRunner {
    func run(plan: ConversionPlan) -> AsyncThrowingStream<ProgressEvent, Error>
    func cancel()                  // SIGTERM → wait 3s → SIGKILL
}
```

- **Bundle resolution** — `BundleResolver.pythonExecutable` returns `Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/python/bin/python3")`. Asserts existence at app startup; fatal "Bundle is corrupt — please reinstall" alert if missing.
- **Process setup** — `Process()` with:
  - `executableURL` = bundled `python3`
  - `arguments` = `["-m", "jang_tools", "convert", src, "-o", out, "-p", profile, "-m", method, "--progress=json", "--quiet-text"] + (hadamard ? ["--hadamard"] : [])`. JANGTQ branch routes by detected `model_type`:
    - `qwen3_5_moe` → `-m jang_tools.convert_qwen35_jangtq <src> <out> <profile>`
    - `minimax_m2`  → `-m jang_tools.convert_minimax_jangtq <src> <out> <profile>`
    - (GLM JANGTQ deferred to v1.1 — JANGTQ tab disabled for `glm_*`.)
  - `environment` = inherits + `PYTHONPATH` set to bundled site-packages, `PYTHONUNBUFFERED=1`, `PYTHONNOUSERSITE=1`, `JANG_PROGRESS=1`
  - `currentDirectoryURL` = chosen output dir's parent (clean relative paths in logs)
- **Pipes** — separate `Pipe()` for stdout (text logs) and stderr (JSONL). Two `Task`s drain each pipe via `FileHandle.bytes.lines` async sequence — never block the main actor.
- **Parser** — stderr lines feed `JSONDecoder().decode(ProgressEvent.self, from: line.data(using: .utf8)!)`. Bad JSON lines surface as a `.parseError` event but don't kill the run.
- **Stream contract** — yields events as they arrive; throws on non-zero exit (`error: ProcessError(code, lastError)`); finishes cleanly on `done.ok == true`.

### 3.4 Cancellation lifecycle

```
User taps Cancel
  → runner.cancel()
  → process.terminate()              (SIGTERM)
  → wait up to 3.0s for waitUntilExit
  → if still running: kill(pid, SIGKILL)
  → drain pipes, flush remaining lines to log
  → emit synthetic {"type":"done","ok":false,"error":"cancelled"}
  → run state = .cancelled
  → banner: "Cancelled. Partial output left at <path>. [Delete] [Keep]"
```

`convert.py` already writes `<out>/jang_config.json` only at the end; per-shard safetensors are written as they finish. SIGTERM is safe (no transactional rollback) — the verifier in Step 5 marks the output invalid (missing `jang_config.json` + shard count mismatch).

### 3.5 Backpressure & log volume

A 397B conversion produces ~50k tqdm ticks. Three guards:

1. Python coalesces ticks: `progress.tick()` skips emission unless ≥0.1s since last tick **or** a tensor finished. Caps at ~10 events/sec.
2. Swift log view is a `TextEditor`-style buffer with a 10k-line ring — older lines dropped from the UI but kept on disk at `~/Library/Logs/JANGStudio/<run-id>.log` (revealed by "Copy Diagnostics").
3. Progress bar updates throttle to 30 Hz on the main actor regardless of event rate.

### 3.6 Testing strategy

- **Unit tests** (`JSONLParserTests`) — feed a captured fixture of JSONL events, assert correct phase/tick/done parsing, version-mismatch handling, malformed-line tolerance.
- **Integration smoke test** — Swift test invokes the bundled `python3 -m jang_tools --version` to prove the bundle works end-to-end. Runs in CI on macos-15 runner.
- **Cancellation test** — start a fake long-running Python child (sleep loop), cancel after 0.5s, assert SIGTERM lands within 3.0s + final `.cancelled` event emitted.
- **No real conversion in CI** — too slow / model-dependent. Manual nightly job on a Mac runner with a 1B test model.

---

## Section 4 — Pre-flight, Verification, Error Handling

### 4.1 Pre-flight checks (Step 3 → Start Conversion gate)

`PreflightRunner` produces `[PreflightCheck]` (`{name, status, hint}`). Statuses: **pass** / **warn** (doesn't block) / **fail** (blocks).

| # | Check | Source | Status if false |
|---|---|---|---|
| 1 | Source dir exists + readable | filesystem | fail |
| 2 | `config.json` parses + has `model_type` | re-run on confirm | fail |
| 3 | Output dir resolvable + writable + not inside `.app` + not equal to source | filesystem | fail |
| 4 | Disk free at output ≥ size estimate × 1.10 | `URLResourceValues.volumeAvailableCapacityForImportantUsage` | fail (with "free X GB needed") |
| 5 | RAM ≥ source on-disk size × 1.5 | `ProcessInfo.physicalMemory` vs source GB | warn (`project_mxtq_load_expansion`: GLM-5 needed 384 GB+; warn but don't block) |
| 6 | Family/architecture compatible: JANGTQ tab requires `model_type ∈ {qwen3_5_moe, minimax_m2}` (v1) | detected summary | fail |
| 7 | JANGTQ source dtype is FP8 or BF16 | sniffed from header | fail |
| 8 | At 2-bit profiles + `model_type` in known-512+-expert list, force bfloat16 (warn if user overrode to float16) | per `project_bfloat16_fix` | warn |
| 9 | Hadamard rotation + 2-bit profile combination | per `project_hadamard_rotation` (Hadamard hurts 2-bit) | warn |
| 10 | Bundled python runtime healthy (`python3 -m jang_tools --version` returns) | one-time at app launch, cached | fail |

The Step 3 panel renders these live as the user changes profile/output/method. The "Start Conversion" button binds to `preflight.allMandatoryPassing`.

### 4.2 Post-convert verifier (Step 5)

`PostConvertVerifier` runs **after** `done.ok == true`. Returns `[VerifyCheck]`:

| # | Check | How | Required? |
|---|---|---|---|
| 1 | `jang_config.json` exists + JSON-valid | filesystem + JSONDecoder | required |
| 2 | `jang_config.format == "jang"` and `format_version >= "2.0"` | parsed config | required |
| 3 | Schema valid via `python -m jang_tools validate <out>` | subprocess (existing validator) | required |
| 4 | `jang_config.capabilities` present and non-empty | parsed config | required |
| 5 | Chat template present (inline in `tokenizer_config.json` OR `chat_template.jinja` exists) | mirrors `convert.py:1031-1038` | required |
| 6 | Tokenizer files complete — at least one of `tokenizer.json` / `tokenizer.model` + `tokenizer_config.json` + `special_tokens_map.json` | filesystem | required |
| 7 | Shard count in `model.safetensors.index.json` matches actual `.safetensors` files on disk | filesystem + JSON parse | required |
| 8 | If detected was VL: `preprocessor_config.json` and `video_preprocessor_config.json` present | filesystem + detected.isVL | required when applicable (`feedback_always_vl`) |
| 9 | If detected was MiniMax-class: `modeling_*.py` + `configuration_*.py` copied | filesystem | required when applicable |
| 10 | `tokenizer_config.json` → `tokenizer_class` is concrete, not `TokenizersBackend` | parsed config | warn (Osaurus serving compat) |
| 11 | Total bytes on disk within ±5% of allocator's pre-quantize estimate | parsed config + filesystem | warn (large drift = silent bug) |
| 12 | Smoke test: `python -m jang_tools inspect <out>` succeeds | subprocess | required (last line of defense) |

UI:
- Required checks fail → **Finish blocked**, banner: *"Output is incomplete — N issues block use."* + Re-run / Open Output / Copy Diagnostics.
- Warn-only checks fail → Finish allowed, but yellow banner with "Review warnings" disclosure.

### 4.3 Error categories & handling

| Category | Trigger | UX |
|---|---|---|
| **PreflightFailure** | Step 3 gate | Inline red row, no run started |
| **BundleCorrupt** | Bundled python missing or `--version` non-zero | Modal alert at app launch: "Reinstall JANG Studio" |
| **SourceUnreadable** | Step 1 detect or step 4 mid-run filesystem error | Step-level error card, "Choose different folder" |
| **PythonException** | non-zero exit, last log line matches `Traceback` | Log view freezes, top banner shows `<ExceptionClass>: <message>`, "Copy Traceback" + "Open Logs Folder" |
| **OOM** | non-zero exit + `MemoryError` or `Killed: 9` in last 50 lines | Special banner: *"Process killed by OS — likely out of RAM. Try a higher-bit profile or close other apps."* |
| **DiskFull** | non-zero exit + `OSError: No space left` | Banner: *"Disk filled mid-conversion — N GB needed."* + "Choose different output dir" |
| **CancelledByUser** | SIGTERM path | Neutral banner, "Delete partial / Keep partial" |
| **JsonParseError** (Swift-side) | Stderr line failed to decode | Logged to disk, surface as a single yellow chip after run ends — never blocks |
| **VersionMismatch** | JSONL `v` field unknown | Modal: *"Update JANG Studio — bundled tools emitted protocol v2."* |

### 4.4 Diagnostics bundle ("Copy Diagnostics" button)

Produces a zip at `~/Desktop/JANGStudio-diagnostics-<timestamp>.zip` containing:
- `plan.json` — the full `ConversionPlan` (sourceURL/outputURL anonymized to last path component if user toggles "anonymize")
- `run.log` — full stdout+stderr from the Python subprocess
- `events.jsonl` — captured JSONL stream
- `system.json` — macOS version, RAM, free disk, CPU, JANG Studio version, bundled jang-tools version, bundled python version
- `verify.json` — the post-convert checklist results (if reached)

### 4.5 Idempotency & re-run

- "Retry" reuses the same `ConversionPlan` and same output dir. The Python converter overwrites existing safetensors per-shard, so a clean re-run from scratch works.
- Optional toggle in Step 4 right rail: **Resume from last completed phase** — only valid for the post-quantize "write" phase (cheapest re-do); deferred to v1.1.

---

## Section 5 — Build, Test, Docs, Distribution

### 5.1 Embedded Python bundle

`Scripts/build-python-bundle.sh` (run during Xcode "Run Script" phase + by CI):

```
1. Download python-build-standalone (3.11.x, aarch64-apple-darwin, install_only)
2. Extract to build/python/
3. Strip site-packages of: pip cache, *.pyc, tests/, __pycache__/
4. pip install --target build/python/lib/python3.11/site-packages \
     "../jang-tools/dist/jang-<ver>-py3-none-any.whl[mlx,vlm]"
   # The [mlx,vlm] extras pull mlx>=0.31.2, mlx-lm>=0.31.3, mlx-vlm>=0.6.3
   # plus numpy, safetensors, tqdm, transformers, tokenizers, sentencepiece
   # (transitive deps, version-pinned in jang-tools/pyproject.toml).
5. Strip build/python/ further — drop unused stdlib (tkinter, idle, test/, ensurepip)
6. Rewrite shebangs in build/python/bin/* to use relative #!/usr/bin/env python3
7. ad-hoc codesign every .dylib + .so under build/python/ (real signing in CI)
8. Copy build/python/ → JANGStudio.app/Contents/Resources/python/
9. Smoke-test: ./JANGStudio.app/.../python/bin/python3 -m jang_tools --version
```

Target bundle size ≤ 200 MB. If >250 MB, fail the build.

`jang-tools` ships a wheel via `pyproject.toml` (already exists). The bundle script depends on `python -m build` having run — `make wheel` target in the repo.

### 5.2 Xcode project setup

- macOS 15+ deployment target (matches `jang-runtime/Package.swift`).
- SwiftUI lifecycle (no AppDelegate unless we need NSOpenPanel customization — can be added later).
- Hardened Runtime entitlements: `com.apple.security.cs.allow-jit` (false), `com.apple.security.cs.disable-library-validation` (TRUE — required for embedded Python loading non-Apple-signed `.dylib`s), `com.apple.security.files.user-selected.read-write` (read source + write output).
- App sandbox **off** for v1 (Python subprocess + arbitrary file paths). Re-evaluate for App Store later.
- Single app target + two test targets (XCTest unit + XCUITest UI).

### 5.3 CI (GitHub Actions, `.github/workflows/jang-studio.yml`)

```yaml
on:
  push:
    paths: [JANGStudio/**, jang-tools/**]
  pull_request: {paths: [JANGStudio/**, jang-tools/**]}

jobs:
  build:
    runs-on: macos-15
    steps:
      - checkout
      - setup-python (3.11)
      - make -C jang-tools wheel
      - cd JANGStudio && Scripts/build-python-bundle.sh
      - xcodebuild -scheme JANGStudio -configuration Debug build
      - xcodebuild test -scheme JANGStudio
      - upload build/JANGStudio.app as artifact

  release:
    needs: build
    runs-on: macos-15
    if: startsWith(github.ref, 'refs/tags/jang-studio-v')
    steps:
      - download .app artifact
      - Scripts/codesign-runtime.sh                # APPLE_DEV_ID secret
      - Scripts/notarize.sh                        # APPLE_NOTARIZE_TOKEN secret
      - create-dmg JANGStudio.app
      - notarize + staple the DMG
      - upload to GitHub Release
```

Secrets in GitHub repo secrets, not in repo.

### 5.4 Test plan

| Layer | Tests | Where |
|---|---|---|
| Unit (Swift) | JSONL parser, ConversionPlan validation, PreflightRunner with mock filesystem, PostConvertVerifier with fixture output dirs, BundleResolver path resolution | `JANGStudioTests/` — every PR |
| Unit (Python) | New `progress.py` emitter — emits well-formed JSON, throttling caps at 10/sec | `jang-tools/tests/test_progress.py` |
| Integration | Bundled `python3 -m jang_tools --version` succeeds; `inspect-source` returns valid JSON for a tiny fixture model | macos-15 CI |
| UI snapshot | XCUITest walks all 5 steps with a stub `PythonRunner` (returns canned events). Asserts each step renders, gates fire, error states display | macos-15 CI |
| Cancellation | Spawn fake long-running subprocess, cancel, assert SIGTERM lands in <3s | unit test with sleep-based fake |
| Manual nightly | Real conversion of a 1B test model (e.g., Llama-3.2-1B) → JANG_4K. Sanity check, not blocking | nightly Mac runner job |
| **Out of scope for CI** | Real 200B+ conversions (too slow, model storage too big) | manual smoke before each release |

### 5.5 Documentation

| File | Purpose |
|---|---|
| `JANGStudio/README.md` | Install, screenshots, "what it does in one paragraph", system requirements |
| `JANGStudio/docs/USER_GUIDE.md` | Step-by-step walkthrough with screenshots of each of the 5 wizard steps + verifier panel |
| `JANGStudio/docs/TROUBLESHOOTING.md` | Common errors → fixes (OOM, disk full, JANGTQ on incompatible arch, missing chat template) |
| `JANGStudio/docs/CONTRIBUTING.md` | Dev setup, how to point at your local `jang-tools` instead of the bundled one (requires unsigned debug builds) |
| `JANGStudio/docs/PROGRESS_PROTOCOL.md` | The JSONL schema (v1) — for anyone replacing the GUI with a different frontend |
| Top-level `README.md` (modify) | Add a "Get JANG Studio" section with DMG link + screenshot under the existing MLX Studio banner |

### 5.6 Out of scope for v1 (deferred)

- HuggingFace Hub download from inside the wizard (Step 1 is local folder only)
- MXTQ → JANG upgrade flow (already exists as `jang upgrade`; will fold in v1.1)
- Profile recommender (auto-pick profile from RAM + arch)
- Resume from last completed phase
- App Store distribution (sandbox required, deferred indefinitely)
- Auto-update inside the app (use Sparkle in v1.1; v1 ships static DMG)
- Multi-conversion queue
- Server mode (talk to remote build host)

### 5.7 Risks & mitigations

| Risk | Mitigation |
|---|---|
| Bundle size > 250 MB after dependencies | Aggressive stdlib stripping; if still over, switch to dynamic download of `jang-tools` wheel on first launch |
| Notarization rejection due to embedded Python | Pre-validated approach — `python-build-standalone` is widely used; deep-sign every `.dylib` |
| `mlx` version drift between bundled python and `jang_tools` requirements | Pin exact versions in `jang-tools/pyproject.toml`; rebuild bundle on every `jang-tools` release |
| User has Rosetta-only Mac (Intel) | macOS 15 is Apple Silicon only — explicit system requirement in DMG |
| New Python `print(...)` line in `convert.py` breaks JSON parser | JSON mode and text mode are independent code paths; CI golden-fixture test catches drift |

---

## Open Items (for the implementation plan)

These were left as "sensible defaults" during brainstorming and should be revisited during implementation:

1. **Branding/theme** — default to system Apple HIG (light/dark mode follows OS); JANG accent color = the blue used in the JANG logo. Confirm with Eric before final asset pass.
2. **Settings persistence** — `UserDefaults` for last source dir, last output dir, last profile, last advanced overrides. Reset via a Settings → Advanced → Reset button.
3. **First-launch experience** — single welcome sheet on first run only: "JANG Studio converts HuggingFace models to JANG/JANGTQ for Apple Silicon. Bundled with python-build-standalone 3.11 + jang-tools v<version>." [Continue].
4. **GLM JANGTQ — deferred to v1.1.** `jang_tools/` ships `convert_minimax_jangtq.py` and `convert_qwen35_jangtq.py` but not `convert_glm_jangtq.py`. Decision (2026-04-18): v1 ships Qwen + MiniMax JANGTQ only. GLM flows through regular `jang convert` (the standard JANG path handles `glm_moe_dsa` via MLA + FP8 dequant — see `scripts/convert-glm51.sh`). JANGTQ tab is disabled for `glm_*`; tooltip says "JANGTQ for GLM coming in v1.1". When v1.1 adds it, update the preflight check in §4.1 row 6 and the JANGTQ tab gate in §2.5.

---

## Acceptance Criteria

A v1 release is considered shippable when:

- ✅ Notarized DMG builds in CI on a tag push
- ✅ A user with a fresh Mac (no Python, no `jang-tools`) can install the DMG and successfully convert a 1B HuggingFace model to JANG_4K end-to-end (5 wizard steps + verifier all green)
- ✅ A user with a Qwen 3.6 model can convert to JANGTQ3; a user with a MiniMax model can convert to JANGTQ2/3/4; JANGTQ tab is correctly disabled for all other architectures (including GLM in v1)
- ✅ Cancellation mid-quantize cleanly terminates the subprocess and offers Delete/Keep
- ✅ All 12 verifier checks fire correctly on a known-good output and a deliberately-broken output (missing chat template, missing tokenizer)
- ✅ Diagnostics bundle reproduces a captured failure end-to-end
- ✅ Bundle size ≤ 200 MB
- ✅ XCTest + XCUITest pass on macos-15 CI
