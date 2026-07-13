# Contributing

## Dev mode (use your own `jang-tools`)

1. `pip install -e jang-tools` in your local venv.
2. Build a debug app:
   ```
   cd JANGStudio && xcodegen generate
   xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build
   ```
3. Launch with override:
   ```
   JANGSTUDIO_PYTHON_OVERRIDE=$(which python3) open build/Debug/JANGStudio.app
   ```
4. JANG Studio will call your local `python3 -m jang_tools ...` instead of the bundled runtime. Edit `jang-tools/` freely.

## Regenerating the Xcode project

After editing `project.yml`:
```
cd JANGStudio && xcodegen generate
```

## Running tests

Swift (40 unit + 1 UI test):
```
cd JANGStudio
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS'
```

Python (151 tests):
```
pytest jang-tools/tests -v
```

Run only one target:
```
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -only-testing:JANGStudioTests
```

## Building the full signed .app locally (optional, ~5 minutes)

```
cd JANGStudio
Scripts/build-python-bundle.sh          # ~3 min, 305 MB bundle
xcodegen generate
xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Release build
# Unsigned .app will be under DerivedData/... — drag into /Applications to run.
```

For real signing + notarization, see `.github/workflows/jang-studio.yml` and the `Scripts/codesign-runtime.sh` + `Scripts/notarize.sh` scripts. Requires an Apple Developer account.

## Releases

Tag `jang-studio-vX.Y.Z`. CI builds the signed+notarized DMG and attaches it to a GitHub release.
Required repo secrets: `APPLE_DEV_ID_CERT_P12`, `APPLE_DEV_ID_CERT_PW`, `APPLE_DEV_ID_APP`, `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD`.

## File layout

```
JANGStudio/
  JANGStudio/
    App/             - @main
    Models/          - ConversionPlan (+ WizardMode), ArchitectureSummary, ProgressEvent
    Runner/          - PythonRunner, JSONLProgressParser, BundleResolver, CLIArgsBuilder, DiagnosticsBundle
    Verify/          - PreflightRunner + PostConvertVerifier
    Wizard/          - WizardCoordinator (Convert vs Expert Lab modes, visibleSteps)
      Steps/         - Source, Profile, Run, Verify (+ Expert/Prune review embedded)
    Resources/       - Info.plist, entitlements, Assets.xcassets
  Tests/
    JANGStudioTests/         - unit tests + fixtures
    JANGStudioUITests/       - XCUITest (Convert sidebar at launch)
  Scripts/
    build-python-bundle.sh   - hermetic python3 + jang[mlx] + mlx-vlm --no-deps
    codesign-runtime.sh      - deep-sign bundled python + app (CI only)
    notarize.sh              - xcrun notarytool wrapper (CI only)
  docs/                      - USER_GUIDE, TROUBLESHOOTING, PROGRESS_PROTOCOL, CONTRIBUTING
  project.yml                - xcodegen source of truth (folder-based sources; no per-step file list)
```

Navigation notes:
- `ConversionPlan.workflowMode` is `.convert` (4 steps) or `.expertLab` (6 steps, MoE only).
- Advanced overrides (force dtype / block size) live on **ProfileStep**, not a separate Architecture step.
- JANGTQ module map is in `CLIArgsBuilder.jangtqModuleByModelType` (hard-fail if unmapped).
