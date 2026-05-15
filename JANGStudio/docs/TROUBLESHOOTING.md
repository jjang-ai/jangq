# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Step 1 says "Not a HuggingFace model" | Folder is missing `config.json` | Point at the actual model directory, not its parent |
| JANGTQ tab greyed out | Your model type is not in `jangtq_families` from `python -m jang_tools profiles --json` | Use a JANG profile instead, or add/prove a family-specific JANGTQ converter first |
| Pre-flight: "Disk free" red | Output volume is too small for the estimated size | Pick a different output folder (external drive is fine) |
| Pre-flight: "RAM adequate" yellow warn | RAM < 1.5x source size | Conversion may swap or OOM. Close other apps, or pick a higher-bit profile |
| Run fails with "Killed: 9" or "MemoryError" | Out of RAM. 397B at JANG_1L needs 128 GB+ | Close other apps, or pick a higher-bit profile |
| Run fails at "Detecting architecture" | `config.json` present but missing `model_type` | Re-download the model — some HF repos drop this key |
| Verify fails "Chat template present" | Source HF repo didn't ship a chat template | Add `chat_template.jinja` to the output dir and re-run verify |
| Verify fails "num_hidden_layers > 0" | `config.json` was malformed or partially downloaded | Re-download the source model |
| Verify fails "Video VL preprocessor" | `video_preprocessor_config.json` missing from output (but source had one) | Re-run conversion; the converter should copy it automatically |
| App won't open: "is damaged and can't be opened" | Notarization cache stale | `xattr -cr "/Applications/JANG Studio.app"` |
| App crashes at launch | Bundled Python runtime missing | Reinstall from the DMG; don't copy the .app out of the DMG |
| "Bundled python3 missing" in pre-flight (dev builds only) | Running unsigned debug build without the bundle | Run `cd JANGStudio && Scripts/build-python-bundle.sh` once, or set `JANGSTUDIO_PYTHON_OVERRIDE=$(which python3)` before launching |
