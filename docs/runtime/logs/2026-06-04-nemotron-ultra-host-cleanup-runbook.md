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
