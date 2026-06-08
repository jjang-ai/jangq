#!/usr/bin/env python3
"""Run a vMLX same-loader AR-vs-native-MTP speed probe for Qwen3.6-27B.

This is a thin launcher around the vMLX source-tree benchmark. It exists in
jang-tools so the MTP artifact handoff has one obvious command for live proof.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = Path("/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP")
DEFAULT_WORKTREE = Path(
    "/Users/eric/.config/superpowers/worktrees/vllm-mlx/mtp-qwen36-depth3-20260516"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--vmlx-worktree",
        type=Path,
        default=Path(os.environ.get("VMLINUX_WORKTREE", DEFAULT_WORKTREE)),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--cache", choices=["off", "on"], default="off")
    ap.add_argument(
        "--prompt",
        default=(
            "Count from 1 to 120, separated by commas. Do not write any words "
            "before or after the comma-separated numbers."
        ),
    )
    args = ap.parse_args()

    bench = args.vmlx_worktree / "bench/native_mtp_speed_ab.py"
    if not bench.exists():
        raise SystemExit(f"missing benchmark script: {bench}")
    if not args.model.exists():
        raise SystemExit(f"missing model path: {args.model}")

    out = args.out
    if out is None:
        out = (
            args.vmlx_worktree
            / "docs/internal/release-gates/qwen36_27b_jang4m_mtp_baseline_vs_spec"
        )

    cmd = [
        sys.executable,
        str(bench),
        str(args.model),
        "--served-name",
        "jangq-qwen36-mtp",
        "--out",
        str(out),
        "--port",
        str(args.port),
        "--cache",
        args.cache,
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--prompt",
        args.prompt,
    ]
    env = os.environ.copy()
    env.setdefault(
        "VMLINUX_BENCH_PYTHON",
        str(args.vmlx_worktree / ".venv/bin/python"),
    )
    subprocess.run(cmd, cwd=args.vmlx_worktree, env=env, check=True)

    result_path = out / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        print(
            json.dumps(
                {
                    "result": str(result_path),
                    "speedup_vs_baseline": result.get("speedup_vs_baseline"),
                    "output_equivalence": result.get("output_equivalence"),
                    "rows": [
                        {
                            "label": row.get("label"),
                            "summary": row.get("summary"),
                            "mtp": (row.get("health_before") or {}).get("mtp"),
                        }
                        for row in result.get("rows", [])
                    ],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
