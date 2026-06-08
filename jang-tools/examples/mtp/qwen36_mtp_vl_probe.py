#!/usr/bin/env python3
"""Probe Qwen3.6 native-MTP artifact routing through the VLM API path."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP")
DEFAULT_WORKTREE = Path(
    "/Users/eric/.config/superpowers/worktrees/vllm-mlx/mtp-qwen36-depth3-20260516"
)

# 1x1 RGB blue PNG.
BLUE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNgYPgP"
    "AAEDAQCtHD2jAAAAAElFTkSuQmCC"
)


def request_json(method: str, url: str, body: Any | None = None, timeout: float = 300.0):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(base_url: str, deadline_s: float):
    end = time.monotonic() + deadline_s
    last_error = None
    while time.monotonic() < end:
        try:
            health = request_json("GET", f"{base_url}/health", timeout=5.0)
            if health.get("status") == "healthy" and health.get("model_loaded"):
                return health
        except Exception as exc:  # noqa: BLE001 - diagnostic launcher
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def terminate(proc: subprocess.Popen[Any]):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--vmlx-worktree",
        type=Path,
        default=Path(os.environ.get("VMLINUX_WORKTREE", DEFAULT_WORKTREE)),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--load-timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"missing model path: {args.model}")
    py = Path(os.environ.get("VMLINUX_BENCH_PYTHON", args.vmlx_worktree / ".venv/bin/python"))
    if not py.exists():
        raise SystemExit(f"missing vMLX python: {py}")

    out = args.out
    if out is None:
        out = args.vmlx_worktree / "docs/internal/release-gates/qwen36_27b_jang4m_mtp_vl"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "server.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VMLINUX_NATIVE_MTP"] = "1"
    cmd = [
        str(py),
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "jangq-qwen36-mtp",
        "--is-mllm",
        "--max-num-seqs",
        "1",
        "--max-tokens",
        "128",
        "--disable-prefix-cache",
        "--log-level",
        "INFO",
    ]
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=args.vmlx_worktree,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        health_before = wait_ready(base_url, args.load_timeout_s)
        image_url = f"data:image/png;base64,{BLUE_PNG_B64}"
        body = {
            "model": "jangq-qwen36-mtp",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is the image? Reply with one word."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = request_json("POST", f"{base_url}/v1/chat/completions", body)
        health_after = request_json("GET", f"{base_url}/health", timeout=30.0)
        cache_stats = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
    finally:
        terminate(proc)

    result = {
        "model": str(args.model),
        "command": cmd,
        "health_before": health_before,
        "health_after": health_after,
        "cache_stats": cache_stats,
        "response": response,
        "server_log": str(log_path),
    }
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    message = ((response.get("choices") or [{}])[0].get("message") or {})
    print(
        json.dumps(
            {
                "result": str(result_path),
                "content": message.get("content"),
                "mtp": health_after.get("mtp"),
                "usage": response.get("usage"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
