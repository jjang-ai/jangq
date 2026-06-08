"""Capture host readiness for Nemotron Ultra runtime speed probes.

This is a cheap no-model-load snapshot for memory pressure, swap, disk
headroom, and high-RSS processes. It helps separate real runtime bottlenecks
from host pressure that can make decode probes noisy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")
DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json")


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output = proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    return proc.returncode, output.strip()


def _parse_memsize(output: str) -> float | None:
    text = output.strip()
    if not text.isdigit():
        return None
    return int(text) / (1024**3)


def _parse_vm_stat(output: str) -> dict[str, Any]:
    page_size = 16384
    page_match = re.search(r"page size of (\d+) bytes", output)
    if page_match:
        page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"Pages ([^:]+):\s+([\d.]+)\.", line.strip())
        if match:
            pages[match.group(1).lower().replace(" ", "_")] = int(match.group(2).replace(".", ""))
    free_like_pages = (
        pages.get("free", 0)
        + pages.get("inactive", 0)
        + pages.get("speculative", 0)
        + pages.get("purgeable", 0)
    )
    active_pages = pages.get("active", 0)
    wired_pages = pages.get("wired_down", pages.get("wired", 0))
    compressed_pages = pages.get("occupied_by_compressor", 0)
    return {
        "page_size": page_size,
        "pages": pages,
        "free_like_gib": free_like_pages * page_size / (1024**3),
        "active_gib": active_pages * page_size / (1024**3),
        "wired_gib": wired_pages * page_size / (1024**3),
        "compressed_gib": compressed_pages * page_size / (1024**3),
    }


def _parse_memory_pressure(output: str) -> dict[str, Any]:
    swapins = None
    swapouts = None
    pressure = "UNKNOWN"
    if "System-wide memory free percentage" in output:
        pressure = "OK"
    if "The system has memory pressure" in output:
        pressure = "PRESSURE"
    for line in output.splitlines():
        if "Swapins:" in line:
            match = re.search(r"Swapins:\s*(\d+)", line)
            if match:
                swapins = int(match.group(1))
        if "Swapouts:" in line:
            match = re.search(r"Swapouts:\s*(\d+)", line)
            if match:
                swapouts = int(match.group(1))
    return {"pressure": pressure, "swapins": swapins, "swapouts": swapouts}


def _parse_df(output: str) -> dict[str, Any] | None:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    parts = lines[-1].split()
    if len(parts) < 6:
        return None
    try:
        return {
            "filesystem": parts[0],
            "size_gib": float(parts[1]),
            "used_gib": float(parts[2]),
            "available_gib": float(parts[3]),
            "capacity": parts[4],
            "mount": parts[-1],
        }
    except ValueError:
        return None


def _parse_ps(output: str, *, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        rows.append({"rss_gib": int(parts[0]) / (1024**2), "command": parts[1]})
    rows.sort(key=lambda item: item["rss_gib"], reverse=True)
    return rows[:limit]


def _collect(bundle: Path, log_dir: Path, *, min_disk_free_gib: float, target_wired_gib: float) -> dict[str, Any]:
    mem_code, mem_out = _run(["sysctl", "-n", "hw.memsize"])
    vm_code, vm_out = _run(["vm_stat"])
    pressure_code, pressure_out = _run(["memory_pressure"])
    df_bundle_code, df_bundle_out = _run(["df", "-g", str(bundle if bundle.exists() else bundle.parent)])
    df_log_code, df_log_out = _run(["df", "-g", str(log_dir if log_dir.exists() else log_dir.parent)])
    ps_code, ps_out = _run(["ps", "-axo", "rss,command"])

    total_gib = _parse_memsize(mem_out) if mem_code == 0 else None
    vm = _parse_vm_stat(vm_out) if vm_code == 0 else {}
    pressure = _parse_memory_pressure(pressure_out) if pressure_code == 0 else {"pressure": "UNKNOWN"}
    bundle_disk = _parse_df(df_bundle_out) if df_bundle_code == 0 else None
    log_disk = _parse_df(df_log_out) if df_log_code == 0 else None
    high_rss = _parse_ps(ps_out) if ps_code == 0 else []

    warnings: list[str] = []
    if total_gib is not None and total_gib < 128:
        warnings.append(f"physical memory is {total_gib:.1f} GiB, below the 128 GiB target")
    if vm.get("free_like_gib") is not None and vm["free_like_gib"] < 16:
        warnings.append(f"free/inactive/speculative/purgeable memory is only {vm['free_like_gib']:.1f} GiB")
    if pressure.get("pressure") == "PRESSURE":
        warnings.append("macOS reports memory pressure")
    if bundle_disk and bundle_disk["available_gib"] < min_disk_free_gib:
        warnings.append(f"bundle volume has only {bundle_disk['available_gib']:.1f} GiB free")
    if log_disk and log_disk["available_gib"] < min_disk_free_gib:
        warnings.append(f"log volume has only {log_disk['available_gib']:.1f} GiB free")
    if vm.get("wired_gib") is not None and vm["wired_gib"] > target_wired_gib:
        warnings.append(f"wired memory is {vm['wired_gib']:.1f} GiB, above target {target_wired_gib:.1f} GiB")
    if high_rss and high_rss[0]["rss_gib"] > 20:
        warnings.append(
            f"largest resident process is {high_rss[0]['rss_gib']:.1f} GiB; close it before loading the 98G bundle if possible"
        )

    status = "READY" if not warnings else "WATCH"
    interpretation = [
        "Saved speed logs still point at MoE and Mamba compute/dispatch as the primary bottlenecks.",
        "Host pressure can add noise or stalls, so rerun expensive probes only when this report is READY or warnings are understood.",
        "Closing high-RSS apps may help stability, but it is not proven to fix the current 65.773 ms MoE and 64.157 ms Mamba buckets.",
    ]
    return {
        "status": status,
        "bundle": str(bundle),
        "log_dir": str(log_dir),
        "thresholds": {
            "min_disk_free_gib": min_disk_free_gib,
            "target_wired_gib": target_wired_gib,
        },
        "memory": {
            "total_gib": total_gib,
            "vm_stat": vm,
            "memory_pressure": pressure,
        },
        "disk": {
            "bundle": bundle_disk,
            "log_dir": log_disk,
        },
        "high_rss_processes": high_rss,
        "warnings": warnings,
        "interpretation": interpretation,
        "commands": {
            "refresh": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/host_runtime_readiness.py "
                f"--bundle {bundle} --log-dir {log_dir}"
            )
        },
    }


def _fmt_gib(value: Any) -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.1f} GiB"


def _render(result: dict[str, Any]) -> str:
    memory = result["memory"]
    vm = memory.get("vm_stat", {})
    pressure = memory.get("memory_pressure", {})
    bundle_disk = result["disk"].get("bundle") or {}
    log_disk = result["disk"].get("log_dir") or {}
    lines = [
        "# Nemotron Ultra Host Runtime Readiness",
        "",
        f"status: `{result['status']}`",
        f"bundle: `{result['bundle']}`",
        f"log_dir: `{result['log_dir']}`",
        "",
        "## Memory",
        f"- total: `{_fmt_gib(memory.get('total_gib'))}`",
        f"- free_like: `{_fmt_gib(vm.get('free_like_gib'))}`",
        f"- active: `{_fmt_gib(vm.get('active_gib'))}`",
        f"- wired: `{_fmt_gib(vm.get('wired_gib'))}`",
        f"- compressed: `{_fmt_gib(vm.get('compressed_gib'))}`",
        f"- memory_pressure: `{pressure.get('pressure', 'UNKNOWN')}`",
        f"- swapins: `{pressure.get('swapins')}`",
        f"- swapouts: `{pressure.get('swapouts')}`",
        "",
        "## Disk",
        f"- bundle_volume_free: `{_fmt_gib(bundle_disk.get('available_gib'))}` at `{bundle_disk.get('mount')}`",
        f"- log_volume_free: `{_fmt_gib(log_disk.get('available_gib'))}` at `{log_disk.get('mount')}`",
        "",
        "## High RSS Processes",
    ]
    for process in result["high_rss_processes"]:
        lines.append(f"- `{process['rss_gib']:.2f} GiB` {process['command']}")
    lines.extend(["", "## Warnings"])
    if result["warnings"]:
        lines.extend(f"- {warning}" for warning in result["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Interpretation"])
    lines.extend(f"- {item}" for item in result["interpretation"])
    lines.extend(["", "## Commands", f"- refresh: `{result['commands']['refresh']}`"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--min-disk-free-gib", type=float, default=25.0)
    ap.add_argument("--target-wired-gib", type=float, default=105.0)
    args = ap.parse_args()

    result = _collect(
        args.bundle,
        args.log_dir,
        min_disk_free_gib=args.min_disk_free_gib,
        target_wired_gib=args.target_wired_gib,
    )
    report = _render(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
