"""Peer discovery → hostfile.json (mlx.launch format).

mlx.launch hostfile schema (per cifar example):
    [
        {"ssh": "host-to-ssh-to", "ips": ["ip-to-bind-to"]},
        ...
    ]

This module walks Tailscale, Bonjour, and a manual override list, then
emits a hostfile pinning each node to its TB5 bridge IP (preferred for
bandwidth) and its Tailscale IP (fallback).

Usage
-----
    python -m jang_tools.distributed.discovery \\
        --out hostfile.json \\
        --pin macstudio:tb5 macbook:tb5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TAILSCALE_BIN = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
TB5_BRIDGE_IFACE_HINT = "bridge"  # macOS Thunderbolt Bridge


@dataclass
class Peer:
    name: str
    ssh: str            # SSH alias (~/.ssh/config)
    tb5_ip: str | None  # Thunderbolt-5 bridge IP (preferred)
    ts_ip: str | None   # Tailscale IP
    lan_ip: str | None  # LAN IP (fallback)


def _tailscale_status() -> dict[str, str]:
    if not Path(TAILSCALE_BIN).exists():
        return {}
    try:
        out = subprocess.check_output([TAILSCALE_BIN, "status", "--json"],
                                      text=True, timeout=5)
    except Exception:
        return {}
    data = json.loads(out)
    peers = {}
    for p in (data.get("Peer") or {}).values():
        host = (p.get("HostName") or "").lower()
        if not host:
            continue
        ips = p.get("TailscaleIPs") or []
        if ips:
            peers[host] = ips[0]
    self_host = (data.get("Self", {}).get("HostName") or "").lower()
    if self_host and (data.get("Self", {}).get("TailscaleIPs") or []):
        peers[self_host] = data["Self"]["TailscaleIPs"][0]
    return peers


def _bonjour_resolve(name: str) -> str | None:
    if shutil.which("dns-sd") is None:
        return None
    return None  # placeholder — resolution is async; SSH config is preferred


def build_hostfile(peers: list[Peer], prefer: str = "tb5") -> list[dict]:
    """Produce mlx.launch-compatible hostfile entries.

    `prefer` selects which IP family to bind the launcher to first:
        "tb5"  → tb5_ip then ts_ip then lan_ip
        "ts"   → ts_ip then tb5_ip then lan_ip
        "lan"  → lan_ip then ts_ip then tb5_ip
    """
    order = {
        "tb5": ("tb5_ip", "ts_ip", "lan_ip"),
        "ts":  ("ts_ip", "tb5_ip", "lan_ip"),
        "lan": ("lan_ip", "ts_ip", "tb5_ip"),
    }[prefer]

    out = []
    for p in peers:
        ips = []
        for k in order:
            v = getattr(p, k)
            if v:
                ips.append(v)
        if not ips:
            raise ValueError(f"peer {p.name}: no usable IP")
        out.append({"ssh": p.ssh, "ips": ips})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hostfile.json")
    ap.add_argument("--prefer", default="tb5", choices=("tb5", "ts", "lan"))
    ap.add_argument("--peers", nargs="*", default=[],
                    help="name,ssh,tb5_ip[,ts_ip[,lan_ip]] tuples; "
                         "e.g. macstudio,macstudio,10.0.42.1,100.107.102.99")
    args = ap.parse_args()

    ts = _tailscale_status()
    peers: list[Peer] = []
    for raw in args.peers:
        bits = raw.split(",")
        name = bits[0]
        ssh = bits[1] if len(bits) > 1 else name
        tb5 = bits[2] if len(bits) > 2 else None
        ts_ip = bits[3] if len(bits) > 3 else ts.get(name.lower())
        lan = bits[4] if len(bits) > 4 else None
        peers.append(Peer(name, ssh, tb5, ts_ip, lan))

    hf = build_hostfile(peers, prefer=args.prefer)
    Path(args.out).write_text(json.dumps(hf, indent=2))
    print(f"wrote {args.out} with {len(hf)} hosts")


if __name__ == "__main__":
    main()
