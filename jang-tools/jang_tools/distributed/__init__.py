"""Distributed inference / convert for large MoE models.

Default reference rig (any two Apple Silicon Macs):
- Node A: primary (Mac Studio / Mac Pro / M-series MacBook with most RAM)
- Node B: secondary
- Interconnect: Thunderbolt 5 / 4 / 3 (Thunderbolt Bridge), or LAN fallback

Stack
-----
1. `discovery.py`   — find peers (Tailscale + Bonjour), emit hostfile.json
2. `tb5_probe.py`   — measure bandwidth + latency before correctness runs
3. `jaccl_init.py`  — `mx.distributed.init(backend="jaccl")`, fallback ring
4. `sharding.py`    — EP/PP/TP plans for MiMoV2 (and any DSV-style MoE)
5. `dist_runtime.py` (per-model, e.g. mimo_v2/dist_runtime.py)
6. `scripts/`       — bash bring-up: TB5 bridge, mlx.launch wrappers

Launch convention (matches mlx-examples/cifar):
    mlx.launch --verbose --hostfile hostfile.json \\
        -m jang_tools.mimo_v2.dist_runtime --src ... --prompt ...

This package is the staging ground; everything ports to vmlx swift/python
once the Python reference is correct.
"""

from .discovery import build_hostfile
from .jaccl_init import init_world, World

__all__ = ["build_hostfile", "init_world", "World"]
