"""Stamp vmlx_mtp_proposal_head.json into JANG bundle roots.

Thin CLI over jang_tools.mtp_head_guard — ALL measurement, eligibility, and
write logic lives there (single source of truth; see its docstring for the
2026-09-03/04 misstamp incident this design prevents). This file must never
grow its own head-layout logic.

Usage:
  python -m jang_tools.stamp_mtp_proposal_head <bundle_dir> [<bundle_dir> ...]

Also runs as a bare file (scp it together with mtp_head_guard.py to any Mac
with plain python3 — both are stdlib-only).
"""

from __future__ import annotations

import sys

try:
    from jang_tools.mtp_head_guard import write_proposal_head_stamp
except ImportError:  # bare-file execution next to mtp_head_guard.py
    import importlib.util
    from pathlib import Path
    _p = Path(__file__).resolve().parent / "mtp_head_guard.py"
    _spec = importlib.util.spec_from_file_location("mtp_head_guard", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    write_proposal_head_stamp = _m.write_proposal_head_stamp


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    failed = False
    for arg in sys.argv[1:]:
        try:
            print(write_proposal_head_stamp(arg))
        except Exception as e:
            failed = True
            print(f"REFUSED {arg}: {e}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
