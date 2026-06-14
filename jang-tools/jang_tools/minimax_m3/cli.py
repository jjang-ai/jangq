"""`jang minimax-m3 <tool>` CLI wiring for the MiniMax-M3 toolchain.

Each underlying tool keeps its own argparse `main()` (so it still runs as
`python -m jang_tools.minimax_m3.<tool>`); this forwards the remaining argv
straight through. Pass `--help` after the tool name to see that tool's own
options, e.g.:

    jang minimax-m3 convert --src <hf> --out <bundle> --keep-experts keep.json
    jang minimax-m3 convert --help
    jang minimax-m3 reap-profile --help

Forwarding is done by `dispatch()`, called from the top-level CLI BEFORE its
strict argparse runs — argparse's REMAINDER cannot capture a leading
``--flag`` (the parent parser consumes it), so the tool args are sliced from
the raw argv instead.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import importlib
import sys

# CLI subcommand name → module under jang_tools.minimax_m3 (each exposes main())
_TOOLS = {
    "convert":      ("convert_jang",  "Convert M3 source → JANG_2L affine bundle (AWQ + REAP prune)"),
    "probe":        ("probe",         "Coherence probe via streamed layer-by-layer forward"),
    "reap-profile": ("reap_profile",  "Compute REAP expert-saliency profile (Σ gate·‖expert(x)‖)"),
    "reap-select":  ("reap_select",   "Select kept experts from a REAP profile → keep map JSON"),
    "awq-capture":  ("awq_capture",   "Capture AWQ activation scales for routed gate/up"),
}


def _print_group_help() -> int:
    print("usage: jang minimax-m3 <tool> [args]   "
          "(append --help after a tool for its options)\n")
    print("MiniMax-M3 (minimax_m3_vl) tools:")
    for name, (_mod, helptext) in _TOOLS.items():
        print(f"  {name:13s} {helptext}")
    return 0


def dispatch(argv: list[str]) -> int:
    """Run `minimax-m3 <tool> <args...>` from the raw argv slice (post `minimax-m3`)."""
    if not argv or argv[0] in ("-h", "--help"):
        return _print_group_help()
    tool = argv[0]
    if tool not in _TOOLS:
        print(f"jang minimax-m3: unknown tool {tool!r}\n")
        _print_group_help()
        return 2
    module_suffix = _TOOLS[tool][0]
    mod = importlib.import_module(f"jang_tools.minimax_m3.{module_suffix}")
    # Forward the rest to the tool's own argparse main().
    sys.argv = [f"jang minimax-m3 {tool}"] + list(argv[1:])
    ret = mod.main()
    return ret if isinstance(ret, int) else 0


def register(subparsers) -> None:
    """Register `minimax-m3` for `jang --help` discoverability.

    Execution is handled by `dispatch()` via early-intercept in __main__
    (so tool --flags forward correctly); this parser exists for help listing
    and as a fallback if the intercept is ever bypassed.
    """
    p = subparsers.add_parser(
        "minimax-m3",
        help="MiniMax-M3 (minimax_m3_vl) convert / probe / REAP / AWQ tooling",
        add_help=False,
    )
    p.set_defaults(func=lambda a: sys.exit(dispatch(getattr(a, "_m3_rest", []) or [])))
