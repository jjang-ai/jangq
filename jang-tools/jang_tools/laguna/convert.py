"""Thin wrapper — delegates to the top-level convert_laguna_jangtq /
convert_laguna_mxfp4 modules. Kept so callers can do
`python -m jang_tools.laguna.convert ...`.
"""
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print("usage: python -m jang_tools.laguna.convert {jangtq|mxfp4} <SRC> <OUT> [extra]")
        sys.exit(2)
    mode = sys.argv[1].lower()
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if mode == "jangtq":
        from jang_tools import convert_laguna_jangtq as _m
    elif mode == "mxfp4":
        from jang_tools import convert_laguna_mxfp4 as _m
    else:
        raise SystemExit(f"unknown mode {mode!r}; use jangtq or mxfp4")
    _m.main()


if __name__ == "__main__":
    main()
