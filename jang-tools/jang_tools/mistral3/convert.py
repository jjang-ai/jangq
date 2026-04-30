"""Thin wrapper — delegates to convert_mistral3_jangtq / _mxfp4."""
import sys


def main():
    if len(sys.argv) < 4:
        print("usage: python -m jang_tools.mistral3.convert {jangtq|mxfp4} <SRC> <OUT> [extra]")
        sys.exit(2)
    mode = sys.argv[1].lower()
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if mode == "jangtq":
        from jang_tools import convert_mistral3_jangtq as _m
    elif mode == "mxfp4":
        from jang_tools import convert_mistral3_mxfp4 as _m
    else:
        raise SystemExit(f"unknown mode {mode!r}; use jangtq or mxfp4")
    _m.main()


if __name__ == "__main__":
    main()
