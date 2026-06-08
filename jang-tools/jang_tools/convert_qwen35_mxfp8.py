"""Qwen3.5/Qwen3.6 VL + MTP -> MXFP8 affine conversion entrypoint."""

from __future__ import annotations

from jang_tools.convert_qwen35_mxfp4 import main as _main


def main() -> None:
    _main(default_bits=8)


if __name__ == "__main__":
    main()
