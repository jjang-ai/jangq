#!/usr/bin/env python3
"""Send image or video content to a multimodal JANG/JANGTQ server.

This requires a bundle and runtime loader that actually expose the requested
modality. Text-only bundles should use openai_chat.py instead.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


def data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif path.suffix.lower() == ".png":
            mime = "image/png"
        elif path.suffix.lower() == ".mp4":
            mime = "video/mp4"
        else:
            mime = "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.getenv("JANG_MODEL_ALIAS", "default"))
    parser.add_argument("--prompt", default="Describe this media.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.image) == bool(args.video):
        raise SystemExit("Pass exactly one of --image or --video")

    content = [{"type": "text", "text": args.prompt}]
    extra_body = None
    if args.image:
        content.append({"type": "image_url", "image_url": {"url": data_url(args.image)}})
    else:
        content.append({"type": "video_url", "video_url": {"url": data_url(args.video)}})
        extra_body = {"video_fps": args.video_fps, "video_max_frames": args.video_max_frames}

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    kwargs = {}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=args.max_tokens,
        temperature=0.0,
        **kwargs,
    )
    print(response.choices[0].message.content or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
