"""Publish a converted JANG/JANGTQ model to HuggingFace Hub.

Wraps huggingface_hub.upload_folder with auto-generated model card.
Requires HF_HUB_TOKEN env var or a token file path via --token.

M43 (iter 23): with `--progress=json`, iterates files via `HfApi.upload_file`
and emits JSONL progress events to stderr (same schema as convert's
5-phase protocol) so Swift PublishService can render a progress bar
instead of a 30-minute spinner. Without the flag, falls back to the
faster `upload_folder` bulk call.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from ._json_utils import read_json_object_safe
from .capabilities import validate_jang_runtime_contract, verify_directory
from .modelcard import generate_card
from .progress import ProgressEmitter


def _upload_with_progress(model_dir: Path, repo_id: str, token: str,
                          emitter: ProgressEmitter,
                          commit_message: str,
                          upload_file: Callable | None = None) -> str:
    """Iterate files under model_dir and upload one at a time, emitting
    JSONL progress between. Returns the repo URL on success.

    `upload_file` is an injection point so tests can mock the HfApi call
    without touching the network. Prod callers leave it None; production
    uses `HfApi(token=token).upload_file`.

    M114 (iter 38): exclude `jang_imatrix.safetensors` from the upload.
    The imatrix is useful LOCALLY during subsequent converts (cached via
    `--imatrix-path`) but is just bloat on HF — it doesn't participate in
    inference and balloons the repo size (per `feedback_model_checklist.md`'s
    "155 GB bloat" incident that this sweep was written to prevent).
    """
    _EXCLUDE_FROM_UPLOAD = {
        "jang_imatrix.safetensors",
    }
    files = sorted(p for p in model_dir.rglob("*")
                   if p.is_file() and p.name not in _EXCLUDE_FROM_UPLOAD)
    if not files:
        raise RuntimeError(f"no files to upload under {model_dir}")
    total_bytes = sum(p.stat().st_size for p in files)

    emitter.phase(1, 3, "scan")
    emitter.event("info", f"enumerated {len(files)} files totalling {total_bytes / 1e9:.2f} GB")

    if upload_file is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        upload_file = api.upload_file

    emitter.phase(2, 3, "upload")
    uploaded_bytes = 0
    for idx, path in enumerate(files, start=1):
        rel = str(path.relative_to(model_dir))
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=repo_id,
            token=token,
            commit_message=f"{commit_message} ({idx}/{len(files)}: {rel})",
        )
        uploaded_bytes += path.stat().st_size
        # Tick throttling in ProgressEmitter handles coalescing — we emit every
        # file, it coalesces to a tick every 100ms (or per JANG_TICK_THROTTLE_MS).
        emitter.tick(uploaded_bytes, total_bytes, label=rel)

    # ProgressEmitter.tick auto-detects is_final via `done >= total - 1` and
    # bypasses the throttle on that branch — so the 100% line always lands.
    emitter.phase(3, 3, "finalize")
    return f"https://huggingface.co/{repo_id}"


def cmd_publish(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)

    # Reject incomplete authoritative metadata before token resolution, model
    # card writes, repository creation, or upload. Truly legacy bundles with
    # neither authoritative schema remain compatible with config.json fallback.
    jang_path = model_dir / "jang_config.json"
    if jang_path.exists():
        jang, error = read_json_object_safe(
            jang_path,
            purpose="jang_config.json publish metadata",
        )
        if error is not None:
            print(f"ERROR: refusing to publish invalid JANG metadata: {error}", file=sys.stderr)
            sys.exit(2)
        ok, message = validate_jang_runtime_contract(jang)
        if not ok:
            print(
                f"ERROR: refusing to publish invalid JANG metadata: {message}",
                file=sys.stderr,
            )
            sys.exit(2)
        if "capabilities" in jang:
            ok, message = verify_directory(model_dir)
            if not ok:
                print(
                    f"ERROR: refusing to publish invalid JANG metadata: {message}",
                    file=sys.stderr,
                )
                sys.exit(2)

    # Resolve HF token. PRIORITY:
    #   1. --token FILEPATH  (read token from a file — safe; file mode 600 recommended)
    #   2. HF_HUB_TOKEN env var  (safe — visible only to the process and root)
    #   3. HUGGING_FACE_HUB_TOKEN env var  (HF's canonical name)
    #
    # Historical note (M41): `--token LITERAL` used to be accepted. It was
    # removed because Swift's Process call would expose the token in `ps aux`
    # to any local user for the ENTIRE duration of a multi-hour upload. The
    # Swift PublishService now always sets HF_HUB_TOKEN in the child env.
    token = None
    if args.token:
        token_path = Path(args.token)
        if token_path.exists():
            token = token_path.read_text().strip()
        else:
            # Refuse to accept literal tokens on argv — they leak via `ps aux`.
            print(
                "ERROR: --token must be a FILE PATH to a token file. "
                "To pass a literal token, set HF_HUB_TOKEN env var instead "
                "(argv is visible to `ps aux` for the full upload window).",
                file=sys.stderr,
            )
            sys.exit(2)
    if not token:
        token = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: set HF_HUB_TOKEN env var or pass --token <filepath>", file=sys.stderr)
        sys.exit(2)

    # Ensure README.md exists with a generated card
    readme = model_dir / "README.md"
    if not readme.exists() or args.regenerate_card:
        try:
            # M202 (iter 138): generate_card returns (card, license_unknown).
            # Surface the license-unknown warning to stderr so the user sees
            # it BEFORE the upload proceeds — publishing a fabricated license
            # to HF is legally concerning (Qwen/Llama derivatives != apache-2.0).
            card, license_unknown = generate_card(model_dir)
            if license_unknown:
                print(
                    "WARNING: source config.json had no `license` key; the "
                    "auto-generated README uses `license: other`. Edit "
                    "README.md's YAML frontmatter BEFORE publishing to match "
                    "the source model's actual license.",
                    file=sys.stderr,
                )
            readme.write_text(card)
        except Exception as e:
            print(f"ERROR: failed to generate model card: {e}", file=sys.stderr)
            sys.exit(3)

    if args.dry_run:
        # M114: dry-run count/size must match what _upload_with_progress
        # actually uploads (imatrix excluded) — otherwise user sees "42 files
        # 187 GB" in preview but 41 files 182 GB actually upload, confusing.
        _EXCLUDE = {"jang_imatrix.safetensors"}
        to_upload = [p for p in model_dir.rglob("*")
                     if p.is_file() and p.name not in _EXCLUDE]
        result = {
            "dry_run": True,
            "repo": args.repo,
            "private": args.private,
            "model_card_path": str(readme),
            "files_count": len(to_upload),
            "total_size_bytes": sum(p.stat().st_size for p in to_upload),
        }
    else:
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
            sys.exit(2)

        try:
            create_repo(repo_id=args.repo, token=token, private=args.private, exist_ok=True)
            commit_message = f"Upload {model_dir.name} via jang-tools"
            if args.progress == "json":
                # Per-file upload with JSONL progress on stderr. Slower than
                # upload_folder's bulk commit but gives live progress for
                # Swift PublishService to render a progress bar instead of a
                # 30-minute spinner. See M43 in ralph audit.
                emitter = ProgressEmitter(json_to_stderr=True, quiet_text=True)
                url = _upload_with_progress(
                    model_dir=model_dir,
                    repo_id=args.repo,
                    token=token,
                    emitter=emitter,
                    commit_message=commit_message,
                )
                emitter.done(ok=True, output=url)
                result = {
                    "dry_run": False,
                    "repo": args.repo,
                    "url": url,
                    "commit_url": url,  # per-file path doesn't expose a single commit
                }
            else:
                api = HfApi(token=token)
                info = api.upload_folder(
                    folder_path=str(model_dir),
                    repo_id=args.repo,
                    token=token,
                    commit_message=commit_message,
                )
                result = {
                    "dry_run": False,
                    "repo": args.repo,
                    "url": f"https://huggingface.co/{args.repo}",
                    "commit_url": str(info),
                }
        except Exception as e:
            # Scrub the token from the exception text before printing — some HF
            # errors include the Authorization header in the exception message.
            msg = f"{type(e).__name__}: {e}"
            if token and token in msg:
                msg = msg.replace(token, "<redacted>")
            print(f"ERROR: publish failed: {msg}", file=sys.stderr)
            sys.exit(3)

    if args.json:
        print(json.dumps(result, indent=None))
    else:
        if result.get("dry_run"):
            print(f"DRY RUN — would publish {result['files_count']} files ({result['total_size_bytes']/1e9:.2f} GB) to {result['repo']}")
        else:
            print(f"Published to: {result['url']}")


def register(subparsers) -> None:
    p = subparsers.add_parser("publish", help="Publish converted model to HuggingFace Hub")
    p.add_argument("--model", required=True, help="Path to converted model dir")
    p.add_argument("--repo", required=True, help="Target HF repo id (e.g., my-org/my-model-JANG_4K)")
    p.add_argument("--private", action="store_true", help="Create as private repo")
    p.add_argument("--token",
                   help="Path to a file containing the HF token. "
                        "For literal tokens set HF_HUB_TOKEN env var "
                        "(argv leaks via `ps aux` for the whole upload).")
    p.add_argument("--regenerate-card", action="store_true",
                   help="Overwrite existing README.md with freshly generated card")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan but don't upload — returns file count + size")
    p.add_argument("--json", action="store_true")
    p.add_argument("--progress", choices=["none", "json"], default="none",
                   help="Stream per-file upload progress as JSONL to stderr. "
                        "Emits the same schema as convert's 5-phase protocol. "
                        "Slower than upload_folder's bulk commit but gives live "
                        "progress for UIs to render a bar instead of a spinner.")
    p.set_defaults(func=cmd_publish)
