"""Shared read-side JSON-loader helpers.

M152 (iter 75): extracted after 5 local copies of the same template
accumulated across the codebase:
  - format/reader.py (M149)
  - jangspec/manifest.py (M148)
  - capabilities.py (M150, tuple-variant)
  - examples.py (M126)
  - loader.py (M151)

Every call site wants the same thing: parse a user-visible JSON file,
fail with an actionable ValueError that names:
  1. the file's path on disk
  2. what the file is for (a short "purpose" tag)
  3. the specific failure mode (disk read / malformed JSON / wrong
     top-level type / schema violation)

Two contracts are supported:
  - :func:`read_json_object` — raise-contract for caller paths that
    propagate errors.
  - :func:`read_json_object_safe` — tuple-return contract for caller
    paths that must never raise (e.g., detection probes,
    verify_directory-style ``-> tuple[bool, str]`` functions).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json_object(path: Path, *, purpose: str) -> dict[str, Any]:
    """Load a JSON file expected to have a dict root.

    Args:
        path: absolute or relative filesystem path. Converted via
            ``Path(path)`` so callers can pass strings too.
        purpose: short noun phrase used in error messages to identify
            what the file is for (e.g., ``"JANG config"``,
            ``"shard index"``, ``"model config"``). Required keyword
            argument because good error messages are not optional.

    Returns:
        The parsed top-level JSON object as a dict.

    Raises:
        ValueError: on read failure (OSError / UnicodeDecodeError),
            parse failure (JSONDecodeError), or non-dict root. Every
            message includes ``path`` and ``purpose``. Wraps the
            original exception via ``from exc`` so debuggers can see
            the underlying cause.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read {purpose} at {p}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{purpose} at {p} is not valid JSON "
            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{purpose} at {p} has a top-level {type(data).__name__}, "
            f"expected a JSON object"
        )
    return data


def read_json_object_safe(path: Path, *, purpose: str) -> tuple[dict[str, Any] | None, str | None]:
    """Contract-preserving variant: returns ``(data, None)`` on success
    and ``(None, error_message)`` on failure.

    Use for callers that have a documented ``-> tuple[bool, str]``
    return type (e.g., ``capabilities.verify_directory``) or for
    detection probes that must return False on any failure rather
    than raise.
    """
    try:
        return read_json_object(path, purpose=purpose), None
    except ValueError as exc:
        return None, str(exc)


def write_json_object_atomic(
    path: Path,
    data: dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    """Atomically replace a model metadata JSON object.

    The temporary file lives beside the destination, so ``os.replace`` stays
    on one filesystem. Readers therefore see either the previous complete
    object or the new complete object, never a truncated/intermediate stamp.
    The file and containing directory are fsynced best-effort before return.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if p.exists():
            os.chmod(temporary, p.stat().st_mode & 0o777)
        os.replace(temporary, p)
        temporary = None
        try:
            directory_fd = os.open(p.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not permit directory fsync. The atomic
            # same-directory replace still prevents torn reader-visible JSON.
            pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
