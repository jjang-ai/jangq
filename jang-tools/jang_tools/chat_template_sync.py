"""Keep a bundle's two chat-template locations from drifting apart.

A bundle can carry its chat template in two places:

  1. ``chat_template.jinja``            — what transformers >= 5 loads, and
                                          what takes precedence.
  2. ``tokenizer_config.json``'s
     ``chat_template`` field            — the older location, still read by
                                          older transformers and by plenty of
                                          third-party tooling.

When those two disagree the bundle serves correctly under a current runtime
and hands every other consumer a DIFFERENT template. Nothing errors; the
model simply behaves oddly somewhere else, which is far harder to diagnose
than a hard failure.

Observed 2026-08-16 across 59 bundles on the model drive: four were
divergent. ``Zaya-8B-JANG_4M``'s embedded copy was byte-identical to its own
``chat_template.jinja.bak-preitemsfix`` (i.e. a fix applied to the ``.jinja``
never reached the embedded copy), and three ``Nemotron-Omni-Nano`` variants
(JANGTQ / JANGTQ4 / MXFP4-CRACK) shared an IDENTICAL divergent pair — one
stale template carried through the pipeline three separate times.

The converters mostly copy config files verbatim from the source model, so
whatever inconsistency the source had is faithfully reproduced. This module
is the shared fix: ``sync_bundle_chat_template`` makes a bundle
self-consistent by writing the ``.jinja`` text into the embedded field.

``convert_inkling_jang_affine`` already does the equivalent inline; this
generalises that behaviour so every bundle can get it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "TEMPLATE_CONSISTENT",
    "TEMPLATE_DIVERGENT",
    "TEMPLATE_EMBEDDED_ONLY",
    "TEMPLATE_JINJA_ONLY",
    "TEMPLATE_NONE",
    "audit_bundle_chat_template",
    "sync_bundle_chat_template",
]

TEMPLATE_CONSISTENT = "consistent"
TEMPLATE_DIVERGENT = "divergent"
TEMPLATE_JINJA_ONLY = "jinja_only"
TEMPLATE_EMBEDDED_ONLY = "embedded_only"
TEMPLATE_NONE = "no_template"


def _digest(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.strip().encode()).hexdigest()[:12]


def _read_jinja(bundle: Path) -> str | None:
    path = bundle / "chat_template.jinja"
    return path.read_text() if path.is_file() else None


def _read_embedded(bundle: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = bundle / "tokenizer_config.json"
    if not path.is_file():
        return None, None
    config = json.loads(path.read_text())
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        return config, None
    return config, template


def audit_bundle_chat_template(bundle: Path) -> dict[str, Any]:
    """Classify a bundle's template sources. Never raises."""
    result: dict[str, Any] = {"status": "unreadable", "jinja": None, "embedded": None}
    try:
        jinja = _read_jinja(bundle)
        _, embedded = _read_embedded(bundle)
        result["jinja"] = _digest(jinja)
        result["embedded"] = _digest(embedded)
        if jinja is None and embedded is None:
            result["status"] = TEMPLATE_NONE
        elif jinja is None:
            result["status"] = TEMPLATE_EMBEDDED_ONLY
        elif embedded is None:
            result["status"] = TEMPLATE_JINJA_ONLY
        elif jinja.strip() == embedded.strip():
            result["status"] = TEMPLATE_CONSISTENT
        else:
            result["status"] = TEMPLATE_DIVERGENT
    except Exception:
        pass
    return result


def sync_bundle_chat_template(bundle: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Make a bundle self-consistent by copying the .jinja into the embedded field.

    The ``.jinja`` is authoritative because it is what a current runtime
    actually loads — syncing the other direction would change serving
    behaviour, which this must never do.

    Only ``divergent`` bundles are rewritten. ``jinja_only`` is left ALONE on
    purpose: 19 of the 59 bundles surveyed ship that way deliberately, and
    adding an embedded copy to them would create a second source of truth
    that can drift later — the very problem this exists to prevent.

    Returns the audit dict with ``changed`` and (when rewritten) ``backup``.
    """
    audit = audit_bundle_chat_template(bundle)
    audit["changed"] = False
    if audit["status"] != TEMPLATE_DIVERGENT:
        return audit

    jinja = _read_jinja(bundle)
    config, _ = _read_embedded(bundle)
    if jinja is None or config is None:
        return audit

    if dry_run:
        audit["changed"] = True
        audit["dry_run"] = True
        return audit

    target = bundle / "tokenizer_config.json"
    backup = target.with_suffix(".json.bak-pre-template-sync")
    if not backup.exists():
        backup.write_text(target.read_text())
        audit["backup"] = backup.name

    config["chat_template"] = jinja
    # Bundles are read by many tools; keep the on-disk shape boring and
    # diff-friendly rather than clever.
    target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    audit["changed"] = True
    audit["status_after"] = audit_bundle_chat_template(bundle)["status"]
    return audit
