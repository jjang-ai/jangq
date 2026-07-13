"""CRACK abliteration probe pack: load, fingerprint, naming.

CRACK is JANG's product label for abliteration / reduced-refusal stance on
Intent Prune (plan §7). The versioned pack lives under
``intent_prune/assets/crack_probes_v1.jsonl``.

When ``safety_stance=crack``, prune plans must:

* attach ``crack_pack`` metadata (name, sha256, prompt_count)
* use the ``-CRACK`` artifact name suffix (plan §15.1)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

CRACK_PACK_NAME = "crack-probes-v1"
CRACK_PACK_VERSION = "v1"
CRACK_PACK_FILENAME = "crack_probes_v1.jsonl"
CRACK_SUFFIX = "CRACK"
CRACK_STANCE = "crack"

# Content classes owned by IP2 (plan §7.4)
CRACK_CLASSES = (
    "over_refusal",
    "benign_dual_use",
    "policy_edge",
    "still_refuse",
)

# Expected response labels for pack rows
EXPECTED_BEHAVIORS = ("comply", "refuse")

# Stable size bounds for the shipped pack
MIN_CRACK_PROBES = 15
MAX_CRACK_PROBES = 30
# At least one still-should-refuse anchor (plan §7.4)
MIN_STILL_REFUSE_ANCHORS = 1

_INTENT_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_INTENT_SLUG_LEN = 48

# Accept common hyphen/alias spellings, map to CRACK_CLASSES
_CLASS_ALIASES: dict[str, str] = {
    "overrefusal": "over_refusal",
    "over_refusals": "over_refusal",
    "benign_dualuse": "benign_dual_use",
    "dual_use": "benign_dual_use",
    "dualuse": "benign_dual_use",
    "policy": "policy_edge",
    "policy_edge_case": "policy_edge",
    "still_should_refuse": "still_refuse",
    "should_refuse": "still_refuse",
    "anchor": "still_refuse",
}


def default_crack_pack_path() -> Path:
    """Absolute path to the shipped ``crack_probes_v1.jsonl`` asset."""
    return Path(__file__).resolve().parent / "assets" / CRACK_PACK_FILENAME


def is_crack_stance(safety_stance: str | None) -> bool:
    """True when the safety stance is CRACK abliteration."""
    return (safety_stance or "").strip().lower() == CRACK_STANCE


def is_default_crack_pack_path(path: str | Path | None) -> bool:
    """True when ``path`` is the shipped default asset (or omitted)."""
    if path is None:
        return True
    try:
        return Path(path).expanduser().resolve() == default_crack_pack_path().resolve()
    except OSError:
        return False


def crack_pack_name_for_path(path: str | Path | None = None) -> str:
    """Pack display name: shipped asset → ``crack-probes-v1``; else filename stem.

    Custom ``--crack-pack`` paths must not claim the shipped pack name.
    Stem hyphens/underscores preserved; ``.jsonl`` suffix stripped.
    """
    if path is None or is_default_crack_pack_path(path):
        return CRACK_PACK_NAME
    pack_path = Path(path).expanduser()
    stem = pack_path.stem.strip()
    if not stem:
        return CRACK_PACK_NAME
    # Normalize underscores to hyphens for stable plan labels
    return stem.replace("_", "-")


def normalize_crack_class(raw: Any) -> str:
    """Map a row ``class`` value to a canonical CRACK_CLASSES member (or '')."""
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not value:
        return ""
    value = _CLASS_ALIASES.get(value, value)
    return value if value in CRACK_CLASSES else value


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file (streaming)."""
    h = hashlib.sha256()
    with Path(path).expanduser().open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(data: bytes | str) -> str:
    """SHA-256 hex digest of raw bytes or UTF-8 text."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def crack_pack_fingerprint(
    path: str | Path | None = None,
    *,
    content: bytes | str | None = None,
) -> str:
    """Fingerprint (sha256) of the CRACK pack bytes.

    Prefer ``path`` (file on disk). If ``content`` is provided, hash that
    instead (useful for tests / in-memory packs). Default path is the
    shipped asset.
    """
    if content is not None:
        return content_sha256(content)
    pack_path = Path(path).expanduser() if path is not None else default_crack_pack_path()
    if not pack_path.is_file():
        raise FileNotFoundError(f"CRACK pack not found: {pack_path}")
    return file_sha256(pack_path)


def _row_expected_behavior(row: Mapping[str, Any], *, probe_class: str) -> str:
    raw = (
        row.get("expected_behavior")
        or row.get("expectedBehavior")
        or row.get("expected")
        or ""
    )
    value = str(raw).strip().lower()
    if value in {"refuse", "refusal", "deny", "block"}:
        return "refuse"
    if value in {"comply", "answer", "allow", "help"}:
        return "comply"
    if probe_class == "still_refuse":
        return "refuse"
    return "comply"


def load_crack_pack(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load and validate CRACK probe pack JSONL.

    Compatible with Expert Lab / expert_lab_vmlx suite rows: each object has
    at least ``id`` and ``prompt`` (or ``text``).

    Structural checks (plan §7.4):

    * size in ``[MIN_CRACK_PROBES, MAX_CRACK_PROBES]``
    * every row ``class`` ∈ :data:`CRACK_CLASSES`
    * at least :data:`MIN_STILL_REFUSE_ANCHORS` rows with ``class=still_refuse``
      and expected refuse behavior
    """
    pack_path = Path(path).expanduser() if path is not None else default_crack_pack_path()
    if not pack_path.is_file():
        raise FileNotFoundError(f"CRACK pack not found: {pack_path}")

    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        pack_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"CRACK pack line {line_number} must be a JSON object")
        if "prompt" not in row and "text" in row:
            row = dict(row)
            row["prompt"] = row["text"]
        prompt_id = str(row.get("id") or row.get("prompt_id") or "").strip()
        if not prompt_id:
            raise ValueError(f"CRACK pack line {line_number} missing non-empty id")
        if prompt_id in seen:
            raise ValueError(f"CRACK pack line {line_number} duplicate id {prompt_id!r}")
        prompt_text = str(row.get("prompt") or "").strip()
        if not prompt_text:
            raise ValueError(f"CRACK pack line {line_number} id {prompt_id!r} has empty prompt")

        probe_class = normalize_crack_class(row.get("class") or row.get("probe_class"))
        if probe_class not in CRACK_CLASSES:
            raise ValueError(
                f"CRACK pack line {line_number} id {prompt_id!r} has invalid class "
                f"{row.get('class')!r}; expected one of {CRACK_CLASSES}"
            )

        # Normalize flags so transition emission treats rows as crack probes
        row = dict(row)
        row["id"] = prompt_id
        row["prompt"] = prompt_text
        row["class"] = probe_class
        row["expected_behavior"] = _row_expected_behavior(row, probe_class=probe_class)
        row.setdefault("crack_probe", True)
        row.setdefault("safety_probe", False)
        tags = row.get("tags")
        if not isinstance(tags, list):
            tags = []
        tag_set = {str(t).strip().lower() for t in tags if str(t).strip()}
        if "crack" not in tag_set:
            tags = list(tags) + ["crack"]
        row["tags"] = tags
        seen.add(prompt_id)
        prompts.append(row)

    n = len(prompts)
    if n < MIN_CRACK_PROBES or n > MAX_CRACK_PROBES:
        raise ValueError(
            f"CRACK pack size {n} outside allowed range "
            f"[{MIN_CRACK_PROBES}, {MAX_CRACK_PROBES}]"
        )

    anchors = [
        r
        for r in prompts
        if r.get("class") == "still_refuse" and r.get("expected_behavior") == "refuse"
    ]
    if len(anchors) < MIN_STILL_REFUSE_ANCHORS:
        raise ValueError(
            f"CRACK pack requires at least {MIN_STILL_REFUSE_ANCHORS} "
            f"still_refuse anchor(s) with expected_behavior=refuse; found {len(anchors)}"
        )
    return prompts


def crack_pack_meta(
    path: str | Path | None = None,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """Plan-ready ``crack_pack`` block: name, sha256, prompt_count, path.

    When ``name`` is omitted, the shipped default pack is labeled
    ``crack-probes-v1``; any other path uses the file stem so custom
    ``--crack-pack`` paths are not mislabeled as the shipped pack.
    """
    pack_path = Path(path).expanduser() if path is not None else default_crack_pack_path()
    rows = load_crack_pack(pack_path)
    pack_name = name if name is not None and str(name).strip() else crack_pack_name_for_path(pack_path)
    return {
        "name": pack_name,
        "version": CRACK_PACK_VERSION,
        "sha256": crack_pack_fingerprint(pack_path),
        "prompt_count": len(rows),
        "path": str(pack_path.resolve()),
        "filename": pack_path.name,
        "classes": sorted({str(r.get("class") or "") for r in rows if r.get("class")}),
    }


def resolve_crack_pack_for_plan(
    safety_stance: str,
    crack_pack: Mapping[str, Any] | None = None,
    *,
    crack_pack_path: str | Path | None = None,
    attach_default: bool = True,
) -> dict[str, Any]:
    """Return crack_pack metadata for plan emission.

    * Non-CRACK stance → ``{}`` unless an explicit pack dict/path is passed.
    * CRACK stance → explicit meta, or default shipped pack when
      ``attach_default`` is true.
    * Custom paths never claim ``crack-probes-v1`` unless that is their stem.
    """
    if crack_pack:
        out = dict(crack_pack)
        if "sha256" not in out and crack_pack_path is not None:
            out["sha256"] = crack_pack_fingerprint(crack_pack_path)
        if "prompt_count" not in out and crack_pack_path is not None:
            out["prompt_count"] = len(load_crack_pack(crack_pack_path))
        if "name" not in out or not str(out.get("name") or "").strip():
            out["name"] = (
                crack_pack_name_for_path(crack_pack_path)
                if crack_pack_path is not None
                else CRACK_PACK_NAME
            )
        return out

    if crack_pack_path is not None:
        return crack_pack_meta(crack_pack_path)

    if is_crack_stance(safety_stance) and attach_default:
        return crack_pack_meta()

    return {}


# ---------------------------------------------------------------------------
# Naming (plan §15.1)
# ---------------------------------------------------------------------------


def normalize_intent_slug(raw: str) -> str:
    """Single intent chip → lowercase slug (hyphenated)."""
    text = str(raw or "").strip().lower().replace("_", "-")
    text = _INTENT_SLUG_RE.sub("-", text).strip("-")
    return text


def join_intent_slugs(
    intents: Sequence[str] | None,
    *,
    max_len: int = _MAX_INTENT_SLUG_LEN,
) -> str:
    """Sorted keep-intent chips joined by ``-`` (plan §15.1), length-clamped."""
    slugs = sorted({normalize_intent_slug(x) for x in (intents or []) if normalize_intent_slug(x)})
    if not slugs:
        return "general"
    joined = "-".join(slugs)
    if len(joined) <= max_len:
        return joined
    # Prefer full leading slugs; truncate if needed
    out: list[str] = []
    budget = max_len
    for slug in slugs:
        piece = slug if not out else f"-{slug}"
        if len(piece) > budget:
            break
        out.append(slug)
        budget -= len(piece)
    if not out:
        return joined[:max_len].rstrip("-")
    return "-".join(out)


def has_crack_suffix(name: str) -> bool:
    """True if basename already ends with ``-CRACK`` (case-insensitive)."""
    base = str(name or "").rstrip("/").split("/")[-1]
    return base.upper().endswith(f"-{CRACK_SUFFIX}") or base.upper().endswith(CRACK_SUFFIX)


def apply_crack_suffix(
    basename: str,
    safety_stance: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Append ``-CRACK`` when stance is crack (or ``force``), idempotent.

    Non-CRACK stances return ``basename`` unchanged unless ``force=True``.
    """
    name = str(basename or "").strip()
    if not name:
        name = "model"
    if not force and not is_crack_stance(safety_stance):
        return name
    if has_crack_suffix(name):
        # Normalize to canonical uppercase suffix
        if name.endswith(f"-{CRACK_SUFFIX}"):
            return name
        lower = name.lower()
        if lower.endswith(f"-{CRACK_SUFFIX.lower()}"):
            return name[: -len(CRACK_SUFFIX)] + CRACK_SUFFIX
        if lower.endswith(CRACK_SUFFIX.lower()) and not lower.endswith(
            f"-{CRACK_SUFFIX.lower()}"
        ):
            return name[: -len(CRACK_SUFFIX)] + CRACK_SUFFIX
        return name
    return f"{name}-{CRACK_SUFFIX}"


def intent_prune_artifact_name(
    basename: str,
    *,
    intents_keep: Sequence[str] | None = None,
    keep_k: int,
    safety_stance: str = "balanced",
) -> str:
    """Build ``{basename}-intent-{slugs}-k{K}[-CRACK]`` (plan §15.1)."""
    base = str(basename or "").strip() or "model"
    # Strip a pre-existing -CRACK so we control placement after k{K}
    if has_crack_suffix(base):
        if base.upper().endswith(f"-{CRACK_SUFFIX}"):
            base = base[: -(len(CRACK_SUFFIX) + 1)]
        elif base.upper().endswith(CRACK_SUFFIX):
            base = base[: -len(CRACK_SUFFIX)].rstrip("-")
    slug = join_intent_slugs(intents_keep)
    k = int(keep_k)
    name = f"{base}-intent-{slug}-k{k}"
    return apply_crack_suffix(name, safety_stance)
