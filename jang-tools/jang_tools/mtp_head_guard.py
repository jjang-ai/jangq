"""MEASURED head truth for every MTP stamp. No exceptions.

════════════════════════════════════════════════════════════════════════════
 THE 2026-09-03/04 MISSTAMP INCIDENT — why this module exists
════════════════════════════════════════════════════════════════════════════
An agent stamped `vmlx_mtp_proposal_head.json` into the dealignai CRACK
bundles by reading the bundle's TOP-LEVEL `quantization.bits` (the tier
default) instead of the per-module `lm_head` override:

  - Qwen3.8-Flash-Next-CRACK-JANG2L shipped `source: q6/g64, ineligible`
    while its real head is q8/g64 → should be ELIGIBLE, proposal_bits 4.
    Every Flash-Next tier keeps an 8-bit/g64 lm_head floor regardless of
    its tier default — the tier default is a lie about the head on every
    tier below 8-bit.
  - Qwen3.8-27B-JANG_2D-CRACK shipped `source: q2` while the head is
    q4/g128 — the wrong validity key defeats the stamp entirely (loaders
    must treat mismatched source as absent and re-derive).

Separately, the earlier misstamp of the SAME class: bundles went out with
blanket `calibrated: true` while modules missing from the capture silently
fell back to uncalibrated quantization. Same disease: METADATA WRITTEN FROM
ASSUMPTION, NOT MEASUREMENT.

════════════════════════════════════════════════════════════════════════════
 THE RULES (also in docs/runtime/MTP-PROPOSAL-HEAD-STAMP-CONTRACT.md,
 AGENTS.md, and the wiki page `mtp-proposal-head-stamp`)
════════════════════════════════════════════════════════════════════════════
1. Never write ANY claim about a head/module's quant layout from a name, a
   tier label, or the config's top-level default. Resolve the per-module
   override, then CROSS-CHECK against the shard bytes:
       bits = packed_u32_cols * 32 / (scales_cols * group_size)
2. Config-vs-shard disagreement is a HARD STOP (`HeadLayoutMismatch`), not
   a value to pick from. A disagreement means the bundle metadata is broken
   (a real case: a speed pack's config said 4-bit, shard packed 8-bit).
3. A stamp is a cache of a measurement, never an authority over the bytes.
   An existing stamp whose `source` matches the measured head is left
   untouched; anything else is rewritten from measured truth.
4. Every stamper that writes an `mtp` block MUST also write the
   proposal-head stamp through `write_proposal_head_stamp()` — forgetting
   the stamp was half of the original incident.

Runtime mirror of the same contract (vmlx Python + Swift): derive the
layout from the LOADED lm_head module, compare all four `source` fields
strictly, and on any mismatch treat the stamp as absent, re-derive with the
pure rule, and atomically overwrite. Fail-open: a stamp must never block or
delay a launch.

stdlib-only on purpose — must run on any Mac with bare python3.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

STAMP_NAME = "vmlx_mtp_proposal_head.json"
STAMP_VERSION = 1

BASIS_ELIGIBLE = (
    "settled 2026-09-03 A/B on Qwen3.8-Flash-Next-JANG_4S fixed-D3: q4 "
    "proposal head +2.3-3.0% count / +1.8-2.8% code over three same-thermal "
    "pairs; target verify keeps the checkpoint head"
)
BASIS_INELIGIBLE = (
    "settled 2026-09-03 A/B on Qwen3.8-Flash-Next-JANG_4S fixed-D3: the q4 "
    "proposal-head gain was measured against a q8/g64 native head; this "
    "bundle's head is already <=6-bit, so a rebuilt proposal copy cannot "
    "reduce proposal cost"
)


class HeadLayoutMismatch(RuntimeError):
    """config.json and the shard bytes disagree about a module's layout.

    Never resolved by picking a side — the bundle metadata is broken and
    must be fixed before any stamp is written.
    """


def shard_header(path: Path | str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def declared_layout(cfg: dict, module: str) -> dict | None:
    """The per-module quantization override for *module*, falling back to
    the top-level spec ONLY when no override exists (MLX semantics). The
    caller must still cross-check the result against the shard bytes —
    a declared layout is a claim, not a measurement."""
    quant = cfg.get("quantization")
    if not isinstance(quant, dict):
        return None
    for key in (module, f"language_model.{module}"):
        if isinstance(quant.get(key), dict):
            return dict(quant[key])
    out = {k: quant[k] for k in ("bits", "group_size", "mode") if k in quant}
    return out or None


def measure_lm_head(bundle: Path | str) -> tuple[int, int, str, bool]:
    """(bits, group_size, mode, tied) of the ACTUAL lm_head in *bundle*.

    Declared layout (per-module override first) cross-checked against the
    packed shard widths; raises HeadLayoutMismatch on disagreement and
    ValueError when the layout cannot be resolved at all.
    """
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    text_cfg = cfg.get("text_config") or cfg
    tied = bool(cfg.get("tie_word_embeddings",
                        text_cfg.get("tie_word_embeddings", False)))

    decl = declared_layout(cfg, "lm_head")
    if not decl or "bits" not in decl or "group_size" not in decl:
        raise ValueError(f"{bundle.name}: cannot resolve lm_head bits/group_size "
                         "from config (no override, no top-level spec)")
    bits, gs = int(decl["bits"]), int(decl["group_size"])
    mode = str(decl.get("mode", "affine"))

    idx = bundle / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        wname = next((k for k in wm if k.endswith("lm_head.weight")), None)
        if wname is not None:
            sname = wname.replace(".weight", ".scales")
            w = shard_header(bundle / wm[wname]).get(wname)
            s = (shard_header(bundle / wm[sname]).get(sname)
                 if sname in wm else None)
            if w is None or w["dtype"] != "U32" or s is None:
                raise HeadLayoutMismatch(
                    f"{bundle.name}: config declares lm_head {bits}b/g{gs} "
                    f"but the shard tensor is not affine-packed")
            measured = w["shape"][-1] * 32 / (s["shape"][-1] * gs)
            if measured != bits:
                raise HeadLayoutMismatch(
                    f"{bundle.name}: config says lm_head {bits}b/g{gs} but "
                    f"the shard packs {measured} bits — fix the bundle "
                    f"metadata first; NEVER stamp either value")
    return bits, gs, mode, tied


def proposal_head_verdict(bits: int, gs: int, mode: str, tied: bool) -> dict:
    """The settled pure eligibility rule. Valid ONLY for calibrated JANG
    bundles (AWQ+imatrix/GPTQ) — do not stamp uncalibrated packs."""
    if tied:
        return {"eligible": False, "reason": "tied_embeddings",
                "basis": BASIS_INELIGIBLE}
    if mode == "affine" and bits == 8 and gs == 64:
        return {"eligible": True, "proposal_bits": 4, "basis": BASIS_ELIGIBLE}
    if bits <= 6:
        return {"eligible": False, "reason": "native_head_already_low_bit",
                "basis": BASIS_INELIGIBLE}
    return {"eligible": False, "reason": f"unmeasured_layout_q{bits}_g{gs}",
            "basis": BASIS_INELIGIBLE}


def write_proposal_head_stamp(bundle: Path | str, family: str | None = None) -> str:
    """Measure the head, derive the verdict, atomically write the stamp.

    Leaves an existing stamp untouched when its `source` matches the
    measured head (the stamp contract: match -> authoritative). Returns a
    one-line human summary. Raises HeadLayoutMismatch/ValueError rather
    than writing anything questionable.
    """
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    if family is None:
        family = cfg.get("model_type") or cfg.get("architectures", [""])[0]

    bits, gs, mode, tied = measure_lm_head(bundle)
    source = {"bits": bits, "group_size": gs, "mode": mode, "tied": tied}
    stamp = {"version": STAMP_VERSION, "family": family, "source": source}
    stamp.update(proposal_head_verdict(bits, gs, mode, tied))

    out = bundle / STAMP_NAME
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("source") == source:
            return f"kept (source matches): {out}"

    fd, tmp = tempfile.mkstemp(dir=bundle, prefix=".mtp_head_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(stamp, f, indent=1)
            f.write("\n")
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    tag = (f"eligible q{bits}->q{stamp['proposal_bits']}" if stamp["eligible"]
           else f"ineligible ({stamp['reason']})")
    return f"stamped {tag}: {out}"
