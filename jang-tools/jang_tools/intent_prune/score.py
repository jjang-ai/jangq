"""Hybrid Intent Prune scorer (hybrid_v1): path + mass + domain + safety.

Normative fusion (plan §11.6 Balanced preset)::

    base = 0.30*ng + 0.20*nm + 0.35*ni - 0.10*nd + 0.05*ng   # backbone uses ng
    Keep:     score = base + 0.15*ns
    Balanced: score = base + 0.05*ns
    CRACK:    score = base - 0.25*ns*(1-ni)

Uniform keep-K per layer with stable tie-break (higher mass, then lower expert id).
Emits ``jang-intent-prune-plan-v1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .graph import (
    DEFAULT_MAX_ITER,
    DEFAULT_TELEPORT,
    DEFAULT_TOL,
    mass_has_signal,
    mass_matrix_from_adjacency,
    path_scores_from_transitions,
    stationary_from_adjacency,
)
from .transitions import (
    CRACK_PROBE_MARKERS,
    SAFETY_PROBE_MARKERS,
    TRANSITION_SCHEMA,
    build_adjacency_from_transitions,
    iter_transition_records,
)

def _normalized_marker(raw: str) -> str:
    """Match transitions._normalized_marker (hyphen/space → underscore)."""
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


# Domain tags that count as safety/crack evidence (shared with transitions emission).
_SAFETY_DOMAIN_MARKERS = frozenset(
    {_normalized_marker(m) for m in SAFETY_PROBE_MARKERS}
    | {_normalized_marker(m) for m in CRACK_PROBE_MARKERS}
)

PLAN_SCHEMA = "jang-intent-prune-plan-v1"
SCORER_NAME = "hybrid_v1"

# ---------------------------------------------------------------------------
# SHIP DEFAULTS — frozen hybrid_v1 fusion weights (IP5 / PR-IP5)
#
# Do not retune casually. These constants are the Studio/CLI ship defaults for
# Intent Prune until a new bake-off re-opens the matrix.
#
# Evidence and freeze rationale:
#   docs/plans/2026-07-12-intent-prune-bakeoff.md
# Normative formulas:
#   docs/plans/2026-07-12-jang-studio-intent-prune-crack.md §10.2 / §11.6
# ---------------------------------------------------------------------------

# SHIP DEFAULTS — Balanced preset (Studio/CLI default for hybrid_v1)
BALANCED_WEIGHTS: dict[str, float] = {
    "path": 0.30,
    "mass": 0.20,
    "intent": 0.35,
    "drop": 0.10,
    "backbone_floor": 0.05,
    "safety_keep": 0.15,
    "safety_balanced": 0.05,
    "safety_crack": 0.25,
}

# SHIP DEFAULTS — Specialist preset (intent-heavy; same safety terms as Balanced)
SPECIALIST_WEIGHTS: dict[str, float] = {
    "path": 0.20,
    "mass": 0.15,
    "intent": 0.50,
    "drop": 0.10,
    "backbone_floor": 0.05,
    "safety_keep": 0.15,
    "safety_balanced": 0.05,
    "safety_crack": 0.25,
}

# SHIP DEFAULTS — preset registry; default key is "balanced"
PRESET_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": BALANCED_WEIGHTS,
    "specialist": SPECIALIST_WEIGHTS,
}

# SHIP DEFAULTS — default preset and safety stance for scorer entry points
DEFAULT_PRESET = "balanced"
DEFAULT_SAFETY_STANCE = "balanced"

SAFETY_STANCES = ("keep", "balanced", "crack")


def norm_layer(values: Sequence[float]) -> list[float]:
    """Per-layer max-normalize to ``[0, 1]``. All-zero → zeros."""
    if not values:
        return []
    peak = 0.0
    for v in values:
        fv = float(v)
        if fv > peak:
            peak = fv
    if peak <= 0.0:
        return [0.0] * len(values)
    inv = 1.0 / peak
    return [float(v) * inv for v in values]


def zeros_like(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[0.0] * len(row) for row in matrix]


def _domain_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        value = raw.strip().lower()
        return {value} if value else set()
    if isinstance(raw, (list, tuple, set)):
        out: set[str] = set()
        for item in raw:
            if item is None:
                continue
            value = str(item).strip().lower()
            if value:
                out.add(value)
        return out
    value = str(raw).strip().lower()
    return {value} if value else set()


def filter_records_for_intents(
    records: Iterable[Mapping[str, Any]],
    intents: Sequence[str],
) -> list[dict[str, Any]]:
    """Keep records whose domains intersect ``intents`` (case-insensitive)."""
    want = _domain_set(intents)
    if not want:
        return []
    out: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        domains = _domain_set(record.get("domains"))
        if domains & want:
            out.append(dict(record))
    return out


def filter_records_for_safety(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Records flagged as safety/crack probes (or carrying safety domains).

    Domain markers are shared with :mod:`transitions` emission
    (``SAFETY_PROBE_MARKERS`` / ``CRACK_PROBE_MARKERS``) so JSONL that only
    carries domain tags still contributes to ``π_S``.
    """
    out: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if record.get("safety_probe") or record.get("crack_probe"):
            out.append(dict(record))
            continue
        domains = {_normalized_marker(d) for d in _domain_set(record.get("domains"))}
        if domains & _SAFETY_DOMAIN_MARKERS:
            out.append(dict(record))
    return out


def _matrix_has_positive(matrix: Sequence[Sequence[float]]) -> bool:
    for row in matrix:
        for value in row:
            if float(value) > 0.0:
                return True
    return False


def resolve_weights(preset: str = DEFAULT_PRESET) -> dict[str, float]:
    key = (preset or DEFAULT_PRESET).strip().lower()
    if key not in PRESET_WEIGHTS:
        raise ValueError(
            f"unknown preset {preset!r}; expected one of {sorted(PRESET_WEIGHTS)}"
        )
    return dict(PRESET_WEIGHTS[key])


def fusion_score_layer(
    pi_g: Sequence[float],
    mass: Sequence[float],
    pi_i: Sequence[float] | None = None,
    pi_d: Sequence[float] | None = None,
    pi_s: Sequence[float] | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    safety_stance: str = DEFAULT_SAFETY_STANCE,
) -> list[float]:
    """Hybrid fusion for one layer (plan §11.6)."""
    w = dict(weights or BALANCED_WEIGHTS)
    stance = (safety_stance or DEFAULT_SAFETY_STANCE).strip().lower()
    if stance not in SAFETY_STANCES:
        raise ValueError(
            f"unknown safety_stance {safety_stance!r}; expected one of {SAFETY_STANCES}"
        )

    n = len(pi_g)
    if len(mass) != n:
        raise ValueError(f"mass length {len(mass)} != pi_g length {n}")

    ng = norm_layer(pi_g)
    nm = norm_layer(mass)
    ni = norm_layer(pi_i if pi_i is not None else [0.0] * n)
    nd = norm_layer(pi_d if pi_d is not None else [0.0] * n)
    ns = norm_layer(pi_s if pi_s is not None else [0.0] * n)

    path_w = float(w.get("path", 0.30))
    mass_w = float(w.get("mass", 0.20))
    intent_w = float(w.get("intent", 0.35))
    drop_w = float(w.get("drop", 0.10))
    backbone_w = float(w.get("backbone_floor", 0.05))
    safety_keep_w = float(w.get("safety_keep", 0.15))
    safety_bal_w = float(w.get("safety_balanced", 0.05))
    safety_crack_w = float(w.get("safety_crack", 0.25))

    scores: list[float] = []
    for e in range(n):
        base = (
            path_w * ng[e]
            + mass_w * nm[e]
            + intent_w * ni[e]
            - drop_w * nd[e]
            + backbone_w * ng[e]
        )
        if stance == "keep":
            score = base + safety_keep_w * ns[e]
        elif stance == "balanced":
            score = base + safety_bal_w * ns[e]
        else:  # crack
            specificity = ns[e] * (1.0 - ni[e])
            score = base - safety_crack_w * specificity
        scores.append(float(score))
    return scores


def select_keep_k(
    scores: Sequence[float],
    mass: Sequence[float],
    k: int,
) -> list[int]:
    """Top-K expert indices. Tie-break: higher mass, then lower expert id."""
    n = len(scores)
    if n == 0:
        return []
    k = max(0, min(int(k), n))
    if k == 0:
        return []
    # Sort key: (-score, -mass, +expert_id)
    order = sorted(
        range(n),
        key=lambda e: (-float(scores[e]), -float(mass[e]) if e < len(mass) else 0.0, e),
    )
    keep = sorted(order[:k])
    return keep


def mass_only_scores(mass: Sequence[Sequence[float]]) -> list[list[float]]:
    """Mass-only ranking signal (for A/B vs hybrid)."""
    return [norm_layer(row) for row in mass]


@dataclass
class HybridScoreResult:
    """In-memory hybrid scoring result before plan emission."""

    scores: list[list[float]]
    keep: dict[int, list[int]]
    pi_g: list[list[float]]
    mass: list[list[float]]
    pi_i: list[list[float]]
    pi_d: list[list[float]]
    pi_s: list[list[float]]
    num_layers: int
    num_experts: int
    keep_k: int
    preset: str
    weights: dict[str, float]
    safety_stance: str
    intents_keep: list[str] = field(default_factory=list)
    intents_drop: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def layers_map(self) -> dict[str, list[int]]:
        return {str(layer): list(experts) for layer, experts in sorted(self.keep.items())}


def _align_matrix(
    matrix: Sequence[Sequence[float]] | None,
    num_layers: int,
    num_experts: int,
) -> list[list[float]]:
    out = [[0.0] * num_experts for _ in range(num_layers)]
    if not matrix:
        return out
    for layer, row in enumerate(matrix):
        if layer >= num_layers:
            break
        for expert, value in enumerate(row):
            if expert >= num_experts:
                break
            out[layer][expert] = float(value)
    return out


def _path_matrix_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    num_experts: int,
    num_layers: int,
    weight_by_gate: bool,
    teleport: float,
    tol: float,
    max_iter: int,
) -> list[list[float]]:
    if not records:
        return [[0.0] * num_experts for _ in range(num_layers)]
    result = path_scores_from_transitions(
        records,
        num_experts=num_experts,
        num_layers=num_layers,
        weight_by_gate=weight_by_gate,
        teleport=teleport,
        tol=tol,
        max_iter=max_iter,
    )
    return _align_matrix(result["pi"], num_layers, num_experts)


def score_hybrid(
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    adjacency: Mapping[str, Any] | None = None,
    num_experts: int | None = None,
    num_layers: int | None = None,
    keep_k: int,
    intents_keep: Sequence[str] | None = None,
    intents_drop: Sequence[str] | None = None,
    safety_stance: str = DEFAULT_SAFETY_STANCE,
    preset: str = DEFAULT_PRESET,
    weights: Mapping[str, float] | None = None,
    weight_by_gate: bool = True,
    teleport: float = DEFAULT_TELEPORT,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    # Optional precomputed signals (testing / advanced)
    pi_g: Sequence[Sequence[float]] | None = None,
    mass: Sequence[Sequence[float]] | None = None,
    pi_i: Sequence[Sequence[float]] | None = None,
    pi_d: Sequence[Sequence[float]] | None = None,
    pi_s: Sequence[Sequence[float]] | None = None,
) -> HybridScoreResult:
    """Score experts with hybrid_v1 and select uniform keep-K per layer."""
    w = dict(weights) if weights is not None else resolve_weights(preset)
    stance = (safety_stance or DEFAULT_SAFETY_STANCE).strip().lower()
    if stance not in SAFETY_STANCES:
        raise ValueError(
            f"unknown safety_stance {safety_stance!r}; expected one of {SAFETY_STANCES}"
        )
    preset_key = (preset or DEFAULT_PRESET).strip().lower()

    keep_intents = [str(x) for x in (intents_keep or []) if str(x).strip()]
    drop_intents = [str(x) for x in (intents_drop or []) if str(x).strip()]
    record_list: list[Mapping[str, Any]] = list(records) if records is not None else []

    # Resolve adjacency / global path + mass
    adj: dict[str, Any] | None = dict(adjacency) if adjacency is not None else None
    if adj is None and record_list:
        e_guess = int(num_experts) if num_experts is not None and int(num_experts) > 0 else None
        adj = build_adjacency_from_transitions(
            record_list,
            num_experts=e_guess,
            weight_by_gate=weight_by_gate,
        )

    e_width = int(
        num_experts
        if num_experts is not None and int(num_experts) > 0
        else (adj.get("num_experts") if adj else 0) or 0
    )
    if e_width <= 0 and pi_g is not None and pi_g:
        e_width = len(pi_g[0])
    if e_width <= 0 and mass is not None and mass:
        e_width = len(mass[0])
    if e_width <= 0:
        raise ValueError("num_experts is required (or provide adjacency/records with width)")

    if pi_g is not None:
        n_layers = len(pi_g)
    elif mass is not None:
        n_layers = len(mass)
    elif adj is not None:
        n_layers = int(num_layers) if num_layers is not None and int(num_layers) > 0 else 0
        if n_layers <= 0:
            n_layers = int(adj.get("num_layers_observed") or 0)
    else:
        n_layers = int(num_layers or 0)

    path_meta: dict[str, Any] = {}
    if pi_g is None:
        if adj is None:
            raise ValueError("records or adjacency or pi_g required for path scores")
        path_result = stationary_from_adjacency(
            adj,
            num_experts=e_width,
            num_layers=n_layers if n_layers > 0 else None,
            teleport=teleport,
            tol=tol,
            max_iter=max_iter,
            require_signal=False,
        )
        pi_g_m = path_result["pi"]
        n_layers = int(path_result["num_layers"])
        path_meta = {
            "edge_count": int(path_result.get("edge_count") or 0),
            "iterations": int(path_result.get("iterations") or 0),
            "converged": bool(path_result.get("converged", True)),
            "delta": float(path_result.get("delta") or 0.0),
            "has_signal": bool(path_result.get("has_signal", False)),
        }
    else:
        pi_g_m = _align_matrix(pi_g, len(pi_g), e_width)
        n_layers = len(pi_g_m)

    if n_layers <= 0:
        raise ValueError(
            "empty graph: num_layers resolved to 0; no edges or mass to score"
        )

    if mass is None:
        if adj is None:
            mass_m = [[0.0] * e_width for _ in range(n_layers)]
        else:
            mass_m = mass_matrix_from_adjacency(
                adj, num_layers=n_layers, num_experts=e_width
            )
    else:
        mass_m = _align_matrix(mass, n_layers, e_width)

    pi_g_m = _align_matrix(pi_g_m, n_layers, e_width)

    # Fail closed: zero-signal graphs must not emit arbitrary keep-[0..K-1] plans.
    # Usable signal = positive path mass, positive selection mass, or any of the
    # conditional path scores (precomputed). Empty edges+mass with only teleport
    # uniforms is rejected.
    has_path = _matrix_has_positive(pi_g_m)
    has_mass = mass_has_signal(mass_m)
    has_precomputed_conditional = any(
        x is not None for x in (pi_i, pi_d, pi_s)
    ) and any(
        _matrix_has_positive(_align_matrix(x, n_layers, e_width))
        for x in (pi_i, pi_d, pi_s)
        if x is not None
    )
    if not has_path and not has_mass and not has_precomputed_conditional:
        raise ValueError(
            "empty graph: no edges or mass (and no precomputed path/mass signals); "
            "refusing to emit an arbitrary keep-K plan"
        )

    # Conditional path scores from record subsets
    if pi_i is not None:
        pi_i_m = _align_matrix(pi_i, n_layers, e_width)
    elif record_list and keep_intents:
        pi_i_m = _path_matrix_from_records(
            filter_records_for_intents(record_list, keep_intents),
            num_experts=e_width,
            num_layers=n_layers,
            weight_by_gate=weight_by_gate,
            teleport=teleport,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        pi_i_m = [[0.0] * e_width for _ in range(n_layers)]

    if pi_d is not None:
        pi_d_m = _align_matrix(pi_d, n_layers, e_width)
    elif record_list and drop_intents:
        pi_d_m = _path_matrix_from_records(
            filter_records_for_intents(record_list, drop_intents),
            num_experts=e_width,
            num_layers=n_layers,
            weight_by_gate=weight_by_gate,
            teleport=teleport,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        pi_d_m = [[0.0] * e_width for _ in range(n_layers)]

    if pi_s is not None:
        pi_s_m = _align_matrix(pi_s, n_layers, e_width)
    elif record_list:
        safety_recs = filter_records_for_safety(record_list)
        if safety_recs:
            pi_s_m = _path_matrix_from_records(
                safety_recs,
                num_experts=e_width,
                num_layers=n_layers,
                weight_by_gate=weight_by_gate,
                teleport=teleport,
                tol=tol,
                max_iter=max_iter,
            )
        else:
            pi_s_m = [[0.0] * e_width for _ in range(n_layers)]
    else:
        pi_s_m = [[0.0] * e_width for _ in range(n_layers)]

    k = int(keep_k)
    if k < 1:
        raise ValueError(f"keep_k must be >= 1, got {keep_k}")
    if k > e_width:
        k = e_width

    scores: list[list[float]] = []
    keep: dict[int, list[int]] = {}
    for layer in range(n_layers):
        layer_scores = fusion_score_layer(
            pi_g_m[layer],
            mass_m[layer],
            pi_i_m[layer],
            pi_d_m[layer],
            pi_s_m[layer],
            weights=w,
            safety_stance=stance,
        )
        scores.append(layer_scores)
        keep[layer] = select_keep_k(layer_scores, mass_m[layer], k)

    return HybridScoreResult(
        scores=scores,
        keep=keep,
        pi_g=pi_g_m,
        mass=mass_m,
        pi_i=pi_i_m,
        pi_d=pi_d_m,
        pi_s=pi_s_m,
        num_layers=n_layers,
        num_experts=e_width,
        keep_k=k,
        preset=preset_key,
        weights=w,
        safety_stance=stance,
        intents_keep=keep_intents,
        intents_drop=drop_intents,
        meta={
            "scorer": SCORER_NAME,
            "teleport": float(teleport),
            "record_count": len(record_list),
            "weight_by_gate": bool(weight_by_gate),
            "has_path_signal": has_path,
            "has_mass_signal": has_mass,
            **path_meta,
        },
    )


def build_prune_plan(
    result: HybridScoreResult,
    *,
    source_model: str | None = None,
    backend: str | None = None,
    suite: Mapping[str, Any] | None = None,
    crack_pack: Mapping[str, Any] | None = None,
    trained_top_k: int | None = None,
    comparison_summary: Mapping[str, Any] | None = None,
    eval_index: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a ``jang-intent-prune-plan-v1`` document from a score result."""
    top_k = int(trained_top_k) if trained_top_k is not None else None
    issues: list[str] = []
    if top_k is not None and result.keep_k < top_k:
        issues.append(
            f"keep_experts_per_layer={result.keep_k} < trained_top_k={top_k}"
        )
    safety_passed = len(issues) == 0

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "scorer": SCORER_NAME,
        "preset": result.preset,
        "weights": dict(result.weights),
        "intents_keep": list(result.intents_keep),
        "intents_drop": list(result.intents_drop),
        "safety_stance": result.safety_stance,
        "keep_experts_per_layer": int(result.keep_k),
        "num_experts_source": int(result.num_experts),
        "num_layers": int(result.num_layers),
        "suite": dict(suite) if suite else {},
        "crack_pack": dict(crack_pack) if crack_pack else {},
        "safety": {
            "passed": safety_passed,
            "minimum_active_experts_per_layer": int(result.keep_k),
            "trained_top_k": top_k if top_k is not None else int(result.keep_k),
            "issues": issues,
        },
        "layers": result.layers_map(),
        "comparison_summary": dict(comparison_summary) if comparison_summary else {},
        "eval_index": dict(eval_index) if eval_index else {},
        "created_at": created_at
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_model": source_model or "",
        "backend": backend or "",
    }
    if extra:
        for key, value in extra.items():
            if key not in plan:
                plan[key] = value
    return plan


def write_prune_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def score_transitions_to_plan(
    transitions_path: str | Path,
    *,
    keep_k: int,
    num_experts: int,
    num_layers: int | None = None,
    intents_keep: Sequence[str] | None = None,
    intents_drop: Sequence[str] | None = None,
    safety_stance: str = DEFAULT_SAFETY_STANCE,
    preset: str = DEFAULT_PRESET,
    weight_by_gate: bool = True,
    source_model: str | None = None,
    backend: str | None = None,
    suite: Mapping[str, Any] | None = None,
    crack_pack: Mapping[str, Any] | None = None,
    trained_top_k: int | None = None,
    teleport: float = DEFAULT_TELEPORT,
) -> dict[str, Any]:
    """Load ``expert_transitions.jsonl``, score hybrid_v1, return plan dict."""
    records = list(iter_transition_records(transitions_path))
    result = score_hybrid(
        records=records,
        num_experts=num_experts,
        num_layers=num_layers,
        keep_k=keep_k,
        intents_keep=intents_keep,
        intents_drop=intents_drop,
        safety_stance=safety_stance,
        preset=preset,
        weight_by_gate=weight_by_gate,
        teleport=teleport,
    )
    return build_prune_plan(
        result,
        source_model=source_model,
        backend=backend,
        suite=suite,
        crack_pack=crack_pack,
        trained_top_k=trained_top_k,
        extra={"transitions": str(Path(transitions_path).expanduser())},
    )


# ---------------------------------------------------------------------------
# Synthetic highway fixture (plan Appendix C) — shared by tests
# ---------------------------------------------------------------------------

# L=3, E=4; H=1 highway; N=3 chatty / dead-end mass
HIGHWAY_H = 1
HIGHWAY_N = 3
HIGHWAY_L = 3
HIGHWAY_E = 4


def build_synthetic_highway_records(
    *,
    highway_tokens: int = 6,
    noise_tokens: int = 40,
    noise_score: float = 5.0,
) -> list[dict[str, Any]]:
    """Construct Appendix C synthetic graph as transition records.

    * Expert **H** (id=1) is rare but sits on every multi-layer path between
      hub experts 0 (layer 0) and 2 (layer 2).
    * Expert **N** (id=3) appears often with high gate mass on single-layer
      dead-end selections (no layer→layer edges), so mass-only ranks N ≫ H
      while path/hybrid ranks H > N.
    """
    records: list[dict[str, Any]] = []
    token = 0
    for _ in range(int(highway_tokens)):
        records.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_id": "highway",
                "domains": ["code"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": token,
                "path": [
                    {"layer": 0, "experts": [0], "scores": [1.0]},
                    {"layer": 1, "experts": [HIGHWAY_H], "scores": [1.0]},
                    {"layer": 2, "experts": [2], "scores": [1.0]},
                ],
            }
        )
        token += 1
        # Alternate reverse hub direction for a bit more path mass on ends
        records.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_id": "highway",
                "domains": ["code"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": token,
                "path": [
                    {"layer": 0, "experts": [2], "scores": [1.0]},
                    {"layer": 1, "experts": [HIGHWAY_H], "scores": [1.0]},
                    {"layer": 2, "experts": [0], "scores": [1.0]},
                ],
            }
        )
        token += 1

    for i in range(int(noise_tokens)):
        # Single-layer dead-end: high mass, no adjacency edges.
        records.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_id": "noise",
                "domains": ["chat"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": token + i,
                "path": [
                    {
                        "layer": 1,
                        "experts": [HIGHWAY_N],
                        "scores": [float(noise_score)],
                    }
                ],
            }
        )
    return records
