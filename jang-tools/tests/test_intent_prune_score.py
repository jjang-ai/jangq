"""Unit tests for hybrid_v1 Intent Prune scorer (path + mass + domain).

Required Appendix C case: synthetic highway expert H (rare, on path) vs
chatty N (high mass, dead-end) — mass-only ranks N>H; hybrid ranks H>N;
keep-K=2 retains H.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from jang_tools.intent_prune.graph import (
    build_row_stochastic,
    mass_matrix_from_adjacency,
    node_id,
    power_iteration,
    stationary_from_adjacency,
)
from jang_tools.intent_prune.score import (
    BALANCED_WEIGHTS,
    HIGHWAY_E,
    HIGHWAY_H,
    HIGHWAY_L,
    HIGHWAY_N,
    PLAN_SCHEMA,
    SCORER_NAME,
    SPECIALIST_WEIGHTS,
    build_prune_plan,
    build_synthetic_highway_records,
    fusion_score_layer,
    mass_only_scores,
    norm_layer,
    score_hybrid,
    select_keep_k,
    write_prune_plan,
)
from jang_tools.intent_prune.transitions import (
    build_adjacency_from_transitions,
    write_transitions_jsonl,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_norm_layer_max_normalizes():
    assert norm_layer([0.0, 2.0, 1.0]) == pytest.approx([0.0, 1.0, 0.5])
    assert norm_layer([0.0, 0.0]) == [0.0, 0.0]
    assert norm_layer([]) == []


def test_power_iteration_uniform_on_complete_graph():
    # Fully connected 3-node graph, equal weights → uniform stationary
    edges = [(i, j, 1.0) for i in range(3) for j in range(3) if i != j]
    p = build_row_stochastic(edges, 3, observed=[0, 1, 2], teleport=0.0)
    pi = power_iteration(p, tol=1e-12, max_iter=100)
    assert sum(pi) == pytest.approx(1.0)
    assert pi == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)


def test_power_iteration_teleport_handles_dangling():
    # Node 0 → 1 only; 1 and 2 dangling. Teleport keeps mass circulating.
    edges = [(0, 1, 1.0)]
    p = build_row_stochastic(edges, 3, observed=[0, 1, 2], teleport=0.15)
    pi = power_iteration(p, tol=1e-12, max_iter=200)
    assert sum(pi) == pytest.approx(1.0)
    assert all(x >= 0.0 for x in pi)
    # Node 1 receives the only structural edge → at least as much as a pure sink-ish node 2
    assert pi[1] >= pi[2] - 1e-9


def test_select_keep_k_tie_break_mass_then_id():
    # Equal scores: higher mass wins; then lower id
    scores = [1.0, 1.0, 1.0, 0.5]
    mass = [0.1, 0.5, 0.5, 9.0]
    keep = select_keep_k(scores, mass, k=2)
    # Among score=1.0: experts 1 and 2 have mass 0.5 > 0.1; lower id between 1,2 is 1
    # top-2 by (-score, -mass, id): (1, mass0.5), (2, mass0.5) → ids 1,2
    assert keep == [1, 2]


def test_fusion_balanced_matches_normative_weights():
    # ng=[1,0], nm=[0,1], rest zero → score_0 = 0.30+0.05=0.35, score_1=0.20
    scores = fusion_score_layer(
        pi_g=[1.0, 0.0],
        mass=[0.0, 1.0],
        pi_i=[0.0, 0.0],
        pi_d=[0.0, 0.0],
        pi_s=[0.0, 0.0],
        weights=BALANCED_WEIGHTS,
        safety_stance="balanced",
    )
    assert scores[0] == pytest.approx(0.30 + 0.05)
    assert scores[1] == pytest.approx(0.20)


def test_fusion_crack_penalizes_safety_specificity():
    # Expert 0: high safety, low intent → CRACK penalty
    # Expert 1: high safety, high intent → little penalty
    keep_scores = fusion_score_layer(
        pi_g=[1.0, 1.0],
        mass=[1.0, 1.0],
        pi_i=[0.0, 1.0],
        pi_s=[1.0, 1.0],
        weights=BALANCED_WEIGHTS,
        safety_stance="keep",
    )
    crack_scores = fusion_score_layer(
        pi_g=[1.0, 1.0],
        mass=[1.0, 1.0],
        pi_i=[0.0, 1.0],
        pi_s=[1.0, 1.0],
        weights=BALANCED_WEIGHTS,
        safety_stance="crack",
    )
    # Under CRACK, expert 0 (safety-specific) drops relative to keep stance
    assert crack_scores[0] < keep_scores[0]
    # Expert 1 has ni=1 so specificity=0 → no crack penalty on safety term
    base_1 = (
        BALANCED_WEIGHTS["path"]
        + BALANCED_WEIGHTS["mass"]
        + BALANCED_WEIGHTS["intent"]
        + BALANCED_WEIGHTS["backbone_floor"]
    )
    assert crack_scores[1] == pytest.approx(base_1)


def test_specialist_preset_weights_differ():
    assert SPECIALIST_WEIGHTS["intent"] > BALANCED_WEIGHTS["intent"]
    assert SPECIALIST_WEIGHTS["path"] < BALANCED_WEIGHTS["path"]


# ---------------------------------------------------------------------------
# Appendix C — synthetic highway (REQUIRED)
# ---------------------------------------------------------------------------


def test_synthetic_highway_mass_only_ranks_n_over_h():
    records = build_synthetic_highway_records()
    adj = build_adjacency_from_transitions(
        records, num_experts=HIGHWAY_E, weight_by_gate=True
    )
    mass = mass_matrix_from_adjacency(
        adj, num_layers=HIGHWAY_L, num_experts=HIGHWAY_E
    )
    # Layer 1 is where H is the highway bridge and N dumps mass
    layer = 1
    m = mass[layer]
    assert m[HIGHWAY_N] > m[HIGHWAY_H], (
        f"mass-only precondition failed: N={m[HIGHWAY_N]} H={m[HIGHWAY_H]}"
    )
    mass_scores = mass_only_scores(mass)[layer]
    # Rank by mass score
    order = sorted(range(HIGHWAY_E), key=lambda e: (-mass_scores[e], e))
    assert order.index(HIGHWAY_N) < order.index(HIGHWAY_H)


def test_synthetic_highway_path_and_hybrid_rank_h_over_n():
    records = build_synthetic_highway_records()
    result = score_hybrid(
        records=records,
        num_experts=HIGHWAY_E,
        num_layers=HIGHWAY_L,
        keep_k=2,
        intents_keep=["code"],
        safety_stance="balanced",
        preset="balanced",
    )
    layer = 1
    pi = result.pi_g[layer]
    assert pi[HIGHWAY_H] > pi[HIGHWAY_N], (
        f"path score should prefer highway H: H={pi[HIGHWAY_H]} N={pi[HIGHWAY_N]}"
    )

    hybrid = result.scores[layer]
    assert hybrid[HIGHWAY_H] > hybrid[HIGHWAY_N], (
        f"hybrid should prefer highway H: H={hybrid[HIGHWAY_H]} N={hybrid[HIGHWAY_N]}"
    )

    # Mass-only still prefers N (sanity vs hybrid disagreement)
    mass_scores = mass_only_scores(result.mass)[layer]
    assert mass_scores[HIGHWAY_N] > mass_scores[HIGHWAY_H]


def test_synthetic_highway_keep_k2_retains_h():
    records = build_synthetic_highway_records()
    result = score_hybrid(
        records=records,
        num_experts=HIGHWAY_E,
        num_layers=HIGHWAY_L,
        keep_k=2,
        intents_keep=["code"],
        safety_stance="balanced",
        preset="balanced",
    )
    # H must be retained at the highway layer under keep-K=2
    assert HIGHWAY_H in result.keep[1], (
        f"keep-K=2 must retain highway expert H; keep={result.keep[1]} "
        f"scores={result.scores[1]} mass={result.mass[1]} pi_g={result.pi_g[1]}"
    )
    # Mass-only keep-K=2 at layer 1 would prefer N (and whatever next-highest mass)
    mass_order = sorted(
        range(HIGHWAY_E),
        key=lambda e: (-result.mass[1][e], e),
    )
    mass_keep = sorted(mass_order[:2])
    assert HIGHWAY_N in mass_keep
    # Hybrid and mass-only disagree on whether H is kept (core MAESTRO win)
    if HIGHWAY_H not in mass_keep:
        assert HIGHWAY_H in result.keep[1]


def test_synthetic_highway_plan_schema():
    records = build_synthetic_highway_records()
    result = score_hybrid(
        records=records,
        num_experts=HIGHWAY_E,
        num_layers=HIGHWAY_L,
        keep_k=2,
        intents_keep=["code"],
        safety_stance="balanced",
        preset="balanced",
    )
    plan = build_prune_plan(
        result,
        source_model="/fake/model",
        backend="test",
        suite={"name": "synthetic-highway", "prompt_count": 2},
        trained_top_k=1,
    )
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["schema_version"] == 1
    assert plan["scorer"] == SCORER_NAME
    assert plan["preset"] == "balanced"
    assert plan["weights"]["path"] == pytest.approx(0.30)
    assert plan["weights"]["intent"] == pytest.approx(0.35)
    assert plan["keep_experts_per_layer"] == 2
    assert plan["num_experts_source"] == HIGHWAY_E
    assert plan["num_layers"] == HIGHWAY_L
    assert plan["intents_keep"] == ["code"]
    assert plan["safety_stance"] == "balanced"
    assert plan["safety"]["passed"] is True
    assert "1" in plan["layers"]
    assert HIGHWAY_H in plan["layers"]["1"]
    assert len(plan["layers"]["1"]) == 2


# ---------------------------------------------------------------------------
# Integration: transitions file → plan; CLI
# ---------------------------------------------------------------------------


def test_score_from_adjacency_only(tmp_path: Path):
    records = build_synthetic_highway_records(highway_tokens=4, noise_tokens=20)
    adj = build_adjacency_from_transitions(
        records, num_experts=HIGHWAY_E, weight_by_gate=True
    )
    result = score_hybrid(
        adjacency=adj,
        num_experts=HIGHWAY_E,
        num_layers=HIGHWAY_L,
        keep_k=2,
        safety_stance="keep",
        preset="balanced",
    )
    assert result.num_layers == HIGHWAY_L
    assert HIGHWAY_H in result.keep[1]


def test_write_prune_plan_roundtrip(tmp_path: Path):
    records = build_synthetic_highway_records()
    result = score_hybrid(
        records=records,
        num_experts=HIGHWAY_E,
        num_layers=HIGHWAY_L,
        keep_k=2,
        intents_keep=["code"],
    )
    plan = build_prune_plan(result, source_model="m", backend="b")
    out = tmp_path / "prune_plan.json"
    write_prune_plan(out, plan)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema"] == PLAN_SCHEMA
    assert loaded["layers"]["1"] == plan["layers"]["1"]


def test_cli_intent_prune_score(tmp_path: Path):
    from jang_tools.intent_prune.cli import _cmd_intent_prune_score

    records = build_synthetic_highway_records()
    transitions = tmp_path / "expert_transitions.jsonl"
    write_transitions_jsonl(transitions, records)
    out = tmp_path / "plan.json"

    args = Namespace(
        transitions=str(transitions),
        adjacency="",
        output=str(out),
        num_experts=HIGHWAY_E,
        num_layers=HIGHWAY_L,
        keep_k=2,
        preset="balanced",
        safety_stance="balanced",
        intents_keep=["code"],
        intents_drop=[],
        uniform_weight=False,
        teleport=1e-2,
        tol=1e-10,
        max_iter=200,
        source_model="/src",
        backend="test",
        trained_top_k=1,
        suite_name="Reviewed Prune 50",
        suite_prompt_count=50,
    )
    _cmd_intent_prune_score(args)

    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["scorer"] == SCORER_NAME
    assert HIGHWAY_H in plan["layers"]["1"]
    assert plan["suite"]["name"] == "Reviewed Prune 50"
    assert plan["suite"]["prompt_count"] == 50


def test_intent_conditioned_boosts_keep_domain():
    # Two parallel paths: code-domain through expert 1, chat through expert 2
    records = []
    for i in range(10):
        records.append(
            {
                "schema": "jang-expert-transitions-v1",
                "prompt_id": "code",
                "domains": ["code"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": i,
                "path": [
                    {"layer": 0, "experts": [0], "scores": [1.0]},
                    {"layer": 1, "experts": [1], "scores": [1.0]},
                ],
            }
        )
    for i in range(10):
        records.append(
            {
                "schema": "jang-expert-transitions-v1",
                "prompt_id": "chat",
                "domains": ["chat"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": 100 + i,
                "path": [
                    {"layer": 0, "experts": [0], "scores": [1.0]},
                    {"layer": 1, "experts": [2], "scores": [1.0]},
                ],
            }
        )

    with_intent = score_hybrid(
        records=records,
        num_experts=4,
        num_layers=2,
        keep_k=2,
        intents_keep=["code"],
        safety_stance="balanced",
        preset="balanced",
    )
    # Intent path π_I should concentrate on expert 1 at layer 1
    assert with_intent.pi_i[1][1] > with_intent.pi_i[1][2]
    # Hybrid with code intent should score expert 1 above expert 2 at layer 1
    assert with_intent.scores[1][1] > with_intent.scores[1][2]


def test_node_id_roundtrip():
    assert node_id(2, 5, num_experts=32) == 2 * 32 + 5
    from jang_tools.intent_prune.graph import decode_node

    assert decode_node(2 * 32 + 5, 32) == (2, 5)
