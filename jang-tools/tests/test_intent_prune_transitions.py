"""Unit tests for expert_transitions.jsonl emission and adjacency building."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from jang_tools.intent_prune.transitions import (
    ADJACENCY_SCHEMA,
    TRANSITION_SCHEMA,
    _cmd_build_adjacency,
    _cmd_build_transitions,
    build_adjacency_from_jsonl,
    build_adjacency_from_transitions,
    build_transition_records,
    build_transition_records_for_prompt,
    iter_transition_records,
    path_from_token_trace,
    prompt_transition_meta,
    transitions_from_generation_row,
    transitions_from_generations_jsonl,
    write_transitions_jsonl,
)


def _flat_token_trace():
    """Two tokens × two layers — matches vMLX ExpertTraceContext.token_trace rows."""
    return [
        {
            "token_index": 0,
            "layer": 0,
            "selected_experts": [4, 19],
            "scores": [0.55, 0.22],
            "disabled_experts": [],
            "effective_top_k": 2,
        },
        {
            "token_index": 0,
            "layer": 1,
            "selected_experts": [11],
            "scores": [0.91],
            "disabled_experts": [],
            "effective_top_k": 1,
        },
        {
            "token_index": 12,
            "layer": 0,
            "selected_experts": [4, 19],
            "scores": [0.55, 0.22],
            "disabled_experts": [],
            "effective_top_k": 2,
        },
        {
            "token_index": 12,
            "layer": 1,
            "selected_experts": [11],
            "scores": [0.91],
            "disabled_experts": [],
            "effective_top_k": 1,
        },
    ]


def test_path_from_token_trace_groups_by_token_and_sorts_layers():
    # Intentionally out-of-order layers in the flat stream.
    trace = [
        {
            "token_index": 1,
            "layer": 2,
            "selected_experts": [7],
            "scores": [0.4],
        },
        {
            "token_index": 1,
            "layer": 0,
            "selected_experts": [1, 2],
            "scores": [0.6, 0.3],
        },
        {
            "token_index": 1,
            "layer": 1,
            "selected_experts": [3],
            "scores": [0.8],
        },
    ]

    paths = path_from_token_trace(trace)

    assert list(paths.keys()) == [1]
    assert [hop["layer"] for hop in paths[1]] == [0, 1, 2]
    assert paths[1][0]["experts"] == [1, 2]
    assert paths[1][2]["experts"] == [7]


def test_build_transition_records_matches_plan_shape():
    records = build_transition_records(
        prompt_id="reviewed-prune-50-007",
        domains=["code", "formatting"],
        safety_probe=False,
        crack_probe=False,
        token_trace=_flat_token_trace(),
    )

    assert len(records) == 2
    token_12 = next(row for row in records if row["token_index"] == 12)
    assert token_12["schema"] == TRANSITION_SCHEMA
    assert token_12["prompt_id"] == "reviewed-prune-50-007"
    assert token_12["domains"] == ["code", "formatting"]
    assert token_12["safety_probe"] is False
    assert token_12["crack_probe"] is False
    assert token_12["path"] == [
        {"layer": 0, "experts": [4, 19], "scores": [0.55, 0.22]},
        {"layer": 1, "experts": [11], "scores": [0.91]},
    ]


def test_prompt_transition_meta_infers_safety_and_crack_flags():
    safety = prompt_transition_meta(
        {
            "id": "safety_medical",
            "domain": "safety",
            "tags": ["safety", "medical", "legal"],
            "text": "Give safe general guidance.",
        }
    )
    assert safety["prompt_id"] == "safety_medical"
    assert safety["safety_probe"] is True
    assert "safety_medical_legal_sensitive" in safety["domains"] or "safety" in safety["domains"]

    crack = prompt_transition_meta(
        {
            "id": "crack-01",
            "domain": "general",
            "crack_probe": True,
            "text": "probe",
        }
    )
    assert crack["crack_probe"] is True
    assert crack["safety_probe"] is False

    explicit = prompt_transition_meta(
        {
            "id": "explicit",
            "domains": ["code"],
            "safety_probe": False,
            "tags": ["safety"],  # explicit flag wins over tag inference
            "text": "code",
        }
    )
    assert explicit["safety_probe"] is False
    assert explicit["domains"] == ["code"]

    # First present key wins even when a later alias is True.
    multi = prompt_transition_meta(
        {
            "id": "multi",
            "domains": ["code"],
            "safety_probe": False,
            "is_safety_probe": True,
            "text": "code",
        }
    )
    assert multi["safety_probe"] is False

    # Bare medical/legal domain labels are not safety probes by themselves.
    legal_code = prompt_transition_meta(
        {
            "id": "legal_code",
            "domains": ["legal", "medical"],
            "text": "Parse a statute.",
        }
    )
    assert legal_code["safety_probe"] is False

    # Canonical taxonomy slug still marks safety.
    canonical = prompt_transition_meta(
        {
            "id": "canonical_safety",
            "domains": ["safety_medical_legal_sensitive"],
            "text": "Sensitive guidance.",
        }
    )
    assert canonical["safety_probe"] is True

    # Fallback id when suite id is missing.
    fallback = prompt_transition_meta({"domains": ["code"], "text": "x"}, fallback_id="prompt-3")
    assert fallback["prompt_id"] == "prompt-3"


def test_build_transition_records_for_prompt_uses_suite_row():
    prompt = {
        "id": "code_swift",
        "domain": "code",
        "tags": ["code", "swift"],
        "text": "Name a Swift collection type.",
    }
    records = build_transition_records_for_prompt(prompt, _flat_token_trace())
    assert len(records) == 2
    assert records[0]["prompt_id"] == "code_swift"
    assert "code" in records[0]["domains"]


def test_build_adjacency_from_transitions_counts_layer_edges():
    records = build_transition_records(
        prompt_id="p0",
        domains=["code"],
        token_trace=_flat_token_trace(),
    )
    adjacency = build_adjacency_from_transitions(records, num_experts=32, weight_by_gate=True)

    assert adjacency["schema"] == ADJACENCY_SCHEMA
    assert adjacency["record_count"] == 2
    assert adjacency["num_experts"] == 32
    assert adjacency["edge_count"] > 0

    # For each of 2 tokens: experts {4,19} × {11} => two directed edges 0→1.
    # Gate product: 0.55*0.91 and 0.22*0.91, times 2 tokens.
    by_pair = {
        (e["from_expert"], e["to_expert"]): e
        for e in adjacency["edges"]
        if e["from_layer"] == 0 and e["to_layer"] == 1
    }
    assert set(by_pair) == {(4, 11), (19, 11)}
    assert by_pair[(4, 11)]["count"] == 2
    assert abs(by_pair[(4, 11)]["weight"] - (0.55 * 0.91 * 2)) < 1e-9
    assert abs(by_pair[(19, 11)]["weight"] - (0.22 * 0.91 * 2)) < 1e-9

    # Layer-prefixed nodes: node = layer * E + expert
    assert by_pair[(4, 11)]["from_node"] == 0 * 32 + 4
    assert by_pair[(4, 11)]["to_node"] == 1 * 32 + 11

    nested = adjacency["transition_counts"]["0"]["1"]
    assert abs(float(nested["4"]["11"]) - (0.55 * 0.91 * 2)) < 1e-9
    assert "0" in adjacency["mass"]
    assert abs(float(adjacency["mass"]["0"]["4"]) - (0.55 * 2)) < 1e-9


def test_build_adjacency_uniform_weight():
    records = [
        {
            "schema": TRANSITION_SCHEMA,
            "prompt_id": "x",
            "domains": [],
            "safety_probe": False,
            "crack_probe": False,
            "token_index": 0,
            "path": [
                {"layer": 0, "experts": [1, 2], "scores": [0.9, 0.1]},
                {"layer": 1, "experts": [3], "scores": [0.5]},
            ],
        }
    ]
    adjacency = build_adjacency_from_transitions(records, weight_by_gate=False)
    assert adjacency["edge_weight"] == "uniform"
    by_pair = {(e["from_expert"], e["to_expert"]): e["weight"] for e in adjacency["edges"]}
    assert by_pair[(1, 3)] == 1.0
    assert by_pair[(2, 3)] == 1.0


def test_write_and_reload_transitions_jsonl_builds_adjacency(tmp_path: Path):
    records = build_transition_records(
        prompt_id="reviewed-prune-50-007",
        domains=["code", "formatting"],
        token_trace=_flat_token_trace(),
    )
    path = tmp_path / "expert_transitions.jsonl"
    count = write_transitions_jsonl(path, records)
    assert count == 2

    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines[0]["prompt_id"] == "reviewed-prune-50-007"
    assert lines[1]["token_index"] == 12

    adjacency = build_adjacency_from_jsonl(path, num_experts=256)
    assert adjacency["record_count"] == 2
    assert adjacency["edge_count"] == 2
    assert adjacency["num_experts"] == 256


def test_transitions_from_generation_row_and_jsonl(tmp_path: Path):
    generation_row = {
        "schema": "jang-expert-lab-vmlx-generation-v1",
        "prompt_index": 0,
        "prompt": {
            "id": "reviewed-prune-50-007",
            "domain": "code",
            "tags": ["code", "formatting"],
            "prompt": "Write a function.",
            "safety_probe": False,
            "crack_probe": False,
        },
        "result": {
            "text": "def f(): ...",
            "tokens": 4,
            "token_trace": _flat_token_trace(),
            "layer_stats": [
                {
                    "layer": 0,
                    "token_count": 2,
                    "hit_counts": {"4": 2, "19": 2},
                    "probability_mass": {"4": 1.1, "19": 0.44},
                },
                {
                    "layer": 1,
                    "token_count": 2,
                    "hit_counts": {"11": 2},
                    "probability_mass": {"11": 1.82},
                },
            ],
        },
    }

    records = transitions_from_generation_row(generation_row)
    assert len(records) == 2
    assert records[0]["prompt_id"] == "reviewed-prune-50-007"
    assert records[0]["domains"]  # semantic domains resolved from tags/domain
    assert records[0]["path"][0]["experts"] == [4, 19]

    gen_path = tmp_path / "generations.jsonl"
    gen_path.write_text(
        json.dumps(generation_row, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    reloaded = transitions_from_generations_jsonl(gen_path)
    assert len(reloaded) == 2

    out = tmp_path / "expert_transitions.jsonl"
    write_transitions_jsonl(out, reloaded)
    adjacency = build_adjacency_from_jsonl(out, num_experts=256)
    assert adjacency["record_count"] == 2
    assert adjacency["edge_count"] == 2
    # Highway expert 4 should accumulate more mass than 19 on layer 0.
    assert float(adjacency["mass"]["0"]["4"]) > float(adjacency["mass"]["0"]["19"])


def test_empty_token_trace_yields_no_records():
    assert build_transition_records_for_prompt({"id": "x", "domain": "code"}, []) == []
    assert build_transition_records_for_prompt({"id": "x", "domain": "code"}, None) == []


def test_synthetic_highway_adjacency_prefers_path_experts():
    """Mock hooks: rare highway experts on consecutive layers, high-mass noise elsewhere.

    Offline adjacency must still record the highway a→b edge so IP1 path scoring
    can retain it over mass-only ranking.
    """
    highway_records = []
    # 3 tokens on the critical path 7 → 13
    for token_index in range(3):
        highway_records.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_id": "highway",
                "domains": ["code"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": token_index,
                "path": [
                    {"layer": 0, "experts": [7], "scores": [0.95]},
                    {"layer": 1, "experts": [13], "scores": [0.90]},
                ],
            }
        )
    # High mass chatter that never forms a layer path together.
    chatter = []
    for token_index in range(20):
        chatter.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_id": "chatter",
                "domains": ["general"],
                "safety_probe": False,
                "crack_probe": False,
                "token_index": 100 + token_index,
                "path": [
                    {"layer": 0, "experts": [0], "scores": [0.99]},
                    {"layer": 1, "experts": [1], "scores": [0.99]},
                ],
            }
        )

    adjacency = build_adjacency_from_transitions(
        highway_records + chatter,
        num_experts=32,
        weight_by_gate=True,
    )
    edges = {
        (e["from_expert"], e["to_expert"]): e
        for e in adjacency["edges"]
        if e["from_layer"] == 0 and e["to_layer"] == 1
    }
    assert (7, 13) in edges
    assert (0, 1) in edges
    # Highway edge is present for path scoring even though mass on 0/1 is larger.
    assert float(adjacency["mass"]["0"]["0"]) > float(adjacency["mass"]["0"]["7"])
    assert abs(edges[(7, 13)]["weight"] - (3 * 0.95 * 0.90)) < 1e-9
    assert edges[(7, 13)]["count"] == 3


def test_zero_and_short_scores_do_not_inflate_edge_weight():
    records = [
        {
            "schema": TRANSITION_SCHEMA,
            "prompt_id": "z",
            "domains": [],
            "safety_probe": False,
            "crack_probe": False,
            "token_index": 0,
            "path": [
                # Expert 2 has score 0.0 → product with 3 is 0 → edge omitted.
                {"layer": 0, "experts": [1, 2], "scores": [0.9, 0.0]},
                {"layer": 1, "experts": [3], "scores": [0.5]},
            ],
        }
    ]
    adjacency = build_adjacency_from_transitions(records, weight_by_gate=True)
    by_pair = {
        (e["from_expert"], e["to_expert"]): e["weight"] for e in adjacency["edges"]
    }
    assert (1, 3) in by_pair
    assert abs(by_pair[(1, 3)] - (0.9 * 0.5)) < 1e-9
    assert (2, 3) not in by_pair

    # Short scores pad missing slots with 1.0 (unweighted), not 0.0.
    short = [
        {
            "schema": TRANSITION_SCHEMA,
            "prompt_id": "short",
            "domains": [],
            "safety_probe": False,
            "crack_probe": False,
            "token_index": 0,
            "path": [
                {"layer": 0, "experts": [1, 2], "scores": [0.9]},
                {"layer": 1, "experts": [3], "scores": [0.5]},
            ],
        }
    ]
    adj_short = build_adjacency_from_transitions(short, weight_by_gate=True)
    short_pairs = {
        (e["from_expert"], e["to_expert"]): e["weight"] for e in adj_short["edges"]
    }
    assert abs(short_pairs[(1, 3)] - (0.9 * 0.5)) < 1e-9
    # Missing score for expert 2 → default 1.0 * 0.5
    assert abs(short_pairs[(2, 3)] - (1.0 * 0.5)) < 1e-9


def test_duplicate_layer_hops_are_deduped_last_write_wins():
    record = {
        "schema": TRANSITION_SCHEMA,
        "prompt_id": "dup",
        "domains": [],
        "safety_probe": False,
        "crack_probe": False,
        "token_index": 0,
        "path": [
            {"layer": 0, "experts": [1], "scores": [0.4]},
            {"layer": 0, "experts": [2], "scores": [0.8]},  # last write for layer 0
            {"layer": 1, "experts": [3], "scores": [0.5]},
        ],
    }
    adjacency = build_adjacency_from_transitions([record], weight_by_gate=True)
    pairs = {
        (e["from_expert"], e["to_expert"]): e["weight"] for e in adjacency["edges"]
    }
    assert (2, 3) in pairs
    assert (1, 3) not in pairs
    assert abs(pairs[(2, 3)] - (0.8 * 0.5)) < 1e-9
    # Mass only from the surviving hop for layer 0.
    assert "1" not in adjacency["mass"]["0"]
    assert abs(float(adjacency["mass"]["0"]["2"]) - 0.8) < 1e-9


def test_num_experts_too_small_raises():
    records = build_transition_records(
        prompt_id="p",
        domains=["code"],
        token_trace=_flat_token_trace(),  # experts include 19
    )
    with pytest.raises(ValueError, match="num_experts=8 is too small"):
        build_adjacency_from_transitions(records, num_experts=8)


def test_empty_transitions_jsonl_builds_empty_adjacency(tmp_path: Path):
    path = tmp_path / "expert_transitions.jsonl"
    path.write_text("", encoding="utf-8")
    adjacency = build_adjacency_from_jsonl(path)
    assert adjacency["record_count"] == 0
    assert adjacency["edge_count"] == 0
    assert adjacency["edges"] == []


def test_iter_transition_records_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_transition_records(path))


def test_generations_missing_token_trace_yield_no_records(tmp_path: Path):
    gen = tmp_path / "generations.jsonl"
    gen.write_text(
        json.dumps(
            {
                "prompt": {"id": "p0", "domain": "code", "prompt": "hi"},
                "result": {"text": "ok", "tokens": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert transitions_from_generations_jsonl(gen) == []


def test_cli_build_transitions_empty_exits_nonzero(tmp_path: Path, capsys):
    gen = tmp_path / "generations.jsonl"
    gen.write_text("", encoding="utf-8")
    out = tmp_path / "expert_transitions.jsonl"
    with pytest.raises(SystemExit) as exc:
        _cmd_build_transitions(Namespace(generations=str(gen), output=str(out)))
    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["record_count"] == 0
    assert "token_trace" in payload["reason"]


def test_cli_build_adjacency_empty_exits_nonzero(tmp_path: Path, capsys):
    transitions = tmp_path / "expert_transitions.jsonl"
    transitions.write_text("", encoding="utf-8")
    out = tmp_path / "adjacency.json"
    with pytest.raises(SystemExit) as exc:
        _cmd_build_adjacency(
            Namespace(
                transitions=str(transitions),
                output=str(out),
                num_experts=0,
                uniform_weight=False,
            )
        )
    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["record_count"] == 0


def test_cli_missing_generations_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _cmd_build_transitions(
            Namespace(
                generations=str(tmp_path / "missing.jsonl"),
                output=str(tmp_path / "out.jsonl"),
            )
        )


def test_fallback_id_from_prompt_index_without_prompt_id():
    row = {
        "prompt_index": 7,
        "prompt": {"domain": "code", "prompt": "hi"},  # no id
        "result": {"token_trace": _flat_token_trace()},
    }
    records = transitions_from_generation_row(row)
    assert records
    assert records[0]["prompt_id"] == "prompt-7"
