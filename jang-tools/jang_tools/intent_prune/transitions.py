"""Expert layer-path transitions for MAESTRO-class Intent Prune scoring.

Schema: ``expert_transitions.jsonl`` (one JSON object per line)

Each record is one token's ordered MoE routing path across layers:

.. code-block:: json

    {
      "schema": "jang-expert-transitions-v1",
      "prompt_id": "reviewed-prune-50-007",
      "domains": ["code", "formatting"],
      "safety_probe": false,
      "crack_probe": false,
      "token_index": 12,
      "path": [
        {"layer": 0, "experts": [4, 19], "scores": [0.55, 0.22]},
        {"layer": 1, "experts": [11], "scores": [0.91]}
      ]
    }

Field notes
-----------
* ``prompt_id`` — suite prompt id (stable across baseline/masked runs).
* ``domains`` — semantic domains from the suite row (canonical slugs).
* ``safety_probe`` / ``crack_probe`` — explicit suite flags or inferred from
  domains/tags; used later for safety-conditional path scores.
* ``token_index`` — generation-relative token position from vMLX hooks.
* ``path`` — hops sorted by ascending MoE layer for that token. ``experts``
  and ``scores`` are the selected top-k at that layer (same order).

Adjacency (offline)
-------------------
``build_adjacency_from_transitions`` counts layer→layer expert edges for
path scoring. With ``num_experts=E``, nodes are layer-prefixed:
``node = layer * E + expert``. Edge weight defaults to the product of the
two gate scores (or 1.0 when scores are missing).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

TRANSITION_SCHEMA = "jang-expert-transitions-v1"
ADJACENCY_SCHEMA = "jang-expert-adjacency-v1"
EXPERT_TRANSITIONS_FILENAME = "expert_transitions.jsonl"


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(raw)


def _as_int_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _as_float_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _domains_from_prompt(prompt: dict[str, Any]) -> list[str]:
    """Resolve semantic domains for a suite prompt row."""
    for key in ("domains", "semantic_domains", "semanticDomains"):
        if key in prompt:
            domains = _string_list(prompt.get(key))
            if domains:
                return sorted(set(domains))

    try:
        from jang_tools.prequant_prune_qwen_moe import _suite_semantic_domains

        semantic = sorted(_suite_semantic_domains(prompt))
        if semantic:
            return semantic
    except Exception:
        pass

    domain = prompt.get("domain")
    if isinstance(domain, str) and domain.strip():
        return [domain.strip()]
    tags = _string_list(prompt.get("tags"))
    return sorted(set(tags))


def _infer_safety_probe(prompt: dict[str, Any], domains: Sequence[str]) -> bool:
    if any(
        key in prompt
        for key in ("safety_probe", "safetyProbe", "is_safety_probe", "isSafetyProbe")
    ):
        return _coerce_bool(
            prompt.get("safety_probe")
            or prompt.get("safetyProbe")
            or prompt.get("is_safety_probe")
            or prompt.get("isSafetyProbe")
        )
    safety_markers = {
        "safety",
        "safety_medical_legal_sensitive",
        "safety-medical-legal-sensitive",
        "medical",
        "legal",
        "sensitive",
    }
    lowered = {str(d).strip().lower().replace(" ", "_") for d in domains}
    tags = {t.strip().lower().replace(" ", "_") for t in _string_list(prompt.get("tags"))}
    return bool(lowered.intersection(safety_markers) or tags.intersection(safety_markers))


def _infer_crack_probe(prompt: dict[str, Any], domains: Sequence[str]) -> bool:
    if any(
        key in prompt
        for key in ("crack_probe", "crackProbe", "is_crack_probe", "isCrackProbe")
    ):
        return _coerce_bool(
            prompt.get("crack_probe")
            or prompt.get("crackProbe")
            or prompt.get("is_crack_probe")
            or prompt.get("isCrackProbe")
        )
    crack_markers = {"crack", "abliteration", "refusal", "jailbreak_probe"}
    lowered = {str(d).strip().lower().replace(" ", "_") for d in domains}
    tags = {t.strip().lower().replace(" ", "_") for t in _string_list(prompt.get("tags"))}
    return bool(lowered.intersection(crack_markers) or tags.intersection(crack_markers))


def prompt_transition_meta(prompt: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt_id / domains / probe flags for transition records."""
    prompt_id = (
        str(prompt.get("id") or prompt.get("prompt_id") or prompt.get("promptID") or "").strip()
    )
    if not prompt_id:
        raise ValueError("prompt is missing a non-empty id / prompt_id")
    domains = _domains_from_prompt(prompt)
    return {
        "prompt_id": prompt_id,
        "domains": domains,
        "safety_probe": _infer_safety_probe(prompt, domains),
        "crack_probe": _infer_crack_probe(prompt, domains),
    }


def _hop_from_trace_row(row: dict[str, Any]) -> dict[str, Any] | None:
    experts = _as_int_list(
        row.get("experts")
        or row.get("selected_experts")
        or row.get("selectedExperts")
        or row.get("indices")
    )
    if not experts:
        return None
    scores = _as_float_list(row.get("scores") or row.get("selected_scores") or row.get("weights"))
    if scores and len(scores) != len(experts):
        # Keep alignment: truncate/pad rather than drop the hop.
        if len(scores) > len(experts):
            scores = scores[: len(experts)]
        else:
            scores = scores + [0.0] * (len(experts) - len(scores))
    layer_raw = row.get("layer")
    if layer_raw is None:
        return None
    try:
        layer = int(layer_raw)
    except (TypeError, ValueError):
        return None
    hop: dict[str, Any] = {
        "layer": layer,
        "experts": experts,
    }
    if scores:
        hop["scores"] = scores
    return hop


def path_from_token_trace(token_trace: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group flat per-layer token_trace rows into ordered paths by token_index.

    Returns ``{token_index: [hop, ...]}`` with hops sorted by layer.
    """
    by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(token_trace):
        if not isinstance(row, dict):
            raise ValueError(f"token_trace[{index}] must be an object")
        token_raw = row.get("token_index", row.get("tokenIndex", row.get("position")))
        if token_raw is None:
            continue
        try:
            token_index = int(token_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token_trace[{index}] has invalid token_index {token_raw!r}") from exc
        hop = _hop_from_trace_row(row)
        if hop is None:
            continue
        by_token[token_index].append(hop)

    paths: dict[int, list[dict[str, Any]]] = {}
    for token_index, hops in by_token.items():
        # One hop per layer (last write wins if hooks double-fire).
        by_layer: dict[int, dict[str, Any]] = {}
        for hop in hops:
            by_layer[int(hop["layer"])] = hop
        ordered = [by_layer[layer] for layer in sorted(by_layer)]
        if ordered:
            paths[int(token_index)] = ordered
    return paths


def build_transition_records(
    *,
    prompt_id: str,
    domains: Sequence[str] | None = None,
    safety_probe: bool = False,
    crack_probe: bool = False,
    token_trace: Sequence[dict[str, Any]] | None = None,
    paths: dict[int, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Build ``expert_transitions.jsonl`` records for one prompt."""
    if paths is None:
        if token_trace is None:
            raise ValueError("build_transition_records requires token_trace or paths")
        paths = path_from_token_trace(token_trace)

    domain_list = sorted({str(d).strip() for d in (domains or []) if str(d).strip()})
    records: list[dict[str, Any]] = []
    for token_index in sorted(paths):
        path = paths[token_index]
        if not path:
            continue
        records.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_id": str(prompt_id),
                "domains": domain_list,
                "safety_probe": bool(safety_probe),
                "crack_probe": bool(crack_probe),
                "token_index": int(token_index),
                "path": path,
            }
        )
    return records


def build_transition_records_for_prompt(
    prompt: dict[str, Any],
    token_trace: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convenience: suite prompt row + vMLX token_trace → transition records."""
    if not token_trace:
        return []
    meta = prompt_transition_meta(prompt)
    return build_transition_records(
        prompt_id=meta["prompt_id"],
        domains=meta["domains"],
        safety_probe=meta["safety_probe"],
        crack_probe=meta["crack_probe"],
        token_trace=token_trace,
    )


def write_transitions_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write transition records as JSONL. Returns the number of lines written."""
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_transition_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield transition records from a JSONL file."""
    for line_number, line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: transition record must be a JSON object")
        yield row


def transitions_from_generation_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Build transitions from one ``generations.jsonl`` row (vMLX schema)."""
    if not isinstance(row, dict):
        raise ValueError("generation row must be a JSON object")
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        # Allow flattened rows that already carry prompt_id.
        prompt = {
            "id": row.get("prompt_id") or row.get("promptID") or row.get("id"),
            "domain": row.get("domain"),
            "domains": row.get("domains") or row.get("semantic_domains") or row.get("semanticDomains"),
            "tags": row.get("tags"),
            "safety_probe": row.get("safety_probe", row.get("safetyProbe")),
            "crack_probe": row.get("crack_probe", row.get("crackProbe")),
        }
    result = row.get("result") if isinstance(row.get("result"), dict) else row
    token_trace = result.get("token_trace") if isinstance(result, dict) else None
    if not isinstance(token_trace, list):
        return []
    return build_transition_records_for_prompt(prompt, token_trace)


def transitions_from_generations_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Rebuild expert transitions from a vMLX ``generations.jsonl`` artifact."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        try:
            records.extend(transitions_from_generation_row(row))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def _path_hops(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = record.get("path")
    if not isinstance(path, list):
        return []
    hops: list[dict[str, Any]] = []
    for entry in path:
        if not isinstance(entry, dict):
            continue
        hop = _hop_from_trace_row(entry)
        if hop is not None:
            hops.append(hop)
    hops.sort(key=lambda item: int(item["layer"]))
    return hops


def _edge_weight(
    experts_a: Sequence[int],
    scores_a: Sequence[float],
    experts_b: Sequence[int],
    scores_b: Sequence[float],
    *,
    weight_by_gate: bool,
) -> Iterator[tuple[int, int, float]]:
    for i, expert_a in enumerate(experts_a):
        score_a = float(scores_a[i]) if i < len(scores_a) else 1.0
        for j, expert_b in enumerate(experts_b):
            score_b = float(scores_b[j]) if j < len(scores_b) else 1.0
            if weight_by_gate:
                weight = max(score_a, 0.0) * max(score_b, 0.0)
                if weight <= 0.0:
                    weight = 1.0
            else:
                weight = 1.0
            yield int(expert_a), int(expert_b), float(weight)


def build_adjacency_from_transitions(
    records: Iterable[dict[str, Any]],
    *,
    num_experts: int | None = None,
    weight_by_gate: bool = True,
) -> dict[str, Any]:
    """Build layer→layer adjacency / transition counts from transition records.

    Returns a serializable adjacency document for IP1 path scoring:

    * ``edges`` — list of ``{from_layer, from_expert, to_layer, to_expert, weight, count}``
    * ``transition_counts`` — nested map ``from_layer → to_layer → from_expert → to_expert → weight``
    * ``nodes`` — optional layer-prefixed node ids when ``num_experts`` is set
    * ``mass`` — per-layer expert selection mass (gate sum or hit count)
    """
    # transition_counts[from_layer][to_layer][from_expert][to_expert] = weight
    transition_counts: dict[int, dict[int, dict[int, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    edge_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    mass: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    record_count = 0
    hop_count = 0
    edge_event_count = 0
    max_expert_seen = -1
    max_layer_seen = -1

    for record in records:
        if not isinstance(record, dict):
            continue
        hops = _path_hops(record)
        if not hops:
            continue
        record_count += 1
        for hop in hops:
            hop_count += 1
            layer = int(hop["layer"])
            max_layer_seen = max(max_layer_seen, layer)
            experts = hop["experts"]
            scores = hop.get("scores") or []
            for idx, expert in enumerate(experts):
                max_expert_seen = max(max_expert_seen, int(expert))
                score = float(scores[idx]) if idx < len(scores) else 1.0
                mass[layer][int(expert)] += max(score, 0.0) if weight_by_gate else 1.0

        for left, right in zip(hops, hops[1:]):
            from_layer = int(left["layer"])
            to_layer = int(right["layer"])
            # Skip non-forward edges (should not happen with sorted hops).
            if to_layer <= from_layer:
                continue
            for expert_a, expert_b, weight in _edge_weight(
                left["experts"],
                left.get("scores") or [],
                right["experts"],
                right.get("scores") or [],
                weight_by_gate=weight_by_gate,
            ):
                transition_counts[from_layer][to_layer][expert_a][expert_b] += weight
                edge_counts[(from_layer, expert_a, to_layer, expert_b)] += 1
                edge_event_count += 1
                max_expert_seen = max(max_expert_seen, expert_a, expert_b)

    inferred_experts = max_expert_seen + 1 if max_expert_seen >= 0 else 0
    expert_width = int(num_experts) if num_experts and num_experts > 0 else inferred_experts

    edges: list[dict[str, Any]] = []
    for (from_layer, from_expert, to_layer, to_expert), count in sorted(edge_counts.items()):
        weight = float(transition_counts[from_layer][to_layer][from_expert][to_expert])
        edge: dict[str, Any] = {
            "from_layer": from_layer,
            "from_expert": from_expert,
            "to_layer": to_layer,
            "to_expert": to_expert,
            "weight": weight,
            "count": int(count),
        }
        if expert_width > 0:
            edge["from_node"] = from_layer * expert_width + from_expert
            edge["to_node"] = to_layer * expert_width + to_expert
        edges.append(edge)

    nested: dict[str, Any] = {}
    for from_layer in sorted(transition_counts):
        nested[str(from_layer)] = {}
        for to_layer in sorted(transition_counts[from_layer]):
            nested[str(from_layer)][str(to_layer)] = {
                str(from_expert): {
                    str(to_expert): float(weight)
                    for to_expert, weight in sorted(
                        transition_counts[from_layer][to_layer][from_expert].items()
                    )
                }
                for from_expert in sorted(transition_counts[from_layer][to_layer])
            }

    mass_out = {
        str(layer): {
            str(expert): float(value) for expert, value in sorted(experts.items())
        }
        for layer, experts in sorted(mass.items())
    }

    nodes: list[dict[str, Any]] = []
    if expert_width > 0 and max_layer_seen >= 0:
        observed: set[tuple[int, int]] = set()
        for layer, experts in mass.items():
            for expert in experts:
                observed.add((int(layer), int(expert)))
        for from_layer, from_expert, to_layer, to_expert in edge_counts:
            observed.add((from_layer, from_expert))
            observed.add((to_layer, to_expert))
        for layer, expert in sorted(observed):
            nodes.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "node": layer * expert_width + expert,
                }
            )

    return {
        "schema": ADJACENCY_SCHEMA,
        "schema_version": 1,
        "num_experts": expert_width if expert_width > 0 else None,
        "num_layers_observed": max_layer_seen + 1 if max_layer_seen >= 0 else 0,
        "edge_weight": "gate_product" if weight_by_gate else "uniform",
        "record_count": record_count,
        "hop_count": hop_count,
        "edge_event_count": edge_event_count,
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "transition_counts": nested,
        "mass": mass_out,
    }


def build_adjacency_from_jsonl(
    path: str | Path,
    *,
    num_experts: int | None = None,
    weight_by_gate: bool = True,
) -> dict[str, Any]:
    """Load ``expert_transitions.jsonl`` and build adjacency."""
    return build_adjacency_from_transitions(
        iter_transition_records(path),
        num_experts=num_experts,
        weight_by_gate=weight_by_gate,
    )


def write_adjacency_json(path: str | Path, adjacency: dict[str, Any]) -> None:
    Path(path).expanduser().write_text(
        json.dumps(adjacency, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def register(subparsers) -> None:
    """Register offline Intent Prune transition / adjacency CLI commands."""
    from_gen = subparsers.add_parser(
        "intent-prune-transitions",
        help="Build expert_transitions.jsonl from vMLX generations.jsonl",
    )
    from_gen.add_argument(
        "--generations",
        required=True,
        help="Path to generations.jsonl (or directory containing it)",
    )
    from_gen.add_argument(
        "--output",
        required=True,
        help="Output expert_transitions.jsonl path",
    )
    from_gen.set_defaults(func=_cmd_build_transitions, json=True)

    adj = subparsers.add_parser(
        "intent-prune-adjacency",
        help="Build layer→layer adjacency from expert_transitions.jsonl",
    )
    adj.add_argument(
        "--transitions",
        required=True,
        help="Path to expert_transitions.jsonl",
    )
    adj.add_argument(
        "--output",
        required=True,
        help="Output adjacency JSON path",
    )
    adj.add_argument(
        "--num-experts",
        type=int,
        default=0,
        help="Expert width E for layer-prefixed node ids (0 = infer from data)",
    )
    adj.add_argument(
        "--uniform-weight",
        action="store_true",
        help="Count each a→b co-selection as 1.0 instead of gate product",
    )
    adj.set_defaults(func=_cmd_build_adjacency, json=True)


def _generations_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_dir():
        candidate = path / "generations.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(f"generations.jsonl not found under {path}")
        return candidate
    return path


def _cmd_build_transitions(args) -> None:
    generations = _generations_path(args.generations)
    records = transitions_from_generations_jsonl(generations)
    count = write_transitions_jsonl(args.output, records)
    summary = {
        "ok": True,
        "schema": TRANSITION_SCHEMA,
        "generations": str(generations),
        "output": str(Path(args.output).expanduser()),
        "record_count": count,
    }
    print(json.dumps(summary, sort_keys=True))


def _cmd_build_adjacency(args) -> None:
    num_experts = int(args.num_experts) if args.num_experts and args.num_experts > 0 else None
    adjacency = build_adjacency_from_jsonl(
        args.transitions,
        num_experts=num_experts,
        weight_by_gate=not bool(args.uniform_weight),
    )
    write_adjacency_json(args.output, adjacency)
    summary = {
        "ok": True,
        "schema": ADJACENCY_SCHEMA,
        "transitions": str(Path(args.transitions).expanduser()),
        "output": str(Path(args.output).expanduser()),
        "record_count": adjacency["record_count"],
        "edge_count": adjacency["edge_count"],
        "num_experts": adjacency["num_experts"],
    }
    print(json.dumps(summary, sort_keys=True))
