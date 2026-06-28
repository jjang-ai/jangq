"""Build a Nex/N2 expert prune map from router trace JSON.

The affine converter already accepts ``--expert-prune-map``. This helper turns
runtime router traces into that format so pruning can be activation-guided
instead of using router-row L2 as a proxy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            out.extend(_as_int_list(item))
        return out
    return []


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_as_float_list(item))
        return out
    return []


def _records_from_layer(layer_data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(layer_data, list):
        for entry in layer_data:
            if isinstance(entry, dict):
                yield entry
        return
    if not isinstance(layer_data, dict):
        return
    for key in ("events", "records", "routes", "tokens"):
        entries = layer_data.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    yield entry
            return
    if "indices" in layer_data or "selected_experts" in layer_data:
        yield layer_data


def _iter_trace_records(data: dict[str, Any]) -> Iterable[tuple[int, list[int], list[float]]]:
    layers = data.get("layers")
    if isinstance(layers, dict):
        for layer_key, layer_data in layers.items():
            layer = int(layer_key)
            for record in _records_from_layer(layer_data):
                indices = _as_int_list(
                    record.get("indices")
                    or record.get("expert_indices")
                    or record.get("selected_experts")
                    or record.get("experts")
                )
                scores = _as_float_list(
                    record.get("scores")
                    or record.get("weights")
                    or record.get("selected_scores")
                    or record.get("probabilities")
                )
                if indices:
                    yield layer, indices, scores
    records = data.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict) or "layer" not in record:
                continue
            indices = _as_int_list(
                record.get("indices")
                or record.get("expert_indices")
                or record.get("selected_experts")
                or record.get("experts")
            )
            scores = _as_float_list(
                record.get("scores")
                or record.get("weights")
                or record.get("selected_scores")
                or record.get("probabilities")
            )
            if indices:
                yield int(record["layer"]), indices, scores


def build_prune_map(
    trace: dict[str, Any],
    *,
    keep_experts: int,
    num_experts: int = 512,
) -> dict[str, Any]:
    hit_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    prob_mass: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    token_counts: dict[int, int] = defaultdict(int)

    for layer, indices, scores in _iter_trace_records(trace):
        token_counts[layer] += 1
        for slot, expert in enumerate(indices):
            if expert < 0 or expert >= num_experts:
                raise ValueError(f"layer {layer}: expert id {expert} outside 0..{num_experts - 1}")
            hit_counts[layer][expert] += 1
            prob_mass[layer][expert] += scores[slot] if slot < len(scores) else 1.0

    if not hit_counts:
        raise ValueError("trace did not contain any router records")

    layers: dict[str, Any] = {}
    for layer in sorted(hit_counts):
        hit_rank = sorted(
            range(num_experts),
            key=lambda expert: (-hit_counts[layer].get(expert, 0), -prob_mass[layer].get(expert, 0.0), expert),
        )
        prob_rank = sorted(
            range(num_experts),
            key=lambda expert: (-prob_mass[layer].get(expert, 0.0), -hit_counts[layer].get(expert, 0), expert),
        )
        layers[str(layer)] = {
            "token_count": token_counts[layer],
            "experts": prob_rank[:keep_experts],
            "keep_by_probability_coverage": {
                str(keep_experts): {
                    "experts": prob_rank[:keep_experts],
                    "observed_experts": sum(1 for expert in range(num_experts) if prob_mass[layer].get(expert, 0.0) > 0),
                }
            },
            "keep_by_hit_coverage": {
                str(keep_experts): {
                    "experts": hit_rank[:keep_experts],
                    "observed_experts": sum(1 for expert in range(num_experts) if hit_counts[layer].get(expert, 0) > 0),
                }
            },
        }

    return {
        "schema": "nex-n2-router-trace-prune-map-v1",
        "source": trace.get("artifact") or trace.get("source") or "router_trace",
        "num_experts": num_experts,
        "keep_experts": keep_experts,
        "selection": "router_probability_mass_then_hit_count",
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--keep-experts", type=int, default=435)
    parser.add_argument("--num-experts", type=int, default=512)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    prune_map = build_prune_map(
        trace,
        keep_experts=args.keep_experts,
        num_experts=args.num_experts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prune_map, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
