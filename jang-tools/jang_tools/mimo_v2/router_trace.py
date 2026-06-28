"""Router trace aggregation for MiMo V2.x source-side profiling."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"expected a 2D list-like value, got {type(value).__name__}")
    if value and not isinstance(value[0], list):
        return [value]
    return value


class RouteAccumulator:
    """Collect per-token top-k expert choices and summarize hot experts."""

    def __init__(self, *, num_experts: int) -> None:
        self.num_experts = int(num_experts)
        self.records: list[dict[str, Any]] = []
        self._hit_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._prob_mass: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._token_counts: dict[int, int] = defaultdict(int)

    def add(self, *, layer: int, indices: Any, weights: Any) -> None:
        index_rows = _rows(indices)
        weight_rows = _rows(weights)
        if len(index_rows) != len(weight_rows):
            raise ValueError("indices and weights must have the same row count")

        layer = int(layer)
        for idx_row, weight_row in zip(index_rows, weight_rows, strict=True):
            idx = [int(v) for v in idx_row]
            w = [float(v) for v in weight_row]
            if len(idx) != len(w):
                raise ValueError("indices and weights rows must have the same width")
            for expert in idx:
                if expert < 0 or expert >= self.num_experts:
                    raise ValueError(
                        f"layer {layer}: expert id {expert} outside 0..{self.num_experts - 1}"
                    )
            self.records.append({"layer": layer, "indices": idx, "weights": w})
            self._token_counts[layer] += 1
            for expert, weight in zip(idx, w, strict=True):
                self._hit_counts[layer][expert] += 1
                self._prob_mass[layer][expert] += weight

    def layers(self) -> Iterable[int]:
        return sorted(self._token_counts)

    def to_trace(self, *, source: str, prompts: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for layer in self.layers():
            hit_counts = self._hit_counts[layer]
            prob_mass = self._prob_mass[layer]
            prob_rank = sorted(
                range(self.num_experts),
                key=lambda expert: (-prob_mass.get(expert, 0.0), -hit_counts.get(expert, 0), expert),
            )
            hit_rank = sorted(
                range(self.num_experts),
                key=lambda expert: (-hit_counts.get(expert, 0), -prob_mass.get(expert, 0.0), expert),
            )
            observed = [expert for expert in prob_rank if prob_mass.get(expert, 0.0) > 0.0]
            layers[str(layer)] = {
                "token_count": self._token_counts[layer],
                "observed_experts": len(observed),
                "prob_rank": observed,
                "hit_rank": [expert for expert in hit_rank if hit_counts.get(expert, 0) > 0],
                "prob_mass_top": [
                    {"expert": expert, "prob_mass": prob_mass[expert], "hits": hit_counts[expert]}
                    for expert in observed[:32]
                ],
            }

        return {
            "schema": "mimo-v2-router-trace-v1",
            "source": source,
            "prompts": prompts,
            "num_experts": self.num_experts,
            "metadata": metadata or {},
            "records": self.records,
            "layers": layers,
        }
