"""Path / stationary scores for Intent Prune (MAESTRO-class).

Builds a row-stochastic transition matrix over layer-prefixed expert nodes
(``node = layer * E + expert``) and computes a PageRank-style stationary
distribution via power iteration. Sparse / disconnected graphs get a small
uniform teleport over observed nodes so a unique stationary exists.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# Defaults from plan §11.3
DEFAULT_TELEPORT = 1e-2
DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 200


def node_id(layer: int, expert: int, num_experts: int) -> int:
    """Layer-prefixed node id: ``layer * E + expert``."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if expert < 0 or expert >= num_experts:
        raise ValueError(
            f"expert id {expert} out of range for num_experts={num_experts}"
        )
    return int(layer) * int(num_experts) + int(expert)


def decode_node(node: int, num_experts: int) -> tuple[int, int]:
    """Inverse of :func:`node_id`."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    layer, expert = divmod(int(node), int(num_experts))
    return layer, expert


def _as_float_map(raw: Any) -> dict[int, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def mass_matrix_from_adjacency(
    adjacency: Mapping[str, Any],
    *,
    num_layers: int | None = None,
    num_experts: int | None = None,
) -> list[list[float]]:
    """Reshape adjacency ``mass`` into ``[L][E]`` dense matrix (zeros for missing)."""
    e_width = int(num_experts if num_experts is not None else (adjacency.get("num_experts") or 0))
    if e_width <= 0:
        raise ValueError("num_experts is required to reshape mass to [L, E]")

    mass_raw = adjacency.get("mass") or {}
    if not isinstance(mass_raw, Mapping):
        mass_raw = {}

    observed_layers = [int(k) for k in mass_raw.keys() if str(k).lstrip("-").isdigit()]
    if num_layers is not None and int(num_layers) > 0:
        n_layers = int(num_layers)
    else:
        n_obs = int(adjacency.get("num_layers_observed") or 0)
        n_layers = max(n_obs, max(observed_layers) + 1 if observed_layers else 0)

    matrix = [[0.0] * e_width for _ in range(n_layers)]
    for layer_key, experts in mass_raw.items():
        try:
            layer = int(layer_key)
        except (TypeError, ValueError):
            continue
        if layer < 0 or layer >= n_layers:
            continue
        for expert, value in _as_float_map(experts).items():
            if 0 <= expert < e_width:
                matrix[layer][expert] = float(value)
    return matrix


def _collect_edges(
    adjacency: Mapping[str, Any],
    *,
    num_experts: int,
) -> list[tuple[int, int, float]]:
    """Return ``(from_node, to_node, weight)`` triples."""
    edges_out: list[tuple[int, int, float]] = []
    edges = adjacency.get("edges")
    if isinstance(edges, list) and edges:
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            try:
                weight = float(edge.get("weight") or 0.0)
            except (TypeError, ValueError):
                continue
            if weight <= 0.0:
                continue
            if "from_node" in edge and "to_node" in edge:
                try:
                    src = int(edge["from_node"])
                    dst = int(edge["to_node"])
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    fl = int(edge["from_layer"])
                    fe = int(edge["from_expert"])
                    tl = int(edge["to_layer"])
                    te = int(edge["to_expert"])
                except (KeyError, TypeError, ValueError):
                    continue
                if fe < 0 or fe >= num_experts or te < 0 or te >= num_experts:
                    continue
                src = node_id(fl, fe, num_experts)
                dst = node_id(tl, te, num_experts)
            edges_out.append((src, dst, weight))
        return edges_out

    # Fallback: nested transition_counts
    nested = adjacency.get("transition_counts")
    if not isinstance(nested, Mapping):
        return edges_out
    for from_layer_s, to_map in nested.items():
        if not isinstance(to_map, Mapping):
            continue
        try:
            from_layer = int(from_layer_s)
        except (TypeError, ValueError):
            continue
        for to_layer_s, from_experts in to_map.items():
            if not isinstance(from_experts, Mapping):
                continue
            try:
                to_layer = int(to_layer_s)
            except (TypeError, ValueError):
                continue
            for from_expert_s, to_experts in from_experts.items():
                if not isinstance(to_experts, Mapping):
                    continue
                try:
                    from_expert = int(from_expert_s)
                except (TypeError, ValueError):
                    continue
                if from_expert < 0 or from_expert >= num_experts:
                    continue
                src = node_id(from_layer, from_expert, num_experts)
                for to_expert_s, weight_raw in to_experts.items():
                    try:
                        to_expert = int(to_expert_s)
                        weight = float(weight_raw)
                    except (TypeError, ValueError):
                        continue
                    if weight <= 0.0 or to_expert < 0 or to_expert >= num_experts:
                        continue
                    dst = node_id(to_layer, to_expert, num_experts)
                    edges_out.append((src, dst, weight))
    return edges_out


def _observed_nodes(
    adjacency: Mapping[str, Any],
    edges: Sequence[tuple[int, int, float]],
    *,
    num_experts: int,
    num_layers: int,
) -> list[int]:
    nodes: set[int] = set()
    raw_nodes = adjacency.get("nodes")
    if isinstance(raw_nodes, list):
        for entry in raw_nodes:
            if not isinstance(entry, Mapping):
                continue
            if "node" in entry:
                try:
                    nodes.add(int(entry["node"]))
                    continue
                except (TypeError, ValueError):
                    pass
            try:
                layer = int(entry["layer"])
                expert = int(entry["expert"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= expert < num_experts and 0 <= layer < num_layers:
                nodes.add(node_id(layer, expert, num_experts))
    for src, dst, _ in edges:
        nodes.add(src)
        nodes.add(dst)
    mass_raw = adjacency.get("mass") or {}
    if isinstance(mass_raw, Mapping):
        for layer_key, experts in mass_raw.items():
            try:
                layer = int(layer_key)
            except (TypeError, ValueError):
                continue
            if layer < 0 or layer >= num_layers:
                continue
            for expert in _as_float_map(experts):
                if 0 <= expert < num_experts:
                    nodes.add(node_id(layer, expert, num_experts))
    return sorted(nodes)


def build_row_stochastic(
    edges: Sequence[tuple[int, int, float]],
    n_nodes: int,
    *,
    observed: Sequence[int] | None = None,
    teleport: float = DEFAULT_TELEPORT,
) -> list[list[float]]:
    """Build dense row-stochastic ``P`` with optional PageRank teleport.

    ``P[i][j]`` = probability of stepping from node ``i`` to node ``j``.
    Rows with no outgoing mass (dangling) teleport uniformly over ``observed``
    (or all nodes if ``observed`` is empty).
    """
    if n_nodes <= 0:
        return []
    teleport = float(teleport)
    if teleport < 0.0 or teleport >= 1.0:
        raise ValueError(f"teleport must be in [0, 1), got {teleport}")

    # Weighted adjacency
    out_weight = [0.0] * n_nodes
    adj: list[dict[int, float]] = [dict() for _ in range(n_nodes)]
    for src, dst, weight in edges:
        if src < 0 or dst < 0 or src >= n_nodes or dst >= n_nodes:
            continue
        w = float(weight)
        if w <= 0.0:
            continue
        adj[src][dst] = adj[src].get(dst, 0.0) + w
        out_weight[src] += w

    if observed:
        obs = [int(i) for i in observed if 0 <= int(i) < n_nodes]
        if not obs:
            obs = list(range(n_nodes))
    else:
        obs = list(range(n_nodes))
    inv_obs = 1.0 / float(len(obs))
    inv_all = 1.0 / float(n_nodes)

    # Precompute uniform teleport target vector over observed nodes
    tele_vec = [0.0] * n_nodes
    for i in obs:
        tele_vec[i] = inv_obs

    p: list[list[float]] = [[0.0] * n_nodes for _ in range(n_nodes)]
    stay = 1.0 - teleport
    for i in range(n_nodes):
        row = p[i]
        if out_weight[i] > 0.0 and stay > 0.0:
            scale = stay / out_weight[i]
            for j, w in adj[i].items():
                row[j] += scale * w
            if teleport > 0.0:
                for j in range(n_nodes):
                    row[j] += teleport * tele_vec[j]
        else:
            # Dangling: full teleport (or uniform over all if no observed)
            if teleport > 0.0 or obs:
                for j in range(n_nodes):
                    row[j] = tele_vec[j]
            else:
                for j in range(n_nodes):
                    row[j] = inv_all
    return p


def power_iteration(
    transition: Sequence[Sequence[float]],
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    start: Sequence[float] | None = None,
) -> list[float]:
    """Left-stationary distribution via power iteration: ``π_{t+1} = π_t P``.

    ``transition[i][j]`` is P(i→j). Returns a probability vector summing to 1.
    """
    n = len(transition)
    if n == 0:
        return []
    if any(len(row) != n for row in transition):
        raise ValueError("transition matrix must be square")

    if start is not None:
        if len(start) != n:
            raise ValueError(f"start length {len(start)} != n={n}")
        pi = [max(float(x), 0.0) for x in start]
        total = sum(pi)
        if total <= 0.0:
            pi = [1.0 / n] * n
        else:
            pi = [x / total for x in pi]
    else:
        pi = [1.0 / n] * n

    tol = float(tol)
    max_iter = max(int(max_iter), 1)

    for _ in range(max_iter):
        nxt = [0.0] * n
        for i, mass_i in enumerate(pi):
            if mass_i == 0.0:
                continue
            row = transition[i]
            for j in range(n):
                nxt[j] += mass_i * float(row[j])
        # Renormalize against numerical drift
        s = sum(nxt)
        if s <= 0.0:
            return pi
        nxt = [x / s for x in nxt]
        delta = 0.0
        for a, b in zip(pi, nxt):
            d = a - b
            if d < 0.0:
                d = -d
            if d > delta:
                delta = d
        pi = nxt
        if delta < tol:
            break
    return pi


def stationary_from_adjacency(
    adjacency: Mapping[str, Any],
    *,
    num_experts: int | None = None,
    num_layers: int | None = None,
    teleport: float = DEFAULT_TELEPORT,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, Any]:
    """Compute path / stationary scores ``π`` reshaped to ``[L][E]``.

    Returns a dict with:

    * ``pi`` — dense ``list[list[float]]`` shape ``[L][E]``
    * ``pi_nodes`` — dense vector over layer-prefixed nodes
    * ``num_layers``, ``num_experts``
    * ``observed_nodes``, ``iterations`` metadata
    """
    e_width = int(
        num_experts if num_experts is not None else (adjacency.get("num_experts") or 0)
    )
    if e_width <= 0:
        raise ValueError(
            "num_experts is required (pass model expert width or include it on adjacency)"
        )

    n_obs_layers = int(adjacency.get("num_layers_observed") or 0)
    if num_layers is not None and int(num_layers) > 0:
        n_layers = int(num_layers)
    else:
        # Infer from mass / edges / nodes
        max_layer = n_obs_layers - 1 if n_obs_layers > 0 else -1
        for edge in adjacency.get("edges") or []:
            if not isinstance(edge, Mapping):
                continue
            for key in ("from_layer", "to_layer"):
                if key in edge:
                    try:
                        max_layer = max(max_layer, int(edge[key]))
                    except (TypeError, ValueError):
                        pass
            if "from_node" in edge:
                try:
                    layer, _ = decode_node(int(edge["from_node"]), e_width)
                    max_layer = max(max_layer, layer)
                except (TypeError, ValueError):
                    pass
            if "to_node" in edge:
                try:
                    layer, _ = decode_node(int(edge["to_node"]), e_width)
                    max_layer = max(max_layer, layer)
                except (TypeError, ValueError):
                    pass
        mass_raw = adjacency.get("mass") or {}
        if isinstance(mass_raw, Mapping):
            for layer_key in mass_raw:
                try:
                    max_layer = max(max_layer, int(layer_key))
                except (TypeError, ValueError):
                    pass
        n_layers = max_layer + 1 if max_layer >= 0 else 0

    if n_layers <= 0:
        return {
            "pi": [],
            "pi_nodes": [],
            "num_layers": 0,
            "num_experts": e_width,
            "observed_nodes": [],
            "teleport": float(teleport),
        }

    n_nodes = n_layers * e_width
    edges = _collect_edges(adjacency, num_experts=e_width)
    observed = _observed_nodes(
        adjacency, edges, num_experts=e_width, num_layers=n_layers
    )
    # Restrict matrix size to full L*E space so reshape is trivial
    p = build_row_stochastic(
        edges,
        n_nodes,
        observed=observed if observed else None,
        teleport=float(teleport),
    )

    # Prefer start mass on observed nodes (uniform); falls back inside power_iteration
    start = None
    if observed:
        start = [0.0] * n_nodes
        share = 1.0 / float(len(observed))
        for idx in observed:
            if 0 <= idx < n_nodes:
                start[idx] = share

    pi_nodes = power_iteration(p, tol=tol, max_iter=max_iter, start=start)

    pi = [[0.0] * e_width for _ in range(n_layers)]
    for node, value in enumerate(pi_nodes):
        layer, expert = divmod(node, e_width)
        if layer < n_layers:
            pi[layer][expert] = float(value)

    return {
        "pi": pi,
        "pi_nodes": pi_nodes,
        "num_layers": n_layers,
        "num_experts": e_width,
        "observed_nodes": observed,
        "teleport": float(teleport),
        "edge_count": len(edges),
    }


def path_scores_from_transitions(
    records: Iterable[Mapping[str, Any]],
    *,
    num_experts: int,
    num_layers: int | None = None,
    weight_by_gate: bool = True,
    teleport: float = DEFAULT_TELEPORT,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, Any]:
    """Build adjacency from transition records then compute path scores."""
    from .transitions import build_adjacency_from_transitions

    adjacency = build_adjacency_from_transitions(
        records,
        num_experts=num_experts,
        weight_by_gate=weight_by_gate,
    )
    result = stationary_from_adjacency(
        adjacency,
        num_experts=num_experts,
        num_layers=num_layers,
        teleport=teleport,
        tol=tol,
        max_iter=max_iter,
    )
    result["adjacency"] = adjacency
    result["mass"] = mass_matrix_from_adjacency(
        adjacency,
        num_layers=result["num_layers"],
        num_experts=num_experts,
    )
    return result
