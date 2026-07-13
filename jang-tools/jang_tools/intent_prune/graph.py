"""Path / stationary scores for Intent Prune (MAESTRO-class).

Builds a (sparse) row-stochastic transition operator over layer-prefixed
expert nodes (``node = layer * E + expert``) and computes a PageRank-style
stationary distribution via power iteration. Sparse / disconnected graphs
get a small uniform teleport over observed nodes so a unique stationary
exists.

Dense ``build_row_stochastic`` remains for small fixtures / tests; production
path scoring uses sparse adjacency-list matvec (O(edges) per iteration).
"""

from __future__ import annotations

import warnings
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


def mass_has_signal(mass: Sequence[Sequence[float]]) -> bool:
    """True if any mass entry is strictly positive."""
    for row in mass:
        for value in row:
            if float(value) > 0.0:
                return True
    return False


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


def _normalize_observed(
    observed: Sequence[int] | None,
    n_nodes: int,
) -> list[int]:
    if observed:
        obs = [int(i) for i in observed if 0 <= int(i) < n_nodes]
        if obs:
            return obs
    return list(range(n_nodes)) if n_nodes > 0 else []


def build_sparse_operator(
    edges: Sequence[tuple[int, int, float]],
    n_nodes: int,
    *,
    observed: Sequence[int] | None = None,
    teleport: float = DEFAULT_TELEPORT,
) -> dict[str, Any]:
    """Build a sparse PageRank-style transition operator.

    Returns a dict consumed by :func:`power_iteration_sparse`:

    * ``out_adj[i]`` — list of ``(j, raw_weight)`` outgoing edges
    * ``out_weight[i]`` — sum of raw outgoing weights (0 ⇒ dangling)
    * ``tele_vec`` — teleport distribution over observed nodes
    * ``teleport`` — damping ε
    """
    if n_nodes <= 0:
        return {
            "n_nodes": 0,
            "out_adj": [],
            "out_weight": [],
            "tele_vec": [],
            "teleport": float(teleport),
            "observed": [],
        }
    teleport = float(teleport)
    if teleport < 0.0 or teleport >= 1.0:
        raise ValueError(f"teleport must be in [0, 1), got {teleport}")

    out_adj: list[dict[int, float]] = [dict() for _ in range(n_nodes)]
    out_weight = [0.0] * n_nodes
    for src, dst, weight in edges:
        if src < 0 or dst < 0 or src >= n_nodes or dst >= n_nodes:
            continue
        w = float(weight)
        if w <= 0.0:
            continue
        out_adj[src][dst] = out_adj[src].get(dst, 0.0) + w
        out_weight[src] += w

    obs = _normalize_observed(observed, n_nodes)
    inv_obs = 1.0 / float(len(obs)) if obs else 0.0
    tele_vec = [0.0] * n_nodes
    for i in obs:
        tele_vec[i] = inv_obs

    # Convert dict adj to sorted list of pairs for stable iteration
    out_list: list[list[tuple[int, float]]] = [
        sorted(adj.items()) for adj in out_adj
    ]
    return {
        "n_nodes": n_nodes,
        "out_adj": out_list,
        "out_weight": out_weight,
        "tele_vec": tele_vec,
        "teleport": teleport,
        "observed": obs,
        "edge_count": sum(len(row) for row in out_list),
    }


def build_row_stochastic(
    edges: Sequence[tuple[int, int, float]],
    n_nodes: int,
    *,
    observed: Sequence[int] | None = None,
    teleport: float = DEFAULT_TELEPORT,
) -> list[list[float]]:
    """Build dense row-stochastic ``P`` with optional PageRank teleport.

    Prefer :func:`build_sparse_operator` + :func:`power_iteration_sparse` for
    production-width graphs. Dense form is kept for small fixtures/tests.

    ``P[i][j]`` = probability of stepping from node ``i`` to node ``j``.
    Rows with no outgoing mass (dangling) teleport uniformly over ``observed``
    (or all nodes if ``observed`` is empty).
    """
    if n_nodes <= 0:
        return []
    op = build_sparse_operator(
        edges, n_nodes, observed=observed, teleport=teleport
    )
    teleport_f = float(op["teleport"])
    stay = 1.0 - teleport_f
    tele_vec: list[float] = op["tele_vec"]
    out_weight: list[float] = op["out_weight"]
    out_adj: list[list[tuple[int, float]]] = op["out_adj"]

    p: list[list[float]] = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        row = p[i]
        if out_weight[i] > 0.0 and stay > 0.0:
            scale = stay / out_weight[i]
            for j, w in out_adj[i]:
                row[j] += scale * w
            if teleport_f > 0.0:
                for j in range(n_nodes):
                    row[j] += teleport_f * tele_vec[j]
        else:
            for j in range(n_nodes):
                row[j] = tele_vec[j]
    return p


def _normalize_start(
    n: int,
    start: Sequence[float] | None,
) -> list[float]:
    if start is not None:
        if len(start) != n:
            raise ValueError(f"start length {len(start)} != n={n}")
        pi = [max(float(x), 0.0) for x in start]
        total = sum(pi)
        if total <= 0.0:
            return [1.0 / n] * n if n > 0 else []
        return [x / total for x in pi]
    return [1.0 / n] * n if n > 0 else []


def power_iteration_sparse(
    operator: Mapping[str, Any],
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    start: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Sparse left-stationary: ``π_{t+1} = π_t P`` with teleport rank-1 update.

    Returns ``{pi, iterations, converged, delta, max_iter, tol}``.
    """
    n = int(operator.get("n_nodes") or 0)
    if n <= 0:
        return {
            "pi": [],
            "iterations": 0,
            "converged": True,
            "delta": 0.0,
            "max_iter": int(max_iter),
            "tol": float(tol),
        }

    out_adj: list[list[tuple[int, float]]] = operator["out_adj"]
    out_weight: list[float] = operator["out_weight"]
    tele_vec: list[float] = operator["tele_vec"]
    teleport = float(operator["teleport"])
    stay = 1.0 - teleport

    pi = _normalize_start(n, start)
    tol = float(tol)
    max_iter = max(int(max_iter), 1)
    delta = float("inf")
    converged = False
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        nxt = [0.0] * n
        tele_coeff = 0.0
        for i, mass_i in enumerate(pi):
            if mass_i == 0.0:
                continue
            ow = out_weight[i]
            if ow > 0.0 and stay > 0.0:
                scale = mass_i * stay / ow
                for j, w in out_adj[i]:
                    nxt[j] += scale * w
                tele_coeff += mass_i * teleport
            else:
                # Dangling: full teleport
                tele_coeff += mass_i
        if tele_coeff != 0.0:
            for j, t in enumerate(tele_vec):
                if t != 0.0:
                    nxt[j] += tele_coeff * t

        s = sum(nxt)
        if s <= 0.0:
            delta = float("inf")
            break
        inv = 1.0 / s
        nxt = [x * inv for x in nxt]

        delta = 0.0
        for a, b in zip(pi, nxt):
            d = a - b
            if d < 0.0:
                d = -d
            if d > delta:
                delta = d
        pi = nxt
        if delta < tol:
            converged = True
            break

    return {
        "pi": pi,
        "iterations": iterations,
        "converged": converged,
        "delta": float(delta if delta != float("inf") else delta),
        "max_iter": max_iter,
        "tol": tol,
    }


def power_iteration(
    transition: Sequence[Sequence[float]],
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    start: Sequence[float] | None = None,
    return_info: bool = False,
) -> list[float] | dict[str, Any]:
    """Left-stationary distribution via power iteration: ``π_{t+1} = π_t P``.

    ``transition[i][j]`` is P(i→j). By default returns the probability vector.
    Pass ``return_info=True`` for ``{pi, iterations, converged, delta, ...}``.

    Note: ``teleport=0`` on periodic/bipartite graphs may not converge; prefer
    default teleport > 0. Non-convergence is reported when ``return_info=True``.
    """
    n = len(transition)
    if n == 0:
        empty = {
            "pi": [],
            "iterations": 0,
            "converged": True,
            "delta": 0.0,
            "max_iter": int(max_iter),
            "tol": float(tol),
        }
        return empty if return_info else empty["pi"]
    if any(len(row) != n for row in transition):
        raise ValueError("transition matrix must be square")

    pi = _normalize_start(n, start)
    tol = float(tol)
    max_iter = max(int(max_iter), 1)
    delta = float("inf")
    converged = False
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        nxt = [0.0] * n
        for i, mass_i in enumerate(pi):
            if mass_i == 0.0:
                continue
            row = transition[i]
            for j in range(n):
                nxt[j] += mass_i * float(row[j])
        s = sum(nxt)
        if s <= 0.0:
            delta = float("inf")
            break
        inv = 1.0 / s
        nxt = [x * inv for x in nxt]
        delta = 0.0
        for a, b in zip(pi, nxt):
            d = a - b
            if d < 0.0:
                d = -d
            if d > delta:
                delta = d
        pi = nxt
        if delta < tol:
            converged = True
            break

    info = {
        "pi": pi,
        "iterations": iterations,
        "converged": converged,
        "delta": float(delta),
        "max_iter": max_iter,
        "tol": tol,
    }
    return info if return_info else pi


def stationary_from_adjacency(
    adjacency: Mapping[str, Any],
    *,
    num_experts: int | None = None,
    num_layers: int | None = None,
    teleport: float = DEFAULT_TELEPORT,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    require_signal: bool = False,
) -> dict[str, Any]:
    """Compute path / stationary scores ``π`` reshaped to ``[L][E]``.

    Returns a dict with:

    * ``pi`` — dense ``list[list[float]]`` shape ``[L][E]``
    * ``pi_nodes`` — dense vector over layer-prefixed nodes
    * ``num_layers``, ``num_experts``
    * ``observed_nodes``, ``edge_count``
    * ``iterations``, ``converged``, ``delta`` — power-iteration metadata
    * ``has_signal`` — True when edges or positive mass exist

    When there are no edges, ``pi`` is all zeros (mass-only graphs should not
    invent uniform path scores via teleport over the full ``L*E`` space).
    Pass ``require_signal=True`` to raise if neither edges nor mass exist.
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

    empty_meta = {
        "pi": [],
        "pi_nodes": [],
        "num_layers": 0,
        "num_experts": e_width,
        "observed_nodes": [],
        "teleport": float(teleport),
        "edge_count": 0,
        "iterations": 0,
        "converged": True,
        "delta": 0.0,
        "has_signal": False,
    }
    if n_layers <= 0:
        if require_signal:
            raise ValueError("empty graph: num_layers resolved to 0 (no edges or mass)")
        return empty_meta

    n_nodes = n_layers * e_width
    edges = _collect_edges(adjacency, num_experts=e_width)
    observed = _observed_nodes(
        adjacency, edges, num_experts=e_width, num_layers=n_layers
    )
    mass_m = mass_matrix_from_adjacency(
        adjacency, num_layers=n_layers, num_experts=e_width
    )
    has_mass = mass_has_signal(mass_m)
    has_edges = len(edges) > 0
    has_signal = has_edges or has_mass

    if require_signal and not has_signal:
        raise ValueError("empty graph: no edges or mass")

    if not has_edges:
        # Mass-only (or fully empty): do not invent uniform path scores.
        pi = [[0.0] * e_width for _ in range(n_layers)]
        return {
            "pi": pi,
            "pi_nodes": [0.0] * n_nodes,
            "num_layers": n_layers,
            "num_experts": e_width,
            "observed_nodes": observed,
            "teleport": float(teleport),
            "edge_count": 0,
            "iterations": 0,
            "converged": True,
            "delta": 0.0,
            "has_signal": has_signal,
            "mass": mass_m,
        }

    op = build_sparse_operator(
        edges,
        n_nodes,
        observed=observed if observed else None,
        teleport=float(teleport),
    )

    start = None
    if observed:
        start = [0.0] * n_nodes
        share = 1.0 / float(len(observed))
        for idx in observed:
            if 0 <= idx < n_nodes:
                start[idx] = share

    info = power_iteration_sparse(op, tol=tol, max_iter=max_iter, start=start)
    pi_nodes: list[float] = info["pi"]

    if not info["converged"]:
        warnings.warn(
            f"power iteration did not converge: delta={info['delta']:.3e} "
            f">= tol={float(tol):.3e} after {info['iterations']} iterations "
            f"(teleport={float(teleport)}; increase max_iter or teleport)",
            stacklevel=2,
        )

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
        "iterations": int(info["iterations"]),
        "converged": bool(info["converged"]),
        "delta": float(info["delta"]),
        "has_signal": True,
        "mass": mass_m,
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
    if "mass" not in result:
        result["mass"] = mass_matrix_from_adjacency(
            adjacency,
            num_layers=result["num_layers"],
            num_experts=num_experts,
        )
    return result
