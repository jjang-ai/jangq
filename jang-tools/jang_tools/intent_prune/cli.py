"""CLI for hybrid_v1 Intent Prune scoring.

Commands (registered on the main jang-tools parser):

* ``intent-prune-score`` — transitions/adjacency → ``jang-intent-prune-plan-v1``
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .graph import DEFAULT_MAX_ITER, DEFAULT_TELEPORT, DEFAULT_TOL
from .score import (
    PLAN_SCHEMA,
    PRESET_WEIGHTS,
    SAFETY_STANCES,
    build_prune_plan,
    score_hybrid,
    write_prune_plan,
)
from .transitions import (
    build_adjacency_from_jsonl,
    iter_transition_records,
)


def register(subparsers) -> None:
    """Register Intent Prune scoring CLI commands."""
    p = subparsers.add_parser(
        "intent-prune-score",
        help="Hybrid_v1 path+mass+domain Intent Prune scorer → prune plan JSON",
    )
    p.add_argument(
        "--transitions",
        default="",
        help="Path to expert_transitions.jsonl (preferred input)",
    )
    p.add_argument(
        "--adjacency",
        default="",
        help="Path to adjacency JSON (optional; used when transitions omitted)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output jang-intent-prune-plan-v1 JSON path",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        required=True,
        help="Model expert width E (layer-prefixed nodes / reshape)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=0,
        help="MoE layer count L (0 = infer from traces)",
    )
    p.add_argument(
        "--keep-k",
        type=int,
        required=True,
        help="Uniform keep-K experts per layer",
    )
    p.add_argument(
        "--preset",
        default="balanced",
        choices=sorted(PRESET_WEIGHTS),
        help="Fusion weight preset (default: balanced)",
    )
    p.add_argument(
        "--safety-stance",
        default="balanced",
        choices=list(SAFETY_STANCES),
        help="Safety stance: keep | balanced | crack (default: balanced)",
    )
    p.add_argument(
        "--intent",
        action="append",
        default=[],
        dest="intents_keep",
        help="Keep-intent domain slug (repeatable)",
    )
    p.add_argument(
        "--drop-intent",
        action="append",
        default=[],
        dest="intents_drop",
        help="Drop-intent domain slug (repeatable)",
    )
    p.add_argument(
        "--uniform-weight",
        action="store_true",
        help="Count transitions as 1.0 instead of gate product",
    )
    p.add_argument(
        "--teleport",
        type=float,
        default=DEFAULT_TELEPORT,
        help=f"PageRank teleport ε over observed nodes (default: {DEFAULT_TELEPORT})",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"Power-iteration L∞ tolerance (default: {DEFAULT_TOL})",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Power-iteration max iterations (default: {DEFAULT_MAX_ITER})",
    )
    p.add_argument(
        "--source-model",
        default="",
        help="Source model path recorded on the plan",
    )
    p.add_argument(
        "--backend",
        default="",
        help="Backend id recorded on the plan (e.g. qwen35_moe_vmlx)",
    )
    p.add_argument(
        "--trained-top-k",
        type=int,
        default=0,
        help="Trained router top-k for safety validation (0 = use keep-k)",
    )
    p.add_argument(
        "--suite-name",
        default="",
        help="Suite name recorded under plan.suite",
    )
    p.add_argument(
        "--suite-prompt-count",
        type=int,
        default=0,
        help="Suite prompt count recorded under plan.suite",
    )
    p.set_defaults(func=_cmd_intent_prune_score, json=True)


def _cmd_intent_prune_score(args) -> None:
    transitions_raw = (args.transitions or "").strip()
    adjacency_raw = (args.adjacency or "").strip()
    if not transitions_raw and not adjacency_raw:
        raise SystemExit("error: provide --transitions and/or --adjacency")

    num_experts = int(args.num_experts)
    if num_experts <= 0:
        raise SystemExit("error: --num-experts must be positive")
    keep_k = int(args.keep_k)
    if keep_k < 1:
        raise SystemExit("error: --keep-k must be >= 1")

    num_layers = int(args.num_layers) if args.num_layers and args.num_layers > 0 else None
    weight_by_gate = not bool(args.uniform_weight)

    records: list[dict[str, Any]] | None = None
    adjacency: dict[str, Any] | None = None

    if transitions_raw:
        transitions_path = Path(transitions_raw).expanduser()
        if not transitions_path.is_file():
            raise FileNotFoundError(f"transitions path not found: {transitions_path}")
        records = list(iter_transition_records(transitions_path))
        if not records:
            warnings.warn(
                "no transition records in expert_transitions.jsonl",
                stacklevel=1,
            )

    if adjacency_raw:
        adjacency_path = Path(adjacency_raw).expanduser()
        if not adjacency_path.is_file():
            raise FileNotFoundError(f"adjacency path not found: {adjacency_path}")
        adjacency = json.loads(adjacency_path.read_text(encoding="utf-8"))
        if not isinstance(adjacency, dict):
            raise ValueError("adjacency file must contain a JSON object")
    elif records is not None and not records:
        # Empty transitions — still allow adjacency-less zero graph if layers known
        adjacency = None

    # When only adjacency is provided, score from it (no conditional π_I/π_D/π_S)
    result = score_hybrid(
        records=records,
        adjacency=adjacency,
        num_experts=num_experts,
        num_layers=num_layers,
        keep_k=keep_k,
        intents_keep=list(args.intents_keep or []),
        intents_drop=list(args.intents_drop or []),
        safety_stance=str(args.safety_stance),
        preset=str(args.preset),
        weight_by_gate=weight_by_gate,
        teleport=float(args.teleport),
        tol=float(args.tol),
        max_iter=int(args.max_iter),
    )

    suite: dict[str, Any] = {}
    if args.suite_name:
        suite["name"] = str(args.suite_name)
    if args.suite_prompt_count and args.suite_prompt_count > 0:
        suite["prompt_count"] = int(args.suite_prompt_count)

    trained_top_k = (
        int(args.trained_top_k) if args.trained_top_k and args.trained_top_k > 0 else None
    )

    plan = build_prune_plan(
        result,
        source_model=str(args.source_model or ""),
        backend=str(args.backend or ""),
        suite=suite or None,
        trained_top_k=trained_top_k,
        extra={
            "transitions": str(Path(transitions_raw).expanduser()) if transitions_raw else "",
            "adjacency": str(Path(adjacency_raw).expanduser()) if adjacency_raw else "",
        },
    )

    write_prune_plan(args.output, plan)

    summary: dict[str, Any] = {
        "ok": True,
        "schema": PLAN_SCHEMA,
        "scorer": plan["scorer"],
        "preset": plan["preset"],
        "safety_stance": plan["safety_stance"],
        "output": str(Path(args.output).expanduser()),
        "num_layers": plan["num_layers"],
        "num_experts_source": plan["num_experts_source"],
        "keep_experts_per_layer": plan["keep_experts_per_layer"],
        "intents_keep": plan["intents_keep"],
        "safety_passed": plan["safety"]["passed"],
        "record_count": result.meta.get("record_count", 0),
    }
    # Sample first layer keep list for CLI visibility
    if plan.get("layers"):
        first = sorted(plan["layers"].keys(), key=lambda x: int(x))[0]
        summary["layer0_keep"] = plan["layers"][first]

    if result.meta.get("record_count", 0) == 0 and not adjacency:
        summary["ok"] = False
        summary["reason"] = "no transitions and no adjacency; plan may be empty"
        warnings.warn(summary["reason"], stacklevel=1)
        print(json.dumps(summary, sort_keys=True))
        raise SystemExit(2)

    print(json.dumps(summary, sort_keys=True))
