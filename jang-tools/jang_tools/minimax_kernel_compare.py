"""Fresh-process MiniMax JANG/JANGTQ kernel comparison harness."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


JANGTQ_KIND = "jangtq"
JANG_KIND = "jang"

JANGTQ_MODES = ("legacy_prefill", "default_prefill", "global_auto")
JANG_MODES = ("affine_default",)

_JANGTQ_ENV_KEYS = (
    "JANGTQ_MPP_NAX",
    "JANGTQ_MPP_NAX_PREFILL",
    "JANGTQ_MPP_NAX_STRICT",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if isinstance(text, dict):
        return text
    language = config.get("language_config")
    if isinstance(language, dict):
        return language
    return config


def detect_model_kind(model_path: str | Path) -> str:
    """Return ``jangtq`` for MXTQ bundles and ``jang`` for affine JANG bundles."""

    path = Path(model_path)
    config = _read_json(path / "config.json")
    jang_config = _read_json(path / "jang_config.json")
    quant = config.get("quantization") if isinstance(config.get("quantization"), dict) else {}

    markers = (
        config.get("weight_format") == "mxtq",
        jang_config.get("weight_format") == "mxtq",
        quant.get("weight_format") == "mxtq",
        "mxtq_bits" in config,
        "mxtq_bits" in jang_config,
        "routed_expert_bits" in config,
        "routed_expert_bits" in jang_config,
    )
    if any(markers):
        return JANGTQ_KIND
    if jang_config:
        return JANG_KIND
    raise ValueError(f"Cannot detect JANG model kind from {path}")


def model_summary(model_path: str | Path) -> dict[str, Any]:
    """Extract the metadata that controls MiniMax routed-expert runtime shape."""

    path = Path(model_path)
    config = _read_json(path / "config.json")
    jang_config = _read_json(path / "jang_config.json")
    text = _text_config(config)
    quant = text.get("quantization")
    if not isinstance(quant, dict):
        quant = config.get("quantization") if isinstance(config.get("quantization"), dict) else {}
    kind = detect_model_kind(path)

    if kind == JANGTQ_KIND:
        weight_format = (
            config.get("weight_format")
            or jang_config.get("weight_format")
            or quant.get("weight_format")
            or "mxtq"
        )
    else:
        weight_format = quant.get("mode") or "affine"

    return {
        "path": str(path),
        "kind": kind,
        "profile": jang_config.get("profile")
        or (jang_config.get("quantization") or {}).get("profile")
        or config.get("profile"),
        "weight_format": weight_format,
        "model_type": text.get("model_type") or config.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size")
        or text.get("moe_intermediate_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_local_experts": text.get("num_local_experts")
        or text.get("num_experts")
        or text.get("n_routed_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok")
        or text.get("num_experts_per_token")
        or text.get("moe_top_k")
        or text.get("num_experts_per_topk"),
        "quantization_bits": quant.get("bits"),
        "quantization_group_size": quant.get("group_size"),
        "mxtq_bits": jang_config.get("mxtq_bits") or config.get("mxtq_bits"),
        "routed_expert_bits": jang_config.get("routed_expert_bits")
        or config.get("routed_expert_bits"),
    }


def mode_labels(kind: str, *, include_global_auto: bool = True) -> list[str]:
    if kind == JANGTQ_KIND:
        labels = list(JANGTQ_MODES)
        if not include_global_auto:
            labels.remove("global_auto")
        return labels
    if kind == JANG_KIND:
        return list(JANG_MODES)
    raise ValueError(f"Unknown model kind: {kind}")


def env_for_mode(mode: str) -> dict[str, str | None]:
    if mode == "legacy_prefill":
        return {
            "JANGTQ_MPP_NAX": None,
            "JANGTQ_MPP_NAX_PREFILL": "0",
            "JANGTQ_MPP_NAX_STRICT": None,
        }
    if mode == "default_prefill":
        return {
            "JANGTQ_MPP_NAX": None,
            "JANGTQ_MPP_NAX_PREFILL": None,
            "JANGTQ_MPP_NAX_STRICT": "1",
        }
    if mode == "global_auto":
        return {
            "JANGTQ_MPP_NAX": "auto",
            "JANGTQ_MPP_NAX_PREFILL": None,
            "JANGTQ_MPP_NAX_STRICT": "1",
        }
    if mode == "affine_default":
        return {}
    raise ValueError(f"Unknown comparison mode: {mode}")


def _apply_mode_env(mode: str) -> dict[str, str | None]:
    requested = env_for_mode(mode)
    prior: dict[str, str | None] = {}
    for key in _JANGTQ_ENV_KEYS:
        prior[key] = os.environ.get(key)
        if key in requested:
            value = requested[key]
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return prior


def _restore_env(prior: dict[str, str | None]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(x) for x in encoded]


def _build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    seed = (
        "You are validating a local MiniMax runtime on Apple Silicon. "
        "Keep the final answer short and say READY then the answer. "
    )
    payload = (
        "Context marker alpha: CERULEAN. "
        "Context marker beta: AMBER. "
        "Context marker gamma: VIOLET. "
        "The arithmetic check is 17 plus 28. "
    )
    text = seed
    while len(_token_ids(tokenizer, text)) < target_tokens:
        text += payload
    ids = _token_ids(tokenizer, text)
    return text, len(ids)


def _chat_prompt(tokenizer: Any, body: str):
    import mlx.core as mx

    messages = [
        {
            "role": "user",
            "content": body
            + "\nFinal answer: repeat CERULEAN and give 17+28 only.",
        }
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        rendered = body + "\nAssistant:"
    ids = _token_ids(tokenizer, rendered)
    return mx.array(ids, dtype=mx.uint32), len(ids), rendered


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        return "".join(tokenizer.decode([tid]) for tid in token_ids)


def _token_to_int(token: Any) -> int:
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def _shape_key(args: tuple[Any, ...]) -> str:
    parts = []
    for arg in args[:3]:
        shape = getattr(arg, "shape", None)
        if shape is not None:
            parts.append(str(tuple(int(x) for x in shape)))
    return " x ".join(parts) if parts else "unknown"


def _reset_counters(counters: dict[str, Any]) -> None:
    for key, value in list(counters.items()):
        if isinstance(value, Counter):
            value.clear()
        elif isinstance(value, dict):
            value.clear()
        elif isinstance(value, int):
            counters[key] = 0


def _jsonable_counters(counters: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in counters.items():
        if isinstance(value, Counter):
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def _install_jangtq_counters() -> dict[str, Any]:
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    import jang_tools.turboquant.gather_tq_kernel as gather_kernel
    import jang_tools.turboquant.tq_kernel as tq_kernel

    counters: dict[str, Any] = {
        "tq_fused_gate_up_calls": 0,
        "tq_switch_gather_calls": 0,
        "tq_fused_gate_up_grouped_mpp_nax_calls": 0,
        "tq_gather_grouped_mpp_nax_calls": 0,
        "tq_dense_mpp_nax_calls": 0,
        "shape_counts": Counter(),
    }

    orig_fused = fused_kernel.fused_gate_up_swiglu_matmul
    orig_tq_gather = tq_kernel._gather_tq_matmul
    orig_grouped_fused = fused_kernel._fused_gate_up_swiglu_mpp_nax_grouped_from_rot
    orig_grouped_gather = gather_kernel._gather_tq_mpp_nax_grouped_from_rot
    orig_dense_nax = tq_kernel._tq_matmul_mpp_nax

    def counted_fused(*args, **kwargs):
        counters["tq_fused_gate_up_calls"] += 1
        counters["shape_counts"][f"fused:{_shape_key(args)}"] += 1
        return orig_fused(*args, **kwargs)

    def counted_tq_gather(*args, **kwargs):
        counters["tq_switch_gather_calls"] += 1
        counters["shape_counts"][f"gather:{_shape_key(args)}"] += 1
        return orig_tq_gather(*args, **kwargs)

    def counted_grouped_fused(*args, **kwargs):
        counters["tq_fused_gate_up_grouped_mpp_nax_calls"] += 1
        counters["shape_counts"][f"grouped_fused:{_shape_key(args)}"] += 1
        return orig_grouped_fused(*args, **kwargs)

    def counted_grouped_gather(*args, **kwargs):
        counters["tq_gather_grouped_mpp_nax_calls"] += 1
        counters["shape_counts"][f"grouped_gather:{_shape_key(args)}"] += 1
        return orig_grouped_gather(*args, **kwargs)

    def counted_dense_nax(*args, **kwargs):
        counters["tq_dense_mpp_nax_calls"] += 1
        counters["shape_counts"][f"dense_nax:{_shape_key(args)}"] += 1
        return orig_dense_nax(*args, **kwargs)

    fused_kernel.fused_gate_up_swiglu_matmul = counted_fused
    tq_kernel._gather_tq_matmul = counted_tq_gather
    fused_kernel._fused_gate_up_swiglu_mpp_nax_grouped_from_rot = counted_grouped_fused
    gather_kernel._gather_tq_mpp_nax_grouped_from_rot = counted_grouped_gather
    tq_kernel._tq_matmul_mpp_nax = counted_dense_nax
    return counters


def _install_affine_counters(mx: Any) -> dict[str, Any]:
    counters: dict[str, Any] = {
        "mlx_quantized_matmul_calls": 0,
        "shape_counts": Counter(),
    }
    orig_quantized_matmul = mx.quantized_matmul

    def counted_quantized_matmul(*args, **kwargs):
        counters["mlx_quantized_matmul_calls"] += 1
        counters["shape_counts"][f"quantized_matmul:{_shape_key(args)}"] += 1
        return orig_quantized_matmul(*args, **kwargs)

    mx.quantized_matmul = counted_quantized_matmul
    return counters


def _run_once(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    *,
    max_tokens: int,
    prefill_step_size: int,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0, top_p=0.0)
    new_ids: list[int] = []
    t0 = time.perf_counter()
    first = None
    for token, _probs in generate_step(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        sampler=sampler,
        prefill_step_size=prefill_step_size,
    ):
        if hasattr(token, "shape"):
            mx.eval(token)
        if first is None:
            first = time.perf_counter()
        new_ids.append(_token_to_int(token))
    end = time.perf_counter()
    ttft = (first - t0) if first is not None else None
    total = end - t0
    decode_window = max(total - (ttft or 0.0), 1e-9)
    return {
        "new_tokens": len(new_ids),
        "ttft_s": ttft,
        "total_s": total,
        "approx_prefill_tok_s": (
            float(prompt.size) / ttft if ttft and ttft > 0 else None
        ),
        "decode_tok_s_after_first": (
            max(len(new_ids) - 1, 0) / decode_window if len(new_ids) > 1 else None
        ),
        "output": _decode(tokenizer, new_ids),
    }


def _memory_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "ru_maxrss": usage.ru_maxrss,
        "ru_maxrss_units": "bytes_on_macos_kb_on_linux",
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    model_path = Path(args.model)
    kind = args.kind or detect_model_kind(model_path)
    prompt_tokens = int(args.prompt_tokens[-1])
    mode = args.mode
    prior = _apply_mode_env(mode)
    try:
        summary = model_summary(model_path)
        counters: dict[str, Any]
        load_t0 = time.perf_counter()
        if kind == JANGTQ_KIND:
            counters = _install_jangtq_counters()
            from jang_tools.load_jangtq import load_jangtq_model

            model, tokenizer = load_jangtq_model(str(model_path))
        elif kind == JANG_KIND:
            counters = _install_affine_counters(mx)
            from jang_tools.loader import load_jang_model

            model, tokenizer = load_jang_model(str(model_path))
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        load_s = time.perf_counter() - load_t0

        body, body_tokens = _build_prompt(tokenizer, prompt_tokens)
        prompt, rendered_tokens, rendered = _chat_prompt(tokenizer, body)
        mx.eval(prompt)

        if args.warmup_tokens > 0:
            _run_once(
                model,
                tokenizer,
                prompt,
                max_tokens=args.warmup_tokens,
                prefill_step_size=args.prefill_step_size,
            )
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
            _reset_counters(counters)

        run = _run_once(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
        )
        return {
            "model": summary,
            "mode": mode,
            "env": env_for_mode(mode),
            "prompt": {
                "target_body_tokens": prompt_tokens,
                "body_tokens": body_tokens,
                "rendered_tokens": rendered_tokens,
                "rendered_prefix": rendered[:500],
                "prefill_step_size": args.prefill_step_size,
            },
            "load_s": load_s,
            "timing": run,
            "kernel_counters": _jsonable_counters(counters),
            "process": {
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "memory": _memory_snapshot(),
            },
        }
    finally:
        _restore_env(prior)


def _apply_env_to_mapping(env: dict[str, str], mode: str) -> dict[str, str]:
    updated = dict(env)
    for key, value in env_for_mode(mode).items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value
    updated["PYTHONUNBUFFERED"] = "1"
    return updated


def _run_job(
    *,
    model: Path,
    kind: str,
    mode: str,
    prompt_tokens: int,
    max_tokens: int,
    prefill_step_size: int,
    warmup_tokens: int,
    timeout_s: int | None,
    tmpdir: Path,
) -> dict[str, Any]:
    out_path = tmpdir / (
        f"{model.name}-{kind}-{mode}-{prompt_tokens}.json".replace("/", "_")
    )
    cmd = [
        sys.executable,
        "-m",
        "jang_tools.minimax_kernel_compare",
        "--worker",
        "--model",
        str(model),
        "--kind",
        kind,
        "--mode",
        mode,
        "--prompt-tokens",
        str(prompt_tokens),
        "--max-tokens",
        str(max_tokens),
        "--prefill-step-size",
        str(prefill_step_size),
        "--warmup-tokens",
        str(warmup_tokens),
        "--json-out",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_apply_env_to_mapping(os.environ, mode),
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        return {
            "model": str(model),
            "kind": kind,
            "mode": mode,
            "prompt_tokens": prompt_tokens,
            "error": {
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "cmd": cmd,
            },
        }
    result = json.loads(out_path.read_text())
    result["worker_stdout_tail"] = proc.stdout[-4000:]
    result["worker_stderr_tail"] = proc.stderr[-4000:]
    return result


def _parent_jobs(args: argparse.Namespace) -> list[tuple[Path, str, str, int]]:
    jobs: list[tuple[Path, str, str, int]] = []
    prompts = args.prompt_tokens or [512, 2048]
    for model in args.jangtq or []:
        for mode in mode_labels(JANGTQ_KIND, include_global_auto=args.include_global_auto):
            for prompt_tokens in prompts:
                jobs.append((Path(model), JANGTQ_KIND, mode, int(prompt_tokens)))
    for model in args.jang or []:
        for mode in mode_labels(JANG_KIND):
            for prompt_tokens in prompts:
                jobs.append((Path(model), JANG_KIND, mode, int(prompt_tokens)))
    if args.model:
        kind = args.kind or detect_model_kind(args.model)
        for mode in mode_labels(kind, include_global_auto=args.include_global_auto):
            for prompt_tokens in prompts:
                jobs.append((Path(args.model), kind, mode, int(prompt_tokens)))
    return jobs


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    jobs = _parent_jobs(args)
    if not jobs:
        raise SystemExit("pass --jangtq, --jang, or --model")

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="minimax_kernel_compare_") as tmp:
        tmpdir = Path(tmp)
        for model, kind, mode, prompt_tokens in jobs:
            result = _run_job(
                model=model,
                kind=kind,
                mode=mode,
                prompt_tokens=prompt_tokens,
                max_tokens=args.max_tokens,
                prefill_step_size=args.prefill_step_size,
                warmup_tokens=args.warmup_tokens,
                timeout_s=args.timeout_s,
                tmpdir=tmpdir,
            )
            results.append(result)

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": time.perf_counter() - started,
        "jobs": len(jobs),
        "results": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MiniMax JANGTQ routed kernels with affine JANG in fresh processes."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--jangtq", action="append", type=Path)
    parser.add_argument("--jang", action="append", type=Path)
    parser.add_argument("--kind", choices=[JANGTQ_KIND, JANG_KIND])
    parser.add_argument(
        "--mode",
        choices=list(JANGTQ_MODES) + list(JANG_MODES),
        default="default_prefill",
    )
    parser.add_argument("--prompt-tokens", action="append", type=int)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument("--timeout-s", type=int)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--include-global-auto",
        dest="include_global_auto",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-global-auto",
        dest="include_global_auto",
        action="store_false",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        if args.model is None:
            raise SystemExit("--worker requires --model")
        if args.prompt_tokens is None:
            args.prompt_tokens = [512]
        result = _run_worker(args)
    else:
        result = _run_parent(args)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
