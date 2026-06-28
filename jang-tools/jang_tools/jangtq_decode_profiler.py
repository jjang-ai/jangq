"""JANGTQ decode profiler for proving fast-path use and output sanity.

This module is intentionally opt-in. It wraps the already-installed
``SwitchGLU.__call__`` at profiling time, records whether each routed MoE call
was eligible for the compiled JANGTQ decode fast path, and writes a JSON report
with full generated text. It does not retune kernels or change production
runtime behavior.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SwitchCallClassification:
    fast_path_eligible: bool
    reason: str
    batch: int
    top_k: int
    bits: int | None
    down_bits: int | None


@dataclass
class SwitchCallStats:
    timing_mode: str = "python_call_no_mx_sync"
    total_calls: int = 0
    fast_calls: int = 0
    slow_calls: int = 0
    dispatch_seconds: float = 0.0
    slow_reasons: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)

    def record(
        self,
        fast_path_eligible: bool,
        seconds: float,
        reason: str,
        *,
        batch: int,
        top_k: int,
        bits: int | None,
        down_bits: int | None,
    ) -> None:
        self.total_calls += 1
        self.dispatch_seconds += float(seconds)
        if fast_path_eligible:
            self.fast_calls += 1
        else:
            self.slow_calls += 1
            self.slow_reasons[reason] += 1
        shape_key = f"batch={batch},k={top_k},bits={bits},down_bits={down_bits}"
        self.shape_counts[shape_key] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "timing_mode": self.timing_mode,
            "timing_note": (
                "SwitchGLU call seconds are measured around the Python call. "
                "MLX execution is lazy unless this profile was run with "
                "--sync-switchglu."
            ),
            "total_calls": self.total_calls,
            "fast_calls": self.fast_calls,
            "slow_calls": self.slow_calls,
            "dispatch_seconds": self.dispatch_seconds,
            "slow_reasons": dict(self.slow_reasons),
            "shape_counts": dict(self.shape_counts),
        }


@dataclass
class CoherencySummary:
    status: str
    visible_chars: int
    repeat_4gram_ratio: float
    repeated_line_max: int
    full_chars: int = 0
    reasoning_chars: int = 0
    visible_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return ()


def _prod(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        if value < 0:
            return -1
        out *= value
    return out


def _array_size(value: Any) -> int:
    explicit = getattr(value, "size", None)
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            return _prod(_shape_of(value))
    return _prod(_shape_of(value))


def classify_switchglu_call(
    switch_module: Any,
    x: Any,
    indices: Any,
    *,
    tq_linear_type: type | tuple[type, ...] | None = None,
) -> SwitchCallClassification:
    """Classify a SwitchGLU call using the same shape contract as the fast path.

    The JANGTQ decode path is only valid for a single flattened token with a
    small top-k routing set. Prefill, batched decode, training, and non-TQ
    modules must go through the slow/dynamic path.
    """
    gate_proj = getattr(switch_module, "gate_proj", None)
    up_proj = getattr(switch_module, "up_proj", None)
    down_proj = getattr(switch_module, "down_proj", None)
    bits = getattr(gate_proj, "bits", None)
    down_bits = getattr(down_proj, "bits", None)

    in_features = int(getattr(gate_proj, "in_features", 0) or 0)
    shape = list(_shape_of(x))
    while len(shape) > 2 and shape[-2] == 1:
        del shape[-2]
    batch = _prod(shape[:-1]) if shape else -1
    if batch == -1 and in_features > 0:
        total = _prod(shape)
        if total > 0 and total % in_features == 0:
            batch = total // in_features

    indices_shape = _shape_of(indices)
    indices_ndim = int(getattr(indices, "ndim", len(indices_shape)) or 0)
    top_k = indices_shape[-1] if indices_shape else 1
    indices_size = _array_size(indices)

    if tq_linear_type is not None and (
        not isinstance(gate_proj, tq_linear_type)
        or not isinstance(up_proj, tq_linear_type)
    ):
        return SwitchCallClassification(False, "non-turboquant-switchglu", batch, top_k, bits, down_bits)
    if getattr(switch_module, "training", False):
        return SwitchCallClassification(False, "training", batch, top_k, bits, down_bits)
    if indices_ndim < 1 or top_k <= 0:
        return SwitchCallClassification(False, "invalid-routing-shape", batch, top_k, bits, down_bits)
    if batch != 1 or indices_size >= 64:
        return SwitchCallClassification(False, "prefill-or-large-routing", batch, top_k, bits, down_bits)
    return SwitchCallClassification(True, "decode-fast-path", batch, top_k, bits, down_bits)


def _repeat_ngram_ratio(words: list[str], n: int = 4) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def split_reasoning_visible_text(text: str) -> tuple[str, str, str]:
    """Return `(full, reasoning, visible)` for generated text.

    Some JANGTQ/MiniMax profiles include a reasoning rail before a closing
    `</think>` marker even when the prompt requested thinking off. Keep the
    raw output in the report, but run visible-output coherency checks on the
    post-`</think>` segment so speed acceptance rows don't mistake hidden
    reasoning for user-visible answer text.
    """
    full = (text or "").strip()
    lowered = full.lower()
    close = lowered.rfind("</think>")
    if close >= 0:
        end = close + len("</think>")
        return full, full[:end].strip(), full[end:].strip()
    if "<think" in lowered:
        return full, full, ""
    return full, "", full


def summarize_coherency(text: str) -> CoherencySummary:
    full, reasoning, visible = split_reasoning_visible_text(text)
    visible_chars = len(visible)
    words = visible.split()
    repeat_ratio = _repeat_ngram_ratio(words)
    line_counts = Counter(line.strip() for line in visible.splitlines() if line.strip())
    repeated_line_max = max(line_counts.values(), default=0)

    lowered_full = full.lower()
    if visible_chars == 0 and not full:
        status = "empty"
    elif "<think" in lowered_full and "</think>" not in lowered_full:
        status = "reasoning_only_or_unclosed"
    elif visible_chars == 0 and reasoning:
        status = "reasoning_only_or_unclosed"
    elif repeat_ratio > 0.60 and visible_chars > 80:
        status = "looping_risk"
    elif repeated_line_max >= 4:
        status = "looping_risk"
    else:
        status = "ok"
    return CoherencySummary(
        status,
        visible_chars,
        repeat_ratio,
        repeated_line_max,
        full_chars=len(full),
        reasoning_chars=len(reasoning),
        visible_text=visible,
    )


class SwitchGLUProfiler:
    """Context manager that records SwitchGLU fast/slow-path eligibility."""

    def __init__(self, *, sync_outputs: bool = False) -> None:
        self._sync_outputs = bool(sync_outputs)
        timing_mode = "synchronized_mx_eval" if self._sync_outputs else "python_call_no_mx_sync"
        self.stats = SwitchCallStats(timing_mode=timing_mode)
        self._switch_cls: Any | None = None
        self._orig_call: Any | None = None

    def __enter__(self) -> "SwitchGLUProfiler":
        from mlx_lm.models.switch_layers import SwitchGLU
        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

        self._switch_cls = SwitchGLU
        self._orig_call = SwitchGLU.__call__
        stats = self.stats
        orig_call = self._orig_call
        sync_outputs = self._sync_outputs

        def _profiled_call(module, x, indices):
            classification = classify_switchglu_call(
                module, x, indices, tq_linear_type=TurboQuantSwitchLinear
            )
            t0 = time.perf_counter()
            out = orig_call(module, x, indices)
            if sync_outputs:
                _sync_mlx_output(out)
            stats.record(
                classification.fast_path_eligible,
                time.perf_counter() - t0,
                classification.reason,
                batch=classification.batch,
                top_k=classification.top_k,
                bits=classification.bits,
                down_bits=classification.down_bits,
            )
            return out

        SwitchGLU.__call__ = _profiled_call
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._switch_cls is not None and self._orig_call is not None:
            self._switch_cls.__call__ = self._orig_call


def _sync_mlx_output(value: Any) -> None:
    """Synchronize an MLX output tree for opt-in kernel timing probes."""
    import mlx.core as mx

    if isinstance(value, (list, tuple)):
        for item in value:
            _sync_mlx_output(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _sync_mlx_output(item)
        return
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        mx.eval(value)


def read_model_info(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    cfg = _read_json(path / "config.json")
    jang_cfg = _read_json(path / "jang_config.json")
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    root = text_cfg or cfg
    return {
        "path": str(path),
        "model_type": root.get("model_type") or cfg.get("model_type"),
        "architectures": root.get("architectures") or cfg.get("architectures"),
        "hidden_size": root.get("hidden_size") or cfg.get("hidden_size"),
        "num_hidden_layers": root.get("num_hidden_layers") or cfg.get("num_hidden_layers"),
        "num_experts": (
            root.get("num_local_experts")
            or root.get("n_routed_experts")
            or cfg.get("num_local_experts")
            or cfg.get("num_experts")
        ),
        "num_experts_per_tok": root.get("num_experts_per_tok") or cfg.get("num_experts_per_tok"),
        "sliding_window": root.get("sliding_window") or cfg.get("sliding_window"),
        "hybrid_override_pattern": root.get("hybrid_override_pattern") or cfg.get("hybrid_override_pattern"),
        "weight_format": cfg.get("weight_format") or jang_cfg.get("weight_format"),
        "mxtq_bits": cfg.get("mxtq_bits") or jang_cfg.get("mxtq_bits"),
    }


def loader_kind_for_model_info(model_info: dict[str, Any]) -> str:
    model_type = str(model_info.get("model_type") or "").lower()
    if model_type == "laguna":
        return "laguna-runtime"
    return "generic-jangtq"


def _load_model_and_tokenizer(model_path: str, model_info: dict[str, Any]):
    loader_kind = loader_kind_for_model_info(model_info)
    if loader_kind == "laguna-runtime":
        from transformers import AutoTokenizer
        from jang_tools.laguna.runtime import load as load_laguna

        model, _cfg, _fmt = load_laguna(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer, loader_kind

    from jang_tools.load_jangtq import load_jangtq_model

    model, tokenizer = load_jangtq_model(model_path)
    return model, tokenizer, loader_kind


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_profile_report(
    path: str | Path,
    *,
    model_path: str,
    prompt: str,
    output_text: str,
    prompt_tokens: int,
    generated_tokens: int,
    total_seconds: float,
    decode_seconds: float,
    switch_stats: SwitchCallStats,
    coherency: CoherencySummary,
    step_records: list[dict[str, Any]],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    decode_tps = generated_tokens / max(decode_seconds, 1e-9)
    total_tps = generated_tokens / max(total_seconds, 1e-9)
    finish_reason = None
    for step in reversed(step_records):
        if step.get("finish_reason"):
            finish_reason = step["finish_reason"]
            break
    acceptance_status = "pass"
    acceptance_notes: list[str] = []
    if finish_reason == "length":
        acceptance_status = "incomplete_length"
        acceptance_notes.append("generation stopped by max_tokens before model stop/eos")
    if coherency.status != "ok":
        acceptance_status = coherency.status
        acceptance_notes.append(f"visible-output coherency status is {coherency.status}")
    report = {
        "schema": "jangtq_decode_profile.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": {"path": model_path, **model_info},
        "prompt": prompt,
        "finish_reason": finish_reason,
        "speed": {
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "total_seconds": float(total_seconds),
            "decode_seconds": float(decode_seconds),
            "decode_tokens_per_second": decode_tps,
            "generated_tokens_per_second_total_wall": total_tps,
        },
        "switchglu": switch_stats.to_dict(),
        "coherency": coherency.to_dict(),
        "acceptance": {
            "status": acceptance_status,
            "requires_review": acceptance_status != "pass",
            "notes": acceptance_notes,
        },
        "steps": step_records,
        "output_text": output_text,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _format_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    except Exception:
        return prompt


def profile_generate(
    *,
    model_path: str,
    prompt: str,
    max_tokens: int,
    output: str | Path,
    use_chat_template: bool = True,
    temperature: float = 0.0,
    sync_switchglu: bool = False,
) -> dict[str, Any]:
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model_info = read_model_info(model_path)
    load_t0 = time.perf_counter()
    model, tokenizer, loader_kind = _load_model_and_tokenizer(model_path, model_info)
    load_seconds = time.perf_counter() - load_t0

    rendered_prompt = _format_prompt(tokenizer, prompt, use_chat_template)
    prompt_ids = tokenizer.encode(rendered_prompt)
    sampler = make_sampler(temp=temperature)

    output_parts: list[str] = []
    step_records: list[dict[str, Any]] = []
    generation_started_at: float | None = None
    total_t0 = time.perf_counter()
    finish_reason: str | None = None
    prompt_tps = 0.0

    with SwitchGLUProfiler(sync_outputs=sync_switchglu) as profiler:
        previous = time.perf_counter()
        for index, response in enumerate(
            stream_generate(
                model,
                tokenizer,
                rendered_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ),
            start=1,
        ):
            now = time.perf_counter()
            if generation_started_at is None:
                generation_started_at = now
            text = response.text or ""
            output_parts.append(text)
            finish_reason = response.finish_reason or finish_reason
            prompt_tps = float(response.prompt_tps or prompt_tps or 0.0)
            step_records.append(
                {
                    "index": index,
                    "token": int(response.token),
                    "text": text,
                    "delta_seconds": now - previous,
                    "cumulative_generation_tps": float(response.generation_tps or 0.0),
                    "finish_reason": response.finish_reason,
                }
            )
            previous = now

    total_seconds = time.perf_counter() - total_t0
    decode_seconds = (
        max(0.0, time.perf_counter() - generation_started_at)
        if generation_started_at is not None
        else 0.0
    )
    output_text = "".join(output_parts)
    coherency = summarize_coherency(output_text)
    report = write_profile_report(
        output,
        model_path=model_path,
        prompt=rendered_prompt,
        output_text=output_text,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(step_records),
        total_seconds=total_seconds,
        decode_seconds=decode_seconds,
        switch_stats=profiler.stats,
        coherency=coherency,
        step_records=step_records,
        model_info={
            **model_info,
            "loader_kind": loader_kind,
            "load_seconds": load_seconds,
            "prompt_tokens_per_second": prompt_tps,
        },
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile JANGTQ decode fast-path use and output sanity.")
    parser.add_argument("model", help="Path to a local JANGTQ model directory")
    parser.add_argument("--prompt", default="Answer only the final number: what is 17 plus 28?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--raw-prompt", action="store_true", help="Do not apply tokenizer chat template")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--sync-switchglu",
        action="store_true",
        help=(
            "Synchronize each profiled SwitchGLU output with mx.eval before "
            "recording elapsed time. This is a kernel probe and changes decode "
            "timing; default timing is Python/lazy-dispatch only."
        ),
    )
    args = parser.parse_args(argv)

    report = profile_generate(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        output=args.output,
        use_chat_template=not args.raw_prompt,
        temperature=args.temperature,
        sync_switchglu=args.sync_switchglu,
    )
    speed = report["speed"]
    print(
        f"[jangtq-profile] generated={speed['generated_tokens']} "
        f"decode_tps={speed['decode_tokens_per_second']:.2f} "
        f"coherency={report['coherency']['status']} "
        f"acceptance={report['acceptance']['status']} "
        f"fast={report['switchglu']['fast_calls']} slow={report['switchglu']['slow_calls']} "
        f"report={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
