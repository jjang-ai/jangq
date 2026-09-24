"""Small DSV4 generation runtime used by examples and smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerateOptions:
    mode: str = "chat"
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    drop_thinking: bool = True


@dataclass
class GenerateResult:
    raw: str = ""
    content: str = ""
    reasoning_content: str = ""
    token_ids: list[int] | None = None
    n_tokens: int = 0
    tok_s: float = 0.0
    finish_reason: str = "length"
    error: str | None = None


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "_tokenizer", tokenizer)


def _encode(tokenizer: Any, text: str) -> list[int]:
    tok = _unwrap_tokenizer(tokenizer)
    if hasattr(tok, "encode"):
        try:
            return list(tok.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tok.encode(text))
    raise TypeError(f"tokenizer {type(tokenizer)!r} has no encode()")


def _decode(tokenizer: Any, ids: list[int]) -> str:
    tok = _unwrap_tokenizer(tokenizer)
    if not ids:
        return ""
    return tok.decode(ids)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _collect_eos_ids(tokenizer: Any, model_path: str | Path) -> set[int]:
    eos: set[int] = set()
    for attr in ("eos_token_ids", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            eos.add(value)
        elif isinstance(value, (list, tuple, set)):
            eos.update(int(x) for x in value if isinstance(x, int))

    root = Path(model_path)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        value = _load_json(root / name).get("eos_token_id")
        if isinstance(value, int):
            eos.add(value)
        elif isinstance(value, list):
            eos.update(int(x) for x in value if isinstance(x, int))
    return eos


def _thinking_mode(opts: GenerateOptions) -> str:
    if opts.enable_thinking is not None:
        return "thinking" if opts.enable_thinking else "chat"
    if opts.mode in ("think", "thinking", "think_max"):
        return "thinking"
    return "chat"


def _reasoning_effort(opts: GenerateOptions) -> str | None:
    if opts.reasoning_effort is not None:
        return opts.reasoning_effort
    if opts.mode == "think_max":
        return "max"
    return None


def _build_prompt(messages: list[dict[str, Any]], opts: GenerateOptions) -> tuple[str, str]:
    from jang_tools.dsv4.encoding_adapter import apply_chat_template

    thinking_mode = _thinking_mode(opts)
    prompt = apply_chat_template(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=opts.drop_thinking,
        reasoning_effort=_reasoning_effort(opts),
    )
    return prompt, thinking_mode


def _logits_array(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def _sample_next(logits: Any, temperature: float) -> Any:
    import mlx.core as mx

    last = logits[:, -1, :]
    if temperature <= 0:
        return mx.argmax(last, axis=-1)
    return mx.random.categorical(last / temperature, axis=-1)


def _inject_chat_template(tokenizer: Any, model_path: str | Path) -> bool:
    """Attach tokenizer_config.json::chat_template when a wrapper missed it."""
    tok = _unwrap_tokenizer(tokenizer)
    if getattr(tok, "chat_template", None):
        return False
    cfg = _load_json(Path(model_path) / "tokenizer_config.json")
    template = cfg.get("chat_template")
    if not template:
        return False
    try:
        tok.chat_template = template
    except Exception:
        return False
    return True


def generate(
    model: Any,
    tokenizer: Any,
    model_path: str | Path,
    *,
    messages: list[dict[str, Any]],
    opts: GenerateOptions | None = None,
) -> GenerateResult:
    import mlx.core as mx

    opts = opts or GenerateOptions()
    _inject_chat_template(tokenizer, model_path)
    eos_ids = _collect_eos_ids(tokenizer, model_path)

    try:
        prompt, thinking_mode = _build_prompt(messages, opts)
        prompt_ids = _encode(tokenizer, prompt)
        if not prompt_ids:
            return GenerateResult(error="empty prompt after DSV4 encoding")

        cache = model.make_cache() if hasattr(model, "make_cache") else None
        x = mx.array([prompt_ids], dtype=mx.int32)
        t0 = time.time()
        logits = _logits_array(model(x, cache=cache) if cache is not None else model(x))

        out_ids: list[int] = []
        finish = "length"
        for _ in range(opts.max_tokens):
            next_arr = _sample_next(logits, opts.temperature)
            mx.eval(next_arr)
            next_id = int(next_arr.reshape(-1)[0].item())
            if next_id in eos_ids:
                finish = "stop"
                break
            out_ids.append(next_id)
            cur = mx.array([[next_id]], dtype=mx.int32)
            logits = _logits_array(model(cur, cache=cache) if cache is not None else model(cur))

        mx.synchronize()
        dt = max(time.time() - t0, 1e-9)
        raw = _decode(tokenizer, out_ids)
        content = raw
        reasoning = ""
        try:
            from jang_tools.dsv4.encoding_adapter import parse_completion

            parsed = parse_completion(raw, thinking_mode=thinking_mode)
            content = parsed.get("content") or raw
            reasoning = parsed.get("reasoning_content") or ""
        except Exception as exc:  # noqa: BLE001 — best-effort-parse fallback
            logger.warning("DSV4 parse_completion failed, falling back to raw: %s", exc)

        return GenerateResult(
            raw=raw,
            content=content,
            reasoning_content=reasoning,
            token_ids=out_ids,
            n_tokens=len(out_ids),
            tok_s=len(out_ids) / dt,
            finish_reason=finish,
        )
    except Exception as exc:
        return GenerateResult(error=f"{type(exc).__name__}: {exc}")
