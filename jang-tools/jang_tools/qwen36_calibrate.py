"""Calibration capture for Qwen3.6-27B (qwen3_5) — one pass, four uses.

Records the **per-input-channel second moment** `E[x_c^2]` for every Linear in
the model. That single statistic serves simultaneously as:

  * **AWQ** salient-channel scales
  * **imatrix** activation weighting
  * the **Hessian diagonal**, since `tr(H) = sum_c E[x_c^2]`
    (see docs/internal/_method/hessian-trace-allocation.md)
  * a per-module **sensitivity score** `tr(H) * ||W||_F^2` for bit allocation

Implementation note: MLX resolves `module(x)` through `type(module).__call__`,
so an instance attribute cannot shadow it. We patch `nn.Linear.__call__` (and
`QuantizedLinear` if present) at class level and dispatch on `id(module)`, which
keeps the hook exact and removable.

Accumulates in float64 to avoid drift over long corpora. Memory is trivial —
one vector of `in_features` per module (607 modules, max 17408 wide).

    python -m jang_tools.qwen36_calibrate <model_dir> <out.safetensors> \
        [--limit N] [--max-tokens N] [--images dir] \
        [--corpus corpus.jsonl] [--corpus-tokens N] [--corpus-gen-tokens N]

`--corpus` draws real domain-weighted text on top of the built-in prompt set.
Volume matters for more than averaging quality: GPTQ needs `tokens >> d_in` or
its Hessian is rank-deficient and the solve silently degrades to RTN. The
built-in prompts alone give ~7 k tokens against a 17408-wide `down_proj`
(0.4x — singular); ~140 k is the 8x floor for this model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# id(module) -> dotted path, for modules we want to record.
_TARGETS: dict[int, str] = {}
_SUMSQ: dict[str, np.ndarray] = {}
_COUNT: dict[str, int] = {}
_PATCHED: list[tuple[type, object]] = []


def _accumulate(path: str, x: mx.array) -> None:
    xf = x.reshape(-1, x.shape[-1])
    s = (xf.astype(mx.float32) ** 2).sum(axis=0)
    mx.eval(s)
    v = np.array(s, dtype=np.float64)
    if path in _SUMSQ:
        _SUMSQ[path] += v
    else:
        _SUMSQ[path] = v
    _COUNT[path] = _COUNT.get(path, 0) + int(xf.shape[0])


def install_hooks(model, include_vision: bool = True) -> int:
    """Patch Linear.__call__ at class level; register target modules by id."""
    classes = [nn.Linear]
    q = getattr(nn, "QuantizedLinear", None)
    if q is not None:
        classes.append(q)

    for cls in classes:
        orig = cls.__call__

        def make(orig=orig):
            def patched(self, x, *a, **k):
                p = _TARGETS.get(id(self))
                if p is not None:
                    try:
                        _accumulate(p, x)
                    except Exception:  # never let capture break the forward
                        pass
                return orig(self, x, *a, **k)
            return patched

        _PATCHED.append((cls, orig))
        cls.__call__ = make()

    n = 0
    for path, mod in model.named_modules():
        if not isinstance(mod, tuple(classes)):
            continue
        if not include_vision and path.startswith("vision_tower"):
            continue
        _TARGETS[id(mod)] = path
        n += 1
    return n


def remove_hooks() -> None:
    for cls, orig in _PATCHED:
        cls.__call__ = orig
    _PATCHED.clear()
    _TARGETS.clear()


# ── calibration corpus ───────────────────────────────────────────────────────
# Matches the DISTRIBUTION THE MODEL IS FOR: thinking traces, agentic coding,
# tool use, long-context recall. Calibrating on bare completions would optimise
# for the wrong thing (the LFM2.5 lesson: in-domain KL came out 2.5x better
# than vendor precisely because the corpus matched).
PROMPTS_REASONING = [
    "A train leaves at 3:15pm travelling 82 km/h. A second leaves 40 minutes later at 110 km/h. When does it catch up? Show your reasoning.",
    "Prove that the square root of 3 is irrational.",
    "A bag has 4 red, 6 blue and 5 green marbles. Two are drawn without replacement. What is P(same colour)?",
    "Explain why gradient descent with momentum converges faster than plain SGD on ill-conditioned problems.",
    "If every A is B, and some B are C, does it follow that some A are C? Explain carefully.",
]
PROMPTS_CODING = [
    "Write a Python function that merges overlapping intervals, then explain its complexity.",
    "Refactor this to remove the nested loop:\n\nfor i in range(n):\n    for j in range(n):\n        if a[i]==b[j]: out.append((i,j))",
    "Implement a thread-safe LRU cache in Python with O(1) get and put.",
    "Find the bug:\n\ndef binsearch(a,t):\n    lo,hi=0,len(a)\n    while lo<hi:\n        m=(lo+hi)//2\n        if a[m]<t: lo=m\n        else: hi=m\n    return lo",
    "Write a SQL query returning the second-highest salary per department, handling ties.",
]
PROMPTS_TOOLS = [
    "What's the weather in Santa Clara and should I bring an umbrella?",
    "Search the repository for every call site of `parse_config` and summarise them.",
    "Read the file at /etc/hosts and tell me which domains are redirected.",
]
PROMPTS_GENERAL = [
    "Summarise the causes of the 1873 financial panic in three paragraphs.",
    "Explain the difference between a B-tree and an LSM tree for storage engines.",
    "Translate to French and explain any idioms: 'Don't count your chickens before they hatch.'",
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


IMAGE_PROMPTS = [
    "Describe this image in detail, including colours and layout.",
    "What text appears in this image? Transcribe it exactly.",
    "Read the chart and describe the trend.",
    "List every distinct shape and where it sits.",
]


# Domain weights for corpus draw. Coding-weighted (calibration mix v4) because
# the speed-targeted bundles are coding/agent artifacts; a corpus that does not
# match the serving distribution optimises for the wrong thing.
CORPUS_MIX = {
    "coding": 0.35, "agentic": 0.20, "academic_mc": 0.15, "general": 0.12,
    "chinese": 0.10, "longctx": 0.04, "science": 0.02, "cybersec": 0.02,
}


def build_text_corpus(tokenizer, corpus_path: Path, target_tokens: int,
                      mix: dict[str, float] | None = None,
                      max_prompt_tokens: int = 2048):
    """Draw ~`target_tokens` of real corpus text, domain-weighted, chat-rendered.

    The hardcoded PROMPTS_* above give the right *shape* (thinking traces, tool
    frames, a non-thinking slice) but only ~7 k tokens, which leaves the Hessian
    of a 17408-wide `down_proj` badly rank-deficient — GPTQ on that silently
    falls back to RTN with no error. This draws the volume from a real corpus
    while keeping the same chat rendering, so both properties hold at once.

    Records are interleaved round-robin across domains rather than drawn
    domain-by-domain, so a truncated run still carries the whole mix.
    """
    mix = mix or CORPUS_MIX
    by_domain: dict[str, list[str]] = {}
    with corpus_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (rec.get("text") or "").strip()
            if text:
                by_domain.setdefault(rec.get("domain", "general"), []).append(text)

    budgets = {d: int(target_tokens * w) for d, w in mix.items() if d in by_domain}
    queues = {d: iter(by_domain[d]) for d in budgets}
    spent = {d: 0 for d in budgets}
    items, total = [], 0
    while queues:
        for d in list(queues):
            if spent[d] >= budgets[d]:
                queues.pop(d)
                continue
            try:
                text = next(queues[d])
            except StopIteration:
                queues.pop(d)
                continue
            ids = tokenizer.encode(text)
            if len(ids) > max_prompt_tokens:
                text = tokenizer.decode(ids[:max_prompt_tokens])
                n = max_prompt_tokens
            else:
                n = len(ids)
            items.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False, enable_thinking=True))
            spent[d] += n
            total += n
    return items, total, spent


def build_image_corpus(proc, model, image_dir: Path):
    """Vision-tower calibration. Video reuses these same Linears (only the
    temporal patching upstream differs), so images cover the tower."""
    from mlx_vlm.prompt_utils import apply_chat_template
    imgs = sorted(str(p) for p in image_dir.glob("*.png"))
    out = []
    for i, img in enumerate(imgs):
        prompt = IMAGE_PROMPTS[i % len(IMAGE_PROMPTS)]
        p = apply_chat_template(proc, model.config,
                                [{"role": "user", "content": prompt}], num_images=1)
        out.append((p, img))
    return out


def build_corpus(tokenizer, limit: int | None = None):
    """Render prompts through the REAL chat template, thinking ON (the default)."""
    items = []
    for p in PROMPTS_REASONING + PROMPTS_CODING + PROMPTS_GENERAL:
        items.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False, enable_thinking=True))
    for p in PROMPTS_TOOLS:
        items.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tools=TOOLS,
            add_generation_prompt=True, tokenize=False, enable_thinking=True))
    # A non-thinking slice too, so the instruct preset is represented.
    for p in PROMPTS_GENERAL:
        items.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False))
    return items[:limit] if limit else items


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    src, out = Path(argv[1]), Path(argv[2])
    limit = None
    max_tokens = 96
    image_dir = None
    corpus_path = None
    corpus_tokens = 0
    corpus_gen_tokens = 8
    for i, a in enumerate(argv):
        if a == "--limit":
            limit = int(argv[i + 1])
        if a == "--max-tokens":
            max_tokens = int(argv[i + 1])
        if a == "--images":
            image_dir = Path(argv[i + 1])
        if a == "--corpus":
            corpus_path = Path(argv[i + 1])
        if a == "--corpus-tokens":
            corpus_tokens = int(argv[i + 1])
        if a == "--corpus-gen-tokens":
            corpus_gen_tokens = int(argv[i + 1])

    from mlx_vlm import load, generate

    print(f"  loading {src.name} ...", flush=True)
    t0 = time.time()
    model, proc = load(str(src))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n = install_hooks(model, include_vision=True)
    print(f"  hooked {n} Linear modules", flush=True)

    corpus = build_corpus(proc.tokenizer, limit)
    print(f"  corpus: {len(corpus)} prompts, generating {max_tokens} tok each "
          f"(captures BOTH prefill and decode activations)", flush=True)

    t0 = time.time()
    for i, prompt in enumerate(corpus, 1):
        generate(model, proc, prompt, max_tokens=max_tokens,
                 temperature=1.0, verbose=False)
        if i % 3 == 0 or i == len(corpus):
            print(f"    text {i}/{len(corpus)}  ({time.time()-t0:.0f}s, "
                  f"{len(_SUMSQ)} modules seen)", flush=True)

    corpus_meta = None
    if corpus_path and corpus_tokens > 0:
        extra, drawn, per_domain = build_text_corpus(
            proc.tokenizer, corpus_path, corpus_tokens)
        corpus_meta = {"path": str(corpus_path), "records": len(extra),
                       "prompt_tokens": drawn, "per_domain": per_domain,
                       "gen_tokens_each": corpus_gen_tokens}
        print(f"  corpus draw: {len(extra)} records, {drawn:,} prompt tokens "
              f"({per_domain})", flush=True)
        t0 = time.time()
        for i, prompt in enumerate(extra, 1):
            generate(model, proc, prompt, max_tokens=corpus_gen_tokens,
                     temperature=1.0, verbose=False)
            if i % 25 == 0 or i == len(extra):
                print(f"    corpus {i}/{len(extra)}  ({time.time()-t0:.0f}s)",
                      flush=True)

    if image_dir and image_dir.is_dir():
        img_corpus = build_image_corpus(proc, model, image_dir)
        print(f"  vision: {len(img_corpus)} images", flush=True)
        for i, (prompt, img) in enumerate(img_corpus, 1):
            generate(model, proc, prompt, image=[img], max_tokens=max_tokens,
                     temperature=1.0, verbose=False)
            print(f"    image {i}/{len(img_corpus)} {Path(img).name} "
                  f"({time.time()-t0:.0f}s, {len(_SUMSQ)} modules seen)", flush=True)

    remove_hooks()

    # ── emit: second moment per channel + trace + module metadata ────────
    tensors, meta = {}, {}
    for path, ssq in _SUMSQ.items():
        cnt = max(_COUNT[path], 1)
        second_moment = (ssq / cnt).astype(np.float32)   # E[x_c^2]
        tensors[f"{path}.second_moment"] = second_moment
        meta[path] = {"count": cnt,
                      "trace": float(second_moment.sum()),   # tr(H)
                      "in_features": int(second_moment.shape[0])}

    from safetensors.numpy import save_file
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    (out.with_suffix(".json")).write_text(json.dumps(
        {"source": str(src), "modules": len(meta), "prompts": len(corpus),
         "max_tokens": max_tokens, "corpus": corpus_meta, "stats": meta}, indent=1))

    print(f"\n  captured {len(meta)} modules -> {out}")
    print(f"  sidecar  -> {out.with_suffix('.json')}")
    tot = sum(v["count"] for v in meta.values())
    print(f"  total row-samples accumulated: {tot:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
