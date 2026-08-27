"""Streaming SOURCE capture for dots3-note: one pass over the fp8 source.

Produces, under <out_dir>:
  x1/layer_NN.npy          f16 [rows, 5120]  post-post_attention_layernorm rows
  router/layer_NN.npz      inds u16 [rows,8], weights f16 [rows,8]
  moments.npz              per-layer per-channel E[x^2] for attn_in / mlp_in
  source_logprobs.npz      eval-slice top-K source logprobs (KL reference)
  manifest.json            corpus + accounting

Corpus: token-balanced subsample of the 2026-04-23 standard mix
(kimi_v3_calib/corpus_v3.jsonl) + chat-templated agentic/tool prompts
(thinking ON, dots XML tools) + a HELD-OUT eval slice for the KL gate.
All sequences <= 2040 tokens => DSA top-2048 is dense-equivalent.

    python -m jang_tools.dots3.capture_dots3 <src_model> <out_dir> \
        [--tokens 100000] [--eval-seqs 24]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

CORPUS = Path.home() / ".cache" / "jang" / "corpus_v3.jsonl"
SEQ_LEN = 2040
TOPK_LOGPROBS = 256

AGENTIC_PROMPTS = [
    "Refactor the function `parse_manifest_v2` in loader.py so that it "
    "streams entries instead of reading the whole file; keep the public "
    "signature identical and update the two call sites.",
    "Debug this: pytest fails with `AssertionError: expected 42 rows, got 41` "
    "in test_ledger_rollup only when TZ=America/New_York. Walk through the "
    "root cause and produce a minimal patch.",
    "Write a Rust CLI that tails a JSONL file and exposes a Prometheus "
    "endpoint with per-key rates. Include Cargo.toml.",
    "Plan and execute: rename config key `max_batch` to `max_batch_size` "
    "across a repo with 60 references, preserving backwards compat for one "
    "release. List every file class you would touch.",
    "使用二分查找在旋转有序数组中找目标值，先给出思路再写出带注释的 Python 实现，并分析边界情况。",
    "Prove that the sum of the first n odd numbers is n^2, then write a "
    "property-based test for it with hypothesis.",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Execute a shell command in the project workspace and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout_s": {"type": "integer", "description": "Timeout seconds"}},
            "required": ["command"]},
    },
}, {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]},
    },
}]

TOOL_PROMPTS = [
    "Find every TODO in the src/ directory, then read the file with the most "
    "of them and summarize what remains to be done.",
    "Check whether the test suite passes; if anything fails, read the failing "
    "test file and propose a fix.",
]


def build_sequences(tok, target_tokens: int, eval_seqs: int, seed: int = 20260814):
    rng = random.Random(seed)
    by_domain: dict[str, list[str]] = defaultdict(list)
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            by_domain[r.get("domain", "general")].append(r["text"])
    mix = {"coding": .24, "agentic": .20, "general": .20, "academic_mc": .10,
           "science": .08, "chinese": .08, "cybersec": .05, "systems": .03,
           "longctx": .02}
    seqs: list[tuple[str, list[int]]] = []

    def pack(domain, texts, budget):
        got = 0
        buf: list[int] = []
        rng.shuffle(texts)
        for t in texts:
            ids = tok(t, add_special_tokens=False).input_ids
            buf.extend(ids + [tok.eos_token_id or 151668])
            while len(buf) >= SEQ_LEN:
                seqs.append((domain, buf[:SEQ_LEN]))
                got += SEQ_LEN
                buf = buf[SEQ_LEN:]
                if got >= budget:
                    return got
        if buf and got < budget:
            if len(buf) >= 256:
                seqs.append((domain, buf))
                got += len(buf)
        return got

    total = 0
    for d, frac in mix.items():
        texts = by_domain.get(d, [])
        if not texts:
            continue
        total += pack(d, texts, int(target_tokens * frac))

    # chat-templated agentic prompts, thinking ON (default) + tools
    for p in AGENTIC_PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False)
        seqs.append(("template", tok(s, add_special_tokens=False).input_ids))
    for p in TOOL_PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": p}],
                                    tools=TOOLS, add_generation_prompt=True,
                                    tokenize=False)
        seqs.append(("template_tools", tok(s, add_special_tokens=False).input_ids))
    # one non-thinking render so the instruct path is represented
    s = tok.apply_chat_template(
        [{"role": "user", "content": AGENTIC_PROMPTS[0]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    seqs.append(("template_nothink", tok(s, add_special_tokens=False).input_ids))

    # held-out eval slice (KL reference) — general+coding+chinese, unseen recs
    eval_pool = (by_domain.get("general", [])[-400:] +
                 by_domain.get("coding", [])[-400:] +
                 by_domain.get("chinese", [])[-200:])
    rng.shuffle(eval_pool)
    evals = []
    for t in eval_pool[: eval_seqs]:
        ids = tok(t, add_special_tokens=False).input_ids[:768]
        if len(ids) >= 128:
            evals.append(("eval", ids))
    return seqs, evals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--eval-seqs", type=int, default=24)
    ap.add_argument("--x1-max-rows", type=int, default=120_000)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from .stream import StreamModel

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "x1").mkdir(exist_ok=True)
    (a.out / "router").mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.src)
    seqs, evals = build_sequences(tok, a.tokens, a.eval_seqs)
    n_cal_tokens = sum(len(s[1]) for s in seqs)
    n_eval_tokens = sum(len(s[1]) for s in evals)
    print(f"corpus: {len(seqs)} calib seqs ({n_cal_tokens} tok), "
          f"{len(evals)} eval seqs ({n_eval_tokens} tok)")

    sm = StreamModel(a.src)
    cfg = sm.cfg
    all_seqs = seqs + evals
    n_eval = len(evals)
    states = sm.embed([ids for _, ids in all_seqs])

    L = cfg.num_hidden_layers
    H = cfg.hidden_size
    mom_attn = np.zeros((L, H), np.float64)
    mom_mlp = np.zeros((L, H), np.float64)
    counts = np.zeros(L, np.int64)
    x1_rows: list[np.ndarray] = []
    r_inds: list[np.ndarray] = []
    r_w: list[np.ndarray] = []
    state = {"layer": -1}

    def flush_layer(i: int):
        if x1_rows:
            x1 = np.concatenate(x1_rows, 0)
            if x1.shape[0] > a.x1_max_rows:
                keep = np.random.default_rng(i).choice(
                    x1.shape[0], a.x1_max_rows, replace=False)
                keep.sort()
                x1 = x1[keep]
                if r_inds:
                    ri = np.concatenate(r_inds, 0)[keep]
                    rw = np.concatenate(r_w, 0)[keep]
            elif r_inds:
                ri = np.concatenate(r_inds, 0)
                rw = np.concatenate(r_w, 0)
            np.save(a.out / "x1" / f"layer_{i:02d}.npy", x1)
            if r_inds:
                np.savez(a.out / "router" / f"layer_{i:02d}.npz",
                         inds=ri, weights=rw)
            x1_rows.clear(); r_inds.clear(); r_w.clear()

    def cb(i: int, j: int, cap: dict):
        if state["layer"] != i:
            if state["layer"] >= 0:
                flush_layer(state["layer"])
            state["layer"] = i
        is_eval = n_eval > 0 and j >= len(all_seqs) - n_eval
        if is_eval:
            return          # eval slice stays fully held out of all fitting
        attn_in = np.asarray(cap["attn_in"].astype(mx.float32))[0]
        mlp_in = np.asarray(cap["mlp_in"].astype(mx.float32))[0]
        mom_attn[i] += (attn_in.astype(np.float64) ** 2).sum(0)
        mom_mlp[i] += (mlp_in.astype(np.float64) ** 2).sum(0)
        counts[i] += mlp_in.shape[0]
        if True:
            x1_rows.append(mlp_in.astype(np.float16))
            if "router_inds" in cap:
                r_inds.append(np.asarray(cap["router_inds"]).astype(np.uint16))
                r_w.append(np.asarray(cap["router_weights"]).astype(np.float16))

    t0 = time.time()
    states = sm.forward_all(states, layer_cb=cb)
    flush_layer(state["layer"])
    np.savez(a.out / "moments.npz", attn_in=mom_attn, mlp_in=mom_mlp,
             counts=counts)

    # source logprobs on the eval slice
    if n_eval:
        ids_list, tops_i, tops_lp, lse = [], [], [], []
        for k in range(n_eval):
            hidden = states[len(seqs) + k]
            logits = sm.head_logits(hidden)[0]          # [S, V] f32
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            top_i = mx.argpartition(-lp, kth=TOPK_LOGPROBS - 1,
                                    axis=-1)[:, :TOPK_LOGPROBS]
            top_lp = mx.take_along_axis(lp, top_i, axis=-1)
            order = mx.argsort(-top_lp, axis=-1)      # sorted: [0] is argmax
            top_i = mx.take_along_axis(top_i, order, axis=-1)
            top_lp = mx.take_along_axis(top_lp, order, axis=-1)
            mx.eval(top_i, top_lp)
            tops_i.append(np.asarray(top_i).astype(np.int32))
            tops_lp.append(np.asarray(top_lp).astype(np.float32))
            ids_list.append(np.array(all_seqs[len(seqs) + k][1], np.int32))
        np.savez(a.out / "source_logprobs.npz",
                 **{f"ids_{k}": ids_list[k] for k in range(n_eval)},
                 **{f"top_i_{k}": tops_i[k] for k in range(n_eval)},
                 **{f"top_lp_{k}": tops_lp[k] for k in range(n_eval)})

    manifest = {
        "calib_seqs": len(seqs), "calib_tokens": n_cal_tokens,
        "eval_seqs": n_eval, "eval_tokens": n_eval_tokens,
        "domains": [d for d, _ in seqs][:64],
        "x1_max_rows": a.x1_max_rows, "topk_logprobs": TOPK_LOGPROBS,
        "elapsed_s": round(time.time() - t0, 1),
        "corpus": str(CORPUS),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"capture complete in {(time.time()-t0)/60:.1f} min -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
