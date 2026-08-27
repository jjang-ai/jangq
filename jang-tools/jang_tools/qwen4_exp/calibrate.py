"""Calibration capture for Qwen3.8-Flash-Next (qwen4_exp) — one streamed pass
feeds ALL of: Hessian-trace allocation, imatrix refit, AWQ scales, GPTQ.

Captured per module (keyed by parameter path):
  diag   — per-input-channel second moment (fp64 accumulate → fp32 emit)
  amax   — per-input-channel running |x| max (AWQ)
  rows   — token count seen
Extra:
  per-expert stats for switch_mlp.{gate,up,down} via the SwitchLinear tap
  (indices are visible there → routing counts + per-expert down diag)
  full XᵀX for trunk linears (d_in ≤ 6144) and ONE shared gate_up Hessian
  per layer (input is identical across the 512 experts → rank = tokens/2560)

Run AFTER the bf16 coherence probe passes:
  python -m jang_tools.qwen4_exp.calibrate --model ~/models/Qwen3.8-Flash-Next \
      --corpus ~/jang/kimi_v3_calib/corpus_v3.jsonl \
      --out ~/models/Logs/q38fn-calib/capture --target-tokens 1200000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.switch_layers import SwitchLinear

from .load import load_model
from .ngram_histogram import iter_texts

# modules that get a full XᵀX (GPTQ). Everything else: diagonal only.
FULL_H_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
    "mlp.switch_mlp.gate_proj",  # shared-input H, one per layer, covers up too
)


class Stats:
    def __init__(self):
        self.diag: dict[str, np.ndarray] = {}
        self.amax: dict[str, np.ndarray] = {}
        self.rows: dict[str, int] = {}
        self.full_h: dict[str, np.ndarray] = {}
        self.expert_diag: dict[str, np.ndarray] = {}   # [E, d_in]
        self.expert_rows: dict[str, np.ndarray] = {}   # [E]

    def add(self, path: str, x: mx.array, want_full: bool):
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        sq = np.asarray((flat * flat).sum(axis=0)).astype(np.float64)
        am = np.asarray(mx.abs(flat).max(axis=0))
        n = flat.shape[0]
        if path not in self.diag:
            self.diag[path] = sq
            self.amax[path] = am
            self.rows[path] = n
        else:
            self.diag[path] += sq
            np.maximum(self.amax[path], am, out=self.amax[path])
            self.rows[path] += n
        if want_full:
            h = np.asarray(flat.T @ flat).astype(np.float64)
            if path not in self.full_h:
                self.full_h[path] = h
            else:
                self.full_h[path] += h

    def add_expert(self, path: str, x_flat: mx.array, inds: np.ndarray, n_exp: int):
        """x_flat: [R, d_in] rows already replicated per (token, expert);
        inds: [R] expert index per row."""
        xf = np.asarray(x_flat.astype(mx.float32))
        sq = xf * xf
        d = xf.shape[-1]
        if path not in self.expert_diag:
            self.expert_diag[path] = np.zeros((n_exp, d), dtype=np.float64)
            self.expert_rows[path] = np.zeros(n_exp, dtype=np.int64)
        np.add.at(self.expert_diag[path], inds, sq)
        np.add.at(self.expert_rows[path], inds, 1)


STATS = Stats()
_ORIG_LINEAR = None
_ORIG_SWITCH = None
_PATHS: dict[int, str] = {}
_N_EXPERTS = 512

ROWS_PER_LAYER = 4096  # reservoir sample of MoE inputs + routing, for the
                       # per-unit option measurement (dots3 calibrate_units)


class RowReservoir:
    """Per-layer reservoir of (x row fp16, top-k inds, top-k weights)."""

    def __init__(self, cap: int = ROWS_PER_LAYER):
        self.cap = cap
        self.data: dict[str, dict] = {}
        self._rng = np.random.default_rng(42)

    def record(self, path: str, x: mx.array, inds: mx.array, scores: mx.array):
        xf = np.asarray(x.astype(mx.float16))
        ii = np.asarray(inds).astype(np.int16)
        ww = np.asarray(scores.astype(mx.float16))
        st = self.data.setdefault(path, {"x": [], "inds": [], "w": [], "seen": 0})
        for r in range(xf.shape[0]):
            st["seen"] += 1
            if len(st["x"]) < self.cap:
                st["x"].append(xf[r]); st["inds"].append(ii[r]); st["w"].append(ww[r])
            else:
                j = int(self._rng.integers(0, st["seen"]))
                if j < self.cap:
                    st["x"][j] = xf[r]; st["inds"][j] = ii[r]; st["w"][j] = ww[r]

    def emit(self, out_dir):
        out = {}
        for path, st in self.data.items():
            out[path + ".rows_x"] = mx.array(np.stack(st["x"]))
            out[path + ".rows_inds"] = mx.array(np.stack(st["inds"]).astype(np.int32))
            out[path + ".rows_w"] = mx.array(np.stack(st["w"]))
        if out:
            mx.save_safetensors(str(out_dir / "moe_rows.safetensors"), out)
        return len(self.data)


ROWS = RowReservoir()


def _install(model, full_h: bool):
    global _ORIG_LINEAR, _ORIG_SWITCH
    for path, module in model.named_modules():
        _PATHS[id(module)] = path

    from . import modeling as M

    def moe_recorder(block, x2d, inds, scores):
        path = _PATHS.get(id(block))
        if path is not None:
            ROWS.record(path, x2d, inds, scores)

    M.MOE_ROW_RECORDER = moe_recorder

    _ORIG_LINEAR = nn.Linear.__call__

    def linear_call(self, x):
        path = _PATHS.get(id(self))
        if path is not None:
            # lm_head input (the final mixed state) feeds the "final" AWQ
            # site and lm_head's own imatrix — diag only, no full H
            want = full_h and any(path.endswith(s) for s in FULL_H_SUFFIXES)
            STATS.add(path, x, want)
        return _ORIG_LINEAR(self, x)

    nn.Linear.__call__ = linear_call

    _ORIG_SWITCH = SwitchLinear.__call__

    def switch_call(self, x, indices, sorted_indices=False):
        path = _PATHS.get(id(self))
        if path is not None:
            # x: [..., 1, d] broadcast over expert slots; indices align with
            # the row layout regardless of gather-sort (order-invariant stats)
            d = x.shape[-1]
            xr = x.reshape(-1, d)
            inds = np.asarray(indices.reshape(-1)).astype(np.int64)
            if xr.shape[0] == inds.shape[0]:
                # down_proj: rows already one-per-(token, expert)
                STATS.add_expert(path, xr, inds, _N_EXPERTS)
            else:
                # gate/up: [T,1,1,d] with k indices per token → replicate
                # token-major to align with indices.reshape(-1)
                k = inds.shape[0] // xr.shape[0]
                STATS.add_expert(path, mx.repeat(xr, k, axis=0), inds, _N_EXPERTS)
            # shared-input Hessian for gate/up: the same token row feeds every
            # routed expert, so one H per layer (rank = tokens/d_in). On the
            # gather-sort path rows arrive replicated k× — a uniform scale
            # GPTQ is invariant to, and the row count divides it back out.
            if full_h and path.endswith("switch_mlp.gate_proj"):
                STATS.add(path + ".sharedH", xr, True)
        return _ORIG_SWITCH(self, x, indices, sorted_indices=sorted_indices)

    SwitchLinear.__call__ = switch_call


def _uninstall():
    if _ORIG_LINEAR is not None:
        nn.Linear.__call__ = _ORIG_LINEAR
    if _ORIG_SWITCH is not None:
        SwitchLinear.__call__ = _ORIG_SWITCH
    from . import modeling as M

    M.MOE_ROW_RECORDER = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=1_200_000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-full-h", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = load_model(args.model, lazy=True)
    global _N_EXPERTS
    _N_EXPERTS = model.args.num_experts
    _install(model, full_h=not args.no_full_h)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tok = 0
    t0 = time.time()
    try:
        for text in iter_texts(args.corpus):
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)[: args.chunk]
            if len(ids) < 8:
                continue
            logits = model(mx.array([ids]))
            mx.eval(logits)
            n_tok += len(ids)
            if n_tok % 50_000 < args.chunk:
                el = time.time() - t0
                print(f"{n_tok/1e6:.2f}M tokens, {el/60:.1f} min "
                      f"({n_tok/max(el,1):.0f} tok/s)", flush=True)
            if n_tok >= args.target_tokens:
                break
    finally:
        _uninstall()

    # emit
    diag_out = {}
    for k, v in STATS.diag.items():
        diag_out[k + ".diag"] = mx.array((v / max(STATS.rows[k], 1)).astype(np.float32))
        diag_out[k + ".amax"] = mx.array(STATS.amax[k])
    for k, v in STATS.expert_diag.items():
        rows = np.maximum(STATS.expert_rows[k][:, None], 1)
        diag_out[k + ".expert_diag"] = mx.array((v / rows).astype(np.float32))
        diag_out[k + ".expert_rows"] = mx.array(STATS.expert_rows[k])
    mx.save_safetensors(str(out_dir / "diag.safetensors"), diag_out)

    fh = {k + ".H": mx.array((v / max(STATS.rows.get(k, 1), 1)).astype(np.float32))
          for k, v in STATS.full_h.items()}
    if fh:
        mx.save_safetensors(str(out_dir / "full_h.safetensors"), fh)

    n_row_layers = ROWS.emit(out_dir)

    meta = {
        "tokens": n_tok,
        "modules_diag": len(STATS.diag),
        "modules_expert": len(STATS.expert_diag),
        "modules_full_h": len(STATS.full_h),
        "moe_row_layers": n_row_layers,
        "rows_per_layer_cap": ROWS.cap,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
