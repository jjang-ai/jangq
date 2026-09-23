"""H200-side calibration capture for Qwen3.8-Flash-Next (qwen4_exp).

Standalone: needs only torch + transformers(main) + safetensors + numpy.
Emits EXACTLY the stat files the Mac-side JANG tools consume (keys use the
MLX runtime naming):

  diag.safetensors      <mod>.diag [d_in] fp32, <mod>.amax [d_in]
                        <switch>.expert_diag [E,d_in], .expert_rows [E]
  full_h.safetensors    <mod>.H (trunk) + <layer gate_proj>.sharedH.H
  moe_rows.safetensors  <layer>.mlp.rows_x/.rows_inds/.rows_w (reservoir)
  meta.json

Modes:
  capture    the main pass (default)
  coherence  chat-template greedy probe, prints text
  klref      teacher-forced top-K logprob reference for the held-out set
             (same npz schema as jang_tools.qwen4_exp.kl_eval 'ref' mode)

  python capture_torch.py --model <dir> --corpus corpus_v3.jsonl \
      --out capture/ --target-tokens 1200000
  python capture_torch.py --mode coherence --model <dir>
  python capture_torch.py --mode klref --model <dir> --prompts kl.jsonl --out ref.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

FULL_H_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)
ROWS_PER_LAYER = 4096
TOPK = 128


def iter_texts(corpus_path):
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, str):
                yield rec
            elif "text" in rec:
                yield rec["text"]
            elif "messages" in rec:
                yield "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else ""
                                for m in rec["messages"])
            elif "prompt" in rec:
                yield str(rec.get("prompt", "")) + str(rec.get("response", ""))


def mlx_name(torch_name: str) -> str:
    # Qwen4ExpForCausalLM: "model.layers.N..." → "language_model.layers.N..."
    if torch_name.startswith("model."):
        return "language_model." + torch_name[len("model."):]
    return torch_name  # lm_head


class Stats:
    def __init__(self):
        self.diag, self.amax, self.rows = {}, {}, {}
        self.full_h = {}
        self.expert_diag, self.expert_rows = {}, {}
        self.moe_rows = {}
        self._rng = np.random.default_rng(42)

    def add(self, path, x: torch.Tensor, want_full: bool):
        """diag/amax accumulate ON-DEVICE (no per-call host syncs); full-H
        accumulates fp64 on CPU with one async transfer per module per chunk
        (GPU fp64 residency for the 6144-wide H's would cost ~14GB)."""
        flat = x.reshape(-1, x.shape[-1]).float()
        sq = flat.pow(2).sum(0).double()
        am = flat.abs().amax(0)
        n = flat.shape[0]
        if path not in self.diag:
            self.diag[path], self.amax[path], self.rows[path] = sq, am, n
        else:
            self.diag[path] = self.diag[path] + sq
            self.amax[path] = torch.maximum(self.amax[path], am)
            self.rows[path] += n
        if want_full:
            h = (flat.T @ flat).double().cpu().numpy()
            self.full_h[path] = self.full_h.get(path, 0) + h

    def add_expert(self, path, x: torch.Tensor, expert: int, n_exp: int, d: int):
        """GPU-resident accumulators — NO per-expert host syncs (the naive
        .cpu()-per-expert version ran the whole capture at 13 tok/s)."""
        if path not in self.expert_diag:
            self.expert_diag[path] = torch.zeros(n_exp, d, dtype=torch.float64,
                                                 device=x.device)
            self.expert_rows[path] = torch.zeros(n_exp, dtype=torch.int64,
                                                 device=x.device)
        self.expert_diag[path][expert] += x.float().pow(2).sum(0).double()
        self.expert_rows[path][expert] += x.shape[0]

    def add_moe_rows(self, path, x, inds, w):
        """Vectorized chunk subsample (128/chunk) instead of a per-row Python
        reservoir — rows are exchangeable, the approximation is harmless."""
        st = self.moe_rows.setdefault(path, {"x": [], "inds": [], "w": [], "seen": 0})
        R = x.shape[0]
        st["seen"] += R
        k = min(128, R)
        sel = torch.from_numpy(self._rng.choice(R, size=k, replace=False)).to(x.device)
        st["x"].append(x[sel].detach().to(torch.float16).cpu().numpy())
        st["inds"].append(inds[sel].detach().to(torch.int32).cpu().numpy())
        st["w"].append(w[sel].detach().to(torch.float16).cpu().numpy())
        if len(st["x"]) * 128 > ROWS_PER_LAYER * 2:
            # keep memory bounded: concatenate and random-downsample to cap
            xs = np.concatenate(st["x"]); ii = np.concatenate(st["inds"]); ww = np.concatenate(st["w"])
            keep = self._rng.choice(xs.shape[0], size=ROWS_PER_LAYER, replace=False)
            st["x"] = [xs[keep]]; st["inds"] = [ii[keep]]; st["w"] = [ww[keep]]


STATS = Stats()


def install_hooks(model, full_h=True):
    handles = []
    for tname, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            path = mlx_name(tname)
            want = full_h and any(path.endswith(s) for s in FULL_H_SUFFIXES)

            def pre_hook(m, args, _path=path, _want=want):
                STATS.add(_path, args[0], _want)

            handles.append(mod.register_forward_pre_hook(pre_hook))

    return handles


EXPERT_PATHS = {}
FULL_H_ON = [True]


def register_expert_paths(model):
    for tname, mod in model.named_modules():
        if type(mod).__name__.endswith("Experts"):
            EXPERT_PATHS[id(mod)] = mlx_name(tname).rsplit(".experts", 1)[0]


def patch_expert_class():
    """CLASS-level replacement, installed BEFORE from_pretrained so accelerate
    wraps THIS forward with its offload hooks (an instance-level replacement
    clobbers the hook and meta-offloaded weights never page in).

    Vectorized padded-bmm computing IDENTICAL math to the HF per-expert
    Python loop (which capped capture at 13 tok/s), with per-expert stats
    fused as single tensor ops."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    def expert_forward(_mod, hidden_states, top_k_index, top_k_weights):
        _layer = EXPERT_PATHS.get(id(_mod), "unknown.mlp")
        _switch = _layer + ".switch_mlp"
        _full_h = FULL_H_ON[0]
        if True:
                STATS.add_moe_rows(_layer, hidden_states, top_k_index, top_k_weights)
                STATS.add(_switch + ".gate_proj.sharedH", hidden_states, _full_h)

                E = _mod.num_experts
                T, d = hidden_states.shape
                k = top_k_index.shape[-1]
                dev = hidden_states.device
                experts = top_k_index.reshape(-1)
                weights_flat = top_k_weights.reshape(-1)
                token_ids = torch.arange(T, device=dev).repeat_interleave(k)

                order = torch.argsort(experts)
                sorted_e = experts[order]
                sorted_tok = token_ids[order]
                counts = torch.bincount(sorted_e, minlength=E)
                maxc = int(counts.max())
                offs = torch.cumsum(counts, 0) - counts
                within = torch.arange(sorted_e.shape[0], device=dev) - offs[sorted_e]

                padded = hidden_states.new_zeros(E, maxc, d)
                padded[sorted_e, within] = hidden_states[sorted_tok]

                gu = torch.bmm(padded, _mod.gate_up_proj.transpose(1, 2))
                gate, up = gu.chunk(2, dim=-1)
                inter = _mod.act_fn(gate) * up
                out = torch.bmm(inter, _mod.down_proj.transpose(1, 2))

                # fused per-expert stats (padding rows are zeros — harmless)
                # per-chunk fp32 sums on GPU, fp64 accumulation on CPU
                # (GPU is packed to ~134GiB with weights; keep transients lean)
                # slab-wise squared sums: avoid full fp32 copies of the
                # padded [512, maxc, d] buffers (OOM'd twice at ~1GB free)
                def sq_sum(t):
                    parts = []
                    for sl in torch.split(t, 64, dim=0):
                        parts.append(sl.pow(2).sum(1, dtype=torch.float32))
                    return torch.cat(parts).cpu().double()

                gsq = sq_sum(padded)
                dsq = sq_sum(inter)
                ccpu = counts.cpu()
                if _switch + ".gate_proj" not in STATS.expert_diag:
                    STATS.expert_diag[_switch + ".gate_proj"] = torch.zeros(E, d, dtype=torch.float64)
                    STATS.expert_rows[_switch + ".gate_proj"] = torch.zeros(E, dtype=torch.int64)
                    STATS.expert_diag[_switch + ".down_proj"] = torch.zeros(E, inter.shape[-1], dtype=torch.float64)
                    STATS.expert_rows[_switch + ".down_proj"] = torch.zeros(E, dtype=torch.int64)
                STATS.expert_diag[_switch + ".gate_proj"] += gsq
                STATS.expert_rows[_switch + ".gate_proj"] += ccpu
                STATS.expert_diag[_switch + ".down_proj"] += dsq
                STATS.expert_rows[_switch + ".down_proj"] += ccpu

                contrib = out[sorted_e, within] * weights_flat[order, None]
                final = torch.zeros_like(hidden_states)
                final.index_add_(0, sorted_tok, contrib.to(final.dtype))
                return final

    M.Qwen4ExpTextExperts.forward = expert_forward
    print("[experts] vectorized class-level forward installed", flush=True)


def save_stats(out_dir: Path):
    from safetensors.numpy import save_file

    diag_out = {}
    for k, v in STATS.diag.items():
        diag_out[k + ".diag"] = (v.cpu().numpy() / max(STATS.rows[k], 1)).astype(np.float32)
        diag_out[k + ".amax"] = STATS.amax[k].float().cpu().numpy().astype(np.float32)
    for k, v in STATS.expert_diag.items():
        er = STATS.expert_rows[k].numpy()
        rows = np.maximum(er[:, None], 1)
        diag_out[k + ".expert_diag"] = (v.numpy() / rows).astype(np.float32)
        diag_out[k + ".expert_rows"] = er
    save_file(diag_out, str(out_dir / "diag.safetensors"))

    if STATS.full_h:
        fh = {k + ".H": (v / max(STATS.rows.get(k, 1), 1)).astype(np.float32)
              for k, v in STATS.full_h.items()}
        save_file(fh, str(out_dir / "full_h.safetensors"))

    rows_out = {}
    rng = np.random.default_rng(7)
    for k, st in STATS.moe_rows.items():
        xs = np.concatenate(st["x"])
        ii = np.concatenate(st["inds"]).astype(np.int32)
        ww = np.concatenate(st["w"])
        if xs.shape[0] > ROWS_PER_LAYER:
            keep = rng.choice(xs.shape[0], size=ROWS_PER_LAYER, replace=False)
            xs, ii, ww = xs[keep], ii[keep], ww[keep]
        rows_out[k + ".rows_x"] = xs
        rows_out[k + ".rows_inds"] = ii
        rows_out[k + ".rows_w"] = ww
    if rows_out:
        save_file(rows_out, str(out_dir / "moe_rows.safetensors"))


def bypass_qsa_for_short_seqs(model, budget=2048):
    """The HF QSA indexer is a per-query PYTHON loop (~13 tok/s for capture).
    For visible lengths ≤ budget the top-512-block selection keeps every
    complete block AND the tail — i.e. it is exactly the causal mask. All our
    capture/klref sequences are ≤ 2048 tokens, so returning 'keep everything'
    is bit-identical and removes the loop entirely."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M

    orig = M.Qwen4ExpTextQSAIndexer.forward

    def fast_forward(self, hidden_states, position_embeddings, attention_mask,
                     past_key_values):
        kv_len = attention_mask.shape[-1]
        if kv_len <= budget + self.compress_ratio - 1:
            if attention_mask.dtype == torch.bool:
                return torch.ones(
                    (hidden_states.shape[0], 1, hidden_states.shape[1], kv_len),
                    dtype=torch.bool, device=attention_mask.device)
            return torch.zeros(
                (hidden_states.shape[0], 1, hidden_states.shape[1], kv_len),
                dtype=attention_mask.dtype, device=attention_mask.device)
        return orig(self, hidden_states, position_embeddings, attention_mask,
                    past_key_values)

    M.Qwen4ExpTextQSAIndexer.forward = fast_forward
    print("[qsa] short-sequence exact bypass installed (budget", budget, ")", flush=True)


def load_model(model_dir, causal=True):
    from transformers import Qwen4ExpForCausalLM
    from accelerate import dispatch_model

    # accelerate auto-placement is unusable here: layer 1 contains the 95GB
    # n-gram table and DecoderLayer is no-split, so its balancer punts ~9
    # layers to CPU at ANY cap (measured: 218.5/265.6 resident), then OOMs
    # onloading them mid-forward. So: load fully on CPU (2TB RAM), then
    # dispatch with a hand-built map — 12 layers per GPU, table pinned CPU.
    model = Qwen4ExpForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    n = torch.cuda.device_count()
    per = 48 // n
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0,
                  "model.hyper_connection_mixer": n - 1, "lm_head": n - 1}
    for i in range(48):
        d = min(i // per, n - 1)
        if i == 1:
            for sub in ("linear_attn", "mlp", "attn_hyper_connection",
                        "mlp_hyper_connection", "ple.key_proj", "ple.value_proj",
                        "ple.norm_key", "ple.norm_query", "ple.norm_conv",
                        "ple.conv1d"):
                device_map[f"model.layers.1.{sub}"] = d
            device_map["model.layers.1.ple.ple_embedding"] = "cpu"
        else:
            device_map[f"model.layers.{i}"] = d
    model = dispatch_model(model, device_map=device_map)
    # "cpu" in dispatch_model means offload-and-ONLOAD — its hook tries to
    # move the whole 95GB table to GPU at first forward. The model's own
    # NGramEmbedding.forward already handles a CPU-resident table (explicit
    # .to() dance), so strip accelerate's hook from that module entirely.
    from accelerate.hooks import remove_hook_from_module
    ple_emb = model.model.layers[1].ple.ple_embedding
    remove_hook_from_module(ple_emb, recurse=True)
    # hash buffers (a few dozen int64s) belong with the compute on GPU;
    # only the 95GB embedding weight stays CPU (its forward handles that)
    for bname, buf in list(ple_emb.named_buffers(recurse=False)):
        setattr(ple_emb, bname, buf.to("cuda:0"))
    gpu_bytes = sum(p_.numel() * p_.element_size() for p_ in model.parameters()
                    if p_.device.type == "cuda")
    print(f"[load] GPU-resident params: {gpu_bytes/2**30:.1f} GiB across {n} GPUs", flush=True)
    if gpu_bytes < 230 * 2**30:
        from collections import Counter
        on_cpu = Counter()
        for pname, p_ in model.named_parameters():
            if p_.device.type != "cuda" and "ngram_embedding" not in pname:
                on_cpu[pname.rsplit(".", 2)[0]] += p_.numel() * p_.element_size()
        for mod, b in sorted(on_cpu.items(), key=lambda kv: -kv[1])[:12]:
            print(f"[cpu] {b/2**30:6.2f} GiB  {mod}", flush=True)
        raise RuntimeError("placement failed — refusing slow offload run")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["capture", "coherence", "klref"], default="capture")
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-tokens", type=int, default=1_200_000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=333)
    ap.add_argument("--no-full-h", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    if args.mode == "coherence":
        model = load_model(args.model)
        text = tok.apply_chat_template(
            [{"role": "user", "content": "Explain why the sky is blue in one sentence."}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False)
        print(tok.decode(out[0][ids.shape[1]:]))
        return

    if args.mode == "klref":
        model = load_model(args.model)
        bypass_qsa_for_short_seqs(model)  # exact for ≤2048-token teacher forcing
        prompts = [json.loads(l)["text"] if l.strip().startswith("{") else l.rstrip("\n")
                   for l in open(args.prompts) if l.strip()]
        store = {}
        n_ok = 0
        for i, p in enumerate(prompts):
            ids = tok.encode(p, add_special_tokens=False)[: args.max_len]
            if len(ids) < 8:
                continue
            with torch.no_grad():
                logits = model(torch.tensor([ids], device=model.device)).logits[0].float()
            logprobs = logits - logits.logsumexp(-1, keepdim=True)
            top_lp, top_idx = logprobs.topk(TOPK, dim=-1)
            store[f"p{i}_ids"] = np.array(ids)
            store[f"p{i}_top_lp"] = top_lp.cpu().numpy().astype(np.float16)
            store[f"p{i}_top_ids"] = top_idx.cpu().numpy().astype(np.int32)
            store[f"p{i}_margin"] = (top_lp[:, 0] - top_lp[:, 1]).cpu().numpy().astype(np.float32)
            n_ok += 1
            if i % 10 == 0:
                print(f"{i}/{len(prompts)}", flush=True)
        np.savez_compressed(args.out, n_prompts=len(prompts), **store)
        print(f"klref saved: {n_ok} prompts → {args.out}")
        return

    # capture
    assert args.chunk <= 2048, "QSA bypass exactness requires chunks ≤ budget"
    FULL_H_ON[0] = not args.no_full_h
    patch_expert_class()          # BEFORE load: accelerate wraps it with hooks
    model = load_model(args.model)
    bypass_qsa_for_short_seqs(model)
    register_expert_paths(model)
    install_hooks(model, full_h=not args.no_full_h)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tok, t0 = 0, time.time()
    eos_id = 248044
    buf = []

    def packed_chunks():
        # pack records into FULL chunks (EOS-separated — the model hashes
        # per-EOS-segment by design); one forward per 270-token record was
        # tanking throughput 147→120 tok/s and falling
        for text in iter_texts(args.corpus):
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) < 8:
                continue
            buf.extend(ids[: args.chunk])
            buf.append(eos_id)
            while len(buf) >= args.chunk:
                yield buf[: args.chunk]
                del buf[: args.chunk]

    for ids in packed_chunks():
        with torch.no_grad():
            # inner model only: the CausalLM wrapper computes 248k-vocab
            # logits every chunk — and with lm_head CPU-offloaded that GEMM
            # ran on CPU cores, capping the whole capture at ~13 tok/s
            out = model.model(input_ids=torch.tensor([ids], device=model.device))
            STATS.add("lm_head", out.last_hidden_state, False)
        n_tok += len(ids)
        if n_tok % 50_000 < args.chunk:
            el = time.time() - t0
            print(f"{n_tok/1e6:.2f}M tokens, {el/60:.1f} min ({n_tok/max(el,1):.0f} tok/s)",
                  flush=True)
        if n_tok >= args.target_tokens:
            break
    save_stats(out_dir)
    meta = {"tokens": n_tok, "modules_diag": len(STATS.diag),
            "modules_expert": len(STATS.expert_diag),
            "modules_full_h": len(STATS.full_h),
            "moe_row_layers": len(STATS.moe_rows),
            "elapsed_s": time.time() - t0, "source": "capture_torch"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
