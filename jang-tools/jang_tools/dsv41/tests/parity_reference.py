"""Parity: release inference/model.py (CPU torch, shimmed kernels) vs the MLX port.

Builds a small random DSV4.1 that exercises every V4.1 mechanism (ratio 0/2/1
layers, KV-source sharing, index sources without their own keys, candidate
blocks, Engram, sliding window), writes it as a JANG-layout bundle (8-bit affine
everywhere, Engram table as an FP8 + E8M0 safetensors file), then compares
prefill and teacher-forced decode logits step by step.

  python jang_tools/dsv41/tests/parity_reference.py --src ~/models/DeepSeek-V4.1-Flash-1017
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def build_config():
    from jang_tools.dsv41.engram import _next_prime

    eng = {
        "engram_layer_ids": [1, 4],
        "engram_max_ngram_size": 3,
        "engram_vocab_size": 1000,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
    }
    # table sizes = sum of the bucket primes, drawn the same way the release does
    seen, sizes = set(), []
    for _ in eng["engram_layer_ids"]:
        tot = 0
        for _ in range(eng["engram_max_ngram_size"] - 1):
            cur = eng["engram_vocab_size"] - 1
            for _ in range(eng["engram_n_heads"]):
                cur = _next_prime(cur, seen)
                seen.add(cur)
                tot += cur
        sizes.append(tot)
    eng["engram_num_embeddings"] = sizes
    text = {
        "vocab_size": 129280, "hidden_size": 256, "moe_intermediate_size": 128, "num_hidden_layers": 6,
        "num_attention_heads": 4, "head_dim": 64, "qk_rope_head_dim": 16, "q_lora_rank": 64,
        "o_lora_rank": 32, "o_groups": 2, "swiglu_limit": 10.0, "rms_norm_eps": 1e-20, "rope_theta": 10000,
        "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 64},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "routed_scaling_factor": 1.5, "norm_topk_prob": True,
        "sliding_window": 16, "compress_ratios": [0, 2, 2, 1, 1, 1], "compress_rope_theta": 40000,
        "kv_source_layer_ids": [1, 3], "index_source_layer_ids": [1, 3, 5], "index_n_heads": 16,
        "index_head_dim": 32, "index_topk": 8, "candidate_source_layer_id": 3, "candidate_topk_blocks": 3,
        "candidate_block_size": 4, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, **eng,
    }
    return text


def ref_args(ref, t):
    rs = t["rope_scaling"]
    return ref.ModelArgs(
        max_batch_size=1, max_seq_len=256, temperature=0, dtype="bf16", expert_dtype=None,
        vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
        n_layers=t["num_hidden_layers"], n_mtp_layers=0, n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"], n_activated_experts=t["num_experts_per_tok"],
        route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"], q_lora_rank=t["q_lora_rank"],
        head_dim=t["head_dim"], rope_head_dim=t["qk_rope_head_dim"], norm_eps=t["rms_norm_eps"],
        o_groups=t["o_groups"], o_lora_rank=t["o_lora_rank"], window_size=t["sliding_window"],
        compress_ratios=tuple(t["compress_ratios"]), kv_source_layers=tuple(t["kv_source_layer_ids"]),
        index_source_layers=tuple(t["index_source_layer_ids"]), compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=rs["original_max_position_embeddings"], rope_theta=t["rope_theta"],
        rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"], index_topk=t["index_topk"],
        candidate_source_layer=t["candidate_source_layer_id"], candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"], hc_mult=t["hc_mult"],
        engram_layer_ids=tuple(t["engram_layer_ids"]), engram_num_embeddings=tuple(t["engram_num_embeddings"]),
        engram_max_ngram_size=t["engram_max_ngram_size"], engram_vocab_size=t["engram_vocab_size"],
        engram_n_heads=t["engram_n_heads"], engram_head_dim=t["engram_head_dim"],
        engram_pad_id=t["engram_pad_token_id"], engram_compressed_vocab_size=t["engram_compressed_vocab_size"],
        vision_n_layers=0, dspark_block_size=0,
    )


def randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype in (torch.float8_e4m3fn,):
                p.copy_((torch.randn(p.shape, generator=g) * 2).to(torch.float8_e4m3fn))
            elif p.dtype == torch.float8_e8m0fnu:
                p.copy_((2.0 ** -torch.randint(1, 4, p.shape, generator=g).float()).to(torch.float8_e8m0fnu))
            elif "norm" in name or name.endswith(("q_weight", "k_weight")):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif "hc_" in name and name.endswith("scale"):
                p.copy_(0.5 + 0.1 * torch.randn(p.shape, generator=g))
            elif "hc_" in name and name.endswith("base"):
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith(("attn_sink", ".bias")):
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1] if p.ndim > 1 else 1
                p.copy_(torch.randn(p.shape, generator=g) / fan_in ** 0.5)


def write_bundle(model, text, src: Path, out: Path):
    import mlx.core as mx
    from jang_tools.dsv41.convert_dsv41_jang import PASSTHROUGH_RE

    out.mkdir(parents=True, exist_ok=True)
    (out / "engram").mkdir()
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    tensors, engram = {}, {}
    experts: dict = {}

    def to_mx(t):
        if t.dtype == torch.bfloat16:
            return mx.array(t.view(torch.uint16).numpy()).view(mx.bfloat16)
        return mx.array(t.numpy())

    def write_back(t, q):
        # the reference then runs on exactly the weights the bundle holds
        deq = mx.dequantize(*q, group_size=64, bits=8).astype(mx.float32)
        with torch.no_grad():
            t.copy_(torch.from_numpy(np.array(deq)).reshape(t.shape).to(t.dtype))

    for name, t in sd.items():
        if ".engram.embed." in name:
            engram[name] = t.contiguous()
            continue
        parts = name.split(".")
        if ".ffn.experts." in name:
            experts.setdefault((int(parts[1]), parts[5]), {})[int(parts[4])] = t
            continue
        if not PASSTHROUGH_RE.search(name) and name.endswith(".weight") and t.ndim == 2:
            qw, s, b = mx.quantize(to_mx(t.to(torch.bfloat16)), group_size=64, bits=8)
            write_back(t, (qw, s, b))
            base = name[: -len(".weight")]
            tensors.update({f"{base}.weight": qw, f"{base}.scales": s, f"{base}.biases": b})
        else:
            tensors[name] = to_mx(t)
    proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    for (layer, w), d in experts.items():
        stack = torch.stack([d[e] for e in sorted(d)]).to(torch.bfloat16)
        qw, s, b = mx.quantize(to_mx(stack), group_size=64, bits=8)
        for i, e in enumerate(sorted(d)):
            write_back(d[e], (qw[i], s[i], b[i]))
        base = f"layers.{layer}.ffn.experts.{proj[w]}"
        tensors.update({f"{base}.weight": qw, f"{base}.scales": s, f"{base}.biases": b})
    for l in range(text["num_hidden_layers"]):
        tensors.setdefault(f"layers.{l}.ffn.gate.bias_vl", mx.zeros((text["n_routed_experts"],)))
    mx.save_safetensors(str(out / "model-00001-of-00001.safetensors"), tensors)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    (out / "config.json").write_text(json.dumps({"eos_token_id": 1, "text_config": text}))
    shutil.copy2(src / "tokenizer.json", out / "tokenizer.json")
    from safetensors.torch import save_file

    save_file(engram, str(out / "engram" / "engram-00001-of-00001.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="release dir (tokenizer + inference/)")
    ap.add_argument("--prompt-len", type=int, default=70)
    ap.add_argument("--decode", type=int, default=40)
    ap.add_argument("--full-selection", action="store_true",
                    help="index_topk / candidate blocks cover every position: no near-tie flips, strict check")
    a = ap.parse_args()
    src = Path(a.src).expanduser()

    sys.path[:0] = [str(HERE / "ref_shim"), str(src / "inference")]
    import model as ref  # the release reference, with shimmed kernels
    from transformers import AutoTokenizer

    torch.set_default_dtype(torch.bfloat16)
    text = build_config()
    if a.full_selection:
        text.update(index_topk=512, candidate_topk_blocks=512)
    tok = AutoTokenizer.from_pretrained(str(src))
    rmodel = ref.Transformer(ref_args(ref, text), tok)
    torch.set_default_dtype(torch.float32)
    randomize(rmodel)

    sample = (src / "encoding" / "README.md").read_text()
    ids = tok.encode(sample, add_special_tokens=False)[: a.prompt_len + a.decode]
    P = a.prompt_len

    from jang_tools.dsv41.model import DSV41

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle"
        write_bundle(rmodel, text, src, bundle)
        ref_logits = []
        with torch.inference_mode():
            x = torch.tensor([ids])
            ref_logits.append(rmodel.forward(x[:, :P], 0)[1][0].float().numpy())
            for i in range(P, len(ids) - 1):
                ref_logits.append(rmodel.forward(x[:, i : i + 1], i)[1][0].float().numpy())
        m = DSV41(bundle, wired=False)
        mine = [np.array(m.forward(ids[:P]))]
        for i in range(P, len(ids) - 1):
            mine.append(np.array(m.forward([ids[i]])))

    worst_cos, agree = 1.0, 0
    for step, (r, o) in enumerate(zip(ref_logits, mine)):
        cos = float(np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o)))
        rel = float(np.linalg.norm(r - o) / np.linalg.norm(r))
        worst_cos = min(worst_cos, cos)
        agree += int(r.argmax() == o.argmax())
        if step in (0, 1, len(mine) - 1) or cos < 0.995:
            print(f"step {step:3d} pos {P + step - 1 if step else P - 1:4d}  cos {cos:.5f}  rel {rel:.4f}  "
                  f"top1 {r.argmax()} vs {o.argmax()}")
    print(f"steps {len(mine)}  worst cos {worst_cos:.5f}  top1 agree {agree}/{len(mine)}")
    sys.exit(0 if worst_cos > (0.999 if a.full_selection else 0.8) else 1)


if __name__ == "__main__":
    main()
