"""Tiny-random parity: MLX glm5_next runtime vs transformers reference.

Builds a small random Glm5NextTextModel (KDA + MLA + dense + MoE + mHC all
exercised), saves it in real-checkpoint naming, loads through our sanitize +
model, and compares final hidden states in fp32. Also checks our cache paths
(prefill == chunked == stepwise).

Run:  PYTHONPATH=/tmp/glm5tf python -m jang_tools.glm5_next.parity_tiny
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/glm5tf")

import numpy as np


def build_torch(tmp: Path):
    import torch
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    cfg = Glm5NextTextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "deepseek_sparse_attention",
                     "linear_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        indexer_types=["full", "full", "full", "full"],
        first_k_dense_replace=1,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1, topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        head_dim=16,
        index_topk=2048,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        linear_attn_config={"num_heads": 4, "head_dim": 16,
                            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                            "kda_layers": [0, 2, 3], "full_attn_layers": [1]},
        vocab_size=512,
        pad_token_id=0,
        rms_norm_eps=1e-5,
        swiglu_limit=10.0,
        attention_bias=False,
        mla_use_nope=True,
    )
    torch.manual_seed(0)
    model = Glm5NextTextModel(cfg).eval().float()
    ids = torch.randint(0, 512, (1, 33))  # deliberately not conv/chunk aligned
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).last_hidden_state.numpy()

    # save in real-checkpoint naming
    from safetensors.torch import save_file
    sd = {}
    for k, v in model.state_dict().items():
        if ".conv1d.weight" in k:  # fused conv [3C,1,W] -> split q/k/v convs
            C = v.shape[0] // 3
            base = k.replace(".conv1d.weight", "")
            for i, p in enumerate("qkv"):
                sd[f"model.language_model.{base}.{p}_conv1d.weight"] = \
                    v[i * C:(i + 1) * C].contiguous()
            continue
        if ".experts.gate_up_proj" in k:  # fused [E,2i,h] -> per-expert gate/up
            base = k.replace(".experts.gate_up_proj", "")
            E, twoi, h = v.shape
            for e in range(E):
                sd[f"model.language_model.{base}.experts.{e}.gate_proj.weight"] = \
                    v[e, : twoi // 2].contiguous()
                sd[f"model.language_model.{base}.experts.{e}.up_proj.weight"] = \
                    v[e, twoi // 2:].contiguous()
            continue
        if ".experts.down_proj" in k:
            base = k.replace(".experts.down_proj", "")
            for e in range(v.shape[0]):
                sd[f"model.language_model.{base}.experts.{e}.down_proj.weight"] = \
                    v[e].contiguous()
            continue
        k2 = k.replace("attn_hc.fn", "hc_attn_fn").replace("attn_hc.base", "hc_attn_base") \
             .replace("attn_hc.scale", "hc_attn_scale").replace("ffn_hc.fn", "hc_ffn_fn") \
             .replace("ffn_hc.base", "hc_ffn_base").replace("ffn_hc.scale", "hc_ffn_scale") \
             .replace("forget_gate.", "").replace("mlp.gate.e_score_correction_bias",
                                                  "mlp.gate.e_score_correction_bias")
        sd[f"model.language_model.{k2}"] = v.contiguous()
    sd["lm_head.weight"] = torch.zeros(512, 64)  # unused; loader wants it
    save_file(sd, str(tmp / "model.safetensors"))
    (tmp / "config.json").write_text(json.dumps({"text_config": cfg.to_dict()}))
    return ids.numpy(), ref


def main():
    tmp = Path(tempfile.mkdtemp(prefix="glm5parity_"))
    ids_np, ref = build_torch(tmp)

    import mlx.core as mx
    from jang_tools.glm5_next.load import load_model

    model = load_model(str(tmp), dtype=mx.float32)
    ids = mx.array(ids_np)
    ours = np.asarray(model.model(ids), dtype=np.float32)

    # NOTE the final mean-stream signal of a RANDOM-INIT mHC model is tiny
    # (rms ~0.02 vs post-norm rms ~1.0), so the last RMSNorm applies ~x40 gain
    # to any absolute fp32 noise. Raw-delta gating there measures the noise
    # floor, not correctness (measured 2026-08-29). Gate on cosine + the
    # stream-level checks below; components are pinned separately.
    num = (ours * ref).sum(-1)
    den = np.linalg.norm(ours, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-12
    cos = num / den
    print(f"prefill vs torch: cosine mean={cos.mean():.5f} min={cos.min():.5f} "
          f"mean|Δ|={np.abs(ours - ref).mean():.3e}")
    ok = cos.mean() > 0.999 and cos.min() > 0.99
    denom = np.abs(ref).mean() + 1e-9

    # cache-path check: stepwise == prefill (our side only)
    cache = model.make_cache()
    outs = []
    for t in range(ids.shape[1]):
        outs.append(np.asarray(model.model(ids[:, t:t + 1], cache=cache)))
    step = np.concatenate(outs, axis=1)
    n2 = (step * ours).sum(-1) / (np.linalg.norm(step, axis=-1)
                                  * np.linalg.norm(ours, axis=-1) + 1e-12)
    print(f"stepwise vs prefill: cosine min={n2.min():.5f}")
    ok = ok and n2.min() > 0.99

    # chunked prefill (split mid-sequence)
    cache = model.make_cache()
    a = np.asarray(model.model(ids[:, :17], cache=cache))
    b = np.asarray(model.model(ids[:, 17:], cache=cache))
    ch = np.concatenate([a, b], axis=1)
    cd = np.abs(ch - ours)
    print(f"chunked vs prefill:  max|Δ|={cd.max():.3e}")
    ok = ok and cd.max() / denom < 5e-3

    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
