"""End-to-end micro round-trip: synthetic hy_v3 source -> convert_hy3_jang
JANG_2L bundle -> load onto jang_tools.hy3.model with the MTP head attached.

Pins the riskiest seam before a real 597GB conversion: the converter's output
tensor names (prestacked switch_mlp, source-style router/shared names,
mtp.0.* final names) must load through Model.sanitize + attach_mtp with zero
remapping surprises, and the quantization override map must dequantize back
to finite logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from safetensors.numpy import save_file  # noqa: E402


VOCAB, H, MOE_I, DENSE_I = 256, 128, 128, 256
N_LAYERS, NE, TOP_K = 2, 4, 2
N_HEADS, N_KV, HD = 4, 2, 32


def _tiny_cfg() -> dict:
    return {
        "model_type": "hy_v3",
        "architectures": ["HYV3ForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": H,
        "intermediate_size": DENSE_I,
        "moe_intermediate_size": MOE_I,
        "expert_hidden_dim": MOE_I,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV,
        "head_dim": HD,
        "num_experts": NE,
        "num_experts_per_tok": TOP_K,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "route_norm": True,
        "router_scaling_factor": 2.826,
        "moe_router_use_sigmoid": True,
        "moe_router_enable_expert_bias": True,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 1,
        "enable_lm_head_fp32": True,
    }


def _write_tiny_source(src: Path) -> None:
    rng = np.random.default_rng(7)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    t: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": w(VOCAB, H),
        "lm_head.weight": w(VOCAB, H),
        "model.norm.weight": np.ones(H, np.float32),
    }

    def add_layer(li: int, moe: bool, mtp: bool) -> None:
        p = f"model.layers.{li}"
        if mtp:
            t[f"{p}.eh_proj.weight"] = w(H, 2 * H)
            for n in ("enorm", "hnorm", "final_layernorm"):
                t[f"{p}.{n}.weight"] = np.ones(H, np.float32)
        t[f"{p}.self_attn.q_proj.weight"] = w(N_HEADS * HD, H)
        t[f"{p}.self_attn.k_proj.weight"] = w(N_KV * HD, H)
        t[f"{p}.self_attn.v_proj.weight"] = w(N_KV * HD, H)
        t[f"{p}.self_attn.o_proj.weight"] = w(H, N_HEADS * HD)
        t[f"{p}.self_attn.q_norm.weight"] = np.ones(HD, np.float32)
        t[f"{p}.self_attn.k_norm.weight"] = np.ones(HD, np.float32)
        t[f"{p}.input_layernorm.weight"] = np.ones(H, np.float32)
        t[f"{p}.post_attention_layernorm.weight"] = np.ones(H, np.float32)
        if not moe:
            for proj, shape in (
                ("gate_proj", (DENSE_I, H)), ("up_proj", (DENSE_I, H)),
                ("down_proj", (H, DENSE_I)),
            ):
                t[f"{p}.mlp.{proj}.weight"] = w(*shape)
            return
        t[f"{p}.mlp.router.gate.weight"] = w(NE, H)
        t[f"{p}.mlp.expert_bias"] = np.zeros(NE, np.float32)
        for proj, shape in (
            ("gate_proj", (MOE_I, H)), ("up_proj", (MOE_I, H)),
            ("down_proj", (H, MOE_I)),
        ):
            t[f"{p}.mlp.shared_mlp.{proj}.weight"] = w(*shape)
            for e in range(NE):
                t[f"{p}.mlp.experts.{e}.{proj}.weight"] = w(*shape)

    add_layer(0, moe=False, mtp=False)
    add_layer(1, moe=True, mtp=False)
    add_layer(2, moe=True, mtp=True)  # MTP layer (index == num_hidden_layers)

    src.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    save_file(t, str(src / shard))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": int(sum(a.nbytes for a in t.values()))},
        "weight_map": {k: shard for k in t},
    }))
    (src / "config.json").write_text(json.dumps(_tiny_cfg()))


def test_convert_and_load_with_native_mtp(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _write_tiny_source(src)

    from jang_tools.convert_hy3_jang import main as convert_main

    convert_main([
        "--src", str(src), "--out", str(out),
        "--profile", "JANG_2L", "--no-awq",
    ])

    # ── bundle surface ──
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    keys = set(idx["weight_map"])
    assert "model.layers.1.mlp.switch_mlp.gate_proj.weight" in keys
    assert "model.layers.1.mlp.router.gate.weight" in keys  # source name kept
    assert "mtp.0.eh_proj.weight" in keys
    assert "mtp.0.block.mlp.switch_mlp.down_proj.weight" in keys
    assert "mtp.0.block.mlp.gate.e_score_correction_bias" in keys
    assert not any(k.startswith("model.layers.2.") for k in keys)

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["runtime"]["mtp_mode"] == "preserved_native_candidate"
    jang_cfg = json.loads((out / "jang_config.json").read_text())
    assert jang_cfg["profile"] == "JANG_2L"
    assert jang_cfg["quantization"]["awq"]["enabled"] is False
    assert jang_cfg["quantization"]["awq"]["scope"] is None

    # vmlx `_jang_default_bits` does int(quant["bits"]) eagerly inside a
    # setdefault(): a dict here is a TypeError at load. Per-role map must live
    # under `bits_by_role`, and NOT under `mxtq_bits` (that key routes the
    # loader into the TurboQuant hydrate path; this is an affine bundle).
    q = jang_cfg["quantization"]
    assert isinstance(q["bits"], int), f"quantization.bits must be scalar, got {q['bits']!r}"
    assert q["bits"] in (2, 3, 4, 5, 6, 8)
    assert q["group_size"] == q["block_size"] == 128
    assert isinstance(q["bits_by_role"], dict)
    assert q["bits_by_role"]["routed_expert"] == 2
    assert q["bits_by_role"]["mtp"] == 8
    assert "mxtq_bits" not in jang_cfg
    assert set(q["bit_widths_used"]) == {2, 6, 8}

    # capabilities must be the canonical block (verify_directory round-trips it)
    for block in (cfg["capabilities"], jang_cfg["capabilities"]):
        assert block["family"] == "hy_v3"
        assert block["has_vision"] is False
        assert block["modalities"] == {
            "text": True, "vision": False, "audio": False, "video": False
        }

    # ── load: construct model with head, sanitize, quantize, load ──
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from safetensors import safe_open

    from jang_tools.hy3.model import Model, ModelArgs

    args = ModelArgs.from_dict(cfg)
    model = Model(args)
    model.attach_mtp()

    weights = {}
    for f in sorted(out.glob("model-*.safetensors")):
        with safe_open(str(f), framework="numpy") as fh:
            for k in fh.keys():
                weights[k] = mx.array(fh.get_tensor(k))
    weights = model.sanitize(weights)

    quant = cfg["quantization"]

    def class_predicate(path, module):
        override = quant.get(path)
        if isinstance(override, dict):
            return override
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=quant["group_size"],
        bits=quant["bits"],
        class_predicate=class_predicate,
    )
    model.load_weights(list(weights.items()))
    model.eval()

    n_params = len(tree_flatten(model.parameters()))
    assert n_params > 0

    # ── forward + MTP draft/verify shape run ──
    from mlx_lm.models.cache import KVCache

    x = mx.array([[3, 5, 7, 11]])
    cache = [KVCache() for _ in model.layers]
    logits, hidden = model(x, cache=cache, return_hidden=True)
    mx.eval(logits, hidden)
    assert logits.shape == (1, 4, VOCAB)
    assert bool(mx.isfinite(logits).all())

    main_tok = mx.argmax(logits[:, -1, :], axis=-1)[None]
    mtp_cache = model.make_mtp_cache()
    draft_logits = model.mtp_forward(hidden[:, -1:, :], main_tok, mtp_cache)
    mx.eval(draft_logits)
    assert draft_logits.shape == (1, 1, VOCAB)
    assert bool(mx.isfinite(draft_logits).all())
    assert mtp_cache[0].offset == 1

    # verify-style 2-token continuation on the base cache stays finite
    draft_tok = mx.argmax(draft_logits[:, -1, :], axis=-1)[None]
    two = mx.concatenate([main_tok, draft_tok], axis=-1)
    v_logits = model(two, cache=cache)
    mx.eval(v_logits)
    assert v_logits.shape == (1, 2, VOCAB)
    assert bool(mx.isfinite(v_logits).all())
