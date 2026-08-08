"""Laguna (poolside) all-affine JANG conversion policy checks.

The JANG_2L policy here is byte-for-byte the map of the shipped
Laguna-M.1-JANG_2L bundle (routed 2/2/3, attention+g_proj 8, shared/dense/
embed 6, lm_head 8, gs 64, norms+router fp16) — the recipe already proven
coherent on this family. S-2.1 (117.5B) reuses it unchanged; profiles only
move the routed-expert bits.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_laguna_jang_converter_exists():
    path = ROOT / "jang_tools" / "convert_laguna_jang.py"

    assert path.exists()


def test_laguna_jang_2l_matches_shipped_m1_recipe():
    from jang_tools.convert_laguna_jang import classify_tensor, profile_policy

    policy = profile_policy("JANG_2L")

    assert policy.group_size == 64
    assert policy.routed_bits == {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 3,
    }
    assert policy.attention_bits == 8
    assert policy.shared_expert_bits == 6
    assert policy.dense_ffn_bits == 6
    assert policy.embed_bits == 6
    assert policy.lm_head_bits == 8

    assert classify_tensor(
        "model.layers.1.mlp.experts.7.gate_proj.weight", policy
    ) == (2, "affine")
    assert classify_tensor(
        "model.layers.1.mlp.experts.7.up_proj.weight", policy
    ) == (2, "affine")
    assert classify_tensor(
        "model.layers.1.mlp.experts.7.down_proj.weight", policy
    ) == (3, "affine")
    assert classify_tensor(
        "model.layers.1.mlp.shared_expert.gate_proj.weight", policy
    ) == (6, "affine")
    assert classify_tensor("model.embed_tokens.weight", policy) == (6, "affine")
    assert classify_tensor("lm_head.weight", policy) == (8, "affine")


def test_laguna_attention_gate_rides_with_attention_bits():
    """g_proj gates the attention output (softplus, per-head on S-2.1,
    per-element on M.1) — errors there scale the whole residual write, so it
    must be classified with attention, never as a generic 8-bit leftover of
    a lower-bit profile tweak."""
    from jang_tools.convert_laguna_jang import classify_tensor, profile_policy

    policy = profile_policy("JANG_2L")

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj"):
        assert classify_tensor(
            f"model.layers.3.self_attn.{proj}.weight", policy
        ) == (policy.attention_bits, "affine")


def test_laguna_router_and_norms_pass_through_fp16():
    from jang_tools.convert_laguna_jang import classify_tensor, profile_policy

    policy = profile_policy("JANG_2L")

    # Router gate (mlp.gate.weight) passes through; the dense layer-0 FFN
    # gate_proj (mlp.gate_proj.weight) must NOT be caught by that rule.
    assert classify_tensor("model.layers.1.mlp.gate.weight", policy) == (
        16,
        "passthrough",
    )
    assert classify_tensor("model.layers.0.mlp.gate_proj.weight", policy) == (
        policy.dense_ffn_bits,
        "affine",
    )
    assert classify_tensor(
        "model.layers.1.mlp.experts.e_score_correction_bias", policy
    ) == (16, "passthrough")
    for name in (
        "model.norm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.1.self_attn.q_norm.weight",
        "model.layers.1.self_attn.k_norm.weight",
    ):
        assert classify_tensor(name, policy) == (16, "passthrough")


def test_laguna_chat_block_passes_vendor_generation_params_through():
    """S-2.1 ships temp 1.0 / top_p 1.0 / min_p 0.0 / top_k 20 and poolside_v1
    parsers. The bundle must carry them VERBATIM (no invented floors) plus
    default_chat_template_kwargs. An explicit generation-config default
    remains authoritative over either historical template fallback."""
    from jang_tools.convert_laguna_jang import build_chat_block

    gen = {
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 20,
        "eos_token_id": [2, 24],
        "tool_call_parser": "poolside_v1",
        "reasoning_parser": "poolside_v1",
        "default_chat_template_kwargs": {"enable_thinking": True},
    }
    chat = build_chat_block(gen)

    assert chat["sampling_defaults"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 20,
    }
    assert chat["template_kwargs_defaults"] == {"enable_thinking": True}
    assert chat["reasoning"]["default_enabled"] is True
    assert chat["reasoning"]["default_mode"] == "think"
    assert chat["reasoning"]["modes"] == ["think", "no_think"]
    # Runtime parser names (vmlx registry): GLM-derivative template →
    # deepseek_r1 think tags; glm47 arg_key/arg_value tool format.
    # Vendor's vLLM names are preserved separately.
    assert chat["reasoning"]["parser"] == "deepseek_r1"
    assert chat["tool_calling"]["parser"] == "glm47"
    assert chat["vendor_parsers"] == {
        "reasoning": "poolside_v1", "tool": "poolside_v1"}


def test_laguna_chat_block_derives_current_template_default_without_kwargs():
    """Absent generation kwargs, mirror the effective copied template."""
    from jang_tools.convert_laguna_jang import build_chat_block

    current = (
        "{%- set enable_thinking = enable_thinking | default(true) -%}\n"
        "{%- set preserve_thinking = preserve_thinking | default(false) -%}"
    )
    chat = build_chat_block({"temperature": 0.7}, template_text=current)

    assert chat["reasoning"]["default_enabled"] is True
    assert chat["reasoning"]["default_mode"] == "think"
    # top_k rides in from the card fallback; the vendor declared no value.
    assert chat["sampling_defaults"] == {"temperature": 0.7, "top_k": 20}
    assert chat["template_kwargs_defaults"] == {}


def test_laguna_chat_block_keeps_old_template_default_and_explicit_override():
    """Older template fallback stays Off; an explicit generation value wins."""
    from jang_tools.convert_laguna_jang import build_chat_block

    old = "{%- set enable_thinking = enable_thinking | default(false) -%}"
    old_chat = build_chat_block({}, template_text=old)
    explicit_off = build_chat_block(
        {"default_chat_template_kwargs": {"enable_thinking": False}},
        template_text=old.replace("default(false)", "default(true)"),
    )

    assert old_chat["reasoning"]["default_enabled"] is False
    assert old_chat["reasoning"]["default_mode"] == "no_think"
    assert explicit_off["reasoning"]["default_enabled"] is False
    assert explicit_off["template_kwargs_defaults"] == {
        "enable_thinking": False,
    }


def test_laguna_chat_block_fills_card_documented_top_k_when_vendor_omits_it():
    """XS-2.1's generation_config.json has no top_k, but its card states all
    XS-2.1 benchmarking ran at temperature=1.0, top_k=20, top_p=1. Shipping
    without it makes runtimes sample the full vocab unfiltered at temp 1.0.
    An explicit vendor value must still win over the card fallback."""
    from jang_tools.convert_laguna_jang import build_chat_block

    xs_gen = {  # verbatim poolside/Laguna-XS-2.1 sampling keys
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "default_chat_template_kwargs": {"enable_thinking": True},
    }
    chat = build_chat_block(xs_gen)

    assert chat["sampling_defaults"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 20,
    }

    explicit = build_chat_block({**xs_gen, "top_k": 5})
    assert explicit["sampling_defaults"]["top_k"] == 5


def test_laguna_template_thinking_default_realigns_to_explicit_vendor_kwarg():
    """Raptor-1.0-16B ships a laguna_glm_thinking_v8-derived template that
    still falls back to default(false) while its generation_config declares
    enable_thinking=true (as the whole shipped Laguna-2.1 family does). The
    copied template must be realigned to the explicit kwarg, or the bundle
    reasons or not depending on which consumer renders it."""
    from jang_tools.convert_laguna_jang import (
        _set_template_thinking_default,
        _template_default_enable_thinking,
    )

    raptor = (
        '{#- Iteration on laguna_glm_thinking_v8/chat_template.jinja -#}\n'
        "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
        "{%- set add_generation_prompt = add_generation_prompt | default(false) -%}\n"
    )
    assert _template_default_enable_thinking(raptor) is False

    aligned, n = _set_template_thinking_default(raptor, True)
    assert n == 1, "must rewrite exactly the thinking fallback"
    assert _template_default_enable_thinking(aligned) is True
    # The neighbouring default(false) call must be left untouched.
    assert "add_generation_prompt | default(false)" in aligned

    # Idempotent, and reversible.
    assert _set_template_thinking_default(aligned, True)[0] == aligned
    assert _set_template_thinking_default(aligned, False)[0] == raptor

    # A template with no such fallback reports 0 so the converter can refuse.
    assert _set_template_thinking_default("{{- 'hi' -}}", True)[1] == 0


def test_laguna_qat_symmetric_grid_maps_losslessly_into_jang_affine():
    """The Path B container swap must be exact: MLX dequantizes affine as
    q*scale + bias, so codes q+8 with bias -8*scale must evaluate to the
    symmetric QAT grid q_sym*scale with ZERO error. If this drifts, every
    Raptor expert silently shifts off the grid it was trained on."""
    import mlx.core as mx
    import numpy as np

    from jang_tools.convert_laguna_jang import pack_affine4

    rng = np.random.default_rng(0)
    q_sym = rng.integers(-8, 8, size=(4, 256)).astype(np.int8)
    scale = (rng.random((4, 2)).astype(np.float32) + 0.1) / 7.5

    packed = pack_affine4((q_sym.astype(np.int16) + 8).astype(np.uint8))
    deq = np.array(mx.dequantize(
        mx.array(packed), mx.array(scale), mx.array(-8.0 * scale),
        group_size=128, bits=4).astype(mx.float32))

    expect = (q_sym.reshape(4, 2, 128) * scale[..., None]).reshape(4, 256)
    assert np.array_equal(deq, expect), "affine container is not lossless"
    # And the grid invariant CONVERT.md gates on.
    assert max(len(np.unique(g)) for g in deq.reshape(-1, 128)) <= 16


def test_laguna_runtime_honours_per_module_group_size_for_mixed_grids():
    """A Raptor bundle legitimately MIXES grids: routed experts are locked to
    the certified QAT grid's 128 while non-experts keep the proven 64. The
    loader read one global group_size, so a mixed bundle could not be
    expressed — and forcing everything to 128 coarsened the non-expert path
    enough to degenerate greedy decode. Pin both the override normalisation
    (config keys carry the `model.` prefix that _remap strips) and the
    in_features shape maths that makes mixed grids load at all."""
    qcfg = {
        "bits": 8, "group_size": 64, "mode": "affine",
        "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 64},
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 4, "group_size": 128},
        "lm_head": {"bits": 8, "group_size": 64},
    }
    # Mirror the loader's normalisation + lookup.
    ovr = {(k[6:] if k.startswith("model.") else k): v
           for k, v in qcfg.items() if isinstance(v, dict)}

    def module_group_size(name):
        g = ovr.get(name, {}).get("group_size")
        return int(g) if g else qcfg["group_size"]

    assert module_group_size("layers.1.mlp.switch_mlp.gate_proj") == 128
    assert module_group_size("layers.0.self_attn.q_proj") == 64
    assert module_group_size("lm_head") == 64
    assert module_group_size("layers.7.mlp.shared_expert.up_proj") == 64  # default

    # Shape round-trip: bits must be recoverable from packed/scales widths
    # using the module's OWN group size. in=2048 at 4-bit/gs128 packs to
    # 2048*4/32 = 256 words with 2048/128 = 16 scales; the same tensor read
    # with the global 64 would derive 8-bit and load garbage.
    in_features, gs, bits_true = 2048, 128, 4
    scales_last = in_features // gs
    packed_last = in_features * bits_true // 32
    derived = round(packed_last * 32 / (scales_last * gs))
    assert derived == bits_true
    wrong = round(packed_last * 32 / (scales_last * 64))
    assert wrong != bits_true, "global-group_size read must be the broken one"


def test_laguna_3l_and_4m_only_move_ffn_bits():
    from jang_tools.convert_laguna_jang import profile_policy

    p3 = profile_policy("JANG_3L")
    assert p3.group_size == 64
    assert p3.routed_bits == {"gate_proj": 3, "up_proj": 3, "down_proj": 4}
    assert p3.attention_bits == 8
    assert p3.shared_expert_bits == 6

    p4 = profile_policy("JANG_4M")
    assert p4.group_size == 64
    assert p4.routed_bits == {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
    assert p4.attention_bits == 8
    assert p4.shared_expert_bits == 8
    assert p4.dense_ffn_bits == 8

    p6 = profile_policy("JANG_6M")
    assert p6.group_size == 64
    assert p6.routed_bits == {"gate_proj": 6, "up_proj": 6, "down_proj": 6}
    assert p6.attention_bits == 8
    assert p6.shared_expert_bits == 8
    assert p6.dense_ffn_bits == 8
