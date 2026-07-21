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
    default_chat_template_kwargs — the vendor serving default is
    enable_thinking=true while the template's own fallback is false, so a
    consumer that drops the kwargs silently runs no-think."""
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


def test_laguna_chat_block_defaults_thinking_off_without_template_kwargs():
    """Without default_chat_template_kwargs the template fallback (false)
    governs — default_enabled must not be invented as true."""
    from jang_tools.convert_laguna_jang import build_chat_block

    chat = build_chat_block({"temperature": 0.7})

    assert chat["reasoning"]["default_enabled"] is False
    assert chat["sampling_defaults"] == {"temperature": 0.7}
    assert chat["template_kwargs_defaults"] == {}


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
