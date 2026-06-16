from jang_tools.allocate import allocate_bits_profile, allocate_bits_profile_compact, classify_tensor, Tier


N2_LINEAR_OUT = "model.language_model.layers.0.linear_attn.out_proj.weight"
N2_LINEAR_IN = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
N2_EXPERT_DOWN = "model.language_model.layers.0.mlp.experts.down_proj.weight"
GENERIC_OUT = "model.language_model.layers.0.some_block.out_proj.weight"


def test_linear_attention_out_proj_is_important_not_generic_compress():
    assert classify_tensor(N2_LINEAR_OUT, num_experts=512) == Tier.IMPORTANT
    assert classify_tensor(N2_LINEAR_IN, num_experts=512) == Tier.IMPORTANT
    assert classify_tensor(GENERIC_OUT, num_experts=512) == Tier.COMPRESS


def test_jang1l_keeps_linear_attention_out_proj_at_8bit_compact():
    bits = allocate_bits_profile_compact(
        [(N2_LINEAR_OUT, 1), (N2_EXPERT_DOWN, 1), (GENERIC_OUT, 1)],
        "JANG_1L",
        num_experts=512,
        apply_mlp_asymmetry=False,
    )

    assert bits[N2_LINEAR_OUT] == 8
    assert bits[N2_EXPERT_DOWN] == 2
    assert bits[GENERIC_OUT] == 2


def test_jang1l_keeps_linear_attention_out_proj_at_8bit_array_allocator():
    bits = allocate_bits_profile(
        [N2_LINEAR_OUT, N2_EXPERT_DOWN, GENERIC_OUT],
        "JANG_1L",
        num_experts=512,
        apply_mlp_asymmetry=False,
    )

    assert bits.tolist() == [8, 2, 2]
