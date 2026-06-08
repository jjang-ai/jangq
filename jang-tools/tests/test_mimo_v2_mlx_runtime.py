import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from jang_tools.mimo_v2.mlx_model import MiMoV2Attention, Model, ModelArgs


def _tiny_mimo_args() -> ModelArgs:
    return ModelArgs(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        head_dim=4,
        v_head_dim=4,
        swa_head_dim=4,
        swa_v_head_dim=4,
        partial_rotary_factor=0.5,
        sliding_window=3,
        add_swa_attention_sink_bias=True,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 0],
        n_routed_experts=4,
        num_experts_per_tok=2,
    )


def test_mimo_v2_make_cache_uses_rotating_cache_for_swa_layers():
    model = Model(_tiny_mimo_args())

    cache = model.make_cache()

    assert isinstance(cache[0], KVCache)
    assert isinstance(cache[1], RotatingKVCache)
    assert cache[1].max_size == 3


def test_mimo_v2_backbone_builds_per_layer_sliding_mask(monkeypatch):
    model = Model(_tiny_mimo_args())
    cache = model.make_cache()
    seen_masks = []

    def capture_attention(self, x, mask=None, cache=None):
        seen_masks.append((self.layer_idx, mask))
        return mx.zeros_like(x)

    monkeypatch.setattr(MiMoV2Attention, "__call__", capture_attention)

    model.model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)

    assert seen_masks[0][0] == 0
    assert seen_masks[0][1] == "causal"
    assert seen_masks[1][0] == 1
    assert seen_masks[1][1] is not None
    assert seen_masks[1][1] != "causal"
    assert tuple(seen_masks[1][1].shape) == (5, 5)
