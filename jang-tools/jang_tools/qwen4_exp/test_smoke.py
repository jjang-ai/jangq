"""Smoke + cache-consistency tests on a tiny random qwen4_exp.

The load-bearing check: logits from a single full prefill must match
(1) chunked prefill and (2) token-by-token cached decode — at lengths that
do NOT divide evenly by the QSA compress ratio, the GDN conv kernel, or the
PLE conv span. This exercises every cache path (GDN state, PLE token ctx,
PLE conv state, QSA KV + indexer keys).
"""

import mlx.core as mx
import numpy as np

from jang_tools.qwen4_exp.modeling import Model, Qwen4ExpTextArgs


def tiny_args() -> Qwen4ExpTextArgs:
    return Qwen4ExpTextArgs(
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=997,
        layer_types=None,
        full_attention_interval=4,
        linear_num_value_heads=6,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=1009,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=12,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,          # topk 2 blocks → selection actually bites
        indexer_compress_ratio=4,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        eos_token_id=7,
    )


def randomize(model: Model, scale=0.5):
    mx.random.seed(11)

    def rnd(p):
        if p.dtype in (mx.int32, mx.int64, mx.uint32):
            return p
        return mx.random.normal(p.shape).astype(p.dtype) * scale / max(1, p.shape[-1]) ** 0.5

    from mlx.utils import tree_map

    model.update(tree_map(rnd, model.parameters()))


def main():
    args = tiny_args()
    model = Model(args)
    randomize(model)
    mx.eval(model.parameters())

    rng = np.random.default_rng(3)
    S = 29  # not divisible by 4 → tail block + non-aligned GDN chunk
    ids = rng.integers(0, args.vocab_size, size=(1, S))
    ids[0, 11] = args.eos_token_id  # exercise EOS segmentation in PLE
    ids_mx = mx.array(ids)

    # 1) single-shot prefill (no cache)
    ref = model(ids_mx)
    mx.eval(ref)
    assert not np.isnan(np.asarray(ref)).any(), "NaN in reference logits"

    # 2) full prefill WITH cache should equal no-cache
    cache = model.make_cache()
    full_cached = model(ids_mx, cache=cache)
    d = np.abs(np.asarray(full_cached - ref)).max()
    assert d < 1e-4, f"cached-prefill mismatch {d}"

    # 3) chunked prefill: 13 + 9 + 7 (all non-aligned)
    cache = model.make_cache()
    outs = []
    for chunk in (ids_mx[:, :13], ids_mx[:, 13:22], ids_mx[:, 22:]):
        outs.append(model(chunk, cache=cache))
    chunked = mx.concatenate(outs, axis=1)
    d = np.abs(np.asarray(chunked - ref)).max()
    assert d < 1e-4, f"chunked-prefill mismatch {d}"

    # 4) token-by-token decode
    cache = model.make_cache()
    outs = [model(ids_mx[:, t: t + 1], cache=cache) for t in range(S)]
    stepped = mx.concatenate(outs, axis=1)
    d = np.abs(np.asarray(stepped - ref)).max()
    assert d < 1e-4, f"stepwise-decode mismatch {d}"

    # 5) long-enough sequence that QSA selection drops blocks (S=61 > budget+)
    S2 = 61
    ids2 = mx.array(rng.integers(0, args.vocab_size, size=(1, S2)))
    ref2 = model(ids2)
    cache = model.make_cache()
    outs = [model(ids2[:, t: t + 1], cache=cache) for t in range(S2)]
    stepped2 = mx.concatenate(outs, axis=1)
    d = np.abs(np.asarray(stepped2 - ref2)).max()
    assert d < 1e-4, f"long stepwise mismatch {d} (QSA selection path)"

    print("ALL SMOKE / CACHE-CONSISTENCY TESTS PASSED")


if __name__ == "__main__":
    main()
