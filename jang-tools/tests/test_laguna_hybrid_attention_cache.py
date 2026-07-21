"""Laguna hybrid attention + cache correctness proofs (tiny random weights).

Laguna interleaves full attention with sliding-window attention (S-2.1:
1:3 ratio, window 512) and serves them from mixed caches (KVCache +
RotatingKVCache). Three failure modes are invisible to short smoke prompts
and only bite past the window:

1. Prefill masking: feeding ONE full causal mask to every layer lets SWA
   layers attend the whole prefix when the prompt exceeds the window.
2. Cache semantics: RotatingKVCache(keep=4) retains the first 4 tokens as
   attention sinks — the HF reference only does that behind
   swa_attention_sink_enabled, which no shipped Laguna config sets.
3. Divergence between the no-cache path (full recompute per step, exercises
   T>1 masks) and the cached path (RotatingKVCache decode) — these are two
   implementations of the same math and must agree token-for-token.

These tests run the real LagunaForCausalLM at toy dimensions so the proofs
execute in milliseconds with no weights on disk. The window here is 8; every
sequence is chosen to CROSS it, because parity below the window is trivially
true for all three bugs.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from jang_tools.laguna.config import LagunaConfig
from jang_tools.laguna.model import LagunaForCausalLM

WINDOW = 8


def _tiny_cfg(layer_types: list[str], heads_per_layer: list[int] | None = None):
    n = len(layer_types)
    return LagunaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=WINDOW,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=list(layer_types),
        mlp_layer_types=["dense"] + ["sparse"] * (n - 1),
        num_attention_heads_per_layer=heads_per_layer or [4] * n,
        rope_parameters={
            # Same shape as S-2.1: full = partial rotary 0.5 at theta 500k,
            # sliding = full rotary at theta 10k.
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0,
                               "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0,
                                  "partial_rotary_factor": 1.0},
        },
    )


def _model(cfg: LagunaConfig, seed: int = 0) -> LagunaForCausalLM:
    mx.random.seed(seed)
    model = LagunaForCausalLM(cfg)
    # Perturb default inits so logits aren't degenerate.
    import mlx.utils as u
    params = u.tree_map(
        lambda p: p + 0.02 * mx.random.normal(p.shape), model.parameters())
    model.update(params)
    mx.eval(model.parameters())
    return model


def _logits_last(model, ids: list[int]) -> mx.array:
    logits, _ = model(mx.array([ids], dtype=mx.uint32), caches=None)
    return logits[0, -1]


# ── 1. sliding mask actually restricts the receptive field ───────────────

def test_swa_prefill_mask_blocks_tokens_beyond_window():
    """Single all-SWA layer -> the receptive field IS the window. Perturbing
    a token >= WINDOW back from the end must not change the last logits;
    perturbing one inside the window must. With the old shared full-causal
    mask, the far perturbation leaks through and this fails."""
    cfg = _tiny_cfg(["sliding_attention"], heads_per_layer=[4])
    cfg.mlp_layer_types = ["dense"]  # single layer: keep it dense
    model = _model(cfg)

    base = list(range(1, 21))          # 20 tokens > window 8
    ref = _logits_last(model, base)

    far = list(base)
    far[len(base) - WINDOW - 3] = 90   # outside the window of the last token
    assert mx.allclose(ref, _logits_last(model, far), atol=1e-5), \
        "token beyond the sliding window changed the output — SWA mask leaks"

    near = list(base)
    near[len(base) - 2] = 90           # inside the window
    assert not mx.allclose(ref, _logits_last(model, near), atol=1e-4), \
        "in-window perturbation had no effect — mask over-restricts"


def test_full_attention_layer_sees_beyond_window():
    """Control for the test above: a full-attention layer MUST see past the
    window — if this fails, the sliding mask got applied to full layers."""
    cfg = _tiny_cfg(["full_attention"], heads_per_layer=[4])
    cfg.mlp_layer_types = ["dense"]
    model = _model(cfg)

    base = list(range(1, 21))
    ref = _logits_last(model, base)
    far = list(base)
    far[2] = 90                        # way outside the window
    assert not mx.allclose(ref, _logits_last(model, far), atol=1e-4)


def test_hybrid_stack_dispatches_windowed_mask_to_swa_layers():
    """Structural proof for the S-2.1-shaped bug: with layer 0 = full
    attention, caches[0] is a plain KVCache, so a port that builds ONE mask
    from caches[0] hands every SWA layer an unwindowed causal mask. The
    behavioral tests above can't isolate this in a hybrid stack (the full
    layer legitimately sees the whole prefix), so record the mask each
    layer actually receives and check the window on the SWA ones."""
    from jang_tools.laguna.model import LagunaAttention

    cfg = _tiny_cfg(
        ["full_attention", "sliding_attention", "sliding_attention",
         "sliding_attention"],
        heads_per_layer=[4, 8, 8, 8],
    )
    model = _model(cfg)

    seen: list[tuple[str, object]] = []
    orig = LagunaAttention.__call__

    def recording(self, x, mask=None, cache=None):
        seen.append((self.layer_type, mask))
        return orig(self, x, mask=mask, cache=cache)

    LagunaAttention.__call__ = recording
    try:
        T = 20  # > window
        model(mx.array([[(i * 5) % 97 for i in range(T)]], dtype=mx.uint32),
              caches=None)
    finally:
        LagunaAttention.__call__ = orig

    assert len(seen) == 4
    for layer_type, mask in seen:
        if layer_type == "full_attention":
            # mlx_lm returns the SDPA fast-path string for plain causal.
            blocked = (mask == "causal") or (
                isinstance(mask, mx.array) and bool(
                    (mask[..., T - 1, 0] < -1e8).item()) is False)
            assert blocked, f"full layer got unexpected mask {mask!r}"
        else:
            assert isinstance(mask, mx.array), (
                f"SWA layer got {mask!r} — the unwindowed shared mask")
            row = mask[..., T - 1, :]
            far, near = row[..., 0], row[..., T - 1]
            if mask.dtype == mx.bool_:
                assert not bool(far.item()) and bool(near.item()), \
                    "SWA mask does not window the prefix"
            else:
                assert bool((far < -1e8).item()) and bool((near > -1e8).item()), \
                    "SWA mask does not window the prefix"


# ── 2 + 3. cached decode == no-cache recompute across the window ─────────

def _greedy_no_cache(model, ids: list[int], n: int) -> list[int]:
    out = list(ids)
    for _ in range(n):
        out.append(int(mx.argmax(_logits_last(model, out)).item()))
    return out


def _greedy_cached(model, ids: list[int], n: int) -> list[int]:
    out = list(ids)
    logits, caches = model(mx.array([ids], dtype=mx.uint32), caches=None)
    for _ in range(n):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        logits, caches = model(mx.array([[nxt]], dtype=mx.uint32), caches=caches)
    return out


@pytest.mark.parametrize("prompt_len", [4, 20])
def test_cached_greedy_matches_no_cache_across_window(prompt_len):
    """S-2.1-shaped hybrid stack (full + 3x SWA, mixed per-layer head counts,
    dense layer 0 + MoE). Decode runs long enough that prompt+decode crosses
    the window even from the short prompt, so RotatingKVCache eviction,
    the banded prefill mask, and the rope offsets all get exercised. The
    keep=4 sink bug and the shared-mask bug each break the long case."""
    cfg = _tiny_cfg(
        ["full_attention", "sliding_attention", "sliding_attention",
         "sliding_attention"],
        heads_per_layer=[4, 8, 8, 8],   # per-layer head counts, like 48/72
    )
    model = _model(cfg)

    prompt = [(i * 7) % 97 for i in range(prompt_len)]
    n_new = WINDOW + 6

    no_cache = _greedy_no_cache(model, prompt, n_new)
    cached = _greedy_cached(model, prompt, n_new)

    assert no_cache == cached, (
        f"cache/no-cache divergence at token "
        f"{next(i for i, (a, b) in enumerate(zip(no_cache, cached)) if a != b)}"
        f" (prompt {prompt_len}, window {WINDOW})"
    )
