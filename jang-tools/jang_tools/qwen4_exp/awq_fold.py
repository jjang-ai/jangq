"""AWQ scale folding for Qwen4-Exp's mHC GatedResidual sites.

Standard AWQ: pick per-channel scales s (salience^α), then
    activations ← activations / s     (folded into the producing norm)
    consumer W  ← W · diag(s)         (input columns scaled up)
so the product is unchanged but salient channels quantize finer.

The mHC twist: hc_norm's output feeds THREE places — the stream mixer
(input_mix_weight_down), the injection gate (block_inject_weight) and, via
the mixed stream mean, the block's input projections. Scaling hc_norm by 1/s
therefore requires EXACT compensation (×s on input columns) of the two gate
linears, which sit BEFORE their nonlinearities, plus the usual ×s on every
consumer of the block input:

  attn site (GDN):   in_proj_qkv, in_proj_z, in_proj_a, in_proj_b
  attn site (QSA):   q_proj, k_proj, v_proj, indexer.index_qk_proj
  mlp  site:         mlp.gate (router!), switch_mlp.gate_proj/up_proj (per
                     expert, input axis), shared_expert.gate_proj/up_proj,
                     shared_expert_gate
  final mixer:       lm_head

The residual stream itself is untouched (only normed copies are scaled), so
every other layer is unaffected. Scales are broadcast ×hc_count across the
stream copies. All folds operate on RUNTIME weights (norm already +1-shifted);
converting back to checkpoint form is (w_run − 1) at save time.
"""

import mlx.core as mx
import numpy as np


def _tile(s: mx.array, hc_count: int) -> mx.array:
    return mx.tile(s, (hc_count,))


def _scale_linear_in(linear, s: mx.array):
    """W[out, in] ← W · diag(s) — also handles SwitchLinear [E, out, in]."""
    linear.weight = (linear.weight * s.astype(linear.weight.dtype)).astype(linear.weight.dtype)


def fold_gated_residual(gr, consumers, s: mx.array):
    """gr: GatedResidual; consumers: list of nn.Linear/SwitchLinear whose
    input is the mixed block input; s: [hidden] positive scales."""
    st = _tile(s, gr.hc_count)
    gr.hc_norm.weight = (gr.hc_norm.weight / st).astype(gr.hc_norm.weight.dtype)
    _scale_linear_in(gr.input_mix_weight_down, st)
    if gr.use_combine:
        _scale_linear_in(gr.block_inject_weight, st)
    for c in consumers:
        _scale_linear_in(c, s)


def attn_site_consumers(layer):
    if layer.is_linear:
        la = layer.linear_attn
        return [la.in_proj_qkv, la.in_proj_z, la.in_proj_a, la.in_proj_b]
    sa = layer.self_attn
    return [sa.q_proj, sa.k_proj, sa.v_proj, sa.indexer.index_qk_proj]


def mlp_site_consumers(layer):
    m = layer.mlp
    return [
        m.gate,
        m.switch_mlp.gate_proj,
        m.switch_mlp.up_proj,
        m.shared_expert.gate_proj,
        m.shared_expert.up_proj,
        m.shared_expert_gate,
    ]


def fold_attn_site(layer, s: mx.array):
    fold_gated_residual(layer.attn_hyper_connection, attn_site_consumers(layer), s)


def fold_mlp_site(layer, s: mx.array):
    fold_gated_residual(layer.mlp_hyper_connection, mlp_site_consumers(layer), s)


def fold_final_mixer(model, s: mx.array):
    fold_gated_residual(model.language_model.hyper_connection_mixer, [model.lm_head], s)


def prove_invariance(model, ids: mx.array) -> float:
    """Fold random positive scales into EVERY site and measure the deviation
    of the LAST hidden state (relative to its own magnitude).

    Measured floor (2026-08-26, tiny random model): s=1.0 is bit-exact;
    any s≠1 rewrites weight bits, and that ulp-level rounding noise —
    magnitude-INDEPENDENT of s — is amplified ~2× per mHC injection site
    (16 sites → ~3e-5 at 8 layers). This is the fp32 noise floor of an
    algebraically exact fold, far below bf16 resolution (2^-8). Judge
    against hidden states, not tiny random logits, whose small magnitude
    inflates the relative number ~20×.
    """
    def last_hidden(m):
        h = m.language_model.embed_tokens(ids)
        h = mx.tile(h, (1, 1, m.args.hc_count))
        for layer in m.language_model.layers:
            h = layer(h, mask=None, cache=None, input_ids=ids)
        return np.asarray(h)

    ref = last_hidden(model)
    rng = np.random.default_rng(7)
    hidden = model.args.hidden_size
    for layer in model.language_model.layers:
        fold_attn_site(layer, mx.array(np.exp(rng.normal(0, 0.5, hidden)).astype(np.float32)))
        fold_mlp_site(layer, mx.array(np.exp(rng.normal(0, 0.5, hidden)).astype(np.float32)))
    fold_final_mixer(model, mx.array(np.exp(rng.normal(0, 0.5, hidden)).astype(np.float32)))
    got = last_hidden(model)
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    return float(rel)
