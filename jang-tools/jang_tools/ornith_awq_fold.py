"""AWQ salient-channel scaling with norm absorption for the qwen3_5 family.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-19.

Applies AWQ to a **loaded, still-sanitized** mlx_vlm model in place, before
`nn.quantize`. Intended to be called from the D-lane builder between load and
quantize.

THE FOLD
========
AWQ scales salient input channels of W up by `s`, which only preserves the
layer function if the activation is divided by the same `s`. That requires a
fold partner. On this family the partner is the producing RMSNorm, and the
grouping is unusually clean because a layer is *either* full-attention *or*
linear-attention, never both:

    input_layernorm          -> {q_proj, k_proj, v_proj}                (full-attn layer)
    input_layernorm          -> {in_proj_qkv, in_proj_z,
                                 in_proj_a, in_proj_b}                  (GDN layer)
    post_attention_layernorm -> {mlp.gate_proj, mlp.up_proj}            (dense MLP)

Confirmed by the capture: within a layer all consumers of a given norm share
one input, and their `tr(H)` values are identical (GDN in_proj_* all 4294.4,
attn q/k/v all 4926.2) — which is exactly what a shared producer implies. All
consumers of one norm therefore MUST receive the SAME `s`; there is only one
norm to absorb it.

NOT folded (no norm producer — left for the imatrix fit to handle):
  * `mlp.down_proj`    — input is the GLU product
  * `self_attn.o_proj`, `linear_attn.out_proj` — inputs are attention outputs
  * the whole vision tower, `lm_head`

THE +1 CONVENTION — why this is safe here
=========================================
`qwen3_5.sanitize()` stores norms zero-centered and the runtime adds +1. The
model in memory is already sanitized, so its norm is `b + 1` for a stored `b`.
We divide the in-memory norm by `s`, and the builder's existing un-sanitize
step subtracts 1.0 at save time, so the value written to disk is:

    stored' = (b + 1) / s - 1

which is exactly the absorption formula verified during the Ornith-397B AWQ
work. Doing the fold on the in-memory (sanitized) tensor is what makes the
convention line up — folding into the stored zero-centered value instead would
be wrong by the +1, and that is the hazard the Qwen3.6 playbook warned about.

ALPHA
=====
Default 0.25 is the canonical AWQ value and is right at 4-bit and above. Our
own Ornith-397B run measured alpha=0.25 producing garbage at 2.15-bit and
alpha=0.05 being a near-no-op, so at 2-3 bit the caller should pass a reduced
alpha and gate the result on KL rather than assume improvement.
"""
from __future__ import annotations

import mlx.core as mx

# suffix -> the norm that produces its input, within the same layer
_FOLD_GROUPS = {
    "input_layernorm": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    ),
    "post_attention_layernorm": (
        # dense MLP (9B)
        "mlp.gate_proj", "mlp.up_proj",
        # MoE (35B): the post-attention norm feeds the routed experts, the
        # shared expert AND the router — they all read the same activation, so
        # they must all take the same s. Omitting the routed experts here would
        # leave AWQ covering ~8% of a 36B model.
        # SwitchLinear weights are (num_experts, out, in); scaling the last
        # axis by s is the same per-input-channel operation as for a 2-D
        # Linear, so the fold is identical.
        "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
        "mlp.gate", "mlp.shared_expert_gate",
    ),
}

# Leaves that read something OTHER than the norm (an attention output or the
# GLU product), so they must NOT be scaled. Used by the completeness check.
_NOT_NORM_FED = (
    "down_proj", "o_proj", "out_proj",
)


def _second_moments(calib_path: str) -> dict:
    from safetensors.numpy import load_file
    raw = load_file(calib_path)
    return {k[: -len(".second_moment")]: v
            for k, v in raw.items() if k.endswith(".second_moment")}


def apply_awq(model, calib_path: str, alpha: float = 0.25,
              verbose: bool = True, scales_out: str | None = None) -> dict:
    """Scale salient channels and absorb the inverse into the producing norm.

    Operates in place on a loaded (sanitized) mlx_vlm model. Returns a summary.

    🚨 ``scales_out`` is NOT optional in the D-lane pipeline. ``qwen36_imatrix_refit``
    re-derives every module's codes from the ORIGINAL SOURCE weight, not from
    the bundle, so a later refit silently reverts ``W*s`` back to ``W`` while
    the folded norm stays divided by ``s`` — leaving the layer off by a factor
    of ``s`` per channel. Persisting the scales lets the refit re-apply them and
    keeps full source precision (refitting from dequantized codes would not).
    Symptom if this is skipped: the refit's reported rel-err is byte-identical
    with and without AWQ, because the codes really are identical.
    """
    sm = _second_moments(calib_path)
    if not sm:
        raise ValueError(f"no second_moment tensors in {calib_path}")

    mods = dict(model.named_modules())
    n_group = n_lin = 0
    skipped_no_stat = []
    saved_scales: dict = {}
    unscaled: set = set()

    for path, mod in list(mods.items()):
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in _FOLD_GROUPS:
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.ndim != 1:
            continue
        prefix = path[: -len(leaf)]          # e.g. "...layers.7."

        # Consumers of THIS norm that actually exist in this layer.
        consumers = []
        for suf in _FOLD_GROUPS[leaf]:
            m = mods.get(prefix + suf)
            if m is not None and getattr(m, "weight", None) is not None:
                consumers.append((prefix + suf, m))
        if not consumers:
            continue

        # 🚨 COMPLETENESS CHECK. The fold is only valid if EVERY module reading
        # this norm is scaled; miss one and it silently receives x/s unscaled.
        # Rather than trust the hand-written list, look for weight-bearing
        # siblings in the block this norm feeds that we did NOT scale.
        block = "mlp." if leaf == "post_attention_layernorm" else None
        if block:
            named = {p for p, _ in consumers}
            for p, m in mods.items():
                if not p.startswith(prefix + block):
                    continue
                # NB: must not shadow `w` — it is the norm's weight and is used
                # for the shape check below.
                mw = getattr(m, "weight", None)
                if mw is None or p in named:
                    continue
                if any(x in p.rsplit(".", 1)[-1] for x in _NOT_NORM_FED):
                    continue
                unscaled.add(p[len(prefix):])

        # One shared scale for the whole group: average the per-channel second
        # moments of the consumers that have a capture entry. They agree by
        # construction (shared input), so the mean is just the robust choice.
        stats = [sm[p] for p, _ in consumers if p in sm]
        if not stats:
            skipped_no_stat.append(prefix + leaf)
            continue
        import numpy as np
        mean_sq = np.mean(np.stack(stats, 0), 0)
        if mean_sq.shape[0] != w.shape[0]:
            skipped_no_stat.append(prefix + leaf)
            continue

        # AWQ scale from mean |x| ~ sqrt(E[x^2]); normalise to geometric mean 1
        # so the fold does not shift the overall magnitude of the layer.
        mag = np.sqrt(np.maximum(mean_sq, 0.0)) + 1e-8
        s = np.power(mag, alpha)
        s = s / np.exp(np.mean(np.log(s)))
        s = np.clip(s, 1e-2, 1e2).astype(np.float32)
        s_mx = mx.array(s)

        # W' = W * s  (scale INPUT channels = last axis)
        for p, m in consumers:
            m.weight = (m.weight.astype(mx.float32) * s_mx).astype(m.weight.dtype)
            saved_scales[p] = s
            n_lin += 1
        # norm' = norm / s   (in-memory value is b+1; builder writes b'=norm'/1 -1)
        mod.weight = (mod.weight.astype(mx.float32) / s_mx).astype(mod.weight.dtype)
        n_group += 1

    mx.eval(model.parameters())

    if scales_out and saved_scales:
        from safetensors.numpy import save_file
        save_file({f"{k}.awq_scale": v for k, v in saved_scales.items()},
                  scales_out)

    if unscaled:
        raise RuntimeError(
            "AWQ fold incomplete: these modules read a folded norm but were "
            "NOT scaled, so they would receive x/s unscaled — "
            f"{sorted(unscaled)}. Add them to _FOLD_GROUPS (or to "
            "_NOT_NORM_FED if they genuinely read a different activation).")

    if verbose:
        import numpy as np
        allv = np.concatenate([v for v in saved_scales.values()]) if saved_scales else np.ones(1)
        print(f"  AWQ alpha={alpha}: folded {n_group} norm groups covering "
              f"{n_lin} linears | s: min={allv.min():.3f} max={allv.max():.3f} "
              f"ratio={allv.max()/max(allv.min(),1e-9):.1f}x std={allv.std():.3f}"
              + (f"; {len(skipped_no_stat)} norms had no capture stat"
                 if skipped_no_stat else ""), flush=True)
        if scales_out:
            print(f"  AWQ scales -> {scales_out}", flush=True)
    return {"alpha": alpha, "groups": n_group, "linears": n_lin,
            "skipped": len(skipped_no_stat),
            "scales_path": scales_out or ""}
