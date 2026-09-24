"""Native MTP (Multi-Token Prediction) head + D2/D3 speculative decode for
Nemotron 3.5 Lightning (nemotron_h).

The checkpoint ships a real MTP head — 1.335 B params, DeepSeek-V3 shape:

    h_mtp  = eh_proj( concat( enorm(embed(t_{i+1})), hnorm(h_i) ) )
    h_mtp  = layers[0](h_mtp)     # GQA attention block, own KV cache
    h_mtp  = layers[1](h_mtp)     # MoE block (128 experts + shared)
    logits = lm_head( final_layernorm(h_mtp) )      -> distribution for t_{i+2}

It shares `backbone.embeddings` and `lm_head` with the main model, which is what
makes recursive re-application (D3) structurally possible from a single head.

`mlx_lm.models.nemotron_h.sanitize()` drops every `mtp.*` key at load, so this
module loads them separately from the bundle and attaches them to a loaded model.

Cache restoration requirements:
  * KV cache (6 attention layers) is position-addressable -> rewind `offset`.
  * Mamba `ArraysCache` (23 layers) is overwritten IN PLACE and CANNOT be
    trimmed -> must be snapshotted before the verify forward and restored on
    rejection. Snapshots are deep-copied AND materialized, because an MLX array
    is a lazy graph node and a bare reference would be mutated by the next update.
  * The MTP head's own KV cache must stay in lockstep with the backbone.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import NemotronHBlock


class _MTPEntryLayer(nn.Module):
    """mtp.layers.0 — embedding/hidden fusion + an attention block."""

    def __init__(self, args):
        super().__init__()
        h = args.hidden_size
        self.enorm = nn.RMSNorm(h, eps=args.layer_norm_epsilon)
        self.hnorm = nn.RMSNorm(h, eps=args.layer_norm_epsilon)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        blk = NemotronHBlock(args, "*")
        self.norm = blk.norm
        self.mixer = blk.mixer

    def fuse(self, embed_next: mx.array, hidden: mx.array) -> mx.array:
        return self.eh_proj(
            mx.concatenate([self.enorm(embed_next), self.hnorm(hidden)], axis=-1)
        )

    def __call__(self, x, mask=None, cache=None):
        return x + self.mixer(self.norm(x), mask=mask, cache=cache)


class _MTPMoELayer(nn.Module):
    """mtp.layers.1 — MoE block + the head's final norm."""

    def __init__(self, args):
        super().__init__()
        blk = NemotronHBlock(args, "E")
        self.norm = blk.norm
        self.mixer = blk.mixer
        self.final_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.layer_norm_epsilon
        )

    def __call__(self, x):
        return x + self.mixer(self.norm(x))


class NemotronHMTP(nn.Module):
    """The `mtp.` subtree. Weight names match the checkpoint exactly."""

    def __init__(self, args):
        super().__init__()
        self.layers = [_MTPEntryLayer(args), _MTPMoELayer(args)]

    def __call__(self, embed_next, hidden, cache=None):
        x = self.layers[0].fuse(embed_next, hidden)
        x = self.layers[0](x, mask=None, cache=cache)
        x = self.layers[1](x)
        return self.layers[1].final_layernorm(x)


def load_mtp(model, bundle: str | Path):
    """Build the MTP head and load `mtp.*` from *bundle* onto *model*.

    Returns the head, or None when the bundle carries no MTP weights.
    Honors the per-module quantization overrides in config.json.
    """
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    qcfg = cfg.get("quantization", {}) or {}

    weights: dict[str, mx.array] = {}
    for sf in sorted(glob.glob(str(bundle / "*.safetensors"))):
        for k, v in mx.load(sf).items():
            if k.startswith("mtp."):
                weights[k[len("mtp."):]] = v
    if not weights:
        return None

    head = NemotronHMTP(model.args)

    # Quantize exactly the modules the bundle quantized, at their own widths.
    def predicate(path: str, module) -> Any:
        ov = qcfg.get(f"mtp.{path}")
        if isinstance(ov, dict):
            return {"group_size": ov["group_size"], "bits": ov["bits"]}
        return False

    if qcfg:
        nn.quantize(
            head,
            group_size=qcfg.get("group_size", 64),
            bits=qcfg.get("bits", 4),
            class_predicate=predicate,
        )

    head.load_weights(list(weights.items()), strict=False)
    mx.eval(head.parameters())
    return head


# ── cache helpers ────────────────────────────────────────────────────────────

def cache_slots(model):
    """(ssm_slot_indices, attn_slot_indices) into the COMPACTED cache list.

    The cache list has one entry per mamba/attention layer only — MoE layers
    get none — so cache index != layer index. For Lightning 3.5 that is 29
    slots, with attention at [3, 7, 11, 15, 19, 24].
    """
    ssm, attn, i = [], [], 0
    for layer in model.layers:
        if layer.block_type == "M":
            ssm.append(i); i += 1
        elif layer.block_type == "*":
            attn.append(i); i += 1
    return ssm, attn


def snapshot_ssm(cache, ssm_slots):
    """Snapshot the 23 Mamba states. **Shallow list copy — no data copy.**

    Safe because of two facts about the mlx_lm Python path, both verified:

      1. `ArraysCache.state` is a plain Python list, and the Mamba mixer updates
         it by REBINDING slots (`cache[0] = ...`, `cache[1] = ...` in
         nemotron_h.py) — never by mutating an array in place.
      2. MLX arrays are immutable values, so the array objects we retain cannot
         be changed underneath us.

    So retaining the old list is a valid point-in-time snapshot at zero cost.
    Measured: a deep copy (`a * 1` + `mx.eval`) of the 47 MB of SSM state cost
    3.89 ms per step — ~48 % of a full single-token forward (8.15 ms) — which
    on its own made D2 speculative decoding a net LOSS on this model.

    NOTE: the Swift runtime genuinely does need the deep copy
    (`MambaCache.recordPrefixCommitState` does `$0 * 1` + `MLX.eval`) because
    that cache exposes a mutable buffer. Do not port this shortcut there
    without re-verifying the same two properties hold.
    """
    return [list(cache[i].state) for i in ssm_slots]


def restore_ssm(cache, ssm_slots, snap):
    for j, i in enumerate(ssm_slots):
        cache[i].state = snap[j]


def rewind_kv(cache, attn_slots, n: int):
    """Rejecting n drafts = drop the last n positions. KV is position-addressable."""
    for i in attn_slots:
        cache[i].offset -= n
