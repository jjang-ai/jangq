"""DeepSeek-V4.1 dSpark (MTP) draft head in MLX + the speculative accept/reject loop.

Faithful port of the release DSparkBlock / forward_embed / forward_head / forward_spec
(inference/model.py). dSpark is three MTP stages appended after the 40-layer backbone. Each stage
mirrors a backbone Block (latent attention over a 128-token window + a 128-expert top-3 MoE + hyper-
connections) but:
  - its attention window is seeded from the backbone's hidden state, not from its own tokens;
  - stage 0 carries main_proj/main_norm that project the concatenated mean-over-hc hidden of the
    backbone target layers (37,38,39) down to one hidden;
  - the last stage carries a final norm, a Markov bias head (embed+head over a rank-256 space) and a
    confidence head.
The stages share the backbone's token embedding and output head.

forward_spec drafts a block of `dspark_block_size` (5) tokens from the last committed token plus the
backbone's target-layer hidden; the Markov head sharpens each step autoregressively and the
confidence head scores the block. The release ships only this forward -- the accept/reject loop that
turns drafts into a speedup is built here (`DSparkRunner`).

The draft is loaded at bf16 from the SOURCE model shards (the JANG bundle drops mtp.*), with the
routed experts quantized to 4-bit in memory so the whole draft is ~8 GB on top of the backbone.
Verification is exact, so draft precision only moves the acceptance rate (the speedup), never the
output: speculative decoding here is provably token-identical to plain greedy decode.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jang_tools.dsv41.model import (
    GROUP_SIZE,
    HC_METAL,
    DenseLinear,
    GrowBuffer,
    RMSNorm,
    Rope,
    SwitchProj,
    _hc_post,
    _hc_pre,
    _hc_split,
    _hc_split_metal,
    sparse_attention,
    yarn_inv_freq,
)
from jang_tools.dsv41.st_io import Shards


# --------------------------------------------------------------------------- hyper-connections
def _hc_mixes(x, fn, scale, base, hc, iters, hc_eps, eps):
    """pre/post/comb mixing weights for one hc sub-block (release Block.hc_mixes)."""
    B, S = x.shape[:2]
    xf = x.reshape(B, S, -1).astype(mx.float32)
    rs = mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + eps)
    m = (xf @ fn.T) * rs
    if hc == 4 and HC_METAL:
        return _hc_split_metal(m, scale, base, iters, hc_eps)
    return _hc_split(hc, iters, hc_eps)(m, scale, base)


# --------------------------------------------------------------------------- draft attention
class DSparkAttention(nn.Module):
    """Latent attention whose KV window is main-hidden-derived; bf16 (draft precision)."""

    def __init__(self, args):
        super().__init__()
        d, H = args.hidden_size, args.num_attention_heads
        self.n_heads, self.head_dim, self.rd = H, args.head_dim, args.qk_rope_head_dim
        self.n_groups, self.o_lora_rank = args.o_groups, args.o_lora_rank
        self.win = args.sliding_window
        self.scale = self.head_dim ** -0.5
        self.attn_sink = mx.zeros((H,), mx.float32)
        self.wq_a = DenseLinear(d, args.q_lora_rank)
        self.q_norm = RMSNorm(args.q_lora_rank, args.rms_norm_eps)
        self.wq_b = DenseLinear(args.q_lora_rank, H * self.head_dim)
        self.wkv = DenseLinear(d, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.wo_a = mx.zeros((self.n_groups, self.o_lora_rank, H * self.head_dim // self.n_groups), mx.bfloat16)
        self.wo_b = DenseLinear(self.n_groups * self.o_lora_rank, d)
        inv = yarn_inv_freq(self.rd, args.rope_theta, 0, 1, 0, 0)  # dspark attn is never compressed -> plain rope
        self._rope = Rope(self.rd, inv)
        self.window = GrowBuffer()  # main-hidden-derived KV for committed positions (rope'd), oldest first

    def seed(self, main_x, positions):
        """Append the committed positions' main-derived KV to the window (release prefill/decode seed)."""
        kv = self._rope(self.kv_norm(self.wkv(main_x)), positions)  # [1, S, head_dim]
        self.window.append(kv.astype(mx.bfloat16))

    def attend(self, x, positions):
        """Draft block attention: queries x [1, bs, d] attend to (last-`win` window ++ own KV)."""
        B, bs, _ = x.shape
        qr = self.q_norm(self.wq_a(x))
        q = self._rope(self.wq_b(qr).reshape(B, bs, self.n_heads, self.head_dim), positions)
        kv = self._rope(self.kv_norm(self.wkv(x)), positions)  # [1, bs, head_dim]
        win = self.window.view()
        win = win[:, -self.win:] if win is not None else None
        full = mx.concatenate([win, kv], axis=1) if win is not None else kv  # [1, L, head_dim]
        L = full.shape[1]
        idxs = mx.broadcast_to(mx.arange(L, dtype=mx.int32)[None, None], (1, bs, L))  # every query sees all
        o = sparse_attention(q, full, self.attn_sink, idxs, self.scale)  # [1, bs, H, head_dim]
        o = self._rope(o, positions, inverse=True).astype(mx.bfloat16)
        o = o.reshape(bs, self.n_groups, -1)  # [bs, groups, (H/groups)*head_dim]
        o = mx.einsum("sgd,grd->sgr", o, self.wo_a)  # [bs, groups, o_lora_rank]
        return self.wo_b(o.reshape(1, bs, -1))  # [1, bs, d]


# --------------------------------------------------------------------------- draft MoE
class DSparkMoE(nn.Module):
    """128-expert top-3 MoE, routed experts 4-bit (gather_qmm), shared expert + gate bf16/f32."""

    def __init__(self, args, n_experts, inter, bits=4):
        super().__init__()
        d = args.hidden_size
        self.topk = args.raw["dspark_num_experts_per_tok"]
        self.route_scale = args.routed_scaling_factor
        self.norm_topk = args.norm_topk_prob
        self.lim = args.swiglu_limit
        self.gate_w = mx.zeros((n_experts, d), mx.bfloat16)
        self.gate_bias = mx.zeros((n_experts,), mx.float32)
        self.gate_proj = SwitchProj(n_experts, d, inter, bits)
        self.up_proj = SwitchProj(n_experts, d, inter, bits)
        self.down_proj = SwitchProj(n_experts, inter, d, bits)
        self.sw1 = DenseLinear(d, inter)
        self.sw3 = DenseLinear(d, inter)
        self.sw2 = DenseLinear(inter, d)

    def __call__(self, x):
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        scores = mx.sqrt(nn.softplus(xf.astype(mx.float32) @ self.gate_w.astype(mx.float32).T))
        idx = mx.argpartition(-(scores + self.gate_bias), kth=self.topk - 1, axis=-1)[:, : self.topk]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk and self.topk > 1:
            w = w / (mx.sum(w, -1, keepdims=True) + 1e-20)
        w = w * self.route_scale
        n = xf.shape[0]
        xe = xf[:, None, None, :]  # [N,1,1,D]; small N (<=bs) so no sort (release threshold 64)
        gate = self.gate_proj(xe, idx, False).astype(mx.float32)
        up = self.up_proj(xe, idx, False).astype(mx.float32)
        lim = self.lim
        if lim > 0:
            up = mx.clip(up, -lim, lim)
            gate = mx.minimum(gate, lim)
        h = nn.silu(gate) * up
        h = h * w[:, :, None, None]
        y = self.down_proj(h.astype(mx.bfloat16), idx, False).astype(mx.float32)
        y = mx.sum(y.reshape(n, self.topk, D), axis=1)
        sg = self.sw1(xf).astype(mx.float32)
        su = self.sw3(xf).astype(mx.float32)
        if lim > 0:
            su = mx.clip(su, -lim, lim)
            sg = mx.minimum(sg, lim)
        y = y + self.sw2((nn.silu(sg) * su).astype(mx.bfloat16)).astype(mx.float32)
        return y.astype(x.dtype).reshape(B, S, D)


# --------------------------------------------------------------------------- one dSpark stage
class DSparkStage(nn.Module):
    def __init__(self, args, n_experts, inter, is_first, is_last, n_targets, markov_rank):
        super().__init__()
        d, hc = args.hidden_size, args.hc_mult
        self.attn = DSparkAttention(args)
        self.moe = DSparkMoE(args, n_experts, inter)
        self.attn_norm = RMSNorm(d, args.rms_norm_eps)
        self.ffn_norm = RMSNorm(d, args.rms_norm_eps)
        mix = (2 + hc) * hc
        for p in ("attn", "ffn"):
            setattr(self, f"hc_{p}_fn", mx.zeros((mix, hc * d), mx.float32))
            setattr(self, f"hc_{p}_base", mx.zeros((mix,), mx.float32))
            setattr(self, f"hc_{p}_scale", mx.zeros((3,), mx.float32))
        self.hc, self.iters, self.hc_eps, self.eps = hc, args.hc_sinkhorn_iters, args.hc_eps, args.rms_norm_eps
        if is_first:
            self.main_proj = DenseLinear(d * n_targets, d)
            self.main_norm = RMSNorm(d, args.rms_norm_eps)
        if is_last:
            self.norm = RMSNorm(d, args.rms_norm_eps)
            self.markov_embed = mx.zeros((args.vocab_size, markov_rank), mx.bfloat16)
            self.markov_head = mx.zeros((args.vocab_size, markov_rank), mx.bfloat16)
            self.conf_proj = mx.zeros((1, d + markov_rank), mx.float32)

    def _mixes(self, x, which):
        return _hc_mixes(x, getattr(self, f"hc_{which}_fn"), getattr(self, f"hc_{which}_scale"),
                         getattr(self, f"hc_{which}_base"), self.hc, self.iters, self.hc_eps, self.eps)

    def forward(self, x, pre_mix, positions):
        """x [1,bs,hc,d]; pre_mix [1,bs,hc]. Window must already hold committed KV. (release Block.forward)."""
        residual = x
        a_pre, a_post, a_comb = self._mixes(x, "attn")
        h = self.attn.attend(self.attn_norm(_hc_pre(x, pre_mix)), positions)
        x = _hc_post(h, residual, a_post, a_comb)
        residual = x
        f_pre, f_post, f_comb = self._mixes(x, "ffn")
        h = self.moe(self.ffn_norm(_hc_pre(x, a_pre)))
        x = _hc_post(h, residual, f_post, f_comb)
        return x, f_pre


# --------------------------------------------------------------------------- dSpark head
class DSpark(nn.Module):
    """The 3-stage dSpark draft head. embed/head are the backbone's (shared)."""

    def __init__(self, args, embed, head):
        super().__init__()
        self.args = args
        self.embed, self.head = embed, head  # backbone QuantizedEmbedding / QuantizedLinear
        self.hc = args.hc_mult
        self.bs = args.raw["dspark_block_size"]
        self.noise = args.raw["dspark_noise_token_id"]
        self.targets = tuple(args.raw["dspark_target_layer_ids"])
        self.n_stages = int(args.raw.get("n_mtp_layers", 3))
        self.markov_rank = args.raw["dspark_markov_rank"]
        n_experts = args.raw["dspark_n_routed_experts"]
        inter = args.raw.get("dspark_moe_inter_dim", args.moe_intermediate_size)
        self.stages = [
            DSparkStage(args, n_experts, inter, i == 0, i == self.n_stages - 1, len(self.targets), self.markov_rank)
            for i in range(self.n_stages)
        ]

    # --- window management (external, keeps forward_spec a pure draft) ---
    def seed_windows(self, main_hidden, positions):
        """Add committed positions' KV to every stage window. main_hidden [1,S,d*n_targets]."""
        main_x = self.stages[0].main_norm(self.stages[0].main_proj(main_hidden))  # [1,S,d]
        pos = mx.array(np.asarray(positions, np.int64))
        for st in self.stages:
            st.attn.seed(main_x, pos)
        mx.eval([st.attn.window.buf for st in self.stages])

    def reset_windows(self):
        for st in self.stages:
            st.attn.window = GrowBuffer()

    def window_checkpoint(self):
        return [(st.attn.window.buf, st.attn.window.n) for st in self.stages]

    def window_restore(self, ck):
        for st, (buf, n) in zip(self.stages, ck):
            st.attn.window.buf, st.attn.window.n = buf, n

    # --- draft one block (release forward_embed -> stages -> forward_head), greedy ---
    def draft(self, seed_tok, start_pos):
        """Draft from `seed_tok` (the backbone's token for position start_pos+1). Windows must hold
        committed KV for [0..start_pos]. Returns (output_ids[bs+1], confidence[bs]) where
        output_ids = [seed_tok(@start_pos+1), draft(@start_pos+2), ..., draft(@start_pos+bs+1)]."""
        bs = self.bs
        ids = np.full(bs, self.noise, np.int32)
        ids[0] = seed_tok
        x = self.embed(mx.array(ids)[None])  # [1,bs,d]
        x = mx.repeat(x[:, :, None, :], self.hc, axis=2)  # [1,bs,hc,d]
        pre_mix = mx.zeros((1, bs, self.hc), mx.float32)
        pre_mix[..., 0] = 1.0
        positions = mx.arange(start_pos + 1, start_pos + 1 + bs)
        for st in self.stages:
            x, pre_mix = st.forward(x, pre_mix, positions)
        last = self.stages[-1]
        xc = _hc_pre(x, pre_mix)  # [1,bs,d] (pre-norm; confidence reads this)
        logits = self.head(last.norm(xc)).astype(mx.float32)[0]  # [bs, vocab]
        out = [int(seed_tok)]
        membeds = []
        for i in range(bs):
            me = last.markov_embed[out[i]].astype(mx.float32)  # [rank]
            bias = me @ last.markov_head.astype(mx.float32).T  # [vocab]
            out.append(int(mx.argmax(logits[i] + bias)))
            membeds.append(me)
        me = mx.stack(membeds, axis=0)  # [bs, rank]
        conf = (mx.concatenate([xc[0], me], axis=-1) @ last.conf_proj.T)[:, 0]  # [bs]
        return out, conf


# --------------------------------------------------------------------------- loading
def _to_bf16(sh: Shards, name):
    return sh.dequant(name, mx.bfloat16)


def _quant_experts(sh: Shards, stage, mat, n_experts, bits=4):
    """Stack the per-expert bf16 weights for one projection and quantize to `bits` (SwitchProj layout)."""
    qs, ss, bsL = [], [], []
    for e in range(n_experts):
        w = sh.dequant(f"mtp.{stage}.ffn.experts.{e}.{mat}.weight", mx.bfloat16)
        qw, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
        qs.append(qw); ss.append(s); bsL.append(b)
        if e % 32 == 31:
            mx.eval(qs[-1], ss[-1], bsL[-1])
    return mx.stack(qs), mx.stack(ss), mx.stack(bsL)


class _DictShards:
    """Shards-compatible view over an in-memory ``{name: mx.array}`` dict (the prestacked
    ``dspark/`` subdir). Only the accessors ``load_dspark`` uses are implemented."""

    def __init__(self, d: dict):
        self.d = d

    def __contains__(self, name) -> bool:
        return name in self.d

    def to_mx(self, name):
        return self.d[name]

    def dequant(self, name, dtype=mx.bfloat16):
        return self.d[name].astype(dtype)


def _resolve_dspark_file(path: Path):
    """Return the prestacked draft safetensors inside a bundle, or None (use source shards)."""
    for cand in (path / "dspark" / "dspark.safetensors", path / "dspark.safetensors"):
        if cand.exists():
            return cand
    return None


def load_dspark(source_dir, args, embed, head):
    """Load the dSpark draft head. Prefers a bundle's self-contained ``dspark/`` subdir
    (routed experts already stacked + 4-bit); otherwise reads the raw ``mtp.*`` from the
    source model shards and quantizes experts on the fly. Both paths are byte-identical."""
    path = Path(source_dir).expanduser()
    ds_file = _resolve_dspark_file(path)
    if ds_file is not None:
        sh = _DictShards(mx.load(str(ds_file)))
    else:
        sh = Shards(path, allow_missing=True)
    ds = DSpark(args, embed, head)
    n_experts = args.raw["dspark_n_routed_experts"]
    for i, st in enumerate(ds.stages):
        p = f"mtp.{i}"
        a = st.attn
        a.wq_a.weight = _to_bf16(sh, f"{p}.attn.wq_a.weight")
        a.wq_b.weight = _to_bf16(sh, f"{p}.attn.wq_b.weight")
        a.wkv.weight = _to_bf16(sh, f"{p}.attn.wkv.weight")
        a.wo_b.weight = _to_bf16(sh, f"{p}.attn.wo_b.weight")
        woa = _to_bf16(sh, f"{p}.attn.wo_a.weight")  # [groups*o_lora_rank, (H/groups)*head_dim]
        a.wo_a = woa.reshape(a.n_groups, a.o_lora_rank, -1)
        a.q_norm.weight = _to_bf16(sh, f"{p}.attn.q_norm.weight")
        a.kv_norm.weight = _to_bf16(sh, f"{p}.attn.kv_norm.weight")
        a.attn_sink = sh.to_mx(f"{p}.attn.attn_sink").astype(mx.float32)
        st.attn_norm.weight = _to_bf16(sh, f"{p}.attn_norm.weight")
        st.ffn_norm.weight = _to_bf16(sh, f"{p}.ffn_norm.weight")
        for which in ("attn", "ffn"):
            setattr(st, f"hc_{which}_fn", sh.to_mx(f"{p}.hc_{which}_fn").astype(mx.float32))
            setattr(st, f"hc_{which}_base", sh.to_mx(f"{p}.hc_{which}_base").astype(mx.float32))
            setattr(st, f"hc_{which}_scale", sh.to_mx(f"{p}.hc_{which}_scale").astype(mx.float32))
        m = st.moe
        m.gate_w = _to_bf16(sh, f"{p}.ffn.gate.weight")
        m.gate_bias = sh.to_mx(f"{p}.ffn.gate.bias").astype(mx.float32)
        m.sw1.weight = _to_bf16(sh, f"{p}.ffn.shared_experts.w1.weight")
        m.sw3.weight = _to_bf16(sh, f"{p}.ffn.shared_experts.w3.weight")
        m.sw2.weight = _to_bf16(sh, f"{p}.ffn.shared_experts.w2.weight")
        for proj, mat in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
            pre = f"{p}.ffn.experts.{proj}.weight"
            if pre in sh:  # bundle dspark/ subdir: experts already stacked + quantized
                qw = sh.to_mx(pre)
                s = sh.to_mx(f"{p}.ffn.experts.{proj}.scales")
                b = sh.to_mx(f"{p}.ffn.experts.{proj}.biases")
            else:  # source shards: stack + quantize per-expert on the fly
                qw, s, b = _quant_experts(sh, i, mat, n_experts)
            sp = getattr(m, proj)
            sp.weight, sp.scales, sp.biases = qw, s, b
        if i == 0:
            st.main_proj.weight = _to_bf16(sh, f"{p}.main_proj.weight")
            st.main_norm.weight = _to_bf16(sh, f"{p}.main_norm.weight")
        if i == ds.n_stages - 1:
            st.norm.weight = _to_bf16(sh, f"{p}.norm.weight")
            st.markov_embed = _to_bf16(sh, f"{p}.markov_head.embed.weight")
            st.markov_head = _to_bf16(sh, f"{p}.markov_head.head.weight")
            st.conf_proj = sh.to_mx(f"{p}.confidence_head.proj.weight").astype(mx.float32)
        mx.eval(st.parameters())
    return ds


# --------------------------------------------------------------------------- accept/reject loop
class DSparkRunner:
    """Speculative greedy decode: output is token-identical to plain greedy, accepting a block of
    dSpark drafts per backbone verify step. Rollback is checkpoint/restore of the (ref-based) caches
    plus a re-feed of exactly the accepted tokens -- correct regardless of the backbone cache's
    internal compression."""

    def __init__(self, dsv41, dspark, width=None, conf_gate=None,
                 adaptive=False, latch_after=2):
        self.m = dsv41
        self.ds = dspark
        self.width = width or dspark.bs
        self.conf_gate = conf_gate  # if set, truncate the proposed block where confidence drops below it
        # Adaptive floors throughput at plain AR: an acceptance EMA shrinks the draft width as it
        # falls, and after ``latch_after`` consecutive zero-accept drafts it LATCHES to bare greedy
        # (no draft/verify/hidden/window overhead) for the rest of the generation -> spec cannot fall
        # below AR on sustained low-accept content (short detection transient amortizes to ~0).
        self.adaptive = adaptive
        self.latch_after = latch_after

    def _greedy(self, logits):
        return int(mx.argmax(logits))

    def stream(self, prompt_ids, max_new, eos_ids=(), stats=None, session=None, start_offset=0):
        """Greedy speculative decode as a GENERATOR: yields committed backbone-greedy tokens one at a
        time (so a server can stream and check stops between/within accepted blocks). The yielded
        sequence is identical to ``generate()['tokens']``. Owns its own runtime ``Session`` +
        checkpoint/restore. If ``stats`` (dict) is passed it is filled with steps/drafted/accepted/
        emitted. ``session`` lets the engine spec-decode generator drive its OWN cache (the engine
        DeepseekV41Cache list, duck-typing the runtime Session) so DSpark output populates the prefix
        cache; with a restored prefix, ``start_offset`` skips the cached prompt and prefills only the
        tail. (See DECODE_DSPARK_METHODS.md M1.)"""
        sess, pending, c = self.prefill(prompt_ids, session=session, start_offset=start_offset)
        yield from self.decode_stream(sess, pending, c, max_new, eos_ids=eos_ids, stats=stats)

    def prefill(self, prompt_ids, session=None, start_offset=0):
        """Forward ``prompt_ids`` into ``session`` (a runtime Session or the engine's _SeqState duck
        type) and seed the draft windows, returning (session, pending_logits, c) at the clean prompt
        boundary (no decode yet, so the engine generator can snapshot the cache here). ``prompt_ids``
        is forwarded AS GIVEN from the session's current absolute position ``sess.pos``: on a prefix
        HIT the scheduler already restored the cache and passes only the UNCACHED TAIL, so the tail is
        appended onto the restored prefix (no slicing/trimming). ``start_offset`` is accepted for
        back-compat but ignored — the session position is authoritative."""
        m, ds = self.m, self.ds
        sess = session if session is not None else m.new_session()
        ds.reset_windows()
        start = int(getattr(sess, "pos", 0) or 0)  # absolute position of the (possibly restored) cache
        ids = list(prompt_ids)
        logits, mh = m.forward(ids, sess, all_logits=False, return_hidden=True)  # appends [start..start+L-1]
        end = start + len(ids)
        ds.seed_windows(mh[None], range(start, end))  # draft windows hold [start..c]; c = end-1 (absolute)
        return sess, logits, end - 1

    def decode_stream(self, sess, pending, c, max_new, eos_ids=(), stats=None):
        """Greedy speculative decode generator from a prefilled (sess, pending, c). Yields committed
        backbone-greedy tokens one at a time (identical to greedy output); no-refeed fast path +
        adaptive latch apply (DECODE_DSPARK_METHODS.md M1/M2)."""
        m, ds = self.m, self.ds
        n_emitted = n_steps = n_drafted = n_accepted = 0
        eos_ids = set(int(e) for e in eos_ids)
        accept_ema, zero_streak, ar_latched = 1.0, 0, False   # adaptive state

        def _rec():
            if stats is not None:
                stats.update(steps=n_steps, drafted=n_drafted, accepted=n_accepted, emitted=n_emitted)

        while n_emitted < max_new:
            n_steps += 1
            p0 = self._greedy(pending)                 # backbone truth for c+1 (free)
            if self.adaptive and ar_latched:
                # Latched pure-AR: after >=latch_after zero-accept drafts, run BARE greedy for the
                # rest -> byte-for-byte AR cost, so spec throughput cannot fall below AR.
                pending = m.forward([p0], sess, all_logits=False)
                c += 1
                yield p0; n_emitted += 1
                if (eos_ids and p0 in eos_ids) or n_emitted >= max_new:
                    _rec(); return
                continue
            draft_ids, conf = self.ds.draft(p0, c)     # output_ids[0]=p0; [1..bs] draft c+2..c+bs+1
            W = self.width                             # number of draft tokens to verify (1..bs)
            if self.adaptive:                          # shrink width as acceptance falls
                W = max(1, min(self.width, round(self.width * accept_ema)))
            if self.conf_gate is not None:
                cg = (conf >= self.conf_gate).tolist()
                W = 1
                while W < self.width and cg[W - 1]:
                    W += 1
            feed = [p0] + draft_ids[1 : 1 + W]         # [p0,d1..dW] at positions c+1..c+1+W
            n_drafted += W
            ck = sess.checkpoint()
            vlog, vmh = m.forward(feed, sess, all_logits=True, return_hidden=True)  # vlog[j] predicts c+2+j
            truth = [int(x) for x in mx.argmax(vlog, axis=-1).tolist()]  # truth for c+2 .. c+2+W
            k = 0
            for j in range(W):                         # draft d_{j+1}=draft_ids[1+j] @ c+2+j vs truth[j]
                if draft_ids[1 + j] == truth[j]:
                    k += 1
                else:
                    break
            n_accepted += k
            accept_ema = 0.7 * accept_ema + 0.3 * (k / max(1, W))  # update rolling acceptance
            if self.adaptive:                          # latch to pure-AR after consecutive rejects
                zero_streak = zero_streak + 1 if k == 0 else 0
                if zero_streak >= self.latch_after:
                    ar_latched = True
            if k == W:
                # whole block accepted: the verify forward already wrote exactly-correct causal KV for
                # every position (no rejected tail to undo), so skip the restore+refeed commit forward
                # -> 1 backbone forward this step. truth[W] (the free bonus) becomes the next p0.
                emit = [p0] + draft_ids[1 : 1 + W]      # W+1 accepted tokens; bonus deferred to next p0
                ds.seed_windows(vmh[None], range(c + 1, c + 1 + len(emit)))
                c += len(emit)
                pending = vlog[W]                       # predicts c+1 (== truth[W], next step's p0)
            else:
                emit = [p0] + draft_ids[1 : 1 + k] + [truth[k]]  # all backbone-greedy truth, c+1..c+k+2
                sess.restore(ck)                        # undo the W+1-token verify (rejected tail present)
                flog, fmh = m.forward(emit, sess, all_logits=True, return_hidden=True)  # commit exactly emit
                ds.seed_windows(fmh[None], range(c + 1, c + 1 + len(emit)))
                c += len(emit)
                pending = flog[-1]                      # predicts c+1 again
            for t in emit:
                yield t; n_emitted += 1
                if (eos_ids and t in eos_ids) or n_emitted >= max_new:
                    _rec(); return
        _rec()

    def generate(self, prompt_ids, max_new, eos_ids=(), verbose=False):
        """Batch convenience wrapper over stream(): collect all tokens + report acceptance/speedup."""
        stats = {}
        t0 = time.time()
        out = list(self.stream(prompt_ids, max_new, eos_ids, stats))
        dt = time.time() - t0
        return {
            "tokens": out,
            "steps": stats.get("steps", 0),
            "drafted": stats.get("drafted", 0),
            "accepted": stats.get("accepted", 0),
            "accept_rate": stats.get("accepted", 0) / max(1, stats.get("drafted", 0)),
            "mean_emit": len(out) / max(1, stats.get("steps", 1)),
            "seconds": dt,
            "tok_per_s": len(out) / dt if dt else 0.0,
        }
