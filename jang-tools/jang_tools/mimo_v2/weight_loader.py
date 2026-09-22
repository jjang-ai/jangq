"""Streaming weight loader for MiMo-V2 source checkpoints.

Hides shard boundaries and transparently dequantizes FP8 e4m3fn weights
(with their fp32 ``*_weight_scale_inv`` companions) into fp32. Plain bf16
weights pass through unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .fp8_block_codec import dequant_fp8_e4m3_scale_inv, dequant_fp8_rank_blocked
from .mxfp4_codec import dequant_mxfp4


_QKV_WEIGHT_RE = re.compile(
    r"^model\.(?P<mtp>mtp\.)?layers\.(?P<layer>\d+)\.self_attn\.qkv_proj\.weight$"
)
_MIMO_V25_QKV_TP_SIZE = 4


def deinterleave_tp_qkv_rows(
    weight: torch.Tensor,
    *,
    q_size: int,
    k_size: int,
    v_size: int,
    tp_size: int = _MIMO_V25_QKV_TP_SIZE,
) -> torch.Tensor:
    """Convert MiMo TP-rank qkv row blocks to runtime [all_q, all_k, all_v].

    MiMo-V2.5 stores fused qkv rows by tensor-parallel rank:
    rank0(Q,K,V), rank1(Q,K,V), ... . The HF and MLX runtime projection split
    the linear output as [all Q, all K, all V], so conversion/probes must
    reorder the decoded source rows before quantizing or comparing.
    """
    expected = q_size + k_size + v_size
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 qkv weight, got shape={tuple(weight.shape)}")
    if weight.shape[0] != expected:
        raise ValueError(f"qkv rows={weight.shape[0]} do not match expected rows={expected}")
    for label, size in {"q_size": q_size, "k_size": k_size, "v_size": v_size}.items():
        if size % tp_size != 0:
            raise ValueError(f"{label}={size} is not divisible by tp_size={tp_size}")

    q_per = q_size // tp_size
    k_per = k_size // tp_size
    v_per = v_size // tp_size
    rank_block = q_per + k_per + v_per
    chunks = weight.reshape(tp_size, rank_block, weight.shape[1])
    q_chunks = chunks[:, :q_per, :]
    k_chunks = chunks[:, q_per : q_per + k_per, :]
    v_chunks = chunks[:, q_per + k_per :, :]
    return torch.cat(
        [
            q_chunks.reshape(q_size, weight.shape[1]),
            k_chunks.reshape(k_size, weight.shape[1]),
            v_chunks.reshape(v_size, weight.shape[1]),
        ],
        dim=0,
    ).contiguous()


class MiMoShardIndex:
    """Lookup table from tensor name → source shard, with cached safe_open handles."""

    def __init__(self, src: str | Path):
        self.src = Path(src).expanduser()
        self.config = json.loads((self.src / "config.json").read_text())
        idx = json.loads((self.src / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        # All tensor names visible in the source (incl. *_weight_scale_inv).
        self.keys: list[str] = sorted(self.weight_map.keys())
        # Weight names that have a companion FP8 scale.
        self.fp8_weight_names: set[str] = {
            k[: -len(".weight_scale_inv")] + ".weight"
            for k in self.keys
            if k.endswith(".weight_scale_inv")
        }
        # MiMo-V2.6 routed experts: MXFP4 uint8 codes + uint8 e8m0 ``weight_scale``.
        self.mxfp4_weight_names: set[str] = {
            k[: -len(".weight_scale")] + ".weight"
            for k in self.keys
            if k.endswith(".weight_scale")
        }
        # Names yielded to callers — drop bare scale tensors, they are read
        # internally when their companion weight is requested.
        self.weight_keys: list[str] = [
            k for k in self.keys
            if not (k.endswith(".weight_scale_inv") or k.endswith(".weight_scale"))
        ]
        self._handles = {}
        self.qkv_tp = self._detect_qkv_tp()

    def _open(self, shard_name: str):
        handle = self._handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(self.src / shard_name), framework="pt", device="cpu")
            self._handles[shard_name] = handle
        return handle

    # ------------------------------------------------------------------
    # Tensor reads
    # ------------------------------------------------------------------

    def is_fp8_weight(self, name: str) -> bool:
        return name in self.fp8_weight_names

    def is_mxfp4_weight(self, name: str) -> bool:
        return name in self.mxfp4_weight_names

    def read_mxfp4_raw(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the untouched (uint8 codes, uint8 e8m0 scales) for an MXFP4 weight."""
        if name not in self.mxfp4_weight_names:
            raise KeyError(f"{name} is not an MXFP4 source weight")
        scale_name = name[: -len(".weight")] + ".weight_scale"
        w = self._open(self.weight_map[name]).get_tensor(name)
        s = self._open(self.weight_map[scale_name]).get_tensor(scale_name)
        return w, s

    def _shape(self, name: str):
        if name not in self.weight_map:
            return None
        return tuple(self._open(self.weight_map[name]).get_slice(name).get_shape())

    def _detect_qkv_tp(self) -> int:
        """Derive the TP degree the fused qkv was stored/quantized for.

        Only full-attention layers disambiguate it: each rank block is padded
        to a 128-row FP8 block on its own, so scale_rows*128 == tp*pad128(rows/tp)
        holds for exactly one tp (4 on Flash). SWA geometry is block-aligned and
        ambiguous. Same derivation as oMLX patches/mimo_v2/fused_qkv_layout.py.
        """
        hybrid = self.config.get("hybrid_layer_pattern") or []
        for layer, is_swa in enumerate(hybrid):
            if is_swa:
                continue
            w = self._shape(f"model.layers.{layer}.self_attn.qkv_proj.weight")
            s = self._shape(f"model.layers.{layer}.self_attn.qkv_proj.weight_scale_inv")
            if w is None or s is None:
                continue
            q, k, v = self.qkv_sizes_for_layer(layer)
            heads = int(self.config["num_attention_heads"])
            kv = int(self.config["num_key_value_heads"])
            hits = []
            for tp in (1, 2, 4, 8, 16, 32):
                if heads % tp or kv % tp or (q + k + v) % tp:
                    continue
                rank = (q + k + v) // tp
                if rank * tp == w[0] and tp * (-(-rank // 128)) == s[0]:
                    hits.append(tp)
            if len(hits) != 1:
                raise ValueError(f"cannot determine qkv TP degree from layer {layer}: w={w} s={s} hits={hits}")
            return hits[0]
        return 1

    def qkv_sizes_for_layer(self, layer: int, *, mtp: bool = False) -> tuple[int, int, int]:
        hybrid = self.config.get("hybrid_layer_pattern") or []
        # MTP blocks use SWA attention geometry (README: SWA MTP); verified by row count at read time.
        is_swa = True if mtp else (bool(hybrid[int(layer)]) if int(layer) < len(hybrid) else False)
        num_heads = int(
            self.config.get("swa_num_attention_heads" if is_swa else "num_attention_heads")
            or self.config["num_attention_heads"]
        )
        num_kv_heads = int(
            self.config.get("swa_num_key_value_heads" if is_swa else "num_key_value_heads")
            or self.config["num_key_value_heads"]
        )
        head_dim = int(
            self.config.get("swa_head_dim" if is_swa else "head_dim")
            or self.config["head_dim"]
        )
        v_head_dim = int(
            self.config.get("swa_v_head_dim" if is_swa else "v_head_dim")
            or self.config.get("v_head_dim")
            or head_dim
        )
        return num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * v_head_dim

    def _maybe_deinterleave_qkv(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        match = _QKV_WEIGHT_RE.match(name)
        if not match:
            return tensor
        q_size, k_size, v_size = self.qkv_sizes_for_layer(
            int(match.group("layer")), mtp=bool(match.group("mtp"))
        )
        return deinterleave_tp_qkv_rows(
            tensor, q_size=q_size, k_size=k_size, v_size=v_size, tp_size=self.qkv_tp
        )

    def read_passthrough(self, name: str, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Read a tensor as-is from its shard. No dequantization."""
        t = self._open(self.weight_map[name]).get_tensor(name)
        if out_dtype is not None and t.dtype != out_dtype:
            t = t.to(out_dtype)
        return self._maybe_deinterleave_qkv(name, t)

    def read_tensor(self, name: str, *, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Read `name` and dequantize from FP8 + block scale_inv if applicable.

        Plain bf16/fp32 tensors are cast to ``out_dtype`` if needed.
        """
        if name in self.mxfp4_weight_names:
            w, s = self.read_mxfp4_raw(name)
            return dequant_mxfp4(w, s, out_dtype=out_dtype)
        if name not in self.fp8_weight_names:
            return self.read_passthrough(name, out_dtype=out_dtype)

        scale_name = name[: -len(".weight")] + ".weight_scale_inv"
        weight_shard = self.weight_map[name]
        scale_shard = self.weight_map[scale_name]
        w = self._open(weight_shard).get_tensor(name)
        s = self._open(scale_shard).get_tensor(scale_name)
        if _QKV_WEIGHT_RE.match(name):
            # Fused qkv is stored AND FP8-quantized per TP rank: dequantize each
            # rank block with its own scale rows, then de-interleave.
            deq = dequant_fp8_rank_blocked(w, s, tp_size=self.qkv_tp, out_dtype=out_dtype)
        else:
            deq = dequant_fp8_e4m3_scale_inv(w, s, out_dtype=out_dtype)
        return self._maybe_deinterleave_qkv(name, deq)
