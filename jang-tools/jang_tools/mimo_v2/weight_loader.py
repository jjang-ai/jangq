"""Streaming weight loader for MiMo-V2 source checkpoints.

Hides shard boundaries and transparently dequantizes FP8 e4m3fn weights
(with their fp32 ``*_weight_scale_inv`` companions) into fp32. Plain bf16
weights pass through unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .fp8_block_codec import (
    dequant_fp8_e4m3_scale_inv,
)


class MiMoShardIndex:
    """Lookup table from tensor name → source shard, with cached safe_open handles."""

    def __init__(self, src: str | Path):
        self.src = Path(src).expanduser()
        idx = json.loads((self.src / "model.safetensors.index.json").read_text())
        self.config = json.loads((self.src / "config.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        # All tensor names visible in the source (incl. *_weight_scale_inv).
        self.keys: list[str] = sorted(self.weight_map.keys())
        # Weight names that have a companion FP8 scale.
        self.fp8_weight_names: set[str] = {
            k[: -len(".weight_scale_inv")] + ".weight"
            for k in self.keys
            if k.endswith(".weight_scale_inv")
        }
        # Names yielded to callers — drop bare scale tensors, they are read
        # internally when their companion weight is requested.
        self.weight_keys: list[str] = [k for k in self.keys if not k.endswith(".weight_scale_inv")]

    # ------------------------------------------------------------------
    # Tensor reads
    # ------------------------------------------------------------------

    def is_fp8_weight(self, name: str) -> bool:
        return name in self.fp8_weight_names

    def _qkv_interleaved_layout(
        self,
        name: str,
        weight_shape: tuple[int, int],
        scale_shape: tuple[int, int],
    ) -> tuple[int, int, int, int, int, int] | None:
        """Return TP-interleaved qkv layout for MiMo-V2.5 source weights.

        MiMo-V2.5 stores fused qkv as TP=4 interleaved shards:

            rank0(Q,K,V), rank1(Q,K,V), rank2(Q,K,V), rank3(Q,K,V)

        The MLX runtime expects logical fused order:

            all_Q, all_K, all_V

        The FP8 scale rows follow the stored TP chunks. Full-attention chunks
        are not a multiple of 128 rows, so each rank chunk has one padded scale
        row. Treating the scale tensor or qkv rows as one flat matrix corrupts
        the projection.
        """
        import math
        import re

        m = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight", name)
        if not m:
            return None

        rows, _ = weight_shape
        scale_rows, _ = scale_shape

        layer = int(m.group(1))
        is_swa = int(self.config["hybrid_layer_pattern"][layer]) == 1
        num_heads = self.config["swa_num_attention_heads"] if is_swa else self.config["num_attention_heads"]
        num_kv_heads = self.config["swa_num_key_value_heads"] if is_swa else self.config["num_key_value_heads"]
        head_dim = self.config["swa_head_dim"] if is_swa else self.config["head_dim"]
        v_head_dim = self.config["swa_v_head_dim"] if is_swa else self.config["v_head_dim"]
        q_rows = int(num_heads) * int(head_dim)
        k_rows = int(num_kv_heads) * int(head_dim)
        v_rows = int(num_kv_heads) * int(v_head_dim)
        if q_rows + k_rows + v_rows != rows:
            raise ValueError(
                f"{name}: q/k/v rows {q_rows}+{k_rows}+{v_rows} do not match weight rows {rows}"
            )

        # The released checkpoint is TP=4-interleaved. This is also the minimum
        # TP accepted by the supported vLLM/SGLang loaders for this fused-qkv
        # checkpoint layout.
        tp = 4
        if q_rows % tp or k_rows % tp or v_rows % tp:
            raise ValueError(f"{name}: q/k/v rows are not divisible by TP={tp}")
        q_per = q_rows // tp
        k_per = k_rows // tp
        v_per = v_rows // tp
        rank_rows = q_per + k_per + v_per
        rank_scale_rows = math.ceil(rank_rows / 128)
        if rank_rows * tp != rows or rank_scale_rows * tp != scale_rows:
            raise ValueError(
                f"{name}: expected TP={tp} interleaved qkv layout with "
                f"rank_rows={rank_rows}, rank_scale_rows={rank_scale_rows}; "
                f"got weight_shape={weight_shape}, scale_shape={scale_shape}"
            )
        return tp, q_per, k_per, v_per, rank_rows, rank_scale_rows

    def read_passthrough(self, name: str, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Read a tensor as-is from its shard. No dequantization."""
        with safe_open(str(self.src / self.weight_map[name]), framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        if out_dtype is not None and t.dtype != out_dtype:
            t = t.to(out_dtype)
        return t

    def read_tensor(self, name: str, *, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Read `name` and dequantize from FP8 + block scale_inv if applicable.

        Plain bf16/fp32 tensors are cast to ``out_dtype`` if needed.
        """
        if name not in self.fp8_weight_names:
            return self.read_passthrough(name, out_dtype=out_dtype)

        scale_name = name[: -len(".weight")] + ".weight_scale_inv"
        weight_shard = self.weight_map[name]
        scale_shard = self.weight_map[scale_name]
        with safe_open(str(self.src / weight_shard), framework="pt", device="cpu") as f:
            w = f.get_tensor(name)
        if scale_shard == weight_shard:
            with safe_open(str(self.src / weight_shard), framework="pt", device="cpu") as f:
                s = f.get_tensor(scale_name)
        else:
            with safe_open(str(self.src / scale_shard), framework="pt", device="cpu") as f:
                s = f.get_tensor(scale_name)
        qkv_layout = self._qkv_interleaved_layout(name, tuple(w.shape), tuple(s.shape))
        if qkv_layout is not None:
            tp, q_per, k_per, v_per, rank_rows, rank_scale_rows = qkv_layout
            q_parts = []
            k_parts = []
            v_parts = []
            for rank in range(tp):
                w_start = rank * rank_rows
                s_start = rank * rank_scale_rows
                rank_w = w[w_start : w_start + rank_rows]
                rank_s = s[s_start : s_start + rank_scale_rows]
                rank_out = dequant_fp8_e4m3_scale_inv(rank_w, rank_s, out_dtype=None)
                q, k, v = rank_out.split([q_per, k_per, v_per], dim=0)
                q_parts.append(q)
                k_parts.append(k)
                v_parts.append(v)
            out = torch.cat([*q_parts, *k_parts, *v_parts], dim=0)
            return out if out_dtype is None else out.to(out_dtype)
        return dequant_fp8_e4m3_scale_inv(w, s, out_dtype=out_dtype)
