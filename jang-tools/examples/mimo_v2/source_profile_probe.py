"""Source-side MiMo profile probe.

This simulates a JANG affine profile directly from the FP8/BF16 source
checkpoint without building a safetensors bundle. It is intentionally slow and
diagnostic: the goal is to decide which profile is worth a full conversion.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from layer_diff_probe import SourceRunner, rmsnorm


_EXPERT_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"
)


@dataclass(frozen=True)
class ProbeProfile:
    name: str
    bookend_bits: int
    default_expert_bits: dict[str, int]
    expert_layer_bits: dict[int, dict[str, int]]
    expert_group_size: int = 128
    bookend_group_size: int = 64
    lm_head_bits: int = 0

    @classmethod
    def parse(cls, raw: str) -> "ProbeProfile":
        key = raw.lower().replace("_", "").replace("-", "")
        base = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
        critical8 = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}

        if key in {"all2", "2all", "2fit"}:
            return cls("ALL2", 8, base, {})

        if key in {"c4", "2lc4"}:
            return cls("C4", 8, base, {layer: critical8 for layer in range(1, 5)})

        m = re.fullmatch(r"c4l([1-4])(?:b([68]))?(?:h8)?", key)
        if m:
            late_count = int(m.group(1))
            bookend_bits = int(m.group(2) or 8)
            late4 = {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
            layers = {layer: critical8 for layer in range(1, 5)}
            layers.update({layer: late4 for layer in range(48 - late_count, 48)})
            lm_head_bits = 8 if key.endswith("h8") else 0
            return cls(
                f"C4L{late_count}B{bookend_bits}" + ("H8" if lm_head_bits else ""),
                bookend_bits,
                base,
                layers,
                lm_head_bits=lm_head_bits,
            )

        m = re.fullmatch(r"c4l([1-4])x3(?:b([68]))?(?:h8)?", key)
        if m:
            late_count = int(m.group(1))
            bookend_bits = int(m.group(2) or 8)
            late3 = {"gate_proj": 3, "up_proj": 3, "down_proj": 3}
            layers = {layer: critical8 for layer in range(1, 5)}
            layers.update({layer: late3 for layer in range(48 - late_count, 48)})
            lm_head_bits = 8 if key.endswith("h8") else 0
            return cls(
                f"C4L{late_count}x3B{bookend_bits}" + ("H8" if lm_head_bits else ""),
                bookend_bits,
                base,
                layers,
                lm_head_bits=lm_head_bits,
            )

        grouped = re.fullmatch(r"([234][234][234])g(32|64|128)", key)
        if grouped:
            digits = grouped.group(1)
            group_size = int(grouped.group(2))
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            return cls(f"{digits.upper()}G{group_size}", 8, bits, {}, expert_group_size=group_size)

        d3e = re.fullmatch(r"322d(?:own)?3e(?:arly)?(\d+)", key)
        if d3e:
            end_layer = int(d3e.group(1))
            if end_layer < 1 or end_layer > 47:
                raise ValueError(f"invalid early down3 end {end_layer}; expected 1..47")
            base = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
            layers = {layer: early for layer in range(1, end_layer + 1)}
            return cls(f"322D3E{end_layer}", 8, base, layers)

        three_digit = re.fullmatch(r"[234][234][234]", key)
        if key == "2l" or three_digit:
            if key == "2l":
                bits = {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
            else:
                bits = {"gate_proj": int(key[0]), "up_proj": int(key[1]), "down_proj": int(key[2])}
            return cls(key.upper(), 8, bits, {})

        raise ValueError(
            f"unknown profile {raw!r}; use c4, c4l1..c4l4, c4l1x3..c4l4x3, "
            "optional b6/b8 and h8 suffixes, 2l, 322/323/333, 322g64, or 322d3eN"
        )

    def expert_bits_for(self, name: str) -> int | None:
        m = _EXPERT_RE.match(name)
        if not m:
            return None
        layer = int(m.group("layer"))
        proj = m.group("proj")
        return self.expert_layer_bits.get(layer, self.default_expert_bits)[proj]


def quant_dequant_affine(weight: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    if bits == 0:
        return weight.float()
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported affine bits={bits}")
    x = weight.float()
    if x.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape={tuple(x.shape)}")
    rows, cols = x.shape
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    groups = cols // group_size
    xr = x.reshape(rows, groups, group_size)
    minv = xr.amin(dim=2, keepdim=True)
    maxv = xr.amax(dim=2, keepdim=True)
    scale = ((maxv - minv) / float((1 << bits) - 1)).clamp_min(1e-7)
    q = torch.round((xr - minv) / scale).clamp_(0, (1 << bits) - 1)
    # Converter writes sidecars as f16; include that loss in the source probe.
    scale = scale.to(torch.float16).to(torch.float32)
    bias = minv.to(torch.float16).to(torch.float32)
    return (q * scale + bias).reshape_as(x)


class QuantizedSourceRunner(SourceRunner):
    def __init__(self, src: Path, profile: ProbeProfile):
        super().__init__(src)
        self.profile = profile

    @lru_cache(maxsize=128)
    def tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = super().tensor(name, dtype=torch.float32)
        expert_bits = self.profile.expert_bits_for(name)
        if expert_bits is not None:
            return quant_dequant_affine(
                w,
                bits=expert_bits,
                group_size=self.profile.expert_group_size,
            )
        if name.endswith(".weight"):
            return quant_dequant_affine(
                w,
                bits=self.profile.bookend_bits,
                group_size=self.profile.bookend_group_size,
            )
        return w

    @lru_cache(maxsize=128)
    def cached_tensor(self, name: str) -> torch.Tensor:
        return self.tensor(name)


def final_logits(runner: SourceRunner, h: torch.Tensor, profile: ProbeProfile | None = None) -> torch.Tensor:
    norm_w = runner.idx.read_passthrough("model.norm.weight", out_dtype=torch.float32)
    h = rmsnorm(h, norm_w, runner.eps)
    lm_head = runner.idx.read_passthrough("lm_head.weight", out_dtype=torch.float32)
    if profile is not None and profile.lm_head_bits:
        lm_head = quant_dequant_affine(
            lm_head,
            bits=profile.lm_head_bits,
            group_size=profile.bookend_group_size,
        )
    return F.linear(h, lm_head)


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 8) -> list[tuple[int, str, float]]:
    vals, idx = torch.topk(logits[0, -1].float(), k=k)
    return [(int(i), tokenizer.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]


def torch_rel_stats(src: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    s = src.float()
    a = actual.float()
    d = s - a
    rmse = torch.sqrt(torch.mean(d * d))
    rms = torch.sqrt(torch.mean(s * s)) + 1e-12
    last_d = s[:, -1, :] - a[:, -1, :]
    last_rmse = torch.sqrt(torch.mean(last_d * last_d))
    last_rms = torch.sqrt(torch.mean(s[:, -1, :] * s[:, -1, :])) + 1e-12
    return float(rmse / rms), float(last_rmse / last_rms), float(d.abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prompt", default="Name the capital city of France.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    parser.add_argument("--report-layers", default="0,1,2,3,4,31,43,44,45,46,47")
    parser.add_argument("--max-layer", type=int, default=47,
                        help="Last decoder layer to run, inclusive. Use <47 for faster screening.")
    args = parser.parse_args()

    profile = ProbeProfile.parse(args.profile)
    report_layers = {int(x) for x in args.report_layers.split(",") if x.strip()}

    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    ids = tokenizer.encode(prompt)

    src = SourceRunner(args.src)
    qsrc = QuantizedSourceRunner(args.src, profile)
    h_src = src.embed(ids)
    h_q = qsrc.embed(ids)
    rel, last_rel, maxerr = torch_rel_stats(h_src, h_q)
    print(f"profile={profile.name} tokens={len(ids)}")
    print(f"embed rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")

    for layer_idx in range(min(src.cfg["num_hidden_layers"], args.max_layer + 1)):
        h_src = src.layer(layer_idx, h_src)
        h_q = qsrc.layer(layer_idx, h_q)
        if layer_idx in report_layers:
            rel, last_rel, maxerr = torch_rel_stats(h_src, h_q)
            print(f"layer {layer_idx:02d} rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")

    if args.max_layer >= src.cfg["num_hidden_layers"] - 1:
        logits_src = final_logits(src, h_src)
        logits_q = final_logits(qsrc, h_q, profile)
        rel, last_rel, maxerr = torch_rel_stats(logits_src, logits_q)
        print(f"final_logits rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")
        print("source_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_src):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
        print("quant_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_q):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
