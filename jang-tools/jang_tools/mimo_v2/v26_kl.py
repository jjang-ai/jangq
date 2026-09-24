"""Held-out KL scoring for MiMo-V2.6 bundles against the EXACT source.

  ref  : stream the source over the KL set once; save the post-norm final
         hidden states (bf16, ~0.27 GB for 32k tokens). Reference logits are
         recomputed exactly from them with the source lm_head, chunk by chunk.
  eval : load a bundle (mlx_lm path + v26 runtime), run the same tokens,
         report mean / median / p90 KL(source || bundle), top-1 agreement and a
         margin-conditioned flip curve (feedback_margin_conditioned_flip_test:
         flips should fall monotonically with the source's top-1 margin).
Padding (<|endoftext|> tail runs) is excluded from every statistic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PAD = 151643


def valid_mask(tokens: np.ndarray) -> np.ndarray:
    m = np.ones_like(tokens, dtype=bool)
    for i, r in enumerate(tokens):
        j = len(r)
        while j > 0 and r[j - 1] == PAD:
            j -= 1
        m[i, j:] = False
    return m


def make_ref(src: Path, tokens: np.ndarray, out: Path):
    from .v26_source import SourceStream, layer_masks
    ss = SourceStream(src)
    a = ss.args
    B, T = tokens.shape
    full_m, swa_m = layer_masks(a, T)
    hs = [ss.embed(mx.array(tokens[i:i + 4])) for i in range(0, B, 4)]
    t0 = time.time()
    for L in range(a.num_hidden_layers):
        lay = ss.build_layer(L)
        mask = swa_m if lay.is_swa else full_m
        for i in range(len(hs)):
            hs[i] = lay(hs[i], mask)
            mx.eval(hs[i])
        del lay
        mx.clear_cache()
        print(f"[klref] layer {L} {time.time()-t0:.0f}s", flush=True)
    nw = ss.dense("model.norm.weight")
    h = mx.concatenate([mx.fast.rms_norm(x, nw, a.layernorm_epsilon) for x in hs], axis=0)
    mx.save_safetensors(str(out), {"hidden": h.astype(mx.bfloat16), "tokens": mx.array(tokens)})
    print(f"[klref] saved {out}", flush=True)


def evaluate(bundle: Path, ref_path: Path, src: Path, out: Path | None, chunk: int = 512, gptq_plan: Path | None = None, token_scores: Path | None = None):
    import jang_tools.mimo_v2.mlx_register  # noqa: F401
    from mlx_lm import load
    from .v26_source import SourceStream
    ref = mx.load(str(ref_path))
    tokens = np.array(ref["tokens"])
    hid = ref["hidden"]
    lm_head = SourceStream(src).dense("lm_head.weight")
    model, _ = load(str(bundle))
    # Reject a default-mode load before giving its KL a quality label.
    cfg = json.loads((bundle / "config.json").read_text())
    quant = cfg["quantization"]
    checked_units = 0
    for name, module in model.named_modules():
        if ".switch_mlp." not in name or not hasattr(module, "bits"):
            continue
        expected = quant.get(name)
        actual = {"mode": module.mode, "bits": module.bits, "group_size": module.group_size}
        if expected != actual:
            raise ValueError(f"Runtime quantization mismatch {name}: {actual} != {expected}")
        checked_units += 1
    expected_units = 3 * sum(bool(x) for x in cfg["moe_layer_freq"])
    if checked_units != expected_units:
        raise ValueError(f"Runtime expert census {checked_units} != {expected_units}")
    overlay = []
    if gptq_plan is not None:
        from .v26_gptq_provenance import validate_conversion, file_sha256
        import re
        plan = json.loads(gptq_plan.read_text())
        directory = Path(plan["gptq_dir"])
        validate_conversion(directory, plan, src)
        from .convert_v26 import load_plan
        resolved = load_plan(gptq_plan, cfg["num_hidden_layers"],
                             [i for i, flag in enumerate(cfg["moe_layer_freq"]) if flag])
        jc = json.loads((bundle / "jang_config.json").read_text())
        if (jc["quantization"]["awq"]["alpha_per_layer"] != (plan.get("awq_alpha") or {})):
            raise ValueError("GPTQ overlay AWQ recipe differs from base bundle")
        for (layer, proj), spec in resolved["_units"].items():
            # Native MXFP4 plans intentionally carry only mode: this is the
            # format's fixed 4-bit / 32-element layout, as in convert_v26.
            expected = ({"mode": "mxfp4", "bits": 4, "group_size": 32}
                        if spec["mode"] == "mxfp4" else
                        {key: spec[key] for key in ("mode", "bits", "group_size")})
            if quant[f"model.layers.{layer}.mlp.switch_mlp.{proj}"] != expected:
                raise ValueError("GPTQ overlay precision recipe differs from base bundle")
        # Controlled in-memory evaluation before spending another full build.
        # A result with overlays is never presented as the on-disk bundle score.
        for path in sorted(directory.glob("L*.safetensors")):
            match = re.fullmatch(r"L(\d+)\.(gate_proj|up_proj|down_proj)\.safetensors", path.name)
            if match is None:
                raise ValueError(f"Unknown GPTQ projection file {path}")
            layer, proj = int(match[1]), match[2]
            module = getattr(model.layers[layer].mlp.switch_mlp, proj)
            values, metadata = mx.load(str(path), return_metadata=True)
            expected_meta = (str(module.bits), str(module.group_size),
                             str(bool((plan.get("awq_alpha") or {}).get(str(layer), 0))))
            if tuple(metadata.get(k) for k in ("bits", "group_size", "awq")) != expected_meta:
                raise ValueError(f"GPTQ overlay metadata mismatch: {path}")
            if module.mode != "affine" or set(values) != {"weight", "scales", "biases"}:
                raise ValueError(f"GPTQ overlay layout mismatch: {path}")
            for name, value in values.items():
                old = getattr(module, name)
                if old.shape != value.shape or old.dtype != value.dtype:
                    raise ValueError(f"GPTQ overlay shape/dtype mismatch: {path}:{name}")
            module.update(values)
            overlay.append({"file": str(path), "sha256": file_sha256(path)})
        if not overlay:
            raise ValueError("GPTQ evaluation requested but no projection codes exist")
    from mlx.utils import tree_flatten
    model_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
    print(f"[kleval] verified {checked_units} expert units; text parameters {model_bytes/2**30:.6f} GiB", flush=True)
    mask = valid_mask(tokens)
    kls, top_agree, margins = [], [], []
    token_records = {key: [] for key in (
        "sequence", "position", "input_token", "target_token", "source_top1",
        "quant_top1", "source_nll", "quant_nll", "target_valid",
    )}
    t0 = time.time()
    for i in range(tokens.shape[0]):
        logits = model(mx.array(tokens[i:i + 1])).astype(mx.float32)[0]
        for c in range(0, tokens.shape[1], chunk):
            rl = (hid[i, c:c + chunk] @ lm_head.T).astype(mx.float32)
            ql = logits[c:c + chunk]
            rlp = rl - mx.logsumexp(rl, -1, keepdims=True)
            qlp = ql - mx.logsumexp(ql, -1, keepdims=True)
            kl = (mx.exp(rlp) * (rlp - qlp)).sum(-1)
            srt = mx.sort(rlp, -1); margin = srt[..., -1] - srt[..., -2]
            agree = mx.argmax(rl, -1) == mx.argmax(ql, -1)
            m = mask[i, c:c + chunk]
            mx.eval(kl, agree, margin)
            kls.append(np.array(kl)[m]); top_agree.append(np.array(agree)[m]); margins.append(np.array(margin)[m])
            # Logits at position p predict token p+1. Never score padding or a
            # target across the end of a packed sequence. Preserve original KL
            # positions separately so existing distribution scores stay comparable.
            positions = np.arange(c, min(c + chunk, tokens.shape[1]))
            next_pos = np.minimum(positions + 1, tokens.shape[1] - 1)
            target_valid = (positions + 1 < tokens.shape[1]) & mask[i, next_pos]
            targets = mx.array(tokens[i, next_pos, None])
            source_nll = -mx.take_along_axis(rlp, targets, axis=-1).squeeze(-1)
            quant_nll = -mx.take_along_axis(qlp, targets, axis=-1).squeeze(-1)
            values = {
                "sequence": np.full(len(positions), i, dtype=np.int32),
                "position": positions, "input_token": tokens[i, positions],
                "target_token": tokens[i, next_pos], "target_valid": target_valid,
                "source_top1": np.array(mx.argmax(rl, -1)),
                "quant_top1": np.array(mx.argmax(ql, -1)),
                "source_nll": np.array(source_nll), "quant_nll": np.array(quant_nll),
            }
            for key, value in values.items():
                token_records[key].append(value[m])
        print(f"[kleval] seq {i} {time.time()-t0:.0f}s", flush=True)
    kl = np.concatenate(kls); ag = np.concatenate(top_agree); mg = np.concatenate(margins)
    if not kl.size or not np.isfinite(kl).all() or not np.isfinite(mg).all():
        raise ValueError("Empty or non-finite held-out result")
    bins = [0, 0.25, 0.5, 1, 2, 4, 1e9]
    flip = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (mg >= lo) & (mg < hi)
        flip.append({"margin": f"[{lo},{hi})", "n": int(sel.sum()), "flip_rate": float(1 - ag[sel].mean()) if sel.any() else None})
    res = {"bundle": str(bundle), "tokens": int(kl.size), "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
           "kl_p90": float(np.quantile(kl, 0.9)), "kl_p99": float(np.quantile(kl, 0.99)), "top1_agree": float(ag.mean()),
           "flip_by_margin": flip, "peak_gib": mx.get_peak_memory() / 2 ** 30,
           "verified_expert_units": checked_units, "text_parameter_bytes": model_bytes,
           "gptq_overlay": overlay, "artifact_score": not bool(overlay)}
    tail_count = max(1, int(np.ceil(kl.size * 0.01)))
    res.update(kl_worst_one_percent_count=tail_count,
               kl_worst_one_percent_mean=float(np.sort(kl)[-tail_count:].mean()),
               kl_max=float(kl.max()), kl_min=float(kl.min()),
               kl_std=float(kl.std()))
    records = {key: np.concatenate(value) for key, value in token_records.items()}
    targets_valid = records["target_valid"]
    res["next_token_positions"] = int(targets_valid.sum())
    for arm in ("source", "quant"):
        nll = records[f"{arm}_nll"][targets_valid]
        if not nll.size or not np.isfinite(nll).all():
            raise ValueError(f"Empty or non-finite {arm} next-token likelihoods")
        res[f"{arm}_nll_mean"] = float(nll.mean())
        res[f"{arm}_perplexity"] = float(np.exp(nll.astype(np.float64).mean()))
    if token_scores is not None:
        token_scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(token_scores, kl=kl, top1_agree=ag, margin=mg, **records)
        res["token_scores"] = str(token_scores)
    print(json.dumps(res, indent=1))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref"); r.add_argument("--src", type=Path, required=True); r.add_argument("--tokens", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("eval"); e.add_argument("--bundle", type=Path, required=True); e.add_argument("--ref", type=Path, required=True); e.add_argument("--src", type=Path, required=True); e.add_argument("--out", type=Path); e.add_argument("--gptq-plan", type=Path); e.add_argument("--token-scores", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "ref":
        make_ref(a.src, np.load(a.tokens), a.out)
    else:
        evaluate(a.bundle, a.ref, a.src, a.out, gptq_plan=a.gptq_plan, token_scores=a.token_scores)


if __name__ == "__main__":
    main()
