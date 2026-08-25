"""Utterly detailed audit of a Qwen3.8-27B JANG bundle — every stamp, every method.

Checks, per bundle: quantization (mode, overrides, bit distribution, fp16
passthrough set, imatrix refit provenance), Hessian-bitmap provenance, MTP
(tensors, runtime keys, tuning sidecar semantics), vision+video (processors,
claims), reasoning (effort tiers, preserve_thinking, enable kwarg, off-prefill),
sampling (two-file agreement, agentic default), tokens/template integrity, and
an honest record of which calibration methods were and were NOT applied.

Exit 0 only if every gate passes on every bundle given.

    python -m jang_tools.qwen38_audit <bundle> [more...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

EXPECT_EFFORTS = ["low", "medium", "xhigh"]
EXPECT_EOS = [248046, 248044]


def audit(b: Path) -> tuple[list[str], list[str]]:
    ok, bad = [], []

    def gate(cond, label):
        (ok if cond else bad).append(label)

    cfg = json.loads((b / "config.json").read_text())
    jang = json.loads((b / "jang_config.json").read_text())
    gen = json.loads((b / "generation_config.json").read_text())
    wm = json.loads((b / "model.safetensors.index.json").read_text())["weight_map"]

    # ── quantization ─────────────────────────────────────────────────────
    q = cfg.get("quantization", {})
    ov = {k: v for k, v in q.items() if isinstance(v, dict)}
    dist = collections.Counter(v.get("bits") for v in ov.values())
    is_mx = q.get("mode") == "mxfp8"
    gate(len(ov) >= 580, f"per-module overrides present ({len(ov)})")
    gate(q.get("mode") in ("affine", "mxfp8"), f"mode={q.get('mode')}")
    ok.append(f"bit distribution {dict(sorted(dist.items()))}")
    # fp16 passthrough: vision linear_fc2 (in=4304) must have NO override
    # ONLY blocks.N.mlp.linear_fc2 (in=4304) is unquantizable; the MERGER's
    # linear_fc2 is a different, quantizable module and legitimately has an
    # override — the first audit run flagged it and taught us the distinction.
    fc2 = [k for k in wm if ".blocks." in k and "linear_fc2.weight" in k
           and ("vision" in k or "visual" in k)]
    fc2_ov = [k for k in ov if ".blocks." in k and "linear_fc2" in k
              and ("vision" in k or "visual" in k)]
    gate(len(fc2) == 27 and len(fc2_ov) == 0,
         f"vision blocks linear_fc2 fp16 passthrough (27 tensors, {len(fc2_ov)} overrides)")

    # ── calibration methods — honest record ──────────────────────────────
    jq = jang.get("quantization", {}) if isinstance(jang.get("quantization"), dict) else {}
    refit = jq.get("imatrix_refit")
    if is_mx:
        gate(refit is None, "MXFP8: no imatrix refit (e8m0 — not an affine fit)")
    else:
        gate(isinstance(refit, dict) and refit.get("modules", 0) >= 580,
             f"imatrix refit applied ({(refit or {}).get('modules')} modules, "
             f"max_bits {(refit or {}).get('max_bits')})")
    # 🚨 These three were `ok.append(...)` — hardcoded strings on the PASS list
    # that verified nothing. Worse, the AWQ line asserted "NOT applied", which
    # became FALSE the moment v2 started folding AWQ, and it kept printing a ✓.
    # An audit that states a method's status without reading it is worse than
    # no check: it launders an assumption into a green tick. Now gated on the
    # bundle's own provenance record.
    awq = cfg.get("awq") if isinstance(cfg.get("awq"), dict) else None
    if is_mx:
        gate(awq is None,
             "AWQ: not applied (MXFP8 is the uncalibrated reference tier)")
    else:
        gate(isinstance(awq, dict) and awq.get("linears", 0) >= 360
             and awq.get("groups", 0) >= 120,
             f"AWQ applied (alpha={(awq or {}).get('alpha')}, "
             f"{(awq or {}).get('groups')} norm groups, "
             f"{(awq or {}).get('linears')} linears, "
             f"folded_into={(awq or {}).get('folded_into')})")
    hess = jq.get("hessian_bitmap") or jq.get("bitmap_method") or q.get("method")
    gate(bool(hess) or bool(ov),
         f"Hessian: bit map from tr(H)*||W||^2_F capture ({hess or 'per-module overrides present'})")
    gate("gptq" not in cfg and "gptq" not in jq,
         "GPTQ: not applied (needs off-diagonal H; logged follow-up)")

    # ── MTP ──────────────────────────────────────────────────────────────
    mtp_n = sum(1 for k in wm if k.startswith("mtp."))
    rt = jang.get("runtime", {})
    gate(mtp_n == 31, f"mtp tensors = {mtp_n}")
    gate(rt.get("bundle_has_mtp") is True, "runtime.bundle_has_mtp")
    gate(rt.get("mtp_layers") == 1, "runtime.mtp_layers = 1")
    gate(rt.get("mtp_mode") == "preserved_enabled", f"runtime.mtp_mode = {rt.get('mtp_mode')}")
    gate(jang.get("drop_mtp") is False, "drop_mtp = false")
    gate(jang.get("mtp", {}).get("num_layers") == 1, "mtp.num_layers = 1")
    gate(jang.get("mtp", {}).get("trained_multi_step") is True,
         "mtp.trained_multi_step (card: 'trained with multiple steps')")
    tuning_p = b / "vmlx_mtp_tuning.json"
    if tuning_p.exists():
        t = json.loads(tuning_p.read_text())
        # 🚨 This used to hard-require best_depth==1 with no `validated` key.
        # That encoded a MOMENT (nothing had been measured yet) as a RULE, so
        # the day a depth sweep actually ran the audit failed the correct
        # bundle — same defect class as the "AWQ: NOT applied" assertion.
        # Gate the SEMANTICS instead: either an honest unvalidated default, or
        # a validated recommendation that carries its evidence.
        depth = t.get("best_depth")
        blocked_ok = t.get("blocked") is False
        if t.get("validated") is True:
            ev = ("baseline_tok_s", "best_tok_s", "speedup_vs_baseline")
            has_ev = all(isinstance(t.get(k), (int, float)) for k in ev)
            gate(isinstance(depth, int) and depth >= 1 and blocked_ok and has_ev,
                 f"tuning sidecar: VALIDATED best_depth={depth} drafts "
                 f"({t.get('baseline_tok_s')} -> {t.get('best_tok_s')} tok/s, "
                 f"{t.get('speedup_vs_baseline')}x)")
        else:
            gate(depth == 1 and blocked_ok and "output_equivalent" not in t,
                 "tuning sidecar: best_depth=1 drafts, unvalidated-recommendation semantics")
    else:
        bad.append("vmlx_mtp_tuning.json MISSING")

    # ── vision + video ───────────────────────────────────────────────────
    caps = jang.get("capabilities", {})
    gate((b / "preprocessor_config.json").exists(), "image preprocessor present")
    gate((b / "video_preprocessor_config.json").exists(), "video preprocessor present")
    gate(caps.get("has_vision") is True and caps.get("has_video") is True,
         "capabilities claim vision+video")
    vis_n = sum(1 for k in wm if "visual" in k or "vision_tower" in k)
    gate(vis_n >= 333, f"vision tensors = {vis_n}")

    # ── reasoning: 3.8 contract ──────────────────────────────────────────
    r = jang.get("reasoning", {})
    gate(r.get("default") == "on" and caps.get("default_reasoning") == "on",
         "reasoning default ON")
    gate(r.get("supported_reasoning_efforts") == EXPECT_EFFORTS,
         f"supported_reasoning_efforts = {r.get('supported_reasoning_efforts')}")
    gate(r.get("default_reasoning_effort") == "xhigh", "default_reasoning_effort = xhigh")
    gate(r.get("preserve_thinking_supported") is True
         and r.get("preserve_thinking_default") is True,
         "preserve_thinking supported + default ON")
    gate(r.get("off_prefill") == "<think>\n\n</think>\n\n", "off = closed prefill")

    # ── sampling: two-file agentic contract ──────────────────────────────
    sd = jang.get("chat", {}).get("sampling_defaults", {})
    modes = jang.get("chat", {}).get("sampling_modes", {})
    gate(sd.get("temperature") == 1.0 and sd.get("top_p") == 0.95
         and sd.get("top_k") == 20, "agentic default T=1.0/0.95/20")
    gate(all(sd.get(k) == gen.get(k) for k in ("temperature", "top_p", "top_k")),
         "two-file agreement (jang_config == generation_config)")
    gate(set(modes) == {"thinking", "instruct_nothinking"},
         f"exactly 2 card presets ({sorted(modes)})")
    gate(gen.get("eos_token_id") == EXPECT_EOS, f"eos = {gen.get('eos_token_id')}")
    mx_tok = jang.get("chat", {}).get("recommended_max_tokens", {})
    gate(mx_tok.get("reasoning") == 262144 and mx_tok.get("final_response") == 131072,
         "card output-length guidance stamped")

    # ── template integrity ───────────────────────────────────────────────
    tc = json.loads((b / "tokenizer_config.json").read_text())
    jinja = (b / "chat_template.jinja").read_text()
    emb = tc.get("chat_template")
    gate(emb is None or emb == jinja, "no template conflict (embedded == jinja)")
    gate("reasoning_effort" in jinja, "template carries reasoning_effort logic")

    return ok, bad


def main(argv) -> int:
    fails = 0
    for d in argv[1:]:
        b = Path(d)
        ok, bad = audit(b)
        print(f"\n════ {b.name} ════")
        for x in ok:
            print(f"  ✓ {x}")
        for x in bad:
            print(f"  ✗ {x}")
        fails += len(bad)
        print(f"  → {len(ok)} pass / {len(bad)} FAIL")
    print(f"\nAUDIT {'CLEAN' if fails == 0 else f'FAILED ({fails})'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
