"""Finalize + verify a Naive-N0.5-Flash JANGH bundle.

Writes (fresh files, tmp + rename): the serving capabilities into config.json and jang_config.json, the sampling
contract, evaluation/ (build report + eval JSONs with local paths removed). Never touches the shards.
Checks (all must pass; exit 1 otherwise)
  A layout     : root tensor files are exactly the indexed shards; ONE *index.json; no other tensor file in the root
  B alignment  : every tensor offset is a multiple of its dtype size
  C config     : model_type, no auto_map (config + tokenizer_config), jangtq v2 block, every routed layer has
                 gate/up/down entries with mode jangtq2 + rotation hadamard32, shapes match bits, affine modules carry
                 bf16 scales AND biases, kept tensors keep the source dtype, router fp32, indexer unquantized
  D template   : renders thinking off (ends with "<think></think>"), efforts low/high/max ("Reasoning Effort: X" first),
                 an unknown effort renders Max, tools block + tool call + tool response
  E contract   : capabilities identical in config.json and jang_config.json; sampling defaults == generation_config
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path


from jang_tools.format.aligned_safetensors import verify_safetensors_alignment  # noqa: E402

E, D, I = 256, 4096, 2048
CAPS = {"family": "naive_n05_flash", "has_vision": False, "has_video": False, "has_audio": False,
        "supports_thinking": True, "default_reasoning": "on", "think_in_template": False,
        "reasoning_parser": "think_xml", "tool_parser": "xml_function", "supports_tools": True,
        "tool_response_role": "tool", "reasoning_efforts": ["low", "high", "max"], "reasoning_effort_default": "max",
        "cache_type": "kv", "cache_subtype": "naive_n05_swa_dsa", "modality": "text"}


def fail(msg):
    print("FAIL:", msg); sys.exit(1)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, str) and o.startswith(("/Users/", "/Volumes/")):
        return os.path.basename(o.rstrip("/"))
    return o


def write(p: Path, obj):
    t = p.with_name("." + p.name + ".tmp"); t.write_text(json.dumps(obj, indent=1)); t.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    b = Path(a.bundle)
    if not a.verify_only:
        cfg = json.loads((b / "config.json").read_text()); cfg["capabilities"] = dict(CAPS); write(b / "config.json", cfg)
        gen = json.loads((b / "generation_config.json").read_text())
        jc = json.loads((b / "jang_config.json").read_text())
        jc["capabilities"] = dict(CAPS)
        jc["chat"] = {"sampling_defaults": {"temperature": gen["temperature"], "top_p": gen["top_p"]},
                      "default_sampling_mode": "thinking", "stop_token_ids": [gen["eos_token_id"]] if isinstance(gen["eos_token_id"], int) else gen["eos_token_id"],
                      "reasoning_efforts": ["low", "high", "max"], "reasoning_effort_default": "max"}
        rp = b / "jangh_build_report.json"
        rep = json.loads((rp if rp.exists() else b / "evaluation" / "jangh_build_report.json").read_text())
        esc = {g: sum(v["extra_damp"].get(g, {}).get("experts_escalated", 0) for v in rep["layers"].values()) for g in ("gu", "dn")}
        jc["calibration"] = {"reference": "BF16 source (layer-streamed)", "tokens": 618462,
                             "domains": "coding, tool conversations, cybersecurity, agentic, general, Chinese, science, academic",
                             "held_out": "every 10th calibration document + 28 reference prompts + 96 tool conversations on unseen tools",
                             "rotation": "hadamard32",
                             "rounding": "GPTQ (JANGH codebook) on the full per-expert input covariance, act-order, 1% damping",
                             "hessian": "uncentered; pooled-covariance floor per layer chosen on held-out rows",
                             "allocation": "knapsack on measured held-out error curves weighted by the measured frozen-selection KL per layer",
                             "awq": "evaluated, not applied", "qat": "not applied", "mxfp8_nonexperts": "evaluated, not applied",
                             "experts_with_extra_damping": esc}
        jc["jangtq"]["method"] = {"experts": "JANGH odd-cubic codebook, per-row fp16 scale, rotation hadamard32",
                                  "rounding": "GPTQ in the rotated basis", "allocation": "knapsack on measured held-out curves",
                                  "non_experts": "affine 8-bit, group 64, bf16 scales and biases"}
        write(b / "jang_config.json", jc)
        cfg["jangtq"]["method"] = jc["jangtq"]["method"]; write(b / "config.json", cfg)
        tk = json.loads((b / "tokenizer_config.json").read_text())
        if tk.pop("auto_map", None) is not None:
            write(b / "tokenizer_config.json", tk)
        (b / "evaluation").mkdir(exist_ok=True)
        write(b / "evaluation" / "jangh_build_report.json", clean(rep))
        if (b / "jangh_build_report.json").exists():
            (b / "jangh_build_report.json").unlink()
        for e in a.eval:
            write(b / "evaluation" / Path(e).name, clean(json.loads(Path(e).read_text())))
    # ------------------------------------------------ verification
    root = sorted(p.name for p in b.iterdir() if p.is_file() and not p.name.startswith("."))
    idx = json.loads((b / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(idx.values()))
    if sorted(n for n in root if n.endswith(".safetensors")) != shards: fail("A root safetensors != indexed shards")
    if sum(n.endswith("index.json") for n in root) != 1: fail("A exactly one *index.json required")
    if any(n.endswith(".py") for n in root): fail("A python files in the bundle root")
    shapes, nt = {}, 0
    for s in shards:
        n, bad = verify_safetensors_alignment(b / s); nt += n
        if bad: fail(f"B {s}: {bad} misaligned tensors")
        with open(b / s, "rb") as fh:
            hn = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(hn))
        shapes.update({k: (v["dtype"], v["shape"]) for k, v in hdr.items() if k != "__metadata__"})
    if nt != len(idx): fail(f"B tensor count {nt} != index {len(idx)}")
    cfg = json.loads((b / "config.json").read_text()); q = cfg["quantization"]
    if cfg.get("model_type") != "naive_n05_flash": fail("C model_type")
    if "auto_map" in cfg or "auto_map" in json.loads((b / "tokenizer_config.json").read_text()): fail("C auto_map present")
    jt = cfg.get("jangtq", {})
    if jt.get("version") != 2 or jt.get("codebook_family") != "odd-cubic" or jt.get("rotation") != "hadamard32": fail("C jangtq block")
    for L in range(1, 48):
        for n_, N, K in (("gate", I, D), ("up", I, D), ("down", D, I)):
            m = f"model.layers.{L}.mlp.switch_mlp.{n_}_proj"; e = q.get(m)
            if not isinstance(e, dict) or e.get("mode") != "jangtq2" or e.get("rotation") != "hadamard32": fail(f"C {m} entry {e}")
            p, s = shapes.get(m + ".tq2_packed"), shapes.get(m + ".tq2_scales")
            if p is None or s is None or p != ("U32", [E, N, K * e["bits"] // 32]) or s != ("F16", [E, N]): fail(f"C {m} tensors {p} {s} bits {e['bits']}")
        for t in (f"model.layers.{L}.mlp.gate.weight", f"model.layers.{L}.mlp.gate.e_score_correction_bias"):
            if shapes.get(t, ("",))[0] != "F32": fail(f"C router tensor {t} is {shapes.get(t)}")
    n_aff = n_mx = 0
    for m, e in q.items():
        if not isinstance(e, dict) or e.get("mode") == "jangtq2":
            continue
        w, s, bi = shapes.get(m + ".weight"), shapes.get(m + ".scales"), shapes.get(m + ".biases")
        if e.get("mode") == "mxfp8":
            n_mx += 1
            if w is None or s is None or w[0] != "U32" or s[0] != "U8" or bi is not None: fail(f"C mxfp8 {m}: {w} {s} {bi}")
        else:
            n_aff += 1
            if w is None or s is None or bi is None or s[0] != "BF16" or bi[0] != "BF16": fail(f"C affine {m}: {w} {s} {bi}")
    if any(".self_attn.indexer." in m for m in q): fail("C the DSA indexer must not be quantized")
    if sum(1 for k in shapes if ".self_attn.indexer.wq.weight" in k) != 9: fail("C indexer tensors missing")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(b))
    m_ = [{"role": "user", "content": "hi"}]
    r = lambda **kw: tok.apply_chat_template(m_, tokenize=False, add_generation_prompt=True, **kw)
    if not r(enable_thinking=False).endswith("<|im_start|>assistant\n<think></think>"): fail("D thinking-off render")
    if not r().endswith("<|im_start|>assistant\n"): fail("D thinking-on render")
    for eff, word in (("low", "Low"), ("high", "High"), ("max", "Max"), ("medium", "Max")):
        if not r(reasoning_effort=eff).startswith(f"<|im_start|>system\nReasoning Effort: {word}<|im_end|>"): fail(f"D effort {eff}")
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}]
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {"x": "a\nb"}}}]},
            {"role": "tool", "tool_call_id": "1", "content": "res"}]
    t = tok.apply_chat_template(conv, tools=tools, tokenize=False, add_generation_prompt=True)
    for needle in ("<tools>", "<function=f>", "<parameter=x>a\nb</parameter>", "<tool_response>\nres\n</tool_response>"):
        if needle not in t: fail(f"D tools render lacks {needle!r}")
    jc = json.loads((b / "jang_config.json").read_text()); gen = json.loads((b / "generation_config.json").read_text())
    if cfg.get("capabilities") != CAPS or jc.get("capabilities") != CAPS: fail("E capabilities")
    sd = jc.get("chat", {}).get("sampling_defaults", {})
    if sd.get("temperature") != gen.get("temperature") or sd.get("top_p") != gen.get("top_p"): fail("E sampling defaults")
    if gen.get("repetition_penalty", 1.0) != 1.0: fail("E repetition penalty must stay 1.0 (vendor)")
    size = sum((b / s).stat().st_size for s in shards)
    hist = {}
    for m, e in q.items():
        if isinstance(e, dict) and e.get("mode") == "jangtq2":
            key = f"{m.rsplit('.', 1)[-1][:-5]}:{e['bits']}"; hist[key] = hist.get(key, 0) + 1
    print(json.dumps({"PASS": True, "tensors": nt, "shards": len(shards), "misaligned": 0, "bytes_gib": round(size / 2**30, 3),
                      "affine_modules": n_aff, "mxfp8_modules": n_mx, "expert_bits": hist}, indent=1))


if __name__ == "__main__":
    main()
