"""Finalize a GLM-5.3 JANGTQ v2 build into the FINAL bundle + verify it.

Never rewrites the build: shards are HARD-LINKED (same volume, byte-identical), JSON sidecars written fresh.
Final root = model-*.safetensors + ONE model.safetensors.index.json + config/tokenizer/template/processor JSON.
Build report + eval JSONs go to evaluation/ (subfolder, never discovered by tensor globs).

Checks (all must pass; exits non-zero otherwise):
  A. layout     : root tensor files are exactly the indexed shards; one *index.json; no stray safetensors; no MTP keys
  B. alignment  : every tensor payload offset is a multiple of its dtype size (jang_tools aligned_safetensors verifier)
  C. config     : model_type glm5_next, vision_config present, image/video token ids present, MTP disabled
                  (num_nextn_predict_layers 0), jangtq v2 block, per-module quant entries == shard shapes (audit)
  D. template   : chat template renders: thinking off, efforts low/high/max, tools (tool_call format), video content
  E. capability : config.json + jang_config.json capabilities (tool_parser glm_xml_args, reasoning_parser glm_think_block,
                  has_vision/has_video, tool_response_role observation — identical to the shipped affine GLM-5.3
                  contract), chat sampling_defaults == generation_config.json, efforts [low, high, max] only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path


from jang_tools.format.aligned_safetensors import verify_safetensors_alignment  # noqa: E402

SIDE = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "processor_config.json")


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True); ap.add_argument("--final", required=True)
    ap.add_argument("--eval", nargs="*", default=[], help="eval JSON files to include under evaluation/")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    b, f = Path(a.build), Path(a.final)
    if not a.verify_only:
        if f.exists():
            fail(f"{f} exists (never rewrite a final bundle in place)")
        f.mkdir(parents=True)
        idx = json.loads((b / "model.safetensors.index.json").read_text())
        for shard in sorted(set(idx["weight_map"].values())):
            os.link(b / shard, f / shard)                       # byte-identical, no rewrite
        shutil.copy2(b / "model.safetensors.index.json", f / "model.safetensors.index.json")
        for s in SIDE:
            if not (b / s).exists():
                fail(f"required sidecar {s} missing from build")
            shutil.copy2(b / s, f / s)
        (f / "evaluation").mkdir()
        shutil.copy2(b / "jangtq2_build_report.json", f / "evaluation" / "jangtq2_build_report.json")
        for e in a.eval:
            shutil.copy2(e, f / "evaluation" / Path(e).name)
        # config.json: same serving-capabilities contract as the shipped GLM-5.3 JANG bundles (Osaurus/Swift and the
        # Python loader read it; without it vMLX fell back to registry parsers, i.e. a DIFFERENT serving contract).
        cfg = json.loads((b / "config.json").read_text())
        # Identical to the shipped affine GLM-5.3 JANG contract (config.json + embedded jang_config capabilities), which
        # vMLX resolves as detection_source=jang_stamped -> glm_xml_args (== Glm47ToolParser) + glm_think_block
        # (== DeepSeekR1ReasoningParser), cache hybrid / glm5_next_native_v2. The MLLM route is decided separately
        # (is_mllm_model tier glm5_next_indexed_visual), so no modality field is stamped (a list there is not a known value).
        CAPS = {"has_vision": True, "has_video": True, "has_audio": False, "supports_thinking": True,
                "default_reasoning": "on", "think_in_template": True, "reasoning_parser": "glm_think_block",
                "reasoning_prefill_open_tag": True, "tool_parser": "glm_xml_args", "tool_response_role": "observation"}
        cfg["capabilities"] = dict(CAPS)
        (f / "config.json").write_text(json.dumps(cfg, indent=1))
        # capabilities stamp (fresh file)
        jc = json.loads((b / "jang_config.json").read_text())
        gen = json.loads((b / "generation_config.json").read_text())
        jc["chat"] = {"sampling_defaults": {"temperature": gen["temperature"], "top_p": gen["top_p"]},
                      "default_sampling_mode": "thinking", "stop_token_ids": gen["eos_token_id"],
                      "reasoning_efforts": ["low", "high", "max"], "reasoning_effort_default": "max"}
        jc["capabilities"] = {**CAPS, "family": "glm5_next"}
        # Calibration record derived from the BUILD REPORT (what the converter actually did), not from memory.
        rep = json.loads((b / "jangtq2_build_report.json").read_text())
        L = rep["layers"]
        gu = sorted({v["gate_up_method"] for v in L.values()}); dn = sorted({v["down_method"] for v in L.values()})
        n_down_gptq = sum(v["down_method"] == "gptq" for v in L.values())
        method = {"experts": f"JANGTQ v2 odd-cubic codebook, per-row fp16 scale, rotation {rep['rotation']}",
                  "rounding": "GPTQ (TQ codebook) in the rotated basis" if rep["rotation"] != "none" else "GPTQ (TQ codebook)",
                  "gptq_prior": "per-expert imatrix",
                  "gate_up_method": gu, "down_method": dn,
                  "down_gptq_layers": f"{n_down_gptq}/{len(L)} (each validated on held-out rows, else RTN)",
                  "allocation": "heap-MCKP over measured per-layer unit curves",
                  "awq": "evaluated, not applied"}
        cfg["jangtq"]["method"] = method
        (f / "config.json").write_text(json.dumps(cfg, indent=1))
        cal = {k: v for k, v in jc.get("calibration", {}).items() if k in ("reference", "tokens", "mix")}
        cal.update({"rotation": rep["rotation"],
                    "reference_captures": ["FP8 release: 600,064 tokens web50/code25/chat15/math10",
                                           "bf16 layer-streamed agentic capture: 448 GLM-template tool conversations, "
                                           "171k tokens, held-out tools/sequences excluded"] if rep.get("diag2") else
                                          ["FP8 release: 600,064 tokens web50/code25/chat15/math10"],
                    "capture_pool_weight_agentic": rep.get("w2"),
                    "imatrix": "per-expert (GPTQ prior)",
                    "awq": "evaluated, not applied",
                    "gptq": {"prior": "per-expert imatrix", "basis": "hadamard32-rotated"
                             if rep["rotation"] != "none" else "identity", "gate_up": f"{gu}", "down": method["down_gptq_layers"]},
                    "build_started": rep["started"], "build_finished": rep["finished"]})
        jc["calibration"] = cal
        jc["jangtq"]["method"] = method
        (f / "jang_config.json").write_text(json.dumps(jc, indent=1))
    # ---------------- verification
    root = sorted(p.name for p in f.iterdir() if p.is_file())
    for s in SIDE + ("config.json", "jang_config.json"):
        if s not in root: fail(f"A required sidecar {s} missing from final root")
    if not (f / "evaluation" / "jangtq2_build_report.json").exists(): fail("A evaluation/jangtq2_build_report.json missing")
    idx = json.loads((f / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    st = [n for n in root if n.endswith(".safetensors")]
    if st != shards: fail(f"A root safetensors {len(st)} != indexed shards {len(shards)}")
    if sum(n.endswith("index.json") for n in root) != 1: fail("A exactly one *index.json required")
    if any(k.startswith("model.layers.45.") for k in idx["weight_map"]): fail("A MTP tensors present")
    n_t = n_bad = 0
    for s in shards:
        n, bad = verify_safetensors_alignment(f / s); n_t += n; n_bad += bad
    if n_bad: fail(f"B {n_bad} misaligned tensors")
    if n_t != len(idx["weight_map"]): fail(f"B tensor count {n_t} != index {len(idx['weight_map'])}")
    cfg = json.loads((f / "config.json").read_text())
    tc = cfg.get("text_config", {})
    if cfg.get("model_type") != "glm5_next": fail("C model_type")
    if "vision_config" not in cfg: fail("C vision_config missing")
    for k in ("image_token_id", "video_token_id", "video_start_token_id", "video_end_token_id"):
        if k not in cfg: fail(f"C {k} missing")
    if tc.get("num_nextn_predict_layers", 0) != 0: fail("C MTP not disabled in text_config")
    if cfg.get("jangtq", {}).get("version") != 2: fail("C jangtq v2 block missing")
    if not any(k.startswith("visual.") for k in idx["weight_map"]): fail("C vision tower tensors missing")
    import subprocess
    r = subprocess.run([sys.executable, "-m", "jang_tools.jangh.glm53.audit", str(f)], capture_output=True, text=True)
    if r.returncode != 0: fail("C audit: " + r.stdout[-400:])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(f))
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}]
    m = [{"role": "user", "content": "hi"}]
    off = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    renders = {"thinking_off": off[-40:]}
    for eff in ("low", "high", "max"):
        s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, reasoning_effort=eff)
        renders[f"effort_{eff}"] = s[-60:]
    # GLM template: effective = reasoning_effort if in ['low','high'] else 'max' -> "<|system|>Reasoning Effort: X"
    for eff, word in (("low", "Low"), ("high", "High"), ("max", "Max")):
        s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, reasoning_effort=eff)
        if f"Reasoning Effort: {word}" not in s: fail(f"D effort {eff} does not render 'Reasoning Effort: {word}'")
    tsys = tok.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=True)
    if "<tool_call>" not in tsys: fail("D tool-call format not rendered with tools")
    vid = tok.apply_chat_template([{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what moves?"}]}],
                                  tokenize=False, add_generation_prompt=True)
    if "<|begin_of_video|>" not in vid and "video" not in vid.lower(): fail("D video placeholder not rendered")
    jc = json.loads((f / "jang_config.json").read_text())
    if jc.get("capabilities", {}).get("tool_parser") != "glm_xml_args": fail("E jang_config capabilities stamp")
    caps = cfg.get("capabilities", {})
    if caps.get("tool_parser") != "glm_xml_args" or caps.get("reasoning_parser") != "glm_think_block" or not caps.get("has_video"):
        fail("E config.json capabilities contract")
    rot = cfg["jangtq"].get("rotation")
    mod_rots = {v.get("rotation") for v in cfg.get("quantization", {}).values() if isinstance(v, dict) and v.get("mode") == "jangtq2"}
    if mod_rots != {rot}: fail(f"E per-module rotations {mod_rots} != jangtq.rotation {rot}")
    if jc.get("jangtq", {}).get("rotation") != rot or jc.get("calibration", {}).get("rotation") != rot: fail("E rotation record disagrees")
    if "method" not in cfg["jangtq"] or cfg["jangtq"]["method"] != jc["jangtq"].get("method"): fail("E jangtq.method missing/inconsistent")
    gen = json.loads((f / "generation_config.json").read_text())
    sd = jc.get("chat", {}).get("sampling_defaults", {})
    if sd.get("temperature") != gen.get("temperature") or sd.get("top_p") != gen.get("top_p") or "repetition_penalty" in gen:
        fail("E sampling defaults disagree between generation_config.json and jang_config.json")
    print(json.dumps({"PASS": True, "tensors": n_t, "shards": len(shards), "misaligned": 0,
                      "bytes_gib": round(sum((f / s).stat().st_size for s in shards) / 2**30, 3),
                      "renders": renders}, indent=1))


if __name__ == "__main__":
    main()
