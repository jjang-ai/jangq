"""Decision-point corpus for the live probes: renders the glm_probe / agentic_eval prompts EXACTLY as a
thinking-off generation prompt (tools in system, add_generation_prompt, enable_thinking=False), then appends "</think>"
so the LAST position predicts the first content token (tool call or prose). Also records the position that predicts
"</think>" itself. All rows are ref (bf16 reference via stream_capture)."""
import argparse, json
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
lookup = {"type": "function", "function": {"name": "lookup_code", "description": "Look up the secret code for a label. The code cannot be known without calling this tool.",
          "parameters": {"type": "object", "properties": {"label": {"type": "string", "description": "the label to look up"}}, "required": ["label"]}}}
probe_tool = {"type": "function", "function": {"name": "lookup_code", "description": "Look up the code for a label.",
              "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}}}
cases = [
    ("probe_p1_single_tool", [probe_tool], "Use lookup_code to retrieve the code for label alpha. Report the returned code."),
    ("glmprobe_p1_single_tool", [lookup], "Use lookup_code to retrieve the code for label alpha. Report the returned code."),
    ("glmprobe_p2_single_tool", [lookup], "What is the secret code for the label 'alpha'? You must call the lookup_code tool; do not guess."),
]
END_THINK = tok.convert_tokens_to_ids("</think>")
with open(a.out, "w") as f:
    for name, tools, q in cases:
        s = tok.apply_chat_template([{"role": "user", "content": q}], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = list(tok(s, add_special_tokens=False)["input_ids"])
        if ids[-1] != END_THINK:
            ids = ids + [END_THINK]
            dec = [[len(ids) - 2, "end_think"], [len(ids) - 1, "call"]]
        else:
            dec = [[len(ids) - 1, "call"]]
        f.write(json.dumps({"ids": ids, "ref": True, "domain": name, "decisions": dec, "rendered_tail": s[-120:]}) + "\n")
        print(name, len(ids), repr(s[-60:]))
