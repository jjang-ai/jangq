"""agentic_ref in the model's NATIVE generation format.

Measured 2026-09-28: after "</think>" the BF16 model predicts "\\n\\n" with p ~ 1.0 (105 of 112 call positions, 64 of 96
answer positions) and decides between <tool_call> and text on the token AFTER it; the chat template renders assistant
turns WITHOUT that "\\n\\n" ("</think><tool_call>"). A decision probe on the template rendering therefore measures
whether the model predicts "\\n\\n" — nothing else. Here every assistant "</think>" of the 96 held-out tool
conversations is followed by the "\\n\\n" token, and the decision position is that token.
"""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
NL = tok("\n\n", add_special_tokens=False)["input_ids"]; assert len(NL) == 1, NL
ENDT = tok.convert_tokens_to_ids("</think>"); TC = tok.convert_tokens_to_ids("<tool_call>")
n = call = ans = 0
with open(sys.argv[3], "w") as f:
    for l in open(sys.argv[2]):
        r = json.loads(l)
        if r["set"] != "agentic_ref":
            continue
        dec = {p: k for p, k in r["decisions"]}
        ids, out = r["ids"], []
        d2 = []
        for p, t in enumerate(ids):
            out.append(t)
            if p in dec:
                assert t == ENDT
                out.append(NL[0]); d2.append([len(out) - 1, dec[p]])
                assert (ids[p + 1] == TC) == (dec[p] == "call")
        f.write(json.dumps({"ids": out, "ref": True, "set": "agentic_ref", "domain": "tool_conv", "decisions": d2}) + "\n")
        n += 1; call += sum(k == "call" for _, k in d2); ans += sum(k == "answer" for _, k in d2)
print(f"{n} conversations, decisions call {call} answer {ans}; example tail:", repr(tok.decode(out[d2[-1][0] - 4: d2[-1][0] + 3])))
