"""DeepSeek-V4.1-Flash serving acceptance against a real `vmlx serve` process.

One model load per run. Reports cold prefill, decode tok/s by content type, TTFT, RAM prefix
reuse, per-tool-call reuse, multiturn recall, multimodal answers, reasoning-parser leaks, and
the server's phys_footprint.

  python jang_tools/dsv41/tests/serve_acceptance.py --bundle <bundle> --image <red.png> \
      --disk-dir <dir> --out result.json [--ar] [--restart-probe]

--ar sets VMLX_DSV41_DSPARK=0 (AR baseline). --restart-probe only re-sends the SSD probe
prompt (run it after a normal run, with the same --disk-dir, to prove SSD persistence).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--image", help="a solid-red PNG (required unless --no-vision)")
ap.add_argument("--no-vision", action="store_true", help="text-only bundle (e.g. DSV4-0731): skip the image checks")
ap.add_argument("--image2", help="a solid-blue PNG (multimodal prefix-reuse miss check)")
ap.add_argument("--disk-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8170)
ap.add_argument("--ar", action="store_true")
ap.add_argument("--restart-probe", action="store_true")
a = ap.parse_args()

env = dict(os.environ)
env.pop("VMLX_DSV41_DSPARK", None)
if a.ar:
    env["VMLX_DSV41_DSPARK"] = "0"
log_path = os.path.splitext(a.out)[0] + ".serve.log"
logf = open(log_path, "w")
proc = subprocess.Popen(["vmlx", "serve", a.bundle, "--port", str(a.port), "--host", "127.0.0.1",
                         "--use-paged-cache", "--enable-block-disk-cache", "--block-disk-cache-dir",
                         a.disk_dir, "--enable-auto-tool-choice"], stdout=logf, stderr=subprocess.STDOUT, env=env)
base = f"http://127.0.0.1:{a.port}"
def _data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


IMG = None if a.no_vision else _data_url(a.image)
R: dict = {"mode": "ar" if a.ar else "dspark"}
MID = None


def post(body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(messages, mx=64, think=False, tools=None, effort=None):
    b = {"model": MID, "messages": messages, "max_tokens": mx, "temperature": 0, "stream": False,
         "chat_template_kwargs": {"enable_thinking": think}}
    if tools:
        b["tools"] = tools
    if effort:
        b["reasoning_effort"] = effort
    t0 = time.time()
    r = json.loads(post(b).read())
    dt = time.time() - t0
    u = r.get("usage") or {}
    m = r["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls"), "ptok": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "s": dt}


def stream_decode(messages, mx=256):
    """Decode tok/s excluding prefill: first delta to last, tokens from usage."""
    b = {"model": MID, "messages": messages, "max_tokens": mx, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    first = last = None
    usage = {}
    with post(b) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or "") + (d.get("reasoning_content") or ""):
                    now = time.time()
                    first = first or now
                    last = now
    n = usage.get("completion_tokens") or 0
    dec = (last - first) if first and last and last > first else None
    return {"ctok": n, "ttft_s": round(first - t0, 3) if first else None,
            "decode_tps": round((n - 1) / dec, 1) if dec and n > 1 else None}


def footprint():
    out = subprocess.run(["vmmap", "--summary", str(proc.pid)], capture_output=True, text=True).stdout
    unit = {"K": 1 / 1048576, "M": 1 / 1024, "G": 1}
    m = re.search(r"Physical footprint:\s+([\d.]+)([KMG])", out)
    p = re.search(r"Physical footprint \(peak\):\s+([\d.]+)([KMG])", out)
    return {"footprint_gb": float(m[1]) * unit[m[2]] if m else None,
            "peak_gb": float(p[1]) * unit[p[2]] if p else None}


SSD_PROMPT = [{"role": "user", "content": "SSD persistence probe. " + " ".join(
    f"Record {i}: sensor value {i * 37 % 101} at station {i % 13}." for i in range(260)) + " Summarize in one line."}]
try:
    for _ in range(600):
        time.sleep(2)
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode}; see {log_path}")
        try:
            urllib.request.urlopen(base + "/health", timeout=5)
            break
        except Exception:
            pass
    MID = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=5).read())["data"][0]["id"]
    chat([{"role": "user", "content": "hi"}], mx=4)
    R["footprint_after_load"] = footprint()
    if a.restart_probe:
        r = chat(SSD_PROMPT, mx=8)
        R["ssd_restart"] = {"ptok": r["ptok"], "cached": r["cached"], "s": round(r["s"], 2)}
    else:
        tag = f"{R['mode']}-{int(time.time())}"  # unique text: no prefix hit, cold Engram rows
        for n in (1000, 4000, 16000):
            body = " ".join(f"Item {tag}-{n}-{i}: the quick brown fox number {i} jumps over lazy dog {i * 7}."
                            for i in range(n // 18))
            r = chat([{"role": "user", "content": body + " Reply OK."}], mx=1)
            R[f"cold_prefill_{n}"] = {"ptok": r["ptok"], "s": round(r["s"], 2), "pp_s": round(r["ptok"] / r["s"], 1)}
        R["decode"] = {name: stream_decode([{"role": "user", "content": msg}]) for name, msg in [
            ("code", "Write a Python function that checks whether a string is a palindrome, with a docstring and two examples."),
            ("code2", "Write a JavaScript function that deep-clones an object, with comments."),
            ("structured", "List the 12 months of the year as a JSON array of objects with name and days."),
            ("prose", "Write a short story about a lighthouse keeper who finds a message in a bottle.")]}
        r1, r2 = chat(SSD_PROMPT, mx=8), chat(SSD_PROMPT, mx=8)
        R["prefix_ram"] = [{"ptok": x["ptok"], "cached": x["cached"], "s": round(x["s"], 2)} for x in (r1, r2)]
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        conv = [{"role": "system", "content": "You are a helpful assistant with tools."},
                {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
        t1 = chat(conv, mx=96, tools=tools)
        R["tool_t1"] = {"tool_calls": t1["tool_calls"], "content": t1["content"][:160]}
        if t1["tool_calls"]:
            tc = t1["tool_calls"]
            conv2 = conv + [{"role": "assistant", "content": t1["content"], "tool_calls": tc},
                            {"role": "tool", "tool_call_id": tc[0].get("id", "call_0"), "content": '{"temp_c": 18, "sky": "cloudy"}'}]
            t2 = chat(conv2, mx=64, tools=tools)
            R["tool_t2"] = {"content": t2["content"][:160], "ptok": t2["ptok"], "cached": t2["cached"], "s": round(t2["s"], 2)}
        mt = [{"role": "user", "content": "My name is Sam and my favorite color is teal. Acknowledge briefly."}]
        x = chat(mt, mx=40)
        mt += [{"role": "assistant", "content": x["content"]}, {"role": "user", "content": "What is my favorite color? One word."}]
        y = chat(mt, mx=10)
        mt += [{"role": "assistant", "content": y["content"]}, {"role": "user", "content": "And my name? One word."}]
        z = chat(mt, mx=10)
        R["multiturn"] = {"color": y["content"], "name": z["content"]}
        if not a.no_vision:
            sysm = {"role": "system", "content": "Answer in one word."}
            u1 = {"role": "user", "content": [{"type": "text", "text": "What color is this image?"},
                                              {"type": "image_url", "image_url": {"url": IMG}}]}
            m1 = chat([sysm, u1], mx=8)
            conv_mm = [sysm, u1, {"role": "assistant", "content": m1["content"]},
                       {"role": "user", "content": "Name a fruit of exactly that color."}]
            m2 = chat(conv_mm, mx=8)
            conv_mm += [{"role": "assistant", "content": m2["content"]}, {"role": "user", "content": "What color was the image again?"}]
            m3 = chat(conv_mm, mx=8)
            R["multimodal"] = {"answers": [m1["content"], m2["content"], m3["content"]],
                               "cached": [m1["cached"], m2["cached"], m3["cached"]]}
            # Multimodal prefix reuse: a conversation long enough to be stored (> 256 tokens). The text
            # follow-up should hit the cache entry from the image turn; a different image must miss.
            long_sys = {"role": "system", "content": "You are a careful visual assistant. " + " ".join(
                f"Guideline {i}: describe only what is visible, answer in one word when asked." for i in range(40))}
            def img_turn(url):
                return {"role": "user", "content": [{"type": "text", "text": "What color is this image? One word."},
                                                    {"type": "image_url", "image_url": {"url": url}}]}
            v1 = chat([long_sys, img_turn(IMG)], mx=8)
            conv_v = [long_sys, img_turn(IMG), {"role": "assistant", "content": v1["content"]},
                      {"role": "user", "content": "Name a fruit of exactly that color. One word."}]
            v2 = chat(conv_v, mx=8)
            R["multimodal_reuse"] = {"answers": [v1["content"], v2["content"]], "ptok": [v1["ptok"], v2["ptok"]],
                                     "cached": [v1["cached"], v2["cached"]], "s": [round(v1["s"], 2), round(v2["s"], 2)]}
            if a.image2:
                v3 = chat([long_sys, img_turn(_data_url(a.image2))], mx=8)
                R["multimodal_reuse"].update(other_image=v3["content"], other_image_cached=v3["cached"])
        rr = chat([{"role": "user", "content": "What is 17*23?"}], mx=300, think=True, effort="high")
        R["reasoning"] = {"reasoning_len": len(rr["reasoning"]), "content": rr["content"][:80],
                          "leak": "<think" in rr["content"] or "</think>" in rr["content"]}
        R["footprint_end"] = footprint()
finally:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    logf.close()
    keep = ("persist_ms", "wait_ms", "paged cache hit", "Stored paged", "DSpark", "Traceback", "Error")
    R["log"] = [l.rstrip()[:240] for l in open(log_path) if any(k in l for k in keep)][-80:]
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "log"}, indent=1))
