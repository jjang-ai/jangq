"""Read JANGQ-AI/Naive-N0.5-Flash-JANGH2 back from the Hub and compare with the local final bundle."""
import hashlib, json, os, sys
from huggingface_hub import HfApi, hf_hub_download
REPO = "JANGQ-AI/Naive-N0.5-Flash-JANGH2"
LOCAL = os.path.expanduser(sys.argv[1])
a = HfApi(); info = a.model_info(REPO, files_metadata=True)
hub = {s.rfilename: s for s in info.siblings}
local = []
for root, dirs, files in os.walk(LOCAL):
    dirs[:] = [d for d in dirs if d != ".cache"]
    for f in files:
        local.append(os.path.relpath(os.path.join(root, f), LOCAL))
bad = []
for rel in sorted(local):
    s = hub.get(rel)
    if s is None: bad.append(("missing on hub", rel)); continue
    p = os.path.join(LOCAL, rel)
    if s.size != os.path.getsize(p): bad.append(("size", rel)); continue
    if s.lfs:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(64 << 20), b""): h.update(chunk)
        if h.hexdigest() != s.lfs.sha256: bad.append(("sha256", rel))
extra = sorted(set(hub) - set(local) - {".gitattributes"})
for n in ("config.json", "jang_config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer_config.json"):
    remote = json.load(open(hf_hub_download(REPO, n, force_download=True)))
    if remote != json.load(open(os.path.join(LOCAL, n))): bad.append(("json content", n))
for n in ("README.md", "chat_template.jinja"):
    if open(hf_hub_download(REPO, n, force_download=True)).read() != open(os.path.join(LOCAL, n)).read(): bad.append(("text content", n))
cfg = json.load(open(hf_hub_download(REPO, "config.json"))); jc = json.load(open(hf_hub_download(REPO, "jang_config.json")))
idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(idx.values())); hub_shards = sorted(f for f in hub if f.endswith(".safetensors"))
if shards != hub_shards: bad.append(("hub tensor files != indexed shards", str(set(hub_shards) ^ set(shards))))
if sum(f.endswith("index.json") for f in hub if "/" not in f) != 1: bad.append(("index count", ""))
print(json.dumps({"repo": REPO, "private": info.private, "gated": info.gated, "hub_files": len(hub), "local_files": len(local),
                  "hub_only": extra, "problems": bad, "jangtq": {k: cfg["jangtq"][k] for k in ("version", "rotation", "codebook_family")},
                  "format": jc["format"], "reasoning_parser": cfg["capabilities"]["reasoning_parser"], "tool_parser": cfg["capabilities"]["tool_parser"],
                  "reasoning_efforts": cfg["capabilities"]["reasoning_efforts"], "caps_equal": cfg["capabilities"] == jc["capabilities"],
                  "shards": len(shards), "tensors": len(idx), "license": (info.card_data or {}).get("license"), "base_model": (info.card_data or {}).get("base_model"),
                  "total_gib": round(sum(s.size or 0 for s in info.siblings) / 2**30, 3)}, indent=1))
sys.exit(1 if bad or extra else 0)
