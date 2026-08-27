"""Bit-exact download verification against the Hugging Face manifest.

- LFS files (safetensors shards): HF tree API publishes lfs.oid = SHA-256 of
  the exact bytes + size. We hash every local file and compare.
- Small files (tokenizer/configs/template/etc.): HF publishes the git blob
  oid = SHA-1 of "blob <size>\\0" + content. Computed and compared.

Any mismatch / missing / size-off file is listed and the exit code is 1.
Files still mid-download (size mismatch) are reported as PENDING, not FAIL,
unless --final is passed (then everything must match).

  python -m jang_tools.qwen4_exp.verify_download --dir ~/models/Qwen3.8-Flash-Next \
      --repo Qwen/Qwen3.8-Flash-Next [--final] [--workers 8]
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch_manifest(repo: str):
    files = {}
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    cursor = None
    while True:
        u = url + (f"&cursor={cursor}" if cursor else "")
        req = urllib.request.Request(u, headers={"User-Agent": "jang-verify"})
        with urllib.request.urlopen(req) as r:
            batch = json.load(r)
            link = r.headers.get("Link", "")
        for f in batch:
            if f["type"] != "file":
                continue
            ent = {"size": f["size"], "oid": f["oid"]}
            if f.get("lfs"):
                ent["sha256"] = f["lfs"]["oid"]
                ent["size"] = f["lfs"]["size"]
            files[f["path"]] = ent
        if 'rel="next"' in link:
            cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
        else:
            break
    return files


def sha256_file(path: Path, bufsize=32 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def verify_one(name: str, ent: dict, local_dir: Path, final: bool):
    p = local_dir / name
    if not p.exists():
        return name, "MISSING"
    sz = p.stat().st_size
    if sz != ent["size"]:
        return name, ("FAIL-SIZE" if final else "PENDING")
    if "sha256" in ent:
        got = sha256_file(p)
        return name, ("OK" if got == ent["sha256"] else "FAIL-SHA256")
    got = git_blob_sha1(p)
    return name, ("OK" if got == ent["oid"] else "FAIL-SHA1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest-out", default=None)
    args = ap.parse_args()

    local_dir = Path(args.dir).expanduser()
    manifest = fetch_manifest(args.repo)
    if args.manifest_out:
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=1))
    skip = {".gitattributes"}
    names = [n for n in sorted(manifest) if n not in skip]
    print(f"manifest: {len(names)} files "
          f"({sum(manifest[n]['size'] for n in names)/2**30:.1f} GiB)")

    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify_one, n, manifest[n], local_dir, args.final): n for n in names}
        done = 0
        for fut in cf.as_completed(futs):
            n, status = fut.result()
            results[n] = status
            done += 1
            if status not in ("OK", "PENDING"):
                print(f"  !! {status}: {n}", flush=True)
            if done % 25 == 0:
                print(f"  ...{done}/{len(names)}", flush=True)

    ok = sum(1 for s in results.values() if s == "OK")
    pend = sum(1 for s in results.values() if s == "PENDING")
    bad = {n: s for n, s in results.items() if s not in ("OK", "PENDING")}
    print(f"\nVERIFIED BIT-EXACT: {ok}/{len(names)}   pending: {pend}   bad: {len(bad)}")
    for n, s in sorted(bad.items()):
        print(f"  {s}: {n}")
    if bad or (args.final and pend):
        raise SystemExit(1)
    print("ALL PRESENT FILES MATCH THE HF SHA MANIFEST" + (" (FINAL: complete)" if args.final else ""))


if __name__ == "__main__":
    main()
