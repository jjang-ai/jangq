"""SHA-256 every downloaded file against the Hub's LFS oids for a pinned revision.

  python -m jang_tools.dsv41.verify_source --dir ~/models/DeepSeek-V4.1-Flash-1017 \
      --repo deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.dir).expanduser()
    url = f"https://huggingface.co/api/models/{a.repo}/revision/{a.revision}?blobs=true"
    info = json.load(urllib.request.urlopen(url))
    expected = {s["rfilename"]: s["lfs"]["sha256"] for s in info["siblings"] if s.get("lfs")}
    files = sorted(expected)
    missing = [f for f in files if not (root / f).exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        got = dict(zip([f for f in files if f not in missing],
                       ex.map(sha256, [root / f for f in files if f not in missing])))
    bad = [f for f, h in got.items() if h != expected[f]]
    report = {"repo": a.repo, "revision": a.revision, "checked": len(got), "missing": missing, "mismatch": bad,
              "sha256": got}
    (root / "sha256-verification.json").write_text(json.dumps(report, indent=1))
    print(f"checked {len(got)} LFS files: {len(bad)} mismatched, {len(missing)} missing")
    for f in bad + missing:
        print("  BAD/MISSING", f)
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
