"""Emit an explicit mlx-convention per-module `quantization` block into a
JANG bundle's config.json, derived from actual shard shapes.

For every U32 packed weight with sibling scales: in_features comes from the
bf16 SOURCE checkpoint headers (bundle names translated back through the
sanitize rules), then  bits = 32*packed_cols/in  and  gs = in/groups.
No guessing, no b*g ambiguity. Fail-closed: any unresolved tensor aborts.
"""
import json
import struct
import sys
from collections import Counter
from pathlib import Path


def read_headers(d: Path) -> dict:
    out = {}
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    for f in sorted(set(idx["weight_map"].values())):
        with open(d / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = v["shape"] if isinstance(v, dict) else v
    return out


def source_candidates(name: str):
    """Bundle tensor name (without .weight) -> (source key, in_axis) candidates."""
    import re
    cands = []
    m = re.search(r"\.ple\.ngram_embedding\.shards\.(\d+)$", name)
    if m:
        base = re.sub(r"\.ple\.ngram_embedding\.shards\.\d+$",
                      f".ple.ple_embedding.ngram_embedding.shard_{m.group(1)}", name)
        return [("model." + base + ".weight", -1), (base + ".weight", -1)]
    if ".switch_mlp.gate_proj" in name or ".switch_mlp.up_proj" in name:
        base = name.replace(".switch_mlp.gate_proj", ".experts.gate_up_proj")
        base = base.replace(".switch_mlp.up_proj", ".experts.gate_up_proj")
        for p in ("model.language_model.", "model.", ""):
            cands.append((p + base.replace("language_model.", "", 1) if p else base, -1))
        cands.append((base, -1))
    elif ".switch_mlp.down_proj" in name:
        base = name.replace(".switch_mlp.down_proj", ".experts.down_proj")
        for p in ("model.language_model.", "model.", ""):
            cands.append((p + base.replace("language_model.", "", 1) if p else base, -1))
        cands.append((base, -1))
    else:
        w = name + ".weight"
        variants = [w]
        if w.startswith("language_model."):
            variants.append("model." + w)
            variants.append("model.language_model." + w[len("language_model."):])
        if w.startswith("visual."):
            variants.append("model." + w)
        if w.startswith("mtp."):
            variants.append("model." + w)
        variants.append("model." + w)
        for v in variants:
            cands.append((v, -1))
        cands.append((name, -1))
        cands.append(("model." + name, -1))
    return cands


def emit(bundle: str, source: str) -> dict:
    b, s = Path(bundle).expanduser(), Path(source).expanduser()
    bh = read_headers(b)
    sh = read_headers(s)
    block, unresolved = {}, []
    for k, shape in bh.items():
        if not k.endswith(".weight"):
            continue
        base = k[:-len(".weight")]
        scales = bh.get(base + ".scales")
        if scales is None:
            continue
        packed, groups = shape[-1], scales[-1]
        in_dim = None
        for src_key, axis in source_candidates(base):
            if src_key + ".weight" in sh:
                in_dim = sh[src_key + ".weight"][axis]
                break
            if src_key in sh:
                in_dim = sh[src_key][axis]
                break
        if in_dim is None:
            legal = [(g, 32 * packed // (groups * g)) for g in (32, 64, 128)
                     if (32 * packed) % (groups * g) == 0
                     and 32 * packed // (groups * g) in (2, 3, 4, 5, 6, 8)]
            if len(legal) == 1:
                gs, bits = legal[0]
                block[base] = {"group_size": gs, "bits": bits}
            else:
                unresolved.append((k, packed, groups, legal))
            continue
        if in_dim % groups or (32 * packed) % in_dim:
            unresolved.append((k, packed, groups, f"in={in_dim} not divisible"))
            continue
        gs, bits = in_dim // groups, 32 * packed // in_dim
        if bits not in (2, 3, 4, 5, 6, 8) or gs not in (32, 64, 128):
            unresolved.append((k, packed, groups, f"derived b={bits} g={gs}"))
            continue
        block[base] = {"group_size": gs, "bits": bits}
    if unresolved:
        for row in unresolved[:20]:
            print("UNRESOLVED:", row)
        raise SystemExit(f"emit_quant_config: {len(unresolved)} unresolved tensors — aborting")
    modal_g = Counter(v["group_size"] for v in block.values()).most_common(1)[0][0]
    modal_b = Counter(v["bits"] for v in block.values()).most_common(1)[0][0]
    q = {"group_size": modal_g, "bits": modal_b}
    q.update(dict(sorted(block.items())))
    cfg_path = b / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization"] = q
    tmp = cfg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    tmp.replace(cfg_path)
    print(f"quantization block: {len(block)} modules, modal {modal_b}b/g{modal_g} -> {cfg_path}")
    return q


if __name__ == "__main__":
    emit(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "~/models/Qwen3.8-Flash-Next")
