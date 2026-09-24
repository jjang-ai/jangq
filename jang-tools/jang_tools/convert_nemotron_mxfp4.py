"""
Nemotron-3-Nano-Omni-30B-A3B (nemotron_h) → MXFP4 (text-only) Conversion
Created by Jinho Jang (eric@jangq.ai)

MXFP4 path: stock MLX 4-bit affine grouped quantization (group_size=32, bits=4),
no TurboQuant codec. Loads via `mlx_lm.load()` directly with the in-tree
`mlx_lm.models.nemotron_h` module — no monkey-patches required.

Design:
  - strip multimodal towers (vision_model RADIO + sound_encoder parakeet)
  - strip `language_model.` prefix
  - PRE-STACK routed experts at convert time so stock mlx_lm.load() works:
      input :  language_model.backbone.layers.N.mixer.experts.E.{up,down}_proj.weight
      output:  backbone.layers.N.mixer.switch_mlp.{fc1,fc2}.{weight,scales,biases}
        where fc1=up_proj, fc2=down_proj (per mlx_lm.nemotron_h.sanitize map)
    Per-expert tensors stacked into (n_experts, out, in) → mx.quantize once.

What's quantized:
  routed experts (stacked), shared experts, attention, mamba in/out_proj,
  embed, lm_head → all 4-bit affine group_size=32

What's passthrough fp16:
  norms, A_log, D, dt_bias, conv1d.{weight,bias}, gate.weight, biases,
  e_score_correction_bias
"""
import sys, json, gc, shutil, re
import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
_args, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

from jang_tools.progress import ProgressEmitter
progress = ProgressEmitter(
    json_to_stderr=(_args.progress == "json"),
    quiet_text=_args.quiet_text,
)

from jang_tools.calibrate import _load_bf16_tensor

if len(sys.argv) < 3:
    print(
        "usage: python -m jang_tools.convert_nemotron_mxfp4 <src_bf16_dir> <out_dir> [bits] [group_size]\n"
        "  bits        default 4\n"
        "  group_size  default 32",
        file=sys.stderr,
    )
    sys.exit(2)
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
GROUP = int(sys.argv[4]) if len(sys.argv) > 4 else 32
# argv[5]: "affine" (default, back-compat) or "mxfp8"/"mxfp4" for true MX
# formats. MX modes return (packed_weight, scales) with NO biases — the scales
# are e8m0 shared exponents, not fp16 affine scales.
MODE = sys.argv[5] if len(sys.argv) > 5 else "affine"
IS_MX = MODE.startswith("mxfp")

# Routed-expert key pattern. group(1)=layer_idx, group(2)=expert_idx, group(3)=proj
EXPERT_KEY_RE = re.compile(
    r"^(?:language_model\.)?backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)
PROJ_TO_FC = {"up_proj": "fc1", "down_proj": "fc2"}

try:
    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC / "config.json") as f:
        full_config = json.load(f)
    # Nemotron-Omni nests the LM under llm_config; Nemotron 3.5 Lightning is flat.
    llm_config = full_config.get("llm_config") or full_config.get("text_config") or full_config
    if "num_hidden_layers" not in llm_config:
        raise SystemExit(f"{SRC}/config.json has no LM config — wrong source?")
    IS_WRAPPED = bool(full_config.get("llm_config"))

    n_layers = llm_config.get("num_hidden_layers", 52)
    n_experts = llm_config.get("n_routed_experts", 128)
    pattern = llm_config.get("hybrid_override_pattern", "")
    if not pattern and llm_config.get("layers_block_type"):
        _M = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
        pattern = "".join(_M[t] for t in llm_config["layers_block_type"])

    def get_method(tensor_name: str) -> str:
        n = tensor_name
        if (n.startswith("vision_model.") or n.startswith("sound_encoder.")
                or n.startswith("mlp1.") or n.startswith("sound_projection.")):
            return "drop"
        if n.startswith("language_model.mtp.") or n.startswith("mtp."):
            # Nemotron-Omni ships no usable MTP; Nemotron 3.5 Lightning ships a
            # real 1.335 B head worth retaining.
            return "drop" if IS_WRAPPED else "affine"
        if EXPERT_KEY_RE.match(n):
            return "expert"  # buffered for stacked quantization
        if n.endswith(".norm.weight") or "norm_f.weight" in n or "mixer.norm.weight" in n:
            return "passthrough"
        if n.endswith(".A_log") or n.endswith(".D") or n.endswith(".dt_bias"):
            return "passthrough"
        if "conv1d.weight" in n or "conv1d.bias" in n:
            return "passthrough"
        if n.endswith(".mixer.gate.weight") or n.endswith(".mixer.gate.e_score_correction_bias"):
            return "passthrough"
        if n.endswith(".bias"):
            return "passthrough"
        return "affine"

    def _quant(w):
        """Quantize under MODE. Returns (qw, qs, qb|None)."""
        if IS_MX:
            qw, qs = mx.quantize(w, group_size=GROUP, bits=BITS, mode=MODE)
            return qw, qs, None
        qw, qs, qb = mx.quantize(w, group_size=GROUP, bits=BITS)
        return qw, qs, qb

    def _emit(base, qw, qs, qb):
        add_tensor(f"{base}.weight", np.array(qw))
        # MX scales are uint8 e8m0 exponents — casting them to fp16 would
        # corrupt the format, so preserve the dtype the kernel expects.
        add_tensor(f"{base}.scales",
                   np.array(qs) if IS_MX else np.array(qs).astype(np.float16))
        if qb is not None:
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))

    def dst_name(src: str) -> str:
        if src.startswith("language_model."):
            return src[len("language_model."):]
        return src

    print("=" * 70)
    print(f"  Nemotron-3-Nano-Omni-30B-A3B → MXFP{BITS} (group_size={GROUP}) Conversion")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 70)
    print(f"  Source: {SRC}")
    print(f"  Output: {OUT}")
    print(f"  Layers: {n_layers}  Experts: {n_experts}  Pattern: {pattern[:30]}...")
    print(flush=True)

    progress.phase(1, 3, "scan")
    print("\n  Scanning source...", flush=True)
    # Two collections: simple per-tensor list, and expert groups by (layer, proj).
    simple_tensors = []  # (key, shape, sf)
    # expert_groups[(layer, proj)] = {expert_idx: (shape, sf)}
    expert_groups: dict[tuple[int, str], dict[int, tuple[list, Path]]] = {}
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                if k.endswith("_scale_inv"):
                    continue
                shape = list(f.get_slice(k).get_shape())
                m = EXPERT_KEY_RE.match(k)
                if m:
                    layer_idx = int(m.group(1))
                    expert_idx = int(m.group(2))
                    proj = m.group(3)
                    expert_groups.setdefault((layer_idx, proj), {})[expert_idx] = (shape, sf)
                else:
                    simple_tensors.append((k, shape, sf))
    n_expert_tensors = sum(len(eg) for eg in expert_groups.values())
    print(f"  Found {len(simple_tensors)} simple tensors + "
          f"{len(expert_groups)} expert groups ({n_expert_tensors} per-expert tensors)",
          flush=True)

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000
    total_affine_simple = 0
    total_affine_grouped = 0
    total_passthrough = 0
    total_dropped = 0
    shard_map: dict[str, str] = {}

    def flush_shard():
        global shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(OUT / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.2f} GB",
              flush=True)
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, arr: np.ndarray):
        global shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    # Resume support
    done_keys: set[str] = set()
    existing_shards = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
    if existing_shards:
        import struct as _struct
        for sf in existing_shards:
            with open(sf, "rb") as f:
                hsize = _struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hsize))
            for k in hdr:
                if k != "__metadata__":
                    done_keys.add(k)
                    shard_map[k] = sf.name
            shard_idx = max(shard_idx, int(sf.name.split("-")[1]))
        print(f"  Resume: {len(done_keys)} keys, continuing from shard {shard_idx + 1}",
              flush=True)

    def is_simple_done(dst_n: str, method: str) -> bool:
        if method == "passthrough":
            return dst_n in done_keys
        if method == "affine":
            base = dst_n[:-len(".weight")] if dst_n.endswith(".weight") else dst_n
            return (f"{base}.weight" in done_keys
                    and f"{base}.scales" in done_keys
                    and (IS_MX or f"{base}.biases" in done_keys))
        return False

    def is_group_done(layer: int, proj: str) -> bool:
        fc = PROJ_TO_FC[proj]
        base = f"backbone.layers.{layer}.mixer.switch_mlp.{fc}"
        return (f"{base}.weight" in done_keys
                and f"{base}.scales" in done_keys
                and (IS_MX or f"{base}.biases" in done_keys))

    progress.phase(2, 3, "convert simple")
    print("\n  Converting simple tensors...", flush=True)
    skipped_resume = 0
    for src_name, shape, sf_path in tqdm(simple_tensors, desc="  Simple"):
        method = get_method(src_name)
        if method == "drop":
            total_dropped += 1
            continue

        dst_n = dst_name(src_name)
        if done_keys and is_simple_done(dst_n, method):
            skipped_resume += 1
            if method == "affine":  total_affine_simple += 1
            else:                   total_passthrough += 1
            continue

        with safe_open(str(sf_path), framework="numpy") as f:
            try:
                tensor = f.get_tensor(src_name)
                if not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor)
            except Exception:
                tensor = _load_bf16_tensor(sf_path, src_name, shape)
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        if method == "passthrough" or tensor.ndim < 2:
            # `e_score_correction_bias` is the DeepSeek-V3 noaux_tc routing bias
            # and is the ONLY fp32 tensor family in the Nemotron 3.5 Lightning
            # checkpoint (24 of 6513) — NVIDIA raised precision there and
            # nowhere else. fp16 has a 10-bit mantissa and min normal ~6e-5;
            # with 128 experts separated by small learned offsets that changes
            # expert selection. Observed live: an fp16-biased MXFP8 bundle
            # declined a tool call that the fp32 JANG bundles answered.
            if dst_n.endswith("e_score_correction_bias") and tensor.dtype == np.float32:
                add_tensor(dst_n, tensor)
            else:
                add_tensor(dst_n, tensor.astype(np.float16))
            total_passthrough += 1
        elif method == "affine":
            w = mx.array(tensor.astype(np.float16))
            qw, qs, qb = _quant(w)
            base = dst_n[:-len(".weight")] if dst_n.endswith(".weight") else dst_n
            _emit(base, qw, qs, qb)
            total_affine_simple += 1
            del w, qw, qs, qb

        del tensor
        if total_affine_simple % 200 == 0:
            gc.collect()

    print(f"\n  Converting expert groups (stacked quantization)...", flush=True)
    sorted_groups = sorted(expert_groups.keys())
    for (layer, proj) in tqdm(sorted_groups, desc="  Expert groups"):
        if done_keys and is_group_done(layer, proj):
            skipped_resume += n_experts
            total_affine_grouped += 1
            continue

        experts_meta = expert_groups[(layer, proj)]
        if len(experts_meta) != n_experts:
            print(f"  WARN: layer {layer} {proj} has {len(experts_meta)} experts, expected {n_experts}",
                  flush=True)

        # Load all experts in expert-index order, stack to (n_experts, out, in).
        stacked = []
        for e in range(n_experts):
            shape, sf_path = experts_meta[e]
            _pfx = "language_model." if IS_WRAPPED else ""
            src_name = f"{_pfx}backbone.layers.{layer}.mixer.experts.{e}.{proj}.weight"
            with safe_open(str(sf_path), framework="numpy") as f:
                try:
                    t = f.get_tensor(src_name)
                    if not isinstance(t, np.ndarray):
                        t = np.array(t)
                except Exception:
                    t = _load_bf16_tensor(sf_path, src_name, shape)
            if t.dtype != np.float32:
                t = t.astype(np.float32)
            stacked.append(t.astype(np.float16))
        stacked_np = np.stack(stacked, axis=0)
        del stacked
        # Quantize the whole 3D tensor at once. mx.quantize groups along the
        # last dim (in_features), produces (n_experts, out, in/(32//bits))
        # packed weights + (n_experts, out, in/group_size) scales/biases.
        w = mx.array(stacked_np)
        del stacked_np
        qw, qs, qb = _quant(w)
        fc = PROJ_TO_FC[proj]
        base = f"backbone.layers.{layer}.mixer.switch_mlp.{fc}"
        _emit(base, qw, qs, qb)
        total_affine_grouped += 1
        del w, qw, qs, qb
        gc.collect()

    flush_shard()
    if skipped_resume > 0:
        print(f"\n  Resume: skipped {skipped_resume} tensor-equivalents", flush=True)

    progress.phase(3, 3, "write")
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}

    total_bytes = sum(sf.stat().st_size for sf in OUT.glob("model-*-of-*.safetensors"))
    index = {"metadata": {"format": "mlx", "total_size": total_bytes},
             "weight_map": shard_map}
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    out_config = dict(llm_config)
    out_config["model_type"] = "nemotron_h"
    out_config["quantization"] = {"group_size": GROUP, "bits": BITS}
    if IS_MX:
        # Declare the MX mode so the runtime dispatches to the right kernel and
        # the bundle is not mistaken for affine (mlxstudio#130-class metadata bug).
        out_config["quantization"]["mode"] = MODE
    out_config["_jang_source"] = full_config.get("model_type", "NemotronH_Nano_Omni_Reasoning_V3")
    out_config["_jang_modality"] = "text"

    with open(OUT / "config.json", "w") as f:
        json.dump(out_config, f, indent=2)

    jang_config = {
        "version": 2,
        "weight_format": "mlx",
        "profile": f"MXFP{BITS}",
        "source_model": {
            "name": "Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
            "org": "nvidia",
            "architecture": "nemotron_h",
            "wrapper_arch": "NemotronH_Nano_Omni_Reasoning_V3",
            "modality": "text",
        },
        "quantization": {
            "method": "affine",
            "group_size": GROUP,
            "bits": BITS,
        },
        "hybrid_pattern": pattern,
    }
    try:
        from jang_tools.capabilities import build_capabilities
        caps = build_capabilities(jang_config, out_config, OUT)
        if caps is not None:
            jang_config["capabilities"] = caps
    except Exception as _e:
        print(f"  capabilities skipped: {_e}", flush=True)

    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "chat_template.jinja", "chat_template.json",
              "merges.txt", "vocab.json",
              "configuration_nemotron_h.py", "modeling_nemotron_h.py"]:
        src_f = SRC / f
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / f))

    print(f"\n  Done!")
    print(f"  Affine simple tensors: {total_affine_simple}")
    print(f"  Affine grouped (per layer×proj, n_experts={n_experts}): {total_affine_grouped}")
    print(f"  Passthrough tensors:   {total_passthrough}")
    print(f"  Dropped (vision/audio): {total_dropped}")
    print(f"  Output: {OUT}")
    progress.done(ok=True, output=str(OUT))

except Exception as _exc:
    progress.done(ok=False, error=f"{type(_exc).__name__}: {_exc}")
    raise
