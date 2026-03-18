"""
JANG Convert — End-to-end model quantization pipeline.
Created by Jinho Jang (eric@jangq.ai)

Takes a HuggingFace model directory and produces a JANG v2 model.
v2 outputs MLX-native safetensors — loads instantly via mx.load() mmap.

Pipeline: detect arch → calibrate → allocate bits → quantize → write
"""

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from safetensors import safe_open

from .format.spec import DEFAULT_BLOCK_SIZE
from .format.writer import write_jang_v2_model
from .architectures import detect_architecture, get_layer_config, get_skip_tensors, summarize_architecture
from .calibrate import calibrate_from_weights, _load_bf16_tensor
from .allocate import allocate_bits_greedy, allocate_bits_profile, summarize_allocation, JANG_PROFILES


# Pattern for per-expert 2D tensors (MiniMax/Mixtral style)
# e.g. model.layers.0.block_sparse_moe.experts.5.w1.weight
_PER_EXPERT_PATTERN = re.compile(
    r"(.+)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\.weight$"
)

# MiniMax/Mixtral name mapping: w1→gate_proj, w2→down_proj, w3→up_proj
_EXPERT_NAME_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _get_tensor_group_size(tensor_name: str, default_gs: int, num_experts: int = 0) -> int:
    """
    Get the optimal group_size for a specific tensor.

    Rules (from CRACK abliteration research on MiniMax, Mar 2026):
    - MoE router/gate: ALWAYS gs=64 (precision-critical, tiny tensor)
    - Expert MLP with 150+ experts at 2-4 bit: gs=128 (gather_qmm cache pressure)
    - Everything else: gs=64 (standard, best precision)

    The default_gs is the model's global group_size (set by auto-detection).
    This function overrides it for specific tensors that need different values.
    """
    name_lower = tensor_name.lower()

    # MoE router/gate — ALWAYS gs=64 for precision (tiny tensor, speed irrelevant)
    # Matches: mlp.gate.weight, block_sparse_moe.gate.weight, shared_expert_gate.weight
    if ".gate." in name_lower or name_lower.endswith(".gate"):
        return 64
    if "shared_expert_gate" in name_lower:
        return 64

    # Everything else uses the model's global group_size
    # (which is 128 for 150+ expert models, 64 otherwise)
    return default_gs


def convert_model(
    model_path: str | Path,
    output_path: str | Path,
    target_bits: float = 2.5,
    block_size: int = DEFAULT_BLOCK_SIZE,
    calibration_method: str = "weights",
    quantization_method: str = "mse",
    imatrix_path: Optional[str | Path] = None,
    use_awq: bool = False,
    awq_alpha: float = 0.25,
    profile: Optional[str] = None,
) -> dict:
    """
    Convert a HuggingFace model to JANG v2 format.

    v2 stores weights as MLX-native uint32 packed + float16 scales/biases
    in standard safetensors. Loading is instant via mx.load() mmap — no
    repacking step at load time.

    Args:
        model_path: path to source model directory
        output_path: path for output JANG model directory
        target_bits: target average bits per weight
        block_size: weights per quantization block
        calibration_method: "weights" (fast) or "activations" (better quality)
        quantization_method: "rtn" (fast) or "mse" (better quality)
        imatrix_path: pre-computed importance matrix (skip calibration if provided)

    Returns:
        dict with conversion results and quality metrics
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"  JANG Convert v2")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print(f"{'='*60}")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Target: {target_bits} bits/weight")
    print(f"  Block size: {block_size}")
    print(f"  Calibration: {calibration_method}")
    print(f"  Quantization: {quantization_method}")
    print(f"  Format: v2 (MLX-native, instant load)")
    print(f"  AWQ scaling: {'yes (alpha=' + str(awq_alpha) + ')' if use_awq else 'no'}")
    print(f"{'='*60}\n")

    # Check source model exists
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"No config.json in {model_path} — is this a HuggingFace model directory?"
        )

    # Check tie_word_embeddings early
    _raw_config = json.loads((model_path / "config.json").read_text())
    _tie_embeddings = _raw_config.get("tie_word_embeddings", False) or _raw_config.get("text_config", {}).get("tie_word_embeddings", False)

    # Step 1: Detect architecture
    print("  [1/5] Detecting architecture...")
    arch_config = detect_architecture(model_path)
    print(f"  {summarize_architecture(arch_config)}\n")

    # Auto-detect optimal group_size for MoE models with many experts.
    # Models with 150+ experts suffer 15-25% speed regression at group_size=64
    # due to gather_qmm kernel cache pressure. group_size=128 eliminates this.
    # Discovered via CRACK abliteration research (Mar 5 2026).
    num_experts = getattr(arch_config, 'num_experts', 0)
    if block_size == DEFAULT_BLOCK_SIZE and arch_config.has_moe_layers:
        if num_experts >= 150:
            block_size = 128
            print(f"  Auto group_size: {num_experts} experts detected → group_size=128 (speed fix)")
        elif num_experts >= 64:
            # 64-149 experts: warn but keep 64 (quality vs speed tradeoff)
            print(f"  Note: {num_experts} experts. Consider -b 128 if speed is priority.")

    # Step 2: Calibrate (compute importance scores)
    # Skip calibration for profile and K-quant allocation — they use tier
    # classification only, never importance scores. Saves 10-30 seconds.
    from .allocate import is_k_quant
    needs_calibration = not (profile and (profile.upper() in JANG_PROFILES or is_k_quant(profile.upper())))
    # Also skip for integer target bits (routed to K-quant in step 3)
    if not needs_calibration and target_bits in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        needs_calibration = False

    if needs_calibration:
        print("  [2/5] Calibrating...")
        if imatrix_path:
            from safetensors.numpy import load_file
            importance_data = load_file(str(imatrix_path))
            print(f"  Loaded pre-computed imatrix: {imatrix_path}")
        elif calibration_method == "weights":
            # Save imatrix to output dir (not source dir) to avoid polluting source
            imatrix_out = output_path / "jang_imatrix.safetensors" if output_path else None
            importance_data = calibrate_from_weights(model_path, block_size, output_path=imatrix_out)
        else:
            from .calibrate import calibrate_with_activations
            imatrix_out = output_path / "jang_imatrix.safetensors" if output_path else None
            importance_data = calibrate_with_activations(model_path, block_size=block_size, output_path=imatrix_out)
    else:
        print("  [2/5] Skipping calibration (not needed for profile/K-quant allocation)")
        importance_data = {}

    # Step 2b: AWQ activation norms (optional)
    awq_norms = {}
    if use_awq:
        print("\n  [2b] Collecting AWQ activation norms...")
        try:
            from .awq import collect_activation_norms_mlx
            awq_norms = collect_activation_norms_mlx(str(model_path))
        except ImportError:
            print("  WARNING: MLX not available, skipping AWQ scaling")
            use_awq = False

    # Step 3: Load weights and allocate bits
    print("\n  [3/5] Allocating bits...")
    skip_patterns = get_skip_tensors(arch_config)
    weight_files = sorted(model_path.glob("*.safetensors"))

    # Collect all weight tensor info
    all_tensors_info = []  # (name, shape, n_blocks)
    all_importance = []
    all_tensor_names_for_alloc = []

    for sf_path in weight_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for tensor_name in f.keys():
                if any(skip in tensor_name for skip in skip_patterns):
                    continue
                if tensor_name.endswith(".bias"):
                    continue
                if "layernorm" in tensor_name.lower() or "rmsnorm" in tensor_name.lower():
                    continue
                if "weight_scale_inv" in tensor_name or "_scale_inv" in tensor_name:
                    continue
                if "lm_head" in tensor_name and _tie_embeddings:
                    continue
                if ".visual." in tensor_name or "vision_tower" in tensor_name or "vision_model" in tensor_name:
                    continue

                shape = f.get_slice(tensor_name).get_shape()
                if len(shape) < 2:
                    continue

                n_weights = 1
                for d in shape:
                    n_weights *= d
                if len(shape) > 2 and n_weights < 100_000:
                    continue

                n_blocks = (n_weights + block_size - 1) // block_size

                base_name = tensor_name
                if base_name.endswith(".weight"):
                    base_name = base_name[:-7]

                imp_key = f"{base_name}.importance"
                if imp_key in importance_data:
                    imp = importance_data[imp_key]
                else:
                    imp = np.ones(n_blocks, dtype=np.float32) * 0.5

                all_tensors_info.append((tensor_name, shape, n_blocks, sf_path))
                all_importance.append(imp)
                all_tensor_names_for_alloc.extend([tensor_name] * n_blocks)

    global_importance = np.concatenate(all_importance)

    layer_indices = set()
    for name in all_tensor_names_for_alloc:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    n_layers = max(layer_indices) + 1 if layer_indices else 1

    # Run bit allocation
    from .allocate import is_k_quant, k_quant_target, allocate_bits_budget
    if profile and is_k_quant(profile):
        k_target = k_quant_target(profile)
        print(f"  Using K-quant: {profile} (target: {k_target} avg bits)")
        global_bit_alloc = allocate_bits_budget(
            all_tensor_names_for_alloc, target_bits=k_target,
        )
    elif profile:
        print(f"  Using profile: {profile}")
        global_bit_alloc = allocate_bits_profile(all_tensor_names_for_alloc, profile)
    elif target_bits in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        print(f"  Using K-quant allocation (target: {target_bits} avg bits)")
        global_bit_alloc = allocate_bits_budget(
            all_tensor_names_for_alloc, target_bits=target_bits,
        )
    else:
        global_bit_alloc = allocate_bits_greedy(
            global_importance, target_bits, all_tensor_names_for_alloc,
            n_layers=n_layers, block_size=block_size,
        )

    alloc_summary = summarize_allocation(global_bit_alloc, all_tensor_names_for_alloc)
    actual_bits = alloc_summary["average_bits"]

    print(f"  Target bits: {target_bits}")
    print(f"  Actual bits: {actual_bits:.2f}")
    print(f"  Total blocks: {alloc_summary['total_blocks']:,}")
    for bw, info in alloc_summary["histogram"].items():
        print(f"    {bw}: {info['count']:,} blocks ({info['percent']}%)")

    # Step 4: Quantize and build MLX-native tensors
    print(f"\n  [4/5] Quantizing to MLX-native format ({quantization_method})...")

    # v2 output: flat dict of tensor_name → numpy array (MLX format)
    v2_tensors = {}
    # Buffer for per-expert 2D tensors that need stacking
    expert_buffer = {}  # (layer_prefix, wtype) → {expert_id: {weight, scales, biases}}
    passthrough = {}
    offset = 0

    for tensor_name, shape, n_blocks, sf_path in tqdm(all_tensors_info, desc="  Quantizing"):
        # Skip vision conv weights — Conv3d/Conv2d needs float, not uint32.
        # These are passthrough tensors that get saved as float16.
        name_lower = tensor_name.lower()
        if ("patch_embed" in name_lower or "temporal_embed" in name_lower) and ".weight" in name_lower:
            with safe_open(str(sf_path), framework="numpy") as f:
                try:
                    w = f.get_tensor(tensor_name)
                except TypeError:
                    w = _load_bf16_tensor(sf_path, tensor_name, shape)
                passthrough[tensor_name] = w.astype(np.float16) if w.dtype != np.float16 else w
            offset += n_blocks
            continue

        with safe_open(str(sf_path), framework="numpy") as f:
            try:
                weights = f.get_tensor(tensor_name).astype(np.float32)
            except (TypeError, AttributeError):
                from .fp8 import load_fp8_tensor
                from .calibrate import _load_bf16_tensor
                scale_key = f"{tensor_name}_scale_inv"
                scale_inv = None
                try:
                    scale_inv = f.get_tensor(scale_key)
                except Exception:
                    pass
                try:
                    weights = load_fp8_tensor(sf_path, tensor_name, shape, scale_inv)
                except Exception:
                    weights = _load_bf16_tensor(sf_path, tensor_name, shape)

        bit_alloc = global_bit_alloc[offset:offset + n_blocks]
        offset += n_blocks

        # AWQ scaling
        awq_scales = None
        if use_awq:
            parts = tensor_name.split(".")
            layer_key = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_key = f"layers.{layer_idx}.attn_input"
                    except ValueError:
                        pass
                    break
            if layer_key and layer_key in awq_norms:
                from .awq import compute_awq_scales, apply_awq_scaling
                in_features = shape[1] if len(shape) == 2 else shape[-1]
                norms = awq_norms[layer_key]
                if len(norms) == in_features:
                    awq_scales = compute_awq_scales(norms, alpha=awq_alpha)
                    weights = apply_awq_scaling(weights, awq_scales)

        bits = int(bit_alloc[0])
        w_shape = weights.shape
        is_3d = len(w_shape) >= 3

        # --- Quantize via mx.quantize() → MLX-native uint32 output ---
        try:
            import mlx.core as mx

            # Per-tensor group_size: router/gate gets gs=64, experts get model default
            tensor_gs = _get_tensor_group_size(tensor_name, block_size, num_experts)

            if is_3d:
                weights = weights.reshape(-1, weights.shape[-1])

            n_elements = weights.shape[0] * weights.shape[1]
            if n_elements > 100_000_000:
                chunk_rows = max(1, 100_000_000 // weights.shape[1])
                all_qw, all_s, all_b = [], [], []
                for start in range(0, weights.shape[0], chunk_rows):
                    end = min(start + chunk_rows, weights.shape[0])
                    chunk = mx.array(weights[start:end].astype(np.float16))
                    cqw, cs, cb = mx.quantize(chunk, group_size=tensor_gs, bits=bits)
                    all_qw.append(np.array(cqw))
                    all_s.append(np.array(cs))
                    all_b.append(np.array(cb))
                    mx.synchronize()
                mlx_weight = np.concatenate(all_qw, axis=0)
                mlx_scales = np.concatenate(all_s, axis=0).astype(np.float16)
                mlx_biases = np.concatenate(all_b, axis=0).astype(np.float16)
            else:
                w_mx = mx.array(weights.astype(np.float16))
                qw, scales, biases = mx.quantize(w_mx, group_size=tensor_gs, bits=bits)
                # Keep as numpy with MLX shapes — uint32 weights, float16 scales/biases
                mlx_weight = np.array(qw)       # uint32, (out_dim, packed_per_row)
                mlx_scales = np.array(scales).astype(np.float16)  # (out_dim, n_groups)
                mlx_biases = np.array(biases).astype(np.float16)  # (out_dim, n_groups)

            # Reshape back to 3D for expert weights
            if is_3d:
                n_experts = w_shape[0]
                expert_out = w_shape[1]
                mlx_weight = mlx_weight.reshape(n_experts, expert_out, -1)
                mlx_scales = mlx_scales.reshape(n_experts, expert_out, -1)
                mlx_biases = mlx_biases.reshape(n_experts, expert_out, -1)

        except (ImportError, ValueError) as _exc:
            # Fallback: RTN quantization → convert to uint32 shaped
            if isinstance(_exc, ImportError):
                print(f"  WARNING: mlx not available for {tensor_name}, using RTN fallback (lower quality)")
                print(f"           Install mlx for best results: pip install 'jang[mlx]'")
            from .quantize import quantize_tensor
            qt = quantize_tensor(weights.reshape(-1, weights.shape[-1]) if is_3d else weights,
                                 bit_alloc, tensor_gs, method="rtn")
            out_dim = weights.reshape(-1, weights.shape[-1]).shape[0] if is_3d else weights.shape[0]
            in_dim = weights.shape[-1]
            packed_per_row = (in_dim * bits + 31) // 32
            n_groups = (in_dim + tensor_gs - 1) // tensor_gs

            packed_bytes = qt.qweight.tobytes()
            pad_needed = (4 - len(packed_bytes) % 4) % 4
            if pad_needed:
                packed_bytes += b'\x00' * pad_needed
            mlx_weight = np.frombuffer(packed_bytes, dtype=np.uint32)[:out_dim * packed_per_row].copy()
            mlx_weight = mlx_weight.reshape(out_dim, packed_per_row)
            mlx_scales = qt.scales.reshape(out_dim, n_groups)
            mlx_biases = qt.biases.reshape(out_dim, n_groups)

            if is_3d:
                n_experts = w_shape[0]
                expert_out = w_shape[1]
                mlx_weight = mlx_weight.reshape(n_experts, expert_out, -1)
                mlx_scales = mlx_scales.reshape(n_experts, expert_out, -1)
                mlx_biases = mlx_biases.reshape(n_experts, expert_out, -1)

        # Clear Metal cache between tensors
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

        # --- Store with MLX-ready names and shapes ---
        base_name = tensor_name
        if base_name.endswith(".weight"):
            base_name = base_name[:-7]

        if awq_scales is not None:
            passthrough[f"{base_name}.awq_scales"] = awq_scales.astype(np.float16)

        # Handle gate_up_proj splitting (Qwen3.5 MoE fused projections)
        if "gate_up_proj" in base_name:
            if is_3d:
                # 3D: (n_experts, 2*inter, packed) → split into gate + up
                mid = mlx_weight.shape[1] // 2
                gate_base = base_name.replace("experts.gate_up_proj", "switch_mlp.gate_proj")
                up_base = base_name.replace("experts.gate_up_proj", "switch_mlp.up_proj")
                v2_tensors[f"{gate_base}.weight"] = mlx_weight[:, :mid, :]
                v2_tensors[f"{gate_base}.scales"] = mlx_scales[:, :mid, :]
                v2_tensors[f"{gate_base}.biases"] = mlx_biases[:, :mid, :]
                v2_tensors[f"{up_base}.weight"] = mlx_weight[:, mid:, :]
                v2_tensors[f"{up_base}.scales"] = mlx_scales[:, mid:, :]
                v2_tensors[f"{up_base}.biases"] = mlx_biases[:, mid:, :]
            else:
                # 2D: (2*inter, packed) → split into gate + up
                mid = mlx_weight.shape[0] // 2
                gate_base = base_name.replace("gate_up_proj", "gate_proj")
                up_base = base_name.replace("gate_up_proj", "up_proj")
                v2_tensors[f"{gate_base}.weight"] = mlx_weight[:mid, :]
                v2_tensors[f"{gate_base}.scales"] = mlx_scales[:mid, :]
                v2_tensors[f"{gate_base}.biases"] = mlx_biases[:mid, :]
                v2_tensors[f"{up_base}.weight"] = mlx_weight[mid:, :]
                v2_tensors[f"{up_base}.scales"] = mlx_scales[mid:, :]
                v2_tensors[f"{up_base}.biases"] = mlx_biases[mid:, :]

        # Handle 3D expert down_proj renaming
        elif is_3d and "experts" in base_name and "down_proj" in base_name:
            sw_base = base_name.replace("experts.down_proj", "switch_mlp.down_proj")
            v2_tensors[f"{sw_base}.weight"] = mlx_weight
            v2_tensors[f"{sw_base}.scales"] = mlx_scales
            v2_tensors[f"{sw_base}.biases"] = mlx_biases

        # Handle per-expert 2D tensors (MiniMax/Mixtral: experts.N.w1)
        elif _PER_EXPERT_PATTERN.match(tensor_name):
            m = _PER_EXPERT_PATTERN.match(tensor_name)
            prefix = m.group(1)
            expert_id = int(m.group(2))
            wtype = m.group(3)
            group_key = (prefix, wtype)
            if group_key not in expert_buffer:
                expert_buffer[group_key] = {}
            expert_buffer[group_key][expert_id] = {
                "weight": mlx_weight,
                "scales": mlx_scales,
                "biases": mlx_biases,
            }

        # Standard tensor — store directly
        else:
            v2_tensors[f"{base_name}.weight"] = mlx_weight
            v2_tensors[f"{base_name}.scales"] = mlx_scales
            v2_tensors[f"{base_name}.biases"] = mlx_biases

        del weights

    # --- Stack per-expert 2D weights into 3D QuantizedSwitchLinear format ---
    if expert_buffer:
        print(f"  Stacking {len(expert_buffer)} expert groups into 3D...")
        for (prefix, wtype), experts in expert_buffer.items():
            new_name = _EXPERT_NAME_MAP.get(wtype, wtype)
            sw_key = f"{prefix}.switch_mlp.{new_name}"
            n_experts = max(experts.keys()) + 1

            v2_tensors[f"{sw_key}.weight"] = np.stack(
                [experts[e]["weight"] for e in range(n_experts)])
            v2_tensors[f"{sw_key}.scales"] = np.stack(
                [experts[e]["scales"] for e in range(n_experts)])
            v2_tensors[f"{sw_key}.biases"] = np.stack(
                [experts[e]["biases"] for e in range(n_experts)])
        expert_buffer.clear()

    # Step 4b: Collect non-quantized tensors (norms, biases, vision)
    print("  Collecting non-quantized tensors...")
    quantized_bases = set()
    for key in v2_tensors:
        if key.endswith(".weight"):
            quantized_bases.add(key[:-7])
        elif key.endswith(".scales") or key.endswith(".biases"):
            quantized_bases.add(key[:-7] if key.endswith(".scales") else key[:-7])

    for sf_path in weight_files:
        # Skip imatrix files (calibration-only, not model weights)
        if sf_path.name == "jang_imatrix.safetensors":
            continue
        with safe_open(str(sf_path), framework="numpy") as f:
            for tensor_name in f.keys():
                # Skip importance tensors (calibration-only)
                if tensor_name.endswith(".importance"):
                    continue
                base = tensor_name.replace(".weight", "")
                if base in quantized_bases:
                    continue

                is_norm = ("norm" in tensor_name.lower())
                is_bias = tensor_name.endswith(".bias")
                is_vision = (".visual." in tensor_name or "vision_tower" in tensor_name
                             or "vision_model" in tensor_name)
                shape = f.get_slice(tensor_name).get_shape()
                is_small = len(shape) == 1
                n_el = 1
                for d in shape:
                    n_el *= d
                is_tiny_nd = len(shape) > 2 and n_el < 100_000

                if is_norm or is_bias or is_small or is_tiny_nd or is_vision:
                    try:
                        tensor = f.get_tensor(tensor_name)
                    except TypeError:
                        tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

                    if tensor.dtype == np.float32:
                        tensor = tensor.astype(np.float16)
                    elif tensor.dtype != np.float16:
                        tensor = tensor.astype(np.float16)

                    passthrough[tensor_name] = tensor

    print(f"  Found {len(passthrough)} non-quantized tensors (norms, biases)")

    # Merge passthrough into v2_tensors
    v2_tensors.update(passthrough)

    # Step 5: Write JANG v2 model
    print(f"\n  [5/5] Writing JANG v2 model (MLX-native)...")

    model_config = json.loads((model_path / "config.json").read_text())
    bit_widths_used = sorted(set(int(b) for b in global_bit_alloc))
    total_weight_bytes = sum(
        arr.nbytes for name, arr in v2_tensors.items()
        if name.endswith(".weight") and arr.dtype == np.uint32
    )

    jang_config = {
        "quantization": {
            "method": "jang-importance",
            "profile": profile if profile else None,
            "target_bits": target_bits,
            "actual_bits": round(actual_bits, 2),
            "block_size": block_size,
            "calibration_method": calibration_method,
            "quantization_method": quantization_method,
            "scoring_method": "weight-magnitude" if calibration_method == "weights" else "awq+hessian",
            "bit_widths_used": bit_widths_used,
            "quantization_scheme": "asymmetric",
            "quantization_backend": "mx.quantize",
        },
        "source_model": {
            "name": model_path.name,
            "dtype": "bfloat16",
            "parameters": _count_params_str(model_config),
        },
        "architecture": {
            "type": arch_config.arch_type.value,
            "attention": arch_config.attention_type.value,
            "has_vision": arch_config.has_vision_encoder,
            "has_ssm": arch_config.has_ssm_layers,
            "has_moe": arch_config.has_moe_layers,
        },
        "runtime": {
            "total_weight_bytes": total_weight_bytes,
            "total_weight_gb": round(total_weight_bytes / (1024 ** 3), 2),
        },
    }

    # Copy tokenizer files
    tokenizer_files = {}
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.model", "merges.txt", "vocab.json"]:
        tok_path = model_path / tok_file
        if tok_path.exists():
            if tok_file.endswith(".json"):
                tokenizer_files[tok_file] = json.loads(tok_path.read_text())
            else:
                tokenizer_files[tok_file] = tok_path.read_text()

    # Copy VL processor files
    output_path.mkdir(parents=True, exist_ok=True)
    for vl_file in ["preprocessor_config.json", "video_preprocessor_config.json",
                     "chat_template.json"]:
        vl_path = model_path / vl_file
        if vl_path.exists():
            shutil.copy2(str(vl_path), str(output_path / vl_file))

    write_jang_v2_model(
        output_dir=output_path,
        tensors=v2_tensors,
        model_config=model_config,
        jang_config=jang_config,
        tokenizer_files=tokenizer_files,
        importance_data=importance_data,
    )

    results = {
        "source_model": str(model_path),
        "output_path": str(output_path),
        "target_bits": target_bits,
        "actual_bits": round(actual_bits, 2),
        "total_blocks": alloc_summary["total_blocks"],
        "total_weight_gb": round(total_weight_bytes / (1024 ** 3), 2),
        "histogram": alloc_summary["histogram"],
        "bit_widths_used": bit_widths_used,
    }

    print(f"\n{'='*60}")
    print(f"  DONE — JANG v2 (MLX-native)")
    print(f"  Output: {output_path}")
    print(f"  Size: {results['total_weight_gb']} GB")
    print(f"  Avg bits: {results['actual_bits']}")
    print(f"  Load time: instant (mx.load mmap)")
    print(f"{'='*60}\n")

    return results


def _count_params_str(config: dict) -> str:
    """Estimate parameter count from model config (best-effort, cosmetic only)."""
    # Check both top-level and text_config for VLM models
    tc = config.get("text_config", {})
    def _get(key, default=0):
        return config.get(key, tc.get(key, default))

    hidden = _get("hidden_size")
    n_layers = _get("num_hidden_layers")
    intermediate = _get("intermediate_size")
    vocab = _get("vocab_size")
    num_experts = _get("num_local_experts", _get("num_experts"))

    attn_params = 4 * hidden * hidden
    mlp_params = 3 * hidden * intermediate
    if num_experts > 1:
        mlp_params *= num_experts  # MoE: multiply by expert count
    layer_params = attn_params + mlp_params
    total = vocab * hidden + n_layers * layer_params

    if total > 1e9:
        return f"{total / 1e9:.1f}B"
    elif total > 1e6:
        return f"{total / 1e6:.0f}M"
    else:
        return f"{total:.0f}"
