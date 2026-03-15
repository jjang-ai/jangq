"""
MLXQ Convert — End-to-end model quantization pipeline.
Created by Eric Jang (eric@vmlx.net)

Takes a HuggingFace model directory and produces a complete .mxq model.
Pipeline: detect arch → calibrate → allocate bits → quantize → pack → write
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from safetensors import safe_open

from .format.spec import DEFAULT_BLOCK_SIZE
from .format.writer import write_mxq_model
from .architectures import detect_architecture, get_layer_config, get_skip_tensors, summarize_architecture
from .calibrate import calibrate_from_weights, _load_bf16_tensor
from .allocate import allocate_bits_greedy, allocate_bits_profile, summarize_allocation
from .quantize import quantize_tensor, QuantizedTensor


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
    Convert a HuggingFace model to MXQ format.

    Args:
        model_path: path to source model directory
        output_path: path for output MXQ model directory
        target_bits: target average bits per weight (e.g., 2.0, 2.5, 3.0, 4.0)
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
    print(f"  MLXQ Convert")
    print(f"  Created by Eric Jang (eric@vmlx.net)")
    print(f"{'='*60}")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Target: {target_bits} bits/weight")
    print(f"  Block size: {block_size}")
    print(f"  Calibration: {calibration_method}")
    print(f"  Quantization: {quantization_method}")
    print(f"  AWQ scaling: {'yes (alpha=' + str(awq_alpha) + ')' if use_awq else 'no'}")
    print(f"{'='*60}\n")

    # Step 1: Detect architecture
    print("  [1/5] Detecting architecture...")
    arch_config = detect_architecture(model_path)
    print(f"  {summarize_architecture(arch_config)}\n")

    # Step 2: Calibrate (compute importance scores)
    print("  [2/5] Calibrating...")
    if imatrix_path:
        from safetensors.numpy import load_file
        importance_data = load_file(str(imatrix_path))
        print(f"  Loaded pre-computed imatrix: {imatrix_path}")
    elif calibration_method == "weights":
        importance_data = calibrate_from_weights(model_path, block_size)
    else:
        from .calibrate import calibrate_with_activations
        importance_data = calibrate_with_activations(model_path, block_size=block_size)

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

                shape = f.get_slice(tensor_name).get_shape()
                if len(shape) != 2:
                    continue

                n_weights = shape[0] * shape[1]
                n_blocks = (n_weights + block_size - 1) // block_size

                base_name = tensor_name
                if base_name.endswith(".weight"):
                    base_name = base_name[:-7]

                imp_key = f"{base_name}.importance"
                if imp_key in importance_data:
                    imp = importance_data[imp_key]
                else:
                    # Default importance if not in imatrix
                    imp = np.ones(n_blocks, dtype=np.float32) * 0.5

                all_tensors_info.append((tensor_name, shape, n_blocks, sf_path))
                all_importance.append(imp)
                all_tensor_names_for_alloc.extend([tensor_name] * n_blocks)

    # Concatenate all importance scores
    global_importance = np.concatenate(all_importance)

    # Determine number of transformer layers
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
    if profile:
        # Profile-based allocation (proven strategy: attn-high, MLP-low)
        print(f"  Using profile: {profile}")
        global_bit_alloc = allocate_bits_profile(all_tensor_names_for_alloc, profile)
    else:
        # Greedy importance-based allocation
        global_bit_alloc = allocate_bits_greedy(
            global_importance,
            target_bits,
            all_tensor_names_for_alloc,
            n_layers=n_layers,
            block_size=block_size,
        )

    alloc_summary = summarize_allocation(global_bit_alloc, all_tensor_names_for_alloc)
    actual_bits = alloc_summary["average_bits"]

    print(f"  Target bits: {target_bits}")
    print(f"  Actual bits: {actual_bits:.2f}")
    print(f"  Total blocks: {alloc_summary['total_blocks']:,}")
    for bw, info in alloc_summary["histogram"].items():
        print(f"    {bw}: {info['count']:,} blocks ({info['percent']}%)")

    # Step 4: Quantize each tensor
    print(f"\n  [4/5] Quantizing ({quantization_method})...")
    quantized_tensors = {}
    non_quantized_tensors = {}
    passthrough = {}  # non-quantized tensors (norms, biases, AWQ scales)
    offset = 0

    for tensor_name, shape, n_blocks, sf_path in tqdm(all_tensors_info, desc="  Quantizing"):
        with safe_open(str(sf_path), framework="numpy") as f:
            try:
                weights = f.get_tensor(tensor_name).astype(np.float32)
            except TypeError:
                # bfloat16 — load raw bytes and convert
                from .calibrate import _load_bf16_tensor
                weights = _load_bf16_tensor(sf_path, tensor_name, shape)

        # Get this tensor's bit allocation slice
        bit_alloc = global_bit_alloc[offset:offset + n_blocks]
        offset += n_blocks

        # Apply AWQ scaling if enabled
        awq_scales = None
        if use_awq:
            # Find matching activation norms for this tensor
            parts = tensor_name.split(".")
            # Extract layer index
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

        # Quantize (weights are AWQ-scaled if enabled)
        qt = quantize_tensor(weights, bit_alloc, block_size, method=quantization_method)

        # If AWQ was applied, we need to reverse it after dequant at runtime.
        # Store the AWQ scales as a companion tensor.
        # The runtime will: dequant(block) / awq_scales[cols]

        base_name = tensor_name
        if base_name.endswith(".weight"):
            base_name = base_name[:-7]
        quantized_tensors[base_name] = qt

        if awq_scales is not None:
            # Store AWQ scales as a passthrough tensor
            passthrough[f"{base_name}.awq_scales"] = awq_scales.astype(np.float16)

    # Step 4b: Collect non-quantized tensors (norms, biases)
    print("  Collecting non-quantized tensors...")
    quantized_bases = set(quantized_tensors.keys())

    for sf_path in weight_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for tensor_name in f.keys():
                base = tensor_name.replace(".weight", "")
                # Skip tensors we already quantized
                if base in quantized_bases:
                    continue

                # Include: norm weights, biases, small tensors
                is_norm = ("norm" in tensor_name.lower())
                is_bias = tensor_name.endswith(".bias")
                shape = f.get_slice(tensor_name).get_shape()
                is_small = len(shape) == 1  # 1D tensors (norms, biases)

                if is_norm or is_bias or is_small:
                    try:
                        tensor = f.get_tensor(tensor_name)
                    except TypeError:
                        tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

                    # Store as float16 for GPU compatibility
                    if tensor.dtype == np.float32:
                        tensor = tensor.astype(np.float16)
                    elif tensor.dtype != np.float16:
                        tensor = tensor.astype(np.float16)

                    passthrough[tensor_name] = tensor

    print(f"  Found {len(passthrough)} non-quantized tensors (norms, biases)")

    # Step 5: Write MXQ model
    print(f"\n  [5/5] Writing MXQ model...")

    # Load model config
    model_config = json.loads((model_path / "config.json").read_text())

    # Build MXQ config
    bit_widths_used = sorted(set(int(b) for b in global_bit_alloc))
    total_weight_bytes = sum(qt.qweight.nbytes for qt in quantized_tensors.values())

    mxq_config = {
        "quantization": {
            "method": "mxq-importance",
            "target_bits": target_bits,
            "actual_bits": round(actual_bits, 2),
            "block_size": block_size,
            "calibration_method": calibration_method,
            "quantization_method": quantization_method,
            "scoring_method": "weight-magnitude" if calibration_method == "weights" else "awq+hessian",
            "bit_widths_used": bit_widths_used,
            "quantization_scheme": "asymmetric",
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

    write_mxq_model(
        output_dir=output_path,
        quantized_tensors=quantized_tensors,
        model_config=model_config,
        mxq_config=mxq_config,
        tokenizer_files=tokenizer_files,
        importance_data=importance_data,
        passthrough_tensors=passthrough,
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
    print(f"  DONE!")
    print(f"  Output: {output_path}")
    print(f"  Size: {results['total_weight_gb']} GB")
    print(f"  Avg bits: {results['actual_bits']}")
    print(f"{'='*60}\n")

    return results


def _count_params_str(config: dict) -> str:
    """Estimate parameter count from model config."""
    hidden = config.get("hidden_size", 0)
    n_layers = config.get("num_hidden_layers", 0)
    intermediate = config.get("intermediate_size", 0)
    vocab = config.get("vocab_size", 0)

    # Rough estimate: embedding + (attention + MLP) * layers + lm_head
    attn_params = 4 * hidden * hidden  # Q, K, V, O
    mlp_params = 3 * hidden * intermediate  # gate, up, down
    layer_params = attn_params + mlp_params
    total = vocab * hidden + n_layers * layer_params + vocab * hidden

    if total > 1e9:
        return f"{total / 1e9:.1f}B"
    elif total > 1e6:
        return f"{total / 1e6:.0f}M"
    else:
        return f"{total:.0f}"
