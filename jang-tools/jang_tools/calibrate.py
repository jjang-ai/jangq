"""
JANG Calibration Engine — Compute importance scores from real model weights.
Created by Jinho Jang (eric@jangq.ai)

Runs calibration data through the full-precision model, collecting activation
statistics that determine which weights are most important. This is the
foundation that makes JANG's mixed-precision allocation work.

Supports multiple backends:
- safetensors: direct weight loading (no inference, activation-magnitude only)
- mlx: full forward pass with MLX (recommended for Apple Silicon)
- torch: full forward pass with PyTorch (fallback)
"""

import json
import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from safetensors.numpy import save_file


def _load_bf16_tensor(sf_path: Path, tensor_name: str, shape: tuple) -> np.ndarray:
    """
    Load a bfloat16 tensor from safetensors by reading raw bytes.

    bfloat16 is not supported by numpy. We read the raw uint16 values
    and convert to float32 by padding with 16 zero bits (bfloat16 is
    the upper 16 bits of a float32).
    """
    import safetensors

    # Read the raw file and parse header to find tensor offset
    with open(sf_path, "rb") as fh:
        header_size = struct.unpack("<Q", fh.read(8))[0]
        header_bytes = fh.read(header_size)
        header = json.loads(header_bytes)
        data_offset = 8 + header_size

        tensor_info = header[tensor_name]
        offsets = tensor_info["data_offsets"]
        start = data_offset + offsets[0]
        end = data_offset + offsets[1]

        fh.seek(start)
        raw_bytes = fh.read(end - start)

    # Convert bfloat16 -> float32
    raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    # bfloat16 = upper 16 bits of float32, pad with zeros
    raw_u32 = raw_u16.astype(np.uint32) << 16
    tensor = np.frombuffer(raw_u32.tobytes(), dtype=np.float32).reshape(shape)
    return tensor

from .format.spec import DEFAULT_BLOCK_SIZE, IMPORTANCE_SUFFIX, ACT_NORMS_SUFFIX
from .architectures import detect_architecture, get_layer_config, get_skip_tensors


def calibrate_from_weights(
    model_path: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    output_path: Optional[str | Path] = None,
) -> dict[str, np.ndarray]:
    """
    Weight-only calibration (no forward pass needed).

    Computes importance scores based on weight magnitude statistics per block.
    This is fast but less accurate than activation-aware calibration.
    Good enough for initial testing and models where forward pass is expensive.

    Importance heuristic:
        score(block) = weight_variance(block) * weight_range(block)

    Blocks with high variance AND high range contain the most information
    and need more bits to represent accurately.

    Args:
        model_path: path to HuggingFace model directory
        block_size: weights per quantization block
        output_path: where to save imatrix (default: model_path/jang_imatrix.safetensors)

    Returns:
        dict of tensor_name.importance -> scores array
    """
    model_path = Path(model_path)
    arch_config = detect_architecture(model_path)
    skip_patterns = get_skip_tensors(arch_config)

    # Load weight tensors from safetensors
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    from safetensors import safe_open

    importance_data = {}
    total_blocks = 0
    total_weights = 0

    print(f"  Calibrating from weights: {model_path.name}")
    print(f"  Architecture: {arch_config.arch_type.value} ({arch_config.attention_type.value})")
    print(f"  Block size: {block_size}")
    print()

    for sf_path in tqdm(weight_files, desc="  Loading weights"):
        with safe_open(str(sf_path), framework="numpy") as f:
            for tensor_name in f.keys():
                # Skip non-weight tensors
                if any(skip in tensor_name for skip in skip_patterns):
                    continue

                # Skip bias, norm weights (tiny tensors)
                if tensor_name.endswith(".bias"):
                    continue
                if "layernorm" in tensor_name.lower() or "rmsnorm" in tensor_name.lower():
                    continue
                # Skip FP8 scale companion tensors (MiniMax, DeepSeek FP8 models)
                if "weight_scale_inv" in tensor_name or "_scale_inv" in tensor_name:
                    continue

                # Get tensor shape without loading
                shape = f.get_slice(tensor_name).get_shape()

                # Skip 1D tensors (norms, biases caught above)
                if len(shape) < 2:
                    continue

                # Skip tiny non-2D tensors that aren't worth quantizing
                # (conv1d [C,1,4], patch_embed [out,3,2,16,16])
                if len(shape) > 2 and math.prod(shape) < 100_000:
                    continue

                # Load tensor — handle bfloat16 and FP8 by reading raw bytes
                try:
                    tensor = f.get_tensor(tensor_name)
                except (TypeError, AttributeError):
                    # Check if FP8 (float8_e4m3fn not supported by numpy)
                    from .fp8 import load_fp8_tensor
                    scale_key = f"{tensor_name}_scale_inv"
                    scale_inv = None
                    try:
                        scale_inv = f.get_tensor(scale_key)
                    except Exception:
                        pass
                    try:
                        tensor = load_fp8_tensor(sf_path, tensor_name, shape, scale_inv)
                    except Exception:
                        # bfloat16 fallback
                        tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

                # For 3D+ tensors (fused MoE experts), reshape to 2D first
                # so block boundaries align with rows, not across experts
                if tensor.ndim >= 3:
                    tensor = tensor.reshape(-1, tensor.shape[-1])

                # Compute per-block importance scores
                flat = tensor.reshape(-1).astype(np.float32)
                n_weights_tensor = len(flat)
                n_blocks = (n_weights_tensor + block_size - 1) // block_size

                # Pad to multiple of block_size
                pad = n_blocks * block_size - n_weights_tensor
                if pad > 0:
                    flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

                blocks = flat.reshape(n_blocks, block_size)

                # Weight-based importance metrics
                block_variance = np.var(blocks, axis=1)   # information content
                block_range = np.ptp(blocks, axis=1)       # dynamic range
                block_magnitude = np.mean(np.abs(blocks), axis=1)  # overall scale

                # Combined score: variance * range * magnitude
                # This captures blocks that are both high-information and high-magnitude
                raw_score = block_variance * block_range * block_magnitude

                # Normalize raw scores to [0, 1] within this tensor,
                # then scale by architecture importance weight so cross-tensor
                # priority is preserved in the greedy allocator
                layer_config = get_layer_config(arch_config, tensor_name)
                max_raw = raw_score.max()
                if max_raw > 0:
                    importance = (raw_score / max_raw) * layer_config.importance_weight
                else:
                    importance = raw_score

                # Store base name (without .weight suffix if present)
                base_name = tensor_name
                if base_name.endswith(".weight"):
                    base_name = base_name[:-7]

                importance_data[f"{base_name}{IMPORTANCE_SUFFIX}"] = importance.astype(np.float32)

                total_blocks += n_blocks
                total_weights += n_weights_tensor

    print(f"\n  Calibrated {len(importance_data)} tensors")
    print(f"  Total blocks: {total_blocks:,}")
    print(f"  Total weights: {total_weights:,}")

    # Save
    if output_path is None:
        output_path = model_path / "jang_imatrix.safetensors"
    output_path = Path(output_path)
    save_file(importance_data, str(output_path))
    print(f"  Saved importance matrix: {output_path}")

    return importance_data


def calibrate_with_activations(
    model_path: str | Path,
    calibration_data: Optional[list[str]] = None,
    n_samples: int = 64,
    seq_len: int = 512,
    block_size: int = DEFAULT_BLOCK_SIZE,
    output_path: Optional[str | Path] = None,
    backend: str = "torch",
) -> dict[str, np.ndarray]:
    """
    Activation-aware calibration (requires forward pass).

    Runs calibration text through the model and records activation magnitudes
    per channel. Combined with weight magnitudes, this gives the AWQ-style
    importance score: importance = ||activations|| * ||weights||

    This is significantly better than weight-only calibration, especially
    at low bit widths (2-3 bit) where every bit allocation decision matters.

    Args:
        model_path: path to HuggingFace model directory
        calibration_data: list of text strings for calibration (default: built-in)
        n_samples: number of calibration samples
        seq_len: max sequence length per sample
        block_size: weights per quantization block
        output_path: where to save imatrix
        backend: "torch" (default)

    Returns:
        dict of tensor_name.importance -> scores array
    """
    if backend == "torch":
        return _calibrate_torch(model_path, calibration_data, n_samples, seq_len, block_size, output_path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch'.")


def _calibrate_torch(
    model_path: str | Path,
    calibration_data: Optional[list[str]],
    n_samples: int,
    seq_len: int,
    block_size: int,
    output_path: Optional[str | Path],
) -> dict[str, np.ndarray]:
    """PyTorch-based calibration with activation hooks."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Torch calibration requires torch and transformers. "
            "Install with: pip install torch transformers"
        )

    model_path = Path(model_path)
    arch_config = detect_architecture(model_path)

    print(f"  Torch calibration: {model_path.name}")
    print(f"  Architecture: {arch_config.arch_type.value}")
    print(f"  Samples: {n_samples}, seq_len: {seq_len}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model.set_mode("inference") if hasattr(model, 'set_mode') else None

    # Prepare calibration data
    if calibration_data is None:
        calibration_data = _default_calibration_texts()
    calibration_data = calibration_data[:n_samples]

    # Collect activation statistics via hooks
    activation_stats = {}

    def make_hook(name):
        def hook(module, input_tensors, output):
            x = input_tensors[0].detach().float()
            sq = (x * x).sum(dim=tuple(range(x.ndim - 1)))  # sum over all dims except last
            sq_np = sq.cpu().numpy()

            if name not in activation_stats:
                activation_stats[name] = {
                    "sum_sq": np.zeros_like(sq_np),
                    "count": 0,
                }
            activation_stats[name]["sum_sq"] += sq_np
            activation_stats[name]["count"] += x.shape[0] * (x.shape[1] if x.ndim > 2 else 1)
        return hook

    hook_handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            h = module.register_forward_hook(make_hook(name))
            hook_handles.append(h)

    # Run forward passes
    print("  Running calibration forward passes...")
    for text in tqdm(calibration_data, desc="  Calibrating"):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        )
        with torch.no_grad():
            model(**tokens)

    # Remove hooks
    for h in hook_handles:
        h.remove()

    # Compute importance scores
    importance_data = {}
    skip_patterns = get_skip_tensors(arch_config)

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(skip in name for skip in skip_patterns):
            continue

        weight = module.weight.detach().float().cpu().numpy()
        if weight.ndim < 2:
            continue

        flat = weight.reshape(-1)
        n_weights = len(flat)
        n_blocks = (n_weights + block_size - 1) // block_size

        pad = n_blocks * block_size - n_weights
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(n_blocks, block_size)

        # Weight magnitude per block
        block_magnitude = np.mean(np.abs(blocks), axis=1)

        # Activation magnitude (if available)
        if name in activation_stats:
            stats = activation_stats[name]
            act_norms = np.sqrt(stats["sum_sq"] / max(stats["count"], 1))

            # AWQ-style importance: activation_norm * weight_magnitude per block
            in_features = weight.shape[1]
            n_blocks_per_row = (in_features + block_size - 1) // block_size

            # Map activation norms to blocks
            block_act_importance = np.zeros(n_blocks, dtype=np.float32)
            for b in range(n_blocks):
                col_start = (b % n_blocks_per_row) * block_size
                col_end = min(col_start + block_size, in_features)
                if col_start < len(act_norms):
                    block_act_importance[b] = float(np.mean(act_norms[col_start:col_end]))

            importance = block_act_importance * block_magnitude
        else:
            # Fall back to weight-only
            block_variance = np.var(blocks, axis=1)
            importance = block_variance * block_magnitude

        # Apply architecture weight
        layer_config = get_layer_config(arch_config, name)
        importance *= layer_config.importance_weight

        # Normalize
        max_imp = importance.max()
        if max_imp > 0:
            importance = importance / max_imp

        base_name = name
        importance_data[f"{base_name}{IMPORTANCE_SUFFIX}"] = importance.astype(np.float32)

        if name in activation_stats:
            stats = activation_stats[name]
            act_norms = np.sqrt(stats["sum_sq"] / max(stats["count"], 1))
            importance_data[f"{base_name}{ACT_NORMS_SUFFIX}"] = act_norms.astype(np.float32)

    # Save
    if output_path is None:
        output_path = model_path / "jang_imatrix.safetensors"
    output_path = Path(output_path)
    save_file(importance_data, str(output_path))
    print(f"\n  Saved importance matrix: {output_path}")
    print(f"  Tensors calibrated: {len([k for k in importance_data if k.endswith(IMPORTANCE_SUFFIX)])}")

    return importance_data


def _default_calibration_texts() -> list[str]:
    """Built-in calibration texts for quick testing."""
    return [
        # General text
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that can capture long-range dependencies in text.",
        "Machine learning models are trained on large datasets to learn patterns and make predictions. The quality of the training data directly affects model performance.",
        # Code
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint([fibonacci(i) for i in range(10)])",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n    return np.dot(A, B)\n\nA = np.random.randn(3, 4)\nB = np.random.randn(4, 5)\nC = matrix_multiply(A, B)",
        # Reasoning
        "Let's solve this step by step. If a train travels at 60 mph for 2 hours, then at 80 mph for 1.5 hours, the total distance is 60*2 + 80*1.5 = 120 + 120 = 240 miles.",
        "To prove that sqrt(2) is irrational, assume the contrary: sqrt(2) = p/q where p, q are integers with no common factors.",
        # Conversation
        "User: What's the best programming language for beginners?\nAssistant: Python is widely recommended for beginners due to its readable syntax and extensive library ecosystem.",
        "User: Explain quantum computing in simple terms.\nAssistant: Quantum computing uses quantum bits (qubits) that can be in multiple states simultaneously, unlike classical bits which are either 0 or 1.",
    ]
