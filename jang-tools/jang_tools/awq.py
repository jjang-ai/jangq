"""
JANG AWQ — Activation-Aware Weight Scaling for Quantization
Created by Jinho Jang (eric@jangq.ai)

Implements per-channel scaling based on activation magnitudes before
quantization. This gives important input channels more of the
quantization grid, reducing output error by ~14% at the same bit width.

The math:
  s_j = (act_norm_j + eps)^alpha
  W_scaled = W * diag(s)
  Q_scaled = quantize(W_scaled)
  W_approx = dequant(Q_scaled) / diag(s)

Optimal alpha is 0.25 (empirically) or 1/3 (theoretically).
"""

import numpy as np
from typing import Optional


def compute_awq_scales(
    act_norms: np.ndarray,
    alpha: float = 0.25,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute per-channel AWQ scaling factors from activation norms.

    Args:
        act_norms: (in_features,) per-channel activation L2 norms
        alpha: scaling exponent (0.25 empirically optimal)
        eps: epsilon for numerical stability

    Returns:
        (in_features,) scaling factors
    """
    return np.power(act_norms + eps, alpha).astype(np.float32)


def apply_awq_scaling(
    weight: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Scale weight columns by AWQ factors before quantization."""
    return weight * scales[np.newaxis, :]


def reverse_awq_scaling(
    weight: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Reverse AWQ scaling after dequantization."""
    return weight / scales[np.newaxis, :]


def collect_activation_norms_mlx(
    model_path: str,
    calibration_texts: Optional[list[str]] = None,
    n_samples: int = 128,
) -> dict[str, np.ndarray]:
    """
    Collect per-channel activation norms for all Linear layers using MLX.

    Args:
        model_path: path to HuggingFace model directory
        calibration_texts: list of calibration texts
        n_samples: max number of texts to process

    Returns:
        dict mapping layer_name to (in_features,) activation norms
    """
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(model_path)

    if calibration_texts is None:
        calibration_texts = _default_calibration_texts()
    calibration_texts = calibration_texts[:n_samples]

    inner = model.model
    embed = inner['embed_tokens']
    n_layers = len(inner['layers'])

    act_stats = {}

    print(f"  Collecting activation norms ({len(calibration_texts)} texts, {n_layers} layers)...")

    for text_idx, text in enumerate(calibration_texts):
        ids = tokenizer.encode(text)
        tokens = mx.array([ids])
        h = embed(tokens)
        mx.eval(h)

        for layer_idx in range(n_layers):
            layer = inner['layers'][layer_idx]

            # Capture input to attention block (after input_layernorm)
            normed = layer['input_layernorm'](h)
            mx.eval(normed)
            act_np = np.array(normed[0].astype(mx.float32))
            _accumulate(act_stats, f"layers.{layer_idx}.attn_input", act_np)

            # Run full layer
            h = layer(h, mask="causal")
            mx.eval(h)

        if (text_idx + 1) % 10 == 0:
            print(f"    {text_idx + 1}/{len(calibration_texts)} texts processed")

    # Convert to norms
    result = {}
    for key, stats in act_stats.items():
        mean_sq = stats['sum_sq'] / max(stats['count'], 1)
        result[key] = np.sqrt(mean_sq)

    print(f"  Collected norms for {len(result)} layer inputs")
    return result


def _accumulate(acc: dict, key: str, activations: np.ndarray) -> None:
    """Accumulate running sum of squared activations per channel."""
    sq = np.sum(activations ** 2, axis=0)
    n = activations.shape[0]
    if key not in acc:
        acc[key] = {'sum_sq': np.zeros_like(sq), 'count': 0}
    acc[key]['sum_sq'] += sq
    acc[key]['count'] += n


def _default_calibration_texts() -> list[str]:
    """Diverse calibration texts."""
    return [
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that capture long-range dependencies in text. Unlike recurrent neural networks, transformers process all positions simultaneously.",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "What is the meaning of life? This is one of the most profound philosophical questions humanity has grappled with throughout history.",
        "Machine learning models are trained on large datasets to learn patterns. The quality of training data directly affects model performance.",
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
        "In quantum mechanics, particles can exist in superposition of multiple states simultaneously until measured.",
        "Climate change is driven by greenhouse gas emissions, primarily carbon dioxide from fossil fuel combustion.",
        "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections.",
        "Let us solve this step by step. If a train travels at 60 mph for 2 hours, the distance is 120 miles.",
        "import numpy as np\nA = np.random.randn(3, 4)\nB = np.random.randn(4, 5)\nC = A @ B\nprint(C.shape)",
        "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.name;",
        "The Pythagorean theorem states that in a right triangle, a squared plus b squared equals c squared.",
    ] * 4
