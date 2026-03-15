"""
MXQ Bit Allocation — Assign bit widths to weight blocks based on importance.
Created by Eric Jang (eric@vmlx.net)

Given importance scores and a target average bit width, determines the
optimal number of bits for each weight block.
"""

import numpy as np
import heapq
from typing import Optional

from .format.spec import ALLOWED_BIT_WIDTHS, DEFAULT_BLOCK_SIZE


# Sorted allowed bit widths for upgrade path: 2 → 3 → 4 → 5 → 6 → 8
BIT_UPGRADE_PATH = sorted(ALLOWED_BIT_WIDTHS)


def _next_bit_width(current: int) -> Optional[int]:
    """Get the next higher allowed bit width, or None if at max."""
    idx = BIT_UPGRADE_PATH.index(current)
    if idx + 1 < len(BIT_UPGRADE_PATH):
        return BIT_UPGRADE_PATH[idx + 1]
    return None


def _prev_bit_width(current: int) -> Optional[int]:
    """Get the next lower allowed bit width, or None if at min."""
    idx = BIT_UPGRADE_PATH.index(current)
    if idx > 0:
        return BIT_UPGRADE_PATH[idx - 1]
    return None


# Layer type classification for applying minimum bit floors
LAYER_PRIORS = {
    "embed_tokens": 4,    # vocabulary representation — critical
    "lm_head": 6,         # output logits — directly affects token probs
    "q_proj": 3,          # attention query — sensitive to quantization
    "k_proj": 3,          # attention key — sensitive
    "v_proj": 3,          # attention value — moderate sensitivity
    "o_proj": 3,          # attention output projection
    "gate_proj": 2,       # MLP gate — most compressible
    "up_proj": 2,         # MLP up — most compressible
    "down_proj": 2,       # MLP down — slightly more sensitive
}

# MLXQ Quantization Profiles
#
# Named: MQ{bits}_{size}  where bits = avg bit width, size = S/M/L
#   S = Small  — smallest model, most compression, attention at minimum viable
#   M = Medium — balanced quality/size, attention boosted for coherence
#   L = Large  — best quality at this bit level, attention at high precision
#
# The key insight (experiment 028): attention is ~12% of params but
# controls output quality (prevents repetition loops, maintains coherence).
# MLP is ~88% of params and tolerates aggressive quantization.
#
# Example HuggingFace names:
#   dealignai/Qwen2.5-72B-MQ4M
#   dealignai/Qwen2.5-72B-MQ3M
#   dealignai/Qwen2.5-72B-MQ2M
#
MLXQ_PROFILES = {
    # ── MQ1: 1-bit MLP (extreme) ────────────────────────────────────
    # Most extreme compression possible. Only viable on very large models (70B+).
    # MLP at 2-bit is the actual minimum — "1-bit" means we push everything
    # else as low as possible while keeping attention at maximum precision.
    "MQ1L": {  # ~2.2 avg — Large: MLP=2bit, attention=8bit, embed/lm_head=8bit
        "embed_tokens": 8, "lm_head": 8,
        "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
        "gate_proj": 2, "up_proj": 2, "down_proj": 2,
    },

    # ── MQ2: 2-bit MLP ─────────────────────────────────────────────
    # Maximum compression. 70B fits in 32GB. Needs 7B+ models.
    "MQ2S": {  # ~2.5 avg — Small: tightest, attention at 6-bit
        "embed_tokens": 4, "lm_head": 6,
        "q_proj": 6, "k_proj": 6, "v_proj": 6, "o_proj": 6,
        "gate_proj": 2, "up_proj": 2, "down_proj": 2,
    },
    "MQ2M": {  # ~2.7 avg — Medium: attention at 8-bit for better coherence
        "embed_tokens": 4, "lm_head": 8,
        "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
        "gate_proj": 2, "up_proj": 2, "down_proj": 2,
    },

    # ── MQ3: 3-bit MLP ─────────────────────────────────────────────
    # Sweet spot for quality/compression. Proven to beat uniform 4-bit.
    "MQ3S": {  # ~3.1 avg — Small: attention at 4-bit (same as MLP neighbor)
        "embed_tokens": 4, "lm_head": 6,
        "q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
        "gate_proj": 3, "up_proj": 3, "down_proj": 3,
    },
    "MQ3M": {  # ~3.4 avg — Medium: attention at 6-bit (validated sweet spot)
        "embed_tokens": 4, "lm_head": 6,
        "q_proj": 6, "k_proj": 6, "v_proj": 6, "o_proj": 6,
        "gate_proj": 3, "up_proj": 3, "down_proj": 3,
    },
    "MQ3L": {  # ~3.6 avg — Large: attention at 8-bit for maximum coherence
        "embed_tokens": 4, "lm_head": 8,
        "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
        "gate_proj": 3, "up_proj": 3, "down_proj": 3,
    },

    # ── MQ4: 4-bit MLP ─────────────────────────────────────────────
    # High quality. Beats uniform 4-bit on hard prompts.
    "MQ4S": {  # ~4.1 avg — Small: attention at 5-bit (proven direct win)
        "embed_tokens": 4, "lm_head": 6,
        "q_proj": 5, "k_proj": 5, "v_proj": 5, "o_proj": 5,
        "gate_proj": 4, "up_proj": 4, "down_proj": 4,
    },
    "MQ4M": {  # ~4.2 avg — Medium: attention at 6-bit
        "embed_tokens": 4, "lm_head": 6,
        "q_proj": 6, "k_proj": 6, "v_proj": 6, "o_proj": 6,
        "gate_proj": 4, "up_proj": 4, "down_proj": 4,
    },
    "MQ4L": {  # ~4.5 avg — Large: attention at 8-bit, same size as uniform 4
        "embed_tokens": 4, "lm_head": 8,
        "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
        "gate_proj": 4, "up_proj": 4, "down_proj": 4,
    },

    # ── MQ6: 6-bit MLP ─────────────────────────────────────────────
    # Near-lossless. For when quality matters more than size.
    "MQ6M": {  # ~6.2 avg — near-lossless
        "embed_tokens": 6, "lm_head": 8,
        "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
        "gate_proj": 6, "up_proj": 6, "down_proj": 6,
    },
}

# Backward compat
MXQ_PROFILES = MLXQ_PROFILES


def classify_layer(tensor_name: str) -> tuple[str, Optional[int], int]:
    """
    Classify a tensor name to determine its layer type and position.

    Returns:
        (layer_type, layer_index, min_bits)
    """
    min_bits = 2  # default minimum

    for key, floor in LAYER_PRIORS.items():
        if key in tensor_name:
            min_bits = floor
            break

    # Extract layer index if present (e.g., "layers.5.self_attn.q_proj")
    layer_idx = None
    parts = tensor_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
            break

    return tensor_name, layer_idx, min_bits


def allocate_bits_greedy(
    importance_scores: np.ndarray,
    target_bits: float,
    tensor_names: list[str],
    n_layers: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    first_last_bonus: int = 2,
    first_last_extra_bits: int = 1,
) -> np.ndarray:
    """
    Greedy bit allocation: start at minimum, upgrade most important blocks.

    Algorithm:
        1. Set all blocks to their minimum bit width (based on layer type)
        2. Apply first/last layer bonus
        3. Sort blocks by importance (descending)
        4. Upgrade the most important under-allocated block until target is reached

    Args:
        importance_scores: float32 array of importance per block
        target_bits: target average bits per weight (e.g., 2.5)
        tensor_names: name of the tensor each block belongs to
        n_layers: total number of transformer layers in the model
        block_size: weights per block
        first_last_bonus: number of first/last layers to protect
        first_last_extra_bits: bonus bits for first/last layers

    Returns:
        uint8 array of bit widths per block
    """
    n_blocks = len(importance_scores)
    bits = np.full(n_blocks, 2, dtype=np.int32)

    # Apply layer-type minimum floors
    block_tensor_map = []
    for i, name in enumerate(tensor_names):
        _, layer_idx, min_bits = classify_layer(name)
        bits[i] = max(bits[i], min_bits)
        block_tensor_map.append((name, layer_idx))

    # Apply first/last layer bonus
    for i, (name, layer_idx) in enumerate(block_tensor_map):
        if layer_idx is not None:
            if layer_idx < first_last_bonus or layer_idx >= n_layers - first_last_bonus:
                new_bits = bits[i] + first_last_extra_bits
                # Snap to nearest allowed
                for b in BIT_UPGRADE_PATH:
                    if b >= new_bits:
                        bits[i] = b
                        break
                else:
                    bits[i] = BIT_UPGRADE_PATH[-1]

    # Calculate remaining budget
    current_avg = float(np.mean(bits))
    if current_avg >= target_bits:
        # Already at or above target from floors alone
        return bits.astype(np.uint8)

    # Use a max-heap (negate importance for min-heap behavior) to upgrade
    # most important blocks first
    # Heap entries: (-importance, block_index)
    heap = [(-float(importance_scores[i]), i) for i in range(n_blocks)]
    heapq.heapify(heap)

    total_bits = int(np.sum(bits))
    target_total = int(target_bits * n_blocks)

    while total_bits < target_total and heap:
        neg_imp, idx = heapq.heappop(heap)

        current = bits[idx]
        next_bw = _next_bit_width(current)
        if next_bw is None:
            continue  # already at max

        cost = next_bw - current
        if total_bits + cost <= target_total + n_blocks * 0.01:
            # Allow slight overshoot for rounding
            bits[idx] = next_bw
            total_bits += cost

            # Re-add to heap if can still upgrade
            if _next_bit_width(next_bw) is not None:
                heapq.heappush(heap, (neg_imp, idx))

    return bits.astype(np.uint8)


def allocate_bits_dp(
    importance_scores: np.ndarray,
    weight_variances: np.ndarray,
    target_bits: float,
    tensor_names: list[str],
    n_layers: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """
    Dynamic programming bit allocation for optimal distortion minimization.

    Minimizes: Σ importance[i] * distortion(bits[i])
    Subject to: mean(bits) = target_bits

    Where distortion(b) ≈ variance * (range / 2^b)² / 12

    This is a bounded knapsack problem, solved via DP.

    For very large models (millions of blocks), falls back to greedy.

    Args:
        importance_scores: float32 array of importance per block
        weight_variances: float32 array of weight variance per block
        target_bits: target average bits
        tensor_names: tensor name per block
        n_layers: total transformer layers
        block_size: weights per block

    Returns:
        uint8 array of bit widths per block
    """
    n_blocks = len(importance_scores)

    # For large models, DP is too expensive — fall back to greedy
    if n_blocks > 50000:
        return allocate_bits_greedy(
            importance_scores, target_bits, tensor_names, n_layers, block_size
        )

    # Precompute distortion at each bit width for each block
    # D(b) ≈ importance * variance * (1/2^b)² ∝ importance * variance * 4^(-b)
    distortion = np.zeros((n_blocks, len(BIT_UPGRADE_PATH)), dtype=np.float64)
    for j, b in enumerate(BIT_UPGRADE_PATH):
        distortion[:, j] = importance_scores * weight_variances * (4.0 ** (-b))

    # Get minimum bits per block from layer priors
    min_bits = np.full(n_blocks, 2, dtype=np.int32)
    for i, name in enumerate(tensor_names):
        _, layer_idx, mb = classify_layer(name)
        min_bits[i] = mb

    # DP over blocks
    target_total = int(round(target_bits * n_blocks))

    # Greedy with distortion-guided priority
    bits = min_bits.copy()
    total = int(np.sum(bits))

    if total >= target_total:
        return bits.astype(np.uint8)

    # Priority: blocks where upgrading gives the most distortion reduction per bit
    # marginal_gain[i] = (D(current) - D(current+1)) / cost
    while total < target_total:
        best_idx = -1
        best_gain = -1.0

        for i in range(n_blocks):
            current = bits[i]
            next_bw = _next_bit_width(current)
            if next_bw is None:
                continue

            curr_j = BIT_UPGRADE_PATH.index(current)
            next_j = BIT_UPGRADE_PATH.index(next_bw)
            gain = (distortion[i, curr_j] - distortion[i, next_j]) / (next_bw - current)

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1:
            break  # all blocks at max

        old = bits[best_idx]
        new = _next_bit_width(old)
        bits[best_idx] = new
        total += new - old

    return bits.astype(np.uint8)


def allocate_bits_profile(
    tensor_names: list[str],
    profile: str = "mxq-3",
) -> np.ndarray:
    """
    Profile-based bit allocation — assigns bits by layer type.

    This is the proven MXQ strategy (experiment 028): give attention
    layers more bits (6-8) and MLP layers fewer bits (2-3). The quality
    improvement comes from the attention/MLP sensitivity asymmetry.

    Args:
        tensor_names: tensor name for each block
        profile: e.g. "MQ3M", "MQ4S", "MQ2M" (case-insensitive)

    Returns:
        uint8 array of bit widths per block
    """
    profile = profile.upper()
    if profile not in MLXQ_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(MLXQ_PROFILES.keys())}")

    layer_bits = MLXQ_PROFILES[profile]
    n_blocks = len(tensor_names)
    bits = np.full(n_blocks, 4, dtype=np.int32)  # default 4-bit

    for i, name in enumerate(tensor_names):
        for pattern, preferred in layer_bits.items():
            if pattern in name:
                bits[i] = preferred
                break

    return bits.astype(np.uint8)


def summarize_allocation(
    bit_map: np.ndarray,
    tensor_names: Optional[list[str]] = None,
) -> dict:
    """
    Generate a summary of the bit allocation.

    Returns dict with:
        - average_bits: actual average
        - histogram: count of blocks at each bit width
        - per_layer_avg: average bits per layer type (if tensor_names given)
    """
    result = {
        "average_bits": float(np.mean(bit_map)),
        "total_blocks": len(bit_map),
        "histogram": {},
    }

    for bw in sorted(ALLOWED_BIT_WIDTHS):
        count = int(np.sum(bit_map == bw))
        if count > 0:
            pct = round(100 * count / len(bit_map), 1)
            result["histogram"][f"{bw}-bit"] = {"count": count, "percent": pct}

    if tensor_names:
        layer_bits = {}
        for i, name in enumerate(tensor_names):
            # Group by layer type
            for key in LAYER_PRIORS:
                if key in name:
                    if key not in layer_bits:
                        layer_bits[key] = []
                    layer_bits[key].append(int(bit_map[i]))
                    break

        result["per_layer_type"] = {
            k: round(float(np.mean(v)), 2) for k, v in sorted(layer_bits.items())
        }

    return result
