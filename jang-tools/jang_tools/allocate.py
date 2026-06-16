"""
JANG Bit Allocation — Assign bit widths to weight blocks based on importance.
Created by Jinho Jang (eric@jangq.ai)

Architecture-aware tier-based allocation that works across all model families:
dense transformers, MoE, hybrid SSM, latent attention (MLA), VL, Mamba, etc.

The key insight: classify every tensor into a sensitivity tier, then profiles
just assign bits per tier. Works regardless of tensor naming conventions.
"""

import numpy as np
import heapq
from enum import IntEnum
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


# ============================================================
# Tier-Based Architecture-Aware Classification
# ============================================================
#
# Every weight tensor is classified into one of three sensitivity tiers:
#
#   CRITICAL  — Directly controls output quality. Quantization errors here
#               cause repetition loops, incoherence, NaN overflow, or total failure.
#               MUST be 4+ bits. At lower bits, the SiLU gate multiplication
#               amplifies quantization errors quadratically → float16 overflow.
#
#               Examples: full softmax attention projections, output heads,
#               MLA latent projections, SSM state matrices, MoE routers,
#               shared experts (always-active MLP — errors compound every layer).
#
#   IMPORTANT — Moderate sensitivity. Errors degrade but don't destroy quality.
#               Examples: embeddings, vision-language connectors,
#               linear attention projections (GatedDeltaNet), SSM timestep.
#
#   COMPRESS  — Most robust to quantization. Can go to 2-bit with minimal
#               quality loss on MoE models (expert redundancy absorbs errors).
#               WARNING: 2-bit on dense model MLP causes quality collapse.
#               WARNING: 2-bit on 512+ expert models may cause NaN (proven on 397B).
#
#               Examples: MLP/FFN layers (dense or expert),
#               vision FFN, generic SSM input/output projections.
#
# ── Precision Floor Rules (proven empirically) ──────────────
#
#   Component              | Min bits | Why
#   -----------------------|----------|--------------------------------------------
#   Shared expert (always  | 4-bit    | SiLU(gate)*up amplifies errors quadratically.
#     active MLP)          |          | 3-bit → 45x output error → float16 inf (397B).
#   MoE router/gate        | 8-bit    | Controls expert routing. Lower → garbage.
#   Attention Q/K/V/O      | 4-bit    | Controls coherence. Lower → repetition loops.
#   Embeddings             | 4-bit    | First layer, errors propagate everywhere.
#   Routed experts (MoE)   | 2-bit*   | Redundancy across K-of-N absorbs errors.
#                          |          | *BUT 2-bit fails on 512+ experts (397B NaN).
#   Dense MLP              | 3-bit    | No redundancy. 2-bit → quality cliff.
#   Linear attention (GDN) | 3-bit    | Always active but lower sensitivity.
#
# ── MLP Asymmetry (256+ expert models) ────────────────────
# Threshold: _MLP_ASYMMETRY_MIN_EXPERTS = 256 (see `_apply_mlp_asymmetry_floor`).
# Was 512 pre-2026-04-08 — lowered after GLM-5.1 (256 experts) degenerated
# into repetition loops at 2-bit MLPs. See `project_mlp_asymmetry.md`.
#
#   Expert MLP tensors are NOT equal in sensitivity:
#
#   gate_proj → SiLU amplifier. Errors get squared through SiLU(gate)*up.
#              At 3-bit on hidden=4096: produces 45x error → float16 overflow.
#              MUST be 4-bit on 256+ expert models.
#
#   up_proj   → Linear multiplicand. Errors are bounded, not amplified.
#              2-bit is safe IF gate_proj has sufficient precision.
#
#   down_proj → Projects back to residual stream. Every subsequent layer
#              sees this output. GGUF always gives +1 bit (Q3_K in Q2_K).
#              3-bit minimum on 256+ expert models.
#
#   Budget-neutral: (gate=4 + up=2 + down=3) / 3 = 3.0 average.
#   Same size as uniform 3-bit, but prevents float16 overflow.
#
#   (gate/down asymmetry empirically required on 512-expert models).
#
#
#

class Tier(IntEnum):
    CRITICAL = 3    # Highest precision
    IMPORTANT = 2   # Moderate precision
    COMPRESS = 1    # Most compressible


# Tier classification rules. Order matters — first match wins.
# Each entry: (substring_pattern, tier)
# More specific patterns MUST come before less specific ones.
TIER_RULES = [
    # ── Norms (skip, but classify as CRITICAL if encountered) ──
    ("layernorm", Tier.CRITICAL),
    ("rmsnorm", Tier.CRITICAL),

    # ── Output head ──────────────────────────────────────────
    ("lm_head", Tier.CRITICAL),

    # ── MLA / Latent Attention (DeepSeek-V2/V3) ─────────────
    # Latent projections compress/decompress the KV cache
    ("kv_a_proj_with_mqa", Tier.CRITICAL),
    ("kv_b_proj", Tier.CRITICAL),
    ("q_a_proj", Tier.CRITICAL),
    ("q_b_proj", Tier.CRITICAL),

    # ── DSA (DeepSeek Sparse Attention) Indexer (GLM-5/V3.2) ─
    # The indexer produces top-k token indices for sparse attention.
    # If quantized too aggressively, it selects WRONG tokens to attend to,
    # causing attention to drift and model to degenerate during generation.
    # Must stay CRITICAL — small tensors, high sensitivity.
    ("indexer.wk", Tier.CRITICAL),
    ("indexer.wq_b", Tier.CRITICAL),
    ("indexer.weights_proj", Tier.CRITICAL),
    ("indexers_proj", Tier.CRITICAL),

    # ── Full Softmax Attention ───────────────────────────────
    # Standard Q/K/V/O projections — critical for coherence
    ("q_proj", Tier.CRITICAL),
    ("k_proj", Tier.CRITICAL),
    ("v_proj", Tier.CRITICAL),
    ("o_proj", Tier.CRITICAL),
    ("self_attn.out_proj", Tier.CRITICAL),
    ("self_attn.g_proj", Tier.CRITICAL),

    # ── SSM State Matrices (Mamba) ───────────────────────────
    ("a_log", Tier.CRITICAL),
    # Mamba D is handled as special case in classify_tensor() — bare ".D" leaf
    ("dt_proj", Tier.IMPORTANT),

    # ── MoE Router / Expert Gate ─────────────────────────────
    # CRITICAL: Controls which experts fire — routing errors cascade.
    # MiniMax specifically requires router at 8-bit or output degrades to garbage.
    # Small tensor but maximum sensitivity.
    ("shared_expert_gate", Tier.CRITICAL),
    # Shared experts are ALWAYS active (process every token). Quantization errors
    # compound through every layer. The SiLU gate multiplication amplifies errors
    # quadratically: gate_error * up_error. At 3-bit on hidden_size>=4096, this
    # causes float16 overflow (proven on 397B). Must be CRITICAL (4+ bit minimum).
    ("shared_expert", Tier.CRITICAL),
    # "gate" alone is tricky — could be MoE router OR MLP gate_proj
    # gate_proj will match gate_proj rule below (COMPRESS) before reaching here

    # ── Embeddings ───────────────────────────────────────────
    ("embed_tokens", Tier.IMPORTANT),
    ("wte", Tier.IMPORTANT),
    ("word_embeddings", Tier.IMPORTANT),

    # ── Vision-Language Connector ────────────────────────────
    ("visual.merger", Tier.IMPORTANT),
    ("multi_modal_projector", Tier.IMPORTANT),
    ("patch_embed", Tier.IMPORTANT),
    ("pos_embed", Tier.IMPORTANT),

    # ── Latent MoE Projections (Nemotron-H) ──────────────────
    # fc1_latent_proj compresses hidden→latent, fc2 decompresses latent→hidden.
    # ALL expert computation flows through this bottleneck. At 2-bit the 1024-dim
    # latent loses too much — must be CRITICAL.
    ("latent_proj", Tier.CRITICAL),

    # ── SSM Projections (Mamba / Nemotron-H) ─────────────────
    # Mamba mixer.in_proj/out_proj are always-active SSM projections.
    # Must come BEFORE generic "proj" catch-all.
    ("mixer.in_proj", Tier.IMPORTANT),
    ("mixer.out_proj", Tier.IMPORTANT),
    ("conv.in_proj", Tier.IMPORTANT),
    ("conv.out_proj", Tier.IMPORTANT),
    ("conv.conv", Tier.CRITICAL),
    ("x_proj", Tier.IMPORTANT),
    ("conv1d", Tier.COMPRESS),

    # ── MoE Router (Gemma 4 uses "router.proj" instead of "gate") ────
    ("router.proj", Tier.CRITICAL),
    ("router", Tier.CRITICAL),

    # ── MoE Expert MLP (fused) ───────────────────────────────
    # Must come before generic gate/up/down to catch fused variants
    ("gate_up_proj", Tier.COMPRESS),

    # ── MLP / FFN (dense or expert) ──────────────────────────
    ("gate_proj", Tier.COMPRESS),
    ("up_proj", Tier.COMPRESS),
    ("down_proj", Tier.COMPRESS),
    ("mlp.fc1", Tier.COMPRESS),
    ("mlp.fc2", Tier.COMPRESS),
    ("w1", Tier.COMPRESS),
    ("w2", Tier.COMPRESS),
    ("w3", Tier.COMPRESS),
    ("wi_0", Tier.COMPRESS),
    ("wi_1", Tier.COMPRESS),
    ("wo", Tier.COMPRESS),

    # ── Linear Attention (RWKV / GatedDeltaNet / Qwen3.5) ───
    # GatedDeltaNet projections are more sensitive than MLP but less than
    # full softmax attention. 2-bit causes repetition loops on 35B.
    ("in_proj_qkv", Tier.IMPORTANT),
    ("in_proj_z", Tier.IMPORTANT),
    ("in_proj_a", Tier.IMPORTANT),
    ("in_proj_b", Tier.IMPORTANT),
    ("linear_attn.out_proj", Tier.IMPORTANT),
    ("delta_net", Tier.IMPORTANT),

    # ── Vision FFN ───────────────────────────────────────────
    ("linear_fc1", Tier.COMPRESS),
    ("linear_fc2", Tier.COMPRESS),

    # ── Generic attention output (must come after specific patterns) ──
    # Catches: generic out_proj variants and vision proj
    ("out_proj", Tier.COMPRESS),

    # ── Vision attention (fused QKV) ─────────────────────────
    ("qkv", Tier.COMPRESS),

    # ── Catch-all projections ────────────────────────────────
    # Generic "proj" — after all specific proj patterns
    ("proj", Tier.COMPRESS),
    ("fc", Tier.COMPRESS),
]

# MoE router needs special handling since "gate" is a substring of "gate_proj"
# We handle it in classify_tensor() directly


def classify_tensor(tensor_name: str, num_experts: int = 0, has_shared_mlp: bool = False) -> Tier:
    """
    Classify a tensor into a sensitivity tier based on its name.

    Uses substring matching with priority ordering. First match wins.
    Handles edge cases like "gate" (MoE router) vs "gate_proj" (MLP).

    Note: For 256+ expert models, MLP asymmetry bit floors are enforced
    in the allocator functions (allocate_bits_profile, allocate_bits_budget),
    not here. This function only handles tier classification.

    Args:
        tensor_name: full tensor name (e.g., "model.layers.5.self_attn.q_proj.weight")
        num_experts: number of MoE experts (0 for dense, passed for future use)
        has_shared_mlp: True if model has parallel dense MLP alongside MoE experts
            (Gemma 4 pattern). When True, mlp.* tensors that are NOT inside
            experts.* are classified as CRITICAL (shared expert equivalent).

    Returns:
        Tier enum value
    """
    name_lower = tensor_name.lower()

    # Gemma 4 shared MLP / GLM-5 first_k_dense_replace:
    # For MoE models, any mlp.{gate,up,down}_proj that is NOT inside experts.*
    # is a dense MLP that is ALWAYS active (either shared MLP like Gemma 4,
    # or first_k_dense_replace layers like GLM-5/DeepSeek V3 where the first
    # N layers use dense MLP instead of MoE). These always-active layers process
    # every token and quantization errors compound through every subsequent layer.
    # Must be CRITICAL, not COMPRESS.
    # Trigger: num_experts > 0 (MoE model) OR has_shared_mlp (Gemma 4 pattern).
    is_moe_model_dense_mlp = (num_experts > 0 or has_shared_mlp) and "experts" not in name_lower
    if is_moe_model_dense_mlp:
        for mlp_pat in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                        "mlp.fc1", "mlp.fc2"):
            if mlp_pat in name_lower:
                return Tier.CRITICAL
        for lfm_dense_pat in (".feed_forward.w1.", ".feed_forward.w2.", ".feed_forward.w3."):
            if lfm_dense_pat in name_lower:
                return Tier.CRITICAL

    for pattern, tier in TIER_RULES:
        if pattern in name_lower:
            return tier

    # Special case: Mamba D parameter — a bare ".D" leaf (e.g., "mamba.D")
    # Can't use TIER_RULES because "d" matches "down_proj", "dt_proj", etc.
    if name_lower.endswith(".d"):
        return Tier.CRITICAL

    # Special case: MoE router — "gate" or "router" that isn't "gate_proj"/"gate_up_proj"
    # Catches: "mlp.gate.weight" (Mixtral/DeepSeek), "feed_forward.router.weight" (Jamba)
    # NOTE: Do NOT add bare "gate" to TIER_RULES — it would match gate_proj too
    if ".gate." in name_lower or name_lower.endswith(".gate"):
        return Tier.CRITICAL  # MoE router — must stay high precision (MiniMax needs 8-bit)
    if ".router." in name_lower or name_lower.endswith(".router"):
        return Tier.CRITICAL  # MoE router

    # Default: COMPRESS (assume it's some form of MLP/projection we don't recognize)
    return Tier.COMPRESS


def _is_routed_expert_mlp(name_lower: str, component: str) -> bool:
    """Check if a tensor is a routed expert MLP component (not shared_expert)."""
    return component in name_lower and "shared_expert" not in name_lower


# ── MLP asymmetry bit floors for 256+ expert models ──────────
# Applied as post-classification floors in allocator functions.
# gate_proj: SiLU amplifier → 4-bit minimum (prevents 45x error on hidden=4096)
# down_proj: residual projection → 3-bit minimum (matches GGUF Q2_K behavior)
# up_proj: linear multiplicand → no floor (2-bit OK when gate is protected)
# Budget impact: depends on profile. For 2-bit profiles, (4+2+3)/3 = 3.0 avg.
MLP_ASYMMETRY_FLOORS = {
    "gate_proj": 4,
    "gate_up_proj": 4,  # Fused variant, conservative (contains gate)
    "w1": 4,            # Mixtral naming for gate_proj
    "down_proj": 3,
    "w2": 3,            # Mixtral naming for down_proj
}


_MLP_ASYMMETRY_MIN_EXPERTS = 256


def _apply_mlp_asymmetry_floor(
    name: str,
    bits: int,
    num_experts: int,
    base_bits: int = 0,
) -> int:
    """
    Apply MLP asymmetry bit floors for 256+ expert models.

    Returns the adjusted bit width (may be higher than input if floor applies).
    Only affects routed expert MLP tensors, not shared_expert.

    Iter-21 memory-cross-ref fix: threshold was CODE 512 but docstring 256 —
    drift from the 2026-04-08 lowering recorded in `project_mlp_asymmetry.md`.
    GLM-5.1 (256 experts) + MiniMax M2.7 (256 experts) + Qwen3.6 (256 routed)
    all need the floor to avoid repetition loops at 2-bit MLPs. Threshold
    constant is now an exported sentinel so a test can pin it.

    K-quant compensation guard (2026-05-04): when ``base_bits >= 4`` (i.e. the
    profile's namesake bits is 4 or more), routed expert ``up_proj`` / ``w3``
    are not allowed to drop below ``base_bits`` during budget compensation.
    Without this guard, JANG_4K on 256-expert MoE pushes a fraction of routed
    experts to 3-bit; rarely-activated experts at 3-bit produce repetition
    loops on trivial prompts. The guard makes JANG_4K behave like JANG_4M on
    MoE (slight ~2 % overshoot vs strict budget) while keeping the existing
    K-quant behaviour for dense models and 2-/3-bit profiles where dropping
    routed ``up_proj`` to a smaller width is intentional.
    """
    if num_experts < _MLP_ASYMMETRY_MIN_EXPERTS:
        return bits
    name_lower = name.lower()
    if "shared_expert" in name_lower:
        return bits  # shared_expert is already CRITICAL, don't touch
    if base_bits >= 4:
        # K-quant compensation guard. Mirrors JANG_4M behaviour on 256+ MoE:
        # routed expert MLP tensors (gate / up / down) stay at the namesake
        # bit width instead of being downgraded to fund the CRITICAL boost.
        for component in (
            "gate_proj", "gate_up_proj", "w1",
            "up_proj", "w3",
            "down_proj", "w2",
        ):
            if _is_routed_expert_mlp(name_lower, component):
                return max(bits, base_bits)
    for component, floor in MLP_ASYMMETRY_FLOORS.items():
        if component in name_lower:
            return max(bits, floor)
    return bits


# ============================================================
# JANG Profiles — Tier-Based
# ============================================================
#
# Each profile defines bits for (CRITICAL, IMPORTANT, COMPRESS) tiers.
# The profile number (2/3/4/6) is the COMPRESS tier bit width.
# S/M/L controls how much extra precision CRITICAL gets above COMPRESS.
# Designed by Jinho Jang — the GGUF equivalent for MLX.
#
# This works for ANY architecture because it doesn't depend on
# tensor naming conventions — just sensitivity classification.

JANG_PROFILES = {
    # ── 2-bit COMPRESS tier ──────────────────────────────────
    # At 2-bit, attention MUST be protected or model breaks completely.
    # IMPORTANT tier also needs protection (linear attention, embeddings).
    "JANG_2S": (6, 4, 2),   # Tightest 2-bit
    "JANG_2M": (8, 4, 2),   # Balanced 2-bit
    "JANG_2L": (8, 6, 2),   # Best quality 2-bit (proven: 73% MMLU on 122B)
    "JANG_1L": (8, 8, 2),   # Maximum quality 2-bit (proven: 6/6 free-form)

    # ── 3-bit COMPRESS tier ──────────────────────────────────
    # At 3-bit+, only CRITICAL (full attention) needs protection.
    # IMPORTANT stays at COMPRESS level — minimal overhead, max quality.
    "JANG_3S": (6, 3, 3),   # Small boost on attention only
    "JANG_3M": (8, 3, 3),   # Full attention at 8-bit, everything else 3-bit
    "JANG_3L": (8, 4, 3),   # Attention 8-bit, embeddings 4-bit

    # ── 4-bit COMPRESS tier ──────────────────────────────────
    # Standard quality. CRITICAL at 8-bit gives attention 2x precision
    # with only ~2% overhead on MoE models.
    "JANG_4S": (6, 4, 4),   # Small boost
    "JANG_4M": (8, 4, 4),   # Full attention at 8-bit, rest at 4-bit (~2% overhead on MoE)
    "JANG_4L": (8, 6, 4),   # Also boost embeddings (for dense models)

    # ── 6-bit COMPRESS tier ──────────────────────────────────
    "JANG_6M": (8, 6, 6),   # Near-lossless
}


# Simple bit-to-profile mapping. User picks a number, gets the best approach.
# At 3-bit+: K-quant style (budget-neutral, same size as MLX, smarter allocation)
# At 2-bit: profile-based (JANG_2S is proven sweet spot)
# At 1-bit: max protection profile
BIT_TO_PROFILE = {
    1: "JANG_1L",   # Extreme 2-bit + max protection
    2: "JANG_2S",   # 2-bit + 6-bit attention (proven: 84% MMLU on 122B, +28 over MLX)
    3: "JANG_3K",   # K-quant 3-bit (budget-neutral)
    4: "JANG_4K",   # K-quant 4-bit (budget-neutral) — THE DEFAULT
    5: "JANG_5K",   # K-quant 5-bit (budget-neutral)
    6: "JANG_6K",   # K-quant 6-bit (budget-neutral)
    7: "JANG_6M",   # Near-lossless
    8: "JANG_6M",   # Near-lossless
}

# K-quant targets: budget-neutral allocation at these bit levels
# Uses allocate_bits_budget() instead of allocate_bits_profile()
JANG_K_TARGETS = {"JANG_3K": 3.0, "JANG_4K": 4.0, "JANG_5K": 5.0, "JANG_6K": 6.0}


def profile_for_bits(target_bits: int) -> str:
    """Map a target bit width (1-8) to the recommended JANG profile."""
    target_bits = max(1, min(8, int(target_bits)))
    return BIT_TO_PROFILE[target_bits]


def is_k_quant(profile: str) -> bool:
    """Check if a profile is a K-quant (budget-neutral) profile."""
    return profile.upper() in JANG_K_TARGETS


def k_quant_target(profile: str) -> float:
    """Get the target bits for a K-quant profile."""
    return JANG_K_TARGETS[profile.upper()]


def estimate_size_gb(total_params: int, profile: str) -> dict:
    """
    Estimate JANG model size for a given parameter count and profile.

    Args:
        total_params: total model parameters
        profile: JANG profile name (e.g. "JANG_2L") or K-quant (e.g. "JANG_4K")

    Returns:
        dict with weight_gb, overhead_gb, total_gb, and profile details
    """
    # Handle K-quant profiles
    if is_k_quant(profile):
        target = k_quant_target(profile)
        weight_bytes = int(total_params * target / 8)
        n_blocks = (total_params + 63) // 64
        overhead_bytes = n_blocks * 4
        return {
            "profile": profile,
            "type": "k-quant",
            "target_bits": target,
            "avg_bits_approx": target,
            "weight_gb": round(weight_bytes / (1024 ** 3), 1),
            "overhead_gb": round(overhead_bytes / (1024 ** 3), 1),
            "total_gb": round((weight_bytes + overhead_bytes) / (1024 ** 3), 1),
        }

    if profile not in JANG_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")

    c, i, comp = JANG_PROFILES[profile]

    # For size estimation, use COMPRESS bits as the dominant factor
    # (94-99% of params in MoE models are COMPRESS tier)
    # This gives a conservative lower bound
    avg_bits_approx = comp + 0.1  # slight overhead from CRITICAL/IMPORTANT tiers

    weight_bytes = int(total_params * avg_bits_approx / 8)
    n_blocks = (total_params + 63) // 64
    overhead_bytes = n_blocks * 4  # scales + zeros (float16 each)

    total_bytes = weight_bytes + overhead_bytes

    return {
        "profile": profile,
        "critical_bits": c,
        "important_bits": i,
        "compress_bits": comp,
        "avg_bits_approx": round(avg_bits_approx, 1),
        "weight_gb": round(weight_bytes / (1024 ** 3), 1),
        "overhead_gb": round(overhead_bytes / (1024 ** 3), 1),
        "total_gb": round(total_bytes / (1024 ** 3), 1),
    }


# Legacy: keep layer priors for greedy allocator
LAYER_PRIORS = {
    "embed_tokens": 4,
    "lm_head": 6,
    "q_proj": 3,
    "k_proj": 3,
    "v_proj": 3,
    "o_proj": 3,
    "gate_proj": 2,
    "up_proj": 2,
    "down_proj": 2,
}


def classify_layer(tensor_name: str) -> tuple[str, Optional[int], int]:
    """
    Classify a tensor name to determine its layer type and position.
    Used by greedy allocator for minimum bit floors.
    """
    min_bits = 2

    for key, floor in LAYER_PRIORS.items():
        if key in tensor_name:
            min_bits = floor
            break

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


def allocate_bits_budget(
    tensor_names: list[str],
    target_bits: float = 4.0,
    num_experts: int = 0,
    has_shared_mlp: bool = False,
) -> np.ndarray:
    """
    Budget-neutral bit allocation — same total bits as uniform, smarter distribution.

    Gives CRITICAL tensors more bits and compensates by giving the least-important
    COMPRESS tensors fewer bits. Total average = target_bits exactly.

    Like GGUF K-quants: same size as uniform, better quality.

    For 256+ expert models, classify_tensor applies MLP asymmetry
    (gate_proj → IMPORTANT) and down_proj gets a 3-bit floor.

    Args:
        tensor_names: tensor name per block (repeated for each block in tensor)
        target_bits: target average bits (e.g., 4.0 to match MLX 4-bit size)
        num_experts: number of MoE experts (0 for dense models)

    Returns:
        uint8 array of bit widths per block
    """
    # Step 1: Classify each unique tensor
    unique_tensors = {}  # name → (tier, n_params, block_indices)
    for i, name in enumerate(tensor_names):
        if name not in unique_tensors:
            unique_tensors[name] = {
                "tier": classify_tensor(name, num_experts, has_shared_mlp),
                "params": 0,
                "indices": [],
            }
        unique_tensors[name]["indices"].append(i)
        unique_tensors[name]["params"] += 1  # each entry = 1 block

    total_blocks = len(tensor_names)
    target_total = target_bits * total_blocks

    # Step 2: Start everything at the target bits (or nearest allowed)
    # Find the allowed bit width at or below target
    base_bits = max(b for b in BIT_UPGRADE_PATH if b <= target_bits) if target_bits >= 2 else 2

    tensor_bits = {}
    for name, info in unique_tensors.items():
        tensor_bits[name] = base_bits

    # Step 3: Boost CRITICAL tensors
    critical_names = [n for n, info in unique_tensors.items() if info["tier"] == Tier.CRITICAL]
    critical_blocks = sum(len(unique_tensors[n]["indices"]) for n in critical_names)
    critical_pct = critical_blocks / total_blocks if total_blocks > 0 else 0

    # Boost cap depends on how much of the model is CRITICAL.
    # MoE models: CRITICAL is 2-5% → can afford +4 steps (e.g. 4→8)
    # Dense models: CRITICAL is 10-25% → limit to +2 steps to avoid over-compensation
    max_boost = 4 if critical_pct < 0.05 else 2

    for name in critical_names:
        boosted = base_bits
        candidate = _next_bit_width(boosted)
        while candidate is not None:
            tensor_bits[name] = candidate
            boosted = candidate
            candidate = _next_bit_width(boosted)
            if boosted - base_bits >= max_boost:
                break

    # Step 3b: First/last layer bonus (+1 step for first/last 2 transformer layers)
    # These layers are empirically most sensitive (same heuristic as GGUF K-quants).
    first_last_bonus = 2

    # Precompute layer index per tensor and total layer count
    tensor_layer_idx = {}
    all_layer_ids = set()
    for name in unique_tensors:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    lid = int(parts[i + 1])
                    tensor_layer_idx[name] = lid
                    all_layer_ids.add(lid)
                except ValueError:
                    pass
                break
    n_layers = max(all_layer_ids) + 1 if all_layer_ids else 1

    for name, layer_idx in tensor_layer_idx.items():
        if layer_idx < first_last_bonus or layer_idx >= n_layers - first_last_bonus:
            current = tensor_bits[name]
            next_bw = _next_bit_width(current)
            if next_bw is not None and next_bw <= 8:
                tensor_bits[name] = next_bw

    # Step 4: Calculate current average and compensate
    current_total = sum(tensor_bits[name] * len(unique_tensors[name]["indices"])
                       for name in unique_tensors)
    overspend = current_total - target_total

    if overspend > 0:
        # Downgrade COMPRESS tensors to compensate.
        # Sort SMALLEST first — accumulate savings gradually, stop closer to exact
        # budget. Biggest-first overshoots because one large tensor can exceed the
        # entire overspend, wasting bits.
        compress_tensors = [
            (name, len(info["indices"]))
            for name, info in unique_tensors.items()
            if info["tier"] == Tier.COMPRESS
        ]
        compress_tensors.sort(key=lambda x: x[1])  # smallest first

        lower_bits = _prev_bit_width(base_bits)
        if lower_bits:
            remaining = overspend
            for name, n_blocks in compress_tensors:
                if remaining <= 0:
                    break
                # Apply MLP asymmetry floor: don't downgrade below the floor.
                # Threading ``base_bits`` lets the floor protect routed
                # ``up_proj`` / ``w3`` at the namesake bit width for K-quant
                # profiles (JANG_4K and above) on 256+ expert MoE models.
                floor = _apply_mlp_asymmetry_floor(
                    name, lower_bits, num_experts, base_bits=base_bits
                )
                if floor > lower_bits:
                    continue  # Can't downgrade this tensor (floor prevents it)
                savings = (base_bits - lower_bits) * n_blocks
                tensor_bits[name] = lower_bits
                remaining -= savings

    # Step 5: Build output array
    bits = np.empty(total_blocks, dtype=np.uint8)
    for name, info in unique_tensors.items():
        for idx in info["indices"]:
            bits[idx] = tensor_bits[name]

    return bits


def allocate_bits_profile(
    tensor_names: list[str],
    profile: str = "JANG_3M",
    num_experts: int = 0,
    has_shared_mlp: bool = False,
    apply_mlp_asymmetry: bool = True,
) -> np.ndarray:
    """
    Tier-based bit allocation — classifies each tensor into a sensitivity
    tier then assigns bits from the profile.

    Works for ANY architecture: dense transformer, MoE, hybrid SSM,
    MLA, VL, Mamba, etc. No tensor name hardcoding needed.

    For 256+ expert models, applies MLP asymmetry fix:
      - gate_proj of routed experts → IMPORTANT tier (4-bit min)
      - down_proj of routed experts → 3-bit floor (prevents residual corruption)
      - up_proj stays COMPRESS (2-bit OK when gate is protected)
      Budget-neutral at (4+2+3)/3 = 3.0 avg for 3-bit profiles.

    Args:
        tensor_names: tensor name for each block
        profile: e.g. "JANG_3M", "JANG_4S", "JANG_2M" (case-insensitive)
        num_experts: number of MoE experts (0 for dense models)

    Returns:
        uint8 array of bit widths per block
    """
    profile = profile.upper()
    if profile not in JANG_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(JANG_PROFILES.keys())}")

    critical_bits, important_bits, compress_bits = JANG_PROFILES[profile]
    tier_to_bits = {
        Tier.CRITICAL: critical_bits,
        Tier.IMPORTANT: important_bits,
        Tier.COMPRESS: compress_bits,
    }

    n_blocks = len(tensor_names)

    # Optimized: classify unique names, detect runs of repeated names,
    # build output with numpy.repeat instead of per-element loop.
    # This turns 3.7B per-element assignments into ~400 repeat operations.
    cache = {}
    runs = []  # [(bits_value, count), ...]
    prev_name = None
    run_count = 0

    for name in tensor_names:
        if name == prev_name:
            run_count += 1
        else:
            if prev_name is not None:
                runs.append((cache[prev_name], run_count))
            if name not in cache:
                assigned = tier_to_bits[classify_tensor(name, num_experts, has_shared_mlp)]
                if apply_mlp_asymmetry:
                    assigned = _apply_mlp_asymmetry_floor(name, assigned, num_experts)
                cache[name] = assigned
            prev_name = name
            run_count = 1
    if prev_name is not None:
        runs.append((cache[prev_name], run_count))

    # Build output array from runs using numpy
    bits = np.empty(n_blocks, dtype=np.uint8)
    offset = 0
    for bval, count in runs:
        bits[offset:offset + count] = bval
        offset += count

    return bits


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
                for b in BIT_UPGRADE_PATH:
                    if b >= new_bits:
                        bits[i] = b
                        break
                else:
                    bits[i] = BIT_UPGRADE_PATH[-1]

    # Calculate remaining budget
    current_avg = float(np.mean(bits))
    if current_avg >= target_bits:
        return bits.astype(np.uint8)

    # Use a max-heap to upgrade most important blocks first
    heap = [(-float(importance_scores[i]), i) for i in range(n_blocks)]
    heapq.heapify(heap)

    total_bits = int(np.sum(bits))
    target_total = int(target_bits * n_blocks)

    while total_bits < target_total and heap:
        neg_imp, idx = heapq.heappop(heap)

        current = bits[idx]
        next_bw = _next_bit_width(current)
        if next_bw is None:
            continue

        cost = next_bw - current
        if total_bits + cost <= target_total + n_blocks * 0.01:
            bits[idx] = next_bw
            total_bits += cost

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

    For very large models (millions of blocks), falls back to greedy.
    """
    n_blocks = len(importance_scores)

    if n_blocks > 50000:
        return allocate_bits_greedy(
            importance_scores, target_bits, tensor_names, n_layers, block_size
        )

    distortion = np.zeros((n_blocks, len(BIT_UPGRADE_PATH)), dtype=np.float64)
    for j, b in enumerate(BIT_UPGRADE_PATH):
        distortion[:, j] = importance_scores * weight_variances * (4.0 ** (-b))

    min_bits = np.full(n_blocks, 2, dtype=np.int32)
    for i, name in enumerate(tensor_names):
        _, layer_idx, mb = classify_layer(name)
        min_bits[i] = mb

    target_total = int(round(target_bits * n_blocks))

    bits = min_bits.copy()
    total = int(np.sum(bits))

    if total >= target_total:
        return bits.astype(np.uint8)

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
            break

        old = bits[best_idx]
        new = _next_bit_width(old)
        bits[best_idx] = new
        total += new - old

    return bits.astype(np.uint8)


def summarize_allocation(
    bit_map: np.ndarray,
    tensor_names: Optional[list[str]] = None,
    num_experts: int = 0,
) -> dict:
    """
    Generate a summary of the bit allocation.

    Returns dict with:
        - average_bits: actual average
        - histogram: count of blocks at each bit width
        - per_tier: average bits per tier (if tensor_names given)
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
        tier_bits = {Tier.CRITICAL: [], Tier.IMPORTANT: [], Tier.COMPRESS: []}
        for i, name in enumerate(tensor_names):
            tier = classify_tensor(name, num_experts)
            tier_bits[tier].append(int(bit_map[i]))

        result["per_tier"] = {}
        for tier in [Tier.CRITICAL, Tier.IMPORTANT, Tier.COMPRESS]:
            vals = tier_bits[tier]
            if vals:
                result["per_tier"][tier.name] = {
                    "avg_bits": round(float(np.mean(vals)), 2),
                    "blocks": len(vals),
                    "pct": round(100 * len(vals) / len(bit_map), 1),
                }

    return result


# ============================================================
# Compact Allocation — Per-Tensor (no per-block arrays)
# ============================================================
#
# For profile and budget allocation, every block in a tensor gets the
# same bit width. The per-block arrays are redundant and catastrophically
# expensive on large models: GLM-5 (256 experts × 78 layers) generates
# 5.89 billion blocks → 100 GB of allocation arrays.
#
# These compact functions classify each TENSOR once and return a dict
# of tensor_name → bits. Memory: ~10 MB instead of ~100 GB.


def allocate_bits_profile_compact(
    tensor_info: list[tuple[str, int]],
    profile: str = "JANG_3M",
    num_experts: int = 0,
    has_shared_mlp: bool = False,
    apply_mlp_asymmetry: bool = True,
) -> dict[str, int]:
    """Per-tensor profile allocation. Returns tensor_name → bits.

    Equivalent to allocate_bits_profile() but without per-block arrays.
    """
    profile = profile.upper()
    if profile not in JANG_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(JANG_PROFILES.keys())}")

    critical_bits, important_bits, compress_bits = JANG_PROFILES[profile]
    tier_to_bits = {
        Tier.CRITICAL: critical_bits,
        Tier.IMPORTANT: important_bits,
        Tier.COMPRESS: compress_bits,
    }

    result = {}
    for name, n_blocks in tensor_info:
        bits = tier_to_bits[classify_tensor(name, num_experts, has_shared_mlp)]
        if apply_mlp_asymmetry:
            bits = _apply_mlp_asymmetry_floor(name, bits, num_experts)
        result[name] = bits
    return result


def allocate_bits_budget_compact(
    tensor_info: list[tuple[str, int]],
    target_bits: float = 4.0,
    num_experts: int = 0,
    has_shared_mlp: bool = False,
) -> dict[str, int]:
    """Per-tensor budget-neutral allocation. Returns tensor_name → bits.

    Equivalent to allocate_bits_budget() but without per-block arrays.
    Same logic: boost CRITICAL, first/last layer bonus, compensate by
    downgrading COMPRESS tensors.
    """
    total_blocks = sum(n for _, n in tensor_info)
    target_total = target_bits * total_blocks

    # Classify each tensor
    classified = {}
    for name, n_blocks in tensor_info:
        tier = classify_tensor(name, num_experts, has_shared_mlp)
        classified[name] = {"tier": tier, "n_blocks": n_blocks}

    # Start at base bits
    base_bits = max(b for b in BIT_UPGRADE_PATH if b <= target_bits) if target_bits >= 2 else 2
    tensor_bits = {name: base_bits for name, _ in tensor_info}

    # Boost CRITICAL
    critical_blocks = sum(info["n_blocks"] for info in classified.values()
                          if info["tier"] == Tier.CRITICAL)
    critical_pct = critical_blocks / total_blocks if total_blocks > 0 else 0
    max_boost = 4 if critical_pct < 0.05 else 2

    for name, info in classified.items():
        if info["tier"] == Tier.CRITICAL:
            boosted = base_bits
            for _ in range(max_boost):
                nxt = _next_bit_width(boosted)
                if nxt is None:
                    break
                tensor_bits[name] = nxt
                boosted = nxt

    # First/last layer bonus (+1 step for first/last 2 layers)
    tensor_layer_idx = {}
    all_layer_ids = set()
    for name, _ in tensor_info:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    lid = int(parts[i + 1])
                    tensor_layer_idx[name] = lid
                    all_layer_ids.add(lid)
                except ValueError:
                    pass
                break
    n_layers = max(all_layer_ids) + 1 if all_layer_ids else 1

    for name, layer_idx in tensor_layer_idx.items():
        if layer_idx < 2 or layer_idx >= n_layers - 2:
            current = tensor_bits[name]
            nxt = _next_bit_width(current)
            if nxt is not None and nxt <= 8:
                tensor_bits[name] = nxt

    # Compensate overspend by downgrading COMPRESS tensors
    current_total = sum(tensor_bits[name] * classified[name]["n_blocks"]
                        for name in classified)
    overspend = current_total - target_total

    if overspend > 0:
        compress_tensors = [
            (name, classified[name]["n_blocks"])
            for name in classified
            if classified[name]["tier"] == Tier.COMPRESS
        ]
        compress_tensors.sort(key=lambda x: x[1])  # smallest first

        lower_bits = _prev_bit_width(base_bits)
        if lower_bits:
            remaining = overspend
            for name, n_blocks in compress_tensors:
                if remaining <= 0:
                    break
                floor = _apply_mlp_asymmetry_floor(
                    name, lower_bits, num_experts, base_bits=base_bits
                )
                if floor > lower_bits:
                    continue
                savings = (base_bits - lower_bits) * n_blocks
                tensor_bits[name] = lower_bits
                remaining -= savings

    return tensor_bits


def summarize_allocation_compact(
    tensor_bits: dict[str, int],
    tensor_info: list[tuple[str, int]],
    num_experts: int = 0,
) -> dict:
    """Compute allocation statistics from per-tensor bit assignments."""
    total_blocks = sum(n for _, n in tensor_info)
    total_bits_sum = sum(tensor_bits[name] * n for name, n in tensor_info)
    avg = total_bits_sum / total_blocks if total_blocks else 0

    histogram = {}
    for name, n_blocks in tensor_info:
        bw = tensor_bits[name]
        key = f"{bw}-bit"
        if key not in histogram:
            histogram[key] = {"count": 0, "percent": 0}
        histogram[key]["count"] += n_blocks

    for key in histogram:
        histogram[key]["percent"] = round(100 * histogram[key]["count"] / total_blocks, 1)

    return {
        "average_bits": avg,
        "total_blocks": total_blocks,
        "histogram": dict(sorted(histogram.items())),
    }
