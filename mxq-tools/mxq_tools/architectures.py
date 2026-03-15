"""
MXQ Architecture Detection and Per-Architecture Quantization Config
Created by Eric Jang (eric@vmlx.net)

Handles the diversity of modern LLM architectures:
- Standard Transformer (Llama, Qwen, Gemma, Phi, Mistral)
- Vision-Language (Qwen-VL, Llama-Vision, Gemma-VL)
- Mamba / State Space Models (Mamba, Mamba2, Jamba)
- Mixture of Experts (Mixtral, DeepSeek-V2/V3, DBRX)
- Hybrid SSM+Attention (Jamba, Nemotron-H, Zamba)

Each architecture type has different:
- Weight tensor names and shapes
- Sensitivity profiles (which weights matter most)
- Minimum bit floors per layer type
- Calibration strategies
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from typing import Optional


class ArchType(Enum):
    """Model architecture families."""
    TRANSFORMER = "transformer"          # Standard dense transformer
    VISION_LANGUAGE = "vision_language"  # Transformer + vision encoder
    MAMBA = "mamba"                      # Pure state space model
    MOE = "moe"                         # Mixture of experts
    HYBRID_SSM = "hybrid_ssm"           # SSM + attention hybrid
    HYBRID_MOE_SSM = "hybrid_moe_ssm"   # SSM + attention + MoE


class AttentionType(Enum):
    """Attention mechanism variants."""
    MHA = "mha"      # Multi-Head Attention (all heads same dim)
    GQA = "gqa"      # Grouped Query Attention (fewer KV heads)
    MQA = "mqa"      # Multi-Query Attention (1 KV head)
    MLA = "mla"      # Multi-head Latent Attention (DeepSeek-V2/V3)
    NONE = "none"    # No attention (pure SSM)


@dataclass
class LayerQuantConfig:
    """Quantization configuration for a specific layer type."""
    min_bits: int = 2
    preferred_bits: int = 4
    max_bits: int = 8
    importance_weight: float = 1.0  # multiplier on importance score
    description: str = ""


@dataclass
class ArchConfig:
    """Architecture-specific quantization configuration."""
    arch_type: ArchType
    attention_type: AttentionType
    model_type: str  # HuggingFace model_type string

    # Layer-type quantization rules
    layer_configs: dict[str, LayerQuantConfig] = field(default_factory=dict)

    # Architecture-specific info
    has_vision_encoder: bool = False
    has_ssm_layers: bool = False
    has_moe_layers: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0

    # Tensor name patterns for identifying layer types
    tensor_patterns: dict[str, str] = field(default_factory=dict)


# ============================================================
# Architecture Configurations
# ============================================================

TRANSFORMER_LAYER_CONFIGS = {
    "embed_tokens": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=2.0,
        description="Token embeddings — vocabulary representation, critical"
    ),
    "lm_head": LayerQuantConfig(
        min_bits=6, preferred_bits=6, importance_weight=3.0,
        description="Output head — directly affects token probabilities"
    ),
    "q_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.5,
        description="Attention query projection — sensitive to quantization"
    ),
    "k_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.4,
        description="Attention key projection — affects attention pattern"
    ),
    "v_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=3, importance_weight=1.2,
        description="Attention value projection — moderate sensitivity"
    ),
    "o_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=3, importance_weight=1.1,
        description="Attention output projection"
    ),
    "gate_proj": LayerQuantConfig(
        min_bits=2, preferred_bits=2, importance_weight=0.8,
        description="MLP gate — most compressible"
    ),
    "up_proj": LayerQuantConfig(
        min_bits=2, preferred_bits=2, importance_weight=0.8,
        description="MLP up projection — most compressible"
    ),
    "down_proj": LayerQuantConfig(
        min_bits=2, preferred_bits=2, importance_weight=0.9,
        description="MLP down projection — slightly more sensitive than gate/up"
    ),
    "input_layernorm": LayerQuantConfig(
        min_bits=8, preferred_bits=8, importance_weight=0.0,
        description="RMSNorm — keep full precision (tiny tensor)"
    ),
    "post_attention_layernorm": LayerQuantConfig(
        min_bits=8, preferred_bits=8, importance_weight=0.0,
        description="RMSNorm — keep full precision (tiny tensor)"
    ),
    "norm": LayerQuantConfig(
        min_bits=8, preferred_bits=8, importance_weight=0.0,
        description="Final norm — keep full precision"
    ),
}

VISION_LAYER_CONFIGS = {
    # Vision encoder weights — typically less compressible than language
    "visual.patch_embed": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=2.0,
        description="Vision patch embedding — first layer of visual processing"
    ),
    "visual.blocks": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.5,
        description="ViT transformer blocks — vision features"
    ),
    "visual.merger": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=1.8,
        description="Vision-language connector — bridges modalities"
    ),
}

MAMBA_LAYER_CONFIGS = {
    # Mamba/SSM specific layers
    "in_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.3,
        description="SSM input projection"
    ),
    "out_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.3,
        description="SSM output projection"
    ),
    "x_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.5,
        description="SSM state projection — maps to B, C, dt"
    ),
    "dt_proj": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=1.8,
        description="SSM timestep projection — critical for state dynamics"
    ),
    "A_log": LayerQuantConfig(
        min_bits=6, preferred_bits=8, importance_weight=3.0,
        description="SSM state matrix (log scale) — DO NOT quantize aggressively"
    ),
    "D": LayerQuantConfig(
        min_bits=8, preferred_bits=8, importance_weight=0.0,
        description="SSM skip connection — keep full precision (tiny)"
    ),
    "conv1d": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=1.5,
        description="SSM causal convolution"
    ),
}

MOE_LAYER_CONFIGS = {
    # Mixture of Experts specific
    "gate": LayerQuantConfig(
        min_bits=6, preferred_bits=8, importance_weight=3.0,
        description="Expert router/gate — critical, determines which experts fire"
    ),
    "experts": LayerQuantConfig(
        min_bits=2, preferred_bits=2, importance_weight=0.7,
        description="Individual expert weights — many experts = more redundancy"
    ),
    "shared_expert": LayerQuantConfig(
        min_bits=3, preferred_bits=3, importance_weight=1.2,
        description="Shared expert (DeepSeek) — always active, more important"
    ),
}

# MLA (Multi-head Latent Attention) specific — DeepSeek-V2/V3
MLA_LAYER_CONFIGS = {
    "kv_a_proj_with_mqa": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=1.5,
        description="MLA KV compression projection"
    ),
    "kv_b_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.3,
        description="MLA KV decompression projection"
    ),
    "q_a_proj": LayerQuantConfig(
        min_bits=4, preferred_bits=4, importance_weight=1.5,
        description="MLA query compression"
    ),
    "q_b_proj": LayerQuantConfig(
        min_bits=3, preferred_bits=4, importance_weight=1.3,
        description="MLA query decompression"
    ),
}


def detect_architecture(model_path: str | Path) -> ArchConfig:
    """
    Detect model architecture from config.json and return quantization config.

    Args:
        model_path: path to model directory (with config.json)

    Returns:
        ArchConfig with architecture-specific quantization settings
    """
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_path}")

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    architectures = config.get("architectures", [])

    # Detect attention type
    num_heads = config.get("num_attention_heads", 0)
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    if num_kv_heads == 0:
        attention_type = AttentionType.NONE
    elif num_kv_heads == 1:
        attention_type = AttentionType.MQA
    elif num_kv_heads < num_heads:
        attention_type = AttentionType.GQA
    else:
        attention_type = AttentionType.MHA

    # Check for MLA (DeepSeek-V2/V3)
    if config.get("kv_lora_rank") or "DeepseekV2" in str(architectures):
        attention_type = AttentionType.MLA

    # Detect architecture type and build config
    arch_config = _classify_architecture(model_type, architectures, config)
    arch_config.attention_type = attention_type
    arch_config.model_type = model_type

    return arch_config


def _classify_architecture(
    model_type: str,
    architectures: list[str],
    config: dict,
) -> ArchConfig:
    """Classify into architecture family and build layer configs."""

    arch_str = str(architectures)

    # --- Vision-Language Models ---
    if any(vl in model_type.lower() for vl in ["qwen2_vl", "llava", "pixtral", "gemma_vl"]):
        layers = {**TRANSFORMER_LAYER_CONFIGS, **VISION_LAYER_CONFIGS}
        return ArchConfig(
            arch_type=ArchType.VISION_LANGUAGE,
            attention_type=AttentionType.GQA,
            model_type=model_type,
            layer_configs=layers,
            has_vision_encoder=True,
        )

    # --- Pure Mamba/SSM ---
    if model_type in ("mamba", "mamba2"):
        layers = {**MAMBA_LAYER_CONFIGS}
        # Mamba still has embed + lm_head
        layers["embed_tokens"] = TRANSFORMER_LAYER_CONFIGS["embed_tokens"]
        layers["lm_head"] = TRANSFORMER_LAYER_CONFIGS["lm_head"]
        layers["norm"] = TRANSFORMER_LAYER_CONFIGS["norm"]
        return ArchConfig(
            arch_type=ArchType.MAMBA,
            attention_type=AttentionType.NONE,
            model_type=model_type,
            layer_configs=layers,
            has_ssm_layers=True,
        )

    # --- Hybrid SSM + Attention (Jamba, Zamba, Nemotron-H) ---
    if model_type in ("jamba", "zamba", "zamba2") or "Jamba" in arch_str:
        layers = {**TRANSFORMER_LAYER_CONFIGS, **MAMBA_LAYER_CONFIGS}
        return ArchConfig(
            arch_type=ArchType.HYBRID_SSM,
            attention_type=AttentionType.GQA,
            model_type=model_type,
            layer_configs=layers,
            has_ssm_layers=True,
        )

    # --- Mixture of Experts ---
    num_experts = config.get("num_local_experts", config.get("num_experts", 0))
    if num_experts > 1:
        layers = {**TRANSFORMER_LAYER_CONFIGS, **MOE_LAYER_CONFIGS}

        # Check for MLA + MoE (DeepSeek-V2/V3)
        if config.get("kv_lora_rank") or "Deepseek" in arch_str:
            layers.update(MLA_LAYER_CONFIGS)

        return ArchConfig(
            arch_type=ArchType.MOE,
            attention_type=AttentionType.GQA,
            model_type=model_type,
            layer_configs=layers,
            has_moe_layers=True,
            num_experts=num_experts,
            num_experts_per_tok=config.get("num_experts_per_tok", 2),
        )

    # --- Standard Transformer (Llama, Qwen, Gemma, Phi, Mistral, etc.) ---
    return ArchConfig(
        arch_type=ArchType.TRANSFORMER,
        attention_type=AttentionType.GQA,
        model_type=model_type,
        layer_configs=TRANSFORMER_LAYER_CONFIGS.copy(),
    )


def get_layer_config(
    arch_config: ArchConfig,
    tensor_name: str,
) -> LayerQuantConfig:
    """
    Get the quantization config for a specific tensor based on architecture.

    Matches tensor name against layer config patterns.
    Falls back to default (2-bit min) if no match.
    """
    for pattern, config in arch_config.layer_configs.items():
        if pattern in tensor_name:
            return config

    # Default: fully compressible
    return LayerQuantConfig(min_bits=2, preferred_bits=2, importance_weight=1.0)


def get_skip_tensors(arch_config: ArchConfig) -> set[str]:
    """
    Get tensor name patterns that should NOT be quantized (kept in fp16/bf16).

    These are typically tiny tensors where quantization overhead > savings:
    - LayerNorm/RMSNorm weights
    - SSM D parameter (skip connection scalar)
    - Bias terms
    """
    skip = set()
    for pattern, config in arch_config.layer_configs.items():
        if config.importance_weight == 0.0:
            skip.add(pattern)
    return skip


def summarize_architecture(arch_config: ArchConfig) -> str:
    """Human-readable architecture summary."""
    lines = [
        f"Architecture: {arch_config.arch_type.value}",
        f"Model type: {arch_config.model_type}",
        f"Attention: {arch_config.attention_type.value}",
    ]
    if arch_config.has_vision_encoder:
        lines.append("Vision encoder: YES")
    if arch_config.has_ssm_layers:
        lines.append("SSM layers: YES")
    if arch_config.has_moe_layers:
        lines.append(f"MoE: {arch_config.num_experts} experts, top-{arch_config.num_experts_per_tok}")

    lines.append("\nLayer quantization rules:")
    for pattern, config in sorted(arch_config.layer_configs.items()):
        lines.append(f"  {pattern:<30s}  min={config.min_bits}  pref={config.preferred_bits}  weight={config.importance_weight:.1f}  {config.description}")

    return "\n".join(lines)
