"""MiMo-V2.5-Pro (Xiaomi) — MLX port.

Architecture: model_type=mimo_v2 (1.02T total / 42B active MoE).
- 70 layers (1 dense + 69 MoE), hidden 6144, dense inter 16384, MoE inter 2048
- 384 routed experts, top-8, no shared, sigmoid + noaux_tc routing
- GQA 128/8, asymmetric Q/K=192 V=128, partial_rotary_factor=0.334
- Hybrid: 10 full + 60 SWA(window=128), split RoPE (full=10M, swa=10K)
- Learnable SWA sink bias, attention_value_scale=0.612
- 3-layer MTP head (separate model_mtp.safetensors)
- FP8 E4M3 source with [128,128] FP32 block scales (NOT UE8M0 like DSV4)
- All `o_proj` layers kept bf16 (per upstream `ignored_layers`)
"""

from .model import MiMoV2Model, MiMoV2ForCausalLM
from .config import MiMoV2Config

__all__ = ["MiMoV2Config", "MiMoV2Model", "MiMoV2ForCausalLM"]
