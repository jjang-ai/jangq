"""MiniMax-M3 (minimax_m3_vl) streaming forward + REAP/probe tooling.

M3 text backbone is a GQA + block-sparse-selection (MSA) MoE with Gemma-style
RMSNorm and swigluoai activations. This subpackage implements a pure-torch,
layer-streamed forward so a 427B checkpoint can be probed (coherence) and
profiled (REAP saliency) without ever materializing the whole model.

Reachable from the unified CLI as `jang minimax-m3 <convert|probe|reap-profile|
reap-select|awq-capture>` (see cli.register), or directly via
`python -m jang_tools.minimax_m3.<tool>`.

Created by Jinho Jang (eric@jangq.ai).
"""

from .cli import register

__all__ = ["register"]
