"""MiniMax-M3 (minimax_m3_vl) streaming forward + REAP/probe tooling.

M3 text backbone is a GQA + block-sparse-selection (MSA) MoE with Gemma-style
RMSNorm and swigluoai activations. This subpackage implements a pure-torch,
layer-streamed forward so a 427B checkpoint can be probed (coherence) and
profiled (REAP saliency) without ever materializing the whole model.

Created by Jinho Jang (eric@jangq.ai).
"""
