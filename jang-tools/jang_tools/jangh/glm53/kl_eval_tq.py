"""KL eval of a JANGTQ v2 GLM-5.3 bundle with the EXACT protocol of the shipped affine GLM-5.3-Flash-JANG
(jang_tools.glm5_next.kl_eval: FP8 klref, 20 prompts / 15,850 positions, top-128 renormalized KL, median/mean,
p50/75/90/95/99/max, top-1 agreement, ref-top1 in top-5/10, margin-conditioned flip curve). Only the loader differs.

  python -m jang_tools.jangh.glm53.kl_eval_tq --bundle DIR --klref klref.safetensors --metrics-out m.json
"""
import sys
from pathlib import Path

import jang_tools.glm5_next.load as ref_load  # noqa: E402
from jang_tools.glm5_next import kl_eval  # noqa: E402
from jang_tools.jangh.glm53.load_tq import load_tq_bundle  # noqa: E402

ref_load.load_bundle = load_tq_bundle          # kl_eval does `from .load import load_bundle` at call time

if __name__ == "__main__":
    kl_eval.main()
