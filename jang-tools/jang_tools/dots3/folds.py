"""AWQ + per-expert diagonal-imatrix folds for dots3 (runtime-neutral).

AWQ (per layer, from mlp_in second moments):
    s_j = clip((sqrt(E[x_j^2]) / geomean)^alpha, lo, hi)
    post_attention_layernorm.weight[j] /= s_j
    mlp.gate.weight[:, j]              *= s_j      (router sees same input)
    routed & shared gate/up_proj[:, j] *= s_j
  Text path only. NEVER applied to the vision tower (qwen36 collapse
  precedent) nor to attention inputs.

Diagonal imatrix (per layer, PER EXPERT, from derived X2 moments):
    d_j = clip((sqrt(E[x2_j^2]) / geomean)^alpha, lo, hi)
    up_proj[j, :]   /= d_j      (limited SwiGLU: only the up factor is linear)
    down_proj[:, j] *= d_j
  Shared expert gets its own d vector (index 256).

Both folds compose exactly; identity audit asserts |W2·silu(XW1)·(XW3/d)·d −
original| ~ 0 on random probes before any quantization.
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.25
CLIP = (0.5, 2.0)


def scales_from_moments(second_moment: np.ndarray, alpha: float = ALPHA,
                        clip: tuple[float, float] = CLIP) -> np.ndarray:
    m = np.sqrt(np.maximum(second_moment.astype(np.float64), 1e-12))
    g = np.exp(np.mean(np.log(m)))
    s = (m / g) ** alpha
    return np.clip(s, clip[0], clip[1]).astype(np.float32)


class Folds:
    """Container for awq[L,H] and imx[L,E+1,I] scale vectors (npz-backed)."""

    def __init__(self, awq: np.ndarray | None, imx: np.ndarray | None):
        self.awq = awq
        self.imx = imx

    @classmethod
    def load(cls, path) -> "Folds":
        z = np.load(path)
        return cls(z.get("awq"), z.get("imx"))

    @classmethod
    def none(cls) -> "Folds":
        return cls(None, None)

    # -- appliers: name is the SOURCE tensor name, W is (out, in) f32 --------
    def apply(self, name: str, W: np.ndarray) -> np.ndarray:
        if self.awq is None:
            return W
        parts = name.split(".")
        if not name.startswith("model.layers."):
            return W
        li = int(parts[2])
        if li >= self.awq.shape[0]:
            return W          # MTP layer: never folded
        s = self.awq[li]
        if name.endswith("post_attention_layernorm.weight"):
            return W / s
        if ".mlp.gate.weight" in name:
            return W * s[None, :]
        expert = None
        if ".mlp.experts." in name:
            expert = int(parts[5])
            proj = parts[6]
        elif ".mlp.shared_experts." in name:
            expert = -1
            proj = parts[5]
        elif parts[3] == "mlp" and parts[4] in ("gate_proj", "up_proj",
                                                "down_proj"):
            return W          # dense layer 0 / MTP dense: no fold (no router)
        else:
            return W
        d = None
        if self.imx is not None:
            d = self.imx[li, expert if expert >= 0 else -1]
        if proj == "gate_proj":
            return W * s[None, :]
        if proj == "up_proj":
            out = W * s[None, :]
            if d is not None:
                out = out / d[:, None]
            return out
        if proj == "down_proj":
            if d is not None:
                return W * d[None, :]
            return W
        return W

    def identity_audit(self, rng=None) -> float:
        """Random-tensor fold identity probe (pre-quantization neutrality)."""
        rng = rng or np.random.default_rng(0)
        H, I = 64, 48
        x = rng.standard_normal((17, H)).astype(np.float64)
        ln = rng.standard_normal(H).astype(np.float64) * 0.1 + 1.0
        W1 = rng.standard_normal((I, H)).astype(np.float64) * 0.1
        W3 = rng.standard_normal((I, H)).astype(np.float64) * 0.1
        W2 = rng.standard_normal((H, I)).astype(np.float64) * 0.1
        s = np.clip(np.exp(rng.standard_normal(H) * 0.2), 0.5, 2.0)
        d = np.clip(np.exp(rng.standard_normal(I) * 0.2), 0.5, 2.0)

        def ffn(ln_w, w1, w3, w2):
            h = x * ln_w                       # rms part cancels in the ratio
            g = h @ w1.T
            act = g / (1 + np.exp(-g)) * (h @ w3.T)
            return act @ w2.T

        ref = ffn(ln, W1, W3, W2)
        folded = ffn(ln / s, W1 * s[None, :],
                     (W3 * s[None, :]) / d[:, None], W2 * d[None, :])
        return float(np.abs(ref - folded).max() / (np.abs(ref).max() + 1e-12))
