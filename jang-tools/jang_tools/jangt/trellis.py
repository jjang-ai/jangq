"""Reference bitshift-trellis encoder / decoder (numpy, exact; used for code search, tests and as the oracle for the
Metal kernels). State = 16-bit window; each step shifts in K new bits: s' = ((s << K) | c) & 0xFFFF.
Predecessors of s' are s = (s' >> K) | (h << (16 - K)), h in [0, 2^K)."""
from __future__ import annotations
import numpy as np

L = 16
NS = 1 << L


def viterbi(x: np.ndarray, code: np.ndarray, K: int, tail_biting: bool = False) -> tuple[np.ndarray, float]:
    """Quantize a sequence x (length T) with the trellis. Returns (states per step, squared error sum)."""
    T = len(x)
    low = NS >> K                                   # 2^(L-K) distinct low parts
    cost = (x[0] - code) ** 2                       # any start state
    back = np.empty((T, NS), dtype=np.uint8 if K <= 8 else np.uint16)
    idx = np.arange(NS)
    for t in range(1, T):
        c = cost.reshape(1 << K, low)               # c[h, m] = cost[h * low + m]
        h = c.argmin(axis=0)                        # best high part for each low part m
        best = c[h, np.arange(low)]
        m = idx >> K                                # s' -> its predecessor low part
        cost = best[m] + (x[t] - code) ** 2
        back[t] = h[m]
    s = int(cost.argmin()); err = float(cost[s])
    states = np.empty(T, dtype=np.int64); states[-1] = s
    for t in range(T - 1, 0, -1):
        s = (s >> K) | (int(back[t, s]) << (L - K)); states[t - 1] = s
    return states, err


def gaussian_mse(code: np.ndarray, K: int, tiles: int = 32, T: int = 256, seed: int = 1,
                 scales=(0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10)) -> tuple[float, float]:
    """Per-weight MSE on iid N(0,1) data at the best global scale (code values scaled by `scale`)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((tiles, T))
    best = (np.inf, None)
    for sc in scales:
        tot = sum(viterbi(x, code * sc, K)[1] for x in X)
        if tot < best[0]:
            best = (tot, sc)
    return best[0] / X.size, best[1]


LLOYD_MAX = {2: 0.1175, 3: 0.03454, 4: 0.009497}   # classical Lloyd-Max values for N(0,1)     # scalar quantizer MSE on N(0,1) (classical values)
RD_BOUND = {K: 2.0 ** (-2 * K) for K in (2, 3, 4)}


def viterbi_v(x: np.ndarray, code2: np.ndarray, K: int, V: int = 2, Lb: int = 16) -> tuple[np.ndarray, float]:
    """Vector trellis: each step shifts in K*V bits and the 16-bit state emits V values code2[s] (shape (NS, V)).
    x length must be a multiple of V. Returns (states per step, squared error)."""
    xs = x.reshape(-1, V); T = len(xs); KV = K * V; NSl = 1 << Lb
    code2 = code2[:NSl]
    low = NSl >> KV
    def step_cost(xv):
        return ((xv[None, :] - code2) ** 2).sum(axis=1)
    cost = step_cost(xs[0]); back = np.empty((T, NSl), dtype=np.uint16); idx = np.arange(NSl)
    for t in range(1, T):
        c = cost.reshape(1 << KV, low); h = c.argmin(axis=0); best = c[h, np.arange(low)]
        m = idx >> KV; cost = best[m] + step_cost(xs[t]); back[t] = h[m]
    s = int(cost.argmin()); err = float(cost[s]); states = np.empty(T, dtype=np.int64); states[-1] = s
    for t in range(T - 1, 0, -1):
        s = (s >> KV) | (int(back[t, s]) << (Lb - KV)); states[t - 1] = s
    return states, err


def gaussian_mse_v(code2: np.ndarray, K: int, V: int = 2, tiles: int = 16, T: int = 256, seed: int = 1,
                   scales=(0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2), Lb: int = 16) -> tuple[float, float]:
    rng = np.random.default_rng(seed); X = rng.standard_normal((tiles, T)); best = (np.inf, None)
    for sc in scales:
        tot = sum(viterbi_v(x, code2 * sc, K, V, Lb)[1] for x in X)
        if tot < best[0]:
            best = (tot, sc)
    return best[0] / X.size, best[1]
