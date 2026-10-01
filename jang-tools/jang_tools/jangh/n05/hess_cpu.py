"""CPU-only (numpy/scipy float64) Hessian handling for JANGH GPTQ — NO MLX import, safe in worker processes.
See gptq_stable.py for the rationale (constant-dominated inputs, TF32, LAPACK safety)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

_HAD = None


def had32() -> np.ndarray:
    global _HAD
    if _HAD is None:
        Hm = np.array([[1.0]])
        for _ in range(5):
            Hm = np.block([[Hm, Hm], [Hm, -Hm]])
        _HAD = Hm / np.sqrt(32.0)
    return _HAD


def rot_both(H: np.ndarray) -> np.ndarray:
    """R H R^T, R = blockwise normalized Hadamard-32 (symmetric), float64."""
    K = H.shape[0]; Hd = had32()
    H1 = (Hd @ H.reshape(K // 32, 32, K)).reshape(K, K)
    return (H1.reshape(K, K // 32, 32) @ Hd).reshape(K, K)


def rot_vec(v: np.ndarray) -> np.ndarray:
    K = v.shape[-1]
    return (v.reshape(*v.shape[:-1], K // 32, 32) @ had32()).reshape(v.shape)


def st_memmap(path: str) -> dict:
    """Read-only numpy views of a safetensors file (no MLX: safe in worker processes, no lazy-load surprises)."""
    import json as _json, struct as _struct
    DT = {"F32": np.float32, "F16": np.float16, "I16": np.int16, "I32": np.int32, "U8": np.uint8, "U32": np.uint32, "F64": np.float64}
    with open(path, "rb") as fh:
        n = _struct.unpack("<Q", fh.read(8))[0]
        hdr = _json.loads(fh.read(n))
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = np.memmap(path, dtype=DT[v["dtype"]], mode="r", offset=8 + n + a, shape=tuple(v["shape"]))
    return out


class HessStore:
    def __init__(self, path: str, which: str):
        self.hs = st_memmap(path); self.w = which
        self.n = int(self.hs[f"{which}_diag"].shape[1])
        self.E = int(self.hs[f"{which}_diag"].shape[0])
        r, c = np.triu_indices(self.n)
        self.iu = (r, c)
        self.rows = np.array(self.hs["rows"], dtype=np.float64)
        self.wsum = np.array(self.hs["wsum"], dtype=np.float64)
        self.mean = np.array(self.hs[f"{which}_mean"], dtype=np.float64)
        self._pool = None

    def cov(self, e: int) -> np.ndarray:
        n, w = self.n, self.w
        d = np.array(self.hs[f"{w}_diag"][e], dtype=np.float64, copy=True)
        tri = np.array(self.hs[f"{w}_tri"][e], dtype=np.float64, copy=True) / 32767.0
        C = np.zeros((n, n)); C[self.iu] = tri
        C = C + C.T; C[np.diag_indices(n)] = tri[np.cumsum(np.r_[0, np.arange(n, 1, -1)])]   # diagonal entries of the tri
        s = np.sqrt(np.maximum(d, 0.0))
        Hm = C * s[:, None] * s[None, :]
        ti = np.array(self.hs[f"{w}_top_idx"][e], copy=True); tr = np.array(self.hs[f"{w}_top_rows"][e], dtype=np.float64, copy=True)
        Hm[ti, :] = tr; Hm[:, ti] = tr.T
        Hm[np.diag_indices(n)] = np.maximum(d, 0.0)
        return Hm

    def pooled(self) -> np.ndarray:
        """Layer-level pooled within-expert covariance (weights: sum w^2 of each expert)."""
        if self._pool is None:
            acc = np.zeros((self.n, self.n)); tot = 0.0
            for e in range(self.E):
                if self.wsum[e] > 0:
                    acc += self.wsum[e] * self.cov(e); tot += self.wsum[e]
            self._pool = acc / max(tot, 1e-300)
        return self._pool


def upper_factor(Hp: np.ndarray) -> np.ndarray:
    """U upper-triangular, U^T U = Hp^-1, via one Cholesky of the order-reversed matrix (float64).
    SAFE LAPACK only: the low-level potrf/dtrtri wrappers with overwrite flags corrupted the heap here
    (objc autorelease page overwritten with doubles -> intermittent SIGBUS/abort, 2026-09-27). Every factor is
    verified on a random probe before it is used."""
    n = Hp.shape[0]
    A = np.array(Hp[::-1, ::-1], dtype=np.float64, order="C", copy=True)
    L = np.linalg.cholesky(A)                                        # raises LinAlgError if not PD
    Li = solve_triangular(L, np.eye(n), lower=True, check_finite=False)
    U = np.array(Li[::-1, ::-1], dtype=np.float64, order="C", copy=True)
    v = np.random.default_rng(0).standard_normal(n)
    chk = Hp @ (U.T @ (U @ v))
    err = np.linalg.norm(chk - v) / np.linalg.norm(v)
    if not np.isfinite(err) or err > 1e-3:        # float64 on matrices with condition numbers up to ~1e10
        raise np.linalg.LinAlgError(f"upper_factor self-check failed: |H U^T U v - v|/|v| = {err:.2e}")
    if np.abs(np.tril(U, -1)).max() != 0.0:
        raise np.linalg.LinAlgError("upper_factor: result is not upper triangular")
    return U


def prepare(H: np.ndarray, rotate: bool, damp: float = 0.01, ridge: float = 0.0, shrink_to: np.ndarray | None = None,
            rows: float = 0.0, tau: float = 0.0, act_order: bool = True):
    """H: float64 centered covariance (unrotated). Returns (U f32, Hp f32, perm int32) in the rotated+permuted basis.
    shrinkage: H <- (rows*H + tau*S_pool) / (rows + tau);  ridge: H += ridge * diag(H);  damp: + damp * diag(H_rot)."""
    if shrink_to is not None and tau > 0:
        H = (rows * H + tau * shrink_to) / (rows + tau)
    if ridge > 0:
        H = H + ridge * np.diag(np.diag(H))
    Hr = rot_both(H) if rotate else H.copy()
    d = np.diag(Hr).copy()
    Hr[np.diag_indices_from(Hr)] += damp * d + 1e-10 * d.mean()
    perm = np.argsort(-d, kind="stable") if act_order else np.arange(len(d))
    Hp = np.ascontiguousarray(Hr[perm][:, perm])
    extra = 0.0
    while True:                                   # never silently: the escalation is returned to the caller
        try:
            U = upper_factor(Hp)
            break
        except np.linalg.LinAlgError:
            step = (0.01 if extra == 0.0 else extra) * 2
            if step > 1.0:
                raise
            Hp[np.diag_indices_from(Hp)] += (step - extra) * d[perm]
            extra = step
    return U.astype(np.float32), Hp.astype(np.float32), perm.astype(np.int32), extra


