"""The JANGH recipe for Naive-N0.5-Flash routed experts — every choice made by held-out measurement
on UNSEEN documents (03-CALIBRATION.md).

  Hessian of expert e for a projection whose input has mean mu_e, centered covariance S_e and n_e calibration rows:
      H_e = (1 - m_e) * S_e + m_e * S_pool * (tr S_e / tr S_pool) + mu_e mu_e^T            (UNcentered, no bias term)
      m_e = max(m_min, tau / (n_e + tau))
  S_pool = the layer's pooled covariance over experts. For gate/up it is a real statistic (all experts read the same
  x, and it contains every document); for down it is only a regularizer (hidden units are expert-specific).
  m_min is chosen PER LAYER and per projection group on held-out rows from the candidate sets below
  (curves.py writes the choice; convert.py reads it). Poorly sampled experts are pulled to the pool by tau.
  Then: Hadamard-32 rotation, 1% diagonal damping (escalated only if the factorization fails, and reported),
  act-order GPTQ with the JANGH odd-cubic codebook, exact H-weighted row scale.
  Centered covariance + per-row bias correction was measured and REJECTED (needs a format extension and loses on
  experts whose inputs are multi-modal: 0.15 vs 0.018 on layer 2's dominant expert).
"""
from __future__ import annotations

import numpy as np

CAND = {"gu": (0.5, 1.0), "dn": (0.0, 0.5, 1.0)}
TAU = {"gu": 4096.0, "dn": 2048.0}


def hessian(S_e: np.ndarray, mu_e: np.ndarray, rows_e: float, S_pool: np.ndarray, m_min: float, tau: float) -> np.ndarray:
    m = max(m_min, tau / (rows_e + tau))
    tr_e, tr_p = np.trace(S_e), np.trace(S_pool)
    if tr_e <= 0 or rows_e <= 0:        # expert never routed in calibration: pooled statistics only
        return S_pool + np.outer(mu_e, mu_e)
    return (1.0 - m) * S_e + m * S_pool * (tr_e / tr_p) + np.outer(mu_e, mu_e)
