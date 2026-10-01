"""Validate EVERY expert of a capture-v3 Hessian file: finite, and positive definite in correlation space
with 1% damping (float64 Cholesky), for the gate/up input and the down input. Exit 1 on any failure.
  python n05/validate_hess.py work/hess/L05.safetensors [more files...]"""
import sys
from pathlib import Path
import numpy as np

from jang_tools.jangh.n05.hess_cpu import HessStore

bad_total = 0
for f in sys.argv[1:]:
    line = []
    for which in ("gu", "dn"):
        HS = HessStore(f, which); fails = []; worst = 1.0; few = 0
        for e in range(HS.E):
            if HS.rows[e] <= 0:
                continue
            S = HS.cov(e); d = np.diag(S)
            if not np.isfinite(S).all() or not np.isfinite(HS.mean[e]).all():
                fails.append((e, "nan")); continue
            live = d > 0
            if HS.rows[e] < 64 or live.sum() < 2:
                few += 1; continue                    # too few rows for a covariance: the recipe uses the pooled one
            C = S[live][:, live] / np.sqrt(np.outer(d[live], d[live]))
            C[np.diag_indices_from(C)] += 0.01
            try:
                Lc = np.linalg.cholesky(C)
                worst = min(worst, float(np.diag(Lc).min() ** 2))
            except np.linalg.LinAlgError:
                fails.append((e, int(HS.rows[e])))
        line.append(f"{which}: {len(fails)} of {int((HS.rows > 0).sum())} experts fail" + (f" {fails[:6]}" if fails else "") + f", smallest pivot {worst:.4f}, experts with < 64 rows {few}")
        bad_total += len(fails)
    print(Path(f).name, "|", " | ".join(line), flush=True)
print("VALIDATION", "FAILED" if bad_total else "PASS", flush=True)
sys.exit(1 if bad_total else 0)
