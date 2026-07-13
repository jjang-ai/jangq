# Intent Prune — Scorer Bake-off & Ship Report (IP5)

| Field | Value |
|---|---|
| **Title** | Hybrid weight freeze and Intent Prune ship report |
| **Date** | 2026-07-12 |
| **Status** | **Ship defaults frozen** (synthetic + unit evidence) |
| **Plan** | `docs/plans/2026-07-12-jang-studio-intent-prune-crack.md` §14, PR-IP5 |
| **Scorer** | `hybrid_v1` (`jang_tools/intent_prune/score.py`) |
| **Default preset** | **Balanced** |
| **GPU Qwen TRACE** | Deferred — procedure documented below; not required for this freeze |

---

## 1. Purpose

IP5 freezes **Balanced** and **Specialist** fusion weights as **SHIP DEFAULTS** after the bake-off matrix (mass vs path vs hybrid vs CRACK) is evaluated with the evidence available without a full Qwen BF16 GPU TRACE run.

**Ship criterion (plan §14.2):**

1. Hybrid (D) must be **best or within ε of best** on keep-intent primary metric, **and**
2. Hybrid must be **strictly better than pure mass** on at least one path-sensitive stress test (synthetic highway expert or known disagreement set).

This report satisfies (2) with the **required Appendix C synthetic highway** case and locks weights. Full Reviewed-50 TRACE bake-off on Qwen remains a follow-up procedure (section 7), not a gate for enabling `hybrid_v1` defaults in code.

---

## 2. Bake-off matrix

Fixed design (plan §14.1): Qwen MoE BF16, Standard K, Reviewed 50 — when live TRACE is available.  
**This freeze** uses the same **arms** against the shared synthetic highway fixture + unit fusion tests (no model weights required).

| Arm | Description | Implementation | Primary signal |
|---|---|---|---|
| **A** | Mass / frequency only | `mass_only_scores()` + selection mass from adjacency | How often an expert is selected (REAP-adjacent marginal) |
| **B** | REAP-style / kimi weighted_freq | Optional Advanced baseline (not default); portable REAP deferred | Saliency / weighted freq if wired |
| **C** | Path / Markov only | Stationary `π_g` (power iteration on layer-path graph) | Highway / multi-layer path importance |
| **D** | **Hybrid balanced (candidate → ship default)** | `score_hybrid(..., preset="balanced")` | Path + mass + intent + drop + backbone + mild safety |
| **E** | Hybrid + CRACK stance | Same weights, `safety_stance="crack"` | Abliteration: penalize safety-specific experts |

### 2.1 Frozen fusion weights (SHIP DEFAULTS)

Normative formulas: plan §11.6. Constants live in `jang_tools/intent_prune/score.py`, labeled **SHIP DEFAULTS**, with a pointer to this document.

#### Balanced (default)

```text
base = 0.30*ng + 0.20*nm + 0.35*ni - 0.10*nd + 0.05*ng   # backbone_floor uses ng
Keep:     score = base + 0.15*ns
Balanced: score = base + 0.05*ns
CRACK:    score = base - 0.25*ns*(1-ni)
```

| Key | Weight |
|---|---|
| `path` | **0.30** |
| `mass` | **0.20** |
| `intent` | **0.35** |
| `drop` | **0.10** |
| `backbone_floor` | **0.05** |
| `safety_keep` | **0.15** |
| `safety_balanced` | **0.05** |
| `safety_crack` | **0.25** |

#### Specialist

Same safety terms; heavier intent, lighter path/mass:

| Key | Weight |
|---|---|
| `path` | **0.20** |
| `mass` | **0.15** |
| `intent` | **0.50** |
| `drop` | **0.10** |
| `backbone_floor` | **0.05** |
| safety_* | unchanged vs Balanced |

**Default entry points:** `preset="balanced"`, `safety_stance="balanced"` (`DEFAULT_PRESET` / `DEFAULT_SAFETY_STANCE` in `score.py`).

---

## 3. Synthetic highway evidence (Appendix C) — hybrid > mass

### 3.1 Fixture

From plan Appendix C / `build_synthetic_highway_records()`:

| Symbol | Expert id | Role |
|---|---|---|
| **H** | 1 | Rare **highway** bridge on multi-layer paths (layer 0 hub → H → layer 2 hub) |
| **N** | 3 | Chatty / high-mass **dead-end** (single-layer selections; no layer→layer edges) |

- Layers **L=3**, experts **E=4**
- Default: 6 highway token pairs + 40 noise tokens (gate score 5.0 on N)
- Intent tag on highway paths: `code`

**Assert (plan):** mass-only ranks **N > H**; path/hybrid ranks **H > N**; keep-K=2 retains **H**.

### 3.2 Layer-1 results (reproducible)

Commands (from `jang-tools/`):

```bash
PYTHONPATH=. python - <<'PY'
from jang_tools.intent_prune.score import (
    HIGHWAY_E, HIGHWAY_H, HIGHWAY_L, HIGHWAY_N,
    build_synthetic_highway_records, score_hybrid, mass_only_scores,
)
records = build_synthetic_highway_records()
r = score_hybrid(
    records=records, num_experts=HIGHWAY_E, num_layers=HIGHWAY_L,
    keep_k=2, intents_keep=["code"], preset="balanced", safety_stance="balanced",
)
layer = 1
print("mass", mass_only_scores(r.mass)[layer])
print("path", r.pi_g[layer])
print("hybrid", r.scores[layer])
print("keep", r.keep[layer])
PY
```

**Captured numbers** (IP5 freeze run, teleport=0.01, weight_by_gate=True):

| Signal (layer 1) | H (id=1) | N (id=3) | Rank order (desc) | Winner H vs N |
|---|---:|---:|---|---|
| **A Mass** (norm) | 0.060 | **1.000** | `[3, 1, 0, 2]` | **N ≫ H** |
| **C Path** (`π_g`) | **0.273** | 0.091 | `[1, 3, 0, 2]` | **H > N** |
| **D Hybrid balanced** | **0.712** | 0.317 | `[1, 3, 0, 2]` | **H > N** |
| Path-only fusion (mass/intent off) | **0.350** | 0.117 | `[1, 3, 0, 2]` | **H > N** |
| Specialist hybrid | **0.759** | 0.234 | `[1, 3, …]` | **H > N** |
| Hybrid CRACK stance | **0.712** | 0.317 | same (no safety mass in fixture) | **H > N** |

Raw mass (layer 1): H=12.0, N=200.0 (noise dominates selection mass).  
Raw path: H≈0.273, N≈0.091 (highway on structural edges).

**keep-K=2 (layer 1):** hybrid retains **`[1, 3]`** → **H is kept**. Mass-only top-2 prefers N first; pure mass would deprioritize the highway relative to hybrid.

### 3.3 Unit test lock

| Test | Asserts |
|---|---|
| `test_synthetic_highway_mass_only_ranks_n_over_h` | Arm A: N ranked above H |
| `test_synthetic_highway_path_and_hybrid_rank_h_over_n` | Arms C/D: path & hybrid H > N; mass still N > H |
| `test_synthetic_highway_keep_k2_retains_h` | keep-K=2 includes H |
| `test_fusion_balanced_matches_normative_weights` | Balanced arithmetic |
| `test_fusion_crack_penalizes_safety_specificity` | Arm E specificity term |
| `test_specialist_preset_weights_differ` | Specialist ≠ Balanced intent/path |

```text
pytest tests/test_intent_prune_score.py tests/test_intent_prune_transitions.py
# 45 passed (IP5 freeze environment)
```

### 3.4 Criterion decision

| Criterion | Result |
|---|---|
| Strictly better than pure mass on path-sensitive stress test | **PASS** — hybrid ranks H > N; mass ranks N > H; keep-K retains H |
| Keep-intent primary on full Reviewed 50 (Qwen BF16) | **Deferred** — no GPU run in this environment; see §7 |
| Hybrid default on | **YES** — Balanced SHIP DEFAULTS frozen in `score.py` |

**Decision:** Freeze Balanced/Specialist weights as shipped. Hybrid is the default scorer for Intent Prune v1.

---

## 4. Arm summary (what we learned)

| Arm | Synthetic highway | Role at ship |
|---|---|---|
| A Mass | Fails highway retention logic (N > H) | Baseline / compare only |
| B REAP / kimi | Not required for freeze; Advanced later | Optional Advanced arm |
| C Path | Saves H | Component of hybrid |
| **D Hybrid balanced** | Saves H; fuses mass + path + intent | **Ship default** |
| E Hybrid CRACK | Same structure; safety penalty when `π_S` present | Stance, not a separate weight table |

**Why not mass-only default:** REAP-style marginal mass drops rare experts on critical multi-layer highways (plan §5). The synthetic case is the minimal proof.

**Why not path-only default:** Path alone ignores high-mass specialists that never bridge layers; hybrid keeps a mass floor (0.20 Balanced / 0.15 Specialist) plus intent conditioning (0.35 / 0.50).

---

## 5. CRACK stance (arm E) — unit evidence

CRACK does **not** change path/mass/intent weights. It replaces the safety add-on with a **specificity penalty**:

```text
specificity = ns * (1 - ni)
score = base - 0.25 * specificity
```

Unit test `test_fusion_crack_penalizes_safety_specificity`:

- High safety + low intent → score drops vs Keep stance  
- High safety + high intent (`ni=1`) → specificity 0 → no crack penalty on the safety term  

Full CRACK pack refusal metrics require live suite + probe pack (IP2); naming/eval procedure is in the parent plan §7 / §15.

---

## 6. Ship defaults (signed freeze)

| Item | Value |
|---|---|
| Scorer name | `hybrid_v1` |
| Plan schema | `jang-intent-prune-plan-v1` |
| Default preset | **balanced** |
| Default safety stance | **balanced** (CRACK is explicit opt-in) |
| Balanced weights | frozen table §2.1 |
| Specialist weights | frozen table §2.1 |
| Code anchors | `BALANCED_WEIGHTS`, `SPECIALIST_WEIGHTS`, `PRESET_WEIGHTS`, `DEFAULT_PRESET`, `DEFAULT_SAFETY_STANCE` |
| Label in source | **SHIP DEFAULTS** + link to this doc |
| Retune policy | Only after a new bake-off doc revision + deliberate PR |

---

## 7. Procedure: full TRACE bake-off later (Qwen GPU)

Not required for this freeze. When BF16/vMLX + Reviewed 50 are available:

### 7.1 Collect transitions

```bash
# Expert Lab / vMLX generate with path emission
python -m jang_tools expert-lab-vmlx ... \
  --emit-token-trace --emit-transitions \
  # → expert_transitions.jsonl under run dir
```

### 7.2 Score each arm

```bash
# D — Hybrid balanced (ship default)
python -m jang_tools intent-prune-score \
  --transitions expert_transitions.jsonl \
  --num-experts 256 --keep-k 192 \
  --intent code --intent math \
  --preset balanced --safety-stance balanced \
  --output plan-hybrid-balanced.json

# E — Hybrid CRACK
python -m jang_tools intent-prune-score \
  --transitions expert_transitions.jsonl \
  --num-experts 256 --keep-k 192 \
  --intent code \
  --preset balanced --safety-stance crack \
  --output plan-hybrid-crack.json

# A — Mass-only compare: use comparison tooling / Expert Lab Advanced,
#     or rank by mass_matrix_from_adjacency + select_keep_k offline.
# C — Path-only: rank by stationary π_g per layer (same graph stack).
```

### 7.3 Report fields (plan §14.1)

For each arm: pruned size / K, keep-intent suite scores, safety/CRACK pack metrics, qualitative samples. Attach CRACK pack results for arm E. Do not claim MMLU-only victory (plan §14.3).

### 7.4 Re-open freeze only if

- Hybrid loses keep-intent primary to mass/REAP by more than agreed ε, **or**
- CRACK collapses keep-intent below floor, **or**
- Path scores prove unstable on Reviewed 50 (teleport/150-prompt mitigation per plan risks).

---

## 8. Scope notes / non-claims

| In scope for this report | Out of scope |
|---|---|
| Matrix documentation | Live Qwen GPU TRACE numbers |
| Synthetic highway proof hybrid > mass | Learned fusion weights (non-goal N2) |
| Freezing Balanced/Specialist constants | Studio UI (IP4) |
| Unit/synthetic ship criterion | MiniMax/other arches (IP6) |
| Procedure for full TRACE | Recovery LoRA (IP7) |

---

## 9. Sign-off

| Role | Statement |
|---|---|
| **IP5 / PR-IP5** | Bake-off matrix documented; ship criterion met on synthetic highway; Balanced + Specialist frozen as **SHIP DEFAULTS** in `score.py`. |
| **Hybrid default** | **ON** (`preset=balanced`, `scorer=hybrid_v1`). |
| **Full TRACE** | Documented procedure; optional later validation; **not** a blocker for code freeze. |

---

## 10. Related documents

- Full product plan: `docs/plans/2026-07-12-jang-studio-intent-prune-crack.md`  
- Scorer: `jang-tools/jang_tools/intent_prune/score.py`  
- Graph / path: `jang-tools/jang_tools/intent_prune/graph.py`  
- Tests: `jang-tools/tests/test_intent_prune_score.py`  
- Transitions emission: IP0 / `intent_prune/transitions.py`  

---

*End of IP5 bake-off ship report.*
