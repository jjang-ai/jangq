"""M113 (iter 14 observation, closed iter 105): coarse count invariant for
`except Exception` sites across jang_tools/.

Companion to JANG Studio's iter-104 M108 try? count-threshold test. Both
enforce "bulk additions of a potentially-silent-swallow pattern trigger
review" without blocking routine work.

**Taxonomy of current 171 sites (2026-05-11 classification):**

1. **Optional imports** — `try: import X ... except Exception: X = None`
   — fall back gracefully when mlx / torch / numpy not available.

2. **Tensor conversion retries** — try primary quantization / dtype-conversion
   path; on failure fall back to slower but more-tolerant path
   (convert.py, convert_*_jangtq.py, calibrate_fp8.py). The re-raise sites
   add diagnostic context so the final error surfaced to the user is
   actionable.

3. **Best-effort parse** — read optional config / header / metadata; if
   parse fails, continue with defaults (inspect_source._sniff_dtype,
   estimate_model._source_bytes_per_weight).

4. **Error wrapping with context** — catch Exception, add "while processing
   file X" context, re-raise as a more specific type. Used throughout
   loader.py and modelcard.py to turn generic Python errors into
   jang-specific typed errors that iter-90 M167's actionable-diagnostic
   rule can pattern-match.

5. **CLI top-level catch** — a few `__main__`-level blocks that catch
   Exception for a clean CLI error output (no traceback to the user).

**The BAD pattern is `except Exception: pass` with no logging and no
re-raise.** Iter-35 M107's Swift class applies here too — silent swallows
in user-action paths. The precise no-bare-pass test below must stay zero;
the count threshold is intentionally coarse and exists only to force review
when broad catches grow materially.

**When this test fails:**
1. Engineer adding new `except Exception` should classify each addition
   per the 5 categories above.
2. If all fit, bump the threshold by a small amount.
3. If any is `except Exception: pass` with no log / no re-raise, fix it
   with either a specific error type OR a stderr log + re-raise.
4. Update category counts in this module's docstring.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "jang_tools"


def _count_except_exception_sites() -> int:
    """Count `except Exception` occurrences in .py files (excluding .pyc
    caches and test files which are expected to use broad catches freely)."""
    total = 0
    for py in SRC_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        content = py.read_text(encoding="utf-8", errors="replace")
        # Match `except Exception` as whole word to avoid matching comments
        # like "except Exceptional" or sub-words.
        matches = re.findall(r"\bexcept\s+Exception\b", content)
        total += len(matches)
    return total


def test_except_exception_site_count_within_threshold():
    """Coarse count invariant. Today's count is 195; threshold is 205 (10
    headroom for routine additions). Bulk additions trigger review per
    the taxonomy in the module docstring above. ZAYA runtime port +
    Hy3/ZAYA1-VL converters added best-effort-parse and optional-import
    sites that fit the taxonomy; the Nex/N2 expert-prune + split-quant
    convert path, the MiniMax-M3 converter/probe, and the multimodal
    capability resolver added further tensor-conversion-retry / optional-
    import / best-effort-parse sites (all audited, none silent)."""
    total = _count_except_exception_sites()
    assert total <= 205, (
        f"except Exception site count ({total}) exceeds threshold of 205 — "
        f"audit new additions per the 5-category taxonomy in this module's "
        f"docstring (M113 iter 105). If all additions fit one of: optional-"
        f"import / tensor-conversion-retry / best-effort-parse / error-"
        f"wrapping-with-context / CLI-top-level — bump the threshold. "
        f"If any new site is `except Exception: pass` with no log/re-raise, "
        f"fix with iter-35 M107 / iter-90 M167's actionable-diagnostic pattern."
    )


def test_no_bare_except_exception_pass():
    """Specific anti-pattern: `except Exception:\\n    pass` with no other
    body — silent swallow, no log, no re-raise. iter-35 M107's Swift-side
    class applied to Python. Iter-105's precise invariant (vs the coarse
    count above).

    Regex matches: `except Exception` optionally with `as <name>`, colon,
    whitespace (including newlines), the literal `pass` keyword, and
    nothing else before the next non-indented line.
    """
    offenders: list[tuple[str, int]] = []
    # Match: except Exception[ as x]:\n    pass  (where pass is the ONLY body)
    pattern = re.compile(
        r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n(\s+)pass\s*\n(?!\1)",
        re.MULTILINE,
    )
    for py in SRC_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        content = py.read_text(encoding="utf-8", errors="replace")
        for m in pattern.finditer(content):
            line_no = content[: m.start()].count("\n") + 1
            offenders.append((str(py.relative_to(REPO_ROOT)), line_no))
    # Explicit allowlist for legitimate best-effort operations where failure
    # is benign (optimization cache clear, optional lookup, last-resort
    # inference). The test catches NEW silent-swallows; these existing
    # sites were audited in iter-105 and classified as acceptable.
    #
    # If a listed site is refactored / line numbers shift, update the tuple.
    # If a new site needs the allowlist, ADD RATIONALE BELOW — don't just
    # grow the set silently.
    allowed: set[tuple[str, int]] = {
        # mx.clear_cache() — Metal cache is an optimization; convert
        # succeeds either way. Exception means MLX unavailable or
        # mid-teardown; safe to continue.
        ("jang_tools/convert.py", 780),
        ("jang_tools/convert_mxtq_to_jang.py", 369),
        # f.get_tensor(scale_key) — probing for optional `_scale_inv`
        # tensor for FP8 scaling. Missing → scale_inv stays None → caller
        # handles the None path. Exception-as-lookup is idiomatic here.
        ("jang_tools/calibrate.py", 146),
        # Last-resort bit-width inference fallback. If the heuristic
        # raises, keep the already-configured bits. No loss of signal.
        ("jang_tools/loader.py", 1568),
        # MiniMax codec-fallback chain: try FP8 codec, then numpy, then the
        # final _load_bf16_tensor (which RAISES on real failure). The two
        # intermediate passes just advance to the next codec — a genuine
        # failure still propagates from the last attempt, so nothing is
        # silently swallowed. (tensor-conversion-retry taxonomy.)
        ("jang_tools/convert_minimax_jang.py", 85),
        ("jang_tools/convert_minimax_jang.py", 93),
        # mx.clear_cache() — Metal cache is an optimization; convert succeeds
        # either way. Same rationale as the convert.py / convert_mxtq entries.
        ("jang_tools/mimo_v2/convert_jang.py", 756),
    }
    remaining = [o for o in offenders if o not in allowed]
    assert not remaining, (
        f"Found {len(remaining)} bare `except Exception: pass` sites — these "
        f"silently swallow errors with no log / re-raise. Fix with either a "
        f"specific error type or stderr log + context (iter-35 M107 / iter-90 "
        f"M167 pattern):\n" + "\n".join(f"  {f}:{l}" for f, l in remaining)
    )
