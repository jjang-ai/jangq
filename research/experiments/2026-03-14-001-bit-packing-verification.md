# Experiment 001: Bit Packing Roundtrip Verification

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PASS

## Hypothesis

The MXQ bit packing engine (`pack.py`) correctly packs and unpacks integer values
at all supported bit widths (2, 3, 4, 5, 6, 8) without data loss, including edge
cases like cross-byte boundaries for non-byte-aligned widths (3, 5, 6).

## Method

- **Test framework**: pytest 9.0.2, Python 3.14.2, macOS (Apple Silicon)
- **Test categories**:
  1. Roundtrip tests: pack then unpack random values at each bit width (block_size=64)
  2. Size verification: packed byte count matches theoretical `ceil(n * bits / 8)`
  3. Known-value tests: 2-bit and 4-bit with hand-computed expected byte values
  4. Scale tests: 100 blocks (6,400 values) at each bit width
  5. Edge cases: all-zero arrays, all-max-value arrays

- **Bit widths tested**: 2, 3, 4, 5, 6, 8
- **Block sizes**: 64 (default MXQ block size)
- **RNG seeds**: 42, 123, 999 (reproducible)

## Results

```
28 tests, 28 passed, 0 failed, 0 errors
Total time: 1.33 seconds
```

### Per-test results:

| Test | Bit widths | Status |
|------|-----------|--------|
| test_roundtrip | 2,3,4,5,6,8 | 6/6 PASS |
| test_packed_size | 2,3,4,5,6,8 | 6/6 PASS |
| test_2bit_specific | 2 | PASS — bytes match hand-computed 0xE4 |
| test_4bit_specific | 4 | PASS — bytes match hand-computed 0xA5, 0xF3 |
| test_block_roundtrip | 2,3,4,5,6,8 | 6/6 PASS |
| test_large_roundtrip | 2,3,4,5,6,8 | 6/6 PASS (6400 values each) |
| test_max_values | all | PASS |
| test_zero_values | all | PASS |

### Packed sizes verified:

| Bit width | Values | Expected bytes | Actual bytes | Match |
|-----------|--------|----------------|-------------|-------|
| 2 | 64 | 16 | 16 | YES |
| 3 | 64 | 24 | 24 | YES |
| 4 | 64 | 32 | 32 | YES |
| 5 | 64 | 40 | 40 | YES |
| 6 | 64 | 48 | 48 | YES |
| 8 | 64 | 64 | 64 | YES |

## Implementation notes

- 2-bit and 4-bit use fast paths (direct bit shifting, no loop)
- 3-bit, 5-bit, 6-bit use general case (per-value bit offset calculation)
- Cross-byte boundary handling uses uint16 read for safe extraction
- Pack uses uint64 intermediate to avoid overflow during shifts

## Analysis

All bit widths produce exact roundtrip results. The packing engine handles
cross-byte boundaries correctly for non-byte-aligned widths (3, 5, 6).
This is critical for MXQ because variable bit-width blocks will have
different block sizes in bytes, and the dequantization kernel must correctly
extract values that span byte boundaries.

## Conclusions

The bit packing engine is correct and ready for use in the quantization pipeline.
No data loss at any bit width. Fast paths for 2-bit and 4-bit provide optimization
for the most common MXQ bit widths.
