# Audit: 05-apple-metal-gpu-computing.md

Fact-check performed 2026-03-14 against publicly available Apple specifications,
Apple Developer documentation, and Metal Shading Language Specification.

---

## 1. Apple Silicon GPU Specs Table (Section 1.1)

### GPU Core Counts

| Chip | Document Claims | Verified | Verdict |
|------|----------------|----------|---------|
| M1 | 7-8 | 7-8 | CORRECT |
| M1 Pro | 14-16 | 14-16 | CORRECT |
| M1 Max | 24-32 | 24-32 | CORRECT |
| M1 Ultra | 48-64 | 48-64 | CORRECT |
| M2 | 8-10 | 8-10 | CORRECT |
| M2 Pro | 16-19 | 16-19 | CORRECT |
| M2 Max | 30-38 | 30-38 | CORRECT |
| M2 Ultra | 60-76 | 60-76 | CORRECT |
| M3 | 8-10 | 8-10 | CORRECT |
| M3 Pro | 11-18 | 14-18 | [INCORRECT] |
| M3 Max | 30-40 | 30-40 | CORRECT |
| M3 Ultra | 60-80 | 60-80 | CORRECT |
| M4 | 10 | 8-10 | [INCORRECT] |
| M4 Pro | 16-20 | 16-20 | CORRECT |
| M4 Max | 32-40 | 32-40 | CORRECT |
| M4 Ultra | 64-80 | Does not exist | [INCORRECT] |

**M3 Pro GPU cores**: The document says "11-18". The 11 is the CPU core count
of the lower-tier M3 Pro, not the GPU core count. The M3 Pro GPU configurations
are 14-core and 18-core. Correct range is **14-18**.

**M4 base GPU cores**: The document says "10" but the M4 is available with
either 8-core or 10-core GPU (the 8-core version ships in base MacBook Air M4).
Correct range is **8-10**.

**M4 Ultra**: [INCORRECT] As of March 2026, Apple has not released an M4 Ultra
chip. Apple has stated that not every generation will have an Ultra variant, and
the M4 Max lacks an UltraFusion connector. The document's entire M4 Ultra row
(64-80 cores, 36.0/72.0 TFLOPS, 819 GB/s, 512 GB) is speculative/fictional.
Apple instead shipped the M3 Ultra alongside M4 Max in the 2025 Mac Studio.
Every claim in the document that references M4 Ultra specifications must be
treated as unverified speculation.

### Peak FP32 TFLOPS

| Chip | Document Claims | Verified | Verdict |
|------|----------------|----------|---------|
| M1 | 2.6 | 2.6 | CORRECT |
| M1 Pro | 5.2 | ~4.5 (14-core) / 5.2 (16-core) | CORRECT (for max config) |
| M1 Max | 10.4 | ~7.8 (24-core) / 10.4 (32-core) | CORRECT (for max config) |
| M1 Ultra | 20.8 | ~15.9 (48-core) / ~21 (64-core) | [UNCERTAIN] |
| M2 | 3.6 | 3.6 | CORRECT |
| M2 Pro | 6.8 | [UNVERIFIED] | Apple did not officially publish this number. Likely derived from per-core scaling but not confirmed. |
| M2 Max | 13.6 | 13.6 | CORRECT |
| M2 Ultra | 27.2 | 27.2 | CORRECT |
| M3 | 4.1 | ~3.6 (from third-party benchmarks) | [INCORRECT] |
| M3 Pro | 7.4 | [UNVERIFIED] | Apple stopped publishing TFLOPS for M3 generation. |
| M3 Max | 16.4 | [UNVERIFIED] | Apple stopped publishing TFLOPS for M3 generation. |
| M3 Ultra | 32.8 | [UNVERIFIED] | Apple stopped publishing TFLOPS for M3 generation. |
| M4 | 4.6 | ~4.3 (third-party sources) | [UNCERTAIN] |
| M4 Pro | 8.3 | [UNVERIFIED] | Not officially published. |
| M4 Max | 18.0 | ~18.4 (third-party sources) | [UNCERTAIN] |
| M4 Ultra | 36.0 | N/A -- chip does not exist | [INCORRECT] |

**M1 Ultra FP32**: The document claims 20.8 TFLOPS. Third-party benchmarks show
~21.2 TFLOPS for the 64-core variant and ~15.9 for the 48-core. Apple's own
press release said "up to 21 TFLOPS". 20.8 appears to be derived by doubling
M1 Max's 10.4 (mathematically clean but slightly off from the measured ~21).
Close enough for a research document but not exactly what Apple stated.

**M3 FP32**: The document claims 4.1 TFLOPS. Apple did not publish TFLOPS for
M3 GPUs. Third-party sources suggest the M3 10-core GPU achieves ~3.6 TFLOPS
FP32 (NotebookCheck, cpu-monkey). 4.1 GHz is the M3's CPU clock speed, which
may have been confused with TFLOPS. The M3 Pro, M3 Max, and M3 Ultra TFLOPS
figures are similarly unverified -- Apple stopped publishing these numbers
starting with M3.

**M4 FP32**: The document claims 4.6 TFLOPS. Third-party sources show ~4.3
TFLOPS. This is uncertain -- the discrepancy could be due to different
measurement methods.

### Peak FP16 TFLOPS

The document assumes FP16 TFLOPS = 2x FP32 TFLOPS throughout. [UNCERTAIN]
This is a reasonable assumption for M1/M2 architecture (which has 2:1 FP16:FP32
throughput ratio), but for M3/M4 (Apple Family 9), the GPU can issue FP32 and
FP16 instructions simultaneously to different datapaths, making the actual FP16
TFLOPS potentially higher than a simple 2x multiple. The 2x ratio is a
conservative lower bound and likely approximately correct for sustained
workloads, but the document should note this is an assumption, not an Apple-published
specification.

### Memory Bandwidth

| Chip | Document Claims | Verified | Verdict |
|------|----------------|----------|---------|
| M1 | 68.25 GB/s | 68.25 GB/s | CORRECT |
| M1 Pro | 200 GB/s | 200 GB/s | CORRECT |
| M1 Max | 400 GB/s | 400 GB/s | CORRECT |
| M1 Ultra | 800 GB/s | 800 GB/s | CORRECT |
| M2 | 100 GB/s | 100 GB/s | CORRECT |
| M2 Pro | 200 GB/s | 200 GB/s | CORRECT |
| M2 Max | 400 GB/s | 400 GB/s | CORRECT |
| M2 Ultra | 800 GB/s | 800 GB/s | CORRECT |
| M3 | 100 GB/s | 100 GB/s | CORRECT |
| M3 Pro | 150 GB/s | 150 GB/s | CORRECT |
| M3 Max | 400 GB/s | 300 GB/s (30-core) / 400 GB/s (40-core) | [INCOMPLETE] |
| M3 Ultra | 800 GB/s | 819 GB/s | [INCORRECT] |
| M4 | 120 GB/s | 120 GB/s | CORRECT |
| M4 Pro | 273 GB/s | 273 GB/s | CORRECT |
| M4 Max | 546 GB/s | 546 GB/s (40-core) / ~410 GB/s (32-core) | [INCOMPLETE] |
| M4 Ultra | 819 GB/s | N/A -- chip does not exist | [INCORRECT] |

**M3 Max bandwidth**: The 30-core M3 Max has 300 GB/s, while the 40-core has
400 GB/s. The document lists only 400 GB/s. This omission could mislead readers
who have the lower-tier M3 Max.

**M3 Ultra bandwidth**: The document says 800 GB/s. Apple's specification is
819 GB/s (matching the M4 Max's stated bandwidth pattern). This is wrong by
~2.4%.

**M4 Max bandwidth**: Similar to M3 Max, the 32-core binned M4 Max has ~410
GB/s, not 546 GB/s. Only the 40-core variant hits 546 GB/s.

### Max Unified RAM

| Chip | Document Claims | Verified | Verdict |
|------|----------------|----------|---------|
| M1 | 16 GB | 16 GB | CORRECT |
| M1 Pro | 32 GB | 32 GB | CORRECT |
| M1 Max | 64 GB | 64 GB | CORRECT |
| M1 Ultra | 128 GB | 128 GB | CORRECT |
| M2 | 24 GB | 24 GB | CORRECT |
| M2 Pro | 32 GB | 32 GB | CORRECT |
| M2 Max | 96 GB | 96 GB | CORRECT |
| M2 Ultra | 192 GB | 192 GB | CORRECT |
| M3 | 36 GB | 24 GB | [INCORRECT] |
| M3 Pro | 36 GB | 36 GB | CORRECT |
| M3 Max | 128 GB | 128 GB | CORRECT |
| M3 Ultra | 192 GB | 512 GB | [INCORRECT] |
| M4 | 32 GB | 32 GB | CORRECT |
| M4 Pro | 48 GB | 64 GB | [INCORRECT] |
| M4 Max | 128 GB | 128 GB | CORRECT |
| M4 Ultra | 512 GB | N/A -- chip does not exist | [INCORRECT] |

**M3 base max RAM**: The document says 36 GB. The base M3 supports up to 24 GB.
36 GB is the max for the M3 Pro. This is a clear error.

**M3 Ultra max RAM**: The document says 192 GB. The M3 Ultra supports up to
512 GB. 192 GB was the M2 Ultra's max. This is wrong.

**M4 Pro max RAM**: The document says 48 GB. The M4 Pro supports up to 64 GB
(available in Mac Mini and MacBook Pro configurations). 48 GB is a common
configuration but not the maximum.

---

## 2. Metal API Names (Section 2)

All core API names verified correct:
- `MTLDevice` -- CORRECT
- `MTLCommandQueue` -- CORRECT
- `MTLCommandBuffer` -- CORRECT
- `MTLComputeCommandEncoder` -- CORRECT
- `MTLComputePipelineState` -- CORRECT
- `MTLBuffer` -- CORRECT
- `MTLEvent` -- CORRECT
- `MTLFence` -- CORRECT
- `MTLBinaryArchiveDescriptor` -- CORRECT
- `MTLCreateSystemDefaultDevice()` -- CORRECT
- `makeCommandQueue()` -- CORRECT
- `makeLibrary(source:options:)` -- CORRECT
- `makeDefaultLibrary()` -- CORRECT
- `makeFunction(name:)` -- CORRECT
- `makeComputePipelineState(function:)` -- CORRECT
- `setComputePipelineState()` -- CORRECT
- `setBuffer(_:offset:index:)` -- CORRECT
- `setBytes(_:length:index:)` -- CORRECT
- `dispatchThreadgroups(_:threadsPerThreadgroup:)` -- CORRECT
- `dispatchThreads(_:threadsPerThreadgroup:)` -- CORRECT
- `endEncoding()` -- CORRECT
- `commit()` -- CORRECT
- `waitUntilCompleted()` -- CORRECT
- `addCompletedHandler()` -- CORRECT
- `memoryBarrier(scope:)` -- CORRECT
- `memoryBarrier(resources:)` -- CORRECT
- `gpuStartTime` / `gpuEndTime` -- CORRECT
- `maxTotalThreadsPerThreadgroup` -- CORRECT
- `threadExecutionWidth` -- CORRECT
- `hasUnifiedMemory` -- CORRECT
- `maxBufferLength` -- CORRECT
- `maxThreadgroupMemoryLength` -- CORRECT
- `maxThreadsPerThreadgroup` -- CORRECT
- `supportsFamily(.apple8)` / `supportsFamily(.apple9)` -- CORRECT
- `storageModeShared` / `storageModePrivate` / `storageModeManaged` -- CORRECT

### Minor API issues

**Section 2.5 MTLEvent usage** (line 383-386): The document shows
`encoder1.encodeSignalEvent(event, value: 1)` and
`encoder2.encodeWaitForEvent(event, value: 1)`. [UNCERTAIN] These are
`MTLComputeCommandEncoder` methods. The actual method names may be
`encodeSignalEvent(_:value:)` and `encodeWaitForEvent(_:value:)`. The syntax
shown is plausible but should be verified against current API docs -- the exact
Swift method names for event signaling on compute encoders have evolved across
macOS versions.

**Section 2.5 MTLFence usage** (line 392-394): `encoder1.updateFence(fence)` and
`encoder2.waitForFence(fence)` -- CORRECT for compute command encoders.

**Section 7.1 maxBufferLength** (line 1716): Claims "Typically 256 GB on Apple
Silicon." [UNVERIFIED] The actual value depends on the device and OS version.
This is a runtime-queryable property and the exact value is not well-documented
publicly. 256 GB seems plausible but cannot be confirmed.

---

## 3. Metal Shading Language Syntax (Section 3)

### Kernel function signature (Section 3.1)

All attribute names verified correct:
- `[[buffer(N)]]` -- CORRECT
- `[[thread_position_in_grid]]` -- CORRECT
- `[[threadgroup_position_in_grid]]` -- CORRECT
- `[[thread_position_in_threadgroup]]` -- CORRECT
- `[[thread_index_in_simdgroup]]` -- CORRECT
- `[[simdgroup_index_in_threadgroup]]` -- CORRECT
- `[[threads_per_grid]]` -- CORRECT
- `[[threads_per_threadgroup]]` -- CORRECT

### Address spaces (Section 3.2)

- `device`, `threadgroup`, `constant`, `thread` -- CORRECT

### SIMD group operations (Section 3.4)

- `simd_sum()` -- CORRECT
- `simd_max()` -- CORRECT
- `simd_min()` -- CORRECT
- `simd_prefix_exclusive_sum()` -- CORRECT
- `simd_shuffle()` -- CORRECT
- `simd_shuffle_down()` -- CORRECT
- `simd_shuffle_up()` -- CORRECT
- `simd_shuffle_xor()` -- CORRECT
- `simd_broadcast()` -- CORRECT
- `simd_all()` -- CORRECT
- `simd_any()` -- CORRECT

**`simd_ballot()` and `simd_vote::vote_t`** (line 580): [UNCERTAIN] Metal does
support ballot-like operations, but the exact type name `simd_vote::vote_t` and
function name `simd_ballot()` could not be definitively confirmed from publicly
available documentation. The Metal Shading Language Specification PDF documents
SIMD vote/ballot operations, but the exact syntax may differ. The document should
note this is based on a specific MSL specification version and verify against
the latest spec.

### Data types (Section 3.6)

- `half` -- CORRECT
- `float` -- CORRECT
- `bfloat` -- CORRECT (supported in MSL)
- `half4`, `float4`, `uint4` vector types -- CORRECT
- `packed_half4`, `packed_float4` -- CORRECT
- Integer types (`uint8_t`, `int8_t`, etc.) -- CORRECT

**`bfloat` literal suffix `1.0bf`** (line 653): [UNCERTAIN] The MSL spec
defines the `bfloat` type, but the literal suffix `bf` for bfloat constants
could not be independently verified. The standard MSL approach may require
explicit cast: `bfloat(1.0)` rather than `1.0bf`. This should be verified
against the MSL 3.1+ specification.

**Claim "M3 added native bfloat16 support"** (line 153): [INCORRECT] The
document claims bfloat16 was added with M3. However, bfloat16 support was
announced at WWDC 2023 and is available from Apple GPU family 7 (which includes
M1 and A14). Some sources indicate Apple6 family already supports bfloat.
The M3 did not "add" bfloat16 -- it was available earlier. What M3 added was
the GPU family 9 architecture with dynamic caching. The bfloat16 support in
Metal predates M3.

### Threadgroup memory and barriers (Section 3.5)

- `threadgroup_barrier(mem_flags::mem_threadgroup)` -- CORRECT
- `mem_flags::mem_device` -- CORRECT
- `mem_flags::mem_texture` -- CORRECT
- `mem_flags::mem_threadgroup_imageblock` -- CORRECT

### Atomic operations (Section 3.8)

- `atomic_uint`, `atomic_fetch_add_explicit` -- CORRECT
- `memory_order_relaxed` -- CORRECT

**"Atomic float add (Metal 3.0+ / Apple GPU family 8+)"** (line 762): [INCORRECT]
According to Apple Developer Forums and documentation, Metal does NOT natively
support atomic float operations. The workaround is to use atomic integer
operations with float-to-int reinterpretation (compare-and-swap loop) or to
scale floats to integers. The type `atomic_float` does not exist in standard
MSL. This claim is wrong.

### Math functions (Section 3.9)

- `fma()` -- CORRECT
- `precise::fma()`, `precise::sqrt()` -- CORRECT
- Standard math functions (`sin`, `exp`, `log`, `rsqrt`, etc.) -- CORRECT
- SiLU/GELU formulas -- CORRECT (standard approximations)

---

## 4. simdgroup_matrix Operations (Section 4.4)

### Types and sizes

- `simdgroup_half8x8` -- CORRECT (8x8 matrix of half, Apple GPU family 7+)
- `simdgroup_float8x8` -- CORRECT (8x8 matrix of float, Apple GPU family 7+)
- `simdgroup_bfloat8x8` -- [UNCERTAIN] The document claims this type exists
  for bf16 input on M3+. Metal does support bfloat in simdgroup_matrix
  operations, but the exact type name `simdgroup_bfloat8x8` should be verified
  against the MSL specification. It may be correct but I cannot confirm the
  exact type name.

### Size support

The document states 8x8 as the matrix size. Apple's Metal documentation
confirms 8x8 is the supported size. The MSL spec also mentions 4x4 matrices
are available. The document only discusses 8x8 which is the primary size for
compute workloads -- this is fine but incomplete.

### Functions

- `simdgroup_load()` -- CORRECT
- `simdgroup_store()` -- CORRECT
- `simdgroup_multiply_accumulate()` -- CORRECT
- `#include <metal_simdgroup_matrix>` -- CORRECT

### GPU family support

The document claims simdgroup_matrix is available on "all Apple Silicon" (lines
1010-1012). This is CORRECT -- simdgroup_matrix was introduced with Apple GPU
family 7 (A14/M1), which covers all M-series chips.

---

## 5. Memory Bandwidth Numbers

The memory bandwidth table analysis is covered in Section 1 above. Summary of
errors:

- M3 Max: Missing 300 GB/s tier for 30-core variant
- M3 Ultra: Claims 800 GB/s, actual is 819 GB/s
- M4 Max: Missing ~410 GB/s tier for 32-core variant
- All M4 Ultra numbers: Chip does not exist

The token generation rate calculations in Section 1.5.4 and Section 8.2 use the
formula `tokens/s = bandwidth / model_size`. This formula is correct in
principle. The "effective bandwidth ~80%" assumption is reasonable. The
calculated rates are internally consistent with the (sometimes incorrect)
bandwidth numbers used.

---

## 6. Metal 3 vs Metal 4 Features (Section 1.7)

**Claim: M3 = "A17 Pro GPU architecture"** (line 149): [UNCERTAIN] The M3 and
A17 Pro share the same GPU architecture generation, but saying M3 IS the
"A17 Pro GPU architecture" is imprecise. They are both Apple GPU family 9 (or
Apple8, depending on the feature level). The M3 is not the A17 Pro; they are
separate chips that share a GPU architecture.

**Claim: "Dynamic caching" on M3** (line 150): CORRECT. This is a well-documented
M3/A17 Pro feature.

**Claim: "BFloat16 support: M3 added native bfloat16"** (line 153): [INCORRECT]
As noted above, bfloat16 support predates M3. It was available from Apple GPU
family 7 (M1/A14) at the API level, though hardware acceleration quality may
have improved with M3.

**Claim: "simdgroup_matrix operations support bf16 on M3+"** (line 153):
[UNCERTAIN] simdgroup_matrix with bfloat may have been added or improved with
M3-era Metal, but claiming it is M3+ exclusive needs verification against the
Metal Feature Set Tables.

**M4 Neural Engine: "38 TOPS on M4 (vs 18 TOPS on M3)"** (line 157): CORRECT.
Both numbers match Apple's published specifications.

**M4 GPU IPC improvement "~15-20%"** (line 158): [UNVERIFIED] Apple did not
publish a specific per-core IPC improvement percentage. This is a reasonable
estimate from third-party benchmarks but is not an official specification.

**Metal 4**: The document does not discuss Metal 4 in detail. For completeness:
Metal 4 was announced at WWDC 2025 and introduces a unified command encoder,
`MTLTensor` for ML workloads, placement sparse resources, neural rendering
support, and MetalFX Frame Interpolation. None of this is mentioned in the
document, which may be fine if the document was written before WWDC 2025.

---

## 7. Unified Memory Details (Section 1.6, 6.1-6.5)

### storageModeShared behavior

**Claim: "storageModeShared is literally the same physical memory pages accessed
by both processors"** (line 137): CORRECT. This is the fundamental property of
Apple's unified memory architecture.

**Claim: "storageModeManaged: macOS discrete GPUs only"** (line 1616): CORRECT.
This mode is only relevant for Macs with discrete GPUs (Intel Macs with AMD
GPUs). Not applicable to Apple Silicon.

### mmap + MTLBuffer zero-copy

**Claim: You can mmap a file and create MTLBuffer with `bytesNoCopy`** (lines
306-313, 1590-1600): CORRECT in principle. This is a valid and well-documented
approach.

**Claim: "MTLBuffer makeBuffer(bytesNoCopy:) requires the pointer to be
page-aligned (16 KB on Apple Silicon)"** (line 1678): [INCORRECT] The Apple
Developer documentation for `makeBuffer(bytesNoCopy:length:options:deallocator:)`
states the pointer must be page-aligned. However, the Metal API historically
checks for 4096-byte (4 KB) alignment, not 16 KB. While macOS on Apple Silicon
uses 16 KB pages at the OS level (`vm_page_size` = 16384), the Metal API's
`bytesNoCopy` alignment requirement is documented and enforced at 4096 bytes in
practice. Error messages from the API say "not 4096 byte aligned." The document
should say the pointer must be page-aligned (which is 16 KB on Apple Silicon
for OS-level operations, but Metal may accept 4 KB alignment). This is a subtle
but important distinction.

**Claim: "macOS page size is 16 KB on Apple Silicon (NOT 4 KB like x86)"**
(line 1677): CORRECT at the OS level. `vm_page_size` is indeed 16384 on Apple
Silicon. However, the conflation with MTLBuffer alignment requirements (above)
introduces confusion.

### mmap concerns

**Claim: "The Data object manages the mmap lifetime"** (line 312, using
`Data(contentsOf:options:.mappedIfSafe)`): [UNCERTAIN] Using
`Data(contentsOf:options:.mappedIfSafe)` and then passing `baseAddress` to
`makeBuffer(bytesNoCopy:)` with `deallocator: nil` is risky. The `Data` object
must be retained for the lifetime of the `MTLBuffer`, and there is no guarantee
that `Data` created with `.mappedIfSafe` will actually mmap the file (it may
copy the data into memory instead if the file is small or the system decides
mmap is not safe). The direct `mmap()` approach shown later in Section 6.1 is
more reliable.

---

## 8. Other Specific Claims

### SIMD width (Section 1.3)

**Claim: "Apple GPUs use a SIMD width of 32 threads"** (line 50): CORRECT.
Verified across multiple sources.

**Claim: "wider than AMD wavefronts on RDNA (32) though narrower than GCN
wavefronts (64)"** (line 56): [INCORRECT] The claim says Apple's 32-wide SIMD
is "wider than" RDNA's 32-wide. They are the same width (both 32). The sentence
should say "the same as RDNA" not "wider than." RDNA uses 32-wide wavefronts
(Wave32), though it can also run in Wave64 mode. GCN uses 64-wide wavefronts.
The comparison is muddled.

### Threadgroup sizes (Section 1.4)

**Claim: "Maximum threads per threadgroup: 1024 on all Apple Silicon"** (line
62): CORRECT.

**Claim: "Maximum threadgroup dimensions: (1024, 1024, 1024)"** (line 63):
CORRECT (product must not exceed 1024).

### Threadgroup memory (Section 1.5.2)

**Claim: "32 KB per threadgroup on all Apple Silicon GPUs"** (line 89):
CORRECT. Verified via Apple documentation and Metal Feature Set Tables.

**Claim: "Latency: ~1-2 cycles"** (line 91): [UNVERIFIED] Apple does not
publish threadgroup memory latency. 1-2 cycles is a reasonable estimate based
on similar GPU architectures but is not official.

**Claim: "Bank conflicts: likely 32 banks on Apple GPUs"** (line 94):
[UNVERIFIED] Apple does not document the number of threadgroup memory banks.
32 banks matching SIMD width is a reasonable assumption from general GPU
architecture principles but is speculative.

### Memory types (Section 1.5.4)

**Claim: "LPDDR4X (M1), LPDDR5 (M2/M3), LPDDR5X (M4)"** (line 110):
CORRECT. M1 uses LPDDR4X-4266, M2/M3 use LPDDR5-6400, M4 uses LPDDR5X-7500.

### Neural Engine (Section 8.5)

| Chip | Document Claims | Verified | Verdict |
|------|----------------|----------|---------|
| M1 | 11 TOPS | 11 TOPS | CORRECT |
| M2 | 15.8 TOPS | 15.8 TOPS | CORRECT |
| M3 | 18 TOPS | 18 TOPS | CORRECT |
| M4 | 38 TOPS | 38 TOPS | CORRECT |

### GPU family mappings (implicit throughout)

The document implies M3 = Apple GPU family 8. [INCORRECT/OVERSIMPLIFIED]
The M3 supports both Apple GPU family 8 and Apple GPU family 9. The A17 Pro
also supports family 9. The M1/M2 are family 7. The M4 supports family 9.
Apple's GPU family numbering does not map 1:1 to chip generations in the way
the document implies.

### Appendix C

The Appendix C table repeats the M4 specifications. The issues are the same as
Section 1 above. Additionally:

**Claim: "Page size: 16 KB"** for all M4 variants: CORRECT at the OS level.

**Claim: "BFloat16: Yes"** for all M4 variants: CORRECT.

---

## 9. Code Correctness

### Swift code examples

The Swift code throughout Sections 2, 6, and 7 is syntactically correct and
follows standard Metal API patterns. The triple-buffering example (Section 2.7),
the model loading examples (Section 6.1-6.3), and the Swift Package setup
(Section 7.6) are all reasonable and follow best practices.

### Metal shader code examples

The Metal kernel examples (naive matmul, tiled matmul, simdgroup matmul,
quantized GEMV/GEMM) are syntactically correct Metal Shading Language. The
tiling logic, barrier usage, and SIMD reduction patterns are standard and
appear correct.

**Appendix B threadgroup binding** (line 2231): `threadgroup float* shmem
[[threadgroup(0)]]` -- [UNCERTAIN] Binding threadgroup memory via kernel
arguments with `[[threadgroup(N)]]` attribute is a valid MSL feature, but it
requires the host side to call `setThreadgroupMemoryLength(_:index:)` on the
compute command encoder. Most Metal code declares threadgroup memory inside the
kernel body instead. The syntax is valid but uncommon.

### Command-line compilation (Section 7.2)

The `xcrun -sdk macosx metal` and `xcrun -sdk macosx metallib` commands are
CORRECT.

---

## 10. Summary of All Errors

### Definite Errors (must fix)

1. **M4 Ultra does not exist** -- All M4 Ultra specs are fictional. Remove the
   entire M4 Ultra row and all references to M4 Ultra throughout the document.
   Replace with M3 Ultra where appropriate as the current high-end option.

2. **M3 Pro GPU cores: 11-18 should be 14-18** -- The "11" is the CPU core
   count, not GPU.

3. **M3 base max RAM: 36 GB should be 24 GB** -- 36 GB is the M3 Pro's max.

4. **M3 Ultra max RAM: 192 GB should be 512 GB** -- 192 GB was M2 Ultra's max.

5. **M4 Pro max RAM: 48 GB should be 64 GB** -- 48 GB is a common config but
   not the maximum.

6. **M4 base GPU cores: "10" should be "8-10"** -- 8-core variant exists.

7. **M3 Ultra bandwidth: 800 GB/s should be 819 GB/s**

8. **RDNA SIMD width comparison**: "wider than AMD RDNA (32)" should be
   "same as AMD RDNA (32)".

9. **atomic_float does not exist in MSL** -- Section 3.8 claims Metal 3.0+ has
   atomic float add. This is incorrect. Remove the `atomic_float` example.

10. **M3 FP32 TFLOPS: 4.1 is likely wrong** -- Third-party sources indicate
    ~3.6 TFLOPS. 4.1 GHz is the CPU clock speed.

11. **bfloat16 was not "added by M3"** -- It predates M3, available from Apple
    GPU family 7 (M1/A14).

### Likely Errors (should verify and probably fix)

12. **M1 Ultra FP32 TFLOPS**: 20.8 vs Apple's stated ~21 TFLOPS.

13. **M4 FP32 TFLOPS**: 4.6 vs third-party ~4.3.

14. **M4 Max FP32 TFLOPS**: 18.0 vs third-party ~18.4.

15. **MTLBuffer bytesNoCopy alignment**: Document says 16 KB. Metal API enforces
    4096-byte alignment. The OS page size is 16 KB but Metal's requirement may
    differ.

### Unverifiable Claims (flag but may be fine)

16. All M3-generation TFLOPS numbers (Apple stopped publishing these).
17. M2 Pro FP32 TFLOPS (6.8 -- derived, not published).
18. M4 Pro FP32 TFLOPS (8.3 -- not published).
19. Register file estimates (32-64 registers, 32-64 KB per core).
20. Threadgroup memory latency (1-2 cycles).
21. Threadgroup memory bank count (32 banks).
22. `simd_ballot` / `simd_vote::vote_t` exact syntax.
23. `simdgroup_bfloat8x8` exact type name.
24. `bfloat` literal suffix `bf`.
25. `maxBufferLength` = 256 GB claim.

---

## 11. Recommendations

1. Remove all M4 Ultra references. Replace the high-end target with M3 Ultra
   (80 GPU cores, 819 GB/s, up to 512 GB RAM) or note M4 Ultra as speculative.

2. Add a note that M3/M4 generation TFLOPS numbers are estimates from
   third-party benchmarks, not Apple-published specifications.

3. Fix the M3 Pro GPU core count, M3/M3 Ultra/M4 Pro RAM numbers.

4. Remove the atomic_float claim or note it as a workaround pattern, not a
   native MSL type.

5. Correct the bfloat16 introduction timeline.

6. Add bandwidth tiers for binned chip variants (M3 Max 30-core at 300 GB/s,
   M4 Max 32-core at ~410 GB/s).

7. Clarify the MTLBuffer alignment requirement (Metal enforces 4096-byte
   alignment for bytesNoCopy, even though the OS page size is 16 KB).
