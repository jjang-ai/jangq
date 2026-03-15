# Apple Metal GPU Computing for ML Inference

> Research document for MLXQ — Mixed-Precision Importance Quantization
> Covers everything needed to design and implement Metal compute kernels for variable bit-width quantized matrix multiplication on Apple Silicon.

---

## 1. Apple Silicon GPU Architecture

### 1.1 GPU Core Counts and Configurations

Apple Silicon integrates the GPU on the same die (or dies, for Ultra variants) as the CPU, sharing a unified memory pool. The GPU cores are not discrete "shader cores" like NVIDIA CUDA cores — they are wider execution units, each containing multiple ALUs organized into SIMD groups.

| Chip | GPU Cores | Peak FP32 TFLOPS | Peak FP16 TFLOPS | Memory BW (GB/s) | Max Unified RAM |
|------|-----------|-------------------|-------------------|-------------------|-----------------|
| M1 | 7-8 | 2.6 | 5.2 | 68.25 | 16 GB |
| M1 Pro | 14-16 | 5.2 | 10.4 | 200 | 32 GB |
| M1 Max | 24-32 | 10.4 | 20.8 | 400 | 64 GB |
| M1 Ultra | 48-64 | 20.8 | 41.6 | 800 | 128 GB |
| M2 | 8-10 | 3.6 | 7.2 | 100 | 24 GB |
| M2 Pro | 16-19 | 6.8 | 13.6 | 200 | 32 GB |
| M2 Max | 30-38 | 13.6 | 27.2 | 400 | 96 GB |
| M2 Ultra | 60-76 | 27.2 | 54.4 | 800 | 192 GB |
| M3 | 8-10 | 4.1 | 8.2 | 100 | 36 GB |
| M3 Pro | 11-18 | 7.4 | 14.8 | 150 | 36 GB |
| M3 Max | 30-40 | 16.4 | 32.8 | 400 | 128 GB |
| M3 Ultra | 60-80 | 32.8 | 65.6 | 800 | 192 GB |
| M4 | 10 | 4.6 | 9.2 | 120 | 32 GB |
| M4 Pro | 16-20 | 8.3 | 16.6 | 273 | 48 GB |
| M4 Max | 32-40 | 18.0 | 36.0 | 546 | 128 GB |
| M4 Ultra | 64-80 | 36.0 | 72.0 | 819 | 512 GB |

**For MXQ**: The M4 Ultra at 512 GB unified RAM and 819 GB/s bandwidth is the high-end target. The M4 Max at 128 GB / 546 GB/s is the sweet spot for 70B models. The M4 Pro at 48 GB / 273 GB/s is the mass-market target where MXQ-2.5bit makes a 70B model barely fit.

### 1.2 GPU Core Types

Apple GPU cores handle three types of work: vertex processing, fragment (pixel) processing, and compute. For ML inference, only compute matters.

Each GPU core contains:
- **ALU execution units**: Organized as SIMD pipelines that execute 32 threads in lockstep
- **Load/store units**: Handle memory access to device (main) memory
- **Threadgroup (shared) memory**: Fast on-chip SRAM accessible to all threads in a threadgroup
- **Texture units**: Optimized for image sampling (not relevant for compute, though texture sampling can be repurposed for lookup tables)
- **Register file**: Per-thread private registers

A compute dispatch occupies GPU cores exclusively — no vertex or fragment work is interleaved. When we dispatch a compute kernel, all active GPU cores run that kernel.

### 1.3 SIMD Width and Execution Model

Apple GPUs use a SIMD width of **32 threads**. This is the fundamental execution unit:

- **SIMD group (warp equivalent)**: 32 threads that execute the same instruction in lockstep
- All threads in a SIMD group execute the same instruction at the same time
- Branch divergence within a SIMD group causes serialization — divergent threads are masked off while the other path executes, then swapped
- SIMD group operations (shuffles, reductions) operate across all 32 threads with hardware support
- This is the same width as NVIDIA warps (32) and wider than AMD wavefronts on RDNA (32) though narrower than GCN wavefronts (64)

**For MXQ**: SIMD width of 32 is critical for kernel design. When we have blocks of 32 or 64 weights (MXQ block size), a single SIMD group can process one or two blocks. SIMD-group reductions (`simd_sum`) let us accumulate dot products without threadgroup barriers.

### 1.4 Threadgroup Sizes

- **Maximum threads per threadgroup**: 1024 on all Apple Silicon GPUs (M1 through M4)
- **Maximum threadgroup dimensions**: (1024, 1024, 1024) — but product cannot exceed 1024
- Common configurations for matmul: 256 (8x32), 512 (16x32), 1024 (32x32)
- The threadgroup is the unit of scheduling — all threads in a threadgroup are guaranteed to run concurrently on the same GPU core
- Multiple threadgroups can run on the same GPU core if resources (registers, shared memory) allow — this is "occupancy"

**Choosing threadgroup size for matmul**:
- Threadgroup size should be a multiple of SIMD width (32)
- Larger threadgroups allow more data reuse from threadgroup memory but reduce occupancy
- For tiled matmul with 32x32 output tiles: 32x32 = 1024 threads (maximum) — each thread computes one output element
- For tiled matmul with 16x16 output tiles using simdgroup_matrix: fewer threads needed since each SIMD group computes an 8x8 block
- Typical sweet spot: 256 threads (8 SIMD groups) — good balance of occupancy and data reuse

### 1.5 Memory Hierarchy

Apple Silicon's memory hierarchy, from fastest to slowest:

#### 1.5.1 Registers (Thread-Private Memory)

- Each thread has its own register file
- Fastest storage — zero latency for ALU operations
- Apple does not publicly document register file size, but empirically each thread can use ~32-64 registers (128-256 bytes) before spilling
- Register pressure affects occupancy: more registers per thread = fewer concurrent threads per core
- For MXQ kernels: keep dequantized values and accumulator in registers

#### 1.5.2 Threadgroup Memory (Shared Memory)

- **Size**: 32 KB per threadgroup on all Apple Silicon GPUs
- On-chip SRAM, accessible to all threads in the threadgroup
- Latency: ~1-2 cycles (similar to L1 cache)
- Requires explicit barriers for synchronization: `threadgroup_barrier(mem_flags::mem_threadgroup)`
- Declared in kernel as: `threadgroup float shared_tile[TILE_SIZE][TILE_SIZE];`
- Bank conflicts: threadgroup memory is banked (likely 32 banks on Apple GPUs, matching SIMD width). Accessing the same bank from different threads in a SIMD group causes serialization. Padding arrays by 1 element per row avoids conflicts for column access patterns.

**For MXQ tiled matmul**:
- 32 KB is enough for two 64x64 tiles of float16 (2 * 64 * 64 * 2 bytes = 16 KB) plus scales/zeros
- Or two 32x32 tiles of float32 (2 * 32 * 32 * 4 bytes = 8 KB) with room for metadata
- Double buffering (prefetch next tile while computing current) fits in 32 KB for tiles up to ~48x48 in float16

#### 1.5.3 Tile Memory (M3+ Only)

- M3 introduced "dynamic caching" which improved register allocation efficiency
- Tile memory is an extension of the tiled rendering architecture into compute — it provides per-tile scratchpad that persists between kernel dispatches within a render pass
- For pure compute kernels (our use case), tile memory is less relevant — threadgroup memory is the primary fast storage
- M3's dynamic caching does help compute kernels by more efficiently allocating the register file, potentially improving occupancy

#### 1.5.4 Device Memory (Unified Main Memory)

- This is the main LPDDR4X (M1), LPDDR5 (M2/M3), or LPDDR5X (M4) memory
- Shared between CPU and GPU — unified memory architecture
- Bandwidth is the defining performance characteristic for inference:

| Chip Tier | Bandwidth | Token/s for 70B fp16 (140 GB) | Token/s for 70B MXQ-2.5 (22 GB) |
|-----------|-----------|-------------------------------|----------------------------------|
| Base (M4) | 120 GB/s | 0.86 (won't fit anyway) | 5.5 |
| Pro (M4 Pro) | 273 GB/s | 1.95 (won't fit) | 12.4 |
| Max (M4 Max) | 546 GB/s | 3.9 | 24.8 |
| Ultra (M4 Ultra) | 819 GB/s | 5.85 | 37.2 |

These are theoretical maximums — real throughput is ~70-85% of peak due to kernel launch overhead, non-matmul operations, and memory access patterns.

**The formula**: `tokens_per_second ≈ memory_bandwidth / model_size_bytes` for autoregressive token generation (batch size 1). This is because each token generation requires reading ALL model weights once. This makes bandwidth the singular bottleneck for inference.

**Why MXQ directly improves throughput**: A 2.5-bit model is 6.4x smaller than fp16. This means 6.4x more tokens per second at the same bandwidth, or equivalently, the model fits on a machine with 6.4x less RAM.

### 1.6 Unified Memory Architecture — Why It Matters for ML

Traditional discrete GPU systems (NVIDIA) have a critical bottleneck: model weights must be copied from CPU RAM to GPU VRAM over PCIe (32 GB/s for PCIe 4.0 x16, 64 GB/s for PCIe 5.0). This means:

1. Model loading time is bottlenecked by PCIe transfer
2. Models must fit entirely in VRAM (typically 24-80 GB for consumer/prosumer GPUs)
3. Offloading layers to CPU memory incurs PCIe latency on every forward pass

Apple's unified memory eliminates all three problems:

1. **Zero-copy model loading**: Weights loaded into RAM by the CPU are immediately accessible to the GPU. No transfer needed. An `MTLBuffer` created with `storageModeShared` is literally the same physical memory pages accessed by both processors.

2. **All RAM is "VRAM"**: A Mac with 128 GB of unified memory can hold a 128 GB model and run it on the GPU. There is no separate, smaller VRAM pool.

3. **No offloading penalty for unified memory**: CPU and GPU access memory at the same bandwidth (though they may contend for bandwidth if both are active). There's no "slow path" for memory that's on the "wrong" side.

4. **Memory-mapped model loading**: We can `mmap()` a model file and create an `MTLBuffer` directly from the mapped pages. The OS handles paging — only the weights currently being accessed need to be in physical RAM. For a 200 GB model on a 128 GB machine, the OS pages weights in and out of swap, with only a performance (not correctness) penalty.

**For MXQ specifically**: Unified memory means we can `mmap` the `.mxq.safetensors` file and create `MTLBuffer` objects pointing directly at the quantized weight data. No copies, no conversion, no staging buffers. The GPU reads quantized data directly from the memory-mapped file pages. This is the fastest possible path from disk to GPU computation.

### 1.7 M3 and M4 Compute-Relevant Features

**M3 (A17 Pro GPU architecture)**:
- **Dynamic caching**: Hardware dynamically allocates register file space based on actual kernel requirements, rather than worst-case static allocation. This improves occupancy for kernels with variable register pressure — directly relevant for MXQ kernels where different bit-width paths use different numbers of registers.
- **Mesh shading**: Not relevant for compute.
- **Ray tracing hardware**: Not relevant for compute.
- **BFloat16 support**: M3 added native bfloat16 (bf16) support in the GPU. This is significant for ML — bf16 has the same dynamic range as fp32 (8-bit exponent) with reduced precision (7-bit mantissa vs 23-bit). `simdgroup_matrix` operations support bf16 on M3+.

**M4**:
- **LPDDR5X memory**: Higher bandwidth (up to 819 GB/s on M4 Ultra) directly improves inference throughput.
- **Enhanced Neural Engine**: 38 TOPS on M4 (vs 18 TOPS on M3). However, the Neural Engine is limited in programmability — it primarily accelerates Core ML models and specific operations. For custom quantized inference, Metal compute kernels are the correct path.
- **Improved GPU IPC**: M4 GPU cores have higher instructions-per-clock than M3, meaning each GPU core is more efficient. Exact improvement varies by workload but benchmarks suggest ~15-20% improvement per core.
- **Larger GPU configurations**: M4 Ultra doubles the M4 Max die, providing up to 80 GPU cores and 512 GB unified memory — the first Apple Silicon chip where truly massive models (400B+) can run entirely in memory.

### 1.8 Why Memory Bandwidth is THE Bottleneck for Inference

During autoregressive generation (producing tokens one at a time), each token requires:

1. Reading the entire model weights (all linear layers) from memory
2. Performing matmul: `output = input_activation @ weight_matrix`
3. The input activation is a single vector (or small batch), so the matmul is actually a matrix-vector multiply (GEMV)

For GEMV, the arithmetic intensity (FLOPs per byte loaded) is extremely low:
- Loading a weight matrix of shape [N, K] in fp16: N * K * 2 bytes
- Computing the matrix-vector product: 2 * N * K FLOPs (multiply + add)
- Arithmetic intensity: 2 * N * K / (N * K * 2) = **1 FLOP/byte**

Compare this to the GPU's compute-to-bandwidth ratio:
- M4 Max: 36 TFLOPS fp16 / 546 GB/s = **66 FLOPs/byte**

The GPU can perform 66 FLOPs for every byte it loads, but GEMV only needs 1 FLOP per byte. The GPU ALUs are idle ~98% of the time during token generation, waiting for memory. **The computation is trivially cheap; loading the weights is the entire cost.**

This is why quantization has an almost linear effect on throughput: halving the weight size (e.g., fp16 -> 8-bit) nearly doubles tokens/s, because the bottleneck is purely the number of bytes read from memory.

For **prefill** (processing the entire input prompt at once), the situation is different. The input is a matrix (not a vector), so we do true GEMM (matrix-matrix multiply). Arithmetic intensity scales with the sequence length:
- Input shape: [seq_len, K], Weight shape: [K, N]
- Bytes loaded: seq_len * K * 2 + K * N * 2
- FLOPs: 2 * seq_len * K * N
- For large seq_len, arithmetic intensity approaches seq_len FLOPs/byte

At seq_len > 66 on M4 Max, prefill becomes compute-bound rather than bandwidth-bound. This means quantization helps prefill less (the weights are smaller to load, but you're limited by TFLOPS not bandwidth). However, quantization still helps prefill by fitting larger models in memory.

---

## 2. Metal Compute Pipeline

### 2.1 The Full Lifecycle

A Metal compute dispatch follows this object hierarchy:

```
MTLDevice                          // Represents the GPU hardware
  └─ MTLCommandQueue               // Serial queue of command buffers
       └─ MTLCommandBuffer         // One "submission" to the GPU
            └─ MTLComputeCommandEncoder  // Records compute commands
                 ├─ setComputePipelineState()   // Which kernel to run
                 ├─ setBuffer()                  // Bind data buffers
                 ├─ setBytes()                   // Small inline constants
                 └─ dispatchThreadgroups()        // Launch the kernel
```

Step-by-step in Swift:

```swift
// 1. Get the GPU device
let device = MTLCreateSystemDefaultDevice()!

// 2. Create a command queue (reuse for the lifetime of the app)
let commandQueue = device.makeCommandQueue()!

// 3. Load shader library (compiled .metallib or .metal source)
let library = try device.makeLibrary(source: metalSource, options: nil)
// Or from precompiled: device.makeDefaultLibrary()
// Or from file: device.makeLibrary(filepath: "kernels.metallib")

// 4. Get a specific kernel function
let function = library.makeFunction(name: "mxq_dequant_matmul")!

// 5. Create pipeline state (expensive — cache this!)
let pipelineState = try device.makeComputePipelineState(function: function)

// 6. Create a command buffer for this batch of work
let commandBuffer = commandQueue.makeCommandBuffer()!

// 7. Create compute encoder
let encoder = commandBuffer.makeComputeCommandEncoder()!

// 8. Set the pipeline (which kernel to run)
encoder.setComputePipelineState(pipelineState)

// 9. Bind buffers (weights, activations, output, metadata)
encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
encoder.setBuffer(activationsBuffer, offset: 0, index: 1)
encoder.setBuffer(outputBuffer, offset: 0, index: 2)
encoder.setBuffer(scalesBuffer, offset: 0, index: 3)
encoder.setBuffer(zerosBuffer, offset: 0, index: 4)
encoder.setBuffer(bitWidthsBuffer, offset: 0, index: 5)

// 10. Set small constants inline (avoids creating a buffer)
var params = KernelParams(M: 1, N: 4096, K: 4096, blockSize: 64)
encoder.setBytes(&params, length: MemoryLayout<KernelParams>.size, index: 6)

// 11. Dispatch
let gridSize = MTLSize(width: N, height: M, depth: 1)
let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
encoder.dispatchThreadgroups(
    MTLSize(width: (N + 255) / 256, height: M, depth: 1),
    threadsPerThreadgroup: threadgroupSize
)

// 12. End encoding and commit
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()  // Block until GPU finishes
```

### 2.2 Loading Shaders: Compile Time vs Runtime

**Build-time compilation** (preferred for production):
- `.metal` source files are compiled to `.metallib` by the Metal compiler (`metal` and `metallib` command-line tools, or Xcode build system)
- Loaded via `device.makeDefaultLibrary()` (from the app bundle) or `device.makeLibrary(URL:)` from a specific `.metallib` file
- Faster startup — no shader compilation at launch
- Function specialization constants can be set at pipeline creation time

**Runtime compilation** (useful for development and dynamic kernels):
- Pass `.metal` source as a string to `device.makeLibrary(source:options:)`
- The Metal compiler is invoked at runtime — takes hundreds of milliseconds for complex kernels
- Useful when kernel parameters need to be baked in (e.g., bit width, block size) for maximum performance
- llama.cpp uses runtime compilation to specialize kernels for specific quantization formats

**For MXQ**: Use build-time compilation for fixed kernels (matmul, dequant). Consider runtime compilation for generating specialized kernel variants for specific bit-width combinations — a kernel that only handles 2-bit and 3-bit blocks can be faster than a generic kernel that handles all bit widths, because the compiler can optimize the unpacking logic.

### 2.3 Setting Buffers and Data

**`setBuffer(buffer, offset, index)`**:
- `buffer`: An `MTLBuffer` containing the data
- `offset`: Byte offset into the buffer (must be 256-byte aligned for some operations; 4-byte aligned minimum)
- `index`: The argument index in the kernel function signature — corresponds to `[[buffer(N)]]` attribute

**`setBytes(bytes, length, index)`**:
- For small data (< 4 KB) that changes every dispatch
- Avoids creating an `MTLBuffer` for ephemeral constants
- Internally uses an argument buffer — no GPU memory allocation

**Buffer creation**:
```swift
// For weights (loaded from file, shared with CPU)
let weightsBuffer = device.makeBuffer(
    bytes: weightData,
    length: weightData.count,
    options: .storageModeShared  // Unified memory — CPU and GPU access
)

// For output (GPU writes, CPU reads results)
let outputBuffer = device.makeBuffer(
    length: outputSize,
    options: .storageModeShared
)

// For memory-mapped model files (zero-copy!)
let fileData = try Data(contentsOf: modelURL, options: .mappedIfSafe)
let mappedBuffer = device.makeBuffer(
    bytesNoCopy: fileData.baseAddress!,
    length: fileData.count,
    options: .storageModeShared,
    deallocator: nil  // The Data object manages the mmap lifetime
)
```

### 2.4 Dispatching Compute

Two dispatch methods:

**`dispatchThreadgroups(_:threadsPerThreadgroup:)`**:
- You specify how many threadgroups to launch and how many threads per threadgroup
- Total threads = threadgroups.x * threadsPerThreadgroup.x (times y, z dimensions)
- You are responsible for handling boundary conditions (threads past the end of data)
- This is the standard dispatch method for most kernels

```swift
// For M x N output matrix, with threadgroup size 16x16
let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
let threadgroups = MTLSize(
    width: (N + 15) / 16,   // Ceiling division
    height: (M + 15) / 16,
    depth: 1
)
encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
```

**`dispatchThreads(_:threadsPerThreadgroup:)`**:
- You specify the total number of threads; Metal calculates threadgroups for you
- Metal handles boundary conditions — threads past the end are not launched
- Requires non-uniform threadgroup size support (all Apple Silicon has this)
- Simpler but gives less control

```swift
let totalThreads = MTLSize(width: N, height: M, depth: 1)
let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
encoder.dispatchThreads(totalThreads, threadsPerThreadgroup: threadgroupSize)
```

**Choosing grid dimensions for matmul**:
- Output matrix is M x N. Each threadgroup computes a TILE_M x TILE_N tile of the output.
- Grid dimensions: `(N / TILE_N, M / TILE_M, 1)` threadgroups
- Threads per threadgroup: depends on how much work each thread does
  - If each thread computes one output element: TILE_M * TILE_N threads
  - If using simdgroup_matrix (each SIMD group computes 8x8): TILE_M * TILE_N / 64 * 32 threads
- For MXQ GEMV (batch=1): M=1, so grid is 1D along N. Each threadgroup handles TILE_N output elements, with K divided among threads for the reduction.

### 2.5 Synchronization

**Command buffer completion**:
```swift
// Option 1: Block the CPU thread
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Option 2: Callback (non-blocking)
commandBuffer.addCompletedHandler { cb in
    // GPU work is done, read results
    let output = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
}
commandBuffer.commit()

// Option 3: Check status
commandBuffer.commit()
while commandBuffer.status != .completed {
    // Poll (don't do this in production)
}
```

**MTLEvent** (cross-command-buffer synchronization):
```swift
let event = device.makeEvent()!
// In command buffer 1:
encoder1.encodeSignalEvent(event, value: 1)
// In command buffer 2:
encoder2.encodeWaitForEvent(event, value: 1)
```

**MTLFence** (within a command buffer, between encoders):
```swift
let fence = device.makeFence()!
// In encoder 1:
encoder1.updateFence(fence)
// In encoder 2:
encoder2.waitForFence(fence)
```

**Memory barriers** (within a compute encoder, between dispatches):
```swift
encoder.memoryBarrier(scope: .buffers)  // Ensure writes are visible
encoder.memoryBarrier(resources: [bufferA, bufferB])  // Specific buffers
```

### 2.6 Multiple Dispatches and Command Buffers

**Single command buffer, multiple dispatches** (preferred for sequential kernels):
```swift
let encoder = commandBuffer.makeComputeCommandEncoder()!

// Dispatch 1: Dequantize
encoder.setComputePipelineState(dequantPipeline)
encoder.setBuffer(quantizedWeights, offset: 0, index: 0)
encoder.setBuffer(dequantizedWeights, offset: 0, index: 1)
encoder.dispatchThreadgroups(...)

// Memory barrier ensures dequant output is visible to matmul
encoder.memoryBarrier(resources: [dequantizedWeights])

// Dispatch 2: Matmul using dequantized weights
encoder.setComputePipelineState(matmulPipeline)
encoder.setBuffer(dequantizedWeights, offset: 0, index: 0)
encoder.setBuffer(activations, offset: 0, index: 1)
encoder.setBuffer(output, offset: 0, index: 2)
encoder.dispatchThreadgroups(...)

encoder.endEncoding()
commandBuffer.commit()
```

**For MXQ**: Use a fused dequant+matmul kernel instead of two separate dispatches. Separate dispatches write dequantized weights to device memory (slow), then read them back (slow again). A fused kernel dequantizes in registers or threadgroup memory and immediately multiplies — the dequantized values never touch device memory.

### 2.7 Triple Buffering for Overlapping CPU/GPU Work

For continuous inference, overlap CPU preparation with GPU execution:

```swift
let inflightSemaphore = DispatchSemaphore(value: 3)  // 3 frames in flight
var bufferIndex = 0
let paramBuffers = (0..<3).map { _ in device.makeBuffer(length: paramSize, options: .storageModeShared)! }

func generateToken() {
    inflightSemaphore.wait()  // Block if 3 command buffers are in flight

    let params = paramBuffers[bufferIndex]
    // CPU: prepare input for this token (update KV cache pointers, etc.)
    prepareInput(into: params)

    let commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.addCompletedHandler { _ in
        inflightSemaphore.signal()  // Release slot when GPU finishes
    }

    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setBuffer(params, offset: 0, index: 6)
    // ... dispatch kernels ...
    encoder.endEncoding()
    commandBuffer.commit()

    bufferIndex = (bufferIndex + 1) % 3
}
```

This ensures the CPU can prepare the next token's input while the GPU is still processing the current one. For autoregressive generation, the overlap is minimal (CPU work is small), but it avoids pipeline bubbles.

---

## 3. Metal Shading Language for Compute

### 3.1 Kernel Function Signature

A Metal compute kernel is declared with the `kernel` qualifier and `void` return type. Arguments are bound by index using attributes:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void mxq_dequant_matmul(
    // Buffer arguments — bound via setBuffer()/setBytes()
    device const uint8_t*  packed_weights  [[buffer(0)]],
    device const half*     scales          [[buffer(1)]],
    device const half*     zeros           [[buffer(2)]],
    device const uint8_t*  bit_widths      [[buffer(3)]],
    device const half*     activations     [[buffer(4)]],
    device half*           output          [[buffer(5)]],
    constant uint&         M               [[buffer(6)]],
    constant uint&         N               [[buffer(7)]],
    constant uint&         K               [[buffer(8)]],

    // Thread indexing — provided by the runtime
    uint3 tid              [[thread_position_in_grid]],
    uint3 tgid             [[threadgroup_position_in_grid]],
    uint3 tid_in_tg        [[thread_position_in_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]]
) {
    // Kernel body
}
```

### 3.2 Address Space Qualifiers

Metal has four address spaces for pointers:

| Qualifier | Scope | Lifetime | Typical Use |
|-----------|-------|----------|-------------|
| `device` | Global GPU memory (unified) | Persistent | Weight buffers, input/output |
| `threadgroup` | Shared within threadgroup | Threadgroup lifetime | Tiles, shared accumulators |
| `constant` | Read-only, optimized for broadcast | Persistent | Kernel parameters, lookup tables |
| `thread` | Private to one thread | Thread lifetime | Local variables (default) |

```metal
// device: main memory buffers
device float* weights [[buffer(0)]];
device const float* input [[buffer(1)]];  // const = read-only

// threadgroup: shared scratchpad
threadgroup float tile[32][33];  // 33 to avoid bank conflicts on 32-bank memory

// constant: kernel parameters (broadcast-optimized, all threads read same value)
constant uint& matrix_size [[buffer(6)]];  // Small, uniform access

// thread: local variables (default, no qualifier needed)
float accumulator = 0.0;
```

**Key distinction for MXQ**: Weight buffers are `device const` (huge, read-only). Scales and zeros are also `device const` (smaller, read-only). Output is `device` (write). Threadgroup memory holds tiles of activations and dequantized weight tiles. Local accumulators are `thread`.

### 3.3 Thread Indexing

Metal provides built-in variables for thread identification:

```metal
// Absolute position in the entire dispatch grid
uint3 tid [[thread_position_in_grid]];           // (x, y, z)

// Position within the threadgroup
uint3 tid_in_tg [[thread_position_in_threadgroup]]; // (x, y, z)

// Which threadgroup this thread belongs to
uint3 tgid [[threadgroup_position_in_grid]];     // (x, y, z)

// SIMD group (warp) identification
uint simd_lane_id [[thread_index_in_simdgroup]];     // 0-31
uint simd_group_id [[simdgroup_index_in_threadgroup]]; // 0-(threadgroup_size/32 - 1)

// Grid and threadgroup dimensions
uint3 grid_size [[threads_per_grid]];
uint3 tg_size [[threads_per_threadgroup]];
```

**For matmul kernel indexing**:
```metal
// Each threadgroup computes a TILE_M x TILE_N output tile
uint row = tgid.y * TILE_M + tid_in_tg.y;  // Output row
uint col = tgid.x * TILE_N + tid_in_tg.x;  // Output col

// For GEMV (M=1): linearize
uint out_idx = tgid.x * TILE_N + tid_in_tg.x;
```

### 3.4 SIMD Group Operations — Critical for Reductions

SIMD group operations execute across all 32 threads in a SIMD group with hardware support — no threadgroup memory or barriers needed. These are essential for efficient reductions.

```metal
// Reduction: sum across all 32 lanes
float sum = simd_sum(value);          // All lanes get the sum
float max_val = simd_max(value);      // All lanes get the max
float min_val = simd_min(value);      // All lanes get the min

// Prefix scan
float prefix = simd_prefix_exclusive_sum(value);

// Shuffle: read another lane's value
float other = simd_shuffle(value, lane_id);      // Read lane_id's value
float down = simd_shuffle_down(value, offset);   // Read value from lane + offset
float up = simd_shuffle_up(value, offset);        // Read value from lane - offset
float xor_val = simd_shuffle_xor(value, mask);   // Read value from lane ^ mask

// Ballot: which lanes have a true predicate
simd_vote::vote_t mask = simd_ballot(predicate);  // Bitmask of lanes where predicate is true

// Broadcast: read lane 0's value
float broadcast = simd_broadcast(value, 0);

// All/any
bool all_true = simd_all(predicate);
bool any_true = simd_any(predicate);
```

**For MXQ dot product reduction**:
```metal
// Each lane computes one element of the dot product
// Then reduce across the SIMD group
float partial = dequantized_weight * activation;
float dot_product = simd_sum(partial);  // Sum across 32 lanes

// For K > 32, each lane accumulates multiple products, then reduce
float acc = 0.0;
for (uint k = simd_lane_id; k < K; k += 32) {
    acc += dequant(weights, k) * activations[k];
}
float total = simd_sum(acc);  // Reduce across lanes
```

### 3.5 Threadgroup Memory and Barriers

```metal
// Declare shared memory
threadgroup float shared_activations[TILE_K][TILE_M];
threadgroup half shared_weights[TILE_K][TILE_N];

// Load data cooperatively (each thread loads a few elements)
shared_activations[tid_in_tg.x][tid_in_tg.y] = activations[global_row * K + k_offset + tid_in_tg.x];

// Barrier: ensure all threads have finished loading before computing
threadgroup_barrier(mem_flags::mem_threadgroup);

// Now compute using shared data
float acc = 0;
for (uint k = 0; k < TILE_K; k++) {
    acc += shared_activations[k][tid_in_tg.y] * shared_weights[k][tid_in_tg.x];
}

// Barrier before next tile load (ensure computation is done before overwriting shared memory)
threadgroup_barrier(mem_flags::mem_threadgroup);
```

**Memory flags for barriers**:
- `mem_flags::mem_threadgroup` — Ensures threadgroup memory writes are visible
- `mem_flags::mem_device` — Ensures device memory writes are visible
- `mem_flags::mem_texture` — Ensures texture writes are visible
- `mem_flags::mem_threadgroup_imageblock` — For imageblock memory

**Bank conflict avoidance**:
```metal
// BAD: column access on 32-wide array causes 32-way bank conflicts
threadgroup float tile[32][32];
float val = tile[k][tid_in_tg.x];  // All 32 threads hit same bank

// GOOD: pad by 1 to offset columns across banks
threadgroup float tile[32][33];  // 33 instead of 32
float val = tile[k][tid_in_tg.x];  // Threads hit different banks
```

### 3.6 Data Types

Metal supports a wide range of data types relevant to ML:

```metal
// Floating point
half    h = 1.0h;     // 16-bit float (fp16): 1 sign, 5 exponent, 10 mantissa
float   f = 1.0f;     // 32-bit float (fp32)
bfloat  b = 1.0bf;    // 16-bit bfloat (bf16): 1 sign, 8 exponent, 7 mantissa (M3+)

// Integer
uint8_t   u8  = 0;    // 8-bit unsigned
uint16_t  u16 = 0;    // 16-bit unsigned
uint32_t  u32 = 0;    // 32-bit unsigned
int8_t    i8  = 0;    // 8-bit signed
int16_t   i16 = 0;    // 16-bit signed
int32_t   i32 = 0;    // 32-bit signed

// Vector types
half4   h4 = half4(1.0h, 2.0h, 3.0h, 4.0h);
float4  f4 = float4(1.0, 2.0, 3.0, 4.0);
uint4   u4 = uint4(0, 1, 2, 3);

// Packed types (tightly packed, no alignment padding)
packed_half4  ph4;     // 8 bytes (vs half4 which may be 8 or 16 depending on context)
packed_float4 pf4;     // 16 bytes
```

**For MXQ quantized weights**:
- Quantized data: `uint8_t` arrays containing packed 2/3/4/5/6/8-bit values
- Scales: `half` (fp16) — one per block (32 or 64 weights)
- Zero points: `half` (fp16) — one per block
- Bit widths: `uint8_t` — one per block
- Activations: `half` (fp16) — the input vectors/matrices
- Accumulator: `float` (fp32) — to avoid fp16 precision loss during accumulation
- Output: `half` (fp16) — final result

**Conversion operations**:
```metal
half h = half(f);           // float -> half (round to nearest)
float f = float(h);         // half -> float (exact)
half h = half(u8);          // uint8 -> half
bfloat b = bfloat(f);       // float -> bfloat (M3+)
```

### 3.7 Pointer Casting and Bit Manipulation for Unpacking

Extracting variable-width quantized values requires bit manipulation:

```metal
// Read 2 bytes from packed data to handle cross-byte boundaries
inline half dequantize_value(
    device const uint8_t* packed,
    uint bit_offset,
    uint bits,
    half scale,
    half zero
) {
    uint byte_offset = bit_offset >> 3;       // bit_offset / 8
    uint bit_shift = bit_offset & 7;          // bit_offset % 8

    // Read 16 bits starting at byte_offset (handles cross-byte boundaries)
    uint16_t raw16 = uint16_t(packed[byte_offset]) | (uint16_t(packed[byte_offset + 1]) << 8);

    // Extract 'bits' bits starting at bit_shift
    uint16_t mask = (uint16_t(1) << bits) - 1;
    uint8_t raw = uint8_t((raw16 >> bit_shift) & mask);

    // Dequantize: value = (raw - zero) * scale
    return (half(raw) - zero) * scale;
}
```

**Optimized unpacking for known bit widths** (much faster than generic):
```metal
// 4-bit: two values per byte, no cross-byte issues
inline half2 unpack_4bit(uint8_t packed_byte, half scale, half zero) {
    uint8_t low = packed_byte & 0x0F;
    uint8_t high = (packed_byte >> 4) & 0x0F;
    return half2(
        (half(low) - zero) * scale,
        (half(high) - zero) * scale
    );
}

// 2-bit: four values per byte
inline half4 unpack_2bit(uint8_t packed_byte, half scale, half zero) {
    return half4(
        (half(packed_byte & 0x03) - zero) * scale,
        (half((packed_byte >> 2) & 0x03) - zero) * scale,
        (half((packed_byte >> 4) & 0x03) - zero) * scale,
        (half((packed_byte >> 6) & 0x03) - zero) * scale
    );
}

// 3-bit: 8 values per 3 bytes (24 bits)
inline void unpack_3bit_8(device const uint8_t* packed, half scale, half zero, thread half* out) {
    uint32_t bits = uint32_t(packed[0]) | (uint32_t(packed[1]) << 8) | (uint32_t(packed[2]) << 16);
    for (int i = 0; i < 8; i++) {
        out[i] = (half((bits >> (i * 3)) & 0x07) - zero) * scale;
    }
}
```

### 3.8 Atomic Operations

Useful for global reductions or shared counters:

```metal
// Atomic operations on device memory
device atomic_uint* counter [[buffer(7)]];
atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);

// Atomic on threadgroup memory
threadgroup atomic_uint shared_counter;
atomic_fetch_add_explicit(&shared_counter, 1, memory_order_relaxed);

// Atomic float add (Metal 3.0+ / Apple GPU family 8+)
device atomic_float* sum [[buffer(8)]];
atomic_fetch_add_explicit(sum, partial_result, memory_order_relaxed);
```

For MXQ kernels, atomics are generally avoided in the hot path — they serialize access. Prefer SIMD reductions and threadgroup reductions using shared memory.

### 3.9 Metal Math Functions

```metal
// Fused multiply-add: a * b + c (single instruction, no intermediate rounding)
float result = fma(a, b, c);       // CRITICAL for matmul accuracy
half result_h = fma(a_h, b_h, c_h);

// Fast math (default in Metal) — may reorder operations, use reciprocal estimates
// Controlled by -ffast-math (default on) or -fno-fast-math compiler flags

// Precise math — when you need exact IEEE 754 behavior
float result = precise::fma(a, b, c);
float result = precise::sqrt(x);

// Common math functions
float s = sin(x);
float e = exp(x);
float l = log(x);
float p = pow(x, y);
float r = rsqrt(x);      // 1.0 / sqrt(x) — fast hardware instruction
float m = max(a, b);
float n = min(a, b);
float c = clamp(x, lo, hi);
float a = abs(x);
float s = sign(x);
float f = floor(x);
float c = ceil(x);

// For softmax / activation functions
float sigmoid = 1.0 / (1.0 + exp(-x));
float silu = x * sigmoid;   // SiLU / Swish activation
float gelu = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
```

---

## 4. Matrix Multiplication on Metal

### 4.1 Naive Matmul — Why It's Terrible

```metal
// Naive: each thread computes one output element
kernel void naive_matmul(
    device const half* A [[buffer(0)]],   // [M x K]
    device const half* B [[buffer(1)]],   // [K x N]
    device half* C [[buffer(2)]],         // [M x N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= N || tid.y >= M) return;

    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        acc += float(A[tid.y * K + k]) * float(B[k * N + tid.x]);
    }
    C[tid.y * N + tid.x] = half(acc);
}
```

Why this is terrible:
1. **No data reuse**: Each thread reads its entire row of A and column of B from device memory. For M=N=K=4096, each thread reads 8192 half values (16 KB). Total memory reads across all threads: 4096 * 4096 * 16 KB = 256 TB of memory traffic for a 32 MB output matrix. The actual data is only 64 MB (two 4096x4096 fp16 matrices).
2. **No memory coalescing**: Adjacent threads read the same row of A but different columns of B. B accesses are strided (stride = N), meaning each SIMD group accesses 32 different cache lines of B per iteration.
3. **Low arithmetic intensity**: Each float value is read from memory many times instead of being reused from fast storage.

### 4.2 Tiled Matmul — The Standard Approach

The key insight: tiles of A and B can be loaded into threadgroup memory and reused by all threads in the threadgroup.

```metal
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

kernel void tiled_matmul(
    device const half* A [[buffer(0)]],   // [M x K]
    device const half* B [[buffer(1)]],   // [K x N]
    device half* C [[buffer(2)]],         // [M x N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]]
) {
    // Shared memory tiles (padded to avoid bank conflicts)
    threadgroup half As[TILE_M][TILE_K + 1];
    threadgroup half Bs[TILE_K][TILE_N + 1];

    // Output position
    uint row = tgid.y * TILE_M + tid_in_tg.y;
    uint col = tgid.x * TILE_N + tid_in_tg.x;

    float acc = 0.0;

    // Iterate over K dimension in tiles
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative load: each thread loads one element of each tile
        uint a_row = tgid.y * TILE_M + tid_in_tg.y;
        uint a_col = k_tile + tid_in_tg.x;
        if (a_row < M && a_col < K)
            As[tid_in_tg.y][tid_in_tg.x] = A[a_row * K + a_col];
        else
            As[tid_in_tg.y][tid_in_tg.x] = 0.0h;

        uint b_row = k_tile + tid_in_tg.y;
        uint b_col = tgid.x * TILE_N + tid_in_tg.x;
        if (b_row < K && b_col < N)
            Bs[tid_in_tg.y][tid_in_tg.x] = B[b_row * N + b_col];
        else
            Bs[tid_in_tg.y][tid_in_tg.x] = 0.0h;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product from this tile
        for (uint k = 0; k < TILE_K; k++) {
            acc += float(As[tid_in_tg.y][k]) * float(Bs[k][tid_in_tg.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (row < M && col < N) {
        C[row * N + col] = half(acc);
    }
}
```

**Memory traffic reduction**:
- Each tile of A (TILE_M x TILE_K) is loaded once by the threadgroup and used by TILE_N threads
- Each tile of B (TILE_K x TILE_N) is loaded once and used by TILE_M threads
- Reuse factor: TILE_M for B, TILE_N for A
- For TILE=32: 32x reduction in global memory traffic vs naive

**Shared memory usage**: 2 * 32 * 33 * 2 bytes = 4.125 KB per tile pair. Well within the 32 KB limit.

### 4.3 Tile Size Selection for Apple GPUs

| Tile Size | Threads/TG | Shared Mem | Reuse Factor | Occupancy | Best For |
|-----------|-----------|------------|--------------|-----------|----------|
| 16x16 | 256 | ~2 KB | 16x | High | Small matrices, GEMV |
| 32x32 | 1024 | ~8 KB | 32x | Medium | General matmul |
| 64x64 | 1024* | ~16 KB | 64x | Lower | Large matmul, GEMM |

*64x64 with 1024 threads: each thread computes a 4x4 sub-tile of the output (4 output elements per thread).

For Apple GPUs, **32x32 tiles with simdgroup_matrix** is typically optimal:
- 32x32 = 1024 threads at max, but with simdgroup_matrix we need fewer threads
- Each SIMD group handles an 8x8 sub-tile using hardware-accelerated matmul
- 32x32 output tile = 16 8x8 sub-tiles = 16 SIMD groups = 512 threads
- Leaves room for occupancy (two threadgroups per core)

### 4.4 SIMD-Group Matrix Operations (simdgroup_matrix)

Apple Silicon has hardware-accelerated small matrix multiply operations that operate at the SIMD group level. This is Apple's equivalent of NVIDIA's Tensor Cores.

```metal
#include <metal_simdgroup_matrix>

// Declare 8x8 matrix types
simdgroup_float8x8 accum;     // 8x8 accumulator in float32
simdgroup_half8x8 a_tile;     // 8x8 tile of A in fp16
simdgroup_half8x8 b_tile;     // 8x8 tile of B in fp16

// Load from memory
// The 32 threads in the SIMD group cooperatively load the 64 elements (2 per thread)
simdgroup_load(a_tile, A_ptr, A_stride);     // Load 8x8 from A
simdgroup_load(b_tile, B_ptr, B_stride);     // Load 8x8 from B

// Initialize accumulator to zero
accum = simdgroup_float8x8(0);

// Hardware-accelerated 8x8 matmul: accum += a_tile * b_tile
simdgroup_multiply_accumulate(accum, a_tile, b_tile, accum);

// Store result
simdgroup_store(accum, C_ptr, C_stride);
```

**Building larger matmul from 8x8 blocks**:

To compute a 32x32 output tile from 32xK input tiles:
```
C[32x32] = A[32xK] * B[KxN]

Break into 8x8 blocks:
C[i*8:(i+1)*8, j*8:(j+1)*8] += A[i*8:(i+1)*8, k*8:(k+1)*8] * B[k*8:(k+1)*8, j*8:(j+1)*8]

For i in 0..3, j in 0..3 (16 output blocks), k in 0..K/8 (reduction steps)
```

```metal
#define TILE 32
#define BM 8   // simdgroup_matrix block size
#define BN 8
#define BK 8

kernel void simdgroup_matmul(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Each SIMD group handles one 8x8 output block
    // 16 SIMD groups per threadgroup = 512 threads
    uint bi = simd_gid / 4;   // Row block index (0-3)
    uint bj = simd_gid % 4;   // Col block index (0-3)

    // Base positions in output
    uint base_row = tgid.y * TILE + bi * BM;
    uint base_col = tgid.x * TILE + bj * BN;

    // Accumulator (per SIMD group, distributed across 32 lanes)
    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    // Iterate over K in steps of 8
    for (uint k = 0; k < K; k += BK) {
        simdgroup_half8x8 a_blk, b_blk;

        // Load 8x8 block of A: rows [base_row..+8], cols [k..k+8]
        simdgroup_load(a_blk, A + base_row * K + k, K);

        // Load 8x8 block of B: rows [k..k+8], cols [base_col..+8]
        simdgroup_load(b_blk, B + k * N + base_col, N);

        // Accumulate: acc += a_blk * b_blk
        simdgroup_multiply_accumulate(acc, a_blk, b_blk, acc);
    }

    // Store result
    simdgroup_store(acc, C + base_row * N + base_col, N);
}
```

**Available simdgroup_matrix types**:
- `simdgroup_half8x8` — fp16 input (all Apple Silicon)
- `simdgroup_float8x8` — fp32 accumulator (all Apple Silicon)
- `simdgroup_bfloat8x8` — bf16 input (M3+ only)

**Loading from threadgroup memory** (preferred for tiled matmul):
```metal
threadgroup half tg_A[32][33];  // In threadgroup memory
simdgroup_half8x8 a_blk;
simdgroup_load(a_blk, &tg_A[bi * 8][k], 33);  // Load from threadgroup mem, stride=33
```

This is faster than loading from device memory because threadgroup memory has ~1-2 cycle latency vs ~100+ cycles for device memory.

### 4.5 Memory Coalescing

Memory coalescing is when adjacent threads access adjacent memory addresses, allowing the hardware to merge multiple small accesses into fewer large transactions.

**Coalesced (good)** — row-major matrix, threads read along a row:
```metal
// Thread 0 reads B[k][0], thread 1 reads B[k][1], ..., thread 31 reads B[k][31]
// These are contiguous in memory: addresses differ by sizeof(half) = 2 bytes
// Hardware merges into one 64-byte cache line fetch
float val = B[k * N + tid_in_tg.x];  // GOOD: contiguous access
```

**Non-coalesced (bad)** — column-major access on row-major data:
```metal
// Thread 0 reads A[0][k], thread 1 reads A[1][k], ..., thread 31 reads A[31][k]
// These are strided: addresses differ by K * sizeof(half) bytes
// Each thread hits a different cache line — 32 separate memory transactions
float val = A[tid_in_tg.y * K + k];  // BAD: strided access (stride = K)
```

**Fix for non-coalesced access**: Load into threadgroup memory with coalesced reads, then access from threadgroup memory (which has no coalescing requirements because it's SRAM, not DRAM):
```metal
// Cooperative coalesced load of A tile (threads read along K dimension)
threadgroup half As[TILE_M][TILE_K + 1];
As[tid_in_tg.x][tid_in_tg.y] = A[(tgid.y * TILE_M + tid_in_tg.x) * K + (k_tile + tid_in_tg.y)];
// tid_in_tg.y varies fastest -> contiguous addresses in A -> coalesced
threadgroup_barrier(mem_flags::mem_threadgroup);
// Now access As with any pattern — it's in fast SRAM
```

**For MXQ**: Quantized weight data is packed tightly. For 4-bit weights, each byte contains 2 weights. Adjacent threads should read adjacent bytes for coalescing. When the block layout stores weights for contiguous output channels sequentially, GEMV access (one thread per output channel, reading the same input channel) is naturally coalesced along the output dimension.

### 4.6 Double Buffering in Threadgroup Memory

Overlap loading the next tile from device memory with computing on the current tile:

```metal
threadgroup half As[2][TILE_M][TILE_K + 1];  // Two slots
threadgroup half Bs[2][TILE_K][TILE_N + 1];
int slot = 0;

// Preload first tile into slot 0
load_tile(As[0], Bs[0], A, B, 0, tgid, tid_in_tg);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
    int next_slot = 1 - slot;

    // Start loading next tile into alternate slot (if not last tile)
    if (k_tile + TILE_K < K) {
        load_tile(As[next_slot], Bs[next_slot], A, B, k_tile + TILE_K, tgid, tid_in_tg);
    }

    // Compute on current slot (overlaps with loads on some architectures)
    compute_tile(As[slot], Bs[slot], &acc, tid_in_tg);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    slot = next_slot;
}
```

**Shared memory cost**: Doubles the threadgroup memory usage. For 32x32 fp16 tiles: 2 * 2 * 32 * 33 * 2 = ~8.25 KB. Still well within 32 KB.

On Apple GPUs, the benefit of double buffering depends on the memory-to-compute ratio. For compute-bound kernels (large GEMM), double buffering hides memory latency effectively. For bandwidth-bound kernels (GEMV), the GPU is already stalled on memory and there's nothing to overlap.

### 4.7 The Memory Bandwidth Wall

The transition from bandwidth-bound to compute-bound depends on arithmetic intensity:

```
Arithmetic intensity = FLOPs / Bytes loaded
Ridge point = Peak TFLOPS / Peak Bandwidth (in FLOP/byte)
```

For M4 Max fp16:
```
Peak fp16 = 36 TFLOPS = 36,000 GFLOP/s
Peak BW = 546 GB/s
Ridge point = 36,000 / 546 ≈ 66 FLOP/byte
```

For GEMV (batch=1, M=1):
```
FLOPs = 2 * K * N
Bytes = K * N * bytes_per_weight + K * 2 (input) + N * 2 (output)
       ≈ K * N * bytes_per_weight  (weights dominate)

Arithmetic intensity = 2 * K * N / (K * N * bpw) = 2 / bpw

fp16: 2/2 = 1 FLOP/byte  << 66  → BANDWIDTH BOUND
4-bit: 2/0.5 = 4 FLOP/byte << 66 → BANDWIDTH BOUND
2-bit: 2/0.25 = 8 FLOP/byte << 66 → STILL BANDWIDTH BOUND
```

GEMV is always bandwidth-bound on Apple Silicon, regardless of quantization. Quantization helps by reducing the bytes loaded, not by changing the computational regime.

For GEMM (batch=B, M=B):
```
FLOPs = 2 * B * K * N
Bytes ≈ K * N * bpw (weights dominate for B << K,N)
Arithmetic intensity ≈ 2 * B / bpw

fp16, B=1: 1 FLOP/byte (BW-bound)
fp16, B=33: 33 FLOP/byte (BW-bound)
fp16, B=66: 66 FLOP/byte (at ridge point!)
fp16, B=128: 128 FLOP/byte (compute-bound)

4-bit, B=1: 4 (BW-bound)
4-bit, B=17: 68 (compute-bound!)
```

**For MXQ**: Token generation (B=1) is always bandwidth-bound. Every byte saved by quantization directly translates to proportionally faster generation. Prefill with large batch/sequence length crosses the ridge point much sooner with quantized weights.

### 4.8 Batched Matmul

For serving multiple sequences simultaneously or processing long prefills:

```metal
// Batched GEMM: C[b] = A[b] * B (B is shared — same weights for all batches)
kernel void batched_matmul(
    device const half* A [[buffer(0)]],     // [batch, M, K]
    device const half* B [[buffer(1)]],     // [K, N] (shared)
    device half* C [[buffer(2)]],           // [batch, M, N]
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]]  // (N_tiles, M_tiles, batch)
) {
    uint b = tgid.z;  // Batch index
    // A for this batch: A + b * M * K
    // C for this batch: C + b * M * N
    // B is the same for all batches
    device const half* A_batch = A + b * M * K;
    device half* C_batch = C + b * M * N;

    // ... tiled matmul as before, using A_batch and shared B ...
}
```

For inference with shared weights across batches, the weight matrix B is read once per threadgroup (cached) and reused across all batch elements. This increases arithmetic intensity by a factor of batch_size, pushing toward compute-bound regime.

---

## 5. Implementing Quantized Matmul on Metal

### 5.1 The Core Challenge

MXQ needs a fused kernel that:
1. Reads quantized weights (2-8 bits per weight) from device memory
2. Reads per-block scale and zero point
3. Reads the bit width for each block (variable per block — this is unique to MXQ)
4. Unpacks and dequantizes in registers
5. Multiplies with input activations
6. Accumulates and writes output

This must happen in a single kernel — writing dequantized weights to device memory and reading them back would negate the bandwidth savings of quantization.

### 5.2 Why Fusion Matters — The Bandwidth Argument

Consider a 4096x4096 weight matrix:

**Unfused (separate dequant + matmul)**:
1. Read quantized weights: 4096 * 4096 * 0.5 bytes (4-bit) = 8 MB
2. Write dequantized fp16 weights: 4096 * 4096 * 2 = 32 MB
3. Read dequantized weights for matmul: 32 MB
4. Total memory traffic: 72 MB

**Fused (dequant + matmul in one kernel)**:
1. Read quantized weights: 8 MB
2. Total memory traffic: 8 MB + negligible activations/output

Fusion provides a **9x reduction in memory traffic** for 4-bit weights. For 2-bit weights, it's even more dramatic (4 MB read vs 68 MB unfused = 17x).

### 5.3 Kernel Structure for Quantized Matmul

#### 5.3.1 GEMV Path (Token Generation, Batch=1)

For autoregressive generation, M=1. The matmul becomes a matrix-vector product. Each output element is a dot product between one row of the weight matrix and the input vector.

```metal
// Quantized GEMV: output[n] = sum_k(dequant(W[n,k]) * input[k])
// Each threadgroup computes TILE_N output elements
// Within each threadgroup, threads cooperatively reduce over K

#define BLOCK_SIZE 64   // MXQ quantization block size
#define TILE_N 4        // Output elements per threadgroup (rows of W)
#define SIMD_SIZE 32

kernel void mxq_gemv(
    device const uint8_t*  packed_weights  [[buffer(0)]],  // Packed quantized data
    device const half*     scales          [[buffer(1)]],  // Per-block scales
    device const half*     zeros           [[buffer(2)]],  // Per-block zeros
    device const uint8_t*  bit_widths      [[buffer(3)]],  // Per-block bit widths
    device const uint32_t* block_offsets   [[buffer(4)]],  // Byte offset for each block's packed data
    device const half*     input           [[buffer(5)]],  // Input vector [K]
    device half*           output          [[buffer(6)]],  // Output vector [N]
    constant uint&         N               [[buffer(7)]],
    constant uint&         K               [[buffer(8)]],
    uint tgid_x   [[threadgroup_position_in_grid]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],
    uint simd_lid  [[thread_index_in_simdgroup]]
) {
    // This threadgroup handles output indices [tgid_x * TILE_N .. + TILE_N)
    // Each SIMD group handles one output element
    uint n = tgid_x * TILE_N + simd_gid;
    if (n >= N) return;

    float acc = 0.0;

    // Number of blocks in the K dimension for this row
    uint num_blocks = K / BLOCK_SIZE;
    uint blocks_per_row = num_blocks;  // Assumes K is a multiple of BLOCK_SIZE
    uint row_block_start = n * blocks_per_row;

    // Each lane processes K/32 elements (strided across the K dimension)
    for (uint block_idx = 0; block_idx < blocks_per_row; block_idx++) {
        uint global_block = row_block_start + block_idx;
        uint bits = bit_widths[global_block];
        half scale = scales[global_block];
        half zero = zeros[global_block];
        uint byte_off = block_offsets[global_block];

        // Each lane handles BLOCK_SIZE/32 = 2 elements within this block
        for (uint i = simd_lid; i < BLOCK_SIZE; i += SIMD_SIZE) {
            uint k = block_idx * BLOCK_SIZE + i;
            if (k >= K) break;

            // Compute bit offset within the packed data for this block
            uint bit_offset = i * bits;
            uint b_off = byte_off + (bit_offset >> 3);
            uint b_shift = bit_offset & 7;

            // Extract value
            uint16_t raw16 = uint16_t(packed_weights[b_off])
                           | (uint16_t(packed_weights[b_off + 1]) << 8);
            uint16_t mask = (uint16_t(1) << bits) - 1;
            half dequant = (half(uint8_t((raw16 >> b_shift) & mask)) - zero) * scale;

            acc += float(dequant) * float(input[k]);
        }
    }

    // Reduce across SIMD group
    acc = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lid == 0) {
        output[n] = half(acc);
    }
}
```

#### 5.3.2 GEMM Path (Prefill / Batched Inference)

For M > 1, we use a tiled approach where the weight tile is dequantized into threadgroup memory:

```metal
#define TILE_M 32
#define TILE_N 32
#define TILE_K 64   // Match block size for aligned dequantization

kernel void mxq_gemm(
    device const uint8_t*  packed_weights  [[buffer(0)]],
    device const half*     scales          [[buffer(1)]],
    device const half*     zeros           [[buffer(2)]],
    device const uint8_t*  bit_widths      [[buffer(3)]],
    device const uint32_t* block_offsets   [[buffer(4)]],
    device const half*     activations     [[buffer(5)]],   // [M x K]
    device half*           output          [[buffer(6)]],   // [M x N]
    constant uint&         M               [[buffer(7)]],
    constant uint&         N               [[buffer(8)]],
    constant uint&         K               [[buffer(9)]],
    uint2 tgid      [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    // Threadgroup memory for tiles
    threadgroup half tg_A[TILE_M][TILE_K + 1];   // Activation tile
    threadgroup half tg_W[TILE_K][TILE_N + 1];   // Dequantized weight tile

    // Each thread accumulates its output element
    float acc = 0.0;

    uint row = tgid.y * TILE_M + tid_in_tg.y;
    uint col = tgid.x * TILE_N + tid_in_tg.x;

    // Iterate over K in tiles of TILE_K
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // === Load activation tile (standard fp16 load) ===
        uint a_row = tgid.y * TILE_M + tid_in_tg.y;
        uint a_col = k_tile + tid_in_tg.x;
        if (a_row < M && a_col < K)
            tg_A[tid_in_tg.y][tid_in_tg.x] = activations[a_row * K + a_col];
        else
            tg_A[tid_in_tg.y][tid_in_tg.x] = 0.0h;

        // === Dequantize weight tile into threadgroup memory ===
        // Weight layout: W[n][k], we need tile W[k_tile..+TILE_K][col_tile..+TILE_N]
        // Each thread dequantizes one or more elements
        uint w_k = k_tile + tid_in_tg.y;
        uint w_n = tgid.x * TILE_N + tid_in_tg.x;

        if (w_k < K && w_n < N) {
            uint block_idx_in_row = w_k / BLOCK_SIZE;
            uint within_block = w_k % BLOCK_SIZE;
            uint global_block = w_n * (K / BLOCK_SIZE) + block_idx_in_row;

            uint bits = bit_widths[global_block];
            half scale = scales[global_block];
            half zero = zeros[global_block];
            uint byte_off = block_offsets[global_block];

            uint bit_offset = within_block * bits;
            uint b_off = byte_off + (bit_offset >> 3);
            uint b_shift = bit_offset & 7;

            uint16_t raw16 = uint16_t(packed_weights[b_off])
                           | (uint16_t(packed_weights[b_off + 1]) << 8);
            uint16_t mask = (uint16_t(1) << bits) - 1;
            half dequant = (half(uint8_t((raw16 >> b_shift) & mask)) - zero) * scale;

            tg_W[tid_in_tg.y][tid_in_tg.x] = dequant;
        } else {
            tg_W[tid_in_tg.y][tid_in_tg.x] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Compute partial product from this tile ===
        for (uint k = 0; k < TILE_K; k++) {
            acc += float(tg_A[tid_in_tg.y][k]) * float(tg_W[k][tid_in_tg.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (row < M && col < N) {
        output[row * N + col] = half(acc);
    }
}
```

#### 5.3.3 Using simdgroup_matrix with Dequantized Data

For maximum throughput, dequantize into threadgroup memory, then use `simdgroup_load` + `simdgroup_multiply_accumulate`:

```metal
// After dequantizing weight tile into tg_W:
threadgroup_barrier(mem_flags::mem_threadgroup);

// Each SIMD group computes an 8x8 sub-tile of the output
uint bi = simd_gid / (TILE_N / 8);  // Row block within tile
uint bj = simd_gid % (TILE_N / 8);  // Col block within tile

simdgroup_float8x8 acc_block = simdgroup_float8x8(0);

for (uint k = 0; k < TILE_K; k += 8) {
    simdgroup_half8x8 a_block, w_block;
    simdgroup_load(a_block, &tg_A[bi * 8][k], TILE_K + 1);
    simdgroup_load(w_block, &tg_W[k][bj * 8], TILE_N + 1);
    simdgroup_multiply_accumulate(acc_block, a_block, w_block, acc_block);
}

// Store the 8x8 output block
uint out_row = tgid.y * TILE_M + bi * 8;
uint out_col = tgid.x * TILE_N + bj * 8;
simdgroup_store(acc_block, output + out_row * N + out_col, N);
```

This is the fastest matmul path on Apple Silicon — the hardware `simdgroup_multiply_accumulate` instruction is significantly faster than manual multiply-add loops.

### 5.4 How llama.cpp Handles Metal Quantized Matmul

llama.cpp's Metal kernels (in `ggml/src/ggml-metal/ggml-metal.metal`) are the most mature quantized matmul implementation on Apple Silicon. Key design decisions:

**Separate kernels per quantization type**: llama.cpp compiles separate kernel functions for each GGUF quantization format — `kernel_mul_mv_q4_0_f32`, `kernel_mul_mv_q2_K_f32`, etc. This avoids branch divergence from runtime type switching.

**Lookup tables for dequantization**: Some formats use `constant` memory lookup tables to convert quantized indices to float values, trading memory for compute.

**SIMD group reductions**: The GEMV kernels use `simd_sum` for the K-dimension reduction within each SIMD group, then use threadgroup memory + barriers for cross-SIMD-group reduction if needed.

**Specialized packing**: Each format has hand-tuned unpacking code. For example, Q4_K stores scales and values in a specific interleaved layout optimized for coalesced access.

**The K-quant formats** (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) use per-block and per-super-block scales with different bit widths for different "importance levels" of a block — conceptually similar to MXQ but at a coarser granularity (per-format, not per-block).

**Relevant patterns for MXQ**:
1. Pre-compute byte offsets for each block to avoid runtime bit-offset arithmetic
2. Use `simdgroup_multiply_accumulate` for the GEMM path
3. Specialize kernels for common bit-width combinations (e.g., "all 2-bit" vs "all 4-bit" vs "mixed")
4. Pack metadata (scale, zero, bit_width) into a single struct per block for cache-friendly access

### 5.5 Variable Bit-Width Challenge — The MXQ-Specific Problem

MXQ's distinguishing feature is per-block bit-width allocation. This creates a challenge: adjacent blocks may have different bit widths, meaning:
1. The byte offset of block N depends on the bit widths of all preceding blocks
2. Threads processing different blocks execute different unpacking code (potential branch divergence)
3. The data layout is irregular — no simple stride pattern

**Option A: Dispatch separate kernels per bit width**

Pre-sort blocks by bit width, dispatch 6 kernels (one for 2-bit, one for 3-bit, etc.):

```swift
// Pre-sort: group all 2-bit blocks, all 3-bit blocks, etc.
for bits in [2, 3, 4, 5, 6, 8] {
    let blocks = blocksWithBitWidth(bits)
    if blocks.isEmpty { continue }
    encoder.setComputePipelineState(kernels[bits]!)
    encoder.setBuffer(blocks.buffer, ...)
    encoder.dispatchThreadgroups(...)
}
```

Pros: No branch divergence, each kernel is fully optimized for its bit width.
Cons: Many small dispatches = high launch overhead. Breaks the matmul structure — you can't compute a complete output row without results from all bit widths.

**This approach doesn't work well for matmul**, because a single output element depends on ALL blocks in a row (which have mixed bit widths). You'd need to accumulate partial results across dispatches.

**Option B: Single kernel with runtime bit-width branching**

One kernel handles all bit widths with a switch statement:

```metal
uint bits = bit_widths[global_block];
half dequant;
switch (bits) {
    case 2: dequant = unpack_2bit(packed, offset, scale, zero); break;
    case 3: dequant = unpack_3bit(packed, offset, scale, zero); break;
    case 4: dequant = unpack_4bit(packed, offset, scale, zero); break;
    case 5: dequant = unpack_5bit(packed, offset, scale, zero); break;
    case 6: dequant = unpack_6bit(packed, offset, scale, zero); break;
    case 8: dequant = unpack_8bit(packed, offset, scale, zero); break;
}
```

Pros: Simple, works with standard matmul tiling.
Cons: Branch divergence if threads in the same SIMD group process blocks with different bit widths. On Apple GPUs, divergent branches cause both paths to execute (masked), doubling or worse the execution time.

**Mitigation**: Since MXQ block size is 64 and SIMD width is 32, a SIMD group processes at most 2 blocks at a time. If we ensure that blocks along the K dimension within a tile have the same bit width (or at most 2 different widths), divergence is limited.

**Option C: Pre-sort blocks by bit width within each row (recommended for MXQ)**

Reorder the weight storage so that within each row (output channel), blocks are sorted by bit width. Within each group of same-width blocks, the standard tiled matmul works without divergence.

```
Original block layout for row n:
  [3-bit] [2-bit] [4-bit] [2-bit] [3-bit] [2-bit] [4-bit] [2-bit]

Sorted layout:
  [2-bit] [2-bit] [2-bit] [2-bit] | [3-bit] [3-bit] | [4-bit] [4-bit]
   ^--- all 2-bit blocks ---^       ^-- 3-bit ---^     ^-- 4-bit --^
```

The matmul kernel processes each homogeneous group with the optimal unpacking code. Since dot product is commutative and associative, reordering the K dimension doesn't change the result.

We store a "permutation index" per block to map back to the original K positions, so we read the correct activation elements:

```metal
uint original_k = permutation[sorted_block_idx * BLOCK_SIZE + within_block];
float act = float(input[original_k]);
```

Pros: Zero branch divergence, optimal unpacking, standard tiled structure.
Cons: Extra memory for permutation indices. Indirect activation access may hurt cache (but activations are small — one vector for GEMV).

**For MXQ, Option C is recommended**. The permutation indices add ~4 bytes per block (one uint32 per element, or one uint32 per block if blocks are contiguous in the sorted layout). For a 4096x4096 matrix with block_size=64, there are 4096*64 = 262,144 blocks. Permutation adds ~1 MB — negligible compared to the weight data.

### 5.6 Performance Optimization

#### 5.6.1 Packing for simdgroup_matrix

After dequantizing a tile of weights into threadgroup memory in fp16, we can directly use `simdgroup_load` + `simdgroup_multiply_accumulate`. The key is ensuring the threadgroup memory layout matches what `simdgroup_load` expects:

```metal
// Dequantize TILE_K x TILE_N weight tile into row-major fp16 in threadgroup memory
threadgroup half tg_W[TILE_K][TILE_N + 1];  // +1 padding for bank conflicts

// ... dequantization code fills tg_W ...

threadgroup_barrier(mem_flags::mem_threadgroup);

// simdgroup_load expects row-major with explicit stride
simdgroup_half8x8 w_block;
simdgroup_load(w_block, &tg_W[k_offset][n_offset], TILE_N + 1);
```

The stride parameter (TILE_N + 1) accounts for the padding. `simdgroup_load` handles the 2D indexing internally — each of the 32 threads loads 2 elements of the 8x8 block.

#### 5.6.2 Overlapping Memory Loads with Compute

Apple GPUs have out-of-order execution within a SIMD group to some degree — memory loads can be issued early and their results consumed later. To exploit this:

```metal
// Issue loads early
half4 packed_data = *(device const half4*)(packed_weights + offset);  // Prefetch

// Do unrelated computation while load completes
float prev_result = simd_sum(prev_acc);  // Use ALU while waiting for memory

// Now use the loaded data
half dequant = unpack(packed_data, ...);
```

More practically, structure the kernel so that the dequantization of the next block's data begins while the current block's multiply-accumulate is executing. This is natural in a loop:

```metal
for (uint block = 0; block < num_blocks; block++) {
    // Load block metadata (will be used next iteration in a double-buffered approach)
    uint next_bits = bit_widths[block + 1];
    half next_scale = scales[block + 1];

    // Compute with current block's already-loaded data
    acc = fma(current_dequant, activation, acc);

    // Dequantize next block (uses the loaded metadata)
    current_dequant = dequantize(packed, next_bits, next_scale, ...);
}
```

#### 5.6.3 Tuning Threadgroup Size for Occupancy

Occupancy = number of active threads / maximum possible threads per GPU core. Higher occupancy helps hide memory latency.

Factors that limit occupancy:
1. **Register usage**: More registers per thread = fewer threads fit. Apple GPUs have a fixed register file per core (exact size undisclosed, but ~32-64 KB estimated). A kernel using 64 registers per thread supports fewer concurrent threads than one using 32.
2. **Threadgroup memory**: 32 KB per threadgroup. If a threadgroup uses 16 KB, only 2 threadgroups can coexist per core.
3. **Threadgroup size**: Maximum 1024. Smaller threadgroups allow more threadgroups per core.

For MXQ kernels:
- **GEMV (token gen)**: Use 128 or 256 threads per threadgroup (4-8 SIMD groups). Low register pressure (just accumulator + dequant temps). Can have 4-8 threadgroups per core. Threadgroup memory: minimal (just the reduction buffer).
- **GEMM (prefill)**: Use 256 or 512 threads per threadgroup. Moderate threadgroup memory (~8-16 KB for tiles). Can have 2-4 threadgroups per core.

**Profiling is essential**: Use Instruments (Metal System Trace) to measure actual occupancy and identify the limiting factor. The `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` property tells you the hardware limit for a specific kernel.

```swift
let maxThreads = pipelineState.maxTotalThreadsPerThreadgroup
// If maxThreads < 1024, register pressure is limiting the threadgroup size
print("Max threads per threadgroup: \(maxThreads)")
print("Thread execution width (SIMD size): \(pipelineState.threadExecutionWidth)")
```

---

## 6. Memory Management for Large Models

### 6.1 Loading Large Models via MTLBuffer

A 70B model at MXQ-2.5bit is approximately 22 GB. The weights must be accessible as one or more `MTLBuffer` objects.

**Approach 1: Load into MTLBuffer directly**
```swift
let data = try Data(contentsOf: modelShardURL)
let buffer = device.makeBuffer(bytes: data.baseAddress!,
                                length: data.count,
                                options: .storageModeShared)
```
Drawback: This allocates new memory and copies the file data. For a 22 GB model, this means 22 GB of file reads + 22 GB of memory allocation. Doubling memory usage during load.

**Approach 2: Memory-mapped (zero-copy) — recommended**
```swift
// mmap the file
let fileHandle = try FileHandle(forReadingFrom: modelShardURL)
let fileSize = try fileHandle.seekToEnd()
fileHandle.seek(toFileOffset: 0)

let mappedData = mmap(nil, Int(fileSize), PROT_READ, MAP_PRIVATE, fileHandle.fileDescriptor, 0)!

// Create MTLBuffer wrapping the mmap'd region — NO COPY
let buffer = device.makeBuffer(bytesNoCopy: mappedData,
                                length: Int(fileSize),
                                options: .storageModeShared,
                                deallocator: { ptr, size in
                                    munmap(ptr, size)
                                })
```

This is the preferred approach for MLXQ:
- Zero memory overhead — the buffer IS the file's page cache
- OS handles paging — only pages accessed by the GPU are loaded from disk
- First access to each page incurs a page fault (~microseconds), but subsequent accesses hit the page cache
- Works with models larger than physical RAM (OS swaps pages in/out)

### 6.2 Storage Mode: storageModeShared

On Apple Silicon, `MTLResourceStorageModeShared` is the only sensible option for compute buffers:

| Storage Mode | CPU Access | GPU Access | Copy Required | Use Case |
|-------------|-----------|-----------|---------------|----------|
| `.storageModeShared` | Yes | Yes | No | Everything on Apple Silicon |
| `.storageModePrivate` | No | Yes | Must blit from shared | Intermediate GPU-only buffers |
| `.storageModeManaged` | Yes | Yes | Must synchronize | macOS discrete GPUs only |

For unified memory, `.storageModeShared` means the CPU and GPU access the exact same physical pages. No copies, no synchronization for read-only data (like model weights).

`.storageModePrivate` buffers cannot be accessed by the CPU but may have slightly better GPU access patterns on some hardware. For model weights (read-only after loading), this provides minimal benefit and prevents zero-copy loading.

### 6.3 Memory-Mapped Buffers — Zero-Copy Model Loading

The ideal MXQ model loading path:

```
Disk (.mxq.safetensors file)
  → mmap() system call
    → Virtual memory pages (not yet in physical RAM)
      → MTLBuffer (bytesNoCopy, storageModeShared)
        → GPU reads weight data
          → Page fault (first access only)
            → OS reads from disk into page cache
              → GPU gets the data
```

**Performance characteristics**:
- First token may be slower (page faults loading weights)
- Subsequent tokens are fast (weights are in page cache)
- If the system is under memory pressure, the OS may evict weight pages and re-read them from disk on next access — causing sporadic slowdowns
- For models that fit comfortably in RAM (e.g., 22 GB model on 128 GB Mac), page eviction is rare after warmup

**safetensors compatibility**: The safetensors format stores tensors contiguously with a JSON header. To mmap:
1. Parse the header to find each tensor's byte offset and size within the file
2. Create `MTLBuffer` with `bytesNoCopy` pointing to the base of the data section
3. When binding a specific tensor's buffer, use `setBuffer(buffer, offset: tensorOffset, index: N)`

The buffer offset in `setBuffer` handles per-tensor addressing within the mmap'd file. This avoids creating separate MTLBuffer objects per tensor.

### 6.4 Handling Models Larger Than Physical RAM

On a 128 GB Mac running a 200 GB model:

- The OS uses swap (SSD-backed virtual memory) to extend available memory
- macOS unified memory swap is fast (NVMe SSD, 3-7 GB/s read) but still 50-200x slower than LPDDR bandwidth
- Each token generation reads all 200 GB of weights. With 128 GB RAM, ~72 GB must be swapped in/out per token.
- At 5 GB/s SSD speed: 72 GB / 5 = ~14.4 seconds per token. Usable for offline processing, terrible for interactive use.

**MXQ mitigates this**: At MXQ-2.5bit, that 200 GB fp16 model becomes ~31 GB. On a 128 GB Mac, it fits entirely in RAM with room to spare. No swapping needed.

**madvise hints for mmap'd models**:
```c
// Tell the OS we'll access the entire file sequentially during each forward pass
madvise(mapped_ptr, file_size, MADV_SEQUENTIAL);

// Or if we know the access pattern is random across layers:
madvise(mapped_ptr, file_size, MADV_RANDOM);

// Pre-fault all pages (forces loading into RAM upfront — avoids per-page fault overhead)
madvise(mapped_ptr, file_size, MADV_WILLNEED);
```

For inference, `MADV_WILLNEED` on startup (pre-loads the model) followed by `MADV_SEQUENTIAL` during inference (optimal read-ahead) is the best strategy.

### 6.5 Memory Alignment Requirements

- **mmap**: Must be page-aligned. macOS page size is 16 KB on Apple Silicon (NOT 4 KB like x86). The `mmap` call automatically returns page-aligned addresses.
- **MTLBuffer**: The `makeBuffer(bytesNoCopy:)` requires the pointer to be page-aligned (16 KB on Apple Silicon).
- **setBuffer offset**: Must be 4-byte aligned minimum. For optimal performance, use 256-byte alignment (matching Apple GPU cache line behavior). safetensors format typically aligns tensors to 64-byte boundaries, which satisfies this.
- **simdgroup_load**: The source pointer must be naturally aligned for the element type (2-byte for half, 4-byte for float).

For MXQ file format design: ensure that each tensor shard's data starts at a 16 KB boundary within the safetensors file. This allows direct mmap-to-MTLBuffer mapping without any alignment fixups.

```swift
// Verify alignment
let pageSize = vm_page_size  // 16384 on Apple Silicon
assert(UInt(bitPattern: mappedPtr) % UInt(pageSize) == 0, "Pointer must be page-aligned")

// For setBuffer offset, ensure tensor offsets are at least 4-byte aligned
assert(tensorOffset % 4 == 0, "Buffer offset must be 4-byte aligned")
```

---

## 7. Swift + Metal Integration Specifics

### 7.1 Using Metal from Swift

**Basic setup**:
```swift
import Metal
import MetalKit

// Get the default GPU device
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
}

// Create command queue (reuse for app lifetime)
guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Failed to create command queue")
}

// Query device capabilities
print("Device name: \(device.name)")
print("Max buffer length: \(device.maxBufferLength)")  // Typically 256 GB on Apple Silicon
print("Max threadgroup memory: \(device.maxThreadgroupMemoryLength)")  // 32768 (32 KB)
print("Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")  // MTLSize(1024, 1024, 1024)
print("Unified memory: \(device.hasUnifiedMemory)")  // true on Apple Silicon
```

**Device feature sets and GPU family**:
```swift
// Check GPU family for feature availability
if device.supportsFamily(.apple8) {
    // M3+ features: dynamic caching, bfloat16 in shaders
    print("Apple GPU family 8+ supported")
}
if device.supportsFamily(.apple9) {
    // M4+ features
    print("Apple GPU family 9+ supported")
}

// Check specific features
if device.supports32BitMSAA { ... }
if device.areBarycentricCoordsSupported { ... }  // Not relevant for compute
```

### 7.2 Compiling Metal Shaders

**Option 1: Xcode build system (default for apps)**
- Add `.metal` files to the Xcode project
- Xcode automatically compiles them into `default.metallib` in the app bundle
- Load via `device.makeDefaultLibrary()`

**Option 2: Command-line compilation (for packages and CI)**
```bash
# Compile .metal source to .air (intermediate)
xcrun -sdk macosx metal -c mxq_dequant.metal -o mxq_dequant.air

# Link .air files into .metallib
xcrun -sdk macosx metallib mxq_dequant.air mxq_matmul.air -o mxq_kernels.metallib

# With optimization flags
xcrun -sdk macosx metal -c -ffast-math -O2 mxq_dequant.metal -o mxq_dequant.air

# With specific target (Apple Silicon GPU)
xcrun -sdk macosx metal -c -std=metal3.1 -target air64-apple-macos14 mxq_dequant.metal -o mxq_dequant.air
```

**Option 3: Runtime compilation from source string**
```swift
let metalSource = """
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(device float* data [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 2.0;
}
"""

let library = try device.makeLibrary(source: metalSource, options: nil)
```

**Option 4: Precompiled binary archive (Metal Binary Archive, iOS 14+ / macOS 11+)**
```swift
let descriptor = MTLBinaryArchiveDescriptor()
let archive = try device.makeBinaryArchive(descriptor: descriptor)
// Add pipeline states to archive, then serialize to disk
// Fastest loading — no compilation at all
```

**For MXQ**: Use command-line compilation in the Swift Package build process. Ship a `.metallib` file as a package resource. Optionally, support runtime compilation for JIT-specialized kernels (e.g., generating a kernel that only handles the specific bit widths present in the loaded model).

### 7.3 MetalKit vs Raw Metal API

**MetalKit (MTKView)**: For rendering — provides a view, draw loop, and display link. Not needed for pure compute.

**Raw Metal API**: For compute-only work (inference), use the Metal framework directly. No MetalKit dependency. This keeps the library lightweight and avoids UIKit/AppKit dependencies.

```swift
// Pure compute — no MetalKit needed
import Metal

class MXQEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineStates: [String: MTLComputePipelineState] = [:]

    init() throws {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!

        let library = try device.makeLibrary(filepath: "mxq_kernels.metallib")
        let functions = ["mxq_gemv_2bit", "mxq_gemv_3bit", "mxq_gemv_4bit",
                         "mxq_gemm_generic"]
        for name in functions {
            let function = library.makeFunction(name: name)!
            pipelineStates[name] = try device.makeComputePipelineState(function: function)
        }
    }
}
```

### 7.4 Swift/Metal Interop — Passing Data

**Swift Array to MTLBuffer**:
```swift
// From Swift array
var weights: [Float16] = loadWeights()
let buffer = device.makeBuffer(bytes: &weights,
                                length: weights.count * MemoryLayout<Float16>.stride,
                                options: .storageModeShared)!

// From UnsafeRawPointer (safetensors data)
let rawPtr: UnsafeRawPointer = safetensorsData.baseAddress!.advanced(by: tensorOffset)
let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: rawPtr),
                                length: tensorSize,
                                options: .storageModeShared,
                                deallocator: nil)
```

**Shared struct between Swift and Metal**:
```swift
// Swift side
struct MXQParams {
    var M: UInt32
    var N: UInt32
    var K: UInt32
    var blockSize: UInt32
    var numBitGroups: UInt32
}
// Pass via setBytes
var params = MXQParams(M: 1, N: 4096, K: 4096, blockSize: 64, numBitGroups: 3)
encoder.setBytes(&params, length: MemoryLayout<MXQParams>.size, index: 6)
```

```metal
// Metal side — must match Swift struct layout exactly
struct MXQParams {
    uint M;
    uint N;
    uint K;
    uint blockSize;
    uint numBitGroups;
};

kernel void mxq_kernel(constant MXQParams& params [[buffer(6)]]) {
    uint N = params.N;
    // ...
}
```

**Reading GPU output from Swift**:
```swift
commandBuffer.waitUntilCompleted()

// storageModeShared: CPU can read directly after GPU completion
let outputPtr = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputCount)
let result = Array(UnsafeBufferPointer(start: outputPtr, count: outputCount))
```

### 7.5 Using Accelerate Alongside Metal

The Accelerate framework provides CPU-optimized BLAS, vDSP, and vImage operations. Useful for:

- **Fallback path**: If Metal kernel fails or is unavailable
- **Small operations**: Operations too small to justify GPU dispatch overhead (e.g., RMSNorm on a single vector, small attention score matmul)
- **Pre/post processing**: Tokenization, sampling, embedding lookup

```swift
import Accelerate

// CPU matmul via BLAS (for small matrices)
var C = [Float](repeating: 0, count: M * N)
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(M), Int32(N), Int32(K),
            1.0,                    // alpha
            A, Int32(K),            // A [M x K]
            B, Int32(N),            // B [K x N]
            0.0,                    // beta
            &C, Int32(N))           // C [M x N]

// vDSP for element-wise ops
var output = [Float](repeating: 0, count: N)
vDSP_vadd(a, 1, b, 1, &output, 1, vDSP_Length(N))  // Element-wise add

// BNNS (CPU neural network ops) for specific layer types
// Can handle quantized operations but limited format support
```

**Decision criterion**: If the operation touches more data than ~64 KB or involves matrices larger than ~128x128, use Metal. For smaller operations, CPU via Accelerate avoids the ~10-50 microsecond GPU dispatch overhead.

### 7.6 Building a Swift Package with Metal Shaders

```
MXQEngine/
  Package.swift
  Sources/
    MXQEngine/
      Engine.swift
      MetalKernels.swift
      ModelLoader.swift
      Resources/
        mxq_kernels.metallib      // Precompiled shaders
    MXQMetal/
      mxq_dequant.metal           // Metal source files
      mxq_matmul.metal
      mxq_gemv.metal
  Tests/
    MXQEngineTests/
      KernelTests.swift
```

**Package.swift**:
```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MXQEngine",
    platforms: [.macOS(.v14)],  // Metal 3.1 minimum
    products: [
        .library(name: "MXQEngine", targets: ["MXQEngine"]),
    ],
    targets: [
        .target(
            name: "MXQEngine",
            dependencies: [],
            resources: [
                .copy("Resources/mxq_kernels.metallib")
            ]
        ),
        // Build Metal sources as part of the package
        // Note: SPM doesn't natively compile .metal files
        // Use a plugin or pre-build step
        .plugin(
            name: "MetalCompilerPlugin",
            capability: .buildTool()
        ),
        .testTarget(
            name: "MXQEngineTests",
            dependencies: ["MXQEngine"]
        ),
    ]
)
```

**Loading metallib from package resources**:
```swift
import Metal

class MXQKernels {
    let library: MTLLibrary

    init(device: MTLDevice) throws {
        // Load from bundle resource
        guard let url = Bundle.module.url(forResource: "mxq_kernels", withExtension: "metallib") else {
            fatalError("Metal library not found in package resources")
        }
        library = try device.makeLibrary(URL: url)
    }
}
```

**Alternative — compile Metal sources at build time using a build plugin**:

Swift Package Manager does not natively compile `.metal` files. Options:
1. Pre-compile to `.metallib` and include as a resource (recommended)
2. Write a Build Tool Plugin that invokes `xcrun metal` and `xcrun metallib`
3. Compile Metal source strings at runtime (slower startup but simplest integration)

For MXQ, pre-compiling to `.metallib` is recommended. The build script:
```bash
#!/bin/bash
# build_metallib.sh — run before swift build
METAL_DIR="Sources/MXQMetal"
OUTPUT="Sources/MXQEngine/Resources/mxq_kernels.metallib"

AIR_FILES=()
for metal_file in "$METAL_DIR"/*.metal; do
    air_file="${metal_file%.metal}.air"
    xcrun -sdk macosx metal -c -ffast-math -std=metal3.1 "$metal_file" -o "$air_file"
    AIR_FILES+=("$air_file")
done

xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$OUTPUT"
rm -f "${AIR_FILES[@]}"
```

### 7.7 Performance Profiling

**Instruments — Metal System Trace**:
The primary tool for profiling Metal performance. Shows:
- GPU timeline: when each command buffer / dispatch executes
- Occupancy: how many threads are active vs. maximum
- Memory bandwidth utilization
- Shader execution time per dispatch

**Instruments — GPU Counters**:
Provides hardware counter data:
- ALU utilization (%)
- Memory read/write bandwidth (GB/s, actual vs peak)
- Cache hit rates
- Threadgroup memory bandwidth
- SIMD group occupancy

**Programmatic GPU timing**:
```swift
// Timestamp the command buffer
let commandBuffer = commandQueue.makeCommandBuffer()!

// Add start timestamp
let startTime = CACurrentMediaTime()

// ... encode work ...

commandBuffer.addCompletedHandler { cb in
    let gpuStart = cb.gpuStartTime
    let gpuEnd = cb.gpuEndTime
    let gpuTime = gpuEnd - gpuStart
    print("GPU execution time: \(gpuTime * 1000) ms")

    let cpuEnd = CACurrentMediaTime()
    print("Total wall time: \((cpuEnd - startTime) * 1000) ms")
}
commandBuffer.commit()
```

**Metal API validation**:
```swift
// Enable in Xcode scheme: Edit Scheme → Run → Diagnostics → Metal → API Validation
// Or via environment variable
// Catches API misuse, buffer overruns, missing synchronization
```

**Metal shader profiling with Xcode**:
1. Xcode → Debug → Capture GPU Frame (for Metal workloads)
2. GPU Debugger shows per-dispatch timing, shader invocations, memory traffic
3. Shader Profiler shows per-line execution time within the kernel
4. Dependency Viewer shows synchronization and data dependencies between dispatches

---

## 8. Optimization Techniques Specific to Apple Silicon

### 8.1 Exploiting Unified Memory for Model Loading

The zero-copy path for MXQ model loading:

```
File on disk (.mxq.safetensors)
  → mmap() (no copy, just virtual address mapping)
    → MTLBuffer(bytesNoCopy:) (no copy, just wraps the mmap)
      → GPU setBuffer() (no copy, just binds the virtual address)
        → GPU reads quantized data (page fault → OS loads from disk → cached in RAM)
          → Dequantize in GPU registers
            → Matmul result
```

Total copies: **zero**. Total extra memory: **zero**. The only RAM used is the page cache, which the OS manages. Compare to CUDA where model loading requires: read file → CPU buffer → cudaMemcpy → GPU VRAM (at least one full copy, often two).

### 8.2 Memory Bandwidth Numbers and Token Generation Rates

The fundamental equation for autoregressive token generation:

```
tokens_per_second = memory_bandwidth / bytes_per_token

bytes_per_token = model_size_bytes  (must read all weights for each token)
```

Effective bandwidth is typically 70-85% of theoretical peak:

| Chip | Peak BW | Effective BW (~80%) | MXQ-2.5 70B (22GB) | MXQ-3 70B (27GB) | fp16 70B (140GB) |
|------|---------|-------------------|---------------------|-------------------|------------------|
| M4 | 120 | 96 | 4.4 tok/s | 3.6 tok/s | Won't fit |
| M4 Pro 48GB | 273 | 218 | 9.9 tok/s | 8.1 tok/s | Won't fit |
| M4 Max 128GB | 546 | 437 | 19.9 tok/s | 16.2 tok/s | 3.1 tok/s |
| M4 Ultra 512GB | 819 | 655 | 29.8 tok/s | 24.3 tok/s | 4.7 tok/s |
| M2 Ultra 192GB | 800 | 640 | 29.1 tok/s | 23.7 tok/s | 4.6 tok/s |

**MXQ-2.5 on M4 Max delivers ~20 tok/s for a 70B model** — fast enough for interactive use. Without quantization, the same model at fp16 would only achieve ~3 tok/s (and barely fits in 128 GB).

**For 120B models (e.g., Nemotron-H)**:
- MXQ-2.5: ~38 GB → M4 Max 128GB: 11.5 tok/s
- MXQ-3: ~45 GB → M4 Max 128GB: 9.7 tok/s
- MXQ-2.5: ~38 GB → M4 Ultra 512GB: 17.2 tok/s

### 8.3 Prefill Performance (Compute-Bound Regime)

During prefill (processing the input prompt), the operation is GEMM (matrix-matrix multiply) with the batch/sequence dimension providing data reuse for weights:

```
prefill_tokens_per_second = peak_tflops * 10^12 / (2 * params * active_ratio)

active_ratio accounts for: not all params are used every token (MoE),
attention is not purely matmul, etc. Typically ~0.7 for dense models.
```

| Chip | fp16 TFLOPS | Prefill tok/s (70B dense) | Prefill tok/s (120B dense) |
|------|-------------|--------------------------|---------------------------|
| M4 Max | 36 | ~170 | ~100 |
| M4 Ultra | 72 | ~340 | ~200 |
| M2 Ultra | 54 | ~260 | ~150 |

Prefill is compute-bound for sequence lengths above ~66 tokens (the ridge point calculated earlier). Quantization provides modest benefit for prefill — the weights are smaller to load from memory, but the computation (dequant + matmul) takes approximately the same time.

However, MXQ does help prefill by enabling larger models to fit in memory. A model that doesn't fit can't do prefill at all.

### 8.4 How Quantization Helps Both Regimes

**Token generation (bandwidth-bound)**:
- Direct, near-linear speedup: N-bit quantization reads N/16 as many bytes as fp16
- MXQ-2.5: 6.4x fewer bytes → ~6.4x faster token generation
- The dequantization compute (bit unpacking, scale/zero application) is essentially free because the GPU ALUs are 98% idle waiting for memory anyway

**Prefill (compute-bound)**:
- Smaller weights load faster, but GEMM throughput is limited by TFLOPS
- Dequantization adds compute overhead during prefill (unlike token gen where compute is free)
- Net effect: ~10-30% improvement for prefill (less memory to load, but extra dequant work)
- The primary benefit is fitting larger models in memory

**Context window / KV cache**:
- Quantized weights leave more RAM for KV cache
- A 70B model at fp16 uses 140 GB for weights, leaving ~0 GB for KV cache on a 128 GB Mac
- The same model at MXQ-2.5 uses 22 GB, leaving 106 GB for KV cache
- At fp16 KV, 106 GB supports ~200K-400K tokens of context for 70B model
- This enables extremely long context windows that are impossible without weight quantization

### 8.5 Neural Engine for Quantized Inference

The Apple Neural Engine (ANE) is a dedicated matrix accelerator present in all Apple Silicon:
- M1: 11 TOPS (int8)
- M2: 15.8 TOPS (int8)
- M3: 18 TOPS (int8)
- M4: 38 TOPS (int8)

**Can the ANE accelerate MXQ inference?**

Limitations:
1. **Core ML only**: The ANE is only programmable via Core ML. No Metal API access. No custom kernels.
2. **Fixed quantization formats**: Core ML supports int8 (uniform), int4 (palettized/LUT), and a few other fixed formats. No variable bit-width support.
3. **Graph-level optimization**: Core ML compiles entire model graphs, not individual operations. You can't use ANE for just the matmul and Metal for everything else (in practice).
4. **Limited precision**: ANE int8 TOPS doesn't help with fp16 accumulation needed for quality.

**Verdict**: The Neural Engine is not suitable for MXQ inference. Its fixed quantization format support cannot handle variable per-block bit widths. Metal compute kernels on the GPU are the correct approach.

However, the ANE could theoretically be used for:
- Running a small draft model for speculative decoding (via Core ML)
- Embedding computation (if the embedding layer is a standard format)
- Pre/post-processing operations (tokenizer, sampling)

For MXQ, all core inference computation should target Metal GPU compute.

### 8.6 Summary: Design Principles for MXQ Metal Kernels

1. **Fuse dequant + matmul**: Never write dequantized weights to device memory. Dequantize in registers or threadgroup memory and immediately multiply.

2. **Sort blocks by bit width**: Reorder the K dimension so that SIMD groups process homogeneous bit-width regions. Eliminates branch divergence.

3. **Use simdgroup_matrix for GEMM**: After dequantizing a tile into threadgroup memory as fp16, use hardware-accelerated 8x8 matmul. This is the fastest path.

4. **Use simd_sum for GEMV**: For single-token generation, SIMD group reductions are more efficient than threadgroup reductions for the dot product.

5. **Exploit zero-copy loading**: mmap the .mxq.safetensors file, create MTLBuffer with bytesNoCopy. No copies, no conversion.

6. **Pre-compute block byte offsets**: Store cumulative byte offsets per block to avoid runtime offset calculation. Adds a small metadata table but eliminates the serial dependency of "sum all previous block sizes to find my offset."

7. **Specialize common paths**: If a model uses only 2-bit and 3-bit blocks (common for MXQ-2.5), compile a specialized kernel that only handles those two widths. The compiler can optimize more aggressively.

8. **Profile with Metal System Trace**: Measure actual bandwidth utilization and occupancy. The GEMV kernel should achieve >70% of peak bandwidth. If it doesn't, the bottleneck is in the access pattern (non-coalesced reads, bank conflicts) rather than raw bandwidth.

9. **Align data to 16 KB page boundaries**: Ensures mmap + MTLBuffer(bytesNoCopy) work without alignment fixups. Align individual tensors within safetensors to at least 256 bytes for optimal GPU cache behavior.

10. **Target fp32 accumulation**: Accumulate matmul results in fp32 to avoid fp16 precision loss over long reductions (K=4096 or larger). Convert back to fp16 only when writing the final output.

---

## Appendix A: Quick Reference — Metal Compute API

```swift
// Device & Queue
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!

// Shader compilation
let library = try device.makeLibrary(source: src, options: nil)
let function = library.makeFunction(name: "kernel_name")!
let pipeline = try device.makeComputePipelineState(function: function)

// Buffer creation
let buf = device.makeBuffer(length: size, options: .storageModeShared)!
let buf = device.makeBuffer(bytes: ptr, length: size, options: .storageModeShared)!
let buf = device.makeBuffer(bytesNoCopy: ptr, length: size, options: .storageModeShared, deallocator: nil)!

// Dispatch
let cb = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
enc.setBuffer(buf, offset: 0, index: 0)
enc.setBytes(&params, length: MemoryLayout<Params>.size, index: 1)
enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
```

## Appendix B: Quick Reference — Metal Shading Language for Compute

```metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Kernel signature
kernel void my_kernel(
    device float* data      [[buffer(0)]],
    constant uint& count    [[buffer(1)]],
    threadgroup float* shmem [[threadgroup(0)]],  // Explicit threadgroup binding
    uint tid                [[thread_position_in_grid]],
    uint tid_in_tg          [[thread_position_in_threadgroup]],
    uint tgid               [[threadgroup_position_in_grid]],
    uint simd_lid           [[thread_index_in_simdgroup]],
    uint simd_gid           [[simdgroup_index_in_threadgroup]],
    uint tg_size            [[threads_per_threadgroup]]
) { ... }

// Threadgroup memory (declared in kernel body)
threadgroup float shared[256];
threadgroup_barrier(mem_flags::mem_threadgroup);

// SIMD operations
float sum = simd_sum(val);
float shuf = simd_shuffle(val, lane);
float down = simd_shuffle_down(val, delta);
bool all = simd_all(pred);
bool any = simd_any(pred);

// simdgroup_matrix (hardware matmul)
simdgroup_half8x8 a, b;
simdgroup_float8x8 c(0);
simdgroup_load(a, src_ptr, stride);
simdgroup_multiply_accumulate(c, a, b, c);
simdgroup_store(c, dst_ptr, stride);
```

## Appendix C: Chip Specifications for MXQ Target Platforms

| Property | M4 | M4 Pro | M4 Max | M4 Ultra |
|----------|-----|--------|--------|----------|
| GPU Cores | 10 | 16-20 | 32-40 | 64-80 |
| fp16 TFLOPS | 9.2 | 16.6 | 36.0 | 72.0 |
| Memory BW | 120 GB/s | 273 GB/s | 546 GB/s | 819 GB/s |
| Max RAM | 32 GB | 48 GB | 128 GB | 512 GB |
| Threadgroup mem | 32 KB | 32 KB | 32 KB | 32 KB |
| SIMD width | 32 | 32 | 32 | 32 |
| Max TG threads | 1024 | 1024 | 1024 | 1024 |
| BFloat16 | Yes | Yes | Yes | Yes |
| Page size | 16 KB | 16 KB | 16 KB | 16 KB |

## Appendix D: Performance Model for MXQ Kernel Design

```
# Token generation (GEMV, bandwidth-bound)
tok_per_sec = effective_bandwidth / model_size_bytes
effective_bandwidth = peak_bandwidth * utilization  (utilization target: 0.80)

# Dequantization overhead (should be negligible for GEMV)
dequant_cycles_per_element = ~2-5 cycles (bit shift + mask + multiply + add)
matmul_cycles_per_element = ~1 cycle (fma)
overhead_ratio = dequant_cycles / total_cycles ≈ 5-10% (acceptable)

# Prefill (GEMM, compute-bound for seq_len > ridge_point)
ridge_point = peak_tflops / peak_bandwidth (in FLOP/byte)
M4 Max fp16: 36000 / 546 ≈ 66

# For seq_len > 66, prefill is compute-bound
prefill_tps = peak_tflops / (2 * model_params * active_ratio)

# Threadgroup memory budget
tile_memory = 2 * TILE_M * (TILE_K+1) * sizeof(half)   // A tile
            + 2 * TILE_K * (TILE_N+1) * sizeof(half)    // W tile (dequantized)
            + block_metadata                              // scales, zeros per tile
must be <= 32768 bytes

# Block metadata per tile
blocks_per_tile = (TILE_K * TILE_N) / BLOCK_SIZE
metadata_per_block = sizeof(half) * 2 + sizeof(uint8_t) + sizeof(uint32_t)  // scale + zero + bits + offset
                   = 4 + 2 + 1 + 4 = 11 bytes
```
