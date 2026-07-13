//
// JANGTQ Swift bindings — codebook matmul + Hadamard butterfly.
// Created by Jinho Jang (eric@jangq.ai).
//
// Three thin wrappers around the kernels in JANGTQMatmul.metal. Mirrors
// the Python jang_tools.turboquant API so a Swift inference loop can call:
//
//     let rot = try ctx.hadamard.run(x: ..., signs: ..., dim: 3072)
//     let act = try ctx.fusedGateUp.run(xRot: rot, gate: ..., up: ...,
//                                        codebook: ..., indices: ..., K: 8)
//     let actRot = try ctx.hadamard.run(x: act, signs: ..., dim: 1536)
//     let y   = try ctx.gather.run(xRot: actRot, packed: ..., norms: ...,
//                                   codebook: ..., indices: ..., K: 8)
//
// Conventions match the Python kernels exactly:
//
//   packed[expert][out, pack_idx] : uint32, 16 × 2-bit indices LSB-first
//   norms [expert][out]           : half, per-row L2 norm
//   codebook                       : float32 (4 entries for 2-bit), keyed
//                                    on (in_features, bits) — gate and down
//                                    use DIFFERENT codebooks (see §10 of
//                                    research/JANGTQ-REFERENCE.md).
//   signs                          : float32 (in_features,) — ±1
//
// Sweet-spot tile sizes (from M3 Ultra sweep, P17):
//   fused_gate_up_swiglu : OPT = 10 outputs per thread
//   gather_tq_matmul     : OPT = 20 outputs per thread
//
// These are kernel-source constants — see JANGTQMatmul.metal. Resweep on
// new Apple GPU generations: P12 found OPT=4 on M4, M3 Ultra wants 10/20.
//
// Threadgroup math: each kernel uses 32 lanes per simd-group, stops at
// 256 threads per threadgroup. Grid.x = ceil(out_features / OPT) × 32,
// grid.y = K (broadcast experts). Per-row mode (down_proj) and broadcast
// mode (gate/up) differ only in how rhs_indices is laid out — the kernel
// math is identical.

import Foundation
import Metal

// MARK: - Hadamard butterfly (P3)

/// Single-dispatch multi-block Hadamard rotation for non-pow2 dimensions.
///
/// Input: x is (batch, dim) half. dim must decompose into pow2 blocks whose
/// largest block is ≤ 8192, matching the Metal threadgroup memory cap.
public struct JANGTQHadamard {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_hadamard_multiblock")
    }

    /// Decompose `dim` into descending pow2 blocks that sum to it.
    public static func decomposePow2(_ dim: Int) -> [Int] {
        var blocks: [Int] = []
        var rem = dim
        while rem > 0 {
            let p = 1 << (Int.bitWidth - 1 - rem.leadingZeroBitCount)
            blocks.append(p)
            rem -= p
        }
        return blocks
    }

    /// Build the meta buffer the kernel expects: [total_d, n_blocks, d_b0, log_b0, d_b1, log_b1, ...]
    public static func makeMeta(totalDim: Int) -> [UInt32] {
        let blocks = decomposePow2(totalDim)
        var meta: [UInt32] = [UInt32(totalDim), UInt32(blocks.count)]
        for d in blocks {
            meta.append(UInt32(d))
            meta.append(UInt32(d.trailingZeroBitCount))
        }
        return meta
    }

    /// Rotate x in place via Hadamard. Returns a new fp32 buffer of the same shape.
    /// `xBuf` is half-precision (2 bytes/elem). `signsBuf` is float32 (4 bytes/elem).
    public func run(
        xBuf: MTLBuffer,
        signsBuf: MTLBuffer,
        batch: Int,
        dim: Int
    ) throws -> MTLBuffer {
        let largestBlock = JANGTQHadamard.decomposePow2(dim).max() ?? dim
        precondition(dim > 0 && largestBlock <= 8192,
            "dim's largest pow2 block must be <= 8192")
        let outBytes = batch * dim * MemoryLayout<Float>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("hadamard out buffer alloc failed")
        }
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("command encoder alloc failed")
        }
        encode(into: enc, xBuf: xBuf, signsBuf: signsBuf, outBuf: outBuf, batch: batch, dim: dim)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return outBuf
    }

    /// Encode into a caller-managed encoder. The output buffer must be at
    /// least `batch * dim * 4` bytes (fp32).
    public func encode(
        into enc: MTLComputeCommandEncoder,
        xBuf: MTLBuffer, signsBuf: MTLBuffer, outBuf: MTLBuffer,
        batch: Int, dim: Int
    ) {
        let largestBlock = JANGTQHadamard.decomposePow2(dim).max() ?? dim
        precondition(dim > 0 && largestBlock <= 8192,
            "dim's largest pow2 block must be <= 8192")
        let meta = JANGTQHadamard.makeMeta(totalDim: dim)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xBuf,    offset: 0, index: 0)
        enc.setBuffer(signsBuf, offset: 0, index: 1)
        meta.withUnsafeBufferPointer { ptr in
            enc.setBytes(ptr.baseAddress!, length: meta.count * MemoryLayout<UInt32>.stride, index: 2)
        }
        enc.setBuffer(outBuf,   offset: 0, index: 3)
        let tgSize = min(1024, max(32, largestBlock))
        enc.dispatchThreads(MTLSizeMake(tgSize, batch, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgSize, 1, 1))
    }
}


// MARK: - Fused gate+up+SwiGLU (P17 OPT=10)

/// Computes `SiLU(gate_proj(x_rot)) * up_proj(x_rot)` for K experts in one
/// Metal dispatch. Output shape is `(K, out_features)` fp32, ready to feed
/// into `down_proj` (after another Hadamard rotation if used).
public struct JANGTQFusedGateUpSwiGLU {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    /// Outputs per thread for the fused kernel — must match JANGTQ_FUSED_OPT
    /// in JANGTQMatmul.metal. P17 sweet spot on M3 Ultra.
    public static let optOutsPerThread: Int = 10

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_fused_gate_up_swiglu")
    }

    public func run(
        xRotBuf: MTLBuffer,
        packedGateBuf: MTLBuffer, normsGateBuf: MTLBuffer,
        packedUpBuf: MTLBuffer,   normsUpBuf: MTLBuffer,
        codebookBuf: MTLBuffer,
        rhsIndicesBuf: MTLBuffer,
        K: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2,
        swigluLimit: Float = 0
    ) throws -> MTLBuffer {
        let outBytes = K * outFeatures * MemoryLayout<Float>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("fused_gate_up out alloc failed")
        }
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("command encoder alloc failed")
        }
        encode(into: enc,
               xRotBuf: xRotBuf, packedGateBuf: packedGateBuf, normsGateBuf: normsGateBuf,
               packedUpBuf: packedUpBuf, normsUpBuf: normsUpBuf,
               codebookBuf: codebookBuf, rhsIndicesBuf: rhsIndicesBuf, outBuf: outBuf,
               K: K, inFeatures: inFeatures, outFeatures: outFeatures,
               bits: bits, swigluLimit: swigluLimit)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return outBuf
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        xRotBuf: MTLBuffer,
        packedGateBuf: MTLBuffer, normsGateBuf: MTLBuffer,
        packedUpBuf: MTLBuffer,   normsUpBuf: MTLBuffer,
        codebookBuf: MTLBuffer,
        rhsIndicesBuf: MTLBuffer,
        outBuf: MTLBuffer,
        K: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2,
        swigluLimit: Float = 0
    ) {
        precondition([2, 3, 4, 8].contains(bits))
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let limitMilli = UInt32(max(0, Int((swigluLimit * 1000).rounded())))
        var meta: [UInt32] = [
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits), limitMilli,
        ]
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xRotBuf,       offset: 0, index: 0)
        enc.setBuffer(packedGateBuf, offset: 0, index: 1)
        enc.setBuffer(normsGateBuf,  offset: 0, index: 2)
        enc.setBuffer(packedUpBuf,   offset: 0, index: 3)
        enc.setBuffer(normsUpBuf,    offset: 0, index: 4)
        enc.setBuffer(codebookBuf,   offset: 0, index: 5)
        enc.setBuffer(rhsIndicesBuf, offset: 0, index: 6)
        meta.withUnsafeBufferPointer { ptr in
            enc.setBytes(ptr.baseAddress!, length: meta.count * MemoryLayout<UInt32>.stride, index: 7)
        }
        enc.setBuffer(outBuf,        offset: 0, index: 8)
        let opt = JANGTQFusedGateUpSwiGLU.optOutsPerThread
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        enc.dispatchThreads(MTLSizeMake(gridX, K, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgX, 1, 1))
    }
}


// MARK: - Gather TQ matmul (P17 OPT=20, down_proj per-row mode)

/// Computes `down[expert] @ x_rot` for K experts in one Metal dispatch.
/// Per-row: each (token, k) pair has its own input row.
public struct JANGTQGatherMatmul {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    /// Outputs per thread — must match JANGTQ_GATHER_OPT in JANGTQMatmul.metal.
    public static let optOutsPerThread: Int = 20

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_gather_tq_matmul")
    }

    public func run(
        xRotBuf: MTLBuffer,
        packedBuf: MTLBuffer,
        normsBuf: MTLBuffer,
        codebookBuf: MTLBuffer,
        rhsIndicesBuf: MTLBuffer,
        K: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) throws -> MTLBuffer {
        let outBytes = K * outFeatures * MemoryLayout<Float>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("gather_tq out alloc failed")
        }
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("command encoder alloc failed")
        }
        encode(into: enc,
               xRotBuf: xRotBuf, packedBuf: packedBuf, normsBuf: normsBuf,
               codebookBuf: codebookBuf, rhsIndicesBuf: rhsIndicesBuf, outBuf: outBuf,
               K: K, inFeatures: inFeatures, outFeatures: outFeatures, bits: bits)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return outBuf
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        xRotBuf: MTLBuffer,
        packedBuf: MTLBuffer,
        normsBuf: MTLBuffer,
        codebookBuf: MTLBuffer,
        rhsIndicesBuf: MTLBuffer,
        outBuf: MTLBuffer,
        K: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) {
        precondition([2, 3, 4, 8].contains(bits))
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        var meta: [UInt32] = [
            1, UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ]
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(xRotBuf,       offset: 0, index: 0)
        enc.setBuffer(packedBuf,     offset: 0, index: 1)
        enc.setBuffer(normsBuf,      offset: 0, index: 2)
        enc.setBuffer(codebookBuf,   offset: 0, index: 3)
        enc.setBuffer(rhsIndicesBuf, offset: 0, index: 4)
        meta.withUnsafeBufferPointer { ptr in
            enc.setBytes(ptr.baseAddress!, length: 5 * MemoryLayout<UInt32>.stride, index: 5)
        }
        enc.setBuffer(outBuf,        offset: 0, index: 6)
        let opt = JANGTQGatherMatmul.optOutsPerThread
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        enc.dispatchThreads(MTLSizeMake(gridX, K, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgX, 1, 1))
    }
}


// MARK: - Convenience bundle

/// Holds compiled pipelines for all three JANGTQ kernels. Reuse across
/// many decode steps — pipeline creation is cheap to amortize but not free
/// to hit per call.
public struct JANGTQKernels {
    public let context: MetalContext
    public let hadamard: JANGTQHadamard
    public let fusedGateUp: JANGTQFusedGateUpSwiGLU
    public let gather: JANGTQGatherMatmul

    public init(context: MetalContext) throws {
        self.context     = context
        self.hadamard    = try JANGTQHadamard(context: context)
        self.fusedGateUp = try JANGTQFusedGateUpSwiGLU(context: context)
        self.gather      = try JANGTQGatherMatmul(context: context)
    }
}
