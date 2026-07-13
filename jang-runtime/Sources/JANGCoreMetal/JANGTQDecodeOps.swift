//
// Swift bindings for the JANGTQ decode-time helper kernels.
// Created by Jinho Jang (eric@jangq.ai).
//
// Wraps `JANGTQDecodeOps.metal` — RMSNorm, RoPE, SDPA, residual add, fp32→fp16.
// These replace the CPU helpers in `JANGTQDecoderEngine.swift` so the per-layer
// hot path runs entirely on Metal at decode time.
//
// All structs hold a single `MTLComputePipelineState` plus the context they
// were created against. Reusable across many decode steps — pipeline creation
// is amortized. Output buffers are allocated per call from shared storage; for
// even tighter loops, callers can reuse pre-allocated buffers via the
// `runInto(...)` variants.
//

import Foundation
import Metal

// MARK: - RMSNorm

public struct JANGTQRMSNormKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_rmsnorm")
    }

    /// Synchronous one-shot — allocates output, dispatches, waits.
    /// Convenient for testing; for tight loops use `encode(into:)`.
    public func run(x: MTLBuffer, gamma: MTLBuffer, dim: Int, eps: Float) throws -> MTLBuffer {
        let outBytes = dim * MemoryLayout<Float16>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("rmsnorm out buffer alloc failed")
        }
        try runInto(x: x, gamma: gamma, out: outBuf, dim: dim, eps: eps)
        return outBuf
    }

    public func runInto(x: MTLBuffer, gamma: MTLBuffer, out: MTLBuffer, dim: Int, eps: Float) throws {
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("rmsnorm encoder alloc failed")
        }
        encode(into: enc, x: x, gamma: gamma, out: out, dim: dim, eps: eps)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Encode this op into a caller-managed command encoder. The caller is
    /// responsible for `endEncoding()`, `commit()`, and `waitUntilCompleted()`.
    /// Use this in the JANGTQModel forward loop to batch every kernel into
    /// one command buffer per token.
    public func encode(
        into enc: MTLComputeCommandEncoder,
        x: MTLBuffer, gamma: MTLBuffer, out: MTLBuffer, dim: Int, eps: Float
    ) {
        var params = (UInt32(dim), eps)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x,     offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(out,   offset: 0, index: 2)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 3)
        }
        let tg = MTLSizeMake(256, 1, 1)
        enc.dispatchThreads(MTLSizeMake(256, 1, 1), threadsPerThreadgroup: tg)
    }
}

// MARK: - RoPE (in-place)

public struct JANGTQRoPEKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_rope")
    }

    /// Apply RoPE in place to `qk` of shape `(nHeads, headDim)` half.
    /// `rotaryDim` is the number of dimensions rotated within each head; it
    /// defaults to the full head width for models without partial RoPE.
    public func run(
        qk: MTLBuffer,
        nHeads: Int,
        headDim: Int,
        rotaryDim: Int? = nil,
        position: Int,
        base: Float
    ) throws {
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("rope encoder alloc failed")
        }
        encode(
            into: enc,
            qk: qk,
            nHeads: nHeads,
            headDim: headDim,
            rotaryDim: rotaryDim,
            position: position,
            base: base
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        qk: MTLBuffer,
        nHeads: Int,
        headDim: Int,
        rotaryDim: Int? = nil,
        position: Int,
        base: Float
    ) {
        let effectiveRotaryDim = rotaryDim ?? headDim
        precondition(effectiveRotaryDim > 0)
        precondition(effectiveRotaryDim <= headDim)
        precondition(effectiveRotaryDim % 2 == 0)
        var params = (
            UInt32(nHeads),
            UInt32(headDim),
            UInt32(effectiveRotaryDim),
            UInt32(position),
            base
        )
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qk, offset: 0, index: 0)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 1)
        }
        let totalPairs = nHeads * (effectiveRotaryDim / 2)
        let tgWidth = min(256, totalPairs)
        enc.dispatchThreads(MTLSizeMake(totalPairs, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(max(1, tgWidth), 1, 1))
    }
}

// MARK: - SDPA (single-token decode)

public struct JANGTQSDPAKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_sdpa_decode")
    }

    public func run(
        q: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        curLen: Int, maxSeq: Int
    ) throws -> MTLBuffer {
        let outBytes = nHeads * headDim * MemoryLayout<Float16>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("sdpa out buffer alloc failed")
        }
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("sdpa encoder alloc failed")
        }
        encode(into: enc, q: q, kCache: kCache, vCache: vCache, out: outBuf,
               nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
               curLen: curLen, maxSeq: maxSeq)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return outBuf
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        q: MTLBuffer, kCache: MTLBuffer, vCache: MTLBuffer, out: MTLBuffer,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        curLen: Int, maxSeq: Int
    ) {
        let scale: Float = 1.0 / Float(headDim).squareRoot()
        var params = (
            UInt32(nHeads), UInt32(nKVHeads), UInt32(headDim),
            UInt32(curLen), UInt32(maxSeq), scale
        )
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q,      offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBuffer(out,    offset: 0, index: 3)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 4)
        }
        enc.dispatchThreadgroups(
            MTLSizeMake(nHeads, 1, 1),
            threadsPerThreadgroup: MTLSizeMake(32, 1, 1)
        )
    }
}

// MARK: - Residual add

public struct JANGTQResidualKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_residual_add")
    }

    public func run(a: MTLBuffer, b: MTLBuffer, dim: Int) throws -> MTLBuffer {
        let outBytes = dim * MemoryLayout<Float16>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("residual out buffer alloc failed")
        }
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("residual encoder alloc failed")
        }
        encode(into: enc, a: a, b: b, out: outBuf, dim: dim)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return outBuf
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, dim: Int
    ) {
        var params = UInt32(dim)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(a,   offset: 0, index: 0)
        enc.setBuffer(b,   offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 3)
        }
        let tgWidth = min(256, dim)
        enc.dispatchThreads(MTLSizeMake(dim, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgWidth, 1, 1))
    }
}

// MARK: - fp32 → fp16 cast

public struct JANGTQF32toF16Kernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_cast_f32_to_f16")
    }

    public func run(src: MTLBuffer, count: Int) throws -> MTLBuffer {
        let outBytes = count * MemoryLayout<Float16>.stride
        guard let outBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("cast out buffer alloc failed")
        }
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("cast encoder alloc failed")
        }
        encode(into: enc, src: src, dst: outBuf, count: count)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return outBuf
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        src: MTLBuffer, dst: MTLBuffer, count: Int
    ) {
        var params = UInt32(count)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 2)
        }
        let tgWidth = min(256, count)
        enc.dispatchThreads(MTLSizeMake(count, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgWidth, 1, 1))
    }
}

// MARK: - Qwen gated-attention helpers

public struct JANGTQGatedQSplitKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_qwen_gated_q_split")
    }

    public func runInto(
        qFull: MTLBuffer,
        q: MTLBuffer,
        gate: MTLBuffer,
        nHeads: Int,
        headDim: Int
    ) throws {
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("gated q split encoder alloc failed")
        }
        encode(into: enc, qFull: qFull, q: q, gate: gate, nHeads: nHeads, headDim: headDim)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        qFull: MTLBuffer,
        q: MTLBuffer,
        gate: MTLBuffer,
        nHeads: Int,
        headDim: Int
    ) {
        var params = (UInt32(nHeads), UInt32(headDim))
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qFull, offset: 0, index: 0)
        enc.setBuffer(q,     offset: 0, index: 1)
        enc.setBuffer(gate,  offset: 0, index: 2)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 3)
        }
        let total = nHeads * headDim
        let tgWidth = min(256, total)
        enc.dispatchThreads(MTLSizeMake(total, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(max(1, tgWidth), 1, 1))
    }
}

public struct JANGTQAttentionGateKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_apply_attention_gate")
    }

    public func runInto(attnOut: MTLBuffer, gate: MTLBuffer, count: Int) throws {
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("attention gate encoder alloc failed")
        }
        encode(into: enc, attnOut: attnOut, gate: gate, count: count)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        attnOut: MTLBuffer,
        gate: MTLBuffer,
        count: Int
    ) {
        var params = UInt32(count)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(attnOut, offset: 0, index: 0)
        enc.setBuffer(gate,    offset: 0, index: 1)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 2)
        }
        let tgWidth = min(256, count)
        enc.dispatchThreads(MTLSizeMake(count, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(max(1, tgWidth), 1, 1))
    }
}

// MARK: - Per-head RMSNorm (q_norm / k_norm)

public struct JANGTQHeadRMSNormKernel {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jangtq_head_rmsnorm")
    }

    public func encode(
        into enc: MTLComputeCommandEncoder,
        qk: MTLBuffer, gamma: MTLBuffer, nHeads: Int, headDim: Int, eps: Float
    ) {
        var params = (UInt32(nHeads), UInt32(headDim), eps)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qk,    offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 2)
        }
        let totalThreads = nHeads * 32
        let tgWidth = min(256, totalThreads)
        enc.dispatchThreads(MTLSizeMake(totalThreads, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgWidth, 1, 1))
    }
}

// MARK: - Bundle

/// Convenience holder for all the decode-time helper pipelines.
public struct JANGTQDecodeOps {
    public let context: MetalContext
    public let rmsnorm: JANGTQRMSNormKernel
    public let headRMSNorm: JANGTQHeadRMSNormKernel
    public let rope: JANGTQRoPEKernel
    public let sdpa: JANGTQSDPAKernel
    public let residual: JANGTQResidualKernel
    public let castF32ToF16: JANGTQF32toF16Kernel
    public let gatedQSplit: JANGTQGatedQSplitKernel
    public let attentionGate: JANGTQAttentionGateKernel

    public init(context: MetalContext) throws {
        self.context = context
        self.rmsnorm = try JANGTQRMSNormKernel(context: context)
        self.headRMSNorm = try JANGTQHeadRMSNormKernel(context: context)
        self.rope = try JANGTQRoPEKernel(context: context)
        self.sdpa = try JANGTQSDPAKernel(context: context)
        self.residual = try JANGTQResidualKernel(context: context)
        self.castF32ToF16 = try JANGTQF32toF16Kernel(context: context)
        self.gatedQSplit = try JANGTQGatedQSplitKernel(context: context)
        self.attentionGate = try JANGTQAttentionGateKernel(context: context)
    }
}
