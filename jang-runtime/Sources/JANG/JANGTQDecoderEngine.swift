/*
 * JANGTQ Decoder Engine — top-level forward pass for JANGTQ MoE models.
 * Created by Jinho Jang (eric@jangq.ai)
 *
 * This is the Swift counterpart to Python's `mlx_lm.generate(model, ...)`
 * for JANGTQ models. One forward pass produces logits for the next token.
 *
 * It assumes:
 *   - A `JANGTQModelBundle` loaded by `JANGTQLoader` (TQ MoE weights +
 *     affine 8-bit attention + half-precision norms + sidecar)
 *   - A `JANGTQKernels` bundle for the codebook+Hadamard kernels
 *   - A `JANGTQAffine8Matmul` for the 8-bit attention/embed/lm_head kernels
 *   - A pre-allocated KV cache sized for `maxSeqLen`
 *
 * Per-layer flow (matches MiniMaxDecoderLayer.__call__ in mlx_lm):
 *
 *   r = x + self_attn(input_layernorm(x))
 *   out = r + moe(post_attention_layernorm(r))
 *
 * The MoE block delegates to `JANGTQMoEBlock.runMLP` for the codebook
 * matmul path and computes the router (gate Linear + sigmoid + topk +
 * normalize) on the CPU because K=8 from 256 experts is trivially small.
 *
 * NOTES:
 *   - Only standard attention (q/k/v/o + optional q_norm/k_norm) is
 *     supported in this initial pass. MLA (GLM 5.1) needs a separate path.
 *   - Decode-only (single token at a time). Prefill must be done one
 *     token at a time too — batched prefill is a follow-up.
 *   - This implementation is intentionally straightforward for
 *     verification. Performance optimizations (KV cache reuse, command
 *     buffer batching, fused norm+matmul) come after correctness.
 */

import Foundation
import Metal
import JANGCoreMetal

/// Per-layer KV cache slot. Each row is `kv_heads × head_dim` half values
/// for one position. Allocated up front for the full `maxSeqLen`.
public final class JANGTQKVCache {
    public let device: MTLDevice
    public let nLayers: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxSeqLen: Int

    /// `keys[layer]` is one MTLBuffer of shape `(maxSeqLen, kv_heads, head_dim)` half.
    public let keys: [MTLBuffer]
    public let values: [MTLBuffer]

    public private(set) var currentLength: Int = 0

    public init(device: MTLDevice, nLayers: Int, kvHeads: Int, headDim: Int, maxSeqLen: Int) throws {
        self.device = device
        self.nLayers = nLayers
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen

        let bytesPerLayer = maxSeqLen * kvHeads * headDim * MemoryLayout<Float16>.stride
        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        for _ in 0..<nLayers {
            guard let k = device.makeBuffer(length: bytesPerLayer, options: .storageModeShared),
                  let v = device.makeBuffer(length: bytesPerLayer, options: .storageModeShared)
            else {
                throw JANGError.bufferAllocationFailed(bytesPerLayer)
            }
            ks.append(k)
            vs.append(v)
        }
        self.keys = ks
        self.values = vs
    }

    public func reset() { currentLength = 0 }

    /// Append one token's K and V (each `kv_heads × head_dim` half) to layer `layer`
    /// at the EXPLICIT `position` slot. Use this when forwarding multi-token sequences
    /// or when a single token's forward pass spans multiple layers (so each layer
    /// writes to the SAME position slot, not consecutive ones).
    public func appendKVAt(layer: Int, position: Int, kBuf: MTLBuffer, vBuf: MTLBuffer) throws {
        precondition(layer < nLayers, "layer \(layer) >= nLayers \(nLayers)")
        precondition(position < maxSeqLen, "KV cache position \(position) >= maxSeqLen \(maxSeqLen)")
        let rowBytes = kvHeads * headDim * MemoryLayout<Float16>.stride
        let offset = position * rowBytes
        let kDst = keys[layer].contents().advanced(by: offset)
        let vDst = values[layer].contents().advanced(by: offset)
        memcpy(kDst, kBuf.contents(), rowBytes)
        memcpy(vDst, vBuf.contents(), rowBytes)
    }

    /// Encode a GPU blit copy of K and V into the cache at the EXPLICIT `position`
    /// slot. The model's top-level forward passes the absolute token position to
    /// each layer; that position is what determines the cache slot, not a
    /// per-layer counter. (The previous shared `currentLength` design was buggy
    /// because each layer would advance it, producing 62 advances per token.)
    public func encodeAppendKVAt(
        into blitEnc: MTLBlitCommandEncoder,
        layer: Int, position: Int,
        kBuf: MTLBuffer, vBuf: MTLBuffer
    ) {
        precondition(layer < nLayers)
        precondition(position < maxSeqLen)
        let rowBytes = kvHeads * headDim * MemoryLayout<Float16>.stride
        let offset = position * rowBytes
        blitEnc.copy(from: kBuf, sourceOffset: 0,
                     to: keys[layer], destinationOffset: offset, size: rowBytes)
        blitEnc.copy(from: vBuf, sourceOffset: 0,
                     to: values[layer], destinationOffset: offset, size: rowBytes)
    }

    /// Advance the shared currentLength counter by ONE TOKEN (not per layer).
    /// The model's top-level forward calls this exactly once per forward pass.
    public func advanceOneToken() {
        currentLength += 1
    }
}

/// CPU helper: dequantize a single embedding row from MLX-format 8-bit
/// quantized embed_tokens. For decode (one token per step) this is a few
/// KB of work — running on CPU is simpler than dispatching a kernel.
public func jangtqDequantizeEmbedRow(
    embed: JANGTQAffineWeight, tokenId: Int, device: MTLDevice
) throws -> MTLBuffer {
    precondition(tokenId >= 0 && tokenId < embed.outFeatures,
        "tokenId \(tokenId) out of range [0, \(embed.outFeatures))")
    let inF = embed.inFeatures
    let gs = embed.groupSize
    let nGroups = inF / gs
    let valsPerU32 = 32 / embed.bits
    let packedIn = inF / valsPerU32

    let qPtr = embed.qweight.contents()
        .advanced(by: tokenId * packedIn * MemoryLayout<UInt32>.stride)
        .bindMemory(to: UInt32.self, capacity: packedIn)
    let sPtr = embed.scales.contents()
        .advanced(by: tokenId * nGroups * MemoryLayout<Float16>.stride)
        .bindMemory(to: Float16.self, capacity: nGroups)
    let bPtr = embed.biases.contents()
        .advanced(by: tokenId * nGroups * MemoryLayout<Float16>.stride)
        .bindMemory(to: Float16.self, capacity: nGroups)

    let outBytes = inF * MemoryLayout<Float16>.stride
    guard let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared) else {
        throw JANGError.bufferAllocationFailed(outBytes)
    }
    let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: inF)

    let mask: UInt32 = (1 << embed.bits) - 1
    for g in 0..<nGroups {
        let scale = Float(sPtr[g])
        let bias  = Float(bPtr[g])
        let gStart = g * gs
        let wordsPerGroup = gs / valsPerU32
        for w in 0..<wordsPerGroup {
            let word = qPtr[(gStart / valsPerU32) + w]
            for k in 0..<valsPerU32 {
                let q = (word >> (k * embed.bits)) & mask
                let dq = Float(q) * scale + bias
                outPtr[gStart + w * valsPerU32 + k] = Float16(dq)
            }
        }
    }
    return outBuf
}

/// CPU RMSNorm — for decode T=1 this is one vector of `hidden` floats and
/// running on CPU avoids a Metal command buffer allocation. Order-of-
/// magnitude cheaper than the GPU path at decode shape.
public func jangtqRMSNormCPU(
    inputBuf: MTLBuffer, gammaBuf: MTLBuffer, dim: Int, eps: Float
) -> MTLBuffer {
    let inPtr  = inputBuf.contents().bindMemory(to: Float16.self, capacity: dim)
    let gPtr   = gammaBuf.contents().bindMemory(to: Float16.self, capacity: dim)
    let device = inputBuf.device
    let outBuf = device.makeBuffer(length: dim * MemoryLayout<Float16>.stride,
                                    options: .storageModeShared)!
    let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: dim)

    var sumSq: Float = 0
    for i in 0..<dim {
        let v = Float(inPtr[i])
        sumSq += v * v
    }
    let rms = 1.0 / Float((sumSq / Float(dim) + eps).squareRoot())
    for i in 0..<dim {
        let scaled = Float(inPtr[i]) * rms * Float(gPtr[i])
        outPtr[i] = Float16(scaled)
    }
    return outBuf
}

/// CPU residual add — `out = a + b`, both half. Tiny per call.
public func jangtqResidualAddCPU(
    a: MTLBuffer, b: MTLBuffer, dim: Int
) -> MTLBuffer {
    let aPtr = a.contents().bindMemory(to: Float16.self, capacity: dim)
    let bPtr = b.contents().bindMemory(to: Float16.self, capacity: dim)
    let device = a.device
    let outBuf = device.makeBuffer(length: dim * MemoryLayout<Float16>.stride,
                                    options: .storageModeShared)!
    let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: dim)
    for i in 0..<dim {
        outPtr[i] = Float16(Float(aPtr[i]) + Float(bPtr[i]))
    }
    return outBuf
}

/// Router variants shipped by this runtime.
///
/// The routed-expert selection + score math differs by arch family:
///   - `.sigmoidBias` : MiniMax / GLM / DeepSeek V3 — sigmoid(logits) + bias,
///     topk on biased values, selected scores come from the ORIGINAL sigmoid
///     (not biased), finally normalized to sum to 1.
///   - `.softmaxTopK` : Qwen3-Next / Qwen3.5 / Qwen3.6 — softmax(logits), topk,
///     optionally renormalize selected scores (`norm_topk_prob`).
public enum JANGTQRouterVariant: Sendable {
    case sigmoidBias
    case softmaxTopK(renormalize: Bool)
}

/// CPU router for JANGTQ MoE models. K is typically 8 of up to 256 experts —
/// staying on CPU avoids two Metal dispatches per layer and beats a tiny Metal
/// argpartition kernel at this shape.
///
/// Returns both `MTLBuffer` views (for kernels that consume indices/scores) and
/// `[Int]/[Float]` copies (for the subsequent CPU combine).
///
/// `eScoreBias` is only used by `.sigmoidBias`; pass `nil` for the Qwen path.
public func jangtqRouterCPU(
    gates: [Float],
    eScoreBias: [Float]?,
    k: Int,
    variant: JANGTQRouterVariant,
    device: MTLDevice,
    disabledExperts: Set<Int> = []
) throws -> (indicesBuf: MTLBuffer, scoresBuf: MTLBuffer, indices: [Int], scores: [Float]) {
    let n = gates.count
    guard k > 0 else {
        throw JANGError.invalidFormat("router top-k must be positive, got \(k)")
    }
    let invalidDisabled = disabledExperts.filter { $0 < 0 || $0 >= n }
    guard invalidDisabled.isEmpty else {
        throw JANGError.invalidFormat(
            "expert mask contains ids outside 0..<\(n): \(invalidDisabled.sorted())"
        )
    }
    let eligible = (0..<n).filter { !disabledExperts.contains($0) }
    guard eligible.count >= k else {
        throw JANGError.invalidFormat(
            "expert mask leaves \(eligible.count) active experts, fewer than top-k \(k)"
        )
    }

    let topIdx: [Int]
    var sel: [Float]

    switch variant {
    case .sigmoidBias:
        guard let bias = eScoreBias, bias.count == n else {
            throw JANGError.invalidFormat(
                "sigmoidBias router requires e_score_correction_bias of length \(n)"
            )
        }
        var sigmoid = [Float](repeating: 0, count: n)
        for i in 0..<n { sigmoid[i] = 1.0 / (1.0 + Foundation.exp(-gates[i])) }
        var biased = [Float](repeating: 0, count: n)
        for i in 0..<n { biased[i] = sigmoid[i] + bias[i] }
        let sortedIdx = eligible.sorted(by: { biased[$0] > biased[$1] })
        topIdx = Array(sortedIdx.prefix(k))
        sel = [Float](repeating: 0, count: k)
        for i in 0..<k { sel[i] = sigmoid[topIdx[i]] }
        var s: Float = 0
        for v in sel { s += v }
        let denom = s + 1e-20
        for i in 0..<k { sel[i] /= denom }

    case .softmaxTopK(let renormalize):
        // Precise softmax with max-subtract for numerical stability.
        var maxV: Float = -.infinity
        for v in gates { if v > maxV { maxV = v } }
        var exps = [Float](repeating: 0, count: n)
        var sumExp: Float = 0
        for i in 0..<n {
            let e = Foundation.exp(gates[i] - maxV)
            exps[i] = e
            sumExp += e
        }
        let invSum = 1.0 / (sumExp + 1e-20)
        var scores = [Float](repeating: 0, count: n)
        for i in 0..<n { scores[i] = exps[i] * invSum }
        let sortedIdx = eligible.sorted(by: { scores[$0] > scores[$1] })
        topIdx = Array(sortedIdx.prefix(k))
        sel = [Float](repeating: 0, count: k)
        for i in 0..<k { sel[i] = scores[topIdx[i]] }
        if renormalize || !disabledExperts.isEmpty {
            var s: Float = 0
            for v in sel { s += v }
            let denom = s + 1e-20
            for i in 0..<k { sel[i] /= denom }
        }
    }

    // Pack into MTLBuffers
    let idxBuf = device.makeBuffer(length: k * MemoryLayout<UInt32>.stride,
                                    options: .storageModeShared)!
    let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: k)
    for i in 0..<k { idxPtr[i] = UInt32(topIdx[i]) }

    let scoresBuf = device.makeBuffer(length: k * MemoryLayout<Float>.stride,
                                       options: .storageModeShared)!
    let scoresPtr = scoresBuf.contents().bindMemory(to: Float.self, capacity: k)
    for i in 0..<k { scoresPtr[i] = sel[i] }

    return (idxBuf, scoresBuf, topIdx, sel)
}

/// CPU half-Linear matmul for the router gate (small: hidden × n_experts).
/// `weight` is (n_experts, hidden) half. Returns (n_experts,) Float.
public func jangtqHalfMatmul(
    x: MTLBuffer, weight: MTLBuffer, inFeatures: Int, outFeatures: Int
) -> [Float] {
    let xPtr = x.contents().bindMemory(to: Float16.self, capacity: inFeatures)
    let wPtr = weight.contents().bindMemory(to: Float16.self, capacity: inFeatures * outFeatures)
    var out = [Float](repeating: 0, count: outFeatures)
    for r in 0..<outFeatures {
        var acc: Float = 0
        for c in 0..<inFeatures {
            acc += Float(wPtr[r * inFeatures + c]) * Float(xPtr[c])
        }
        out[r] = acc
    }
    return out
}

/// Top-level decoder for JANGTQ MoE models.
public final class JANGTQDecoderEngine {
    public let bundle: JANGTQModelBundle
    public let kernels: JANGTQKernels
    public let affine8: JANGTQAffine8Matmul
    public let ops: JANGTQDecodeOps
    public let context: MetalContext
    public let cache: JANGTQKVCache

    /// Per-layer cached MoE block (lazy — created on first use).
    private var moeBlocks: [Int: JANGTQMoEBlock] = [:]
    /// Per-layer staging buffer for fp32→fp16 cast between fused_gu and hadamard.
    private var stagingBuffers: [Int: MTLBuffer] = [:]
    private let moePrefix: String
    private let swigluLimit: Float

    public init(
        bundle: JANGTQModelBundle,
        context: MetalContext,
        kernels: JANGTQKernels,
        affine8: JANGTQAffine8Matmul,
        ops: JANGTQDecodeOps,
        cache: JANGTQKVCache,
        moePrefix: String = "block_sparse_moe",
        swigluLimit: Float = 0
    ) {
        self.bundle = bundle
        self.context = context
        self.kernels = kernels
        self.affine8 = affine8
        self.ops = ops
        self.cache = cache
        self.moePrefix = moePrefix
        self.swigluLimit = swigluLimit
    }

    /// Lazily build the MoE block for a given layer.
    private func moeBlock(for layer: Int, topK: Int) throws -> JANGTQMoEBlock {
        if let b = moeBlocks[layer] { return b }
        let b = try JANGTQMoEBlock(
            layerIndex: layer,
            layerPrefix: resolvedMoEPrefix(for: layer),
            bundle: bundle, kernels: kernels,
            topK: topK,
            swigluLimit: swigluLimit
        )
        moeBlocks[layer] = b
        return b
    }

    private func resolvedMoEPrefix(for layer: Int) -> String {
        let candidates = [
            "language_model.model.layers.\(layer).mlp",
            "model.layers.\(layer).mlp",
            "language_model.layers.\(layer).mlp",
            "model.layers.\(layer).\(moePrefix)",
        ]
        return candidates.first { prefix in
            bundle.halfTensors["\(prefix).gate.weight"] != nil
                || bundle.weights["\(prefix).switch_mlp.gate_proj"] != nil
        } ?? candidates.last!
    }

    /// CPU helper to convert an fp32 MTLBuffer to half MTLBuffer.
    private func float32ToHalf(_ src: MTLBuffer, count: Int) -> MTLBuffer {
        let device = context.device
        let dst = device.makeBuffer(length: count * MemoryLayout<Float16>.stride,
                                     options: .storageModeShared)!
        let srcPtr = src.contents().bindMemory(to: Float.self, capacity: count)
        let dstPtr = dst.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { dstPtr[i] = Float16(srcPtr[i]) }
        return dst
    }

    /// Run one MoE block (router + JANGTQMoEBlock + combine) on a half buffer.
    /// Returns a half buffer of shape `(hidden,)`.
    /// Uses the encode-into-cb fast path for the 4 MLP kernels (1 sync per layer
    /// instead of 4) plus CPU router and CPU combine.
    ///
    /// Auto-selects router variant based on what the bundle provides:
    ///   - If `e_score_correction_bias` is in `halfTensors` → sigmoid+bias (MiniMax/GLM).
    ///   - Else → softmax top-k (Qwen3-Next/3.5/3.6). Renormalize based on config.
    ///
    /// When the MoE block has a `shared_expert`, its output is added on top of
    /// the routed combine BEFORE the final return (gated by sigmoid(shared_expert_gate(x))).
    public func runMoE(
        layer: Int,
        normedX: MTLBuffer,
        hidden: Int,
        k: Int,
        traceConfig: JANGExpertTraceConfig? = nil,
        traceCollector: JANGExpertTraceCollector? = nil,
        tokenIndex: Int? = nil
    ) throws -> MTLBuffer {
        let topKOverride = traceConfig?.mask?.topKOverride
        if let topKOverride {
            guard topKOverride > 0 else {
                throw JANGError.invalidFormat("topKOverride must be positive, got \(topKOverride)")
            }
            guard topKOverride <= k else {
                throw JANGError.invalidFormat(
                    "topKOverride \(topKOverride) cannot exceed trained top-k \(k)"
                )
            }
        }
        let effectiveK = topKOverride ?? k
        let disabledExperts = traceConfig?.mask?.disabledExperts(for: layer) ?? []

        // Layer prefix: support VLM Qwen (`language_model.model.layers`),
        // text-only Qwen (`model.layers`), rare `language_model.layers`, and
        // the older MiniMax/GLM `block_sparse_moe` layout.
        let prefix = resolvedMoEPrefix(for: layer)

        // Router gate weight (half, shape [n_experts, hidden])
        guard let gateWBuf = bundle.halfTensors["\(prefix).gate.weight"] else {
            throw JANGError.tensorNotFound("\(prefix).gate.weight")
        }
        let nExperts = gateWBuf.length / (hidden * MemoryLayout<Float16>.stride)
        let gates = jangtqHalfMatmul(
            x: normedX, weight: gateWBuf, inFeatures: hidden, outFeatures: nExperts
        )

        // Variant = sigmoidBias if bias present, else softmax (+ renormalize per config)
        let biasKey = "\(prefix).e_score_correction_bias"
        let biasBuf = bundle.halfTensors[biasKey]
        let variant: JANGTQRouterVariant
        var biasArr: [Float]? = nil
        if let biasBuf = biasBuf {
            var ba = [Float](repeating: 0, count: nExperts)
            let p = biasBuf.contents().bindMemory(to: Float16.self, capacity: nExperts)
            for i in 0..<nExperts { ba[i] = Float(p[i]) }
            biasArr = ba
            variant = .sigmoidBias
        } else {
            // Match mlx-lm/vMLX ModelArgs: missing norm_topk_prob defaults to false.
            // Some Qwen checkpoints omit the key; renormalizing those top-k scores
            // changes every MoE residual scale and breaks parity.
            let renorm = bundle.config.model.normTopkProb ?? false
            variant = .softmaxTopK(renormalize: renorm)
        }

        let routed = try jangtqRouterCPU(
            gates: gates, eScoreBias: biasArr, k: effectiveK,
            variant: variant, device: context.device,
            disabledExperts: disabledExperts
        )

        if let traceCollector, let tokenIndex {
            traceCollector.record(
                JANGExpertRouteRecord(
                    tokenIndex: tokenIndex,
                    layer: layer,
                    selectedExperts: routed.indices,
                    scores: routed.scores,
                    disabledExperts: disabledExperts.sorted(),
                    effectiveTopK: effectiveK,
                    entropy: Self.routerEntropy(routed.scores)
                ),
                config: traceConfig
            )
        }

        // Run the MoE block via single-cb encode path.
        let block = try moeBlock(for: layer, topK: k)

        // Get or create per-layer staging buffer (K * out_features * sizeof(half))
        let stagingBytes = effectiveK * block.outFeatures * MemoryLayout<Float16>.stride
        let staging: MTLBuffer
        if let cached = stagingBuffers[layer], cached.length >= stagingBytes {
            staging = cached
        } else {
            guard let newBuf = context.device.makeBuffer(length: stagingBytes,
                                                          options: .storageModeShared) else {
                throw JANGError.bufferAllocationFailed(stagingBytes)
            }
            stagingBuffers[layer] = newBuf
            staging = newBuf
        }

        // ONE command buffer for the entire MoE MLP path
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGError.inferenceError("moe cb alloc failed")
        }
        block.encode(
            into: enc,
            xHalfBuf: normedX,
            selectedExpertsBuf: routed.indicesBuf,
            K: effectiveK,
            ops: ops,
            xActHalfStaging: staging
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Combine routed experts: out[h] = sum_k scores[k] * y[k, h]   (CPU — small)
        let yPtr = block.yOut.contents().bindMemory(to: Float.self, capacity: effectiveK * hidden)
        let combined = context.device.makeBuffer(
            length: hidden * MemoryLayout<Float16>.stride, options: .storageModeShared
        )!
        let outPtr = combined.contents().bindMemory(to: Float16.self, capacity: hidden)
        for h in 0..<hidden {
            var acc: Float = 0
            for ki in 0..<effectiveK {
                acc += routed.scores[ki] * yPtr[ki * hidden + h]
            }
            outPtr[h] = Float16(acc)
        }

        // Shared expert (Qwen3-Next/3.5/3.6 etc): y += sigmoid(gate(x)) * shared(x).
        // Uses affine 8-bit GEMVs through `affine8`, then a small CPU combine.
        if block.hasSharedExpert,
           let shG = block.sharedGateProj,
           let shU = block.sharedUpProj,
           let shD = block.sharedDownProj
        {
            let sharedY = try runSharedExpert(
                xHalfBuf: normedX,
                gate: shG, up: shU, down: shD,
                gateScalar: block.sharedGateScalar,
                gateScalarHalf: block.sharedGateScalarHalf,
                hidden: hidden,
                swigluLimit: block.swigluLimit
            )
            let shPtr = sharedY.contents().bindMemory(to: Float16.self, capacity: hidden)
            for h in 0..<hidden {
                outPtr[h] = Float16(Float(outPtr[h]) + Float(shPtr[h]))
            }
        }

        return combined
    }

    private static func routerEntropy(_ scores: [Float]) -> Float {
        var entropy: Float = 0
        for score in scores where score > 0 {
            entropy -= score * Foundation.log(score)
        }
        return entropy
    }

    /// Run the dense shared-expert MLP: `sigmoid(gate_scalar(x)) * down(SiLU(gate(x)) * up(x))`.
    /// Returns a half MTLBuffer of shape `(hidden,)`.
    ///
    /// All 4 matmuls go through the 8-bit affine GEMV kernel (outputs fp32).
    /// SwiGLU + scalar-gate compose on CPU because the intermediate dim is
    /// tiny (512 on Qwen3.6) and we've already paid a sync per matmul.
    private func runSharedExpert(
        xHalfBuf: MTLBuffer,
        gate: JANGTQAffineWeight,
        up: JANGTQAffineWeight,
        down: JANGTQAffineWeight,
        gateScalar: JANGTQAffineWeight?,
        gateScalarHalf: MTLBuffer?,
        hidden: Int,
        swigluLimit: Float = 0
    ) throws -> MTLBuffer {
        precondition(gate.bits == 8 && up.bits == 8 && down.bits == 8,
            "shared_expert affine weights must be 8-bit")
        let interFeat = gate.outFeatures
        precondition(interFeat == up.outFeatures, "shared_expert gate/up inter-dim mismatch")

        func affineRun(_ w: JANGTQAffineWeight, _ x: MTLBuffer) throws -> MTLBuffer {
            return try affine8.run(
                qweightBuf: w.qweight, scalesBuf: w.scales, biasesBuf: w.biases,
                xBuf: x,
                inFeatures: w.inFeatures, outFeatures: w.outFeatures,
                groupSize: w.groupSize
            )
        }

        // 1. gate_proj(x), up_proj(x) → fp32 (inter,)
        let gOutF32 = try affineRun(gate, xHalfBuf)
        let uOutF32 = try affineRun(up,   xHalfBuf)

        // 2. SwiGLU into a fresh f16 buffer so down_proj can read it.
        let gPtr = gOutF32.contents().bindMemory(to: Float.self, capacity: interFeat)
        let uPtr = uOutF32.contents().bindMemory(to: Float.self, capacity: interFeat)
        let actBuf = context.device.makeBuffer(
            length: interFeat * MemoryLayout<Float16>.stride, options: .storageModeShared
        )!
        let aPtr = actBuf.contents().bindMemory(to: Float16.self, capacity: interFeat)
        for i in 0..<interFeat {
            var gv = gPtr[i]
            var uv = uPtr[i]
            if swigluLimit > 0 {
                gv = min(gv, swigluLimit)
                uv = max(min(uv, swigluLimit), -swigluLimit)
            }
            let silu = gv / (1.0 + Foundation.exp(-gv))
            aPtr[i] = Float16(silu * uv)
        }

        // 3. down_proj(act) → (hidden,) fp32
        let dOutF32 = try affineRun(down, actBuf)

        // 4. Optional scalar sigmoid gate. `gateScalar.weight` is shape (1, hidden).
        var gateVal: Float = 1.0
        if let gs = gateScalar {
            precondition(gs.bits == 8, "shared_expert_gate must be 8-bit")
            let gateOutF32 = try affineRun(gs, xHalfBuf)   // (1,) fp32
            let gp = gateOutF32.contents().bindMemory(to: Float.self, capacity: 1)
            gateVal = 1.0 / (1.0 + Foundation.exp(-gp[0]))
        } else if let gs = gateScalarHalf {
            let gateOut = jangtqHalfMatmul(
                x: xHalfBuf,
                weight: gs,
                inFeatures: hidden,
                outFeatures: 1
            )
            gateVal = 1.0 / (1.0 + Foundation.exp(-gateOut[0]))
        }

        // 5. Write final half buffer, applying scalar gate.
        let outBuf = context.device.makeBuffer(
            length: hidden * MemoryLayout<Float16>.stride, options: .storageModeShared
        )!
        let dp = dOutF32.contents().bindMemory(to: Float.self, capacity: hidden)
        let op = outBuf.contents().bindMemory(to: Float16.self, capacity: hidden)
        for h in 0..<hidden {
            op[h] = Float16(dp[h] * gateVal)
        }
        return outBuf
    }
}
