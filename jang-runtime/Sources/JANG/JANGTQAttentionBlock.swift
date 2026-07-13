/*
 * JANGTQ Attention Block — single-layer self-attention forward pass.
 * Created by Jinho Jang (eric@jangq.ai)
 *
 * Implements the MiniMax M2.7 attention block:
 *
 *   queries = q_proj(x)        # 8-bit affine GEMV
 *   keys    = k_proj(x)        # 8-bit affine GEMV
 *   values  = v_proj(x)        # 8-bit affine GEMV
 *   if use_qk_norm:
 *       queries = q_norm(queries)
 *       keys    = k_norm(keys)
 *   reshape Q to (n_heads, head_dim), K/V to (n_kv_heads, head_dim)
 *   Q = rope(Q, offset=cache.length)
 *   K = rope(K, offset=cache.length)
 *   cache.append(K, V)
 *   out = sdpa(Q, cache.K[:offset+1], cache.V[:offset+1], scale=1/sqrt(head_dim))
 *   reshape out to (n_heads * head_dim,)
 *   return o_proj(out)         # 8-bit affine GEMV
 *
 * Decode-only (T=1). Prefill needs a separate path. Most of the math runs
 * on CPU because for T=1 + small head_dim (64-128) it's ~µs per layer and
 * avoids the GPU command-buffer dispatch overhead.
 *
 * The four matmuls (q/k/v/o) DO go to GPU via JANGTQAffine8Matmul because
 * the weights are large (3 GB total for 62 layers).
 */

import Foundation
import Metal
import JANGCoreMetal

public final class JANGTQAttentionBlock {
    public let layerIndex: Int
    public let hidden: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let useQKNorm: Bool
    public let ropeBase: Float
    /// Dimensions rotated by RoPE per head. For Qwen3.6 with
    /// `partial_rotary_factor: 0.25` and `head_dim: 256` this is 64.
    public let ropeDim: Int
    public let scale: Float

    /// True when `q_proj` outputs `2 × nHeads × headDim` (Qwen3-Next-style
    /// per-head sigmoid gate on attention output).
    public let attnOutputGate: Bool

    public let qProj: JANGTQAffineWeight
    public let kProj: JANGTQAffineWeight
    public let vProj: JANGTQAffineWeight
    public let oProj: JANGTQAffineWeight

    public let inputLayernorm: MTLBuffer
    public let qNorm: MTLBuffer?
    public let kNorm: MTLBuffer?

    public let normEps: Float
    public let affine8: JANGTQAffine8Matmul
    public let ops: JANGTQDecodeOps

    // Pre-allocated per-layer scratch buffers (allocated once at init).
    // Reused across decode steps to avoid per-call allocation overhead.
    public let normedX: MTLBuffer       // (hidden,) fp16
    /// `qFp32` capacity is `qProj.outFeatures * 4`. For MiniMax that's `nHeads*headDim`;
    /// for Qwen3.6 with `attnOutputGate` that's `2 * nHeads * headDim`.
    public let qFp32: MTLBuffer
    public let kFp32: MTLBuffer         // (nKVHeads * headDim,) fp32
    public let vFp32: MTLBuffer         // (nKVHeads * headDim,) fp32
    public let qHalfFull: MTLBuffer     // full qProj output, fp16 — includes gate if present
    public let qHalf: MTLBuffer         // (nHeads * headDim,) query half buffer
    public let gateHalf: MTLBuffer?     // (nHeads * headDim,) gate half buffer, nil when no gate
    public let kHalf: MTLBuffer
    public let vHalf: MTLBuffer
    public let attnOut: MTLBuffer       // (nHeads * headDim,) fp16
    public let oFp32: MTLBuffer         // (hidden,) fp32
    public let oHalf: MTLBuffer         // (hidden,) fp16

    /// Legacy initializer (MiniMax/GLM — `model.layers.L.self_attn.*`, no output gate).
    public convenience init(
        layerIndex: Int,
        config: ModelConfig,
        bundle: JANGTQModelBundle,
        affine8: JANGTQAffine8Matmul,
        ops: JANGTQDecodeOps
    ) throws {
        try self.init(
            layerIndex: layerIndex, config: config, bundle: bundle,
            layerPrefix: "model.layers.\(layerIndex).self_attn",
            inputLayernormPath: "model.layers.\(layerIndex).input_layernorm.weight",
            attnOutputGate: config.hasAttnOutputGate,
            ropeDim: nil,
            affine8: affine8, ops: ops
        )
    }

    /// Full initializer — explicit prefix + optional attn_output_gate + partial-rope dim.
    ///
    /// `layerPrefix` should include `.self_attn` (e.g.,
    /// `"language_model.model.layers.3.self_attn"`).
    /// `ropeDim` defaults to `config.headDim`; pass `nil` to auto-derive from
    /// `partial_rotary_factor` in the config when present.
    public init(
        layerIndex: Int,
        config: ModelConfig,
        bundle: JANGTQModelBundle,
        layerPrefix: String,
        inputLayernormPath: String,
        attnOutputGate: Bool,
        ropeDim explicitRopeDim: Int?,
        affine8: JANGTQAffine8Matmul,
        ops: JANGTQDecodeOps
    ) throws {
        self.layerIndex = layerIndex
        self.affine8 = affine8
        self.ops = ops

        self.hidden = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.normEps = config.normEps
        self.ropeBase = config.ropeBase
        self.attnOutputGate = attnOutputGate
        self.scale = 1.0 / Float(self.headDim).squareRoot()

        if let rd = explicitRopeDim {
            self.ropeDim = rd
        } else if let prf = config.partialRotaryFactor, prf > 0 && prf < 1 {
            self.ropeDim = Int((Double(self.headDim) * prf).rounded())
        } else {
            self.ropeDim = self.headDim
        }

        guard let q = bundle.affineWeights["\(layerPrefix).q_proj"],
              let k = bundle.affineWeights["\(layerPrefix).k_proj"],
              let v = bundle.affineWeights["\(layerPrefix).v_proj"],
              let o = bundle.affineWeights["\(layerPrefix).o_proj"]
        else {
            throw JANGError.tensorNotFound("attention layer \(layerIndex) missing q/k/v/o_proj at \(layerPrefix)")
        }
        self.qProj = q
        self.kProj = k
        self.vProj = v
        self.oProj = o

        // Verify qProj output dim matches the gate flag: 2× heads when gated.
        let qBase = self.nHeads * self.headDim
        let expectedQOut = attnOutputGate ? (2 * qBase) : qBase
        guard qProj.outFeatures == expectedQOut else {
            throw JANGError.invalidFormat(
                "q_proj.outFeatures=\(qProj.outFeatures) ≠ expected \(expectedQOut) " +
                "(attn_output_gate=\(attnOutputGate), nHeads=\(nHeads), headDim=\(headDim))"
            )
        }

        guard let normW = bundle.halfTensors[inputLayernormPath] else {
            throw JANGError.tensorNotFound(inputLayernormPath)
        }
        self.inputLayernorm = normW

        self.qNorm = bundle.halfTensors["\(layerPrefix).q_norm.weight"]
        self.kNorm = bundle.halfTensors["\(layerPrefix).k_norm.weight"]
        self.useQKNorm = (qNorm != nil) || (kNorm != nil)

        // Pre-allocate scratch buffers
        let dev = affine8.context.device
        let f16 = MemoryLayout<Float16>.stride
        let f32 = MemoryLayout<Float>.stride
        let qOutDim = qProj.outFeatures
        let kvDim = self.nKVHeads * self.headDim
        func mkBuf(_ bytes: Int) throws -> MTLBuffer {
            guard let b = dev.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw JANGError.bufferAllocationFailed(bytes) }
            return b
        }
        self.normedX    = try mkBuf(self.hidden * f16)
        self.qFp32      = try mkBuf(qOutDim * f32)
        self.kFp32      = try mkBuf(kvDim   * f32)
        self.vFp32      = try mkBuf(kvDim   * f32)
        self.qHalfFull  = try mkBuf(qOutDim * f16)
        self.qHalf      = try mkBuf(qBase * f16)
        self.kHalf      = try mkBuf(kvDim   * f16)
        self.vHalf      = try mkBuf(kvDim   * f16)
        self.attnOut    = try mkBuf(qBase   * f16)
        self.oFp32      = try mkBuf(self.hidden * f32)
        self.oHalf      = try mkBuf(self.hidden * f16)

        if attnOutputGate {
            self.gateHalf = try mkBuf(qBase * f16)
        } else {
            self.gateHalf = nil
        }
    }

    /// Run the attention block for a single decode token.
    /// Qwen gated attention stays on Metal: q_proj is split per head into
    /// query/gate buffers, SDPA is gated, then o_proj consumes the gated output.
    ///
    /// `position` is the absolute token position (0-indexed) — used as the
    /// KV cache slot index. The model passes the SAME position to every
    /// layer's attention.forward (because each layer is processing the
    /// same token), and the per-token cache advance happens at the model
    /// level after all layers have written their slot.
    public func forward(
        x: MTLBuffer,
        cache: JANGTQKVCache,
        position: Int
    ) throws -> MTLBuffer {
        precondition(position < cache.maxSeqLen,
                     "position \(position) >= maxSeqLen \(cache.maxSeqLen)")

        guard let cb = affine8.context.queue.makeCommandBuffer() else {
            throw JANGError.inferenceError("attn cb alloc failed")
        }

        // === Compute encoder 1: norm + Q/K/V + cast + optional gated-Q split + qk_norm + RoPE ===
        guard let enc1 = cb.makeComputeCommandEncoder() else {
            throw JANGError.inferenceError("attn enc1 alloc failed")
        }
        ops.rmsnorm.encode(into: enc1, x: x, gamma: inputLayernorm,
                           out: normedX, dim: hidden, eps: normEps)
        affine8.encode(into: enc1,
                       qweightBuf: qProj.qweight, scalesBuf: qProj.scales,
                       biasesBuf: qProj.biases, xBuf: normedX, yBuf: qFp32,
                       inFeatures: qProj.inFeatures, outFeatures: qProj.outFeatures,
                       groupSize: qProj.groupSize)
        affine8.encode(into: enc1,
                       qweightBuf: kProj.qweight, scalesBuf: kProj.scales,
                       biasesBuf: kProj.biases, xBuf: normedX, yBuf: kFp32,
                       inFeatures: kProj.inFeatures, outFeatures: kProj.outFeatures,
                       groupSize: kProj.groupSize)
        affine8.encode(into: enc1,
                       qweightBuf: vProj.qweight, scalesBuf: vProj.scales,
                       biasesBuf: vProj.biases, xBuf: normedX, yBuf: vFp32,
                       inFeatures: vProj.inFeatures, outFeatures: vProj.outFeatures,
                       groupSize: vProj.groupSize)
        if attnOutputGate {
            ops.castF32ToF16.encode(into: enc1, src: qFp32, dst: qHalfFull, count: Int(qProj.outFeatures))
        } else {
            ops.castF32ToF16.encode(into: enc1, src: qFp32, dst: qHalf, count: nHeads * headDim)
        }
        ops.castF32ToF16.encode(into: enc1, src: kFp32, dst: kHalf, count: nKVHeads * headDim)
        ops.castF32ToF16.encode(into: enc1, src: vFp32, dst: vHalf, count: nKVHeads * headDim)
        if attnOutputGate, let gate = gateHalf {
            ops.gatedQSplit.encode(
                into: enc1,
                qFull: qHalfFull,
                q: qHalf,
                gate: gate,
                nHeads: nHeads,
                headDim: headDim
            )
        }
        if let qN = qNorm {
            ops.headRMSNorm.encode(into: enc1, qk: qHalf, gamma: qN,
                                   nHeads: nHeads, headDim: headDim, eps: normEps)
        }
        if let kN = kNorm {
            ops.headRMSNorm.encode(into: enc1, qk: kHalf, gamma: kN,
                                   nHeads: nKVHeads, headDim: headDim, eps: normEps)
        }
        ops.rope.encode(into: enc1, qk: qHalf, nHeads: nHeads, headDim: headDim,
                        rotaryDim: ropeDim, position: position, base: ropeBase)
        ops.rope.encode(into: enc1, qk: kHalf, nHeads: nKVHeads, headDim: headDim,
                        rotaryDim: ropeDim, position: position, base: ropeBase)
        enc1.endEncoding()

        // === Blit encoder: append K, V to cache at THIS token's position slot ===
        guard let blitEnc = cb.makeBlitCommandEncoder() else {
            throw JANGError.inferenceError("attn blit alloc failed")
        }
        cache.encodeAppendKVAt(into: blitEnc, layer: layerIndex,
                                position: position, kBuf: kHalf, vBuf: vHalf)
        blitEnc.endEncoding()

        // === Compute encoder 3: SDPA ===
        // SDPA reads positions [0..position] inclusive (current token included)
        let curLen = position + 1
        guard let enc3 = cb.makeComputeCommandEncoder() else {
            throw JANGError.inferenceError("attn enc3 alloc failed")
        }
        ops.sdpa.encode(into: enc3, q: qHalf,
                        kCache: cache.keys[layerIndex], vCache: cache.values[layerIndex],
                        out: attnOut,
                        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        curLen: curLen, maxSeq: cache.maxSeqLen)
        if attnOutputGate, let gate = gateHalf {
            ops.attentionGate.encode(
                into: enc3,
                attnOut: attnOut,
                gate: gate,
                count: nHeads * headDim
            )
        }
        affine8.encode(into: enc3,
                       qweightBuf: oProj.qweight, scalesBuf: oProj.scales,
                       biasesBuf: oProj.biases, xBuf: attnOut, yBuf: oFp32,
                       inFeatures: oProj.inFeatures, outFeatures: oProj.outFeatures,
                       groupSize: oProj.groupSize)
        ops.castF32ToF16.encode(into: enc3, src: oFp32, dst: oHalf, count: hidden)
        enc3.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // NOTE: cache advance happens at the model level (one advance per
        // token, not per layer). We do NOT call cache.advance() here.
        return oHalf
    }

    // MARK: - CPU helpers

    private func float32ToHalf(_ src: MTLBuffer, count: Int, device: MTLDevice) -> MTLBuffer {
        let buf = device.makeBuffer(length: count * MemoryLayout<Float16>.stride,
                                     options: .storageModeShared)!
        let s = src.contents().bindMemory(to: Float.self, capacity: count)
        let d = buf.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { d[i] = Float16(s[i]) }
        return buf
    }

    /// Per-head RMSNorm: each head_dim chunk normalized independently.
    private func applyQKNormCPU(buf: MTLBuffer, gamma: MTLBuffer, totalDim: Int, eps: Float) {
        // gamma length must equal totalDim
        let bufPtr = buf.contents().bindMemory(to: Float16.self, capacity: totalDim)
        let gPtr = gamma.contents().bindMemory(to: Float16.self, capacity: totalDim)
        let nHeadsHere = totalDim / headDim
        for h in 0..<nHeadsHere {
            var sumSq: Float = 0
            for i in 0..<headDim {
                let v = Float(bufPtr[h * headDim + i])
                sumSq += v * v
            }
            let rms = 1.0 / Float((sumSq / Float(headDim) + eps).squareRoot())
            for i in 0..<headDim {
                let scaled = Float(bufPtr[h * headDim + i]) * rms * Float(gPtr[h * headDim + i])
                bufPtr[h * headDim + i] = Float16(scaled)
            }
        }
    }

    /// Standard RoPE: rotate pairs of dimensions in each head by `theta_i * pos`.
    /// Frequency convention: theta_i = base^(-2i/head_dim). Matches mlx_lm's
    /// `nn.RoPE(traditional=False)` which uses interleaved-half layout:
    /// the first half of head_dim is the "real" component, the second half the
    /// "imaginary". For position `p` and dim `i` in [0, head_dim/2):
    ///     freq = base^(-2*i/head_dim)
    ///     angle = p * freq
    ///     (real, imag) → (real*cos - imag*sin, real*sin + imag*cos)
    private func applyRoPECPU(
        buf: MTLBuffer, nHeads: Int, headDim: Int, position: Int, base: Float
    ) {
        let total = nHeads * headDim
        let p = buf.contents().bindMemory(to: Float16.self, capacity: total)
        let half = headDim / 2
        for h in 0..<nHeads {
            let rowOff = h * headDim
            for i in 0..<half {
                let freq = Foundation.pow(base, -2.0 * Float(i) / Float(headDim))
                let angle = Float(position) * freq
                let c = Foundation.cos(angle)
                let s = Foundation.sin(angle)
                let realIdx = rowOff + i
                let imagIdx = rowOff + i + half
                let r = Float(p[realIdx])
                let im = Float(p[imagIdx])
                p[realIdx] = Float16(r * c - im * s)
                p[imagIdx] = Float16(r * s + im * c)
            }
        }
    }

    /// CPU SDPA for decode (T=1, length cache_len). GQA-aware.
    /// Returns flat half buffer of shape (nHeads * headDim,).
    private func sdpaCPU(
        q: MTLBuffer, cache: JANGTQKVCache, curLen: Int, device: MTLDevice
    ) throws -> MTLBuffer {
        let qPtr = q.contents().bindMemory(to: Float16.self, capacity: nHeads * headDim)
        let kPtr = cache.keys[layerIndex].contents().bindMemory(
            to: Float16.self, capacity: cache.maxSeqLen * nKVHeads * headDim
        )
        let vPtr = cache.values[layerIndex].contents().bindMemory(
            to: Float16.self, capacity: cache.maxSeqLen * nKVHeads * headDim
        )

        let outBuf = device.makeBuffer(length: nHeads * headDim * MemoryLayout<Float16>.stride,
                                        options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: nHeads * headDim)
        let groupSize = nHeads / nKVHeads

        for h in 0..<nHeads {
            let kvHead = h / groupSize
            // logits[t] = q[h] · k[t, kv_head] / sqrt(d)
            var logits = [Float](repeating: 0, count: curLen)
            for t in 0..<curLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += Float(qPtr[h * headDim + d])
                         * Float(kPtr[t * nKVHeads * headDim + kvHead * headDim + d])
                }
                logits[t] = dot * scale
            }
            // softmax(logits)
            var maxLog: Float = -Float.infinity
            for v in logits { if v > maxLog { maxLog = v } }
            var sumExp: Float = 0
            for t in 0..<curLen {
                logits[t] = Foundation.exp(logits[t] - maxLog)
                sumExp += logits[t]
            }
            for t in 0..<curLen { logits[t] /= sumExp }
            // out[h] = sum_t softmax[t] * v[t, kv_head]
            for d in 0..<headDim {
                var acc: Float = 0
                for t in 0..<curLen {
                    acc += logits[t]
                         * Float(vPtr[t * nKVHeads * headDim + kvHead * headDim + d])
                }
                outPtr[h * headDim + d] = Float16(acc)
            }
        }
        return outBuf
    }
}
