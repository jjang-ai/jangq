/*
 * MXQ Inference Engine — Forward Pass Execution
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Executes the transformer forward pass by dispatching Metal kernels.
 * This is where the MXQ dequant kernels do their work — every linear
 * layer uses fused dequant+matmul on quantized weights.
 *
 * Forward pass per layer:
 *   1. RMSNorm (input_layernorm)
 *   2. Q/K/V projections (3 dequant+GEMV/GEMM)
 *   3. RoPE on Q and K
 *   4. Attention: Q×K^T, softmax, ×V
 *   5. O projection (1 dequant+GEMV/GEMM)
 *   6. Residual add
 *   7. RMSNorm (post_attention_layernorm)
 *   8. Gate + Up projections (2 dequant+GEMV/GEMM)
 *   9. SiLU(gate) * up
 *  10. Down projection (1 dequant+GEMV/GEMM)
 *  11. Residual add
 */

import Metal
import Foundation

/// Manages inference state and executes forward passes.
public final class MXQInferenceEngine {
    private let metalDevice: MXQMetalDevice
    private let model: MXQModel
    private let config: ModelConfig

    // KV cache buffers — pre-allocated for max sequence length
    private var kvCacheKeys: [[MTLBuffer]]    // [layer][position]
    private var kvCacheValues: [[MTLBuffer]]  // [layer][position]
    private var currentPosition: Int = 0
    private let maxSeqLen: Int

    // Reusable intermediate buffers
    private var hiddenBuffer: MTLBuffer       // (seq_len, hidden_size)
    private var residualBuffer: MTLBuffer
    private var normBuffer: MTLBuffer
    private var qBuffer: MTLBuffer
    private var kBuffer: MTLBuffer
    private var vBuffer: MTLBuffer
    private var attnOutBuffer: MTLBuffer
    private var gateBuffer: MTLBuffer
    private var upBuffer: MTLBuffer
    private var mlpBuffer: MTLBuffer
    private var logitsBuffer: MTLBuffer

    public init(model: MXQModel, metalDevice: MXQMetalDevice, maxSeqLen: Int = 2048) throws {
        self.model = model
        self.metalDevice = metalDevice
        self.config = model.config.model
        self.maxSeqLen = maxSeqLen

        let hidden = config.hiddenSize
        let intermediate = config.intermediateSize
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        let nHeads = config.numAttentionHeads
        let vocab = config.vocabSize
        let device = metalDevice.device

        // Allocate intermediate buffers (sized for single-token decode)
        let halfSize = MemoryLayout<Float16>.stride

        self.hiddenBuffer = try Self.makeBuffer(device, hidden * halfSize)
        self.residualBuffer = try Self.makeBuffer(device, hidden * halfSize)
        self.normBuffer = try Self.makeBuffer(device, hidden * halfSize)
        self.qBuffer = try Self.makeBuffer(device, nHeads * headDim * halfSize)
        self.kBuffer = try Self.makeBuffer(device, kvHeads * headDim * halfSize)
        self.vBuffer = try Self.makeBuffer(device, kvHeads * headDim * halfSize)
        self.attnOutBuffer = try Self.makeBuffer(device, hidden * halfSize)
        self.gateBuffer = try Self.makeBuffer(device, intermediate * halfSize)
        self.upBuffer = try Self.makeBuffer(device, intermediate * halfSize)
        self.mlpBuffer = try Self.makeBuffer(device, hidden * halfSize)
        self.logitsBuffer = try Self.makeBuffer(device, vocab * halfSize)

        // Allocate KV cache
        self.kvCacheKeys = []
        self.kvCacheValues = []
        let kvBytesPerPos = kvHeads * headDim * halfSize

        for _ in 0..<config.numHiddenLayers {
            let keyBuf = try Self.makeBuffer(device, maxSeqLen * kvBytesPerPos)
            let valBuf = try Self.makeBuffer(device, maxSeqLen * kvBytesPerPos)
            kvCacheKeys.append([keyBuf])
            kvCacheValues.append([valBuf])
        }

        print("  Inference engine initialized")
        print("  KV cache: \(config.numHiddenLayers) layers × \(maxSeqLen) positions")
        let kvMB = Double(config.numHiddenLayers * 2 * maxSeqLen * kvBytesPerPos) / (1024 * 1024)
        print("  KV cache memory: \(String(format: "%.1f", kvMB)) MB")
    }

    /// Enable to print per-layer hidden state norms during forward pass.
    public var debugLayers: Bool = false

    /// Helper to get a fresh command buffer.
    private func makeCmd() throws -> MTLCommandBuffer {
        guard let cmd = metalDevice.commandQueue.makeCommandBuffer() else {
            throw MXQError.inferenceError("Failed to create command buffer")
        }
        return cmd
    }

    /// Run a single-token forward pass (for autoregressive generation).
    /// Returns logits buffer (vocab_size float16 values).
    public func forward(tokenId: Int) throws -> MTLBuffer {
        var cmdBuffer = try makeCmd()

        // Embedding lookup
        try dispatchEmbedding(cmdBuffer: cmdBuffer, tokenId: tokenId)

        // Copy hidden state to residual buffer for first residual connection
        try dispatchCopy(cmdBuffer: cmdBuffer, from: hiddenBuffer, to: residualBuffer,
                         bytes: config.hiddenSize * MemoryLayout<Float16>.stride)

        // Process each transformer layer
        let numLayers = config.numHiddenLayers  // TODO: was config.numHiddenLayers
        for layerIdx in 0..<numLayers {
            let layer = model.layers[layerIdx]

            // 1. Input LayerNorm
            try dispatchRMSNorm(cmdBuffer: cmdBuffer,
                                input: hiddenBuffer,
                                gamma: layer.inputNorm,
                                output: normBuffer,
                                hiddenSize: config.hiddenSize)

            // 2. Q/K/V projections (dequant + GEMV + bias)
            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.qProj,
                                     output: qBuffer,
                                     K: config.hiddenSize,
                                     N: config.numAttentionHeads * config.headDim)
            if let qBias = layer.qBias {
                try dispatchAddInPlace(cmdBuffer: cmdBuffer,
                                        buffer: qBuffer, bias: qBias,
                                        count: config.numAttentionHeads * config.headDim)
            }

            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.kProj,
                                     output: kBuffer,
                                     K: config.hiddenSize,
                                     N: config.kvHeads * config.headDim)
            if let kBias = layer.kBias {
                try dispatchAddInPlace(cmdBuffer: cmdBuffer,
                                        buffer: kBuffer, bias: kBias,
                                        count: config.kvHeads * config.headDim)
            }

            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.vProj,
                                     output: vBuffer,
                                     K: config.hiddenSize,
                                     N: config.kvHeads * config.headDim)
            if let vBias = layer.vBias {
                try dispatchAddInPlace(cmdBuffer: cmdBuffer,
                                        buffer: vBuffer, bias: vBias,
                                        count: config.kvHeads * config.headDim)
            }

            // 3. RoPE on Q and K
            try dispatchRoPE(cmdBuffer: cmdBuffer,
                             qk: qBuffer,
                             seqLen: 1,
                             nHeads: config.numAttentionHeads,
                             headDim: config.headDim,
                             posOffset: currentPosition)

            try dispatchRoPE(cmdBuffer: cmdBuffer,
                             qk: kBuffer,
                             seqLen: 1,
                             nHeads: config.kvHeads,
                             headDim: config.headDim,
                             posOffset: currentPosition)

            // 4. Store K, V in cache
            try storeKVCache(cmdBuffer: cmdBuffer, layer: layerIdx)

            // 5. Attention: Q × K^T, softmax, × V
            // TODO: implement attention kernel
            // For now, use a simplified single-head attention
            try dispatchAttention(cmdBuffer: cmdBuffer, layer: layerIdx)

            // 6. O projection
            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: attnOutBuffer,
                                     weight: layer.oProj,
                                     output: normBuffer,  // reuse normBuffer as temp
                                     K: config.hiddenSize,
                                     N: config.hiddenSize)

            // 7. Residual add: hidden = residual + attn_output
            try dispatchAdd(cmdBuffer: cmdBuffer,
                            a: residualBuffer, b: normBuffer,
                            output: hiddenBuffer,
                            count: config.hiddenSize)

            // Save for next residual
            try dispatchCopy(cmdBuffer: cmdBuffer, from: hiddenBuffer, to: residualBuffer,
                             bytes: config.hiddenSize * MemoryLayout<Float16>.stride)

            // 8. Post-attention LayerNorm
            try dispatchRMSNorm(cmdBuffer: cmdBuffer,
                                input: hiddenBuffer,
                                gamma: layer.postAttnNorm,
                                output: normBuffer,
                                hiddenSize: config.hiddenSize)

            // 9. Gate + Up projections
            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.gateProj,
                                     output: gateBuffer,
                                     K: config.hiddenSize,
                                     N: config.intermediateSize)

            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.upProj,
                                     output: upBuffer,
                                     K: config.hiddenSize,
                                     N: config.intermediateSize)

            // 10. SiLU(gate) * up
            try dispatchSiLUMul(cmdBuffer: cmdBuffer,
                                gate: gateBuffer, up: upBuffer,
                                output: mlpBuffer,
                                count: config.intermediateSize)

            // 11. Down projection
            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: mlpBuffer,
                                     weight: layer.downProj,
                                     output: normBuffer,
                                     K: config.intermediateSize,
                                     N: config.hiddenSize)

            // 12. Residual add
            try dispatchAdd(cmdBuffer: cmdBuffer,
                            a: residualBuffer, b: normBuffer,
                            output: hiddenBuffer,
                            count: config.hiddenSize)

            // Save for next layer's residual
            try dispatchCopy(cmdBuffer: cmdBuffer, from: hiddenBuffer, to: residualBuffer,
                             bytes: config.hiddenSize * MemoryLayout<Float16>.stride)

            // Debug: dump per-layer hidden state norm
            if debugLayers {
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()
                let ptr = hiddenBuffer.contents().bindMemory(to: Float16.self,
                                                              capacity: config.hiddenSize)
                var sumSq: Float = 0
                for i in 0..<config.hiddenSize { sumSq += Float(ptr[i]) * Float(ptr[i]) }
                let norm = sqrt(sumSq)
                let first4 = (0..<4).map { String(format: "%.4f", Float(ptr[$0])) }.joined(separator: ", ")
                print("  L\(String(format: "%02d", layerIdx)): norm=\(String(format: "%8.2f", norm))  [\(first4)]")

                cmdBuffer = try makeCmd()
            }
        }

        // Final RMSNorm
        try dispatchRMSNorm(cmdBuffer: cmdBuffer,
                            input: hiddenBuffer,
                            gamma: model.finalNorm,
                            output: normBuffer,
                            hiddenSize: config.hiddenSize)

        // LM Head projection → logits
        // For tied embeddings, use the embedding table as lm_head
        let lmHeadWeight = model.lmHead ?? model.embedTokens
        try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                 input: normBuffer,
                                 weight: lmHeadWeight,
                                 output: logitsBuffer,
                                 K: config.hiddenSize,
                                 N: config.vocabSize)

        // Submit and wait
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        currentPosition += 1
        return logitsBuffer
    }

    /// Reset the KV cache (start a new sequence).
    public func reset() {
        currentPosition = 0
    }

    /// Dump the first N float16 values from a buffer (for debugging).
    public func dumpBuffer(_ buffer: MTLBuffer, name: String, count: Int = 8) {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        var values: [Float] = []
        for i in 0..<min(count, buffer.length / 2) {
            values.append(Float(ptr[i]))
        }
        let formatted = values.map { String(format: "%.4f", $0) }.joined(separator: ", ")
        print("  DEBUG \(name)[\(count)]: [\(formatted)]")
    }

    /// Run embedding only and dump results (for verifying GPU vs CPU).
    public func debugEmbedding(tokenId: Int) throws {
        guard let cmdBuffer = metalDevice.commandQueue.makeCommandBuffer() else {
            throw MXQError.inferenceError("Failed to create command buffer")
        }

        try dispatchEmbedding(cmdBuffer: cmdBuffer, tokenId: tokenId)
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        dumpBuffer(hiddenBuffer, name: "embed[\(tokenId)]")
    }

    /// Run one full forward layer step by step, dumping intermediate values.
    public func debugForwardOneLayer(tokenId: Int) throws {
        let layer = model.layers[0]

        // Helper to run one kernel and dump
        func step(_ name: String, _ block: (MTLCommandBuffer) throws -> Void,
                  buffer: MTLBuffer, count: Int = 8) throws {
            guard let cmd = metalDevice.commandQueue.makeCommandBuffer() else {
                throw MXQError.inferenceError("Failed to create command buffer")
            }
            try block(cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            dumpBuffer(buffer, name: name, count: count)
        }

        // 1. Embedding
        try step("embed", { cmd in
            try dispatchEmbedding(cmdBuffer: cmd, tokenId: tokenId)
        }, buffer: hiddenBuffer)

        // 2. Copy to residual
        try step("residual", { cmd in
            try dispatchCopy(cmdBuffer: cmd, from: hiddenBuffer, to: residualBuffer,
                             bytes: config.hiddenSize * MemoryLayout<Float16>.stride)
        }, buffer: residualBuffer)

        // 3. RMSNorm
        try step("norm0", { cmd in
            try dispatchRMSNorm(cmdBuffer: cmd, input: hiddenBuffer,
                                gamma: layer.inputNorm, output: normBuffer,
                                hiddenSize: config.hiddenSize)
        }, buffer: normBuffer)

        // 4. Q projection
        try step("q_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: normBuffer,
                                     weight: layer.qProj, output: qBuffer,
                                     K: config.hiddenSize,
                                     N: config.numAttentionHeads * config.headDim)
        }, buffer: qBuffer)

        // Dump q_proj weight metadata for verification
        let qw = layer.qProj
        print("  DEBUG q_proj weight: qweight=\(qw.qweight.length)B, " +
              "scales=\(qw.scales.length)B, blocks=\(qw.numBlocks)")

        // Read first block metadata from GPU
        let bitsPtr = qw.bitMap.contents().bindMemory(to: UInt8.self, capacity: 4)
        let scalesPtr = qw.scales.contents().bindMemory(to: Float16.self, capacity: 4)
        let zerosPtr = qw.zeros.contents().bindMemory(to: Float16.self, capacity: 4)
        print("  DEBUG q_proj block0: bits=\(bitsPtr[0]), scale=\(Float(scalesPtr[0])), zero=\(Float(zerosPtr[0]))")

        // 5. K projection
        try step("k_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: normBuffer,
                                     weight: layer.kProj, output: kBuffer,
                                     K: config.hiddenSize,
                                     N: config.kvHeads * config.headDim)
        }, buffer: kBuffer)

        // 6. V projection
        try step("v_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: normBuffer,
                                     weight: layer.vProj, output: vBuffer,
                                     K: config.hiddenSize,
                                     N: config.kvHeads * config.headDim)
        }, buffer: vBuffer)

        // 7. RoPE on Q
        try step("q_rope", { cmd in
            try dispatchRoPE(cmdBuffer: cmd, qk: qBuffer, seqLen: 1,
                             nHeads: config.numAttentionHeads,
                             headDim: config.headDim, posOffset: currentPosition)
        }, buffer: qBuffer)

        // 8. RoPE on K
        try step("k_rope", { cmd in
            try dispatchRoPE(cmdBuffer: cmd, qk: kBuffer, seqLen: 1,
                             nHeads: config.kvHeads,
                             headDim: config.headDim, posOffset: currentPosition)
        }, buffer: kBuffer)

        // 9. Store KV cache
        try step("kv_store", { cmd in
            try storeKVCache(cmdBuffer: cmd, layer: 0)
        }, buffer: kvCacheKeys[0][0], count: 4)

        // 10. Attention
        try step("attn_out", { cmd in
            try dispatchAttention(cmdBuffer: cmd, layer: 0)
        }, buffer: attnOutBuffer)

        // 11. O projection
        try step("o_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: attnOutBuffer,
                                     weight: layer.oProj, output: normBuffer,
                                     K: config.hiddenSize,
                                     N: config.hiddenSize)
        }, buffer: normBuffer)

        // 12. Residual add
        try step("residual_add1", { cmd in
            try dispatchAdd(cmdBuffer: cmd, a: residualBuffer, b: normBuffer,
                            output: hiddenBuffer, count: config.hiddenSize)
        }, buffer: hiddenBuffer)

        // 13. Post-attention norm
        try step("post_norm", { cmd in
            try dispatchCopy(cmdBuffer: cmd, from: hiddenBuffer, to: residualBuffer,
                             bytes: config.hiddenSize * MemoryLayout<Float16>.stride)
        }, buffer: residualBuffer)

        try step("mlp_norm", { cmd in
            try dispatchRMSNorm(cmdBuffer: cmd, input: hiddenBuffer,
                                gamma: layer.postAttnNorm, output: normBuffer,
                                hiddenSize: config.hiddenSize)
        }, buffer: normBuffer)

        // 14. Gate + Up
        try step("gate_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: normBuffer,
                                     weight: layer.gateProj, output: gateBuffer,
                                     K: config.hiddenSize,
                                     N: config.intermediateSize)
        }, buffer: gateBuffer)

        try step("up_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: normBuffer,
                                     weight: layer.upProj, output: upBuffer,
                                     K: config.hiddenSize,
                                     N: config.intermediateSize)
        }, buffer: upBuffer)

        // 15. SiLU * mul
        try step("silu_mul", { cmd in
            try dispatchSiLUMul(cmdBuffer: cmd, gate: gateBuffer, up: upBuffer,
                                output: mlpBuffer, count: config.intermediateSize)
        }, buffer: mlpBuffer)

        // 16. Down projection
        try step("down_proj", { cmd in
            try dispatchDequantGEMV(cmdBuffer: cmd, input: mlpBuffer,
                                     weight: layer.downProj, output: normBuffer,
                                     K: config.intermediateSize,
                                     N: config.hiddenSize)
        }, buffer: normBuffer)

        // 17. Residual add 2
        try step("layer_out", { cmd in
            try dispatchAdd(cmdBuffer: cmd, a: residualBuffer, b: normBuffer,
                            output: hiddenBuffer, count: config.hiddenSize)
        }, buffer: hiddenBuffer)

        currentPosition += 1
        print("  Layer 0 complete. All intermediate values above.")
    }

    // MARK: - Kernel Dispatch Helpers

    private func dispatchEmbedding(cmdBuffer: MTLCommandBuffer, tokenId: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let embed = model.embedTokens
        let pipeline = try metalDevice.pipeline(for: "mxq_embedding_dequant")
        encoder.setComputePipelineState(pipeline)

        encoder.setBuffer(embed.qweight, offset: 0, index: 0)
        encoder.setBuffer(embed.scales, offset: 0, index: 1)
        encoder.setBuffer(embed.zeros, offset: 0, index: 2)
        encoder.setBuffer(embed.bitMap, offset: 0, index: 3)
        encoder.setBuffer(embed.blockOffsets, offset: 0, index: 4)
        encoder.setBuffer(hiddenBuffer, offset: 0, index: 5)

        var tid = UInt32(tokenId)
        var hidden = UInt32(config.hiddenSize)
        var blocksPerRow = UInt32((config.hiddenSize + 63) / 64)  // ceil(hidden / block_size)
        encoder.setBytes(&tid, length: 4, index: 6)
        encoder.setBytes(&hidden, length: 4, index: 7)
        encoder.setBytes(&blocksPerRow, length: 4, index: 8)

        let gridSize = MTLSize(width: config.hiddenSize, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(config.hiddenSize, 256), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func dispatchRMSNorm(cmdBuffer: MTLCommandBuffer,
                                  input: MTLBuffer, gamma: MTLBuffer,
                                  output: MTLBuffer, hiddenSize: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let pipeline = try metalDevice.pipeline(for: "mxq_rms_norm")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gamma, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var hidden = UInt32(hiddenSize)
        encoder.setBytes(&hidden, length: 4, index: 3)

        var eps = config.normEps
        encoder.setBytes(&eps, length: 4, index: 4)

        // Dispatch: 1 row, hiddenSize threads
        let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(hiddenSize, 256), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func dispatchDequantGEMV(cmdBuffer: MTLCommandBuffer,
                                      input: MTLBuffer, weight: MXQWeight,
                                      output: MTLBuffer,
                                      K: Int, N: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let pipeline = try metalDevice.pipeline(for: "mxq_dequant_gemv")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight.qweight, offset: 0, index: 0)
        encoder.setBuffer(weight.scales, offset: 0, index: 1)
        encoder.setBuffer(weight.zeros, offset: 0, index: 2)
        encoder.setBuffer(weight.bitMap, offset: 0, index: 3)
        encoder.setBuffer(weight.blockOffsets, offset: 0, index: 4)
        encoder.setBuffer(input, offset: 0, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)

        var kVal = UInt32(K)
        var nVal = UInt32(N)
        encoder.setBytes(&kVal, length: 4, index: 7)
        encoder.setBytes(&nVal, length: 4, index: 8)

        // One threadgroup per output row, 256 threads per threadgroup
        let gridSize = MTLSize(width: N, height: 1, depth: 1)
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func dispatchRoPE(cmdBuffer: MTLCommandBuffer,
                               qk: MTLBuffer, seqLen: Int, nHeads: Int,
                               headDim: Int, posOffset: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let pipeline = try metalDevice.pipeline(for: "mxq_rope")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qk, offset: 0, index: 0)

        var sl = UInt32(seqLen)
        var nh = UInt32(nHeads)
        var hd = UInt32(headDim)
        var po = UInt32(posOffset)
        var theta = config.ropeBase
        encoder.setBytes(&sl, length: 4, index: 1)
        encoder.setBytes(&nh, length: 4, index: 2)
        encoder.setBytes(&hd, length: 4, index: 3)
        encoder.setBytes(&po, length: 4, index: 4)
        encoder.setBytes(&theta, length: 4, index: 5)

        let halfDim = headDim / 2
        let gridSize = MTLSize(width: halfDim, height: nHeads, depth: seqLen)
        let tgSize = MTLSize(width: min(halfDim, 64), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func storeKVCache(cmdBuffer: MTLCommandBuffer, layer: Int) throws {
        // Copy K and V into the cache at the current position
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        let bytesPerPos = kvHeads * headDim * MemoryLayout<Float16>.stride
        let offset = currentPosition * bytesPerPos

        guard let blitEncoder = cmdBuffer.makeBlitCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create blit encoder")
        }

        blitEncoder.copy(from: kBuffer, sourceOffset: 0,
                         to: kvCacheKeys[layer][0], destinationOffset: offset,
                         size: bytesPerPos)
        blitEncoder.copy(from: vBuffer, sourceOffset: 0,
                         to: kvCacheValues[layer][0], destinationOffset: offset,
                         size: bytesPerPos)
        blitEncoder.endEncoding()
    }

    private func dispatchAttention(cmdBuffer: MTLCommandBuffer, layer: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let pipeline = try metalDevice.pipeline(for: "mxq_attention_decode")
        encoder.setComputePipelineState(pipeline)

        let nHeads = config.numAttentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim

        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(kvCacheKeys[layer][0], offset: 0, index: 1)
        encoder.setBuffer(kvCacheValues[layer][0], offset: 0, index: 2)
        encoder.setBuffer(attnOutBuffer, offset: 0, index: 3)

        var nh = UInt32(nHeads)
        var nkv = UInt32(kvHeads)
        var hd = UInt32(headDim)
        var sl = UInt32(max(currentPosition + 1, 1))  // seq_len = filled positions
        var scale = 1.0 / sqrt(Float(headDim))        // 1/sqrt(d_k)

        encoder.setBytes(&nh, length: 4, index: 4)
        encoder.setBytes(&nkv, length: 4, index: 5)
        encoder.setBytes(&hd, length: 4, index: 6)
        encoder.setBytes(&sl, length: 4, index: 7)
        encoder.setBytes(&scale, length: 4, index: 8)

        // One threadgroup per head, 256 threads per threadgroup
        let gridSize = MTLSize(width: nHeads, height: 1, depth: 1)
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    /// Add bias to buffer in-place: buffer[i] += bias[i]
    private func dispatchAddInPlace(cmdBuffer: MTLCommandBuffer,
                                     buffer: MTLBuffer, bias: MTLBuffer,
                                     count: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        // Use mxq_add: output = a + b, with output == buffer (in-place via temp)
        // Actually we need in-place add. Let's use a simple approach:
        // Read buffer into temp, add bias, write back.
        // But we can just use mxq_add with buffer as both input a and output,
        // since Metal processes all threads before writing.
        // Wait — that's not safe. Let's dispatch mxq_add with buffer as 'a',
        // bias as 'b', and a temp buffer as output, then copy back.
        // Actually, the simplest correct approach: Metal guarantees that
        // dispatchThreads processes independently per thread, and reading
        // buffer[i] before writing buffer[i] in the same dispatch is safe
        // as long as each thread reads/writes only its own index.
        // So we CAN use buffer as both a and output.

        let pipeline = try metalDevice.pipeline(for: "mxq_add")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        encoder.setBuffer(buffer, offset: 0, index: 2)  // output = buffer (in-place)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(count, 256), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func dispatchSiLUMul(cmdBuffer: MTLCommandBuffer,
                                  gate: MTLBuffer, up: MTLBuffer,
                                  output: MTLBuffer, count: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let pipeline = try metalDevice.pipeline(for: "mxq_silu_mul")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(count, 256), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func dispatchAdd(cmdBuffer: MTLCommandBuffer,
                              a: MTLBuffer, b: MTLBuffer,
                              output: MTLBuffer, count: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create compute encoder")
        }

        let pipeline = try metalDevice.pipeline(for: "mxq_add")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(count, 256), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }

    private func dispatchCopy(cmdBuffer: MTLCommandBuffer,
                               from: MTLBuffer, to: MTLBuffer,
                               bytes: Int) throws {
        guard let blitEncoder = cmdBuffer.makeBlitCommandEncoder() else {
            throw MXQError.inferenceError("Failed to create blit encoder")
        }
        blitEncoder.copy(from: from, sourceOffset: 0,
                         to: to, destinationOffset: 0,
                         size: bytes)
        blitEncoder.endEncoding()
    }

    private static func makeBuffer(_ device: MTLDevice, _ size: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw MXQError.bufferAllocationFailed(size)
        }
        return buffer
    }
}
