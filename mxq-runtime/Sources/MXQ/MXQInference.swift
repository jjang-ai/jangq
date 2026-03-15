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

    /// Run a single-token forward pass (for autoregressive generation).
    /// Returns logits buffer (vocab_size float16 values).
    public func forward(tokenId: Int) throws -> MTLBuffer {
        guard let cmdBuffer = metalDevice.commandQueue.makeCommandBuffer() else {
            throw MXQError.inferenceError("Failed to create command buffer")
        }

        // Embedding lookup
        try dispatchEmbedding(cmdBuffer: cmdBuffer, tokenId: tokenId)

        // Copy hidden state to residual buffer for first residual connection
        try dispatchCopy(cmdBuffer: cmdBuffer, from: hiddenBuffer, to: residualBuffer,
                         bytes: config.hiddenSize * MemoryLayout<Float16>.stride)

        // Process each transformer layer
        for layerIdx in 0..<config.numHiddenLayers {
            let layer = model.layers[layerIdx]

            // 1. Input LayerNorm
            try dispatchRMSNorm(cmdBuffer: cmdBuffer,
                                input: hiddenBuffer,
                                gamma: layer.inputNorm,
                                output: normBuffer,
                                hiddenSize: config.hiddenSize)

            // 2. Q/K/V projections (dequant + GEMV)
            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.qProj,
                                     output: qBuffer,
                                     K: config.hiddenSize,
                                     N: config.numAttentionHeads * config.headDim)

            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.kProj,
                                     output: kBuffer,
                                     K: config.hiddenSize,
                                     N: config.kvHeads * config.headDim)

            try dispatchDequantGEMV(cmdBuffer: cmdBuffer,
                                     input: normBuffer,
                                     weight: layer.vProj,
                                     output: vBuffer,
                                     K: config.hiddenSize,
                                     N: config.kvHeads * config.headDim)

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
