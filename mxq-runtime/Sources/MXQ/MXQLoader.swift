/*
 * MLXQ Model Loader
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Loads an MXQ model directory into GPU memory for inference.
 * Uses memory-mapped safetensors for zero-copy loading on Apple Silicon.
 *
 * Loading pipeline:
 *   1. Parse mxq_config.json + config.json
 *   2. Memory-map all .mxq.safetensors shard files
 *   3. Create Metal buffers for each tensor
 *   4. Build layer structure matching model architecture
 */

import Foundation
import Metal

/// A loaded MXQ quantized weight tensor with all companion data.
public struct MXQWeight: @unchecked Sendable {
    public let name: String
    public let qweight: MTLBuffer       // packed quantized data
    public let scales: MTLBuffer        // per-block scale factors (float16)
    public let zeros: MTLBuffer         // per-block zero points (float16)
    public let bitMap: MTLBuffer        // per-block bit widths (uint8)
    public let blockOffsets: MTLBuffer  // byte offsets per block (uint32)
    public let numBlocks: Int
    public let outFeatures: Int         // rows of weight matrix
    public let inFeatures: Int          // cols of weight matrix
}

/// A single transformer layer's weights.
public struct TransformerLayerWeights: @unchecked Sendable {
    public let index: Int

    // Attention
    public let qProj: MXQWeight
    public let kProj: MXQWeight
    public let vProj: MXQWeight
    public let oProj: MXQWeight

    // Attention biases (Qwen uses biases on Q/K/V)
    public let qBias: MTLBuffer?
    public let kBias: MTLBuffer?
    public let vBias: MTLBuffer?

    // MLP
    public let gateProj: MXQWeight
    public let upProj: MXQWeight
    public let downProj: MXQWeight

    // Norms (kept in float16, not quantized)
    public let inputNorm: MTLBuffer     // RMSNorm weight
    public let postAttnNorm: MTLBuffer  // RMSNorm weight
}

/// Complete loaded model ready for inference.
public struct MXQModel: @unchecked Sendable {
    public let config: MXQModelConfig
    public let embedTokens: MXQWeight        // embedding weight (quantized)
    public let finalNorm: MTLBuffer          // final RMSNorm weight
    public let lmHead: MXQWeight?            // output head (nil if tied with embeddings)
    public let layers: [TransformerLayerWeights]
    public let device: MTLDevice

    /// Total GPU memory used by this model.
    public var memoryBytes: Int {
        var total = embedTokens.qweight.length + finalNorm.length
        if let lm = lmHead {
            total += lm.qweight.length + lm.scales.length + lm.zeros.length
                   + lm.bitMap.length + lm.blockOffsets.length
        }
        for layer in layers {
            for w in [layer.qProj, layer.kProj, layer.vProj, layer.oProj,
                      layer.gateProj, layer.upProj, layer.downProj] {
                total += w.qweight.length + w.scales.length + w.zeros.length
                       + w.bitMap.length + w.blockOffsets.length
            }
            total += layer.inputNorm.length + layer.postAttnNorm.length
        }
        return total
    }

    public var memoryMB: Double {
        return Double(memoryBytes) / (1024.0 * 1024.0)
    }
}

/// Load an MXQ model from a directory.
public func loadModel(path: String, device: MTLDevice) throws -> MXQModel {
    let url = URL(fileURLWithPath: path)
    return try loadModel(url: url, device: device)
}

public func loadModel(url: URL, device: MTLDevice) throws -> MXQModel {
    let startTime = CFAbsoluteTimeGetCurrent()

    // 1. Load configs
    let config = try MXQModelConfig.load(from: url)
    let modelConfig = config.model

    print("  Loading: \(config.quant.sourceModelName)")
    print("  Architecture: \(modelConfig.modelType ?? "unknown")")
    print("  Layers: \(modelConfig.numHiddenLayers)")
    print("  Hidden: \(modelConfig.hiddenSize)")
    print("  Heads: \(modelConfig.numAttentionHeads) Q, \(modelConfig.kvHeads) KV")
    print("  Bits: \(config.quant.actualBits)")

    // 2. Load safetensors shards
    let shards = try loadSafetensorsShards(from: url)
    let allTensors = buildTensorIndex(shards: shards)

    // 3. Load embedding (quantized in MXQ format)
    let embedTokens = try loadMXQWeight(named: "model.embed_tokens",
                                         from: allTensors, device: device)

    // 4. Load final norm
    let finalNorm = try loadFloat16Tensor(named: "model.norm.weight",
                                           from: allTensors, device: device)

    // 5. Load lm_head (if not tied)
    var lmHead: MXQWeight? = nil
    if !modelConfig.tiedEmbeddings {
        if hasMXQWeight(named: "lm_head", in: allTensors) {
            lmHead = try loadMXQWeight(named: "lm_head", from: allTensors, device: device)
        }
    }

    // 6. Load transformer layers
    var layers: [TransformerLayerWeights] = []
    for i in 0..<modelConfig.numHiddenLayers {
        let prefix = "model.layers.\(i)"

        // Load biases if they exist (Qwen uses Q/K/V biases)
        let qBias = try? loadFloat16Tensor(named: "\(prefix).self_attn.q_proj.bias",
                                            from: allTensors, device: device)
        let kBias = try? loadFloat16Tensor(named: "\(prefix).self_attn.k_proj.bias",
                                            from: allTensors, device: device)
        let vBias = try? loadFloat16Tensor(named: "\(prefix).self_attn.v_proj.bias",
                                            from: allTensors, device: device)

        let layer = try TransformerLayerWeights(
            index: i,
            qProj: loadMXQWeight(named: "\(prefix).self_attn.q_proj",
                                  from: allTensors, device: device),
            kProj: loadMXQWeight(named: "\(prefix).self_attn.k_proj",
                                  from: allTensors, device: device),
            vProj: loadMXQWeight(named: "\(prefix).self_attn.v_proj",
                                  from: allTensors, device: device),
            oProj: loadMXQWeight(named: "\(prefix).self_attn.o_proj",
                                  from: allTensors, device: device),
            qBias: qBias,
            kBias: kBias,
            vBias: vBias,
            gateProj: loadMXQWeight(named: "\(prefix).mlp.gate_proj",
                                     from: allTensors, device: device),
            upProj: loadMXQWeight(named: "\(prefix).mlp.up_proj",
                                   from: allTensors, device: device),
            downProj: loadMXQWeight(named: "\(prefix).mlp.down_proj",
                                     from: allTensors, device: device),
            inputNorm: loadFloat16Tensor(named: "\(prefix).input_layernorm.weight",
                                          from: allTensors, device: device),
            postAttnNorm: loadFloat16Tensor(named: "\(prefix).post_attention_layernorm.weight",
                                             from: allTensors, device: device)
        )
        layers.append(layer)
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime

    let model = MXQModel(
        config: config,
        embedTokens: embedTokens,
        finalNorm: finalNorm,
        lmHead: lmHead,
        layers: layers,
        device: device
    )

    print("  Loaded in \(String(format: "%.2f", elapsed))s")
    print("  GPU memory: \(String(format: "%.1f", model.memoryMB)) MB")

    return model
}

// MARK: - Private helpers

/// Index of tensor name → (shard, tensorInfo).
private typealias TensorIndex = [String: (SafetensorsFile, TensorInfo)]

private func buildTensorIndex(shards: [SafetensorsFile]) -> TensorIndex {
    var index: TensorIndex = [:]
    for shard in shards {
        for (name, info) in shard.tensors {
            index[name] = (shard, info)
        }
    }
    return index
}

private func hasMXQWeight(named baseName: String, in index: TensorIndex) -> Bool {
    return index["\(baseName).qweight"] != nil
}

private func loadMXQWeight(named baseName: String,
                            from index: TensorIndex,
                            device: MTLDevice) throws -> MXQWeight {
    let qweightName = "\(baseName).qweight"
    let scalesName = "\(baseName).scales"
    let zerosName = "\(baseName).zeros"
    let bitMapName = "\(baseName).bit_map"
    let offsetsName = "\(baseName).block_offsets"

    guard let (qShard, _) = index[qweightName] else {
        throw MXQError.tensorNotFound(qweightName)
    }

    let qweight = try qShard.makeMetalBuffer(name: qweightName, device: device)
    let scales = try loadBuffer(named: scalesName, from: index, device: device)
    let zeros = try loadBuffer(named: zerosName, from: index, device: device)
    let bitMap = try loadBuffer(named: bitMapName, from: index, device: device)
    let blockOffsets = try loadBuffer(named: offsetsName, from: index, device: device)

    // Determine dimensions from bit_map tensor shape
    guard let (_, bitMapInfo) = index[bitMapName] else {
        throw MXQError.tensorNotFound(bitMapName)
    }
    let numBlocks = bitMapInfo.shape[0]

    // We need to figure out outFeatures/inFeatures from the config
    // For now, store numBlocks and derive dimensions later
    return MXQWeight(
        name: baseName,
        qweight: qweight,
        scales: scales,
        zeros: zeros,
        bitMap: bitMap,
        blockOffsets: blockOffsets,
        numBlocks: numBlocks,
        outFeatures: 0,  // TODO: derive from model config
        inFeatures: 0     // TODO: derive from model config
    )
}

private func loadBuffer(named name: String,
                          from index: TensorIndex,
                          device: MTLDevice) throws -> MTLBuffer {
    guard let (shard, _) = index[name] else {
        throw MXQError.tensorNotFound(name)
    }
    return try shard.makeMetalBuffer(name: name, device: device)
}

private func loadFloat16Tensor(named name: String,
                                from index: TensorIndex,
                                device: MTLDevice) throws -> MTLBuffer {
    guard let (shard, _) = index[name] else {
        throw MXQError.tensorNotFound(name)
    }

    // Float16 tensors are stored directly — load as-is
    // BFloat16 tensors need conversion (we handle this in the safetensors reader)
    return try shard.makeMetalBuffer(name: name, device: device)
}
