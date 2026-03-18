/*
 * JANG Model Configuration
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Parses config.json and jang_config.json to determine model
 * architecture, dimensions, and quantization parameters.
 */

import Foundation

/// HuggingFace model configuration (from config.json).
public struct ModelConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int?
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Double?
    public let rmsNormEps: Double?
    public let modelType: String?
    public let architectures: [String]?
    public let tieWordEmbeddings: Bool?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case modelType = "model_type"
        case architectures
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Number of KV heads (defaults to num_attention_heads if not specified = MHA).
    public var kvHeads: Int {
        return numKeyValueHeads ?? numAttentionHeads
    }

    /// Head dimension.
    public var headDim: Int {
        return hiddenSize / numAttentionHeads
    }

    /// RoPE base frequency.
    public var ropeBase: Float {
        return Float(ropeTheta ?? 10000.0)
    }

    /// RMSNorm epsilon.
    public var normEps: Float {
        return Float(rmsNormEps ?? 1e-5)
    }

    /// Whether embeddings and lm_head share weights.
    public var tiedEmbeddings: Bool {
        return tieWordEmbeddings ?? false
    }
}

/// MXQ quantization configuration (from jang_config.json).
public struct JANGQuantConfig: Sendable {
    public let formatVersion: String
    public let targetBits: Float
    public let actualBits: Float
    public let blockSize: Int
    public let bitWidthsUsed: [Int]
    public let sourceModelName: String
    public let totalWeightBytes: Int

    public init(from dict: [String: Any]) throws {
        guard let format = dict["format"] as? String, format == "jang" else {
            throw JANGError.invalidFormat("format field must be 'mxq'")
        }

        self.formatVersion = dict["format_version"] as? String ?? "1.0"

        guard let quant = dict["quantization"] as? [String: Any] else {
            throw JANGError.invalidFormat("missing 'quantization' section")
        }

        self.targetBits = (quant["target_bits"] as? NSNumber)?.floatValue ?? 2.5
        self.actualBits = (quant["actual_bits"] as? NSNumber)?.floatValue ?? self.targetBits
        self.blockSize = (quant["block_size"] as? Int) ?? 64
        self.bitWidthsUsed = (quant["bit_widths_used"] as? [Int]) ?? [2, 3, 4]

        let source = dict["source_model"] as? [String: Any] ?? [:]
        self.sourceModelName = source["name"] as? String ?? "unknown"

        let runtime = dict["runtime"] as? [String: Any] ?? [:]
        self.totalWeightBytes = runtime["total_weight_bytes"] as? Int ?? 0
    }
}

/// Combined model + quantization config.
public struct JANGModelConfig: Sendable {
    public let model: ModelConfig
    public let quant: JANGQuantConfig
    public let modelPath: URL

    public static func load(from path: URL) throws -> JANGModelConfig {
        // Load config.json
        let configURL = path.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let decoder = JSONDecoder()
        let model = try decoder.decode(ModelConfig.self, from: configData)

        // Load jang_config.json
        let mxqConfigURL = path.appendingPathComponent("jang_config.json")
        let mxqData = try Data(contentsOf: mxqConfigURL)
        guard let mxqDict = try JSONSerialization.jsonObject(with: mxqData) as? [String: Any] else {
            throw JANGError.invalidFormat("jang_config.json is not a valid JSON object")
        }
        let quant = try JANGQuantConfig(from: mxqDict)

        return JANGModelConfig(model: model, quant: quant, modelPath: path)
    }
}
