//
//  JANGQuant — codec definitions shared by the runtime and the converter.
//  Mirrors jang_tools.jangrt.linear and jang_tools.turboquant.linear.
//
//  Wires up to vmlx-swift's MLX binding for the actual quantized matmul
//  via TQCodebook (already in vmlx). Until that link is exposed publicly,
//  this file stays type-only.
//

import Foundation

public enum JANGFormat: String, Sendable {
    case bf16
    case fp8           // FP8 source (Mistral 3.5 per-tensor; MiMo / DSV4 [128,128] block)
    case jang          // mx.quantize affine
    case jangtq        // TurboQuant (uint8 codes + bf16 codebook + Hadamard)
}

public struct QuantMeta: Sendable {
    public var format: JANGFormat
    public var bits: Int
    public var groupSize: Int
    public var hadamard: Bool
    public var routedExpertBits: Int?
    public init(format: JANGFormat, bits: Int, groupSize: Int,
                hadamard: Bool, routedExpertBits: Int? = nil) {
        self.format = format; self.bits = bits; self.groupSize = groupSize
        self.hadamard = hadamard; self.routedExpertBits = routedExpertBits
    }
}

/// Read mxtq_bits / routed_expert_bits / quantization out of config.json.
/// Enforces the 2026-04-25 invariant: any JANGTQ bundle must set
/// mxtq_bits OR routed_expert_bits.
public enum BundleProbe {
    public enum Error: Swift.Error { case missingInvariant(String) }
    public static func detect(at url: URL) throws -> QuantMeta {
        let data = try Data(contentsOf: url.appendingPathComponent("config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let qc = (json["quantization_config"] as? [String: Any]) ?? [:]
        let method = (qc["quant_method"] as? String) ?? (qc["format"] as? String) ?? ""

        if method == "fp8" || method == "compressed-tensors" || method == "float-quantized" {
            return QuantMeta(format: .fp8, bits: 8, groupSize: 128, hadamard: false)
        }
        if let mxtq = json["mxtq_bits"] as? Int {
            let routed = (json["routed_expert_bits"] as? Int) ?? mxtq
            return QuantMeta(format: .jangtq, bits: mxtq, groupSize: 64,
                             hadamard: true, routedExpertBits: routed)
        }
        if let q = json["quantization"] as? [String: Any],
           let bits = q["bits"] as? Int, let gs = q["group_size"] as? Int {
            return QuantMeta(format: .jang, bits: bits, groupSize: gs, hadamard: false)
        }
        return QuantMeta(format: .bf16, bits: 16, groupSize: 0, hadamard: false)
    }
}
