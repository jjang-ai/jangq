// JANGStudio/JANGStudio/Runner/ProfilesService.swift
import Foundation
import Observation

struct ProfileInfo: Codable, Equatable, Identifiable, Hashable, Sendable {
    let name: String
    let criticalBits: Int?
    let importantBits: Int?
    let compressBits: Int?
    let avgBits: Double
    let description: String
    let isDefault: Bool
    let isKquant: Bool

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name
        case criticalBits = "critical_bits"
        case importantBits = "important_bits"
        case compressBits = "compress_bits"
        case avgBits = "avg_bits"
        case description
        case isDefault = "is_default"
        case isKquant = "is_kquant"
    }
}

struct JANGTQProfileInfo: Codable, Equatable, Identifiable, Hashable, Sendable {
    let name: String
    let bits: Int
    let avgBits: Double?
    let minSourceDtype: [String]
    let description: String

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name
        case bits
        case avgBits = "avg_bits"
        case minSourceDtype = "min_source_dtype"
        case description
    }
}

struct JANGTQFamilyInfo: Codable, Equatable, Hashable, Sendable {
    let converter: String
    let profiles: [String]
    let defaultProfile: String
    let invocation: String
    let status: String
    let note: String

    enum CodingKeys: String, CodingKey {
        case converter, profiles, invocation, status, note
        case defaultProfile = "default_profile"
    }
}

struct Profiles: Codable, Equatable, Sendable {
    let jang: [ProfileInfo]
    let jangtq: [JANGTQProfileInfo]
    let jangtqFamilies: [String: JANGTQFamilyInfo]
    let defaultProfile: String
    let bitToProfile: [String: String]

    enum CodingKeys: String, CodingKey {
        case jang, jangtq
        case jangtqFamilies = "jangtq_families"
        case defaultProfile = "default_profile"
        case bitToProfile = "bit_to_profile"
    }

    func jangtqProfileNames(for modelType: String?) -> [String] {
        guard let modelType, let info = jangtqFamilies[modelType] else {
            return jangtq.map(\.name)
        }
        return info.profiles
    }

    func supportsJANGTQProfile(_ profile: String, for modelType: String?) -> Bool {
        guard let modelType, let info = jangtqFamilies[modelType] else { return false }
        return info.profiles.contains(profile)
    }

    /// Frozen fallback. Must stay in sync with jang-tools/jang_tools/profiles_cli.py.
    static let frozen: Profiles = .init(
        jang: [
            .init(name: "JANG_1L", criticalBits: 8, importantBits: 8, compressBits: 2, avgBits: 2.6, description: "Maximum-protection 1-bit tier", isDefault: false, isKquant: false),
            .init(name: "JANG_2S", criticalBits: 6, importantBits: 4, compressBits: 2, avgBits: 2.5, description: "Tightest 2-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_2M", criticalBits: 8, importantBits: 4, compressBits: 2, avgBits: 2.7, description: "Balanced 2-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_2L", criticalBits: 8, importantBits: 6, compressBits: 2, avgBits: 2.9, description: "Best-quality 2-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_3S", criticalBits: 6, importantBits: 3, compressBits: 3, avgBits: 3.15, description: "Small boost on attention only", isDefault: false, isKquant: false),
            .init(name: "JANG_3M", criticalBits: 8, importantBits: 3, compressBits: 3, avgBits: 3.25, description: "Full attention at 8-bit, rest 3-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_3L", criticalBits: 8, importantBits: 4, compressBits: 3, avgBits: 3.4, description: "Attention 8-bit, embeddings 4-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_3K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 3.0, description: "K-quant 3-bit (budget-neutral)", isDefault: false, isKquant: true),
            .init(name: "JANG_4S", criticalBits: 6, importantBits: 4, compressBits: 4, avgBits: 4.1, description: "Small boost", isDefault: false, isKquant: false),
            .init(name: "JANG_4M", criticalBits: 8, importantBits: 4, compressBits: 4, avgBits: 4.2, description: "Full attention at 8-bit, rest 4-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_4L", criticalBits: 8, importantBits: 6, compressBits: 4, avgBits: 4.4, description: "Attention 8-bit, embeddings 6-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_4K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 4.0, description: "K-quant 4-bit — THE DEFAULT", isDefault: true, isKquant: true),
            .init(name: "JANG_5K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 5.0, description: "K-quant 5-bit", isDefault: false, isKquant: true),
            .init(name: "JANG_6K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 6.0, description: "K-quant 6-bit", isDefault: false, isKquant: true),
            .init(name: "JANG_6M", criticalBits: 8, importantBits: 6, compressBits: 6, avgBits: 6.2, description: "Near-lossless", isDefault: false, isKquant: false),
        ],
        jangtq: [
            .init(name: "JANGTQ1", bits: 1, avgBits: 1.0, minSourceDtype: ["bfloat16"], description: "1-bit routed-expert TurboQuant; experimental and family-gated"),
            .init(name: "JANGTQ2", bits: 2, avgBits: 2.0, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "2-bit TurboQuant routed experts"),
            .init(name: "JANGTQ3", bits: 3, avgBits: 3.0, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "3-bit TurboQuant routed experts; only exposed where packing is proven"),
            .init(name: "JANGTQ4", bits: 4, avgBits: 4.0, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "4-bit TurboQuant routed experts"),
            .init(name: "JANGTQ_K", bits: 3, avgBits: 2.67, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "Mixed routed experts: gate/up 2-bit, down 4-bit"),
            .init(name: "JANGTQ_1L", bits: 2, avgBits: 2.6, minSourceDtype: ["bfloat16"], description: "Kimi legacy 1L policy"),
            .init(name: "JANGTQ_2L", bits: 2, avgBits: 3.0, minSourceDtype: ["bfloat16"], description: "Kimi legacy 2L policy"),
            .init(name: "JANGTQ_3L", bits: 3, avgBits: 4.0, minSourceDtype: ["bfloat16"], description: "Kimi legacy 3L policy"),
        ],
        jangtqFamilies: [
            "qwen3_5_moe": .init(converter: "jang_tools.convert_qwen35_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Qwen3.5/3.6 MoE hybrid converter"),
            "minimax_m2": .init(converter: "jang_tools.convert_minimax_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "MiniMax M2.x converter"),
            "minimax_m2_5": .init(converter: "jang_tools.convert_minimax_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "MiniMax M2.x alias"),
            "minimax": .init(converter: "jang_tools.convert_minimax_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "MiniMax M2.x alias"),
            "hy_v3": .init(converter: "jang_tools.convert_hy3_jangtq", profiles: ["JANGTQ2", "JANGTQ_K", "JANGTQ4", "JANGTQ1"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "JANGTQ1 is experimental"),
            "deepseek_v4": .init(converter: "jang_tools.dsv4.convert_dsv4_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ2", invocation: "dsv4_flags", status: "supported", note: "DSV4 flag-style converter"),
            "kimi_k25": .init(converter: "jang_tools.kimi_prune.convert_kimi_jangtq", profiles: ["JANGTQ_K", "JANGTQ_1L", "JANGTQ_2L", "JANGTQ_3L"], defaultProfile: "JANGTQ_K", invocation: "src_dst_profile_flags", status: "supported", note: "Kimi K2.6/KimiK25 wrapper"),
            "zaya": .init(converter: "jang_tools.convert_zaya_jangtq", profiles: ["JANGTQ2", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ_K", invocation: "positional_progress", status: "supported", note: "JANGTQ3 intentionally excluded"),
            "zaya1_vl": .init(converter: "jang_tools.convert_zaya1_vl_jangtq", profiles: ["JANGTQ2", "JANGTQ4", "JANGTQ_K"], defaultProfile: "JANGTQ_K", invocation: "positional_progress", status: "supported", note: "Vision sidecars preserved; JANGTQ3 intentionally excluded"),
            "bailing_hybrid": .init(converter: "jang_tools.convert_ling_jangtq", profiles: ["JANGTQ2", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "JANGTQ3 excluded until packing path is proven"),
            "bailing_moe_v2_5": .init(converter: "jang_tools.convert_ling_jangtq", profiles: ["JANGTQ2", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Ling/Bailing alias"),
            "nemotron_h": .init(converter: "jang_tools.convert_nemotron_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Text-only Nemotron-H/Omni converter"),
            "nemotron_h_v2": .init(converter: "jang_tools.convert_nemotron_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Nemotron-H alias"),
            "laguna": .init(converter: "jang_tools.convert_laguna_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Laguna XS.2 routed experts"),
            "mistral3": .init(converter: "jang_tools.convert_mistral3_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Mistral3/Pixtral converter"),
            "mistral4": .init(converter: "jang_tools.convert_mistral3_jangtq", profiles: ["JANGTQ2", "JANGTQ3", "JANGTQ4"], defaultProfile: "JANGTQ2", invocation: "positional_progress", status: "supported", note: "Mistral4 alias"),
        ],
        defaultProfile: "JANG_4K",
        bitToProfile: ["1": "JANG_1L", "2": "JANG_2S", "3": "JANG_3K", "4": "JANG_4K", "5": "JANG_5K", "6": "JANG_6K", "7": "JANG_6M", "8": "JANG_6M"]
    )
}

/// M129 (iter 51): typed error parity — see CapabilitiesServiceError.
enum ProfilesServiceError: Error, LocalizedError {
    case cliError(code: Int32, stderr: String)

    var errorDescription: String? {
        switch self {
        case .cliError(let code, let stderr):
            return "jang-tools profiles exited \(code): \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
    }
}

@Observable
@MainActor
final class ProfilesService {
    private(set) var profiles: Profiles = .frozen
    private(set) var isFromBundle: Bool = false
    private(set) var lastError: String? = nil

    func refresh() async {
        if isFromBundle { return }
        do {
            let data = try await Self.invokeCLI(args: ["-m", "jang_tools", "profiles", "--json"])
            let decoded = try JSONDecoder().decode(Profiles.self, from: data)
            self.profiles = decoded
            self.isFromBundle = true
            self.lastError = nil
        } catch let e as ProfilesServiceError {
            // M129: use errorDescription so the banner reads cleanly.
            self.lastError = e.errorDescription ?? "\(e)"
        } catch {
            self.lastError = "\(error)"
        }
    }

    private static func invokeCLI(args: [String]) async throws -> Data {
        // M153 (iter 76): migrated to shared PythonCLIInvoker.
        try await PythonCLIInvoker.invoke(args: args) { code, stderr in
            ProfilesServiceError.cliError(code: code, stderr: stderr)
        }
    }
}
