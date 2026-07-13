// JANGStudio/JANGStudio/Runner/CapabilitiesService.swift
import Foundation
import Observation

/// Capabilities fetched from `jang-tools capabilities --json`. Cached for the
/// app lifetime after first successful load. Falls back to a frozen snapshot
/// if the CLI invocation fails (e.g., Debug builds without a bundled runtime).
struct Capabilities: Codable, Equatable, Sendable {
    let jangtqWhitelist: [String]
    let knownExpert512Types: [String]
    let supportedSourceDtypes: [DtypeInfo]
    let blockSizes: [Int]
    let defaultBlockSize: Int
    let methods: [MethodInfo]
    let defaultMethod: String
    let tokenizerClassBlocklist: [String]
    let hadamardDefaultForBitTier: [String: Bool]

    struct DtypeInfo: Codable, Equatable, Sendable {
        let name: String
        let alias: String
        let description: String
    }
    struct MethodInfo: Codable, Equatable, Sendable {
        let name: String
        let description: String
    }

    enum CodingKeys: String, CodingKey {
        case jangtqWhitelist = "jangtq_whitelist"
        case knownExpert512Types = "known_512_expert_types"
        case supportedSourceDtypes = "supported_source_dtypes"
        case blockSizes = "block_sizes"
        case defaultBlockSize = "default_block_size"
        case methods
        case defaultMethod = "default_method"
        case tokenizerClassBlocklist = "tokenizer_class_blocklist"
        case hadamardDefaultForBitTier = "hadamard_default_for_bit_tier"
    }

    /// Frozen fallback used when the CLI can't be reached. Must stay in sync with
    /// jang-tools/jang_tools/capabilities_cli.py — treat this as "last known good"
    /// for offline UI rendering; real behavior still requires a working runtime.
    static let frozen: Capabilities = .init(
        jangtqWhitelist: ["minimax_m2", "qwen3_5_moe", "qwen3_5_moe_text"],
        knownExpert512Types: ["minimax_m2", "glm_moe_dsa"],
        supportedSourceDtypes: [
            .init(name: "bfloat16", alias: "bf16", description: "HuggingFace standard for modern LLMs"),
            .init(name: "float16", alias: "fp16", description: "Older standard; overflow risk on 512+ expert models"),
            .init(name: "float8_e4m3fn", alias: "fp8", description: "8-bit float (MiniMax, DeepSeek native)"),
            .init(name: "float8_e5m2", alias: "fp8-e5m2", description: "Alternative FP8 encoding"),
        ],
        blockSizes: [32, 64, 128],
        defaultBlockSize: 64,
        methods: [
            .init(name: "mse", description: "Minimum-square-error weight search (default)"),
            .init(name: "rtn", description: "Round-to-nearest (fastest)"),
            .init(name: "mse-all", description: "MSE across all layer classes"),
        ],
        defaultMethod: "mse",
        tokenizerClassBlocklist: ["TokenizersBackend"],
        hadamardDefaultForBitTier: ["2": false, "3": true, "4": true, "5": true, "6": true]
    )
}

/// M129 (iter 51): typed error parity with RecommendationService /
/// ExamplesService / ModelCardService. Previously `invokeCLI` threw a raw
/// `NSError(domain: "CapabilitiesService")` which made `refresh()`'s
/// `self.lastError = "\(error)"` stringify into an ugly
/// `Error Domain=CapabilitiesService Code=1 "(null)" UserInfo={…}`
/// banner. The typed case gives a clean `errorDescription`:
///   "jang-tools capabilities exited N: <stderr>"
/// matching the peer adoption services' UX.
enum CapabilitiesServiceError: Error, LocalizedError {
    case cliError(code: Int32, stderr: String)

    var errorDescription: String? {
        switch self {
        case .cliError(let code, let stderr):
            return "jang-tools capabilities exited \(code): \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
    }
}

@Observable
@MainActor
final class CapabilitiesService {
    private(set) var capabilities: Capabilities = .frozen
    private(set) var isFromBundle: Bool = false   // true when loaded from CLI, false when frozen fallback
    private(set) var lastError: String? = nil

    /// Refresh by invoking `python -m jang_tools capabilities --json`.
    /// Safe to call multiple times; no-op'd if already loaded.
    func refresh() async {
        if isFromBundle { return }
        do {
            let data = try await Self.invokeCLI(args: ["-m", "jang_tools", "capabilities", "--json"])
            let decoded = try JSONDecoder().decode(Capabilities.self, from: data)
            self.capabilities = decoded
            self.isFromBundle = true
            self.lastError = nil
        } catch let e as CapabilitiesServiceError {
            // M129: use errorDescription so the banner reads cleanly.
            self.lastError = e.errorDescription ?? "\(e)"
        } catch {
            self.lastError = "\(error)"
            // Stick with .frozen
        }
    }

    private static func invokeCLI(args: [String]) async throws -> Data {
        // M153 (iter 76): migrated to shared PythonCLIInvoker.
        try await PythonCLIInvoker.invoke(args: args) { code, stderr in
            CapabilitiesServiceError.cliError(code: code, stderr: stderr)
        }
    }
}
