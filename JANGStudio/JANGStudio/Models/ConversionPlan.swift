// JANGStudio/JANGStudio/Models/ConversionPlan.swift
import Foundation
import Observation

enum Family: String, Codable, CaseIterable { case jang, jangtq }
enum QuantMethod: String, Codable, CaseIterable { case mse, rtn, mseAll }
enum SourceDtype: String, Codable { case bf16, fp16, fp8, jangV2, unknown }
enum RunState: String, Codable { case idle, running, succeeded, failed, cancelled }
enum ExpertReviewIntent: String, Codable { case none, smartPrequantPrune }

/// Per-conversion product path. Independent of `ExpertReviewIntent`
/// (intent is the in-lab prune workflow state; mode is navigation).
enum WizardMode: String, Codable, CaseIterable {
    case convert
    case expertLab
}

struct ArchitectureSummary: Codable, Equatable {
    let modelType: String
    let textModelType: String?
    let isMoE: Bool
    let numExperts: Int
    let numHiddenLayers: Int?
    let numExpertsPerTok: Int?
    let isVL: Bool
    let isVideoVL: Bool
    let hasGenerationConfig: Bool
    let dtype: SourceDtype
    let totalBytes: Int64
    let shardCount: Int

    init(modelType: String, isMoE: Bool, numExperts: Int, isVL: Bool,
         isVideoVL: Bool = false, hasGenerationConfig: Bool = false,
         dtype: SourceDtype, totalBytes: Int64, shardCount: Int,
         textModelType: String? = nil, numHiddenLayers: Int? = nil,
         numExpertsPerTok: Int? = nil) {
        self.modelType = modelType
        self.textModelType = textModelType
        self.isMoE = isMoE
        self.numExperts = numExperts
        self.numHiddenLayers = numHiddenLayers
        self.numExpertsPerTok = numExpertsPerTok
        self.isVL = isVL
        self.isVideoVL = isVideoVL
        self.hasGenerationConfig = hasGenerationConfig
        self.dtype = dtype
        self.totalBytes = totalBytes
        self.shardCount = shardCount
    }

    var routedExpertTotal: Int? {
        guard isMoE,
              numExperts > 0,
              let layers = numHiddenLayers,
              layers > 0 else {
            return nil
        }
        return layers * numExperts
    }

    var architectureModelTypes: [String] {
        [modelType, textModelType].compactMap { raw in
            let normalized = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return normalized?.isEmpty == false ? normalized : nil
        }
    }

    var parameterSummary: String {
        guard isMoE else { return "Dense" }

        var parts = ["MoE"]
        if let layers = numHiddenLayers, layers > 0 {
            var shape = "\(layers) layers x \(numExperts) experts"
            if let total = routedExpertTotal {
                shape += " (\(Self.groupedDecimal(total)) total)"
            }
            parts.append(shape)
        } else {
            parts.append("\(numExperts) experts")
        }
        if let topK = numExpertsPerTok, topK > 0 {
            parts.append("top-\(topK)")
        }
        return parts.joined(separator: " · ")
    }

    private static func groupedDecimal(_ value: Int) -> String {
        let raw = String(value)
        var result = ""
        for (index, character) in raw.reversed().enumerated() {
            if index > 0, index.isMultiple(of: 3) {
                result.insert(",", at: result.startIndex)
            }
            result.insert(character, at: result.startIndex)
        }
        return result
    }
}

struct ArchitectureOverrides: Codable, Equatable {
    var forceDtype: SourceDtype? = nil
    var forceBlockSize: Int? = nil
    var skipPatterns: [String] = []
    // M200 (iter 137): `calibrationJSONL` removed — zero downstream
    // consumers across the entire JANGStudio codebase, would have
    // been a settings-UI lie if any wizard step had offered a picker.
    // No UI currently exposes it, so removing the declaration doesn't
    // affect any user-visible surface; this is pure dead-field cleanup
    // bundled with the defaultCalibrationSamples removal.
    // Codable forward-compat: pre-M200 persisted ArchitectureOverrides
    // JSON blobs (e.g., via plan export) that contain the key will
    // decode cleanly because JSONDecoder tolerates unknown keys.
}

@Observable
final class ConversionPlan: Codable {
    var sourceURL: URL?
    var detected: ArchitectureSummary?
    var overrides = ArchitectureOverrides()
    var family: Family = .jang
    var profile: String = "JANG_4K"
    var method: QuantMethod = .mse
    var hadamard: Bool = false
    var outputURL: URL?
    var run: RunState = .idle
    /// Convert (4-step) vs Expert Lab MoE (6-step) navigation mode.
    /// Default `.convert` keeps the short happy path.
    var workflowMode: WizardMode = .convert
    var expertReviewIntent: ExpertReviewIntent = .none
    var expertReviewBundleProfile: String = "JANGTQ3"
    var expertReviewBundleURL: URL?
    var expertReviewSourceURL: URL?
    var expertReviewPlanURL: URL?
    var expertReviewOriginalSourceURL: URL?
    var expertReviewPrunedSourceURL: URL?
    var expertReviewPrunePlanURL: URL?
    var expertReviewPruneReportURL: URL?

    init() {}

    /// Seed a fresh plan with the user's configured defaults. Only fields that
    /// are safe to auto-populate at wizard entry get touched — specifically
    /// the knobs that live in Settings → General → Defaults. `sourceURL`,
    /// `detected`, `outputURL`, and `run` are intentionally untouched: those
    /// are per-conversion state, not user defaults.
    ///
    /// Introduced iter 10 (M62 chain): previously `defaultProfile`,
    /// `defaultFamily`, `defaultMethod`, `defaultHadamardEnabled` were
    /// persisted but never read anywhere — the wizard always started on
    /// `JANG_4K` / `jang` / `mse` / hadamard=false regardless of what the
    /// user set in Settings.
    @MainActor
    func applyDefaults(from settings: AppSettings) {
        // Profile: only apply if the settings value is non-empty — defends
        // against first-launch or corrupted UserDefaults where profile is "".
        if !settings.defaultProfile.isEmpty {
            profile = settings.defaultProfile
        }
        if let fam = Family(rawValue: settings.defaultFamily) {
            family = fam
        }
        switch settings.defaultMethod.lowercased() {
        case "mse": method = .mse
        case "rtn": method = .rtn
        case "mse-all", "mseall", "mse_all": method = .mseAll
        default: break   // unknown value → keep current default
        }
        hadamard = settings.defaultHadamardEnabled
    }

    /// Step 1 completes only when we've picked a folder AND detection found a
    /// real model there — meaning at least one .safetensors shard is present.
    /// A folder with just a config.json and nothing else is NOT a complete step 1;
    /// surfacing this prevents silent progression when the user picks the wrong
    /// folder (e.g., a parent directory, a docs folder, or a broken download).
    var isStep1Complete: Bool {
        sourceURL != nil && detected != nil && (detected?.shardCount ?? 0) > 0
    }
    /// Historically “architecture confirmed”. Architecture is no longer a
    /// wizard step; profile remains unlocked once source detection succeeds.
    var isStep2Complete: Bool { isStep1Complete }
    var isStep3Complete: Bool { isStep2Complete && outputURL != nil }
    var isStep4Complete: Bool { run == .succeeded }

    /// True when an Expert Lab session is mid-flight (not merely post-adopt
    /// pruned-source rails). Used by Codable migration heal.
    var hasInProgressExpertLabSession: Bool {
        expertReviewIntent == .smartPrequantPrune
            || (expertReviewPlanURL != nil && expertReviewSourceURL != nil)
    }

    func adoptReviewedPrunedSource(_ prunedURL: URL) {
        let originalSource = expertReviewSourceURL
        let prunedPlan = prunedURL.appendingPathComponent("prune_plan.json")
        let reviewBundle = expertReviewBundleURL ?? outputURL
        sourceURL = prunedURL
        outputURL = nil
        run = .idle
        expertReviewIntent = .none
        expertReviewBundleURL = reviewBundle
        expertReviewSourceURL = nil
        expertReviewPlanURL = nil
        expertReviewOriginalSourceURL = originalSource
        expertReviewPrunedSourceURL = prunedURL
        expertReviewPrunePlanURL = prunedPlan
        expertReviewPruneReportURL = prunedURL.appendingPathComponent("expert_lab_prune_report.md")
        // Prune done; final quantize is the Convert path (hide Expert steps).
        workflowMode = .convert
    }

    func isJANGTQAllowed(for whitelist: [String]) -> Bool {
        guard let detected else { return false }
        return detected.architectureModelTypes.contains { whitelist.contains($0) }
    }

    @available(*, deprecated, message: "Use isJANGTQAllowed(for:) with CapabilitiesService.capabilities.jangtqWhitelist")
    var isJANGTQAllowed: Bool {
        guard let detected else { return false }
        let whitelist = ["qwen3_5_moe", "qwen3_5_moe_text", "minimax_m2"]
        return detected.architectureModelTypes.contains { whitelist.contains($0) }
    }

    // MARK: - UserDefaults persistence

    // @Observable rewrites stored properties so synthesized Codable doesn't work.
    // Provide explicit encode/decode keyed on the same names.
    enum CodingKeys: String, CodingKey {
        case sourceURL, detected, overrides, family, profile, method, hadamard, outputURL
        case workflowMode
        case expertReviewIntent, expertReviewBundleProfile, expertReviewBundleURL
        case expertReviewSourceURL, expertReviewPlanURL
        case expertReviewOriginalSourceURL, expertReviewPrunedSourceURL
        case expertReviewPrunePlanURL, expertReviewPruneReportURL
    }

    required init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sourceURL  = try c.decodeIfPresent(URL.self,                  forKey: .sourceURL)
        detected   = try c.decodeIfPresent(ArchitectureSummary.self,  forKey: .detected)
        overrides  = try c.decodeIfPresent(ArchitectureOverrides.self, forKey: .overrides) ?? ArchitectureOverrides()
        family     = try c.decodeIfPresent(Family.self,               forKey: .family)    ?? .jang
        profile    = try c.decodeIfPresent(String.self,               forKey: .profile)   ?? "JANG_4K"
        method     = try c.decodeIfPresent(QuantMethod.self,          forKey: .method)    ?? .mse
        hadamard   = try c.decodeIfPresent(Bool.self,                 forKey: .hadamard)  ?? false
        outputURL  = try c.decodeIfPresent(URL.self,                  forKey: .outputURL)
        workflowMode = try c.decodeIfPresent(WizardMode.self,         forKey: .workflowMode) ?? .convert
        expertReviewIntent = try c.decodeIfPresent(ExpertReviewIntent.self, forKey: .expertReviewIntent) ?? .none
        expertReviewBundleProfile = try c.decodeIfPresent(String.self, forKey: .expertReviewBundleProfile) ?? "JANGTQ3"
        expertReviewBundleURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewBundleURL)
        expertReviewSourceURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewSourceURL)
        expertReviewPlanURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewPlanURL)
        expertReviewOriginalSourceURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewOriginalSourceURL)
        expertReviewPrunedSourceURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewPrunedSourceURL)
        expertReviewPrunePlanURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewPrunePlanURL)
        expertReviewPruneReportURL = try c.decodeIfPresent(URL.self, forKey: .expertReviewPruneReportURL)

        // Migration / heal (K12): if mode is convert but an in-progress
        // Expert Lab session is present (missing key OR explicit convert
        // with residual session fields), upgrade to expertLab. Do NOT
        // upgrade solely on expertReviewPrunedSourceURL (post-adopt).
        // Explicit expertLab is never downgraded here.
        if workflowMode == .convert {
            let inProgressSession =
                expertReviewIntent == .smartPrequantPrune
                || (expertReviewPlanURL != nil && expertReviewSourceURL != nil)
            if inProgressSession {
                workflowMode = .expertLab
            }
        }
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(sourceURL, forKey: .sourceURL)
        try c.encodeIfPresent(detected,  forKey: .detected)
        try c.encode(overrides,          forKey: .overrides)
        try c.encode(family,             forKey: .family)
        try c.encode(profile,            forKey: .profile)
        try c.encode(method,             forKey: .method)
        try c.encode(hadamard,           forKey: .hadamard)
        try c.encodeIfPresent(outputURL, forKey: .outputURL)
        try c.encode(workflowMode,       forKey: .workflowMode)
        try c.encode(expertReviewIntent, forKey: .expertReviewIntent)
        try c.encode(expertReviewBundleProfile, forKey: .expertReviewBundleProfile)
        try c.encodeIfPresent(expertReviewBundleURL, forKey: .expertReviewBundleURL)
        try c.encodeIfPresent(expertReviewSourceURL, forKey: .expertReviewSourceURL)
        try c.encodeIfPresent(expertReviewPlanURL, forKey: .expertReviewPlanURL)
        try c.encodeIfPresent(expertReviewOriginalSourceURL, forKey: .expertReviewOriginalSourceURL)
        try c.encodeIfPresent(expertReviewPrunedSourceURL, forKey: .expertReviewPrunedSourceURL)
        try c.encodeIfPresent(expertReviewPrunePlanURL, forKey: .expertReviewPrunePlanURL)
        try c.encodeIfPresent(expertReviewPruneReportURL, forKey: .expertReviewPruneReportURL)
    }

    func encodeForDefaults() throws -> Data { try JSONEncoder().encode(self) }
    static func decodeFromDefaults(_ data: Data) throws -> ConversionPlan {
        try JSONDecoder().decode(ConversionPlan.self, from: data)
    }
}
