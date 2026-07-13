// JANGStudio/JANGStudio/Wizard/IntentPruneModels.swift
// Intent Prune v1 product types (plan §6): chips, safety stance, budget → keep-K.
import Foundation

/// Capability chips shown in Intent Prune UI. Maps to `ExpertDomainTaxonomy` /
/// suite domain slugs consumed by `jang intent-prune-score --intent`.
enum IntentPruneChip: String, CaseIterable, Identifiable, Hashable, Sendable {
    case coding
    case math
    case writing
    case science
    case multilingual
    case tools
    case longContext
    case generalist

    var id: String { rawValue }

    var title: String {
        switch self {
        case .coding: return "Coding"
        case .math: return "Math"
        case .writing: return "Writing"
        case .science: return "Science/Bio"
        case .multilingual: return "Multilingual"
        case .tools: return "Tools/Agentic"
        case .longContext: return "Long context"
        case .generalist: return "Generalist"
        }
    }

    /// Domain keys passed to hybrid scorer (`--intent` / intents_keep).
    var domainKeys: [String] {
        switch self {
        case .coding: return ["code", "coding", "formatting"]
        case .math: return ["math", "reasoning"]
        case .writing: return ["instruction_following", "creative"]
        case .science: return ["knowledge", "medical_sensitive"]
        case .multilingual: return ["multilingual", "non_english", "chinese", "translation", "english_dominant"]
        case .tools: return ["tools", "reasoning"]
        case .longContext: return ["long_context"]
        case .generalist: return [] // increases global path weight slightly via backbone_floor
        }
    }

    /// Short slug for artifact folder names (`-intent-{slug}-k{K}`).
    var artifactSlug: String {
        switch self {
        case .coding: return "coding"
        case .math: return "math"
        case .writing: return "writing"
        case .science: return "science"
        case .multilingual: return "multilingual"
        case .tools: return "tools"
        case .longContext: return "long-context"
        case .generalist: return "generalist"
        }
    }
}

/// Safety stance (plan §7). Default when opening Intent Prune is **Keep**.
enum IntentPruneSafetyStance: String, CaseIterable, Identifiable, Hashable, Sendable {
    case keep
    case balanced
    case crack

    var id: String { rawValue }

    var title: String {
        switch self {
        case .keep: return "Keep"
        case .balanced: return "Balanced"
        case .crack: return "CRACK"
        }
    }

    var subtitle: String {
        switch self {
        case .keep: return "Protect safety-path experts (conservative)"
        case .balanced: return "Mild safety protection"
        case .crack: return "Abliteration — down-rank refusal-path specialists"
        }
    }

    /// CLI `--safety-stance` value.
    var cliValue: String { rawValue }

    var isCrack: Bool { self == .crack }
}

/// Size budget presets → keep fraction → uniform keep-K (plan §6.4).
enum IntentPruneBudget: String, CaseIterable, Identifiable, Hashable, Sendable {
    case light
    case standard
    case aggressive

    var id: String { rawValue }

    var title: String {
        switch self {
        case .light: return "Light"
        case .standard: return "Standard"
        case .aggressive: return "Aggressive"
        }
    }

    /// Target keep fraction of experts per layer.
    var keepFraction: Double {
        switch self {
        case .light: return 0.90
        case .standard: return 0.75
        case .aggressive: return 0.60
        }
    }

    /// Uniform keep-K, clamped to `[max(trainedTopK, 1), E]`.
    func keepK(expertsPerLayer: Int, trainedTopK: Int = 1) -> Int {
        let e = max(expertsPerLayer, 1)
        let raw = Int((Double(e) * keepFraction).rounded())
        let floor = max(trainedTopK, 1)
        return min(e, max(floor, raw))
    }
}

/// Progress phases during Intent Prune run (plan §6.5).
enum IntentPrunePhase: String, CaseIterable, Identifiable, Sendable {
    case idle
    case preparing
    case scoring
    case writingPlan
    case hardPruning
    case verifying
    case ready
    case failed

    var id: String { rawValue }

    var label: String {
        switch self {
        case .idle: return "Ready"
        case .preparing: return "Preparing work directory…"
        case .scoring: return "Scoring hybrid (path + mass + domain)…"
        case .writingPlan: return "Writing prune plan…"
        case .hardPruning: return "Hard-pruning BF16/F16 source…"
        case .verifying: return "Verifying pruned source structure…"
        case .ready: return "Ready for Convert"
        case .failed: return "Failed"
        }
    }

    /// 0…1 progress for UI bars (approximate stage index).
    var progress: Double {
        switch self {
        case .idle: return 0
        case .preparing: return 0.08
        case .scoring: return 0.30
        case .writingPlan: return 0.45
        case .hardPruning: return 0.70
        case .verifying: return 0.90
        case .ready: return 1.0
        case .failed: return 0
        }
    }
}
