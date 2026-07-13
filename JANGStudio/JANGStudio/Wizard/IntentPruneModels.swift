// JANGStudio/JANGStudio/Wizard/IntentPruneModels.swift
// Intent Prune product types: keep/drop chips, stance, budget, workflow stages.
import Foundation
import SwiftUI

/// Capability chips shown in Intent Prune UI (keep intents).
/// Maps to suite domain slugs for `jang intent-prune-score --intent`.
enum IntentPruneChip: String, CaseIterable, Identifiable, Hashable, Sendable {
    case coding
    case math
    case writing
    case science
    case multilingual
    case tools
    case longContext
    case generalist
    case english
    case reasoning

    var id: String { rawValue }

    var title: String {
        switch self {
        case .coding: return "Coding"
        case .math: return "Math"
        case .writing: return "Writing"
        case .science: return "Science / Bio"
        case .multilingual: return "Multilingual"
        case .tools: return "Tools / Agentic"
        case .longContext: return "Long context"
        case .generalist: return "Generalist"
        case .english: return "English-dominant"
        case .reasoning: return "Reasoning"
        }
    }

    /// One-line description for the Shape selection surface.
    var detail: String {
        switch self {
        case .coding: return "Code gen, debugging, formats"
        case .math: return "Arithmetic, algebra, quantitative"
        case .writing: return "Prose, instructions, creative"
        case .science: return "Factual / scientific knowledge"
        case .multilingual: return "Non-English + translation"
        case .tools: return "Tool use, agents, planning"
        case .longContext: return "Long-doc / long-range use"
        case .generalist: return "Broader backbone (less specialist)"
        case .english: return "English-first chat / QA"
        case .reasoning: return "Logic / multi-step (beyond math)"
        }
    }

    /// Domain keys passed to hybrid scorer (`--intent` / intents_keep).
    var domainKeys: [String] {
        switch self {
        case .coding: return ["code", "coding", "formatting"]
        case .math: return ["math"]
        case .writing: return ["instruction_following", "creative"]
        case .science: return ["knowledge", "medical_sensitive"]
        case .multilingual: return ["multilingual", "non_english", "chinese", "translation", "english_dominant"]
        case .tools: return ["tools"]
        case .longContext: return ["long_context"]
        case .generalist: return []
        case .english: return ["english_dominant"]
        case .reasoning: return ["reasoning"]
        }
    }

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
        case .english: return "english"
        case .reasoning: return "reasoning"
        }
    }

    /// Semantic accent for keep tiles (design-b domain colors).
    var accentColor: Color {
        switch self {
        case .coding: return Color(red: 0.31, green: 0.76, blue: 0.97)
        case .math: return Color(red: 0.67, green: 0.28, blue: 0.74)
        case .writing: return Color(red: 0.81, green: 0.58, blue: 0.85)
        case .science: return Color(red: 0.55, green: 0.43, blue: 0.39)
        case .multilingual: return Color(red: 1.0, green: 0.54, blue: 0.40)
        case .tools: return Color(red: 0.15, green: 0.78, blue: 0.85)
        case .longContext: return Color(red: 0.56, green: 0.64, blue: 0.68)
        case .generalist: return Color(red: 0.33, green: 0.43, blue: 0.48)
        case .english: return Color(red: 0.47, green: 0.56, blue: 0.61)
        case .reasoning: return Color(red: 0.40, green: 0.73, blue: 0.42)
        }
    }
}

/// Drop / deprioritize intents → `jang intent-prune-score --drop-intent`.
enum IntentPruneDropChip: String, CaseIterable, Identifiable, Hashable, Sendable {
    case chinese
    case multilingual
    case translation
    case spanish
    case creative
    case knowledge
    case tools
    case longContext
    case safetyHeavy

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chinese: return "Chinese"
        case .multilingual: return "All multilingual"
        case .translation: return "Translation"
        case .spanish: return "Spanish / Romance"
        case .creative: return "Creative writing"
        case .knowledge: return "Trivia / knowledge"
        case .tools: return "Tools / agent"
        case .longContext: return "Long context"
        case .safetyHeavy: return "Safety-heavy (CRACK)"
        }
    }

    var detail: String {
        switch self {
        case .chinese: return "ZH generation / ZH routing mass"
        case .multilingual: return "Non-English broadly"
        case .translation: return "Translate-heavy paths"
        case .spanish: return "ES and related (when tagged)"
        case .creative: return "Story / roleplay mass"
        case .knowledge: return "Factual dump paths"
        case .tools: return "If you want chat-only"
        case .longContext: return "If short-form only"
        case .safetyHeavy: return "Prefer CRACK stance for abliteration"
        }
    }

    /// Domain keys for `--drop-intent` (safetyHeavy is stance-only).
    var domainKeys: [String] {
        switch self {
        case .chinese: return ["chinese"]
        case .multilingual: return ["multilingual", "non_english"]
        case .translation: return ["translation"]
        case .spanish: return ["spanish", "non_english"]
        case .creative: return ["creative"]
        case .knowledge: return ["knowledge"]
        case .tools: return ["tools"]
        case .longContext: return ["long_context"]
        case .safetyHeavy: return []
        }
    }

    var switchesToCrackStance: Bool { self == .safetyHeavy }

    var accentColor: Color {
        switch self {
        case .chinese, .multilingual, .translation, .spanish:
            return Color(red: 1.0, green: 0.54, blue: 0.40)
        case .creative: return Color(red: 0.81, green: 0.58, blue: 0.85)
        case .knowledge: return Color(red: 0.55, green: 0.43, blue: 0.39)
        case .tools: return Color(red: 0.15, green: 0.78, blue: 0.85)
        case .longContext: return Color(red: 0.56, green: 0.64, blue: 0.68)
        case .safetyHeavy: return Color(red: 0.94, green: 0.33, blue: 0.31)
        }
    }
}

/// Quick plan presets for the Shape surface.
struct IntentPrunePreset: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let keep: Set<IntentPruneChip>
    let drop: Set<IntentPruneDropChip>

    static let all: [IntentPrunePreset] = [
        IntentPrunePreset(id: "coding-math", title: "Coding + Math", keep: [.coding, .math], drop: []),
        IntentPrunePreset(id: "coding-only", title: "Coding specialist", keep: [.coding, .tools], drop: [.creative, .chinese]),
        IntentPrunePreset(id: "en-agent", title: "EN agent", keep: [.coding, .tools, .english, .reasoning], drop: [.chinese, .multilingual]),
        IntentPrunePreset(id: "drop-zh", title: "Drop Chinese", keep: [.coding, .math, .english], drop: [.chinese]),
        IntentPrunePreset(id: "general", title: "Generalist keep", keep: [.generalist, .english, .reasoning], drop: []),
    ]
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

    var keepFraction: Double {
        switch self {
        case .light: return 0.90
        case .standard: return 0.75
        case .aggressive: return 0.60
        }
    }

    func keepK(expertsPerLayer: Int, trainedTopK: Int = 1) -> Int {
        let e = max(expertsPerLayer, 1)
        let raw = Int((Double(e) * keepFraction).rounded())
        let floor = max(trainedTopK, 1)
        return min(e, max(floor, raw))
    }
}

/// Workflow stages (design-b Intent Prune quality loop).
enum IntentPruneStage: Int, CaseIterable, Identifiable, Sendable {
    case evidence = 0
    case shape = 1
    case prune = 2
    case quality = 3
    case convert = 4

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .evidence: return "Evidence"
        case .shape: return "Shape"
        case .prune: return "Prune"
        case .quality: return "Quality"
        case .convert: return "Convert"
        }
    }

    var headline: String {
        switch self {
        case .evidence: return "Collect real routing evidence"
        case .shape: return "Choose what to keep & drop"
        case .prune: return "Hard-prune BF16 source"
        case .quality: return "Prove quality on holdouts"
        case .convert: return "Convert pruned model"
        }
    }

    var dockSymbol: String {
        switch self {
        case .evidence: return "waveform.path.ecg"
        case .shape: return "circle.grid.cross"
        case .prune: return "scissors"
        case .quality: return "checkmark.seal"
        case .convert: return "arrow.right.circle"
        }
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

    var progress: Double {
        switch self {
        case .idle: return 0
        case .preparing: return 0.1
        case .scoring: return 0.35
        case .writingPlan: return 0.5
        case .hardPruning: return 0.75
        case .verifying: return 0.9
        case .ready: return 1.0
        case .failed: return 0
        }
    }
}
