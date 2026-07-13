// JANGStudio/JANGStudio/Runner/IntentPruneCLIArgsBuilder.swift
// Pure argv builders for Intent Prune pipeline (plan §12 / IP4).
// No I/O — unit-testable; ViewModel shells these via PythonCLIInvoker.
import Foundation

/// Builds `python -m jang_tools …` argument lists for Intent Prune.
enum IntentPruneCLIArgsBuilder {
    /// Hybrid scorer: transitions/adjacency → `jang-intent-prune-plan-v1`.
    ///
    /// Mirrors `jang intent-prune-score` (see `jang_tools.intent_prune.cli`).
    static func scoreArgs(
        transitionsPath: String?,
        adjacencyPath: String? = nil,
        outputPlanPath: String,
        numExperts: Int,
        numLayers: Int?,
        keepK: Int,
        safetyStance: IntentPruneSafetyStance,
        intentsKeep: [String],
        intentsDrop: [String] = [],
        sourceModelPath: String? = nil,
        trainedTopK: Int? = nil,
        suiteName: String = "Reviewed Prune 50",
        suitePromptCount: Int = 50,
        preset: String = "balanced",
        backend: String = "qwen35_moe_vmlx",
        crackPackPath: String? = nil,
        noDefaultCrackPack: Bool = false
    ) -> [String] {
        var args: [String] = [
            "-m", "jang_tools",
            "--quiet-text",
            "intent-prune-score",
            "--output", outputPlanPath,
            "--num-experts", "\(numExperts)",
            "--keep-k", "\(keepK)",
            "--preset", preset,
            "--safety-stance", safetyStance.cliValue,
            "--backend", backend,
        ]
        if let transitionsPath, !transitionsPath.isEmpty {
            args.append(contentsOf: ["--transitions", transitionsPath])
        }
        if let adjacencyPath, !adjacencyPath.isEmpty {
            args.append(contentsOf: ["--adjacency", adjacencyPath])
        }
        if let numLayers, numLayers > 0 {
            args.append(contentsOf: ["--num-layers", "\(numLayers)"])
        }
        for intent in intentsKeep where !intent.isEmpty {
            args.append(contentsOf: ["--intent", intent])
        }
        for intent in intentsDrop where !intent.isEmpty {
            args.append(contentsOf: ["--drop-intent", intent])
        }
        if let sourceModelPath, !sourceModelPath.isEmpty {
            args.append(contentsOf: ["--source-model", sourceModelPath])
        }
        if let trainedTopK, trainedTopK > 0 {
            args.append(contentsOf: ["--trained-top-k", "\(trainedTopK)"])
        }
        if !suiteName.isEmpty {
            args.append(contentsOf: ["--suite-name", suiteName])
        }
        if suitePromptCount > 0 {
            args.append(contentsOf: ["--suite-prompt-count", "\(suitePromptCount)"])
        }
        if let crackPackPath, !crackPackPath.isEmpty {
            args.append(contentsOf: ["--crack-pack", crackPackPath])
        }
        if noDefaultCrackPack {
            args.append("--no-default-crack-pack")
        }
        return args
    }

    /// Hard-prune BF16/F16 Qwen MoE with an Intent keep-map.
    ///
    /// Reuses `prequant-prune-qwen-moe`. Does **not** pass
    /// `--require-reviewed-comparison` — intent plans carry hybrid safety +
    /// suite fingerprints, not Expert Lab A/B `comparison_summary` (see
    /// `test_prequant_prune_accepts_intent_plan_flat_keep_lists`).
    static func hardPruneArgs(
        sourcePath: String,
        outputPath: String,
        keepExperts: Int,
        prunePlanPath: String
    ) -> [String] {
        [
            "-m", "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            sourcePath,
            outputPath,
            "--keep-experts", "\(keepExperts)",
            "--json",
            "--keep-map", prunePlanPath,
        ]
    }

    /// Artifact folder basename: `{source}-intent-{slugs}[-drop-{drops}]-k{K}[-CRACK]`.
    static func artifactFolderName(
        sourceBaseName: String,
        chips: [IntentPruneChip],
        keepK: Int,
        safetyStance: IntentPruneSafetyStance,
        dropChips: [IntentPruneDropChip] = []
    ) -> String {
        let slugs = chips
            .filter { $0 != .generalist || chips.count == 1 }
            .map(\.artifactSlug)
            .sorted()
        let slugPart = slugs.isEmpty ? "general" : slugs.joined(separator: "-")
        var name = "\(sourceBaseName)-intent-\(slugPart)"
        let dropSlugs = dropChips
            .filter { !$0.switchesToCrackStance }
            .map(\.rawValue)
            .sorted()
        if !dropSlugs.isEmpty {
            name += "-drop-\(dropSlugs.joined(separator: "-"))"
        }
        name += "-k\(keepK)"
        if safetyStance.isCrack, !name.uppercased().hasSuffix("-CRACK") {
            name += "-CRACK"
        }
        return name
    }

    /// Expand selected chips into unique domain keys for `--intent`.
    static func domainKeys(for chips: Set<IntentPruneChip>) -> [String] {
        var seen = Set<String>()
        var keys: [String] = []
        for chip in IntentPruneChip.allCases where chips.contains(chip) {
            for key in chip.domainKeys where seen.insert(key).inserted {
                keys.append(key)
            }
            // Generalist alone still needs a token so the plan records intent.
            if chip == .generalist, chip.domainKeys.isEmpty, chips.count == 1 {
                if seen.insert("general").inserted {
                    keys.append("general")
                }
            }
        }
        return keys
    }

    /// Expand drop chips into unique domain keys for `--drop-intent`.
    static func dropDomainKeys(for chips: Set<IntentPruneDropChip>) -> [String] {
        var seen = Set<String>()
        var keys: [String] = []
        for chip in IntentPruneDropChip.allCases where chips.contains(chip) {
            for key in chip.domainKeys where seen.insert(key).inserted {
                keys.append(key)
            }
        }
        return keys
    }

    /// Whether Source architecture is Intent Prune v1 capable (Qwen MoE BF16/F16).
    static func supportsIntentPrune(_ detected: ArchitectureSummary) -> Bool {
        SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(detected)
    }
}
