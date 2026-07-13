// JANGStudio/Tests/JANGStudioTests/IntentPruneUITests.swift
// IP4: Intent Prune CTA presence, CRACK confirm gate, CLI argv builder.
import XCTest
@testable import JANGStudio

final class IntentPruneUITests: XCTestCase {

    // MARK: - Source inspection (CTA presence)

    func test_sourceStep_showsIntentPrunePrimaryCTA_forQwenMoE() throws {
        let source = try loadWizardSource("Steps/SourceStep.swift")

        XCTAssertTrue(source.contains("Shape model (Intent Prune)"))
        XCTAssertTrue(source.contains("showingIntentPrune"))
        XCTAssertTrue(source.contains("IntentPruneView("))
        XCTAssertTrue(source.contains("supportsIntentPrune"))
        XCTAssertTrue(source.contains("Advanced Expert Lab") || source.contains("DisclosureGroup(\"Advanced\")"))
        XCTAssertTrue(source.contains("Direct Convert") || source.contains("goDirectConvert"))
        // Expert Lab demoted to Advanced secondary, not removed.
        XCTAssertTrue(source.contains("Text(\"Expert Lab\").tag(WizardMode.expertLab)"))
        XCTAssertTrue(source.contains("enterExpertLabReview"))
        // Adopt path → Convert
        XCTAssertTrue(source.contains("adoptReviewedPrunedSource(url: prunedURL)"))
    }

    func test_intentPruneView_exposesCRACKConfirmAndRunControls() throws {
        let view = try loadWizardSource("IntentPruneView.swift")
        let vm = try loadWizardSource("IntentPruneViewModel.swift")
        let models = try loadWizardSource("IntentPruneModels.swift")

        XCTAssertTrue(view.contains("Run Intent Prune"))
        XCTAssertTrue(view.contains("Preview scores"))
        XCTAssertTrue(view.contains("Safety stance") || view.contains("safetyStance"))
        XCTAssertTrue(view.contains("CRACK"))
        XCTAssertTrue(view.contains("I understand CRACK is abliteration and want to proceed"))
        XCTAssertTrue(view.contains("Convert pruned model"))
        XCTAssertTrue(view.contains("Advanced Expert Lab"))

        XCTAssertTrue(vm.contains("crackConfirmed"))
        XCTAssertTrue(vm.contains("crackConfirmRequired") || vm.contains("CRACK abliteration requires"))
        XCTAssertTrue(vm.contains("canRun"))
        XCTAssertTrue(vm.contains("safetyStance.isCrack"))

        XCTAssertTrue(models.contains("case keep"))
        XCTAssertTrue(models.contains("case balanced"))
        XCTAssertTrue(models.contains("case crack"))
        XCTAssertTrue(models.contains("case light"))
        XCTAssertTrue(models.contains("case standard"))
        XCTAssertTrue(models.contains("case aggressive"))
    }

    func test_userGuide_documentsIntentPruneAndCRACK() throws {
        let guideURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("docs/USER_GUIDE.md")
        let guide = try String(contentsOf: guideURL, encoding: .utf8)

        XCTAssertTrue(guide.contains("Intent Prune"))
        XCTAssertTrue(guide.contains("Shape model (Intent Prune)"))
        XCTAssertTrue(guide.contains("CRACK"))
        XCTAssertTrue(guide.contains("abliteration") || guide.contains("Abliteration"))
        XCTAssertTrue(guide.contains("Advanced Expert Lab") || guide.contains("Expert Lab path (MoE, Advanced)"))
    }

    // MARK: - Budget / keep-K

    func test_budgetKeepK_clampsToTopKAndExperts() {
        // E=256, Standard ~0.75 → 192
        XCTAssertEqual(
            IntentPruneBudget.standard.keepK(expertsPerLayer: 256, trainedTopK: 8),
            192
        )
        XCTAssertEqual(
            IntentPruneBudget.light.keepK(expertsPerLayer: 256, trainedTopK: 8),
            230
        )
        XCTAssertEqual(
            IntentPruneBudget.aggressive.keepK(expertsPerLayer: 256, trainedTopK: 8),
            154
        )
        // Never below trained top-k
        XCTAssertEqual(
            IntentPruneBudget.aggressive.keepK(expertsPerLayer: 10, trainedTopK: 8),
            8
        )
        // Never above E
        XCTAssertEqual(
            IntentPruneBudget.light.keepK(expertsPerLayer: 8, trainedTopK: 1),
            7 // 0.9 * 8 = 7.2 → 7, still < E; floor 1
        )
        // Small E with high top-k
        XCTAssertEqual(
            IntentPruneBudget.light.keepK(expertsPerLayer: 8, trainedTopK: 8),
            8
        )
    }

    // MARK: - CRACK confirm required

    @MainActor
    func test_viewModel_crackConfirmRequired_blocksCanRun() {
        let detected = ArchitectureSummary(
            modelType: "qwen3_5_moe",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1_000_000_000,
            shardCount: 2,
            numHiddenLayers: 40,
            numExpertsPerTok: 8
        )
        let source = URL(fileURLWithPath: "/tmp/Fake-Qwen-MoE")
        let vm = IntentPruneViewModel(sourceURL: source, detected: detected)
        // Attach fake transitions so missing-transitions is not the blocker.
        vm.transitionsURL = URL(fileURLWithPath: "/tmp/expert_transitions.jsonl")
        vm.selectedChips = [.coding]
        vm.setSafetyStance(.crack)
        vm.crackConfirmed = false

        XCTAssertFalse(vm.canRun, "CRACK without confirm must block Run")
        XCTAssertNotNil(vm.crackGateMessage)
        XCTAssertFalse(vm.canPreviewScores)

        vm.crackConfirmed = true
        XCTAssertTrue(vm.canRun)
        XCTAssertTrue(vm.canPreviewScores)
        XCTAssertNil(vm.crackGateMessage)

        // Keep does not require confirm
        vm.setSafetyStance(.keep)
        XCTAssertFalse(vm.crackConfirmed)
        XCTAssertTrue(vm.canRun)
    }

    @MainActor
    func test_viewModel_requiresCapabilityChip() {
        let detected = ArchitectureSummary(
            modelType: "qwen3_5_moe_text",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1_000,
            shardCount: 1,
            numHiddenLayers: 40,
            numExpertsPerTok: 8
        )
        let vm = IntentPruneViewModel(
            sourceURL: URL(fileURLWithPath: "/tmp/model"),
            detected: detected
        )
        vm.transitionsURL = URL(fileURLWithPath: "/tmp/expert_transitions.jsonl")
        vm.selectedChips = []
        XCTAssertFalse(vm.canRun)
        vm.selectedChips = [.math]
        XCTAssertTrue(vm.canRun)
    }

    // MARK: - CLI argv builder

    func test_scoreArgs_includeStanceIntentsAndKeepK() {
        let args = IntentPruneCLIArgsBuilder.scoreArgs(
            transitionsPath: "/tmp/expert_transitions.jsonl",
            outputPlanPath: "/tmp/plan.json",
            numExperts: 256,
            numLayers: 40,
            keepK: 192,
            safetyStance: .crack,
            intentsKeep: ["code", "coding"],
            sourceModelPath: "/tmp/Qwen",
            trainedTopK: 8,
            suiteName: "Reviewed Prune 50",
            suitePromptCount: 50
        )

        XCTAssertTrue(args.contains("intent-prune-score"))
        XCTAssertTrue(args.contains("--safety-stance"))
        XCTAssertEqual(args[args.firstIndex(of: "--safety-stance")! + 1], "crack")
        XCTAssertTrue(args.contains("--keep-k"))
        XCTAssertEqual(args[args.firstIndex(of: "--keep-k")! + 1], "192")
        XCTAssertTrue(args.contains("--intent"))
        XCTAssertTrue(args.contains("code"))
        XCTAssertTrue(args.contains("--transitions"))
        XCTAssertTrue(args.contains("--suite-name"))
        XCTAssertTrue(args.contains("Reviewed Prune 50"))
        XCTAssertTrue(args.contains("--num-layers"))
        XCTAssertEqual(args[args.firstIndex(of: "--num-layers")! + 1], "40")
    }

    func test_hardPruneArgs_wireKeepMapAndJSON() {
        let args = IntentPruneCLIArgsBuilder.hardPruneArgs(
            sourcePath: "/tmp/src",
            outputPath: "/tmp/out",
            keepExperts: 192,
            prunePlanPath: "/tmp/plan.json"
        )
        XCTAssertTrue(args.contains("prequant-prune-qwen-moe"))
        XCTAssertTrue(args.contains("--keep-map"))
        XCTAssertEqual(args[args.firstIndex(of: "--keep-map")! + 1], "/tmp/plan.json")
        // Intent plans are not Expert Lab A/B plans — do not require comparison_summary.
        XCTAssertFalse(args.contains("--require-reviewed-comparison"))
        XCTAssertTrue(args.contains("--json"))
        XCTAssertTrue(args.contains("/tmp/src"))
        XCTAssertTrue(args.contains("/tmp/out"))
    }

    func test_artifactFolderName_appendsCRACK() {
        let base = IntentPruneCLIArgsBuilder.artifactFolderName(
            sourceBaseName: "Qwen3.6-35B-A3B",
            chips: [.coding, .math],
            keepK: 192,
            safetyStance: .keep
        )
        XCTAssertEqual(base, "Qwen3.6-35B-A3B-intent-coding-math-k192")
        XCTAssertFalse(base.contains("CRACK"))

        let crack = IntentPruneCLIArgsBuilder.artifactFolderName(
            sourceBaseName: "Qwen3.6-35B-A3B",
            chips: [.coding],
            keepK: 192,
            safetyStance: .crack
        )
        XCTAssertTrue(crack.hasSuffix("-CRACK"))
        XCTAssertTrue(crack.contains("intent-coding-k192"))
    }

    func test_domainKeys_fromChips() {
        let keys = IntentPruneCLIArgsBuilder.domainKeys(for: [.coding, .math])
        XCTAssertTrue(keys.contains("code") || keys.contains("coding"))
        XCTAssertTrue(keys.contains("math"))
        // no duplicates
        XCTAssertEqual(keys.count, Set(keys).count)
    }

    func test_supportsIntentPrune_matchesQwenRawMoE() {
        let qwen = ArchitectureSummary(
            modelType: "qwen3_5_moe",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1,
            shardCount: 1
        )
        let fp8 = ArchitectureSummary(
            modelType: "qwen3_5_moe",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .fp8,
            totalBytes: 1,
            shardCount: 1
        )
        let minimax = ArchitectureSummary(
            modelType: "minimax_m2",
            isMoE: true,
            numExperts: 64,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1,
            shardCount: 1
        )
        XCTAssertTrue(IntentPruneCLIArgsBuilder.supportsIntentPrune(qwen))
        XCTAssertFalse(IntentPruneCLIArgsBuilder.supportsIntentPrune(fp8))
        XCTAssertFalse(IntentPruneCLIArgsBuilder.supportsIntentPrune(minimax))
    }

    // MARK: - Helpers

    private func loadWizardSource(_ relative: String) throws -> String {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        return try String(
            contentsOf: wizardRoot.appendingPathComponent(relative),
            encoding: .utf8
        )
    }
}
