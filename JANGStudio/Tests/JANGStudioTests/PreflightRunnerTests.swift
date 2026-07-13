// JANGStudio/Tests/JANGStudioTests/PreflightRunnerTests.swift
import CryptoKit
import XCTest
@testable import JANGStudio

final class PreflightRunnerTests: XCTestCase {
    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    func test_missingSourceDirFails() {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/tmp/definitely_missing_xyz")
        plan.outputURL = tmp
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .sourceReadable && $0.status == .fail })
    }

    func test_jangtqOnLlamaFails() throws {
        let src = tmp.appendingPathComponent("src"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .jangtqArchSupported && $0.status == .fail })
    }

    func test_jangtqAcceptsFP16MoESource() throws {
        let src = tmp.appendingPathComponent("src"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 64, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .fp16, totalBytes: 0, shardCount: 1)
        plan.family = .jangtq
        plan.profile = "JANGTQ3"

        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)

        XCTAssertEqual(checks.first { $0.id == .jangtqArchSupported }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .jangtqSourceDtype }?.status, .pass)
    }

    func test_jangtqArchSupported_acceptsVLWrapperViaTextModelType() throws {
        // Outer modelType is not on the whitelist/module map; textModelType is.
        // Preflight must use architectureModelTypes (same as isJANGTQAllowed / CLIArgsBuilder).
        let src = tmp.appendingPathComponent("src-wrapper"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe_wrapper"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-wrapper")
        plan.detected = .init(modelType: "qwen3_5_moe_wrapper", isMoE: true, numExperts: 256, isVL: true,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1,
                              textModelType: "qwen3_5_moe_text")
        plan.family = .jangtq
        plan.profile = "JANGTQ3"

        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let arch = try XCTUnwrap(checks.first { $0.id == .jangtqArchSupported })
        XCTAssertEqual(arch.status, .pass,
                       "wrapper + textModelType qwen3_5_moe_text should pass arch check, hint=\(arch.hint ?? "nil")")
    }

    func test_jangtqArchSupported_failsUnknownModelTypeNilText() throws {
        let src = tmp.appendingPathComponent("src-unknown"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"some_other_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-unknown")
        plan.detected = .init(modelType: "some_other_moe", isMoE: true, numExperts: 64, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1,
                              textModelType: nil)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"

        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let arch = try XCTUnwrap(checks.first { $0.id == .jangtqArchSupported })
        XCTAssertEqual(arch.status, .fail)
        XCTAssertTrue(arch.hint?.contains("some_other_moe") == true,
                      "fail hint should mention detected type, got: \(arch.hint ?? "nil")")
    }

    func test_jangtqArchSupported_failsWhenWhitelistedButNoModuleMapping() throws {
        // Synthetic: put an unmapped type on the whitelist without a Studio module entry.
        // Preflight must still fail with the no-module mapping hint (whitelist ∩ module).
        let src = tmp.appendingPathComponent("src-nomodule"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"future_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-nomodule")
        plan.detected = .init(modelType: "future_moe", isMoE: true, numExperts: 64, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"

        let f = Capabilities.frozen
        let caps = Capabilities(
            jangtqWhitelist: f.jangtqWhitelist + ["future_moe"],
            knownExpert512Types: f.knownExpert512Types,
            supportedSourceDtypes: f.supportedSourceDtypes,
            blockSizes: f.blockSizes,
            defaultBlockSize: f.defaultBlockSize,
            methods: f.methods,
            defaultMethod: f.defaultMethod,
            tokenizerClassBlocklist: f.tokenizerClassBlocklist,
            hadamardDefaultForBitTier: f.hadamardDefaultForBitTier
        )
        XCTAssertTrue(plan.isJANGTQAllowed(for: caps.jangtqWhitelist),
                      "precondition: future_moe is on the synthetic whitelist")
        XCTAssertNil(CLIArgsBuilder.jangtqModule(for: plan),
                     "precondition: future_moe has no Studio module mapping")

        let checks = PreflightRunner().run(plan: plan, capabilities: caps)
        let arch = try XCTUnwrap(checks.first { $0.id == .jangtqArchSupported })
        XCTAssertEqual(arch.status, .fail,
                       "whitelisted but unmapped arch must fail preflight")
        XCTAssertTrue(arch.hint?.lowercased().contains("module") == true,
                      "hint should mention module mapping, got: \(arch.hint ?? "nil")")
    }

    func test_outputSameAsSourceFails() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src   // same!
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .outputUsable && $0.status == .fail })
    }

    func test_reviewedPrunePreflightFailsWhenVerificationMissing() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("verification.json") == true)
    }

    func test_reviewedPrunePreflightFailsWhenVerificationFailed() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try #"{"ok":false,"errors":["router rows mismatch"]}"#
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("router rows mismatch") == true)
    }

    func test_reviewedPrunePreflightFailsWhenRequiredVerificationCheckMissing() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try #"{"ok":true,"checks":{"router_rows_match":true}}"#
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("config_parses") == true)
    }

    func test_reviewedPrunePreflightFailsWhenVerificationCheckFailsDespiteOK() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":false}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("expert_rows_match") == true)
    }

    func test_reviewedPrunePreflightPassesOnlyForVerifiedPrunedSource() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .pass, reviewed.hint ?? "")
    }


    func test_intentPrunePlanPreflightPassesWithoutSemanticEvidenceRows() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validIntentPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .pass, reviewed.hint ?? "")
    }

    func test_intentPrunePlanPreflightAcceptsUnknownScorerField() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validIntentPrunePlanJSON(scorer: "hybrid_v2_experimental")
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .pass, reviewed.hint ?? "")
    }

    func test_intentPrunePlanPreflightRequiresCrackPackWhenStanceIsCrack() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validIntentPrunePlanJSON(safetyStance: "crack", includeCrackPack: false)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("crack_pack") == true, reviewed.hint ?? "")
    }

    func test_intentPrunePlanPreflightPassesWithCrackFingerprint() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validIntentPrunePlanJSON(safetyStance: "crack", includeCrackPack: true)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .pass, reviewed.hint ?? "")
    }

    func test_reviewedPrunePreflightRejectsComparisonSafeDropMaskMismatch() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_comparison_summary.json")) { json in
            json["safeDropCandidates"] = [["layer": 0, "expert": 0]]
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("safe-drop candidates do not match mask.json disabled experts") == true,
            reviewed.hint ?? ""
        )
    }

    func test_reviewedPrunePreflightRejectsPlanDropsOutsideSameSuiteSafeDrops() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try rewriteEmbeddedPlanDropExperts(planURL, experts: [0])

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("drops experts outside same-suite safe-drop candidates") == true,
            reviewed.hint ?? ""
        )
    }

    func test_reviewedPrunePreflightRejectsPrunePlanEvalIndexSidecarDrift() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try reorderEmbeddedPlanPromptIDs(planURL)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("embedded eval_index does not match expert_lab_eval_index.json") == true,
            reviewed.hint ?? ""
        )
        XCTAssertTrue(reviewed.hint?.contains("prompt IDs differ") == true, reviewed.hint ?? "")
    }

    func test_reviewedPrunePreflightRejectsEvalIndexWithoutDecodeSettingsEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json.removeValue(forKey: "generation_settings_checked")
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval_index.json is missing decode settings evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsEvalIndexSuiteFingerprintDrift() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let originalSuite = try String(contentsOf: suite, encoding: .utf8)
        try originalSuite
            .replacingOccurrences(
                of: "Return only the number: 17 * 23.",
                with: "Return only the number: 19 * 29."
            )
            .write(to: suite, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval_index.json suite.jsonl fingerprint does not match suite.jsonl") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSuiteFingerprintDrift() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let originalSuite = try String(contentsOf: suite, encoding: .utf8)
        try originalSuite
            .replacingOccurrences(
                of: "Return only the number: 17 * 23.",
                with: "Return only the number: 19 * 29."
            )
            .write(to: suite, atomically: true, encoding: .utf8)
        let updatedSuiteSHA256 = try Self.fileSHA256(suite)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json["suite_sha256"] = updatedSuiteSHA256
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source reviewed prompt suite fingerprint does not match reviewed suite") == true)
    }

    func test_reviewedPrunePreflightRejectsExternalReviewSidecarPaths() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let externalEvidence = try makeModelDir("external-evidence")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: externalEvidence)
        try FileManager.default.copyItem(
            at: externalEvidence.appendingPathComponent("expert_lab_review_summary.json"),
            to: pruned.appendingPathComponent("expert_lab_review_summary.json")
        )

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("review summary pruned source path does not match the selected pruned BF16/F16 source") == true
        )
    }

    func test_reviewedPrunePreflightRejectsReviewSummaryPrunedSourceMismatch() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let otherPruned = try makeModelDir("other-pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_review_summary.json")) { json in
            json["pruned_source"] = otherPruned.path
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("review summary pruned source path does not match the selected pruned BF16/F16 source") == true
        )
    }

    func test_reviewedPrunePreflightRejectsPrunedGenerationSummarySourceMismatch() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let otherPruned = try makeModelDir("other-pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")) { json in
            json["pruned_source"] = otherPruned.path
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("pruned-source generation summary path does not match the selected pruned BF16/F16 source") == true
        )
    }

    func test_reviewedPrunePreflightRejectsEvalIndexPromptOrderMismatch() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeEvalIndexPromptOrder(in: pruned, indices: Array((0..<50).reversed()))

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval_index.json prompt order does not match suite.jsonl") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSourceGenerationPromptOrderMismatch() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writePrunedGenerationPromptOrder(in: pruned, indices: [1, 0] + Array(2..<50))

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation prompt order does not match reviewed suite") == true)
    }

    func test_reviewedPrunePreflightRejectsExtraEvalIndexRowsBeyondComparisonSummary() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try expandReviewSidecarsTo51Prompts(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("eval_index.json covers 51 of 50 compared prompts") == true
                || reviewed.hint?.contains("embedded eval_index does not match expert_lab_eval_index.json") == true,
            reviewed.hint ?? ""
        )
    }

    func test_reviewedPrunePreflightRejectsComparisonSummaryDriftFromEvalRows() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try markFirstEvalRowHighRisk(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(
            reviewed.hint?.contains("comparison summary high-risk domains do not match eval.jsonl") == true,
            reviewed.hint ?? "missing hint"
        )
    }

    func test_reviewedPrunePreflightFailsWhenSuiteLacksRequiredSemanticProbes() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned, includeRequiredSemanticProbes: false)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("suite.jsonl is missing required semantic prompt probes") == true)
    }

    func test_reviewedPrunePreflightRequiresPlanSidecarInPrunedSource() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let externalPlanURL = tmp.appendingPathComponent("review-run-prune-plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: externalPlanURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: externalPlanURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("Reviewed prune plan is missing or unreadable") == true)
    }

    func test_reviewedPrunePreflightFailsWhenPlanLacksSemanticEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON(includeSemanticEvidence: false)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("activation lift evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWhenPlanPromptTagsAreEmpty() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON(includePromptTags: false)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("prompt tags/examples") == true)
    }

    func test_reviewedPrunePreflightFailsWhenPlanLacksMaskedImpactEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON(includeMaskedImpact: false)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("masked-output impact evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWhenPlanLacksMaskedImpactScope() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON(includeMaskedImpactScope: false)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("masked-output impact scope evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWithoutPrunedSourceSuiteVerification() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let summary = """
        {
          "schema": "jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready": true,
          "review_sidecars_ready": true,
          "pruned_suite_verification_ready": false,
          "pruned_suite_verification_issue": "pruned BF16/F16 same-suite vMLX verification has not run yet",
          "pruned_source": "\(pruned.path)",
          "prompt_count": 50,
          "suite_jsonl": "\(pruned.appendingPathComponent("expert_lab_suite.jsonl").path)",
          "comparison_summary": "\(pruned.appendingPathComponent("expert_lab_comparison_summary.json").path)",
          "eval_jsonl": "\(pruned.appendingPathComponent("expert_lab_eval.jsonl").path)",
          "eval_trace_jsonl": "\(pruned.appendingPathComponent("expert_lab_eval_trace.jsonl").path)",
          "eval_index": "\(pruned.appendingPathComponent("expert_lab_eval_index.json").path)"
        }
        """
        try summary.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned BF16/F16 same-suite vMLX verification has not run yet") == true)
    }

    func test_reviewedPrunePreflightRejectsExtraPrunedSourceGenerationRows() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let generations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let extra = #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"extra","text":"Unexpected prompt."},"result":{"text":"extra","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"}}}"#
        let updated = try String(contentsOf: generations, encoding: .utf8) + extra + "\n"
        try updated.write(to: generations, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation JSONL has 51 rows for 50 prompts") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSourceGenerationPromptIDDrift() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let generations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let drifted = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"other\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"}}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try drifted.write(to: generations, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation missing suite prompt IDs") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSourceGenerationRowsMissingDecodeSettings() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")) { json in
            json["generation_defaults"] = [
                "max_tokens": 96,
                "temperature": 0.0
            ]
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation row is missing decode settings evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSourceGenerationRowsMissingRuntimeEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let generations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let stripped = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try stripped.write(to: generations, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation is missing per-prompt runtime evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSourceGenerationRowsMissingLayerStatsWhenLayerCountKnown() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeKnownLayerHookEvidence(in: pruned, layerCount: 2)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation is missing per-prompt routed-layer stats") == true)
    }

    func test_reviewedPrunePreflightRejectsPrunedSourceGenerationRowsMissingTokenTraceWhenLayerCountKnown() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeKnownLayerHookEvidence(in: pruned, layerCount: 1)
        try writePrunedGenerationsWithLayerStats(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("pruned-source generation is missing per-prompt token_trace routing evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWhenPlanSafetyBelowTopK() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON(keepExperts: 4, trainedTopK: 8)
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("trained top-k") == true)
    }

    func test_reviewedPrunePreflightFailsWhenPlanLacksEmbeddedSameSuiteEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.safetyOnlyReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("embedded same-suite comparison evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexHasNoTokenDepthEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"native_jangtq_review_bundle","runtime_backend":"jangtq","runtime_device":"Unit Metal","runtime_metal_enabled":true}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("generation-depth token evidence") == true, reviewed.hint ?? "missing hint")
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexHasNoRuntimeEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json"}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("runtime device evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexHasNoPackageVersionEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("vMLX package version evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsTopKOnlyMaskEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"hooked_moe_layers":40,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":0,"top_k_override":4}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)
        try #"{"disabled_by_layer":{}}"#
            .write(to: pruned.appendingPathComponent("mask.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("top-k-only comparisons cannot authorize hard pruning") == true, reviewed.hint ?? "missing hint")
    }

    func test_reviewedPrunePreflightRejectsEvalRowsMissingRuntimeEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let evalRows = (0..<50)
            .map { #"{"promptID":"p\#($0)","baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"# }
            .joined(separator: "\n")
            .appending("\n")
        try evalRows.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval.jsonl is missing per-prompt runtime device evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsPartialVMLXHookCoverage() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let comparison = pruned.appendingPathComponent("expert_lab_comparison_summary.json")
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTrace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndex = pruned.appendingPathComponent("expert_lab_eval_index.json")
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"hooked_moe_layers":12,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: evalIndex, atomically: true, encoding: .utf8)
        try """
        {
          "schema": "jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready": true,
          "review_sidecars_ready": true,
          "review_sidecars_issue": null,
          "pruned_suite_verification_ready": true,
          "pruned_suite_verification_issue": null,
          "pruned_source": "\(pruned.path)",
          "pruned_suite_summary": "\(prunedSummary.path)",
          "pruned_suite_generations": "\(prunedGenerations.path)",
          "prompt_count": 50,
          "layer_count": 40,
          "source_model_path": "/tmp/jang-unit-bf16-source",
          "suite_jsonl": "\(suite.path)",
          "comparison_summary": "\(comparison.path)",
          "eval_jsonl": "\(eval.path)",
          "eval_trace_jsonl": "\(evalTrace.path)",
          "eval_index": "\(evalIndex.path)"
        }
        """
            .write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("vMLX hook coverage 12 of 40 routed layers") == true)
    }

    func test_reviewedPrunePreflightRejectsIncompleteVMLXHookCoverageFlag() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let comparison = pruned.appendingPathComponent("expert_lab_comparison_summary.json")
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTrace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndex = pruned.appendingPathComponent("expert_lab_eval_index.json")
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"hooked_moe_layers":40,"expected_moe_layers":40,"hook_coverage_complete":false,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: evalIndex, atomically: true, encoding: .utf8)
        try """
        {
          "schema": "jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready": true,
          "review_sidecars_ready": true,
          "review_sidecars_issue": null,
          "pruned_suite_verification_ready": true,
          "pruned_suite_verification_issue": null,
          "pruned_source": "\(pruned.path)",
          "pruned_suite_summary": "\(prunedSummary.path)",
          "pruned_suite_generations": "\(prunedGenerations.path)",
          "prompt_count": 50,
          "layer_count": 40,
          "source_model_path": "/tmp/jang-unit-bf16-source",
          "suite_jsonl": "\(suite.path)",
          "comparison_summary": "\(comparison.path)",
          "eval_jsonl": "\(eval.path)",
          "eval_trace_jsonl": "\(evalTrace.path)",
          "eval_index": "\(evalIndex.path)"
        }
        """
            .write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("incomplete vMLX routed-layer hook coverage") == true)
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexHasNoRoutingEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("routing record evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsPartialRouteCoverage() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":49,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("routing record evidence for every indexed prompt") == true)
    }

    func test_reviewedPrunePreflightRejectsEvalTraceRouteCountMismatch() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json["baseline_route_record_count"] = 51
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval_trace.jsonl has 50 baseline routing records for 51 indexed baseline route records") == true)
    }

    func test_reviewedPrunePreflightRejectsPartialEvalIndexLayerStatsCoverage() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json["baseline_layer_stats_prompt_count"] = 50
            json["masked_layer_stats_prompt_count"] = 49
        }

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval_index.json layer-stat coverage is incomplete for indexed prompts") == true)
    }

    func test_reviewedPrunePreflightRejectsPartialEvalRowLayerStatsEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeEvalRowsWithPartialLayerStats(in: pruned)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval.jsonl layer-stat evidence is incomplete for baseline/masked prompts") == true)
    }

    func test_reviewedPrunePreflightFailsWhenSameSuiteComparisonHasHighRiskDomains() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(
            in: pruned,
            comparisonJSON: #"{"promptCount":50,"passRateBaseline":1.0,"passRateMasked":1.0,"meanTextDelta":0.7,"highRiskDomains":["math"],"safeDropCandidates":[]}"#
        )

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("high-risk domains") == true)
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexStillHasRiskyPrompts() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":["p13"],"high_risk_domains":["math"],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"native_jangtq_review_bundle","runtime_backend":"jangtq","runtime_device":"Unit Metal","runtime_metal_enabled":true}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("risky prompt IDs") == true, reviewed.hint ?? "missing hint")
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexDoesNotListComparedPrompts() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try #"{"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":["p0"],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"eval_jsonl":"expert_lab_eval.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"native_jangtq_review_bundle","runtime_backend":"jangtq","runtime_device":"Unit Metal","runtime_metal_enabled":true}"#
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("lists 1 prompt IDs for 50 indexed prompts") == true)
    }

    func test_reviewedPrunePreflightFailsWhenEvalIndexPromptIDsDoNotMatchEvalRows() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let mismatchedEvalText = (0..<50)
            .map { #"{"promptID":"q\#($0)","baselineText":"hello","maskedText":"hello"}"# }
            .joined(separator: "\n")
            .appending("\n")
        try mismatchedEvalText.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("missing from eval.jsonl") == true)
    }

    func test_reviewedPrunePreflightRejectsEvalRowsOutsideIndexedSuite() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let extra = #"{"promptID":"outside-suite","baselineText":"hello","maskedText":"hello","baselineRouteRecordCount":1,"maskedRouteRecordCount":1}"#
        try (String(contentsOf: eval, encoding: .utf8) + extra + "\n")
            .write(to: eval, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval.jsonl prompt IDs outside eval_index.json") == true)
    }

    func test_reviewedPrunePreflightRejectsTraceRowsOutsideIndexedSuite() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let trace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let extra = #"{"promptID":"outside-suite","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#
        try (String(contentsOf: trace, encoding: .utf8) + extra + "\n")
            .write(to: trace, atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("eval_trace.jsonl prompt IDs outside eval_index.json") == true)
    }

    func test_reviewedPrunePreflightRejectsTraceMissingMaskedVariant() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let baselineOnlyTrace = (0..<50)
            .map { #"{"promptID":"p\#($0)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try baselineOnlyTrace.write(to: pruned.appendingPathComponent("expert_lab_eval_trace.jsonl"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("missing masked routing records") == true)
    }

    func test_reviewedPrunePreflightRejectsMaskedTraceWithoutMaskEvidence() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let traceWithoutMaskedDisabledExperts = (0..<50)
            .flatMap { index in
                [
                    #"{"promptID":"p\#(index)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#,
                    #"{"promptID":"p\#(index)","domain":"general","variant":"masked","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#
                ]
            }
            .joined(separator: "\n")
            .appending("\n")
        try traceWithoutMaskedDisabledExperts.write(to: pruned.appendingPathComponent("expert_lab_eval_trace.jsonl"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("masked routing records are missing mask evidence") == true)
    }

    func test_reviewedPrunePreflightRejectsMaskedTraceSelectingDisabledExperts() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let traceSelectingDisabledExperts = (0..<50)
            .flatMap { index in
                [
                    #"{"promptID":"p\#(index)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#,
                    #"{"promptID":"p\#(index)","domain":"general","variant":"masked","record":{"tokenIndex":0,"layer":0,"selectedExperts":[1],"scores":[1.0],"disabledExperts":[1],"effectiveTopK":1}}"#
                ]
            }
            .joined(separator: "\n")
            .appending("\n")
        try traceSelectingDisabledExperts.write(to: pruned.appendingPathComponent("expert_lab_eval_trace.jsonl"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("selected disabled experts") == true)
    }

    func test_reviewedPrunePreflightRejectsMaskedTraceThatDoesNotMatchMaskJSON() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let traceWithWrongDisabledExpert = (0..<50)
            .flatMap { index in
                [
                    #"{"promptID":"p\#(index)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#,
                    #"{"promptID":"p\#(index)","domain":"general","variant":"masked","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[2],"effectiveTopK":1}}"#
                ]
            }
            .joined(separator: "\n")
            .appending("\n")
        try traceWithWrongDisabledExpert.write(to: pruned.appendingPathComponent("expert_lab_eval_trace.jsonl"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("mask.json evidence") == true)
    }

    func test_reviewedPrunePreflightFailsWhenSameSuiteComparisonIsUnderCovered() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try #"{"version":1}"#.write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(
            in: pruned,
            comparisonJSON: #"{"promptCount":1,"passRateBaseline":1.0,"passRateMasked":1.0,"meanTextDelta":0.0,"highRiskDomains":[],"safeDropCandidates":[{"layer":0,"expert":1}]}"#
        )

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("compare at least 50 prompts") == true)
    }

    func test_reviewedPrunePreflightFallsBackToPrunedSourcePlanSidecar() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        try Self.validReviewedPrunePlanJSON().write(
            to: pruned.appendingPathComponent("prune_plan.json"),
            atomically: true,
            encoding: .utf8
        )
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = reviewedPrunePlan(
            original: original,
            pruned: pruned,
            prunePlan: pruned.appendingPathComponent("missing-original-run-plan.json")
        )
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .pass)
    }

    func test_reviewedPrunePreflightFailsWithoutSameSuiteReviewSidecars() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try #"{"version":1}"#.write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("expert_lab_review_summary.json") == true)
    }

    func test_reviewedPrunePreflightFailsIfSourceNoLongerMatchesPrunedSource() throws {
        let original = try makeModelDir("original")
        let pruned = try makeModelDir("pruned")
        let other = try makeModelDir("other")
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try #"{"version":1}"#.write(to: planURL, atomically: true, encoding: .utf8)
        try #"{"ok":true}"#.write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)

        let plan = reviewedPrunePlan(original: original, pruned: pruned, prunePlan: planURL)
        plan.sourceURL = other
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let reviewed = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerified })
        XCTAssertEqual(reviewed.status, .fail)
        XCTAssertTrue(reviewed.hint?.contains("verified pruned BF16/F16") == true)
    }

    // MARK: - M139 (iter 61): reject nested src/dst
    //
    // Pre-iter-61 outputUsable only rejected `dst == src`. A user picking
    // output INSIDE the source tree (e.g., src=/models/foo, dst=/models/foo/out)
    // passed preflight, let convert write shards into a subdir of the source.
    // Confusing + risky if any future cleanup pass rglobs.

    func test_outputInsideSourceFails() throws {
        let src = tmp.appendingPathComponent("foo")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src.appendingPathComponent("out")   // nested inside src
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let failure = checks.first { $0.id == .outputUsable && $0.status == .fail }
        XCTAssertNotNil(failure, "outputUsable should fail when output is inside source")
        XCTAssertEqual(failure?.hint, "Output cannot be inside the source folder")
    }

    func test_sourceInsideOutputFails() throws {
        // Symmetric: user picks output as the parent of source.
        let parent = tmp.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let src = parent.appendingPathComponent("hf-model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = parent   // source nested inside output
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let failure = checks.first { $0.id == .outputUsable && $0.status == .fail }
        XCTAssertNotNil(failure, "outputUsable should fail when source is inside output")
        XCTAssertEqual(failure?.hint, "Source cannot be inside the output folder")
    }

    func test_siblingPrefixPathsDoNotTrigger() throws {
        // Regression: `/a/b` must NOT be rejected as "inside /a/bc". The
        // check uses path + "/" specifically to prevent this.
        let srcParent = tmp.appendingPathComponent("abc")
        try FileManager.default.createDirectory(at: srcParent, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: srcParent.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = srcParent
        plan.outputURL = tmp.appendingPathComponent("abcd")   // sibling with shared prefix
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let found = checks.first { $0.id == .outputUsable }
        XCTAssertEqual(found?.status, .pass,
            "Sibling directories with a shared string prefix (abc vs abcd) must not trigger the nested-path check. Got \(found?.hint ?? "nil")")
    }

    // MARK: - M140 (iter 62): symmetric to M131 — preflight must detect
    // 512+ expert MoE dynamically, not just via a hardcoded name list.
    //
    // Pre-iter-62 check passed on ANY model_type not in the capabilities
    // `knownExpert512Types` hardcoded list. A future 512-expert
    // qwen3_5_moe or deepseek_v3 variant would skip the warning while the
    // underlying recommend.py-side already forces bfloat16 for the same
    // dynamic reason. Symmetric bug across the boundary; fix the preflight
    // to match.

    func test_bf16Warning_fires_on_dynamic_512_experts() throws {
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        // 512-expert qwen3_5_moe — NOT in the default .frozen whitelist
        // `knownExpert512Types: ["minimax_m2", "glm_moe_dsa"]`, but the
        // num_experts count should trigger the dynamic check.
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 512, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16   // user forced fp16 — should warn
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .warn,
            "512-expert qwen3_5_moe + forced fp16 must warn about bfloat16. Got status=\(String(describing: bf?.status)).")
        XCTAssertNotNil(bf?.hint)
        XCTAssertTrue(bf?.hint?.contains("512 experts") ?? false,
            "Hint must include the dynamic expert count so the user understands what triggered the warning. Got hint=\(bf?.hint ?? "nil")")
    }

    func test_bf16Warning_still_fires_for_named_whitelist_types() throws {
        // Regression guard: the named-family path must keep working after
        // iter-62's dynamic extension. minimax_m2 is in the frozen
        // whitelist — should warn on fp16 override.
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"minimax_m2"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        // Named whitelist entry, but with numExperts unknown (0).
        plan.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 0, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .fp8, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .warn)
    }

    func test_bf16Warning_skips_small_moe() throws {
        // Regression guard: 256-expert qwen3_5_moe must stay passing — no
        // over-warn on smaller MoEs.
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .pass,
            "256-expert qwen3_5_moe must not fire the 512+ warning.")
    }

    // MARK: - M141 (iter 63): diskSpace preflight actually gates
    //
    // Pre-iter-63 PreflightRunner.run always passed `estimated: 0` to the
    // diskSpace check, which short-circuited to `.pass` unconditionally.
    // User with near-full disk got no warning — convert started, filled
    // the disk, crashed mid-shard. Iter 63 wires profile-aware estimation
    // via `estimateOutputBytes(plan:, profiles:)` using the same formula
    // as jang_tools/estimate_model.predict.

    func test_estimateOutputBytes_scales_by_profile_avgBits() {
        // 100 GB bf16 source × (4/16) × 1.05 = 26.25 GB for JANG_4K.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        // 100 GB * 4/16 * 1.05 = 26.25 GB
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000,
                       "JANG_4K on 100 GB bf16 source should predict ~26 GB output")
    }

    func test_estimateOutputBytes_uses_real_avgBits_for_JANG_2L() {
        // JANG_2L is 2.9 bits/weight avg → 100 GB × 2.9/16 × 1.05 = 19.03 GB
        let plan = ConversionPlan()
        plan.profile = "JANG_2L"
        plan.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertGreaterThan(est, 15_000_000_000)
        XCTAssertLessThan(est, 25_000_000_000)
    }

    func test_estimateOutputBytes_returns_zero_before_source_inspected() {
        // Regression: pre-inspection state (detected=nil) must return 0 so
        // the disk-space check falls back to its `estimated <= 0` .pass
        // short-circuit — we can't gate until we know the source size.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        // detected stays nil
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 0,
            "Without a detected source size, the estimator must return 0 so preflight doesn't falsely fail.")
    }

    // MARK: - Iter 102 M175: ramAdequate sibling of M05 (ambiguous-pass sweep)

    func test_ramAdequate_pre_inspection_warns_about_uncheckable_state() {
        // Pre-M175 this returned .pass with nil hint when totalBytes was
        // unknown — user saw ✓ but the check hadn't actually evaluated.
        // OOM mid-convert is arguably worse than disk-full since the OS
        // may kill the subprocess before it can surface a clean error.
        let src = tmp.appendingPathComponent("src-ram-uninspected")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-ram-uninspected")
        plan.profile = "JANG_4K"
        // detected stays nil → totalBytes unknown
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        guard let ram = checks.first(where: { $0.id == .ramAdequate }) else {
            XCTFail("no ramAdequate check in preflight output")
            return
        }
        XCTAssertEqual(ram.status, .warn,
            "pre-inspection ramAdequate must warn, not silently pass (M175)")
        XCTAssertTrue(ram.hint?.contains("no estimate") == true,
            "warn hint must explain the uncheckable state. Got: \(ram.hint ?? "nil")")
    }

    func test_ramAdequate_post_inspection_with_room_still_passes() {
        // Regression guard for the real-check branch.
        let src = tmp.appendingPathComponent("src-ram-ok")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-ram-ok")
        plan.profile = "JANG_4K"
        // 10 MB source → needed ~15 MB → any Mac has this much RAM
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 10_000_000, shardCount: 1)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        let ram = checks.first(where: { $0.id == .ramAdequate })
        XCTAssertEqual(ram?.status, .pass,
            "post-inspection with ample RAM must still pass (regression guard)")
    }

    // MARK: - Iter 101 M05: diskSpace disambiguates "no estimate" from "enough room"

    func test_diskSpace_pre_inspection_warns_about_uncheckable_state() {
        // Plan with output set but no detected source → estimator returns 0.
        // Pre-M05 this produced `.pass` which visually matched a real passing
        // check. M05 changes to `.warn` with "(no estimate yet)" hint so the
        // user knows the system didn't actually verify sufficient space.
        let src = tmp.appendingPathComponent("src-uninspected")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-uninspected")
        plan.profile = "JANG_4K"
        // detected stays nil → totalBytes unknown → estimator returns 0
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        guard let disk = checks.first(where: { $0.id == .diskSpace }) else {
            XCTFail("no diskSpace check in preflight output")
            return
        }
        XCTAssertEqual(disk.status, .warn,
            "pre-inspection diskSpace must warn, not silently pass — user needs to see 'no estimate' state")
        XCTAssertTrue(disk.hint?.contains("no estimate") == true,
            "warn hint must explain WHY the check is uncheckable. Got: \(disk.hint ?? "nil")")
    }

    func test_diskSpace_post_inspection_with_room_still_passes() {
        // Regression guard: with a real detected source size + sufficient free
        // space, the check must still produce `.pass` (M05 only changed the
        // uncheckable branch).
        let src = tmp.appendingPathComponent("src-with-shards")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-with-shards")
        plan.profile = "JANG_4K"
        // Tiny model — estimate will be comfortably under free space
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 10_000_000, shardCount: 1)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        let disk = checks.first(where: { $0.id == .diskSpace })
        XCTAssertEqual(disk?.status, .pass,
            "with a real estimate + enough free space, diskSpace must still pass (regression guard)")
    }

    // MARK: - Iter 99 M173: estimate honors source dtype (FP8 / BF16 / FP32)

    func test_estimateOutputBytes_fp8_source_uses_8bit_divisor() {
        // For FP8 source, src_bytes = weights × 1 (not × 2). Output at 4 bits
        // avg = weights × 0.5. So the formula needs src_bytes × (4/8) × 1.05
        // = 0.525 × src_bytes, NOT the BF16-assuming 0.26 × src_bytes. Pre-
        // M173 the user got a prediction half the real need → "plenty of
        // disk" green check → convert fails mid-way on disk-full.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "deepseek_v3", isMoE: true, numExperts: 256,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .fp8, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        // 100 GB FP8 × 4/8 × 1.05 = 52.5 GB
        XCTAssertEqual(est, 52_500_000_000, accuracy: 1_000_000_000,
            "FP8 source must use /8 divisor — pre-M173 this returned ~26 GB (half the real need)")
    }

    func test_estimateOutputBytes_bf16_source_matches_pre_M173_behavior() {
        // Regression: BF16 source (the case the original formula was written
        // for) must still produce the same answer post-M173.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000)
    }

    func test_estimateOutputBytes_fp16_source_same_as_bf16() {
        // FP16 is also 2 bytes/weight — same /16 divisor as BF16.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .fp16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000)
    }

    func test_estimateOutputBytes_unknown_dtype_falls_back_to_16bit_assumption() {
        // Safety default: if source dtype is unknown (older models, detection
        // drift), assume 16-bit — the common case historically. Conservative:
        // over-estimate is better than under-estimate for the disk-space gate
        // (better to refuse a safe convert than to permit a disk-full crash).
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .unknown, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000,
            "Unknown dtype must fall back to /16 (BF16/FP16 assumption) for safety")
    }

    func test_estimateOutputBytes_returns_zero_for_unknown_profile() {
        // Regression: unknown profile must also produce 0 rather than
        // guessing a bits value — prevents false positives from typos.
        let plan = ConversionPlan()
        plan.profile = "JANG_UNKNOWN_99X"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 0,
            "Unknown profile must return 0 (caller falls back to pass). Don't guess a bit-width.")
    }

    // MARK: - M142 (iter 64): hadamardVsLowBits uses compress bits, not substring
    //
    // Pre-iter-64 the check was:
    //   let is2bit = plan.profile.contains("_2")
    //              || plan.profile == "JANG_1L"
    //              || plan.profile == "JANGTQ2"
    // Brittle: a future "JANG_20" (20-bit) would trip as 2-bit; a future
    // "JANG_0S" (<1-bit) wouldn't be flagged. Fix: look up compressBits
    // from ProfilesService — single source of truth with Python-side
    // allocate.JANG_PROFILES.

    func test_compressBitsForProfile_JANG_2L() {
        // JANG_2L is (8, 6, 2) → compress=2.
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_2L", profiles: .frozen), 2)
    }

    func test_compressBitsForProfile_JANG_1L() {
        // JANG_1L is (8, 8, 2) → compress=2 (low-bit flagged).
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_1L", profiles: .frozen), 2)
    }

    func test_compressBitsForProfile_JANG_4M() {
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_4M", profiles: .frozen), 4)
    }

    func test_compressBitsForProfile_K_quant() {
        // JANG_4K has compressBits=nil in the schema; derive from avgBits=4.0.
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_4K", profiles: .frozen), 4)
    }

    func test_compressBitsForProfile_JANGTQ2() {
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANGTQ2", profiles: .frozen), 2)
    }

    func test_compressBitsForProfile_unknown_returns_nil() {
        // Iter-54 M132 fix: typo defense. Unknown profile must return nil
        // so the caller can distinguish "unknown" from "known and high-bit".
        XCTAssertNil(PreflightRunner.compressBitsForProfile("JANG_BOGUS_99Y", profiles: .frozen))
    }

    func test_hadamardAt2bitWarns() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.profile = "JANG_2S"
        plan.hadamard = true
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .hadamardVsLowBits && $0.status == .warn })
    }

    private func makeModelDir(_ name: String) throws -> URL {
        let url = tmp.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: url.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        return url
    }

    private func writeKnownLayerHookEvidence(in pruned: URL, layerCount: Int) throws {
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_review_summary.json")) { json in
            json["layer_count"] = layerCount
        }
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json["hooked_moe_layers"] = layerCount
            json["expected_moe_layers"] = layerCount
            json["hook_coverage_complete"] = true
        }
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")) { json in
            json["hooked_moe_layers"] = layerCount
            json["expected_moe_layers"] = layerCount
            json["hook_coverage_complete"] = true
        }
    }

    private func writePrunedGenerationsWithLayerStats(in pruned: URL) throws {
        let text = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"},"layer_stats":[{"layer":0,"token_count":1,"hit_counts":{"0":1},"probability_mass":{"0":1.0}}]}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try text.write(to: pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl"), atomically: true, encoding: .utf8)
    }

    private func updateJSONFile(_ url: URL, update: (inout [String: Any]) -> Void) throws {
        let data = try Data(contentsOf: url)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        update(&json)
        let updated = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try updated.write(to: url)
    }

    private func reorderEmbeddedPlanPromptIDs(_ planURL: URL) throws {
        let data = try Data(contentsOf: planURL)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var evalIndex = try XCTUnwrap(json["eval_index"] as? [String: Any])
        var promptIDs = try XCTUnwrap(evalIndex["prompt_ids"] as? [String])
        guard promptIDs.count >= 2 else {
            XCTFail("expected at least two prompt IDs in embedded eval_index")
            return
        }
        promptIDs.swapAt(0, 1)
        evalIndex["prompt_ids"] = promptIDs
        json["eval_index"] = evalIndex
        let updated = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try updated.write(to: planURL)
    }

    private func rewriteEmbeddedPlanDropExperts(_ planURL: URL, experts: [Int]) throws {
        let data = try Data(contentsOf: planURL)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var layers = try XCTUnwrap(json["layers"] as? [String: Any])
        var layer = try XCTUnwrap(layers["0"] as? [String: Any])
        layer["drop"] = experts
        layers["0"] = layer
        json["layers"] = layers
        let updated = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try updated.write(to: planURL)
    }

    private func markFirstEvalRowHighRisk(in pruned: URL) throws {
        let url = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        var lines = try String(contentsOf: url, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map(String.init)
        var first = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(lines[0].utf8)) as? [String: Any]
        )
        first["domain"] = "domain"
        first["semanticDomains"] = ["safety_medical_legal_sensitive"]
        first["risk"] = "regression"
        first["regressionSeverity"] = "high"
        first["baselinePassed"] = true
        first["maskedPassed"] = false
        first["baselineQualified"] = true
        first["promptClassification"] = "degraded"
        first["safeDropEvidenceEligible"] = false
        first["textDelta"] = 0.9
        let data = try JSONSerialization.data(withJSONObject: first, options: [.sortedKeys])
        lines[0] = String(data: data, encoding: .utf8) ?? "{}"
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private func writeExpertLabReviewSidecars(
        in pruned: URL,
        comparisonJSON: String? = nil,
        includeRequiredSemanticProbes: Bool = true
    ) throws {
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let comparison = pruned.appendingPathComponent("expert_lab_comparison_summary.json")
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTrace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndex = pruned.appendingPathComponent("expert_lab_eval_index.json")
        let mask = pruned.appendingPathComponent("mask.json")
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        let suiteText = (0..<50)
            .map { index -> String in
                switch index {
                case 0 where includeRequiredSemanticProbes:
                    return #"{"id":"p0","domain":"math","subdomain":"arithmetic","text":"Return only the number: 17 * 23.","tags":["arithmetic"]}"#
                case 1 where includeRequiredSemanticProbes:
                    return #"{"id":"p1","domain":"coding","subdomain":"swift","text":"Write a Swift function that groups strings by first character.","tags":["swift","code"]}"#
                case 2 where includeRequiredSemanticProbes:
                    return #"{"id":"p2","domain":"structured","subdomain":"json","text":"Return JSON fields decision and rationale for this status.","tags":["json","formatting"]}"#
                case 3 where includeRequiredSemanticProbes:
                    return #"{"id":"p3","domain":"instruction","subdomain":"hierarchy","text":"Explain how to respond when a user asks you to ignore a system rule.","tags":["instruction-following"]}"#
                case 4 where includeRequiredSemanticProbes:
                    return #"{"id":"p4","domain":"reasoning","subdomain":"logic","text":"If A implies B and B is false, what follows about A?","tags":["logic"]}"#
                case 5 where includeRequiredSemanticProbes:
                    return #"{"id":"p5","domain":"robustness","subdomain":"medicine-safety","text":"Explain safe next steps for incomplete symptoms without diagnosing.","tags":["safety","medical","legal"]}"#
                case 6 where includeRequiredSemanticProbes:
                    return #"{"id":"p6","domain":"multilingual","subdomain":"chinese","text":"Translate this status update into Simplified Chinese: Build succeeded.","tags":["chinese","translation","non_english"]}"#
                case 7 where includeRequiredSemanticProbes:
                    return #"{"id":"p7","domain":"general","subdomain":"explanation","text":"Classify this prompt as English dominant.","tags":["english_dominant"]}"#
                case 8 where includeRequiredSemanticProbes:
                    return #"{"id":"p8","domain":"multilingual","subdomain":"unknown-language-role","text":"Classify whether this text is mixed or unknown language role: Bonjour, build succeeded.","tags":["unknown_language_role","non_english"]}"#
                default:
                    return #"{"id":"p\#(index)","domain":"general","text":"Say hello."}"#
                }
            }
            .joined(separator: "\n")
            .appending("\n")
        try suiteText.write(to: suite, atomically: true, encoding: .utf8)
        try (comparisonJSON ?? Self.validComparisonSummaryJSON())
            .write(to: comparison, atomically: true, encoding: .utf8)
        let suiteSHA256 = try Self.fileSHA256(suite)
        let evalText = (0..<50)
            .map { index in
                #"{"promptID":"p\#(index)","domain":"general","semanticDomains":[\#(Self.semanticDomainsJSON(index: index))],"expectedKind":"exact","expected":"hello","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"#
            }
            .joined(separator: "\n")
            .appending("\n")
        try evalText.write(to: eval, atomically: true, encoding: .utf8)
        let evalTraceText = (0..<50)
            .flatMap { index in
                [
                    #"{"promptID":"p\#(index)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#,
                    #"{"promptID":"p\#(index)","domain":"general","variant":"masked","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[1],"effectiveTopK":1}}"#
                ]
            }
            .joined(separator: "\n")
            .appending("\n")
        try evalTraceText.write(to: evalTrace, atomically: true, encoding: .utf8)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_semantic_coverage":[],"validator_schema":"jang-expert-lab-validator-v1","validator_available_prompt_count":50,"prompt_classification_counts":{\(Self.classificationCountsJSON(count: 50))},"baseline_qualified_prompt_count":50,"baseline_qualified_prompt_ids":[\(promptIDs)],"baseline_invalid_prompt_ids":[],"inconclusive_prompt_ids":[],"preserved_prompt_ids":[\(promptIDs)],"degraded_prompt_ids":[],"baseline_qualified_masked_pass_rate":1.0,"baseline_qualified_semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_baseline_qualified_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"generation_settings_checked":true,"suite_sha256":"\(suiteSHA256)","eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: evalIndex, atomically: true, encoding: .utf8)
        try #"{"disabled_by_layer":{"0":[1]}}"#
            .write(to: mask, atomically: true, encoding: .utf8)
        let prunedGenerationText = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"}}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try prunedGenerationText.write(to: prunedGenerations, atomically: true, encoding: .utf8)
        try """
        {
          "schema": "jang-expert-lab-pruned-bf16-suite-v1",
          "ready": true,
          "pruned_source": "\(pruned.path)",
          "suite_sha256": "\(suiteSHA256)",
          "prompt_count": 50,
          "generation_count": 50,
          "runtime_mode": "bf16_vmlx",
          "runtime_backend": "vmlx",
          "runtime_device": "Unit Metal",
          "runtime_metal_enabled": true,
          "jang_tools_version": "2.5.31",
          "mlx_version": "0.31.2",
          "mlx_lm_version": "0.31.3",
          "runtime_source_model_path": "\(pruned.path)",
          "reviewed_masked_comparison_count": 50,
          "reviewed_masked_mean_text_delta": 0.0,
          "reviewed_masked_max_text_delta": 0.0,
          "pruned_validator_outcomes_checked": true,
          "baseline_qualified_prompt_count": 50,
          "pruned_baseline_qualified_pass_rate": 1.0,
          "pruned_classification_counts": {\(Self.classificationCountsJSON(count: 50))},
          "baseline_invalid_prompt_ids": [],
          "inconclusive_prompt_ids": [],
          "pruned_preserved_prompt_ids": [\(promptIDs)],
          "pruned_degraded_prompt_ids": [],
          "baseline_qualified_semantic_coverage": [\(Self.requiredSemanticCoverageJSON())],
          "missing_baseline_qualified_semantic_coverage": [],
          "reviewed_masked_eval_trace_jsonl": "\(evalTrace.path)",
          "generations_jsonl": "\(prunedGenerations.path)"
        }
        """
            .write(to: prunedSummary, atomically: true, encoding: .utf8)
        let summary = """
        {
          "schema": "jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready": true,
          "review_sidecars_ready": true,
          "review_sidecars_issue": null,
          "pruned_suite_verification_ready": true,
          "pruned_suite_verification_issue": null,
          "pruned_source": "\(pruned.path)",
          "pruned_suite_summary": "\(prunedSummary.path)",
          "pruned_suite_generations": "\(prunedGenerations.path)",
          "prompt_count": 50,
          "source_model_path": "/tmp/jang-unit-bf16-source",
          "suite_jsonl": "\(suite.path)",
          "comparison_summary": "\(comparison.path)",
          "eval_jsonl": "\(eval.path)",
          "eval_trace_jsonl": "\(evalTrace.path)",
          "eval_index": "\(evalIndex.path)",
          "mask_json": "\(mask.path)",
          "mask": "\(mask.path)"
        }
        """
        try summary.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
    }

    private func writeEvalRowsWithPartialLayerStats(in pruned: URL) throws {
        let evalText = (0..<50)
            .map { index -> String in
                let layerStats = index == 0
                    ? #","baselineLayerStats":[{"layer":0}],"maskedLayerStats":[{"layer":0}]"#
                    : ""
                return #"{"promptID":"p\#(index)","domain":"general","semanticDomains":[\#(Self.semanticDomainsJSON(index: index))],"expectedKind":"exact","expected":"hello","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"\#(layerStats)}"#
            }
            .joined(separator: "\n")
            .appending("\n")
        try evalText.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)
    }

    private func expandReviewSidecarsTo51Prompts(in pruned: URL) throws {
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTrace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndex = pruned.appendingPathComponent("expert_lab_eval_index.json")
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")

        try appendLine(#"{"id":"p50","domain":"general","text":"Say one more hello."}"#, to: suite)
        try appendLine(#"{"promptID":"p50","domain":"general","semanticDomains":["general"],"expectedKind":"exact","expected":"hello","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"#, to: eval)
        try appendLine(#"{"promptID":"p50","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#, to: evalTrace)
        try appendLine(#"{"promptID":"p50","domain":"general","variant":"masked","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[1],"effectiveTopK":1}}"#, to: evalTrace)
        try appendLine(#"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p50","text":"Say one more hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"}}}"#, to: prunedGenerations)

        let promptIDs = Self.promptIDJSON(count: 51)
        let suiteSHA256 = try Self.fileSHA256(suite)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":51,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_semantic_coverage":[],"validator_schema":"jang-expert-lab-validator-v1","validator_available_prompt_count":51,"prompt_classification_counts":{\(Self.classificationCountsJSON(count: 51))},"baseline_qualified_prompt_count":51,"baseline_qualified_prompt_ids":[\(promptIDs)],"baseline_invalid_prompt_ids":[],"inconclusive_prompt_ids":[],"preserved_prompt_ids":[\(promptIDs)],"degraded_prompt_ids":[],"baseline_qualified_masked_pass_rate":1.0,"baseline_qualified_semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_baseline_qualified_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":51,"masked_route_record_count":51,"generation_settings_checked":true,"suite_sha256":"\(suiteSHA256)","eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: evalIndex, atomically: true, encoding: .utf8)
        try """
        {
          "schema": "jang-expert-lab-pruned-bf16-suite-v1",
          "ready": true,
          "pruned_source": "\(pruned.path)",
          "suite_sha256": "\(suiteSHA256)",
          "prompt_count": 51,
          "generation_count": 51,
          "runtime_mode": "bf16_vmlx",
          "runtime_backend": "vmlx",
          "runtime_device": "Unit Metal",
          "runtime_metal_enabled": true,
          "jang_tools_version": "2.5.31",
          "mlx_version": "0.31.2",
          "mlx_lm_version": "0.31.3",
          "runtime_source_model_path": "\(pruned.path)",
          "reviewed_masked_comparison_count": 51,
          "reviewed_masked_mean_text_delta": 0.0,
          "reviewed_masked_max_text_delta": 0.0,
          "pruned_validator_outcomes_checked": true,
          "baseline_qualified_prompt_count": 51,
          "pruned_baseline_qualified_pass_rate": 1.0,
          "pruned_classification_counts": {\(Self.classificationCountsJSON(count: 51))},
          "baseline_invalid_prompt_ids": [],
          "inconclusive_prompt_ids": [],
          "pruned_preserved_prompt_ids": [\(promptIDs)],
          "pruned_degraded_prompt_ids": [],
          "baseline_qualified_semantic_coverage": [\(Self.requiredSemanticCoverageJSON())],
          "missing_baseline_qualified_semantic_coverage": [],
          "reviewed_masked_eval_trace_jsonl": "\(evalTrace.path)",
          "generations_jsonl": "\(prunedGenerations.path)"
        }
        """
            .write(to: prunedSummary, atomically: true, encoding: .utf8)
    }

    private func appendLine(_ line: String, to url: URL) throws {
        let existing = try String(contentsOf: url, encoding: .utf8)
        try existing
            .appending(line)
            .appending("\n")
            .write(to: url, atomically: true, encoding: .utf8)
    }

    private func writeEvalIndexPromptOrder(in pruned: URL, indices: [Int]) throws {
        let promptIDs = Self.promptIDJSON(indices: indices)
        let suiteSHA256 = try Self.fileSHA256(pruned.appendingPathComponent("expert_lab_suite.jsonl"))
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_semantic_coverage":[],"validator_schema":"jang-expert-lab-validator-v1","validator_available_prompt_count":50,"prompt_classification_counts":{\(Self.classificationCountsJSON(count: 50))},"baseline_qualified_prompt_count":50,"baseline_qualified_prompt_ids":[\(promptIDs)],"baseline_invalid_prompt_ids":[],"inconclusive_prompt_ids":[],"preserved_prompt_ids":[\(promptIDs)],"degraded_prompt_ids":[],"baseline_qualified_masked_pass_rate":1.0,"baseline_qualified_semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_baseline_qualified_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"generation_settings_checked":true,"suite_sha256":"\(suiteSHA256)","eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)
    }

    private func writePrunedGenerationPromptOrder(in pruned: URL, indices: [Int]) throws {
        let text = indices
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"}}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try text.write(to: pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl"), atomically: true, encoding: .utf8)
    }

    private func reviewedPrunePlan(original: URL, pruned: URL, prunePlan: URL) -> ConversionPlan {
        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = tmp.appendingPathComponent("converted")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 128,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 1024, shardCount: 1)
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = prunePlan
        plan.expertReviewPruneReportURL = pruned.appendingPathComponent("expert_lab_prune_report.md")
        return plan
    }

    private static let requiredSemanticDomains = [
        "math",
        "code",
        "formatting",
        "instruction_following",
        "reasoning",
        "safety_medical_legal_sensitive",
        "chinese",
        "non_english",
        "multilingual",
        "translation",
        "english_dominant",
        "unknown_language_role"
    ]

    private static func promptIDJSON(count: Int) -> String {
        (0..<count).map { #""p\#($0)""# }.joined(separator: ",")
    }

    private static func promptIDJSON(indices: [Int]) -> String {
        indices.map { #""p\#($0)""# }.joined(separator: ",")
    }

    private static func stringArrayJSON(_ values: [String]) -> String {
        values.map { #""\#($0)""# }.joined(separator: ",")
    }

    private static func requiredSemanticCoverageJSON() -> String {
        stringArrayJSON(requiredSemanticDomains)
    }

    private static func semanticDomainsJSON(index: Int) -> String {
        let domains = index < requiredSemanticDomains.count ? [requiredSemanticDomains[index]] : ["general"]
        return stringArrayJSON(domains)
    }

    private static func classificationCountsJSON(count: Int) -> String {
        #""baseline_invalid":0,"preserved":\#(count),"degraded":0,"inconclusive":0"#
    }

    private static func validComparisonSummaryJSON(promptCount: Int = 50) -> String {
        """
        {
          "promptCount": \(promptCount),
          "passRateBaseline": 1.0,
          "passRateMasked": 1.0,
          "validatorAvailablePromptCount": \(promptCount),
          "classificationCounts": {\(classificationCountsJSON(count: promptCount))},
          "baselineQualifiedPromptCount": \(promptCount),
          "baselineQualifiedMaskedPassRate": 1.0,
          "baselineQualifiedPromptIDs": [\(promptIDJSON(count: promptCount))],
          "baselineInvalidPromptIDs": [],
          "inconclusivePromptIDs": [],
          "preservedPromptIDs": [\(promptIDJSON(count: promptCount))],
          "degradedPromptIDs": [],
          "baselineQualifiedSemanticCoverage": [\(requiredSemanticCoverageJSON())],
          "missingBaselineQualifiedSemanticCoverage": [],
          "meanTextDelta": 0.0,
          "highRiskDomains": [],
          "safeDropCandidates": [{"layer": 0, "expert": 1}]
        }
        """
    }

    private static func fileSHA256(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func validReviewedPrunePlanJSON(
        keepExperts: Int = 128,
        trainedTopK: Int = 8,
        includeSemanticEvidence: Bool = true,
        includePromptTags: Bool = true,
        includeMaskedImpact: Bool = true,
        includeMaskedImpactScope: Bool = true,
        includeReviewedMaskMember: Bool = true
    ) -> String {
        let promptIDs = promptIDJSON(count: 50)
        let promptTags = includePromptTags ? #""translation", "non_english""# : ""
        let maskedImpactFields = [
            includeMaskedImpact ? #""ablation_delta": 0.0"# : nil,
            includeMaskedImpact && includeMaskedImpactScope ? #""masked_impact_scope": "same_suite_mask_mean_text_delta""# : nil,
            includeMaskedImpact && includeReviewedMaskMember ? #""reviewed_mask_member": true"# : nil
        ].compactMap { $0 }
        let maskedImpactLine = maskedImpactFields.isEmpty
            ? ""
            : maskedImpactFields.joined(separator: ", ") + ","
        let semanticEvidence = includeSemanticEvidence
            ? """
              {
                "expert": 0,
                "hits": 18,
                "probabilityMass": 0.62,
                "frequency": 0.36,
                "router_mass": 0.62,
                \(maskedImpactLine)
                "domains": {"multilingual": 8, "chinese": 5, "non_english": 8},
                "domain_lift": {"chinese": 2.4, "non_english": 1.8, "multilingual": 1.5},
                "prompt_evidence": [
                  {
                    "promptID": "p0",
                    "domain": "multilingual",
                    "subdomain": "chinese",
                    "tags": [\(promptTags)],
                    "promptExcerpt": "Translate the sentence into Simplified Chinese.",
                    "hits": 5
                  }
                ],
                "label": "chinese-specialist",
                "reason": "kept by reviewed BF16/vMLX prompt evidence",
                "kept": true
              }
            """
            : """
              {
                "expert": 0,
                "hits": 18,
                "probabilityMass": 0.62,
                "frequency": 0.36,
                "router_mass": 0.62,
                \(maskedImpactLine)
                "domains": {"multilingual": 8, "chinese": 5, "non_english": 8},
                "label": "chinese-specialist",
                "reason": "legacy row without semantic proof",
                "kept": true
              }
            """
        return """
        {
          "version": 1,
          "method": "prompt_trace_hits_mass_domain_lift_v1",
          "source_model": "/tmp/jang-unit-bf16-source",
          "promptCount": 50,
          "keepExpertsPerLayer": \(keepExperts),
          "comparison_summary": {
            "promptCount": 50,
            "passRateBaseline": 1.0,
            "passRateMasked": 1.0,
            "validatorAvailablePromptCount": 50,
            "classificationCounts": {\(classificationCountsJSON(count: 50))},
            "baselineQualifiedPromptCount": 50,
            "baselineQualifiedMaskedPassRate": 1.0,
            "baselineQualifiedPromptIDs": [\(promptIDs)],
            "baselineInvalidPromptIDs": [],
            "inconclusivePromptIDs": [],
            "preservedPromptIDs": [\(promptIDs)],
            "degradedPromptIDs": [],
            "baselineQualifiedSemanticCoverage": [\(requiredSemanticCoverageJSON())],
            "missingBaselineQualifiedSemanticCoverage": [],
            "meanTextDelta": 0.0,
            "meanLatencyDeltaPct": 0.0,
            "highRiskDomains": [],
            "safeDropCandidates": [{"layer": 0, "expert": 1}]
          },
          "eval_index": {
            "schema": "jang-expert-lab-eval-index-v1",
            "prompt_count": 50,
            "prompt_ids": [\(promptIDs)],
            "risky_prompt_ids": [],
            "high_risk_domains": [],
            "semantic_coverage": [\(requiredSemanticCoverageJSON())],
            "missing_semantic_coverage": [],
            "validator_schema": "jang-expert-lab-validator-v1",
            "validator_available_prompt_count": 50,
            "prompt_classification_counts": {\(classificationCountsJSON(count: 50))},
            "baseline_qualified_prompt_count": 50,
            "baseline_qualified_prompt_ids": [\(promptIDs)],
            "baseline_invalid_prompt_ids": [],
            "inconclusive_prompt_ids": [],
            "preserved_prompt_ids": [\(promptIDs)],
            "degraded_prompt_ids": [],
            "baseline_qualified_masked_pass_rate": 1.0,
            "baseline_qualified_semantic_coverage": [\(requiredSemanticCoverageJSON())],
            "missing_baseline_qualified_semantic_coverage": [],
            "min_baseline_tokens": 12,
            "min_masked_tokens": 12,
            "mean_baseline_tokens": 12.0,
            "mean_masked_tokens": 12.0,
            "baseline_route_record_count": 50,
            "masked_route_record_count": 50,
            "eval_trace_jsonl": "expert_lab_eval_trace.jsonl",
            "runtime_mode": "bf16_vmlx",
            "runtime_backend": "vmlx",
            "runtime_device": "Unit Metal",
            "runtime_metal_enabled": true,
            "hooked_moe_layers": 1,
            "jang_tools_version": "2.5.31",
            "mlx_version": "0.31.2",
            "mlx_lm_version": "0.31.3",
            "source_model_path": "/tmp/jang-unit-bf16-source",
            "mask_applied": true,
            "disabled_expert_count": 1
          },
          "safety": {
            "passed": true,
            "minimum_active_experts_per_layer": \(keepExperts),
            "trained_top_k_by_layer": {"0": \(trainedTopK)},
            "issues": []
          },
          "target": {"type": "keep_per_layer", "keep_experts_per_layer": \(keepExperts)},
          "layers": {
            "0": {
              "layer": 0,
              "num_source_experts": 256,
              "keep": [0],
              "drop": [1],
              "evidence": [
                \(semanticEvidence),
                {"expert": 1, "hits": 0, "probabilityMass": 0.0, "domains": {}, "label": "unobserved", "kept": false}
              ]
            }
          }
        }
        """
    }


    private static func validIntentPrunePlanJSON(
        keepExperts: Int = 1,
        trainedTopK: Int = 1,
        safetyStance: String = "balanced",
        includeCrackPack: Bool? = nil,
        scorer: String = "hybrid_v1"
    ) -> String {
        let promptIDs = promptIDJSON(count: 50)
        let includeCrack = includeCrackPack ?? (safetyStance == "crack")
        let crackBlock: String
        if includeCrack {
            crackBlock = """
              "crack_pack": {
                "name": "crack-probes-v1",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "prompt_count": 18
              },
            """
        } else {
            crackBlock = """
              "crack_pack": {},
            """
        }
        return """
        {
          "schema": "jang-intent-prune-plan-v1",
          "schema_version": 1,
          "scorer": "\(scorer)",
          "preset": "balanced",
          "weights": {
            "path": 0.30,
            "mass": 0.20,
            "intent": 0.35,
            "drop": 0.10,
            "backbone_floor": 0.05,
            "safety_keep": 0.15,
            "safety_balanced": 0.05,
            "safety_crack": 0.25
          },
          "intents_keep": ["code", "math"],
          "intents_drop": [],
          "safety_stance": "\(safetyStance)",
          "keep_experts_per_layer": \(keepExperts),
          "num_experts_source": 2,
          "num_layers": 1,
          "suite": {
            "name": "Reviewed Prune 50",
            "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "prompt_count": 50
          },
          \(crackBlock)
          "source_model": "/tmp/jang-unit-bf16-source",
          "backend": "qwen35_moe_vmlx",
          "prompt_count": 50,
          "comparison_summary": {
            "promptCount": 50,
            "passRateBaseline": 1.0,
            "passRateMasked": 1.0,
            "validatorAvailablePromptCount": 50,
            "classificationCounts": {\(classificationCountsJSON(count: 50))},
            "baselineQualifiedPromptCount": 50,
            "baselineQualifiedMaskedPassRate": 1.0,
            "baselineQualifiedPromptIDs": [\(promptIDs)],
            "baselineInvalidPromptIDs": [],
            "inconclusivePromptIDs": [],
            "preservedPromptIDs": [\(promptIDs)],
            "degradedPromptIDs": [],
            "baselineQualifiedSemanticCoverage": [\(requiredSemanticCoverageJSON())],
            "missingBaselineQualifiedSemanticCoverage": [],
            "meanTextDelta": 0.0,
            "meanLatencyDeltaPct": 0.0,
            "highRiskDomains": [],
            "safeDropCandidates": [{"layer": 0, "expert": 1}]
          },
          "eval_index": {
            "schema": "jang-expert-lab-eval-index-v1",
            "prompt_count": 50,
            "prompt_ids": [\(promptIDs)],
            "risky_prompt_ids": [],
            "high_risk_domains": [],
            "semantic_coverage": [\(requiredSemanticCoverageJSON())],
            "missing_semantic_coverage": [],
            "validator_schema": "jang-expert-lab-validator-v1",
            "validator_available_prompt_count": 50,
            "prompt_classification_counts": {\(classificationCountsJSON(count: 50))},
            "baseline_qualified_prompt_count": 50,
            "baseline_qualified_prompt_ids": [\(promptIDs)],
            "baseline_invalid_prompt_ids": [],
            "inconclusive_prompt_ids": [],
            "preserved_prompt_ids": [\(promptIDs)],
            "degraded_prompt_ids": [],
            "baseline_qualified_masked_pass_rate": 1.0,
            "baseline_qualified_semantic_coverage": [\(requiredSemanticCoverageJSON())],
            "missing_baseline_qualified_semantic_coverage": [],
            "min_baseline_tokens": 12,
            "min_masked_tokens": 12,
            "mean_baseline_tokens": 12.0,
            "mean_masked_tokens": 12.0,
            "baseline_route_record_count": 50,
            "masked_route_record_count": 50,
            "eval_trace_jsonl": "expert_lab_eval_trace.jsonl",
            "runtime_mode": "bf16_vmlx",
            "runtime_backend": "vmlx",
            "runtime_device": "Unit Metal",
            "runtime_metal_enabled": true,
            "hooked_moe_layers": 1,
            "jang_tools_version": "2.5.31",
            "mlx_version": "0.31.2",
            "mlx_lm_version": "0.31.3",
            "source_model_path": "/tmp/jang-unit-bf16-source",
            "mask_applied": true,
            "disabled_expert_count": 1
          },
          "safety": {
            "passed": true,
            "minimum_active_experts_per_layer": \(keepExperts),
            "trained_top_k": \(trainedTopK),
            "issues": []
          },
          "layers": {
            "0": [0]
          }
        }
        """
    }

    private static func safetyOnlyReviewedPrunePlanJSON(
        keepExperts: Int = 128,
        trainedTopK: Int = 8
    ) -> String {
        """
        {
          "version": 1,
          "method": "prompt_trace_hits_mass_domain_lift_v1",
          "promptCount": 50,
          "keepExpertsPerLayer": \(keepExperts),
          "safety": {
            "passed": true,
            "minimum_active_experts_per_layer": \(keepExperts),
            "trained_top_k_by_layer": {"0": \(trainedTopK)},
            "issues": []
          },
          "target": {"type": "keep_per_layer", "keep_experts_per_layer": \(keepExperts)},
          "layers": {}
        }
        """
    }
}
