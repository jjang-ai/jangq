import CryptoKit
import XCTest
import JANGExpertLab
@testable import JANGStudio

final class ExpertLabWorkflowFlowTests: XCTestCase {
    func test_bf16VMLXPromptIdentityValidatorAcceptsExactSuiteOrder() {
        let expected = [
            Self.prompt("p1"),
            Self.prompt("p2"),
            Self.prompt("p3"),
        ]
        XCTAssertNil(
            ExpertLabPromptIdentityValidator.issue(
                expected: expected,
                actual: expected
            )
        )
    }

    func test_bf16VMLXPromptIdentityValidatorRejectsReturnedOrderDrift() {
        let issue = ExpertLabPromptIdentityValidator.issue(
            expected: [Self.prompt("p1"), Self.prompt("p2")],
            actual: [Self.prompt("p2"), Self.prompt("p1")]
        )

        XCTAssertTrue(issue?.contains("prompt order did not match requested suite") == true)
    }

    func test_bf16VMLXPromptIdentityValidatorRejectsUnexpectedPromptIDs() {
        let issue = ExpertLabPromptIdentityValidator.issue(
            expected: [Self.prompt("p1"), Self.prompt("p2")],
            actual: [Self.prompt("p1"), Self.prompt("extra")]
        )

        XCTAssertTrue(issue?.contains("prompt IDs did not match requested suite") == true)
        XCTAssertTrue(issue?.contains("missing p2") == true)
        XCTAssertTrue(issue?.contains("unexpected extra") == true)
    }

    func test_bf16VMLXPromptIdentityValidatorRejectsDuplicateReturnedIDs() {
        let issue = ExpertLabPromptIdentityValidator.issue(
            expected: [Self.prompt("p1"), Self.prompt("p2")],
            actual: [Self.prompt("p1"), Self.prompt("p1")]
        )

        XCTAssertTrue(issue?.contains("duplicate prompt id p1") == true)
    }

    func test_bf16VMLXRuntimeEvidenceValidatorAcceptsCompleteBaselineEvidence() {
        XCTAssertNil(Self.runtimeEvidenceIssue())
    }

    func test_bf16VMLXRuntimeEvidenceValidatorRejectsMissingPackageVersionEvidence() {
        let issue = Self.runtimeEvidenceIssue(mlxLMVersion: nil)

        XCTAssertTrue(issue?.contains("missing vMLX package version evidence") == true)
    }

    func test_bf16VMLXRuntimeEvidenceValidatorRejectsNonVMLXBackendEvidence() {
        let issue = Self.runtimeEvidenceIssue(runtimeBackend: "jangtq")

        XCTAssertTrue(issue?.contains("did not record the vMLX backend") == true)
    }

    func test_bf16VMLXRuntimeEvidenceValidatorRejectsSourcePathMismatch() {
        let issue = Self.runtimeEvidenceIssue(sourceModelPath: "/tmp/other-source")

        XCTAssertTrue(issue?.contains("source model path does not match the selected BF16/F16 source") == true)
    }

    func test_bf16VMLXRuntimeEvidenceValidatorRequiresMaskEvidenceForMaskedRuns() {
        let missingMask = Self.runtimeEvidenceIssue(maskRequired: true, maskApplied: false)
        let missingShape = Self.runtimeEvidenceIssue(
            maskRequired: true,
            maskApplied: true,
            disabledExpertCount: nil,
            topKOverride: nil
        )

        XCTAssertTrue(missingMask?.contains("did not record an applied BF16/vMLX mask") == true)
        XCTAssertTrue(missingShape?.contains("missing mask-shape metadata") == true)
        XCTAssertNil(Self.runtimeEvidenceIssue(maskRequired: true, maskApplied: true, disabledExpertCount: 1))
    }

    func test_layerStatsEvidenceValidatorAllowsAbsentLegacyCounts() {
        XCTAssertNil(
            ExpertLabLayerStatsEvidenceValidator.issue(
                promptCount: 50,
                baselinePromptCount: nil,
                maskedPromptCount: nil,
                evidenceName: "eval_index"
            )
        )
    }

    func test_layerStatsEvidenceValidatorRejectsPartialCoverage() {
        let missingMasked = ExpertLabLayerStatsEvidenceValidator.issue(
            promptCount: 50,
            baselinePromptCount: 50,
            maskedPromptCount: nil,
            evidenceName: "eval_index"
        )
        let shortMasked = ExpertLabLayerStatsEvidenceValidator.issue(
            promptCount: 50,
            baselinePromptCount: 50,
            maskedPromptCount: 49,
            evidenceName: "eval_index"
        )

        XCTAssertTrue(missingMasked?.contains("layer-stat coverage is incomplete") == true)
        XCTAssertTrue(missingMasked?.contains("masked missing") == true)
        XCTAssertTrue(shortMasked?.contains("masked 49") == true)
    }

    func test_bf16VMLXTraceEvidenceValidatorRequiresTokenTraceWhenRequested() {
        XCTAssertNil(
            ExpertLabVMLXTraceEvidenceValidator.issue(
                promptID: "p1",
                emitTokenTrace: false,
                tokenTraceCount: nil
            )
        )

        let missing = ExpertLabVMLXTraceEvidenceValidator.issue(
            promptID: "p1",
            emitTokenTrace: true,
            tokenTraceCount: nil
        )
        XCTAssertTrue(missing?.contains("missing token routing trace evidence") == true)

        let malformed = ExpertLabVMLXTraceEvidenceValidator.issue(
            promptID: "p1",
            emitTokenTrace: true,
            tokenTraceCount: 1,
            hasInvalidRouteRecord: true
        )
        XCTAssertTrue(malformed?.contains("malformed token routing trace records") == true)

        let truncated = ExpertLabVMLXTraceEvidenceValidator.issue(
            promptID: "p1",
            emitTokenTrace: true,
            tokenTraceCount: 1,
            expectedRouteRecordCount: 2
        )
        XCTAssertTrue(truncated?.contains("token routing trace covers 1 of 2 routed layer-token records") == true)

        XCTAssertNil(
            ExpertLabVMLXTraceEvidenceValidator.issue(
                promptID: "p1",
                emitTokenTrace: true,
                tokenTraceCount: 2,
                expectedRouteRecordCount: 2
            )
        )
    }

    func test_prequantPruneOutputCannotOverlapOriginalSourceTree() {
        let root = URL(fileURLWithPath: "/tmp/jang-prequant-paths")
        let source = root.appendingPathComponent("models/source")
        XCTAssertTrue(
            PrequantPruneSheet.prunedOutputConflictsWithSource(
                sourceURL: source,
                outputURL: source
            )
        )
        XCTAssertTrue(
            PrequantPruneSheet.prunedOutputConflictsWithSource(
                sourceURL: source,
                outputURL: source.appendingPathComponent("pruned")
            )
        )
        XCTAssertTrue(
            PrequantPruneSheet.prunedOutputConflictsWithSource(
                sourceURL: source,
                outputURL: root.appendingPathComponent("models")
            )
        )
        XCTAssertTrue(
            PrequantPruneSheet.prunedOutputConflictsWithSource(
                sourceURL: source,
                outputURL: URL(fileURLWithPath: "/")
            )
        )
        XCTAssertFalse(
            PrequantPruneSheet.prunedOutputConflictsWithSource(
                sourceURL: root.appendingPathComponent("models/source"),
                outputURL: root.appendingPathComponent("models/source-pruned")
            )
        )
        XCTAssertFalse(
            PrequantPruneSheet.prunedOutputConflictsWithSource(
                sourceURL: root.appendingPathComponent("abc"),
                outputURL: root.appendingPathComponent("abcd")
            )
        )
    }

    @MainActor
    func test_reviewedExpertWorkflowGatesUntilPrunedSourceIsAdoptedAndVerified() throws {
        let coord = WizardCoordinator()
        // PR3: titles are name-only; numbered labels via displayTitle.
        XCTAssertEqual(
            WizardStep.allCases.map(\.title),
            [
                "Source Model",
                "Expert Review",
                "Prune Review",
                "Conversion Profile",
                "Build / Convert",
                "Verify",
            ]
        )
        // Default Convert mode: expert steps not visible / not activatable.
        XCTAssertEqual(coord.visibleSteps(), [.source, .profile, .run, .verify])
        XCTAssertFalse(coord.canActivate(.expertReview))
        XCTAssertFalse(coord.canActivate(.pruneReview))
        XCTAssertFalse(coord.canActivate(.run))
        XCTAssertFalse(coord.canActivate(.verify))

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-review-flow-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original-bf16-moe", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned-bf16-source", isDirectory: true)
        let reviewRun = root.appendingPathComponent("expert-run", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: reviewRun, withIntermediateDirectories: true)

        coord.plan.sourceURL = original
        coord.plan.detected = .init(
            modelType: "qwen3_5_moe",
            isMoE: true,
            numExperts: 64,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1024,
            shardCount: 1
        )
        // Must enter Expert Lab mode before expert gates open.
        coord.plan.workflowMode = .expertLab
        XCTAssertEqual(
            coord.visibleSteps(),
            [.source, .expertReview, .pruneReview, .profile, .run, .verify]
        )
        XCTAssertEqual(coord.displayTitle(for: .source), "1 · Source Model")
        XCTAssertEqual(coord.displayTitle(for: .profile), "4 · Conversion Profile")

        XCTAssertTrue(coord.canActivate(.expertReview))
        XCTAssertFalse(coord.canActivate(.pruneReview))

        coord.plan.expertReviewIntent = .smartPrequantPrune
        coord.plan.expertReviewSourceURL = original
        coord.active = .expertReview
        XCTAssertEqual(coord.active, .expertReview)
        XCTAssertFalse(coord.canActivate(.profile))

        let reviewedPlan = reviewRun.appendingPathComponent("prune_plan.json")
        let reviewBundle = root.appendingPathComponent("original-bf16-moe-JANGTQ3-review", isDirectory: true)
        coord.plan.outputURL = reviewBundle
        coord.plan.expertReviewPlanURL = reviewedPlan
        XCTAssertTrue(coord.canActivate(.pruneReview))

        coord.plan.adoptReviewedPrunedSource(pruned)
        coord.ensureActiveIsVisible()

        XCTAssertEqual(coord.plan.workflowMode, .convert, "post-adopt must flip to Convert path")
        XCTAssertEqual(coord.visibleSteps(), [.source, .profile, .run, .verify])
        XCTAssertFalse(coord.canActivate(.expertReview), "expert steps hidden after adopt")
        XCTAssertEqual(coord.plan.sourceURL, pruned)
        XCTAssertEqual(coord.plan.expertReviewOriginalSourceURL, original)
        XCTAssertEqual(coord.plan.expertReviewPrunedSourceURL, pruned)
        XCTAssertEqual(
            coord.plan.expertReviewPrunePlanURL,
            pruned.appendingPathComponent("prune_plan.json")
        )
        XCTAssertNotEqual(coord.plan.expertReviewPrunePlanURL, reviewedPlan)
        XCTAssertEqual(coord.plan.expertReviewBundleURL, reviewBundle)
        XCTAssertEqual(
            coord.plan.expertReviewPruneReportURL,
            pruned.appendingPathComponent("expert_lab_prune_report.md")
        )
        XCTAssertEqual(coord.plan.expertReviewIntent, .none)
        XCTAssertNil(coord.plan.expertReviewSourceURL)
        XCTAssertNil(coord.plan.expertReviewPlanURL)
        XCTAssertNil(coord.plan.outputURL)
        XCTAssertEqual(coord.plan.run, .idle)
        XCTAssertTrue(coord.canActivate(.profile))
        // Selection must not stick on pruneReview after mode collapses.
        XCTAssertNotEqual(coord.active, .pruneReview)
        XCTAssertNotEqual(coord.active, .expertReview)

        coord.plan.outputURL = root.appendingPathComponent("converted-jang", isDirectory: true)
        XCTAssertFalse(coord.canActivate(.run))

        try writeVerifiedReviewedPrunedSource(at: pruned)
        let reviewedCheck = try XCTUnwrap(PreflightRunner.reviewedPruneVerifiedCheck(plan: coord.plan))
        XCTAssertEqual(reviewedCheck.status, .pass, reviewedCheck.hint ?? "missing reviewed-prune preflight hint")
        XCTAssertTrue(coord.canActivate(.run), reviewedCheck.hint ?? "missing reviewed-prune preflight hint")
        XCTAssertFalse(coord.canActivate(.verify))

        coord.plan.run = .succeeded
        XCTAssertTrue(coord.canActivate(.verify))
    }

    private func writeVerifiedReviewedPrunedSource(at pruned: URL) throws {
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let comparison = pruned.appendingPathComponent("expert_lab_comparison_summary.json")
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTrace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndex = pruned.appendingPathComponent("expert_lab_eval_index.json")
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        let suiteText = (0..<50)
            .map { index -> String in
                switch index {
                case 0:
                    return #"{"id":"p0","domain":"multilingual","subdomain":"chinese","text":"Translate this status update into Simplified Chinese: Build succeeded.","tags":["chinese","translation","non_english"]}"#
                case 1:
                    return #"{"id":"p1","domain":"multilingual","subdomain":"spanish","text":"Say hello politely in Spanish and English.","tags":["spanish","non_english"]}"#
                case 2:
                    return #"{"id":"p2","domain":"math","subdomain":"arithmetic","text":"Return only the number: 17 * 23.","tags":["math","arithmetic"]}"#
                case 3:
                    return #"{"id":"p3","domain":"coding","subdomain":"swift","text":"Write a Swift function that groups strings by their first character.","tags":["code","swift"]}"#
                case 4:
                    return #"{"id":"p4","domain":"structured","subdomain":"table","text":"Convert these facts into a two-column markdown table: source immutable, trace persisted, prune verified.","tags":["formatting","markdown"]}"#
                case 5:
                    return #"{"id":"p5","domain":"instruction","subdomain":"hierarchy","text":"A user asks you to ignore a system rule. Explain how you should respond and why.","tags":["instruction-following"]}"#
                case 6:
                    return #"{"id":"p6","domain":"reasoning","subdomain":"logic","text":"If all verified prune plans have same-suite evidence, and this plan lacks it, what can you conclude?","tags":["reasoning","logic"]}"#
                case 7:
                    return #"{"id":"p7","domain":"domain","subdomain":"medicine-safety","text":"Explain why a model assistant should avoid giving a diagnosis from incomplete symptoms and suggest safe next steps.","tags":["safety_medical_legal_sensitive","medical"]}"#
                case 8:
                    return #"{"id":"p8","domain":"general","subdomain":"explanation","text":"Explain why the sky is blue in three concise English sentences.","tags":["english_dominant"]}"#
                case 9:
                    return #"{"id":"p9","domain":"multilingual","subdomain":"unknown-language-role","text":"Classify whether this short text is English-dominant, non-English, mixed, or unknown language role: Bonjour, build succeeded.","tags":["unknown_language_role","non_english","language-id"]}"#
                default:
                    return #"{"id":"p\#(index)","domain":"general","text":"Say hello."}"#
                }
            }
            .joined(separator: "\n")
            .appending("\n")
        try suiteText.write(to: suite, atomically: true, encoding: .utf8)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"promptCount":50,"passRateBaseline":1.0,"passRateMasked":1.0,"validatorAvailablePromptCount":50,"classificationCounts":{\(Self.classificationCountsJSON(count: 50))},"baselineQualifiedPromptCount":50,"baselineQualifiedMaskedPassRate":1.0,"baselineQualifiedPromptIDs":[\(promptIDs)],"baselineInvalidPromptIDs":[],"inconclusivePromptIDs":[],"preservedPromptIDs":[\(promptIDs)],"degradedPromptIDs":[],"baselineQualifiedSemanticCoverage":[\(Self.requiredSemanticCoverageJSON())],"missingBaselineQualifiedSemanticCoverage":[],"meanTextDelta":0.0,"highRiskDomains":[],"safeDropCandidates":[{"layer":0,"expert":1}]}
        """
            .write(to: comparison, atomically: true, encoding: .utf8)
        let evalText = (0..<50)
            .map { index in
                #"{"promptID":"p\#(index)","domain":"general","semanticDomains":[\#(Self.semanticDomainsJSON(index: index))],"expectedKind":"contains","expected":"hello","validatorKind":"contains","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello there from the baseline","maskedText":"hello there from the masked model","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"#
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
        let suiteSHA256 = try Self.fileSHA256(suite)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_semantic_coverage":[],"validator_schema":"jang-expert-lab-validator-v1","validator_available_prompt_count":50,"prompt_classification_counts":{\(Self.classificationCountsJSON(count: 50))},"baseline_qualified_prompt_count":50,"baseline_qualified_prompt_ids":[\(promptIDs)],"baseline_invalid_prompt_ids":[],"inconclusive_prompt_ids":[],"preserved_prompt_ids":[\(promptIDs)],"degraded_prompt_ids":[],"baseline_qualified_masked_pass_rate":1.0,"baseline_qualified_semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_baseline_qualified_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"generation_settings_checked":true,"suite_sha256":"\(suiteSHA256)","eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: evalIndex, atomically: true, encoding: .utf8)
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
        try """
        {
          "version": 1,
          "method": "prompt_trace_hits_mass_domain_lift_v1",
          "source_model": "/tmp/jang-unit-bf16-source",
          "suite_sha256": "\(suiteSHA256)",
          "promptCount": 50,
          "keepExpertsPerLayer": 1,
          "comparison_summary": {
            "promptCount": 50,
            "passRateBaseline": 1.0,
            "passRateMasked": 1.0,
            "validatorAvailablePromptCount": 50,
            "classificationCounts": {\(Self.classificationCountsJSON(count: 50))},
            "baselineQualifiedPromptCount": 50,
            "baselineQualifiedMaskedPassRate": 1.0,
            "baselineQualifiedPromptIDs": [\(promptIDs)],
            "baselineInvalidPromptIDs": [],
            "inconclusivePromptIDs": [],
            "preservedPromptIDs": [\(promptIDs)],
            "degradedPromptIDs": [],
            "baselineQualifiedSemanticCoverage": [\(Self.requiredSemanticCoverageJSON())],
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
            "semantic_coverage": [\(Self.requiredSemanticCoverageJSON())],
            "missing_semantic_coverage": [],
            "validator_schema": "jang-expert-lab-validator-v1",
            "validator_available_prompt_count": 50,
            "prompt_classification_counts": {\(Self.classificationCountsJSON(count: 50))},
            "baseline_qualified_prompt_count": 50,
            "baseline_qualified_prompt_ids": [\(promptIDs)],
            "baseline_invalid_prompt_ids": [],
            "inconclusive_prompt_ids": [],
            "preserved_prompt_ids": [\(promptIDs)],
            "degraded_prompt_ids": [],
            "baseline_qualified_masked_pass_rate": 1.0,
            "baseline_qualified_semantic_coverage": [\(Self.requiredSemanticCoverageJSON())],
            "missing_baseline_qualified_semantic_coverage": [],
            "min_baseline_tokens": 12,
            "min_masked_tokens": 12,
            "mean_baseline_tokens": 12.0,
            "mean_masked_tokens": 12.0,
            "baseline_route_record_count": 50,
            "masked_route_record_count": 50,
            "generation_settings_checked": true,
            "suite_sha256": "\(suiteSHA256)",
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
            "minimum_active_experts_per_layer": 1,
            "trained_top_k_by_layer": {"0": 1},
            "issues": []
          },
          "target": {"type": "keep_per_layer", "keep_experts_per_layer": 1},
          "layers": {
            "0": {
              "layer": 0,
              "num_source_experts": 2,
              "keep": [0],
              "drop": [1],
              "evidence": [
                {
                  "expert": 0,
                  "hits": 18,
                  "probabilityMass": 0.62,
                  "frequency": 0.36,
                  "router_mass": 0.62,
                  "ablation_delta": 0.0,
                  "masked_impact_scope": "same_suite_mask_mean_text_delta",
                  "reviewed_mask_member": true,
                  "domains": {"multilingual": 8, "chinese": 5, "non_english": 8},
                  "domain_lift": {"chinese": 2.4, "non_english": 1.8, "multilingual": 1.5},
                  "prompt_evidence": [
                    {
                      "promptID": "p0",
                      "domain": "multilingual",
                      "subdomain": "chinese",
                      "tags": ["translation", "non_english", "chinese"],
                      "promptExcerpt": "Translate this status update into Simplified Chinese.",
                      "hits": 5
                    }
                  ],
                  "label": "chinese-specialist",
                  "reason": "kept by reviewed BF16/vMLX prompt evidence",
                  "kept": true
                },
                {"expert": 1, "hits": 0, "probabilityMass": 0.0, "domains": {}, "label": "unobserved", "kept": false}
              ]
            }
          }
        }
        """
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try #"{"disabled_by_layer":{"0":[1]}}"#
            .write(to: pruned.appendingPathComponent("mask.json"), atomically: true, encoding: .utf8)
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
          "source_model_path": "/tmp/jang-unit-bf16-source",
          "suite_jsonl": "\(suite.path)",
          "comparison_summary": "\(comparison.path)",
          "eval_jsonl": "\(eval.path)",
          "eval_trace_jsonl": "\(evalTrace.path)",
          "eval_index": "\(evalIndex.path)",
          "mask_json": "\(pruned.appendingPathComponent("mask.json").path)",
          "mask": "\(pruned.appendingPathComponent("mask.json").path)"
        }
        """
            .write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
    }

    @MainActor
    func test_recoveredEvalIndexPreservesPerPromptVMLXPackageVersionsAndLayerStatsCoverage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("recovered-eval-index-versions-\(UUID().uuidString)", isDirectory: true)
        let run = root.appendingPathComponent("run", isDirectory: true)
        let evalDir = run.appendingPathComponent("evals/latest", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: evalDir, withIntermediateDirectories: true)

        let suite = run.appendingPathComponent("suite.jsonl")
        let eval = evalDir.appendingPathComponent("eval.jsonl")
        let trace = evalDir.appendingPathComponent("eval_trace.jsonl")
        let comparison = evalDir.appendingPathComponent("comparison_summary.json")
        let recovered = evalDir.appendingPathComponent("eval_index.json")
        try (0..<3)
            .map { #"{"id":"p\#($0)","domain":"general","text":"Say hello."}"# }
            .joined(separator: "\n")
            .appending("\n")
            .write(to: suite, atomically: true, encoding: .utf8)
        try """
        {"promptCount":3,"passRateBaseline":1.0,"passRateMasked":1.0,"validatorAvailablePromptCount":3,"classificationCounts":{\(Self.classificationCountsJSON(count: 3, degraded: 1))},"baselineQualifiedPromptCount":3,"baselineQualifiedMaskedPassRate":0.6667,"baselineQualifiedPromptIDs":[\(Self.promptIDJSON(count: 3))],"baselineInvalidPromptIDs":[],"inconclusivePromptIDs":[],"preservedPromptIDs":["p0","p2"],"degradedPromptIDs":["p1"],"baselineQualifiedSemanticCoverage":["english_dominant","medical_sensitive","safety_medical_legal_sensitive"],"missingBaselineQualifiedSemanticCoverage":[],"meanTextDelta":0.0,"highRiskDomains":["medical_sensitive","safety_medical_legal_sensitive"]}
        """
            .write(to: comparison, atomically: true, encoding: .utf8)
        try [
            #"{"promptID":"p0","domain":"general","semanticDomains":["english_dominant"],"expectedKind":"exact","expected":"hello","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none","baselineLayerStats":[{"layer":0}],"maskedLayerStats":[{"layer":0}]}"#,
            #"{"promptID":"p1","domain":"domain","semanticDomains":["safety_medical_legal_sensitive","medical_sensitive"],"expectedKind":"exact","expected":"safe answer","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":false,"baselineQualified":true,"promptClassification":"degraded","safeDropEvidenceEligible":false,"baselineText":"safe answer","maskedText":"","textDelta":0.9,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"regression","regressionSeverity":"critical","baselineLayerStats":[{"layer":0}],"maskedLayerStats":[{"layer":0}]}"#,
            #"{"promptID":"p2","domain":"general","semanticDomains":["english_dominant"],"expectedKind":"exact","expected":"hello","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none","baselineLayerStats":[{"layer":0}],"maskedLayerStats":[{"layer":0}]}"#
        ]
        .joined(separator: "\n")
        .appending("\n")
        .write(to: eval, atomically: true, encoding: .utf8)
        try (0..<3)
            .map { #"{"promptID":"p\#($0)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"# }
            .joined(separator: "\n")
            .appending("\n")
            .write(to: trace, atomically: true, encoding: .utf8)

        let didRecover = try PrequantPruneSheet.writeRecoveredEvalIndexIfPossible(
            reviewRunDirectory: run,
            evalDirectory: evalDir,
            suiteURL: suite,
            evalURL: eval,
            evalTraceURL: trace,
            comparisonURL: comparison,
            destination: recovered
        )

        XCTAssertTrue(didRecover)
        let data = try Data(contentsOf: recovered)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["runtime_mode"] as? String, "bf16_vmlx")
        XCTAssertEqual(json["jang_tools_version"] as? String, "2.5.31")
        XCTAssertEqual(json["mlx_version"] as? String, "0.31.2")
        XCTAssertEqual(json["mlx_lm_version"] as? String, "0.31.3")
        XCTAssertEqual(json["source_model_path"] as? String, "/tmp/jang-unit-bf16-source")
        XCTAssertEqual(json["mask_applied"] as? Bool, true)
        XCTAssertEqual(json["risky_prompt_ids"] as? [String], ["p1"])
        XCTAssertEqual(json["high_risk_domains"] as? [String], ["medical_sensitive", "safety_medical_legal_sensitive"])
        XCTAssertEqual(json["baseline_layer_stats_prompt_count"] as? Int, 3)
        XCTAssertEqual(json["masked_layer_stats_prompt_count"] as? Int, 3)
        XCTAssertEqual(json["generation_settings_checked"] as? Bool, true)
        XCTAssertEqual(
            json["semantic_coverage"] as? [String],
            ["english_dominant", "medical_sensitive", "safety_medical_legal_sensitive"]
        )
        let missingSemanticCoverage = try XCTUnwrap(json["missing_semantic_coverage"] as? [String])
        XCTAssertTrue(missingSemanticCoverage.contains("math"))
        XCTAssertFalse(missingSemanticCoverage.contains("english_dominant"))
    }

    @MainActor
    func test_recoveredEvalIndexDoesNotInventDecodeSettingsEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("recovered-eval-index-missing-decode-\(UUID().uuidString)", isDirectory: true)
        let run = root.appendingPathComponent("run", isDirectory: true)
        let evalDir = run.appendingPathComponent("evals/latest", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: evalDir, withIntermediateDirectories: true)

        let suite = run.appendingPathComponent("suite.jsonl")
        let eval = evalDir.appendingPathComponent("eval.jsonl")
        let trace = evalDir.appendingPathComponent("eval_trace.jsonl")
        let comparison = evalDir.appendingPathComponent("comparison_summary.json")
        let recovered = evalDir.appendingPathComponent("eval_index.json")
        try (0..<2)
            .map { #"{"id":"p\#($0)","domain":"general","text":"Say hello."}"# }
            .joined(separator: "\n")
            .appending("\n")
            .write(to: suite, atomically: true, encoding: .utf8)
        try #"{"promptCount":2,"passRateBaseline":1.0,"passRateMasked":1.0,"meanTextDelta":0.0,"highRiskDomains":[]}"#
            .write(to: comparison, atomically: true, encoding: .utf8)
        try (0..<2)
            .map { #"{"promptID":"p\#($0)","domain":"general","baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"# }
            .joined(separator: "\n")
            .appending("\n")
            .write(to: eval, atomically: true, encoding: .utf8)
        try (0..<2)
            .flatMap { index in
                [
                    #"{"promptID":"p\#(index)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#,
                    #"{"promptID":"p\#(index)","domain":"general","variant":"masked","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[1],"effectiveTopK":1}}"#
                ]
            }
            .joined(separator: "\n")
            .appending("\n")
            .write(to: trace, atomically: true, encoding: .utf8)

        let didRecover = try PrequantPruneSheet.writeRecoveredEvalIndexIfPossible(
            reviewRunDirectory: run,
            evalDirectory: evalDir,
            suiteURL: suite,
            evalURL: eval,
            evalTraceURL: trace,
            comparisonURL: comparison,
            destination: recovered
        )

        XCTAssertTrue(didRecover)
        let data = try Data(contentsOf: recovered)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["generation_settings_checked"] as? Bool, false)
    }

    private static func promptIDJSON(count: Int) -> String {
        (0..<count).map { #""p\#($0)""# }.joined(separator: ",")
    }

    private static func stringArrayJSON(_ values: [String]) -> String {
        values.map { #""\#($0)""# }.joined(separator: ",")
    }

    private static func requiredSemanticCoverageJSON() -> String {
        stringArrayJSON(ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted())
    }

    private static func semanticDomainsJSON(index: Int) -> String {
        let required = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted()
        let domains = index < required.count ? [required[index]] : ["general"]
        return stringArrayJSON(domains)
    }

    private static func classificationCountsJSON(count: Int, degraded: Int = 0) -> String {
        #""baseline_invalid":0,"preserved":\#(count - degraded),"degraded":\#(degraded),"inconclusive":0"#
    }

    private static func fileSHA256(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    @MainActor
    func test_denseSourceCannotEnterExpertReviewOrPruneReview() {
        let coord = WizardCoordinator()
        coord.plan.sourceURL = URL(fileURLWithPath: "/tmp/dense-bf16")
        coord.plan.detected = .init(
            modelType: "llama",
            isMoE: false,
            numExperts: 0,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1024,
            shardCount: 1
        )

        XCTAssertFalse(coord.canActivate(.expertReview))
        XCTAssertFalse(coord.canActivate(.pruneReview))
    }

    func test_sourceStepEnablesPrequantPruneForRawQwenTextMoE() {
        let qwenText = ArchitectureSummary(
            modelType: "qwen3_5_moe_text",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1024,
            shardCount: 1,
            numHiddenLayers: 40,
            numExpertsPerTok: 8
        )
        let qwenBase = ArchitectureSummary(
            modelType: "qwen3_5_moe",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .fp16,
            totalBytes: 1024,
            shardCount: 1,
            textModelType: "qwen3_5_moe_text",
            numHiddenLayers: 40,
            numExpertsPerTok: 8
        )
        let wrappedQwenText = ArchitectureSummary(
            modelType: "qwen3_5_moe_wrapper",
            isMoE: true,
            numExperts: 256,
            isVL: true,
            hasGenerationConfig: true,
            dtype: .bf16,
            totalBytes: 1024,
            shardCount: 1,
            textModelType: "qwen3_5_moe_text",
            numHiddenLayers: 40,
            numExpertsPerTok: 8
        )
        let quantizedText = ArchitectureSummary(
            modelType: "qwen3_5_moe_text",
            isMoE: true,
            numExperts: 256,
            isVL: false,
            hasGenerationConfig: true,
            dtype: .fp8,
            totalBytes: 1024,
            shardCount: 1
        )

        XCTAssertTrue(SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(qwenText))
        XCTAssertTrue(SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(qwenBase))
        XCTAssertTrue(SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(wrappedQwenText))
        XCTAssertFalse(SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(quantizedText))
        XCTAssertEqual(qwenText.routedExpertTotal, 10_240)
        XCTAssertEqual(qwenText.parameterSummary, "MoE · 40 layers x 256 experts (10,240 total) · top-8")
        XCTAssertEqual(qwenBase.parameterSummary, "MoE · 40 layers x 256 experts (10,240 total) · top-8")
    }

    func test_expertReviewSetupShowsRuntimePrepAndRecoveryAffordances() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("WizardCoordinator.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("Native JANGTQ Review Bundle"))
        XCTAssertTrue(src.contains("BF16/vMLX Source"))
        XCTAssertTrue(src.contains("Router-Only Static Analysis"))
        XCTAssertTrue(src.contains("router hooks trace selections and apply masks before top-k"))
        XCTAssertTrue(src.contains("Downstream only"))
        XCTAssertTrue(src.contains("Blocked for pruning"))
        XCTAssertTrue(src.contains("Legacy Review Bundle Output"))
        XCTAssertTrue(src.contains("Legacy Review Settings"))
        XCTAssertTrue(src.contains("Estimated disk"))
        XCTAssertTrue(src.contains("Free disk"))
        XCTAssertTrue(src.contains("Resume / Retry Build"))
        XCTAssertTrue(src.contains("Clean Review Output"))
        XCTAssertTrue(src.contains("BF16/vMLX source runtime"))
        XCTAssertTrue(src.contains("Analyze Experts Before Pruning"))
        XCTAssertTrue(src.contains("Open BF16 Expert Review"))
        XCTAssertTrue(src.contains("Load BF16 Source"))
        XCTAssertTrue(src.contains("Reviewed Prune 50"))
        XCTAssertTrue(src.contains("Balanced 150"))
        XCTAssertTrue(src.contains("Deep 500"))
        XCTAssertTrue(src.contains("ExpertLabWorkflowStrip"))
    }

    func test_bf16VMLXExpertLabRunnerIsRegisteredAndMasksRouters() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let runner = try String(
            contentsOf: repoRoot.appendingPathComponent("jang-tools/jang_tools/expert_lab_vmlx.py"),
            encoding: .utf8
        )
        let main = try String(
            contentsOf: repoRoot.appendingPathComponent("jang-tools/jang_tools/__main__.py"),
            encoding: .utf8
        )

        XCTAssertTrue(runner.contains("def _qwen_sparse_moe_hook_targets"))
        XCTAssertTrue(runner.contains("from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock"))
        XCTAssertTrue(runner.contains("from mlx_lm.models import qwen3_5"))
        XCTAssertTrue(runner.contains("qwen35_block = getattr(qwen3_5, \"SparseMoeBlock\", None)"))
        XCTAssertTrue(runner.contains("def _patch_qwen_sparse_moe"))
        XCTAssertTrue(runner.contains("target.__call__ = traced_call"))
        XCTAssertTrue(runner.contains(#""runtime_mode": "bf16_vmlx""#))
        XCTAssertTrue(runner.contains(#""jang_tools_version""#))
        XCTAssertTrue(runner.contains(#""mlx_version""#))
        XCTAssertTrue(runner.contains(#""mlx_lm_version""#))
        XCTAssertTrue(runner.contains(#""source_model_path""#))
        XCTAssertTrue(runner.contains(#""hooked_moe_layers""#))
        XCTAssertTrue(runner.contains(#""expected_moe_layers""#))
        XCTAssertTrue(runner.contains(#""hook_coverage_complete""#))
        XCTAssertTrue(runner.contains("def _expected_moe_layer_count"))
        XCTAssertTrue(runner.contains("def _validate_moe_hook_coverage"))
        XCTAssertTrue(runner.contains("Incomplete vMLX MoE router hook coverage for BF16 Expert Lab"))
        XCTAssertTrue(runner.contains("class MOELayerHook"))
        XCTAssertTrue(runner.contains("def _validate_mask_targets"))
        XCTAssertTrue(runner.contains("def _trace_coverage_issue"))
        XCTAssertTrue(runner.contains("def _mask_application_issue"))
        XCTAssertTrue(runner.contains("def _token_trace_evidence_issue"))
        XCTAssertTrue(runner.contains("requires token_trace routing evidence"))
        XCTAssertTrue(runner.contains("increase --max-trace-tokens"))
        XCTAssertTrue(runner.contains("max_trace_tokens: int = 32768"))
        XCTAssertTrue(runner.contains(#"default=32768"#))
        XCTAssertTrue(runner.contains("BF16/vMLX mask targets unknown MoE layer"))
        XCTAssertTrue(runner.contains("disabled experts were selected"))
        XCTAssertTrue(runner.contains(#""mask_applied""#))
        XCTAssertTrue(runner.contains(#""disabled_expert_count""#))
        XCTAssertTrue(runner.contains("Hooked {hooked_layers} Qwen MoE router layers."))
        XCTAssertTrue(runner.contains("mx.where(available, gates, 0)"))
        XCTAssertTrue(runner.contains(#"parser.add_argument("--mask""#))
        XCTAssertTrue(runner.contains("stream_generate("))
        XCTAssertTrue(main.contains("from .expert_lab_vmlx import register as _register_expert_lab_vmlx"))
        XCTAssertTrue(main.contains(#""expert-lab-vmlx""#))
    }

    func test_expertLabVisualContractKeepsPrimaryPathAndStageRailReadable() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let coordinator = try String(
            contentsOf: wizardRoot.appendingPathComponent("WizardCoordinator.swift"),
            encoding: .utf8
        )
        let source = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/SourceStep.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(coordinator.contains("struct ExpertLabWorkflowStrip"))
        XCTAssertTrue(coordinator.contains(".lineLimit(2)"))
        XCTAssertTrue(coordinator.contains("case \"BF16/vMLX Review\""))
        XCTAssertTrue(coordinator.contains("return \"Legacy\\nBundle\""))
        XCTAssertTrue(coordinator.contains("replacingOccurrences(of: \"/\", with: \"/\\n\")"))
        XCTAssertTrue(coordinator.contains(".fixedSize(horizontal: false, vertical: true)"))
        // PR3: Convert is default; Expert Lab kicker is "Expert Lab path" under that segment.
        XCTAssertTrue(source.contains("Expert Lab path") || source.contains("Workflow"))
        XCTAssertTrue(source.contains("Text(\"Convert\").tag(WizardMode.convert)"))
        XCTAssertTrue(source.contains("Text(\"Expert Lab\").tag(WizardMode.expertLab)"))
        XCTAssertTrue(source.contains("Continue to Profile") || source.contains("goDirectConvert"))
        XCTAssertTrue(source.contains("source stays immutable"))
        XCTAssertTrue(source.contains("prune before quantize"))
        XCTAssertTrue(source.contains("Router-Only Prune"))
        XCTAssertTrue(source.contains("Fallback uses router-row strength only; it does not probe prompts."))
        XCTAssertTrue(source.contains("It cannot unlock final quantization."))
        XCTAssertTrue(source.contains("You can still inspect Expert Lab traces"))
        XCTAssertTrue(source.contains("adoptReviewedPrunedSource(url: prunedURL)"))
        XCTAssertTrue(source.contains("coord.plan.adoptReviewedPrunedSource(url)"))
        XCTAssertTrue(source.contains("enterExpertLabReview") || coordinator.contains("enterExpertLabReview"))
        XCTAssertTrue(source.contains("workflowMode = .convert") || source.contains("setWorkflowMode(.convert)"))
    }

    func test_reviewBundleFailureUsesRecoverableExpertReviewLanguage() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/RunStep.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("Legacy review bundle failed"))
        XCTAssertTrue(src.contains("Legacy review bundle build failed"))
        XCTAssertTrue(src.contains("The legacy review bundle was not written"))
        XCTAssertTrue(src.contains("BF16/vMLX Expert Review can still run from the original source"))
        XCTAssertFalse(src.contains("The temporary review bundle was not written"))
        XCTAssertTrue(src.contains("failureIssueLine"))
        XCTAssertTrue(src.contains(".foregroundStyle(Color.white.opacity(0.84))"))
        XCTAssertTrue(src.contains("Copy Diagnostics"))
    }

    func test_expertLabSheetExposesRecoverableFailureControls() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("ExpertLabSubPanel(title: \"Recovery\""))
        XCTAssertTrue(src.contains("Label(\"Retry\""))
        XCTAssertTrue(src.contains("Label(\"Folder\""))
        XCTAssertTrue(src.contains("copyRecoveryDiagnostics"))
        XCTAssertTrue(src.contains("cleanPartialRun"))
        XCTAssertTrue(src.contains("lastError != nil || partialRunDirectory != nil"))
        XCTAssertTrue(src.contains("failureStage: \"trace_failed\""))
        XCTAssertTrue(src.contains("Partial eval artifacts were saved"))
        XCTAssertTrue(src.contains("Stepper(value: $vm.maxTraceTokens, in: 64...65536, step: 512)"))
        XCTAssertTrue(src.contains("var maxTraceTokens: Int = 32768"))
    }

    func test_expertLabMatchesCompactDockDesignDirection() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let wizardRoot = repoRoot.appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )
        let coordinator = try String(
            contentsOf: wizardRoot.appendingPathComponent("WizardCoordinator.swift"),
            encoding: .utf8
        )
        let design = try String(
            contentsOf: repoRoot.appendingPathComponent("docs/DESIGN_DIRECTION.md"),
            encoding: .utf8
        )

        XCTAssertTrue(design.contains("JANGStudio/demos/design-b-v3.html"))
        XCTAssertTrue(sheet.contains("private var topBar"))
        XCTAssertTrue(sheet.contains("private var dock"))
        XCTAssertTrue(sheet.contains("private var quickProbeBar"))
        XCTAssertTrue(sheet.contains("private func atlasFilterButton"))
        XCTAssertTrue(sheet.contains("private var atlasQueryBar"))
        XCTAssertTrue(sheet.contains("private func atlasFilterField"))
        XCTAssertTrue(sheet.contains("var atlasLayerFilterText"))
        XCTAssertTrue(sheet.contains("var atlasExpertFilterText"))
        XCTAssertTrue(sheet.contains("var atlasDomainFilterText"))
        XCTAssertTrue(sheet.contains("var atlasPromptFilterText"))
        XCTAssertTrue(sheet.contains("var atlasSort: ExpertAtlasSort"))
        XCTAssertTrue(sheet.contains("private enum ExpertAtlasSort"))
        XCTAssertTrue(sheet.contains("private static func matchesIntegerFilter"))
        XCTAssertTrue(sheet.contains("private static func matchesDomainFilter"))
        XCTAssertTrue(sheet.contains("private static func matchesPromptFilter"))
        XCTAssertTrue(sheet.contains("private func sortedAtlasOrder"))
        XCTAssertTrue(sheet.contains("Label(\"Clear filters\""))
        XCTAssertTrue(sheet.contains("private var shouldShowCompareTray"))
        XCTAssertTrue(sheet.contains("private var shouldShowRightWorkflowControls"))
        XCTAssertTrue(sheet.contains(".frame(minWidth: 220, idealWidth: 240, maxWidth: 320)"))
        XCTAssertTrue(sheet.contains(".frame(minWidth: 260, idealWidth: 280, maxWidth: 380)"))
        XCTAssertTrue(sheet.contains("private var embeddedHeader"))
        XCTAssertTrue(sheet.contains("case embeddedInWizard"))
        XCTAssertTrue(sheet.contains("Text(\"Type a prompt to see which experts light up...\""))
        XCTAssertTrue(sheet.contains("domainColor(for entry"))
        XCTAssertTrue(sheet.contains("case .drops: \"Drops\""))
        XCTAssertTrue(sheet.contains("case .locked: \"Locked\""))
        XCTAssertTrue(coordinator.contains("static let canvas = Color(red: 7 / 255"))
        XCTAssertTrue(coordinator.contains("static let line = Color.white.opacity(0.05)"))
    }

    func test_expertLabOpensIdleWithExplicitAtlasPrompt() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )
        let verify = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/VerifyStep.swift"),
            encoding: .utf8
        )
        let coordinator = try String(
            contentsOf: wizardRoot.appendingPathComponent("WizardCoordinator.swift"),
            encoding: .utf8
        )

        XCTAssertFalse(sheet.contains("autoRunOnOpen"))
        XCTAssertFalse(verify.contains("autoRunOnOpen"))
        XCTAssertFalse(coordinator.contains("autoRunOnOpen"))
        XCTAssertTrue(sheet.contains("Run Prompts to Generate the Expert Map"))
        XCTAssertTrue(sheet.contains("Choose a prompt suite, then run the prompts."))
        XCTAssertTrue(sheet.contains("The atlas will appear here when the run finishes."))
    }

    func test_smartExpertReviewStaysEmbeddedInWizard() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )
        let verify = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/VerifyStep.swift"),
            encoding: .utf8
        )
        let coordinator = try String(
            contentsOf: wizardRoot.appendingPathComponent("WizardCoordinator.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(sheet.contains("case embeddedInWizard"))
        XCTAssertTrue(sheet.contains("private var embeddedHeader"))
        XCTAssertTrue(sheet.contains("if presentation == .standalone"))
        XCTAssertTrue(coordinator.contains("presentation: .embeddedInWizard"))
        XCTAssertTrue(verify.contains("coord.active = .expertReview"))
        XCTAssertTrue(verify.contains("if coord.plan.expertReviewIntent == .smartPrequantPrune"))
        XCTAssertTrue(verify.contains("private var standaloneExpertLabSheetBinding"))
        XCTAssertTrue(verify.contains("showingExpertLab && coord.plan.expertReviewIntent != .smartPrequantPrune"))
    }

    func test_finalQuantVerifyShowsPostQuantExpertLabEvidence() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let verify = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/VerifyStep.swift"),
            encoding: .utf8
        )
        let verifier = try String(
            contentsOf: wizardRoot
                .deletingLastPathComponent()
                .appendingPathComponent("Verify/PostConvertVerifier.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(verify.contains("Section(\"Post-Quant Expert Lab\")"))
        XCTAssertTrue(verify.contains("if !busy, isFinalQuantFromReviewedPrune"))
        XCTAssertTrue(verify.contains("postQuantExpertLabSummary"))
        XCTAssertTrue(verify.contains("postQuantPromptCheck"))
        XCTAssertTrue(verify.contains("expertLabFinalReportCheck"))
        XCTAssertTrue(verify.contains("expertLabFinalComparisonCheck"))
        XCTAssertTrue(verify.contains("checks.first { $0.id == .expertLabNativeSmoke }"))
        XCTAssertTrue(verify.contains("checks.first { $0.id == .expertLabFinalReport }"))
        XCTAssertTrue(verify.contains("checks.first { $0.id == .expertLabFinalComparison }"))
        XCTAssertTrue(verify.contains("postQuantPromptCheck?.status == .pass && expertLabFinalComparisonCheck?.status == .pass"))
        XCTAssertTrue(verify.contains("Post-quant same-suite prompt and pruned BF16 comparison passed."))
        XCTAssertTrue(verify.contains("Post-quant final same-suite comparison failed."))
        XCTAssertTrue(verify.contains("Post-quant same-suite prompt verification failed."))
        XCTAssertTrue(verify.contains("Reveal Expert Lab Final Report"))
        XCTAssertTrue(verifier.contains("\"suite_sha256\""))
        XCTAssertTrue(verifier.contains("struct ExpertLabSmokeGenerationSettings"))
        XCTAssertTrue(verifier.contains("case generationSettings = \"generation_settings\""))
        XCTAssertTrue(verifier.contains("temperature: prompt.temperature ?? 0"))
        XCTAssertTrue(verifier.contains("temperature: temperature"))
        XCTAssertTrue(verifier.contains("postQuantGenerationSettingsIssue("))
        XCTAssertTrue(verifier.contains("post-quant reviewed prompt suite is missing decode settings evidence"))
        XCTAssertTrue(verifier.contains("pruned-source generation row is missing decode settings evidence"))
        XCTAssertTrue(verifier.contains("\"post_quant_reviewed_suite_sha256\""))
        XCTAssertTrue(verifier.contains("post-quant reviewed prompt suite fingerprint was not recorded"))
        XCTAssertTrue(verifier.contains("post-quant reviewed prompt suite fingerprint does not match reviewed suite"))
        XCTAssertTrue(verifier.contains("Post-quant prompt suite SHA256"))
        XCTAssertTrue(verifier.contains("private static func fileSHA256(_ url: URL) -> String?"))
    }

    func test_expertLabSupportsLivePromptTracingIntoAtlas() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(sheet.contains("private var quickProbeBar"))
        XCTAssertTrue(sheet.contains("private var livePromptOutputPanel"))
        XCTAssertTrue(sheet.contains("private func livePromptOutputColumn"))
        XCTAssertTrue(sheet.contains("if vm.hasLivePromptOutput"))
        XCTAssertTrue(sheet.contains("var hasLivePromptOutput"))
        XCTAssertTrue(sheet.contains("baselineOutputReady"))
        XCTAssertTrue(sheet.contains("maskedOutputReady"))
        XCTAssertTrue(sheet.contains("|| vm.baselineOutputReady"))
        XCTAssertTrue(sheet.contains("|| vm.maskedOutputReady"))
        XCTAssertTrue(sheet.contains("baselineOutputReady || maskedOutputReady"))
        XCTAssertTrue(sheet.contains("lastEvalSummary.hasPrefix(\"Live prompt\")"))
        XCTAssertTrue(sheet.contains("TextEditor(text: $vm.livePromptText)"))
        XCTAssertTrue(sheet.contains("Type a prompt to see which experts light up..."))
        XCTAssertTrue(sheet.contains("Label(\"Probe\", systemImage: \"bolt.fill\")"))
        XCTAssertTrue(sheet.contains("Original response"))
        XCTAssertTrue(sheet.contains("Model returned an empty response."))
        XCTAssertTrue(sheet.contains("Masked model returned an empty response."))
        XCTAssertTrue(sheet.contains("func runLivePrompt() async"))
        XCTAssertTrue(sheet.contains("domain: \"manual\""))
        XCTAssertTrue(sheet.contains("tags: [\"live\", \"manual\"]"))
        XCTAssertTrue(sheet.contains("ExpertPromptSuite(name: \"Live Prompt\", prompts: [prompt])"))
        XCTAssertTrue(sheet.contains("baselineText = baseline.text"))
        XCTAssertTrue(sheet.contains("baselineOutputReady = true"))
        XCTAssertTrue(sheet.contains("maskedText = masked.text"))
        XCTAssertTrue(sheet.contains("maskedOutputReady = true"))
        XCTAssertTrue(sheet.contains("Live prompt output from the original model. Disable experts, then run Probe again to compare."))
        XCTAssertTrue(sheet.contains("Live prompt A/B:"))
        XCTAssertTrue(sheet.contains("lastEvalDirectory = try persistComparison"))
        XCTAssertTrue(sheet.contains("currentRuntimeMask()"))
        XCTAssertTrue(sheet.contains("blockingRuntimeMaskIssue()"))
        XCTAssertTrue(sheet.contains("private func samplingConfig(for prompt: ExpertPrompt)"))
        XCTAssertTrue(sheet.contains("if let promptMaxTokens = prompt.maxNewTokens, promptMaxTokens > 0"))
        XCTAssertTrue(sheet.contains("if let promptTemperature = prompt.temperature"))
        XCTAssertTrue(sheet.contains("config.temperature = promptTemperature"))
        XCTAssertTrue(sheet.contains("Self.maximumPromptSuiteMaxTokens"))
        XCTAssertTrue(sheet.contains("With disabled experts"))
        XCTAssertTrue(sheet.contains("lastEvalDirectory = nil"))
        XCTAssertTrue(sheet.contains("selectedExpert = nil"))
        XCTAssertTrue(sheet.contains("suite: suite"))
    }

    func test_expertLabKeepsPromptOutputAndCompareLanesVisible() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(sheet.contains("ExpertLabSectionHeader(title: \"Prompt Output\""))
        XCTAssertTrue(sheet.contains("livePromptOutputColumn("))
        XCTAssertTrue(sheet.contains("title: \"Original response\""))
        XCTAssertTrue(sheet.contains("title: vm.hasMask ? \"With disabled experts\" : \"Disabled experts\""))
        XCTAssertTrue(sheet.contains("ExpertLabSectionHeader(title: \"Compare\""))
        XCTAssertTrue(sheet.contains("compareColumn("))
        XCTAssertTrue(sheet.contains("title: \"Original\""))
        XCTAssertTrue(sheet.contains("Run Probe or Compare to see the original output."))
        XCTAssertTrue(sheet.contains("Masked model returned an empty response."))
        XCTAssertTrue(sheet.contains("title: vm.comparisonRiskRows.isEmpty ? \"Eval Evidence\" : \"Eval Regressions\""))
        XCTAssertTrue(sheet.contains("ForEach(vm.comparisonRowsForDisplay.prefix(4), id: \\.promptID)"))
        XCTAssertTrue(sheet.contains("comparisonRowsOmittedCount"))
        XCTAssertTrue(sheet.contains("Text(vm.comparisonEvidenceSummary)"))
        XCTAssertTrue(sheet.contains("Text(vm.comparisonArtifactSummary)"))
        XCTAssertTrue(sheet.contains("Eval artifacts:"))
        XCTAssertTrue(sheet.contains("severity \\(severity)"))
        XCTAssertTrue(sheet.contains("eval.jsonl + eval_trace.jsonl + eval_index.json"))
        XCTAssertTrue(sheet.contains("baselineLayerStats: record.baseline.layerStats"))
        XCTAssertTrue(sheet.contains("maskedLayerStats: record.masked.layerStats"))
        XCTAssertTrue(sheet.contains("baselineLayerStatsPromptCount: Self.layerStatsPromptCount"))
        XCTAssertTrue(sheet.contains("maskedLayerStatsPromptCount: Self.layerStatsPromptCount"))
        XCTAssertTrue(sheet.contains("case baselineLayerStatsPromptCount = \"baseline_layer_stats_prompt_count\""))
        XCTAssertTrue(sheet.contains("case maskedLayerStatsPromptCount = \"masked_layer_stats_prompt_count\""))
        XCTAssertTrue(sheet.contains("row.resolvedRegressionSeverity"))
        XCTAssertTrue(sheet.contains("ExpertLabSectionHeader(title: \"Comparison Impact\""))
        XCTAssertTrue(sheet.contains("selectedExpertReviewStatus(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertComparedMaskStatus(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertSafeDropStatus(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertMaskedImpactSummary(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertRegressionSeveritySummary(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertRegressionSeverityIsHigh(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertPassRateSummary(for: entry)"))
        XCTAssertTrue(sheet.contains("selectedExpertRiskSummary(for: entry)"))
        XCTAssertTrue(sheet.contains("Mask-level evidence, not single-expert ablation."))
        XCTAssertTrue(sheet.contains("drop candidate / user-forced drop"))
        XCTAssertTrue(sheet.contains("same-suite safe-drop"))
        XCTAssertTrue(sheet.contains("case safeDrop"))
        XCTAssertTrue(sheet.contains("isSafeDropCandidate(layer: entry.layer, expert: entry.expert)"))
        XCTAssertTrue(sheet.contains("safeDrop ? ExpertLabVisual.good"))
        XCTAssertTrue(sheet.contains("regression_severity"))
        XCTAssertTrue(sheet.contains("avg tokens"))
        XCTAssertTrue(sheet.contains("token depth missing"))
        XCTAssertTrue(sheet.contains("eval_index.json"))
        XCTAssertTrue(sheet.contains("Text(vm.runtimeInfoSummary)"))
        XCTAssertTrue(sheet.contains("runtimeSummary(from info: JANGKit.ModelRuntimeInfo)"))
        XCTAssertTrue(sheet.contains("runtimeSummary(from info: ExpertLabVMLXRuntimeInfo)"))
        XCTAssertTrue(sheet.contains("expectedSourcePath: modelPath.path"))
        XCTAssertTrue(sheet.contains("maskRequired: mask != nil"))
        XCTAssertTrue(sheet.contains("ExpertLabVMLXRuntimeEvidenceValidator.issue"))
        XCTAssertTrue(sheet.contains("runtimeBackend: runtime.backend"))
        XCTAssertTrue(sheet.contains("did not record the vMLX backend"))
        XCTAssertTrue(sheet.contains("recorded incomplete routed-layer hook coverage"))
        XCTAssertTrue(sheet.contains("source model path does not match the selected BF16/F16 source"))
        XCTAssertTrue(sheet.contains("did not record an applied BF16/vMLX mask"))
        XCTAssertTrue(sheet.contains("hookedMOELayers"))
        XCTAssertTrue(sheet.contains("expectedMOELayers"))
        XCTAssertTrue(sheet.contains("hookCoverageComplete"))
        XCTAssertTrue(sheet.contains("maskApplied"))
        XCTAssertTrue(sheet.contains("disabledExpertCount"))
        XCTAssertTrue(sheet.contains("sourceModelPath"))
    }

    func test_expertLabReopensPersistedRunWithEvalAndOutputVisible() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(sheet.contains("restoreGenerationPreview(from: dir)"))
        XCTAssertTrue(sheet.contains("restoreLatestComparison(from: dir)"))
        XCTAssertTrue(sheet.contains("restoredRunEvidenceIssue(from: dir, suite: loadedSuite)"))
        XCTAssertTrue(sheet.contains("Loaded run, but \\(issue)"))
        XCTAssertTrue(sheet.contains("generations.jsonl contains duplicate prompt IDs"))
        XCTAssertTrue(sheet.contains("suite.jsonl contains duplicate prompt IDs"))
        XCTAssertTrue(sheet.contains("private static func restoredRuns(from runDirectory: URL, suite: ExpertPromptSuite) -> [ExpertPromptRun]"))
        XCTAssertTrue(sheet.contains("private static func restoredRunEvidenceIssue(from runDirectory: URL, suite: ExpertPromptSuite) -> String?"))
        XCTAssertTrue(sheet.contains("jsonlRecords(StoredGenerationRecord.self, from: generationURL)"))
        XCTAssertTrue(sheet.contains("jsonlRecords(StoredTraceRecord.self, from: traceURL)"))
        XCTAssertTrue(sheet.contains("layerStats: restoredLayerStats(from: generation, traces: traces)"))
        XCTAssertTrue(sheet.contains("private static func restoredLayerStats("))
        XCTAssertTrue(sheet.contains("if let layerStats = generation.layerStats, !layerStats.isEmpty"))
        XCTAssertTrue(sheet.contains("tokenTrace: traces.isEmpty ? nil : traces"))
        XCTAssertTrue(sheet.contains("private static func layerStats(from traces: [JANGKit.ExpertRouteRecord]) -> [JANGKit.ExpertLayerStats]"))
        XCTAssertTrue(sheet.contains("loadMaskIfPresent(from: dir)\n            restoreLatestComparison(from: dir)"))
        XCTAssertTrue(sheet.contains("selectedRunArtifactSummary"))
        XCTAssertTrue(sheet.contains("selectedRunArtifactEvidenceSummary(runDirectory: runDirectory)"))
        XCTAssertTrue(sheet.contains("evalIndexSemanticCoverageSummary("))
        XCTAssertTrue(sheet.contains("semantic probes ready"))
        XCTAssertTrue(sheet.contains("semantic probes missing; rerun Compare Suite"))
        XCTAssertTrue(sheet.contains("let derivedMissing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains"))
        XCTAssertTrue(sheet.contains("let recordedMissing = Set("))
        XCTAssertTrue(sheet.contains("let missing = derivedMissing.union(recordedMissing)"))
        XCTAssertTrue(sheet.contains("run.json"))
        XCTAssertTrue(sheet.contains("suite.jsonl \\(lineCount(suiteURL)) prompts"))
        XCTAssertTrue(sheet.contains("trace.jsonl \\(lineCount(traceURL)) routes"))
        XCTAssertTrue(sheet.contains("comparison_summary.json"))
        XCTAssertTrue(sheet.contains("eval_trace.jsonl \\(lineCount(evalTraceURL)) routes"))
        XCTAssertTrue(sheet.contains("mask.json"))
        XCTAssertTrue(sheet.contains("StoredGenerationRecord"))
        XCTAssertTrue(sheet.contains("let layerStats: [JANGKit.ExpertLayerStats]?"))
        XCTAssertTrue(sheet.contains("let jangToolsVersion: String?"))
        XCTAssertTrue(sheet.contains("jangToolsVersion: first.jangToolsVersion"))
        XCTAssertTrue(sheet.contains("jangToolsVersion: run.jangToolsVersion"))
        XCTAssertTrue(sheet.contains("generations.jsonl"))
        XCTAssertTrue(sheet.contains("latestComparisonDirectory(in:"))
        XCTAssertTrue(sheet.contains("comparisonMaskMatchesCurrent(latest)"))
        XCTAssertTrue(sheet.contains("Latest saved eval was for a different mask. Rerun Compare Suite before pruning."))
        XCTAssertTrue(sheet.contains("private func comparisonMaskMatchesCurrent(_ directory: URL) -> Bool"))
        XCTAssertTrue(sheet.contains("let saved = try? JSONDecoder().decode(JANGKit.ExpertMask.self"))
        XCTAssertTrue(sheet.contains("comparisonMaskMatchesCurrent(lastEvalDirectory)"))
        XCTAssertTrue(sheet.contains("StoredEvalRecord.self"))
        XCTAssertTrue(sheet.contains("eval.jsonl"))
        XCTAssertTrue(sheet.contains("eval_index.json"))
        XCTAssertTrue(sheet.contains("StoredEvalIndex("))
        XCTAssertTrue(sheet.contains("jang-expert-lab-eval-index-v1"))
        XCTAssertTrue(sheet.contains("semanticCoverage: semanticCoverage"))
        XCTAssertTrue(sheet.contains("missingSemanticCoverage: Self.missingSemanticCoverage(for: semanticCoverage)"))
        XCTAssertTrue(sheet.contains("case semanticCoverage = \"semantic_coverage\""))
        XCTAssertTrue(sheet.contains("case missingSemanticCoverage = \"missing_semantic_coverage\""))
        XCTAssertTrue(sheet.contains("sourceModelPath: runtimeInfo?.sourceModelPath"))
        XCTAssertTrue(sheet.contains("hookedMOELayers: runtimeInfo?.hookedMOELayers"))
        XCTAssertTrue(sheet.contains("expectedMOELayers: runtimeInfo?.expectedMOELayers"))
        XCTAssertTrue(sheet.contains("hookCoverageComplete: runtimeInfo?.hookCoverageComplete"))
        XCTAssertTrue(sheet.contains("sourceModelPath: first.sourceModelPath"))
        XCTAssertTrue(sheet.contains("hookedMOELayers: first.hookedMOELayers"))
        XCTAssertTrue(sheet.contains("expectedMOELayers: first.expectedMOELayers"))
        XCTAssertTrue(sheet.contains("hookCoverageComplete: first.hookCoverageComplete"))
        XCTAssertTrue(sheet.contains("sourceModelPath: run.sourceModelPath"))
        XCTAssertTrue(sheet.contains("hookedMOELayers: run.hookedMOELayers"))
        XCTAssertTrue(sheet.contains("expectedMOELayers: run.expectedMOELayers"))
        XCTAssertTrue(sheet.contains("hookCoverageComplete: run.hookCoverageComplete"))
        XCTAssertTrue(sheet.contains("runtimeEvidenceParts("))
        XCTAssertTrue(sheet.contains("source \\(URL(fileURLWithPath: sourceModelPath).lastPathComponent)"))
        XCTAssertTrue(sheet.contains("MoE layers \\(coverage)"))
        XCTAssertTrue(sheet.contains("maskApplied: runtimeInfo?.maskApplied"))
        XCTAssertTrue(sheet.contains("disabledExpertCount: runtimeInfo?.disabledExpertCount"))
        XCTAssertTrue(sheet.contains("comparisonRuntimeInfo(record)"))
        XCTAssertTrue(sheet.contains("record.masked.runtimeInfo?.maskApplied == true"))
        XCTAssertTrue(sheet.contains("backfillEvalIndexIfNeeded("))
        XCTAssertTrue(sheet.contains("private static func writeEvalIndex("))
        XCTAssertTrue(sheet.contains("Loaded eval, but could not rebuild eval_index.json"))
        XCTAssertTrue(sheet.contains("comparisonPreviewRows = Self.jsonlRecords(StoredEvalRecord.self"))
        XCTAssertTrue(sheet.contains("comparisonPreviewRows = Self.storedEvalRecords(from: records)"))
        XCTAssertTrue(sheet.contains("lastEvalSummary = Self.suiteComparisonProgressSummary("))
        XCTAssertTrue(sheet.contains("latestPromptID: prompt.id"))
        XCTAssertTrue(sheet.contains("Comparing suite: %d/%d prompts complete, latest %@"))
        XCTAssertTrue(sheet.contains("severity %@, %d regression row%@ visible"))
        XCTAssertTrue(sheet.contains("speed %.2f/%.2f t/s"))
        XCTAssertTrue(sheet.contains("regression row%@ visible"))
        XCTAssertTrue(sheet.contains("statusText = Self.traceProgressSummary("))
        XCTAssertTrue(sheet.contains("Traced %d/%d prompts, latest %@"))
        XCTAssertTrue(sheet.contains("baselineText = latest.result.text"))
        XCTAssertTrue(sheet.contains("var partialEvalDirectory: URL?"))
        XCTAssertTrue(sheet.contains("partialEvalDirectory = try persistComparison("))
        XCTAssertTrue(sheet.contains("directory: partialEvalDirectory"))
        XCTAssertTrue(sheet.contains("directory existingDirectory: URL?"))
        XCTAssertTrue(sheet.contains("if let existingDirectory"))
        XCTAssertTrue(sheet.contains("baselineOutputReady = true"))
        XCTAssertTrue(sheet.contains("maskedOutputReady = true"))
        XCTAssertTrue(sheet.contains("runtimeInfoSummary = Self.runtimeSummary("))
        XCTAssertTrue(sheet.contains("runtime_metal_enabled"))
        XCTAssertTrue(sheet.contains("reviewRuntimeTargetSummary"))
        XCTAssertTrue(sheet.contains("BF16/vMLX source: \\(modelPath.lastPathComponent)"))
        XCTAssertTrue(sheet.contains("Legacy review bundle: \\(modelPath.lastPathComponent)"))
        XCTAssertTrue(sheet.contains("qwenVMLXConfigIssue"))
        XCTAssertTrue(sheet.contains("shared_expert_intermediate_size"))
        XCTAssertTrue(sheet.contains("vMLX config incomplete"))
        XCTAssertTrue(sheet.contains("compact UI fixture"))
        XCTAssertTrue(sheet.contains("BF16/F16 authority"))
        XCTAssertTrue(sheet.contains("Original BF16/F16 source runs through vMLX/mlx_lm"))
        XCTAssertTrue(sheet.contains("ExpertLabStatRow(label: \"Entropy\""))
        XCTAssertTrue(sheet.contains("ExpertLabStatRow(label: \"Depth\""))
        XCTAssertTrue(sheet.contains("func tokenDepthSummary(for entry: ExpertAtlasEntry) -> String"))
        XCTAssertTrue(sheet.contains("ExpertLabStatRow(label: \"Lift\""))
        XCTAssertTrue(sheet.contains("ExpertLabStatRow(label: \"Evidence\""))
        XCTAssertTrue(sheet.contains("func evidenceCount(for entry: ExpertAtlasEntry) -> Int"))
        XCTAssertTrue(sheet.contains("ExpertLabSectionHeader(title: \"Prompt Evidence\")"))
        XCTAssertTrue(sheet.contains("entry.topPrompts"))
        XCTAssertTrue(sheet.contains("let config = samplingConfig(for: prompt)"))
        XCTAssertTrue(sheet.contains("Comparison gate: per-prompt token counts are missing"))
        XCTAssertTrue(sheet.contains("Comparison gate: per-prompt routing records are missing"))
        XCTAssertTrue(sheet.contains("Comparison gate: per-prompt routed-layer stats"))
        XCTAssertTrue(sheet.contains("Comparison gate: runtime device evidence is missing"))
        XCTAssertTrue(sheet.contains("Comparison gate: masked compare did not record a Metal runtime"))
        XCTAssertTrue(sheet.contains("Comparison gate: vMLX package version evidence is missing"))
        XCTAssertTrue(sheet.contains("Comparison gate: eval rows did not record the BF16/vMLX runtime"))
        XCTAssertTrue(sheet.contains("Comparison gate: eval rows did not record the vMLX backend"))
        XCTAssertTrue(sheet.contains("Comparison gate: per-prompt vMLX routed-layer hook evidence is missing"))
        XCTAssertTrue(sheet.contains("Comparison gate: vMLX routed-layer hook coverage is incomplete"))
        XCTAssertTrue(sheet.contains("Comparison gate: vMLX hook coverage does not cover every routed layer"))
        XCTAssertTrue(sheet.contains("Comparison gate: eval rows are missing source model path evidence"))
        XCTAssertTrue(sheet.contains("Comparison gate: masked compare did not record an applied BF16/vMLX mask"))
        XCTAssertTrue(sheet.contains("Comparison gate: reviewed prune requires disabled expert evidence in every masked eval row"))
        XCTAssertTrue(sheet.contains("runtime device, vMLX hook coverage, and plan safety gates passed"))
        XCTAssertTrue(sheet.contains("runtimeMode: runtimeRecord?.runtimeMode"))
        XCTAssertTrue(sheet.contains("runtimeMetalEnabled: runtimeRecord?.runtimeMetalEnabled"))
        XCTAssertTrue(sheet.contains("jangToolsVersion: runtimeRecord?.jangToolsVersion"))
        XCTAssertTrue(sheet.contains("Comparison gate: average generated depth is"))
        XCTAssertTrue(sheet.contains("minimumReviewedPruneMeanTokens"))
        XCTAssertTrue(sheet.contains("Loaded suite eval:"))
        XCTAssertTrue(sheet.contains("Saved prune plan is legacy and missing same-suite comparison evidence"))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing routing record evidence for every indexed prompt."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing eval_trace.jsonl evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing decode settings evidence."))
        XCTAssertTrue(sheet.contains("eval.jsonl is missing per-prompt runtime device evidence."))
        XCTAssertTrue(sheet.contains("eval.jsonl is missing per-prompt source model path evidence."))
        XCTAssertTrue(sheet.contains("eval.jsonl did not record an applied BF16/vMLX mask."))
        XCTAssertTrue(sheet.contains("eval.jsonl is missing per-prompt regression flag evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval.jsonl is missing baseline/masked decode settings evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval.jsonl baseline/masked decode settings do not match."))
        XCTAssertTrue(sheet.contains("evalIndexSemanticCoverageIssue("))
        XCTAssertTrue(sheet.contains("is missing semantic coverage evidence."))
        XCTAssertTrue(sheet.contains("records missing semantic prompt probes:"))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: same-suite comparison found no safe drop candidates."))
        XCTAssertTrue(sheet.contains("Saved prune plan is legacy and missing per-prompt eval_index evidence"))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing generation-depth evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing runtime device evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index did not record a Metal runtime."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index did not record the BF16/vMLX runtime."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index did not record the vMLX backend."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing vMLX routed-layer hook evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index recorded incomplete vMLX routed-layer hook coverage."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index vMLX hook coverage"))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing vMLX package version evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing source model path evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index did not record an applied BF16/vMLX mask."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index did not record disabled expert evidence"))
        XCTAssertTrue(sheet.contains("selectedRunSuiteFingerprintIssue("))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index is missing suite.jsonl fingerprint evidence."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index suite.jsonl fingerprint does not match suite.jsonl."))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: eval_index\""))
        XCTAssertTrue(sheet.contains("ExpertLabLayerStatsEvidenceValidator.issue"))
        XCTAssertTrue(sheet.contains("var comparisonRiskRows: [StoredEvalRecord]"))
        XCTAssertTrue(sheet.contains("var isRisky: Bool"))
        XCTAssertTrue(sheet.contains("let semanticDomains = comparisonSemanticDomains(for: record.prompt)"))
        XCTAssertTrue(sheet.contains("semanticDomains: semanticDomains"))
        XCTAssertTrue(sheet.contains("private static func highRiskDomains(from records: [ExpertComparisonPromptRecord]) -> [String]"))
        XCTAssertTrue(sheet.contains("private static func highRiskDomains(from records: [StoredEvalRecord]) -> [String]"))
        XCTAssertTrue(sheet.contains("private static func semanticCoverage(from records: [StoredEvalRecord]) -> [String]"))
        XCTAssertTrue(sheet.contains("private static func missingSemanticCoverage(for semanticCoverage: [String]) -> [String]"))
        XCTAssertTrue(sheet.contains("var semanticDomainsForRisk: [String]"))
        XCTAssertTrue(sheet.contains("ExpertDomainTaxonomy.canonicalSemanticDomain(domain)"))
        XCTAssertTrue(sheet.contains("var reviewedPrunePromptCount: Int"))
        XCTAssertTrue(sheet.contains("let prompts = reviewedPrunePromptsForCoverage()"))
        XCTAssertTrue(sheet.contains("private func reviewedPrunePromptsForCoverage() -> [ExpertPrompt]"))
        XCTAssertTrue(sheet.contains("let suiteURL = lastRunDirectory.appendingPathComponent(\"suite.jsonl\")"))
        XCTAssertTrue(sheet.contains("return suite.prompts"))
        XCTAssertTrue(sheet.contains("var reviewedPruneDomainCount: Int"))
        XCTAssertTrue(sheet.contains("Set(reviewedPrunePromptsForCoverage().map(\\.domain)).count"))
        XCTAssertTrue(sheet.contains("ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains"))
        XCTAssertTrue(sheet.contains("var reviewedPruneSemanticDomains: Set<String>"))
        XCTAssertTrue(sheet.contains("reviewedPrunePromptsForCoverage().flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) }"))
        XCTAssertTrue(sheet.contains("if reviewedPrunePromptCount < Self.minimumReviewedPrunePromptCount"))
        XCTAssertTrue(sheet.contains("if reviewedPruneDomainCount < Self.minimumReviewedPruneDomainCount"))
        XCTAssertTrue(sheet.contains("Coverage gate: prompt suite contains empty prompt IDs"))
        XCTAssertTrue(sheet.contains("Coverage gate: prompt suite contains duplicate prompt IDs"))
        XCTAssertTrue(sheet.contains("Coverage gate: include required semantic prompt probes before reviewed prune planning"))
    }

    func test_expertLabAtlasUsesConfiguredFullBundleDimensions() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let sheet = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(sheet.contains("let expectedLayers: Int?"))
        XCTAssertTrue(sheet.contains(#"intValue(textConfig, keys: ["num_hidden_layers", "n_layer", "num_layers"])"#))
        XCTAssertTrue(sheet.contains("configuredExpertsByLayer(observedLayers:"))
        XCTAssertTrue(sheet.contains("return Dictionary(uniqueKeysWithValues: (0..<expectedLayers).map { ($0, expectedExperts) })"))
        XCTAssertTrue(sheet.contains("atlas = completedAtlas("))
        XCTAssertTrue(sheet.contains("reviewedPruneAtlasIssues"))
        XCTAssertTrue(sheet.contains("Atlas gate: full source expert grid is incomplete"))
        XCTAssertTrue(sheet.contains("Atlas gate: atlas contains cells outside the source expert grid"))
        XCTAssertTrue(sheet.contains("let decodedAtlas = try decoder.decode(ExpertAtlas.self, from: atlasData)"))
        XCTAssertTrue(sheet.contains("let loadedAtlas = completedAtlas(decodedAtlas)"))
        XCTAssertTrue(sheet.contains("try persistAtlas(loadedAtlas, to: dir)"))
        XCTAssertTrue(sheet.contains("Loaded run, but could not persist completed atlas"))
        XCTAssertTrue(sheet.contains("private func persistAtlas(_ atlas: ExpertAtlas, to runDirectory: URL) throws"))
        XCTAssertTrue(sheet.contains("selectedRunEvidenceSummary"))
        XCTAssertTrue(sheet.contains("refreshSelectedRunEvidence()"))
        XCTAssertTrue(sheet.contains("Text(vm.selectedRunEvidenceSummary)"))
        XCTAssertTrue(sheet.contains("matchingSourcePath: artifactSourcePath"))
        XCTAssertTrue(sheet.contains("reviewBundlePath: artifactReviewBundlePath"))
        XCTAssertFalse(sheet.contains("matchingSourcePath: sourceModelPath?.path"))
        XCTAssertTrue(sheet.contains("let completed = completedAtlas(decodedAtlas)"))
        XCTAssertTrue(sheet.contains("let sourceGrid = Self.intLayerMap(completed.sourceNumExpertsByLayer)"))
        XCTAssertTrue(sheet.contains("if let shape = Self.uniformGridShape(from: sourceGrid)"))
        XCTAssertTrue(sheet.contains("sourceNumExpertsByLayer: current.sourceNumExpertsByLayer"))
        XCTAssertTrue(sheet.contains("sourceNumExpertsByLayer: Self.stringLayerMap(expected)"))
        XCTAssertTrue(sheet.contains("selectedRunEvidenceSummary = \"\\(base) · \\(gridState): \\(layers) layers x \\(experts) experts"))
        XCTAssertTrue(sheet.contains("selectedRunEvidenceSummary = \"\\(base) · \\(gridState): \\(shape.layers) layers x \\(shape.experts) experts"))
        XCTAssertTrue(sheet.contains("layers x \\(experts) experts"))
        XCTAssertTrue(sheet.contains("complete grid"))
        XCTAssertTrue(sheet.contains("incomplete grid"))
        XCTAssertTrue(sheet.contains("\\(actualCells)/\\(expectedCells) cells"))
        XCTAssertTrue(sheet.contains("selectedRunComparisonEvidenceSummary(runDirectory:"))
        XCTAssertTrue(sheet.contains("latestComparisonDirectory(in: evalsDir)"))
        XCTAssertTrue(sheet.contains("selectedRunRuntimeEvidenceSummary"))
        XCTAssertTrue(sheet.contains("device not recorded"))
        XCTAssertTrue(sheet.contains("\"compare \\(summary.promptCount) prompts\""))
        XCTAssertTrue(sheet.contains("\"baseline \\(formatPassRate(summary.passRateBaseline))\""))
        XCTAssertTrue(sheet.contains("\"masked \\(formatPassRate(summary.passRateMasked))\""))
        XCTAssertTrue(sheet.contains("\"avg tokens %.1f/%.1f\""))
        XCTAssertTrue(sheet.contains("token depth missing; rerun Compare Suite"))
        XCTAssertTrue(sheet.contains("\"eval runtime \\(runtime)\""))
        XCTAssertTrue(sheet.contains("eval device not recorded"))
        XCTAssertTrue(sheet.contains("eval runtime not recorded; rerun Compare Suite"))
        XCTAssertTrue(sheet.contains("eval_index missing; rerun Compare Suite"))
        XCTAssertTrue(sheet.contains("risk \\(index.highRiskDomains.sorted().joined(separator: \", \"))"))
        XCTAssertTrue(sheet.contains("selectedRunEvidenceWarning"))
        XCTAssertTrue(sheet.contains("selectedRunAuthorityWarning(selectedRunSummary)"))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: runtime is"))
        XCTAssertTrue(sheet.contains("not BF16/vMLX authority"))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: vMLX backend evidence is missing or not authoritative."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: runtime device evidence is missing."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: Metal runtime evidence is missing."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: vMLX package version evidence is missing."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: runtime source path evidence is missing."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: vMLX routed-layer hook evidence is missing."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: vMLX routed-layer hook coverage is incomplete."))
        XCTAssertTrue(sheet.contains("Saved run is inspectable only: vMLX hook coverage"))
        XCTAssertTrue(sheet.contains("selectedRunPrunePlanWarning(runDirectory:"))
        XCTAssertTrue(sheet.contains("Saved prune plan is legacy and missing top-k safety evidence"))
        XCTAssertTrue(sheet.contains("Saved prune plan is blocked: trained top-k evidence is missing."))
        XCTAssertTrue(sheet.contains("Label(vm.selectedRunEvidenceWarning, systemImage: \"exclamationmark.triangle.fill\")"))
    }

    func test_expertLabPrimaryPrunePathRequiresCurrentComparison() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("A/B comparison required before reviewed prune plan."))
        XCTAssertTrue(src.contains("canGenerateReviewedPrunePlan"))
        XCTAssertTrue(src.contains("Label(vm.reviewedPruneReadiness.message"))
        XCTAssertTrue(src.contains("var reviewedPruneReadiness: (message: String, systemImage: String, isReady: Bool)"))
        XCTAssertTrue(src.contains("Reviewed prune ready: BF16/vMLX authority, coverage, semantic evidence, A/B-safe drops, masked compare, token depth, runtime device, vMLX hook coverage, and plan safety gates passed."))
        XCTAssertTrue(src.contains("reviewedPruneAuthorityIssues"))
        XCTAssertTrue(src.contains("Authority gate: reviewed pruning must start from the original BF16/F16 source through BF16/vMLX"))
        XCTAssertTrue(src.contains("Authority gate: original BF16/F16 source path is missing"))
        XCTAssertTrue(src.contains("loadedRunAuthorityIssue()"))
        XCTAssertTrue(src.contains("Authority gate: loaded run was captured with"))
        XCTAssertTrue(src.contains("Authority gate: loaded run did not record the vMLX backend"))
        XCTAssertTrue(src.contains("Authority gate: loaded run is missing runtime device evidence"))
        XCTAssertTrue(src.contains("Authority gate: loaded run is missing Metal runtime evidence"))
        XCTAssertTrue(src.contains("Authority gate: loaded run is missing vMLX package version evidence"))
        XCTAssertTrue(src.contains("Authority gate: loaded run still references a legacy review bundle"))
        XCTAssertTrue(src.contains("Authority gate: loaded run is missing runtime source path evidence"))
        XCTAssertTrue(src.contains("Authority gate: loaded run runtime source path does not match the selected BF16/F16 source"))
        XCTAssertTrue(src.contains("Authority gate: loaded run is missing vMLX routed-layer hook evidence"))
        XCTAssertTrue(src.contains("Authority gate: loaded run recorded incomplete vMLX routed-layer hook coverage"))
        XCTAssertTrue(src.contains("Authority gate: loaded run vMLX hook coverage"))
        XCTAssertTrue(src.contains("private static func runtimeModeDisplayName(_ mode: String) -> String"))
        XCTAssertTrue(src.contains("ForEach(vm.reviewedPruneAtlasIssues.prefix(3)"))
        XCTAssertTrue(src.contains("&& reviewedPruneAtlasIssues.isEmpty"))
        XCTAssertTrue(src.contains("+ reviewedPruneAtlasIssues"))
        XCTAssertTrue(src.contains("Atlas gate: source expert grid is missing"))
        XCTAssertTrue(src.contains("Atlas gate: source expert grid covers"))
        XCTAssertTrue(src.contains("Atlas gate: source expert grid does not match the configured"))
        XCTAssertTrue(src.contains("Reviewed prune waiting: run a prompt suite trace to build the full expert atlas."))
        XCTAssertTrue(src.contains("Reviewed prune blocked: run Compare Suite so baseline and masked outputs are saved for the traced prompts."))
        XCTAssertTrue(src.contains("Reviewed prune blocked: \\(firstIssue)"))
        XCTAssertTrue(src.contains("missingComparison"))
        XCTAssertTrue(src.contains("reviewedPruneCoverageIssues"))
        XCTAssertTrue(src.contains("reviewedPruneSemanticEvidenceIssues"))
        XCTAssertTrue(src.contains("ForEach(vm.reviewedPruneSemanticEvidenceIssues.prefix(3)"))
        XCTAssertTrue(src.contains("Semantic evidence gate: gate mass is missing"))
        XCTAssertTrue(src.contains("Semantic evidence gate: masked-output impact evidence is missing or invalid"))
        XCTAssertTrue(src.contains("Semantic evidence gate: activation lift is missing"))
        XCTAssertTrue(src.contains("Semantic evidence gate: prompt examples are missing"))
        XCTAssertTrue(src.contains("Semantic evidence gate: prompt tags/examples are incomplete"))
        XCTAssertTrue(src.contains("private static func expertCoordinatePreview"))
        XCTAssertTrue(src.contains("reviewedPruneComparisonIssues"))
        XCTAssertTrue(src.contains("minimumReviewedPrunePromptCount = 50"))
        XCTAssertTrue(src.contains("minimumReviewedPruneDomainCount = 6"))
        XCTAssertTrue(src.contains("Comparison gate: compare at least"))
        XCTAssertTrue(src.contains("Comparison gate: rerun A/B compare for all traced prompts"))
        XCTAssertTrue(src.contains("Comparison gate: masked validator pass rate"))
        XCTAssertTrue(src.contains("Comparison gate: masked outputs regressed in high-risk domains"))
        XCTAssertTrue(src.contains("Comparison gate: A/B comparison found no safe drop candidates"))
        XCTAssertTrue(src.contains("Treat this as a valid review outcome"))
        XCTAssertTrue(src.contains("Comparison gate: per-prompt eval rows are missing"))
        XCTAssertTrue(src.contains("Comparison gate: per-prompt eval rows cover"))
        XCTAssertTrue(src.contains("currentComparisonArtifactIntegrityIssue(comparisonSummary: summary)"))
        XCTAssertTrue(src.contains("jsonlObjects(from: evalURL)"))
        XCTAssertTrue(src.contains("evalRowEvidenceIssue("))
        XCTAssertTrue(src.contains("sourceMismatchIssue: \"Comparison gate: eval.jsonl source model path does not match the reviewed BF16/F16 source.\""))
        XCTAssertTrue(src.contains("Comparison gate: eval.jsonl prompt IDs are unreadable."))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt runtime device evidence."))
        XCTAssertTrue(src.contains("Comparison gate: eval.jsonl source model path does not match the reviewed BF16/F16 source."))
        XCTAssertTrue(src.contains("Comparison gate: eval.jsonl is unreadable."))
        XCTAssertTrue(src.contains("Comparison gate: mask.json is unreadable."))
        XCTAssertTrue(src.contains("Comparison gate: mask.json does not match the selected BF16/vMLX mask."))
        XCTAssertTrue(src.contains("Comparison gate: mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning."))
        XCTAssertTrue(src.contains("Comparison gate: mask.json disabled expert count does not match eval.jsonl."))
        XCTAssertTrue(src.contains("currentComparisonSummaryIntegrityIssue("))
        XCTAssertTrue(src.contains("high-risk domains do not match eval.jsonl."))
        XCTAssertTrue(src.contains("safe-drop candidates do not match the current mask and eval rows."))
        XCTAssertTrue(src.contains("Comparison gate: eval_index.json is unreadable."))
        XCTAssertTrue(src.contains("Comparison gate: eval_index.json is missing reviewed prompt suite fingerprint."))
        XCTAssertTrue(src.contains("Comparison gate: eval_index.json reviewed prompt suite fingerprint does not match loaded suite."))
        XCTAssertTrue(src.contains("evidenceName: \"Comparison gate: eval_index.json\""))
        XCTAssertTrue(src.contains("Comparison gate: eval_index.json semantic coverage does not match eval.jsonl."))
        XCTAssertTrue(src.contains("Comparison gate: eval_index.json routing record counts do not match eval.jsonl."))
        XCTAssertTrue(src.contains("Comparison gate: eval_index.json does not point to the persisted same-suite eval artifacts."))
        XCTAssertTrue(src.contains("persistedIndex.maskJSON == \"mask.json\""))
        XCTAssertTrue(src.contains("strictJSONLRecords(StoredEvalRecord.self"))
        XCTAssertTrue(src.contains("currentComparisonTraceIntegrityIssue(comparisonSummary: summary)"))
        XCTAssertTrue(src.contains("issuePrefix: \"Comparison gate\""))
        XCTAssertTrue(src.contains("expectedDisabledByLayer: expectedMask.layers"))
        XCTAssertTrue(src.contains("traceRowDisabledSelectionIssue("))
        XCTAssertTrue(src.contains(": eval_trace.jsonl masked routing records are missing mask evidence for prompt IDs:"))
        XCTAssertTrue(src.contains(": eval_trace.jsonl masked routing records are missing mask.json evidence for prompt"))
        XCTAssertTrue(src.contains("eval_trace.jsonl masked routing records selected disabled experts for prompt"))
        XCTAssertTrue(src.contains(": eval_trace.jsonl has no routing records."))
        XCTAssertTrue(src.contains(": eval_trace.jsonl missing masked routing records"))
        XCTAssertTrue(src.contains("Comparison gate: eval rows did not record the vMLX backend"))
        XCTAssertTrue(src.contains("Comparison gate: per-prompt vMLX routed-layer hook evidence is missing"))
        XCTAssertTrue(src.contains("Comparison gate: vMLX routed-layer hook coverage is incomplete"))
        XCTAssertTrue(src.contains("Comparison gate: vMLX hook coverage does not cover every routed layer"))
        XCTAssertTrue(src.contains("comparisonPromptIDCoverageIssue()"))
        XCTAssertTrue(src.contains("Comparison gate: traced prompt suite contains duplicate prompt IDs"))
        XCTAssertTrue(src.contains("Comparison gate: compared eval rows contain duplicate prompt IDs"))
        XCTAssertTrue(src.contains("private static func firstDuplicatePromptID(in ids: [String]) -> String?"))
        XCTAssertTrue(src.contains("compared prompt IDs do not match the traced suite"))
        XCTAssertTrue(src.contains("tracedPromptIDsForReview()"))
        XCTAssertTrue(src.contains("lastRunDirectory.appendingPathComponent(\"suite.jsonl\")"))
        XCTAssertTrue(src.contains("Rerun Compare Suite for this trace run."))
        XCTAssertTrue(src.contains("per-prompt regression row"))
        XCTAssertTrue(src.contains("reviewedPruneExportBlockReason()"))
        XCTAssertTrue(src.contains("throw ExpertPrunePlanExportError.blocked"))
        XCTAssertTrue(src.contains("case missingSourceModel"))
        XCTAssertTrue(src.contains("throw ExpertPrunePlanExportError.missingSourceModel"))
        XCTAssertTrue(src.contains("private var planSourceModelPath: String?"))
        XCTAssertTrue(src.contains("sourceModelPath: planSourceModelPath"))
        XCTAssertTrue(src.contains("sourceModelPath: sourceModelPath"))
        XCTAssertFalse(src.contains("sourceModelPath: (sourceModelPath ?? modelPath).path"))
        XCTAssertTrue(src.contains("evalIndex: currentEvalIndexSummary(comparisonSummary: comparisonSummary)"))
        XCTAssertTrue(src.contains("evalArtifactPath: lastEvalDirectory?.path"))
        XCTAssertTrue(src.contains("private func currentEvalIndexSummary(comparisonSummary: ExpertComparisonSummary) -> ExpertEvalIndexSummary?"))
        XCTAssertTrue(src.contains("selectedRunEvalTraceIntegrityIssue("))
        XCTAssertTrue(src.contains("selectedRunEvalDecodeSettingsIssue("))
        XCTAssertTrue(src.contains("selectedRunEvalRowEvidenceIssue("))
        XCTAssertTrue(src.contains("selectedRunEvalURL("))
        XCTAssertTrue(src.contains("evalTraceIntegrityIssue("))
        XCTAssertTrue(src.contains("issuePrefix: \"Saved prune plan is blocked\""))
        XCTAssertTrue(src.contains("Saved prune plan is blocked: eval_index lists"))
        XCTAssertTrue(src.contains("Saved prune plan is blocked: eval_index contains duplicate prompt IDs."))
        XCTAssertTrue(src.contains(": eval_trace.jsonl missing baseline routing records"))
        XCTAssertTrue(src.contains("indexed baseline route records"))
        XCTAssertTrue(src.contains("indexed masked route records"))
        XCTAssertTrue(src.contains("runtimeMaskLayers"))
        XCTAssertTrue(src.contains("Mask changed after the last comparison"))
        XCTAssertTrue(src.contains("let sidecarURL = try writeCanonicalPrunePlan(plan)"))
        XCTAssertTrue(src.contains("Run sidecar refreshed"))
        XCTAssertTrue(src.contains("private func writeCanonicalPrunePlan(_ plan: ExpertPrunePlan) throws -> URL"))
        XCTAssertTrue(src.contains("private func canonicalPrunePlanURL() throws -> URL"))
        XCTAssertTrue(src.contains("directory.appendingPathComponent(\"prune_plan.json\")"))
        XCTAssertTrue(src.contains("ExpertLabSubPanel(title: \"Selection\""))
        XCTAssertTrue(src.contains("No range selected. Drag across cells to review a group."))
        XCTAssertTrue(src.contains("Use Plan for BF16/F16 Prune"))
        XCTAssertTrue(src.contains("accessibilityLabel(\"Layer"))
    }

    func test_reviewedHardPrunePersistsPlanSidecarForFinalQuantVerification() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("PrequantPruneSheet.swift"),
            encoding: .utf8
        )
        let repoRoot = wizardRoot
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let vmlx = try String(
            contentsOf: repoRoot.appendingPathComponent("jang-tools/jang_tools/expert_lab_vmlx.py"),
            encoding: .utf8
        )
        let runtime = try String(
            contentsOf: repoRoot.appendingPathComponent("jang-runtime/Sources/JANGExpertLab/JANGExpertLab.swift"),
            encoding: .utf8
        )
        let preflight = try String(
            contentsOf: wizardRoot
                .deletingLastPathComponent()
                .appendingPathComponent("Verify/PreflightRunner.swift"),
            encoding: .utf8
        )
        let postConvert = try String(
            contentsOf: wizardRoot
                .deletingLastPathComponent()
                .appendingPathComponent("Verify/PostConvertVerifier.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("copyReviewedPlanSidecarIfNeeded()"))
        XCTAssertTrue(src.contains("materializedReviewedPrunePlanExists(at: sidecarURL)"))
        XCTAssertTrue(src.contains("private static func materializedReviewedPrunePlanExists(at url: URL)"))
        XCTAssertTrue(src.contains("private static func materializeReviewedPrunePlanSidecar("))
        XCTAssertTrue(src.contains("private static func materializedMaskSidecarName("))
        XCTAssertTrue(src.contains("materializedReviewEvidenceSidecars("))
        XCTAssertTrue(src.contains("private static func materializedReviewEvidenceSidecars("))
        XCTAssertTrue(src.contains("private static func materializedSidecarURL("))
        XCTAssertTrue(src.contains("private static func materializedSidecarsEmbeddedIssue("))
        XCTAssertTrue(src.contains("private static func embeddedSidecarIfInside("))
        XCTAssertTrue(src.contains("materialized Expert Lab sidecars are missing"))
        XCTAssertTrue(src.contains("materialized Expert Lab sidecar paths must be embedded in the pruned BF16/F16 source"))
        XCTAssertTrue(src.contains("\"reviewed_evidence_sidecars\""))
        XCTAssertTrue(src.contains("guard let sidecars = plan[\"reviewed_evidence_sidecars\"] as? [String: Any] else { return false }"))
        XCTAssertTrue(src.contains("sidecarMaskName == evalIndexMaskName"))
        XCTAssertTrue(src.contains("evalIndex[\"mask_json\"] = maskName"))
        XCTAssertTrue(src.contains("plan[\"reviewed_evidence_sidecars\"] = ["))
        XCTAssertTrue(src.contains("PrequantPruneError.materialization"))
        XCTAssertTrue(src.contains("persistReviewEvidenceSidecarsIfNeeded()"))
        XCTAssertTrue(src.contains("persistPrunedSourceSuiteVerificationIfNeeded("))
        XCTAssertTrue(src.contains("updateReviewSummaryWithPrunedSuiteVerification("))
        XCTAssertTrue(src.contains("reviewEvidenceReady"))
        XCTAssertTrue(src.contains("prunedSuiteEvidenceReady"))
        XCTAssertTrue(src.contains("prunedSuiteEvidenceIssue"))
        XCTAssertTrue(src.contains("prunedSuiteArtifactDescription"))
        XCTAssertTrue(src.contains("LabeledContent(\"Pruned BF16/vMLX generation\""))
        XCTAssertTrue(src.contains("Text(\"Pruned artifacts: \\(prunedSuiteArtifactDescription)\""))
        XCTAssertTrue(src.contains("prunedSuiteSummaryURL = prunedSuite.summaryURL"))
        XCTAssertTrue(src.contains("prunedSuiteGenerationsURL = prunedSuite.generationsURL"))
        XCTAssertTrue(src.contains("canAdoptPrunedSource"))
        XCTAssertTrue(src.contains("return prunePlanURL != nil && reviewEvidenceReady && prunedSuiteEvidenceReady"))
        XCTAssertTrue(src.contains("reviewEvidenceReady = copiedEvidence.ready"))
        XCTAssertTrue(src.contains("reviewEvidenceIssue = copiedEvidence.issue"))
        XCTAssertTrue(src.contains("Router-only fallback output is inspectable only."))
        XCTAssertTrue(src.contains("does not test what experts do on prompts or unlock final quantization"))
        XCTAssertTrue(src.contains("outputConflictsWithSource"))
        XCTAssertTrue(src.contains("prunedOutputConflictsWithSource(sourceURL: sourceURL, outputURL: outputURL)"))
        XCTAssertTrue(src.contains("path(outputPath, isInsideOrEqualTo: sourcePath)"))
        XCTAssertTrue(src.contains("path(sourcePath, isInsideOrEqualTo: outputPath)"))
        XCTAssertTrue(src.contains("Expert pruning never writes into or above the original BF16/F16 source."))
        XCTAssertTrue(src.contains("For smart pruning: open the original BF16/F16 source in Expert Lab, run BF16/vMLX prompt-suite traces"))
        XCTAssertFalse(src.contains("For smart pruning: build a JANGTQ Expert Lab bundle"))
        XCTAssertTrue(src.contains("\"--require-reviewed-comparison\""))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"prune_plan.json\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_review_summary.json\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_suite.jsonl\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_comparison_summary.json\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_eval.jsonl\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_eval_index.json\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_pruned_generation_summary.json\")"))
        XCTAssertTrue(src.contains("outputURL.appendingPathComponent(\"expert_lab_pruned_generations.jsonl\")"))
        XCTAssertTrue(src.contains("\"eval_index\""))
        XCTAssertTrue(src.contains("\"pruned_suite_verification_ready\""))
        XCTAssertTrue(src.contains("\"pruned_suite_summary\""))
        XCTAssertTrue(src.contains("\"pruned_suite_generations\""))
        XCTAssertTrue(src.contains("\"suite_sha256\""))
        XCTAssertTrue(src.contains("\"generation_defaults\""))
        XCTAssertTrue(src.contains("\"reviewed_masked_eval_jsonl\""))
        XCTAssertTrue(src.contains("\"reviewed_masked_comparison_count\""))
        XCTAssertTrue(src.contains("\"reviewed_masked_mean_text_delta\""))
        XCTAssertTrue(src.contains("\"reviewed_masked_max_text_delta\""))
        XCTAssertTrue(src.contains("\"reviewed_ab_comparison_count\""))
        XCTAssertTrue(src.contains("\"reviewed_ab_mean_text_delta\""))
        XCTAssertTrue(src.contains("\"reviewed_ab_max_text_delta\""))
        XCTAssertTrue(src.contains("\"reviewed_behavior_reference\""))
        XCTAssertTrue(src.contains("\"expert-lab-vmlx\""))
        XCTAssertTrue(src.contains("\"--emit-token-trace\""))
        XCTAssertTrue(src.contains("\"--max-trace-tokens\", \"32768\""))
        XCTAssertTrue(src.contains("token_trace routing evidence"))
        XCTAssertTrue(src.contains("case domainLift = \"domain_lift\""))
        XCTAssertTrue(src.contains("case promptEvidence = \"prompt_evidence\""))
        XCTAssertTrue(src.contains("masked_impact_scope"))
        XCTAssertTrue(src.contains("reviewed_mask_member"))
        XCTAssertTrue(src.contains("private struct ImportedPromptEvidence"))
        XCTAssertTrue(src.contains("var semanticProofDescription: String?"))
        XCTAssertTrue(src.contains("private static func semanticEvidenceIssue("))
        XCTAssertTrue(src.contains("missing masked-output impact evidence"))
        XCTAssertTrue(src.contains("missing masked-output impact scope evidence"))
        XCTAssertTrue(src.contains("missing reviewed mask membership evidence"))
        XCTAssertTrue(src.contains("throw ImportedPrunePlanError.semanticEvidenceRejected(issue)"))
        XCTAssertTrue(src.contains("plan failed the semantic evidence gate"))
        XCTAssertTrue(src.contains("private static func prunedSourceSuiteIssue("))
        XCTAssertTrue(src.contains("private static func prunedReviewedBehaviorComparison("))
        XCTAssertTrue(src.contains("reviewed prompt suite contains duplicate prompt IDs"))
        XCTAssertTrue(src.contains("reviewed eval JSONL contains duplicate prompt IDs"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation JSONL contains duplicate prompt IDs"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation JSONL has prompt IDs outside the reviewed suite"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation is missing vMLX Metal runtime evidence"))
        XCTAssertTrue(src.contains("reviewed eval sidecar is missing"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation failed validators for baseline-qualified prompts"))
        XCTAssertTrue(src.contains("pruned BF16/F16 vMLX summary is missing source model path evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 vMLX summary did not record BF16/vMLX runtime evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 vMLX summary did not record vMLX backend evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 vMLX summary is missing package version evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation is missing source model path evidence"))
        XCTAssertTrue(src.contains("Self.fileSHA256(suiteURL)"))
        XCTAssertTrue(vmlx.contains("\"generation_settings\""))
        XCTAssertTrue(vmlx.contains("\"generation_defaults\""))
        XCTAssertTrue(vmlx.contains("baselineGenerationSettings"))
        XCTAssertTrue(vmlx.contains("maskedGenerationSettings"))
        XCTAssertTrue(vmlx.contains("baseline/masked generation"))
        XCTAssertTrue(vmlx.contains("does not match suite.jsonl"))
        XCTAssertTrue(runtime.contains("if let temperature = prompt.temperature"))
        XCTAssertTrue(runtime.contains("promptConfig.temperature = temperature"))
        XCTAssertTrue(preflight.contains("pruned-source generation source path does not match the pruned BF16/F16 source"))
        XCTAssertTrue(src.contains("\"review_eval_directory\""))
        XCTAssertTrue(src.contains("let planEvalPath = prunePlanSummary?.evalArtifactPath?.trimmingCharacters"))
        XCTAssertTrue(src.contains("Self.planEvalDirectory(from: planEvalPath)"))
        XCTAssertTrue(src.contains("recorded eval_artifact is missing required same-suite files"))
        XCTAssertTrue(src.contains("private static func planEvalDirectory(from path: String?) -> URL?"))
        XCTAssertTrue(src.contains("case evalArtifact = \"eval_artifact\""))
        XCTAssertTrue(src.contains("LabeledContent(\"Eval evidence\""))
        XCTAssertTrue(src.contains("\"same_suite_verification_ready\""))
        XCTAssertTrue(src.contains("\"same_suite_verification_issue\""))
        XCTAssertTrue(src.contains("\"source_model_path\""))
        XCTAssertTrue(src.contains("\"runtime_source_model_path\""))
        XCTAssertTrue(src.contains("\"jang_tools_version\""))
        XCTAssertTrue(src.contains("\"mlx_version\""))
        XCTAssertTrue(src.contains("\"mlx_lm_version\""))
        XCTAssertTrue(src.contains("sameSuiteEvidenceIssue("))
        XCTAssertTrue(src.contains("suiteSemanticCoverageIssue("))
        XCTAssertTrue(src.contains("suite.jsonl is missing required semantic prompt probes"))
        XCTAssertTrue(postConvert.contains("let semanticCoverageIssue = expectedSuiteURL.flatMap { suiteSemanticCoverageIssue($0) }"))
        XCTAssertTrue(postConvert.contains("sameSuiteIssue ?? prunedBehaviorComparison.issue ?? semanticCoverageIssue"))
        XCTAssertTrue(src.contains("reviewEvidenceIssue"))
        XCTAssertTrue(src.contains("return ReviewEvidenceSidecars("))
        XCTAssertTrue(src.contains("eval_index.json prompt IDs missing from eval.jsonl"))
        XCTAssertTrue(src.contains("eval_index.json prompt IDs missing from eval_trace.jsonl"))
        XCTAssertTrue(src.contains("eval.jsonl prompt IDs outside eval_index.json"))
        XCTAssertTrue(src.contains("eval_trace.jsonl prompt IDs outside eval_index.json"))
        XCTAssertTrue(src.contains("eval_trace.jsonl missing baseline routing records"))
        XCTAssertTrue(src.contains("eval_trace.jsonl missing masked routing records"))
        XCTAssertTrue(src.contains("indexed baseline route records"))
        XCTAssertTrue(src.contains("indexed masked route records"))
        XCTAssertTrue(src.contains("eval_trace.jsonl masked routing records are missing mask evidence"))
        XCTAssertTrue(src.contains("eval_index.json prompt IDs missing suite.jsonl prompts"))
        XCTAssertTrue(src.contains("eval_index.json prompt IDs outside suite.jsonl"))
        XCTAssertTrue(src.contains("\"suite_sha256\""))
        XCTAssertTrue(src.contains("eval_index.json is missing suite.jsonl fingerprint evidence"))
        XCTAssertTrue(src.contains("eval_index.json suite.jsonl fingerprint does not match suite.jsonl"))
        XCTAssertTrue(src.contains("writeRecoveredEvalIndexIfPossible("))
        XCTAssertTrue(src.contains("\"jang-expert-lab-eval-index-v1\""))
        XCTAssertTrue(src.contains("\"eval_trace_jsonl\""))
        XCTAssertTrue(src.contains("evalRowEvidenceIssue("))
        XCTAssertTrue(src.contains("jsonlObjects("))
        XCTAssertTrue(src.contains("promptIDs == suiteIDs"))
        XCTAssertTrue(src.contains("Set(traceIDs) == Set(promptIDs)"))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt baseline/masked output text"))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt runtime device evidence"))
        XCTAssertTrue(src.contains("eval.jsonl did not record per-prompt vMLX backend evidence"))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt vMLX package version evidence"))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt source model path evidence"))
        XCTAssertTrue(src.contains("eval.jsonl did not record an applied BF16/vMLX mask"))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt regression flag evidence"))
        XCTAssertTrue(src.contains("eval_index.json is missing generation-depth token evidence"))
        XCTAssertTrue(src.contains("eval_index.json is missing routing record evidence"))
        XCTAssertTrue(src.contains("eval_index.json layer-stat coverage is incomplete for indexed prompts"))
        XCTAssertTrue(src.contains("eval_index.json is missing eval_trace.jsonl evidence"))
        XCTAssertTrue(src.contains("eval_index.json average generated depth"))
        XCTAssertTrue(src.contains("eval_index.json is missing runtime device evidence"))
        XCTAssertTrue(src.contains("eval_index.json did not record a Metal runtime"))
        XCTAssertTrue(src.contains("eval_index.json did not record BF16/vMLX runtime evidence"))
        XCTAssertTrue(src.contains("eval_index.json did not record vMLX backend evidence"))
        XCTAssertTrue(src.contains("eval_index.json is missing vMLX routed-layer hook evidence"))
        XCTAssertTrue(src.contains("eval_index.json vMLX hook coverage"))
        XCTAssertTrue(src.contains("eval_index.json recorded incomplete vMLX routed-layer hook coverage"))
        XCTAssertTrue(preflight.contains("eval_index.json is missing vMLX package version evidence"))
        XCTAssertTrue(src.contains("eval_index.json is missing source model path evidence"))
        XCTAssertTrue(src.contains("eval_index.json did not record an applied BF16/vMLX mask"))
        XCTAssertTrue(src.contains("eval_index.json did not record disabled expert evidence"))
        XCTAssertTrue(src.contains("eval.jsonl is missing per-prompt disabled expert evidence"))
        XCTAssertTrue(src.contains("eval.jsonl layer-stat evidence is incomplete for baseline/masked prompts"))
        XCTAssertTrue(src.contains("eval_index.json source model path does not match reviewed source"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation did not record BF16/vMLX runtime evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation did not record vMLX backend evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation is missing vMLX routed-layer hook evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation vMLX hook coverage"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation recorded incomplete vMLX routed-layer hook coverage"))
        XCTAssertTrue(preflight.contains("pruned-source generation is missing vMLX package version evidence"))
        XCTAssertTrue(src.contains("pruned BF16/F16 generation source path does not match the pruned source"))
        XCTAssertTrue(src.contains("eval_index.json still has high-risk domains"))
        XCTAssertTrue(src.contains("masked comparison severity"))
        XCTAssertTrue(src.contains("eval_index regression severity"))
        XCTAssertTrue(src.contains("\"regression_severity\""))
        XCTAssertTrue(src.contains("\"runtime_metal_enabled\""))
        XCTAssertTrue(src.contains("let evalIndex: ImportedEvalIndexSummary?"))
        XCTAssertTrue(src.contains("private struct ImportedEvalIndexSummary"))
        XCTAssertTrue(src.contains("plan is missing per-prompt eval_index evidence"))
        XCTAssertTrue(src.contains("eval_index still has risky prompt IDs"))
        XCTAssertTrue(src.contains("evalIndex.highRiskDomains"))
        XCTAssertTrue(src.contains("eval_index is missing generation-depth token evidence"))
        XCTAssertTrue(src.contains("eval_index is missing routing record evidence for every indexed prompt."))
        XCTAssertTrue(src.contains("eval_index layer-stat coverage is incomplete for indexed prompts."))
        XCTAssertTrue(src.contains("eval_index is missing eval_trace.jsonl evidence."))
        XCTAssertTrue(src.contains("eval_index did not record BF16/vMLX runtime evidence"))
        XCTAssertTrue(src.contains("eval_index did not record vMLX backend evidence."))
        XCTAssertTrue(src.contains("eval_index is missing vMLX routed-layer hook evidence."))
        XCTAssertTrue(src.contains("eval_index is missing vMLX package version evidence"))
        XCTAssertTrue(src.contains("eval_index is missing source model path evidence"))
        XCTAssertTrue(src.contains("eval_index did not record an applied BF16/vMLX mask"))
        XCTAssertTrue(src.contains("eval_index did not record disabled expert evidence"))
        XCTAssertTrue(src.contains("let baselineRouteRecordCount: Int?"))
        XCTAssertTrue(src.contains("let maskedRouteRecordCount: Int?"))
        XCTAssertTrue(src.contains("let baselineLayerStatsPromptCount: Int?"))
        XCTAssertTrue(src.contains("let maskedLayerStatsPromptCount: Int?"))
        XCTAssertTrue(src.contains("let suiteJSONL: String?"))
        XCTAssertTrue(src.contains("let suiteSHA256: String?"))
        XCTAssertTrue(src.contains("let semanticCoverage: [String]"))
        XCTAssertTrue(src.contains("let missingSemanticCoverage: [String]?"))
        XCTAssertTrue(src.contains("let evalJSONL: String?"))
        XCTAssertTrue(src.contains("let evalTraceJSONL: String?"))
        XCTAssertTrue(src.contains("let comparisonSummary: String?"))
        XCTAssertTrue(src.contains("let maskJSON: String?"))
        XCTAssertTrue(src.contains("let jangToolsVersion: String?"))
        XCTAssertTrue(src.contains("case baselineRouteRecordCountSnake = \"baseline_route_record_count\""))
        XCTAssertTrue(src.contains("case baselineLayerStatsPromptCountSnake = \"baseline_layer_stats_prompt_count\""))
        XCTAssertTrue(src.contains("case maskedLayerStatsPromptCountSnake = \"masked_layer_stats_prompt_count\""))
        XCTAssertTrue(src.contains("case suiteJSONLSnake = \"suite_jsonl\""))
        XCTAssertTrue(src.contains("case suiteSHA256Snake = \"suite_sha256\""))
        XCTAssertTrue(src.contains("case semanticCoverageSnake = \"semantic_coverage\""))
        XCTAssertTrue(src.contains("case missingSemanticCoverageSnake = \"missing_semantic_coverage\""))
        XCTAssertTrue(src.contains("case evalJSONLSnake = \"eval_jsonl\""))
        XCTAssertTrue(src.contains("case evalTraceJSONLSnake = \"eval_trace_jsonl\""))
        XCTAssertTrue(src.contains("case comparisonSummarySnake = \"comparison_summary\""))
        XCTAssertTrue(src.contains("case maskJSONSnake = \"mask_json\""))
        XCTAssertTrue(src.contains("case jangToolsVersionSnake = \"jang_tools_version\""))
        XCTAssertTrue(src.contains("eval_index is missing suite.jsonl evidence."))
        XCTAssertTrue(src.contains("eval_index is missing suite.jsonl fingerprint evidence."))
        XCTAssertTrue(src.contains("eval_index is missing semantic coverage evidence."))
        XCTAssertTrue(src.contains("eval_index semantic coverage is missing required probes"))
        XCTAssertTrue(src.contains("eval_index is missing missing-semantic-coverage evidence."))
        XCTAssertTrue(src.contains("eval_index records missing semantic prompt probes"))
        XCTAssertTrue(src.contains("eval_index is missing comparison_summary evidence."))
        XCTAssertTrue(src.contains("eval_index is missing eval.jsonl evidence."))
        XCTAssertTrue(src.contains("eval_index is missing mask.json evidence."))
        XCTAssertTrue(src.contains("minimumReviewedPruneMeanTokens"))
        XCTAssertTrue(src.contains("minimumReviewedPrunePromptCount = 50"))
        XCTAssertTrue(src.contains("plan is missing top-k safety evidence"))
        XCTAssertTrue(src.contains("sourceURL.standardizedFileURL.resolvingSymlinksInPath().path"))
        XCTAssertTrue(src.contains("case sourceMismatch(planSource: String, selectedSource: String)"))
        XCTAssertTrue(src.contains("plan source_model does not match the selected BF16/F16 source"))
        XCTAssertFalse(src.contains("legacy plan; pruner re-checks top-k"))
        XCTAssertTrue(src.contains("comparisonRejected"))
        XCTAssertTrue(src.contains("plan failed the same-suite A/B comparison gate"))
        XCTAssertTrue(src.contains("A/B comparison found no safe drop candidates"))
        XCTAssertTrue(src.contains("Structural verification passed, but same-suite Expert Lab evidence is not ready"))
        XCTAssertTrue(src.contains("Structural verification passed, but pruned BF16/F16 same-suite vMLX generation is not ready"))
        XCTAssertTrue(src.contains("FileManager.default.copyItem(at: prunePlanURL, to: sidecarURL)"))
        XCTAssertTrue(src.contains("This pruned BF16/F16 source is now eligible for final JANG/JANGTQ conversion."))
    }

    func test_finalProfileSelectionHappensAfterReviewedPrune() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/ProfileStep.swift"),
            encoding: .utf8
        )
        let runStep = try String(
            contentsOf: wizardRoot.appendingPathComponent("Steps/RunStep.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("Reviewed BF16/F16 Source"))
        XCTAssertTrue(src.contains("Final conversion profile"))
        XCTAssertTrue(src.contains("Choose the final JANG or JANGTQ profile"))
        XCTAssertTrue(src.contains("Quantize with JANG"))
        XCTAssertTrue(src.contains("Quantize with JANGTQ"))
        XCTAssertTrue(src.contains("Any legacy review bundle is no longer used as pruning authority."))
        XCTAssertTrue(src.contains("This opens the original BF16/F16 source in Expert Lab for vMLX prompt-suite expert probing."))
        XCTAssertTrue(src.contains("Legacy JANGTQ tracing is downstream compatibility evidence only."))
        XCTAssertTrue(src.contains("Reviewed hard pruning starts by opening the original BF16/F16 source in BF16/vMLX Expert Review before any final conversion."))
        XCTAssertTrue(src.contains("For reviewed hard pruning, go back to Source and run BF16/vMLX Expert Review on the original BF16/F16 source first."))
        XCTAssertFalse(src.contains("This will build a temporary JANGTQ review bundle"))
        XCTAssertFalse(src.contains("open Expert Lab for trace/mask evaluation on this quantized bundle"))
        XCTAssertFalse(src.contains("choose JANGTQ above"))
        XCTAssertTrue(src.contains("Same-suite Expert Lab evidence ready"))
        XCTAssertTrue(src.contains("Same-suite Expert Lab evidence incomplete"))
        XCTAssertTrue(src.contains("Pruned BF16/F16 vMLX generation ready"))
        XCTAssertTrue(src.contains("Pruned BF16/F16 vMLX generation missing"))
        XCTAssertTrue(src.contains("Pruned artifacts: \\(evidence.prunedSuiteArtifactDescription)"))
        XCTAssertTrue(src.contains("Reviewed suite SHA-256: \\(evidence.reviewedSuiteFingerprintDescription)"))
        XCTAssertTrue(src.contains("Same-suite Expert Lab evidence summary is missing from the pruned source."))
        XCTAssertTrue(src.contains("Masked compare: \\(compare.promptCount) prompts"))
        XCTAssertTrue(src.contains("Eval deltas: mean text %.4f, risky prompts %d, avg tokens %.1f / %.1f"))
        XCTAssertTrue(src.contains("Risk domains: \\(compare.riskDomainDescription)"))
        XCTAssertTrue(src.contains("Artifacts: \\(compare.artifactDescription)"))
        XCTAssertTrue(src.contains("Masked compare sidecars are not readable from the pruned source."))
        XCTAssertTrue(src.contains("var reviewedPruneEvidence: ReviewedPruneEvidenceSummary?"))
        XCTAssertTrue(src.contains("guard !preflight.isEmpty else { return false }"))
        XCTAssertTrue(src.contains("if isFinalConversionFromReviewedPrune"))
        XCTAssertTrue(src.contains("$0.id == .reviewedPruneVerified && $0.status == .pass"))
        XCTAssertTrue(src.contains("expert_lab_review_summary.json"))
        XCTAssertTrue(src.contains("struct ReviewedPruneEvidenceSummary"))
        XCTAssertTrue(src.contains("struct ReviewedPruneComparisonEvidence"))
        XCTAssertTrue(src.contains("case sameSuiteVerificationReady = \"same_suite_verification_ready\""))
        XCTAssertTrue(src.contains("case suiteSHA256 = \"suite_sha256\""))
        XCTAssertTrue(src.contains("case comparisonSummary = \"comparison_summary\""))
        XCTAssertTrue(src.contains("case evalTraceJSONL = \"eval_trace_jsonl\""))
        XCTAssertTrue(src.contains("case evalIndex = \"eval_index\""))
        XCTAssertTrue(src.contains("case maskJSON = \"mask_json\""))
        XCTAssertTrue(src.contains("case prunedSuiteVerificationReady = \"pruned_suite_verification_ready\""))
        XCTAssertTrue(src.contains("case prunedSuiteSummary = \"pruned_suite_summary\""))
        XCTAssertTrue(src.contains("case prunedSuiteGenerations = \"pruned_suite_generations\""))
        XCTAssertTrue(src.contains("evidence.evalTraceJSONL"))
        XCTAssertTrue(src.contains("evidence.maskJSON ?? evidence.mask"))
        XCTAssertTrue(runStep.contains("isFinalQuantFromReviewedPrune"))
        XCTAssertTrue(runStep.contains("PreflightRunner.reviewedPruneVerifiedCheck(plan: coord.plan)"))
        XCTAssertTrue(runStep.contains("Reviewed BF16/F16 prune verification blocked final quantization."))
        XCTAssertTrue(runStep.contains("Quantize with \\(coord.plan.family == .jangtq ? \"JANGTQ\" : \"JANG\")"))
        XCTAssertTrue(runStep.contains("Post-quant same-suite verification ready"))
        XCTAssertTrue(runStep.contains("scan verified pruned source"))
        XCTAssertTrue(runStep.contains("quantize pruned source"))
        XCTAssertTrue(runStep.contains("validate post-quant bundle"))
    }

    func test_defaultExpertIdentificationSuitesAreThorough() throws {
        let wizardRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(
            contentsOf: wizardRoot.appendingPathComponent("ExpertLabSheet.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(src.contains("domainFingerprintSuite(name: \"Domain Fingerprint 72\")"))
        XCTAssertTrue(src.contains("domainFingerprintFacets(domain: seed.domain, prompt: text)"))
        XCTAssertTrue(src.contains("generatedProbeSuite(name: \"Reviewed Prune 50\", promptCount: 50)"))
        XCTAssertTrue(src.contains("private static let domainFingerprintSeeds"))
        XCTAssertTrue(src.contains("domain: \"coding\""))
        XCTAssertTrue(src.contains("domain: \"math\""))
        XCTAssertTrue(src.contains("domain: \"safety\""))
        XCTAssertTrue(src.contains("domain: \"tools\""))
        XCTAssertTrue(src.contains("generatedProbeSuite(name: \"Balanced 150\", promptCount: 150)"))
        XCTAssertTrue(src.contains("generatedProbeSuite(name: \"Fast 50\", promptCount: 50)"))
        XCTAssertTrue(src.contains("generatedProbeSuite(name: \"Deep 500\", promptCount: 500)"))
        XCTAssertTrue(src.contains("generatedProbeSuite(name: \"Smoke 21\", promptCount: 21)"))
        XCTAssertTrue(src.contains("Text(vm.selectedSuiteEvidenceSummary)"))
        XCTAssertTrue(src.contains("var selectedSuiteEvidenceSummary: String"))
        XCTAssertTrue(src.contains("\"\\(prompts.count) prompts / \\(domainCount) domains / \\(tokenSummary) / \\(semanticSummary)\""))
        XCTAssertTrue(src.contains("private static func semanticCoverageSummary(for prompts: [ExpertPrompt]) -> String"))
        XCTAssertTrue(src.contains("ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains"))
        XCTAssertTrue(src.contains("return \"semantic probes ready\""))
        XCTAssertTrue(src.contains("return \"missing semantic probes: \\(names)\""))
        XCTAssertTrue(src.contains("maxNewTokens: 96"))
        XCTAssertTrue(src.contains("maximumPromptSuiteMaxTokens = 512"))
        XCTAssertTrue(src.contains("@State private var atlasDisplayMode: ExpertAtlasDisplayMode = .map"))
        XCTAssertTrue(src.contains("private enum ExpertAtlasDisplayMode"))
        XCTAssertTrue(src.contains("case map"))
        XCTAssertTrue(src.contains("case table"))
        XCTAssertTrue(src.contains("Picker(\"Atlas view\", selection: $atlasDisplayMode)"))
        XCTAssertTrue(src.contains("atlasDisplayMode == .table"))
        XCTAssertTrue(src.contains("atlasMapPanel(rows: vm.atlasGridRows, metricScales: vm.atlasLayerMetricScales)"))
        XCTAssertTrue(src.contains("ExpertLabSectionHeader(title: \"Expert Table\", systemImage: \"tablecells\")"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Domain\", width: 116)"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Coactive\", width: 104)"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Compared\", width: 156)"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Evidence\", width: 64)"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Conf\", width: 52)"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Annot\", width: 64)"))
        XCTAssertTrue(src.contains("atlasTableHeader(\"Severity\", width: 84)"))
        XCTAssertTrue(src.contains("@State private var showsAtlasLegend = false"))
        XCTAssertTrue(src.contains("Label(\"Semantic colors\", systemImage: \"info.circle\")"))
        XCTAssertTrue(src.contains(".popover(isPresented: $showsAtlasLegend"))
        XCTAssertTrue(src.contains("private var atlasLegendPopover: some View"))
        XCTAssertTrue(src.contains("private var atlasLegendDomains: [String]"))
        XCTAssertTrue(src.contains("ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains"))
        XCTAssertTrue(src.contains("atlasLegendRow(domain)"))
        XCTAssertTrue(src.contains("Picker(\"Metric\", selection: $vm.atlasMetric)"))
        XCTAssertTrue(src.contains("private struct ExpertAtlasLayerRow"))
        XCTAssertTrue(src.contains("private struct ExpertAtlasMetricScale"))
        XCTAssertTrue(src.contains("var atlasGridRows: [ExpertAtlasLayerRow]"))
        XCTAssertTrue(src.contains("var atlasLayerMetricScales: [Int: ExpertAtlasMetricScale]"))
        XCTAssertTrue(src.contains("func metricIntensity("))
        XCTAssertTrue(src.contains("scales: [Int: ExpertAtlasMetricScale]"))
        XCTAssertTrue(src.contains("previewLasso(rect: rect, rows: rows)"))
        XCTAssertFalse(src.contains("ExpertCellFramePreferenceKey"))
        XCTAssertFalse(src.contains("atlasCellFrames"))
        XCTAssertTrue(src.contains("var atlasTableEntries: [ExpertAtlasEntry]"))
        XCTAssertTrue(src.contains("ForEach(Array(vm.atlasTableEntries.prefix(atlasTableDisplayLimit)))"))
        XCTAssertTrue(src.contains("atlasFilterField(\"Domain\", text: $vm.atlasDomainFilterText)"))
        XCTAssertTrue(src.contains("atlasFilterField(\"Prompt\", text: $vm.atlasPromptFilterText)"))
        XCTAssertTrue(src.contains("Self.matchesDomainFilter(entry, text: atlasDomainFilterText)"))
        XCTAssertTrue(src.contains("Self.matchesPromptFilter(entry, text: atlasPromptFilterText)"))
        XCTAssertTrue(src.contains("atlasTableCell(vm.dominantDomainSummary(for: entry), width: 116)"))
        XCTAssertTrue(src.contains("atlasTableCell(vm.coactivationSummary(for: entry), width: 104)"))
        XCTAssertTrue(src.contains("atlasTableCell(\"\\(vm.evidenceCount(for: entry))\", width: 64)"))
        XCTAssertTrue(src.contains("atlasTableCell(String(format: \"%.2f\", entry.confidenceScore), width: 52)"))
        XCTAssertTrue(src.contains("vm.manualAnnotationSummary(for: entry)"))
        XCTAssertTrue(src.contains("vm.selectedExpertRegressionSeveritySummary(for: entry)"))
        XCTAssertTrue(src.contains("func selectedExpertRegressionSeveritySummary(for entry: ExpertAtlasEntry) -> String"))
        XCTAssertTrue(src.contains("func coactivationSummary(for entry: ExpertAtlasEntry) -> String"))
        XCTAssertTrue(src.contains("private static func strongestCoactivationScore"))
        XCTAssertTrue(src.contains("private static func sortedCoactivationNeighbors"))
        XCTAssertTrue(src.contains("func manualAnnotationSummary(for entry: ExpertAtlasEntry) -> String"))
        XCTAssertTrue(src.contains("func dominantDomainSummary(for entry: ExpertAtlasEntry) -> String"))
        XCTAssertTrue(src.contains("ExpertDomainTaxonomy.canonicalSemanticDomain(evidence.domain)"))
        XCTAssertTrue(src.contains("private static func promptFilterHaystack(for entry: ExpertAtlasEntry) -> String"))
        XCTAssertTrue(src.contains("parts.append(evidence.promptID)"))
        XCTAssertTrue(src.contains("parts.append(evidence.promptExcerpt)"))
        XCTAssertTrue(src.contains("normalizedDomainFilterText(parts.joined(separator: \" \"))"))
        XCTAssertTrue(src.contains("return \"label+note\""))
        XCTAssertTrue(src.contains("case activationRank"))
        XCTAssertTrue(src.contains("case tokenDepth"))
        XCTAssertTrue(src.contains("case coactivation"))
        XCTAssertTrue(src.contains("case regressionSeverity"))
        XCTAssertTrue(src.contains("case .activationRank: \"Activation rank\""))
        XCTAssertTrue(src.contains("case .tokenDepth: \"Token depth\""))
        XCTAssertTrue(src.contains("case .regressionSeverity: \"Regression severity\""))
        XCTAssertTrue(src.contains("switch ExpertDomainTaxonomy.canonicalSemanticDomain(raw)"))
        XCTAssertTrue(src.contains("case \"chinese\":"))
        XCTAssertTrue(src.contains("case \"non_english\":"))
        XCTAssertTrue(src.contains("case \"unknown_language_role\":"))
        XCTAssertTrue(src.contains("case \"safety_medical_legal_sensitive\", \"safety_sensitive\", \"medical_sensitive\", \"legal_sensitive\":"))
        XCTAssertTrue(src.contains("cleaned == \"manual\" || cleaned == \"user-label\""))
        XCTAssertTrue(src.contains("selectedExpertRegressionSeverityRank(for: lhs)"))
        XCTAssertTrue(src.contains("lhs.hits > 0 ? lhs.meanSelectedRank : Float.greatestFiniteMagnitude"))
        XCTAssertTrue(src.contains("lhs.meanTokenIndex ?? Float.greatestFiniteMagnitude"))
        XCTAssertTrue(src.contains("vm.selectedExpertComparedMaskStatus(for: entry)"))
        XCTAssertTrue(src.contains("vm.selectedExpertMaskedImpactSummary(for: entry)"))
        XCTAssertTrue(src.contains("vm.selectedExpertReviewStatus(for: entry)"))
        XCTAssertTrue(src.contains("ExpertDomainTaxonomy.dominantDomain"))
        XCTAssertTrue(src.contains("instruction-following"))
        XCTAssertTrue(src.contains("model-prune"))
        XCTAssertTrue(src.contains("\"chinese\", \"translation\", \"non_english\""))
        XCTAssertTrue(src.contains("\"unknown_language_role\", \"non_english\", \"language-id\""))
        XCTAssertTrue(src.contains("\"english_dominant\""))
        XCTAssertTrue(src.contains("\"safety_medical_legal_sensitive\""))
        XCTAssertTrue(src.contains("\"json\", \"formatting\""))
        XCTAssertTrue(src.contains("non_english"))
        XCTAssertTrue(src.contains("unknown_language_role"))
        XCTAssertTrue(src.contains("safety_medical_legal_sensitive"))
        XCTAssertTrue(src.contains("ExpertDomainTaxonomy.displayName(for: domain)"))
        XCTAssertTrue(src.contains("let promptEvidence = entry.promptEvidence ?? []"))
        XCTAssertTrue(src.contains("evidence.promptExcerpt"))
        XCTAssertTrue(src.contains("legal-safety"))
        XCTAssertTrue(src.contains("english_dominant"))
        XCTAssertTrue(src.contains("temperature: 0.0"))
    }

    private static func prompt(_ id: String) -> ExpertPrompt {
        ExpertPrompt(id: id, domain: "general", text: "Prompt \(id)")
    }

    private static func runtimeEvidenceIssue(
        runtimeMode: String? = "bf16_vmlx",
        runtimeBackend: String? = "vmlx",
        runtimeMetalEnabled: Bool? = true,
        deviceName: String? = "Unit Metal",
        jangToolsVersion: String? = "2.5.31",
        mlxVersion: String? = "0.31.2",
        mlxLMVersion: String? = "0.31.3",
        sourceModelPath: String? = "/tmp/source",
        hookedMOELayers: Int? = 40,
        expectedMOELayers: Int? = 40,
        hookCoverageComplete: Bool? = true,
        maskRequired: Bool = false,
        maskApplied: Bool? = nil,
        disabledExpertCount: Int? = nil,
        topKOverride: Int? = nil,
        expectedLayers: Int? = 40,
        expectedSourcePath: String? = "/tmp/source"
    ) -> String? {
        ExpertLabVMLXRuntimeEvidenceValidator.issue(
            promptID: "p1",
            runtimeMode: runtimeMode,
            runtimeBackend: runtimeBackend,
            runtimeMetalEnabled: runtimeMetalEnabled,
            deviceName: deviceName,
            jangToolsVersion: jangToolsVersion,
            mlxVersion: mlxVersion,
            mlxLMVersion: mlxLMVersion,
            sourceModelPath: sourceModelPath,
            hookedMOELayers: hookedMOELayers,
            expectedMOELayers: expectedMOELayers,
            hookCoverageComplete: hookCoverageComplete,
            maskRequired: maskRequired,
            maskApplied: maskApplied,
            disabledExpertCount: disabledExpertCount,
            topKOverride: topKOverride,
            expectedLayers: expectedLayers,
            expectedSourcePath: expectedSourcePath
        )
    }
}
