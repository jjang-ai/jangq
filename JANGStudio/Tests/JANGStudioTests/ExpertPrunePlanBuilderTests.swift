// JANGStudio/Tests/JANGStudioTests/ExpertPrunePlanBuilderTests.swift
import JANGExpertLab
import XCTest

final class ExpertPrunePlanBuilderTests: XCTestCase {
    func test_buildPlanKeepsPromptActiveExpertsAndRecordsEvidence() throws {
        let atlas = ExpertAtlas(
            promptCount: 3,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 1,
                    probabilityMass: 10,
                    tokenCount: 6,
                    domains: ["general": 1],
                    label: "mixed",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 8,
                    probabilityMass: 2,
                    tokenCount: 6,
                    domains: ["coding": 8],
                    domainLift: ["coding": 2.25, "math": 0.4],
                    promptEvidence: [
                        ExpertPromptEvidence(
                            promptID: "code-1",
                            domain: "coding",
                            subdomain: "swift",
                            tags: ["instruction_following", "formatting"],
                            promptExcerpt: "Write a Swift function and explain the edge cases.",
                            hits: 5
                        )
                    ],
                    evidenceCount: 3,
                    label: "coding-specialist",
                    userLabel: "reviewed-coding-core",
                    userNotes: "Reviewer confirmed this expert carries Swift formatting examples.",
                    isDead: false,
                    isHot: true
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 2,
                    hits: 4,
                    probabilityMass: 3,
                    tokenCount: 6,
                    domains: ["math": 4],
                    label: "math-specialist",
                    isDead: false,
                    isHot: false
                )
            ]
        )

        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            sourceModelPath: "/tmp/model"
        )

        let layer = try XCTUnwrap(plan.layers["0"])
        XCTAssertEqual(plan.method, "prompt_trace_hits_mass_domain_lift_v1")
        XCTAssertEqual(plan.keepExpertsPerLayer, 2)
        XCTAssertEqual(layer.keep, [1, 2])
        XCTAssertEqual(layer.drop, [0, 3])
        XCTAssertTrue(layer.evidence.first { $0.expert == 1 }?.kept == true)
        XCTAssertEqual(layer.evidence.first { $0.expert == 1 }?.label, "reviewed-coding-core")
        XCTAssertEqual(
            layer.evidence.first { $0.expert == 1 }?.userNotes,
            "Reviewer confirmed this expert carries Swift formatting examples."
        )
        XCTAssertEqual(layer.evidence.first { $0.expert == 1 }?.evidenceCount, 3)
        let codingEvidence = try XCTUnwrap(layer.evidence.first { $0.expert == 1 })
        XCTAssertEqual(codingEvidence.domainLift["coding"] ?? 0, 2.25, accuracy: 0.0001)
        let promptEvidence = try XCTUnwrap(codingEvidence.promptEvidence.first)
        XCTAssertEqual(promptEvidence.promptID, "code-1")
        XCTAssertEqual(promptEvidence.domain, "coding")
        XCTAssertEqual(promptEvidence.subdomain, "swift")
        XCTAssertEqual(promptEvidence.tags, ["instruction_following", "formatting"])
        XCTAssertTrue(promptEvidence.promptExcerpt.contains("Swift function"))
        XCTAssertEqual(promptEvidence.hits, 5)
        XCTAssertTrue(layer.evidence.first { $0.expert == 3 }?.label == "unobserved")

        let data = try JSONEncoder().encode(plan)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let layers = try XCTUnwrap(json["layers"] as? [String: Any])
        let layerJSON = try XCTUnwrap(layers["0"] as? [String: Any])
        let evidenceJSON = try XCTUnwrap(layerJSON["evidence"] as? [[String: Any]])
        let codingJSON = try XCTUnwrap(evidenceJSON.first { $0["expert"] as? Int == 1 })
        let liftJSON = try XCTUnwrap(codingJSON["domain_lift"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(liftJSON["coding"] as? Double), 2.25, accuracy: 0.0001)
        let promptJSON = try XCTUnwrap(codingJSON["prompt_evidence"] as? [[String: Any]])
        XCTAssertEqual(promptJSON.first?["promptID"] as? String, "code-1")
        XCTAssertEqual(promptJSON.first?["domain"] as? String, "coding")
        XCTAssertEqual(
            codingJSON["user_notes"] as? String,
            "Reviewer confirmed this expert carries Swift formatting examples."
        )
    }

    func test_prunePlanEvidenceDecodesLegacyRowsWithoutSemanticProofFields() throws {
        let data = Data("""
        {
          "expert": 7,
          "hits": 2,
          "probabilityMass": 0.5,
          "domains": {"general": 2},
          "label": "legacy-general",
          "kept": false
        }
        """.utf8)

        let evidence = try JSONDecoder().decode(ExpertPrunePlanEvidence.self, from: data)

        XCTAssertEqual(evidence.expert, 7)
        XCTAssertEqual(evidence.frequency, 0)
        XCTAssertEqual(evidence.routerMass, 0.5, accuracy: 0.0001)
        XCTAssertNil(evidence.maskedImpactScope)
        XCTAssertFalse(evidence.reviewedMaskMember)
        XCTAssertEqual(evidence.domainLift, [:])
        XCTAssertEqual(evidence.promptEvidence, [])
        XCTAssertEqual(evidence.reason, "")
        XCTAssertNil(evidence.userNotes)
        XCTAssertFalse(evidence.userForcedDrop)
    }

    func test_buildPlanTreatsForceDroppedExpertsAsUserDisabledDrops() throws {
        let atlas = ExpertAtlas(
            promptCount: 1,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 20,
                    probabilityMass: 9,
                    tokenCount: 10,
                    domains: ["coding": 20],
                    label: "coding-specialist",
                    isDead: false,
                    isHot: true
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 4,
                    probabilityMass: 3,
                    tokenCount: 10,
                    domains: ["general": 4],
                    label: "mixed",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 2,
                    hits: 3,
                    probabilityMass: 2,
                    tokenCount: 10,
                    domains: ["math": 3],
                    label: "math-specialist",
                    isDead: false,
                    isHot: false
                )
            ]
        )

        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            forceDropByLayer: [0: [0]]
        )

        let layer = try XCTUnwrap(plan.layers["0"])
        XCTAssertEqual(layer.keep, [1, 2])
        XCTAssertEqual(layer.drop, [0, 3])
        let disabledEvidence = try XCTUnwrap(layer.evidence.first { $0.expert == 0 })
        XCTAssertFalse(disabledEvidence.kept)
        XCTAssertTrue(disabledEvidence.label.contains("user-disabled"))
        XCTAssertTrue(disabledEvidence.userForcedDrop)
        XCTAssertTrue(disabledEvidence.reason.contains("user-forced"))
    }

    func test_buildPlanUsesSafeABCandidatesAsDropPreferenceAndEvidence() throws {
        let atlas = ExpertAtlas(
            promptCount: 50,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 100,
                    probabilityMass: 100,
                    tokenCount: 20,
                    domains: ["general": 100],
                    label: "general-hot",
                    isDead: false,
                    isHot: true
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 90,
                    probabilityMass: 90,
                    tokenCount: 20,
                    domains: ["coding": 90],
                    label: "coding",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 2,
                    hits: 10,
                    probabilityMass: 10,
                    tokenCount: 20,
                    domains: ["general": 10],
                    label: "low-general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 3,
                    hits: 5,
                    probabilityMass: 5,
                    tokenCount: 20,
                    domains: ["general": 5],
                    label: "low",
                    isDead: false,
                    isHot: false
                )
            ]
        )
        let comparison = reviewedComparison(
            meanTextDelta: 0.07,
            meanLatencyDeltaPct: -8,
            safeDropCandidates: [
                ExpertCoordinate(layer: 0, expert: 0),
                ExpertCoordinate(layer: 0, expert: 3)
            ]
        )

        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(
                meanTextDelta: 0.07,
                disabledExpertCount: 2
            ),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )

        let layer = try XCTUnwrap(plan.layers["0"])
        XCTAssertEqual(layer.keep, [1, 2])
        XCTAssertEqual(layer.drop, [0, 3])
        let safeDropEvidence = try XCTUnwrap(layer.evidence.first { $0.expert == 0 })
        XCTAssertFalse(safeDropEvidence.kept)
        XCTAssertEqual(safeDropEvidence.ablationDelta ?? -1, 0.07, accuracy: 0.0001)
        XCTAssertEqual(safeDropEvidence.maskedImpactScope, "same_suite_mask_mean_text_delta")
        XCTAssertTrue(safeDropEvidence.reviewedMaskMember)
        XCTAssertTrue(safeDropEvidence.reason.contains("masked A/B safe"))
        XCTAssertTrue(safeDropEvidence.reason.contains("masked pass 100%"))
        XCTAssertFalse(safeDropEvidence.userForcedDrop)
        XCTAssertTrue(layer.evidence.first { $0.expert == 3 }?.reviewedMaskMember == true)
        let keptEvidence = try XCTUnwrap(layer.evidence.first { $0.expert == 1 })
        XCTAssertTrue(keptEvidence.kept)
        XCTAssertEqual(keptEvidence.ablationDelta ?? -1, 0.07, accuracy: 0.0001)
        XCTAssertEqual(keptEvidence.maskedImpactScope, "same_suite_mask_mean_text_delta")
        XCTAssertFalse(keptEvidence.reviewedMaskMember)
    }

    func test_buildPlanRejectsComparisonWithoutEvalIndexEvidence() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("missing per-prompt eval_index evidence"))
        }
    }

    func test_buildPlanRejectsEvalIndexWithoutBF16VMLXRuntimeEvidence() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(
                runtimeMode: "native_jangtq_review_bundle",
                runtimeBackend: "jangtq"
            ),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("runtime mode must be bf16_vmlx"))
        }
    }

    func test_buildPlanRejectsEvalIndexWithoutTokenDepthEvidence() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(
                meanBaselineTokens: 4,
                meanMaskedTokens: 12
            ),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("token depth"))
        }
    }

    func test_buildPlanRejectsEvalIndexSourceMismatch() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(sourceModelPath: "/tmp/other-bf16-source"),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("source model path does not match"))
        }
    }

    func test_buildPlanRejectsEvalIndexWithoutSemanticCoverageEvidence() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(
                semanticCoverage: ["code"],
                missingSemanticCoverage: []
            ),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("semantic coverage"))
            XCTAssertTrue(error.localizedDescription.contains("missing required probes"))
        }
    }

    func test_buildPlanRejectsEvalIndexWithoutSuiteFingerprintEvidence() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(suiteSHA256: nil),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("suite_jsonl fingerprint"))
        }
    }

    func test_buildPlanRejectsEvalIndexWithoutLayerStatsCoverage() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(maskedLayerStatsPromptCount: 49),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("layer-stat coverage"))
        }
    }

    func test_buildPlanRejectsEvalIndexWithoutFullHookCoverage() throws {
        let atlas = reviewedTwoExpertAtlas()
        let comparison = reviewedComparison(
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(
                hookedMOELayers: 12,
                expectedMOELayers: 40
            ),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("vMLX hook coverage 12 of 40"))
        }
    }

    func test_buildPlanRejectsComparisonDropsOutsideSafeABCandidates() throws {
        let atlas = ExpertAtlas(
            promptCount: 50,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 100,
                    probabilityMass: 100,
                    tokenCount: 20,
                    domains: ["general": 100],
                    label: "general-hot",
                    isDead: false,
                    isHot: true
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 90,
                    probabilityMass: 90,
                    tokenCount: 20,
                    domains: ["coding": 90],
                    label: "coding",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 2,
                    hits: 10,
                    probabilityMass: 10,
                    tokenCount: 20,
                    domains: ["general": 10],
                    label: "low-general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 3,
                    hits: 5,
                    probabilityMass: 5,
                    tokenCount: 20,
                    domains: ["general": 5],
                    label: "low",
                    isDead: false,
                    isHot: false
                )
            ]
        )
        let comparison = reviewedComparison(
            meanTextDelta: 0.07,
            meanLatencyDeltaPct: -8,
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 0)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(
                meanTextDelta: 0.07,
                disabledExpertCount: 1
            ),
            sourceModelPath: "/tmp/jang-unit-bf16-source"
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("outside the same-suite safe-drop set"))
            XCTAssertTrue(error.localizedDescription.contains("3"))
        }
    }

    func test_buildPlanRejectsComparisonWithMaskedPassRateBelowBaseline() throws {
        let atlas = ExpertAtlas(
            promptCount: 50,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 8,
                    probabilityMass: 8,
                    tokenCount: 10,
                    domains: ["general": 8],
                    label: "general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 2,
                    probabilityMass: 2,
                    tokenCount: 10,
                    domains: ["general": 2],
                    label: "low",
                    isDead: false,
                    isHot: false
                )
            ]
        )
        let comparison = reviewedComparison(
            maskID: "eval-a",
            passRateMasked: 0.98,
            baselineQualifiedMaskedPassRate: 0.98,
            safeDropCandidates: [ExpertCoordinate(layer: 0, expert: 1)]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 2],
            comparisonSummary: comparison
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("masked validator pass rate is below 100%"))
        }
    }

    func test_buildPlanEmbedsComparisonSummaryInExportedJSON() throws {
        let atlas = ExpertAtlas(
            promptCount: 50,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 8,
                    probabilityMass: 8,
                    tokenCount: 10,
                    domains: ["general": 8],
                    label: "general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 2,
                    probabilityMass: 2,
                    tokenCount: 10,
                    domains: ["general": 2],
                    label: "low",
                    isDead: false,
                    isHot: false
                )
            ]
        )
        let comparison = reviewedComparison(
            maskID: "eval-a",
            safeDropCandidates: [
                ExpertCoordinate(layer: 0, expert: 0),
                ExpertCoordinate(layer: 0, expert: 2)
            ]
        )

        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 3],
            comparisonSummary: comparison,
            evalIndex: reviewedEvalIndex(maskID: "eval-a"),
            sourceModelPath: "/tmp/jang-unit-bf16-source",
            evalArtifactPath: "/tmp/review-run/evals/eval-a"
        )
        XCTAssertEqual(plan.comparisonSummary?.maskID, "eval-a")
        XCTAssertEqual(plan.evalIndex?.promptCount, 50)
        XCTAssertEqual(plan.evalArtifactPath, "/tmp/review-run/evals/eval-a")

        let data = try JSONEncoder().encode(plan)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["eval_artifact"] as? String, "/tmp/review-run/evals/eval-a")
        let embedded = try XCTUnwrap(json["comparison_summary"] as? [String: Any])
        XCTAssertEqual(embedded["maskID"] as? String, "eval-a")
        XCTAssertEqual(embedded["promptCount"] as? Int, 50)
        XCTAssertEqual(try XCTUnwrap(embedded["meanTextDelta"] as? Double), 0.03, accuracy: 0.0001)
        let safe = try XCTUnwrap(embedded["safeDropCandidates"] as? [[String: Any]])
        XCTAssertEqual(safe.first?["expert"] as? Int, 0)
        let evalIndex = try XCTUnwrap(json["eval_index"] as? [String: Any])
        XCTAssertEqual(evalIndex["prompt_count"] as? Int, 50)
        XCTAssertEqual(evalIndex["risky_prompt_ids"] as? [String], [])
        XCTAssertEqual(evalIndex["suite_jsonl"] as? String, "/tmp/review-run/suite.jsonl")
        XCTAssertEqual(evalIndex["eval_jsonl"] as? String, "/tmp/review-run/evals/eval-a/eval.jsonl")
        XCTAssertEqual(evalIndex["eval_trace_jsonl"] as? String, "/tmp/review-run/evals/eval-a/eval_trace.jsonl")
        XCTAssertEqual(evalIndex["comparison_summary"] as? String, "/tmp/review-run/evals/eval-a/comparison_summary.json")
        XCTAssertEqual(evalIndex["mask"] as? String, "/tmp/review-run/evals/eval-a/mask.json")
        XCTAssertEqual(evalIndex["mask_json"] as? String, "/tmp/review-run/evals/eval-a/mask.json")
        XCTAssertEqual(
            evalIndex["semantic_coverage"] as? [String],
            ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted()
        )
        XCTAssertEqual(evalIndex["missing_semantic_coverage"] as? [String], [])
        XCTAssertEqual(evalIndex["runtime_mode"] as? String, "bf16_vmlx")
        XCTAssertEqual(evalIndex["runtime_device"] as? String, "Unit Metal")
        XCTAssertEqual(evalIndex["runtime_metal_enabled"] as? Bool, true)
        XCTAssertEqual(evalIndex["jang_tools_version"] as? String, "2.5.31")
        XCTAssertEqual(evalIndex["mlx_version"] as? String, "0.31.2")
        XCTAssertEqual(evalIndex["mlx_lm_version"] as? String, "0.31.3")
        XCTAssertEqual(evalIndex["source_model_path"] as? String, "/tmp/jang-unit-bf16-source")
        XCTAssertEqual(evalIndex["hooked_moe_layers"] as? Int, 40)
        XCTAssertEqual(evalIndex["mask_applied"] as? Bool, true)
        XCTAssertEqual(evalIndex["disabled_expert_count"] as? Int, 1)
    }

    func test_buildPlanRejectsHighRiskComparisonSummary() throws {
        let atlas = ExpertAtlas(
            promptCount: 50,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 1,
                    probabilityMass: 1,
                    tokenCount: 20,
                    domains: ["math": 1],
                    label: "math-low-activity",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 20,
                    probabilityMass: 20,
                    tokenCount: 20,
                    domains: ["general": 20],
                    label: "general-a",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 2,
                    hits: 19,
                    probabilityMass: 19,
                    tokenCount: 20,
                    domains: ["general": 19],
                    label: "general-b",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 3,
                    hits: 18,
                    probabilityMass: 18,
                    tokenCount: 20,
                    domains: ["general": 18],
                    label: "general-c",
                    isDead: false,
                    isHot: false
                )
            ]
        )
        let comparison = reviewedComparison(
            maskID: "mask-risk",
            meanTextDelta: 0.60,
            meanLatencyDeltaPct: -5,
            highRiskDomains: ["math"],
            safeDropCandidates: [
                ExpertCoordinate(layer: 0, expert: 2),
                ExpertCoordinate(layer: 0, expert: 3)
            ]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            comparisonSummary: comparison
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("high-risk domains"))
            XCTAssertTrue(error.localizedDescription.contains("math"))
        }
    }

    func test_buildPlanHonorsLockedKeepAndDetectsConflicts() throws {
        let atlas = ExpertAtlas(
            promptCount: 1,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 1,
                    probabilityMass: 1,
                    tokenCount: 4,
                    domains: [:],
                    label: "low",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 10,
                    probabilityMass: 10,
                    tokenCount: 4,
                    domains: ["coding": 10],
                    label: "coding-specialist",
                    isDead: false,
                    isHot: true
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 2,
                    hits: 9,
                    probabilityMass: 9,
                    tokenCount: 4,
                    domains: ["math": 9],
                    label: "math-specialist",
                    isDead: false,
                    isHot: true
                )
            ]
        )

        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            lockedKeepByLayer: [0: [0]]
        )

        let layer = try XCTUnwrap(plan.layers["0"])
        XCTAssertTrue(layer.keep.contains(0))
        XCTAssertEqual(layer.lockedKeep, [0])

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            forceDropByLayer: [0: [0]],
            lockedKeepByLayer: [0: [0]]
        ))
    }

    func test_buildPlanEmbedsTopKSafetyInExportedJSON() throws {
        let atlas = ExpertAtlas(
            promptCount: 2,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 6,
                    probabilityMass: 6,
                    tokenCount: 8,
                    domains: ["general": 6],
                    label: "general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 5,
                    probabilityMass: 5,
                    tokenCount: 8,
                    domains: ["coding": 5],
                    label: "coding",
                    isDead: false,
                    isHot: false
                )
            ]
        )

        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 2,
            sourceNumExpertsByLayer: [0: 4],
            trainedTopKByLayer: [0: 2]
        )

        XCTAssertEqual(plan.safety?.passed, true)
        XCTAssertEqual(plan.safety?.minimumActiveExpertsPerLayer, 2)
        XCTAssertEqual(plan.safety?.trainedTopKByLayer["0"], 2)

        let data = try JSONEncoder().encode(plan)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let safety = try XCTUnwrap(json["safety"] as? [String: Any])
        XCTAssertEqual(safety["passed"] as? Bool, true)
        XCTAssertEqual(safety["minimum_active_experts_per_layer"] as? Int, 2)
        let trainedTopK = try XCTUnwrap(safety["trained_top_k_by_layer"] as? [String: Int])
        XCTAssertEqual(trainedTopK["0"], 2)
    }

    func test_buildPlanRejectsKeepCountBelowTrainedTopK() throws {
        let atlas = ExpertAtlas(
            promptCount: 1,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 3,
                    probabilityMass: 3,
                    tokenCount: 4,
                    domains: ["general": 3],
                    label: "general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 2,
                    probabilityMass: 2,
                    tokenCount: 4,
                    domains: ["math": 2],
                    label: "math",
                    isDead: false,
                    isHot: false
                )
            ]
        )

        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: 1,
            sourceNumExpertsByLayer: [0: 4],
            trainedTopKByLayer: [0: 2]
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("top-k"))
            XCTAssertTrue(error.localizedDescription.contains("Layer 0"))
        }
    }

    private func reviewedTwoExpertAtlas(promptCount: Int = 50) -> ExpertAtlas {
        ExpertAtlas(
            promptCount: promptCount,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 0,
                    hits: 8,
                    probabilityMass: 8,
                    tokenCount: 12,
                    domains: ["general": 8],
                    label: "general",
                    isDead: false,
                    isHot: false
                ),
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 2,
                    probabilityMass: 2,
                    tokenCount: 12,
                    domains: ["general": 2],
                    label: "low",
                    isDead: false,
                    isHot: false
                )
            ]
        )
    }

    private func reviewedComparison(
        promptCount: Int = 50,
        maskID: String = "mask-safe",
        passRateBaseline: Double = 1,
        passRateMasked: Double = 1,
        baselineQualifiedMaskedPassRate: Double = 1,
        meanTextDelta: Double = 0.03,
        meanLatencyDeltaPct: Double = -4,
        highRiskDomains: [String] = [],
        degradedPromptIDs: [String] = [],
        safeDropCandidates: [ExpertCoordinate]
    ) -> ExpertComparisonSummary {
        let promptIDs = (0..<promptCount).map { "p\($0)" }
        let degraded = Set(degradedPromptIDs)
        let preserved = promptIDs.filter { !degraded.contains($0) }
        return ExpertComparisonSummary(
            baselineRunID: "run-a",
            maskID: maskID,
            promptCount: promptCount,
            passRateBaseline: passRateBaseline,
            passRateMasked: passRateMasked,
            baselineQualifiedPromptCount: promptCount,
            baselineQualifiedMaskedPassRate: baselineQualifiedMaskedPassRate,
            validatorAvailablePromptCount: promptCount,
            classificationCounts: [
                "baseline_invalid": 0,
                "preserved": preserved.count,
                "degraded": degradedPromptIDs.count,
                "inconclusive": 0
            ],
            baselineQualifiedPromptIDs: promptIDs,
            baselineInvalidPromptIDs: [],
            inconclusivePromptIDs: [],
            preservedPromptIDs: preserved,
            degradedPromptIDs: degradedPromptIDs,
            baselineQualifiedSemanticCoverage: ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted(),
            missingBaselineQualifiedSemanticCoverage: [],
            meanTextDelta: meanTextDelta,
            meanLatencyDeltaPct: meanLatencyDeltaPct,
            highRiskDomains: highRiskDomains,
            safeDropCandidates: safeDropCandidates
        )
    }

    private func reviewedEvalIndex(
        promptCount: Int = 50,
        maskID: String = "mask-safe",
        passRateBaseline: Double = 1,
        passRateMasked: Double = 1,
        meanTextDelta: Double = 0.03,
        sourceModelPath: String = "/tmp/jang-unit-bf16-source",
        runtimeMode: String = "bf16_vmlx",
        runtimeBackend: String = "vmlx",
        runtimeDevice: String = "Unit Metal",
        runtimeMetalEnabled: Bool = true,
        meanBaselineTokens: Double = 12,
        meanMaskedTokens: Double = 12,
        baselineRouteRecordCount: Int? = nil,
        maskedRouteRecordCount: Int? = nil,
        baselineLayerStatsPromptCount: Int? = nil,
        maskedLayerStatsPromptCount: Int? = nil,
        generationSettingsChecked: Bool = true,
        suiteSHA256: String? = "unit-suite-sha",
        semanticCoverage: [String] = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted(),
        missingSemanticCoverage: [String]? = [],
        baselineQualifiedSemanticCoverage: [String]? = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted(),
        missingBaselineQualifiedSemanticCoverage: [String]? = [],
        degradedPromptIDs: [String] = [],
        baselineQualifiedMaskedPassRate: Double = 1,
        hookedMOELayers: Int? = 40,
        expectedMOELayers: Int? = 40,
        hookCoverageComplete: Bool? = true,
        maskApplied: Bool = true,
        disabledExpertCount: Int = 1
    ) -> ExpertEvalIndexSummary {
        let promptIDs = (0..<promptCount).map { "p\($0)" }
        let degraded = Set(degradedPromptIDs)
        let preserved = promptIDs.filter { !degraded.contains($0) }
        return ExpertEvalIndexSummary(
            promptCount: promptCount,
            promptIDs: promptIDs,
            riskyPromptIDs: [],
            highRiskDomains: [],
            passRateBaseline: passRateBaseline,
            passRateMasked: passRateMasked,
            validatorSchema: "jang-expert-lab-validator-v1",
            validatorAvailablePromptCount: promptCount,
            promptClassificationCounts: [
                "baseline_invalid": 0,
                "preserved": preserved.count,
                "degraded": degradedPromptIDs.count,
                "inconclusive": 0
            ],
            baselineQualifiedPromptCount: promptCount,
            baselineQualifiedPromptIDs: promptIDs,
            baselineInvalidPromptIDs: [],
            inconclusivePromptIDs: [],
            preservedPromptIDs: preserved,
            degradedPromptIDs: degradedPromptIDs,
            baselineQualifiedMaskedPassRate: baselineQualifiedMaskedPassRate,
            baselineQualifiedSemanticCoverage: baselineQualifiedSemanticCoverage,
            missingBaselineQualifiedSemanticCoverage: missingBaselineQualifiedSemanticCoverage,
            meanTextDelta: meanTextDelta,
            minBaselineTokens: 8,
            minMaskedTokens: 8,
            meanBaselineTokens: meanBaselineTokens,
            meanMaskedTokens: meanMaskedTokens,
            baselineRouteRecordCount: baselineRouteRecordCount ?? promptCount,
            maskedRouteRecordCount: maskedRouteRecordCount ?? promptCount,
            baselineLayerStatsPromptCount: baselineLayerStatsPromptCount ?? promptCount,
            maskedLayerStatsPromptCount: maskedLayerStatsPromptCount ?? promptCount,
            generationSettingsChecked: generationSettingsChecked,
            suiteJSONL: "/tmp/review-run/suite.jsonl",
            suiteSHA256: suiteSHA256,
            evalJSONL: "/tmp/review-run/evals/\(maskID)/eval.jsonl",
            evalTraceJSONL: "/tmp/review-run/evals/\(maskID)/eval_trace.jsonl",
            comparisonSummary: "/tmp/review-run/evals/\(maskID)/comparison_summary.json",
            mask: "/tmp/review-run/evals/\(maskID)/mask.json",
            maskJSON: "/tmp/review-run/evals/\(maskID)/mask.json",
            semanticCoverage: semanticCoverage,
            missingSemanticCoverage: missingSemanticCoverage,
            runtimeMode: runtimeMode,
            runtimeBackend: runtimeBackend,
            runtimeDevice: runtimeDevice,
            runtimeMetalEnabled: runtimeMetalEnabled,
            jangToolsVersion: "2.5.31",
            mlxVersion: "0.31.2",
            mlxLMVersion: "0.31.3",
            sourceModelPath: sourceModelPath,
            hookedMOELayers: hookedMOELayers,
            expectedMOELayers: expectedMOELayers,
            hookCoverageComplete: hookCoverageComplete,
            maskApplied: maskApplied,
            disabledExpertCount: disabledExpertCount,
            regressionSeverity: "none"
        )
    }
}
