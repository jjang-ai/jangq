// JANGStudio/Tests/JANGStudioTests/PostConvertVerifierTests.swift
import CryptoKit
import XCTest
@testable import JANGStudio

final class PostConvertVerifierTests: XCTestCase {
    private func fixture(_ name: String) -> URL {
        Bundle(for: Self.self).url(forResource: name, withExtension: nil, subdirectory: nil)
            ?? Bundle(for: Self.self).bundleURL.appendingPathComponent("Fixtures/\(name)")
    }

    func test_goodOutputAllRequiredPass() async throws {
        let url = fixture("good_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let requiredFails = checks.filter { $0.required && $0.status == .fail }
        XCTAssertTrue(requiredFails.isEmpty, "unexpected required fails: \(requiredFails.map(\.id))")
        XCTAssertTrue(checks.contains { $0.id == .generationConfig && $0.status == .pass },
                      "good output should have generation_config.json")
    }

    func test_jangtqOutputWithTokenizerConfigSpecialTokensPassesVerify() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-jangtq-verify-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try """
        {"model_type":"qwen3_5_moe","weight_format":"mxtq","mxtq_bits":4,"text_config":{"model_type":"qwen3_5_moe_text","num_hidden_layers":40,"num_experts":256,"dtype":"bfloat16"}}
        """.write(to: url.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try """
        {"version":2,"weight_format":"mxtq","profile":"JANGTQ4","source_model":{"name":"Qwen3.6-35B-A3B","architecture":"qwen3_5_moe_text"},"quantization":{"method":"affine+mxtq","group_size":64,"bits_default":4},"capabilities":{"reasoning_parser":"qwen3","tool_parser":"qwen","think_in_template":true,"supports_tools":true,"supports_thinking":true,"family":"qwen3_5_moe","modality":"text","cache_type":"hybrid"}}
        """.write(to: url.appendingPathComponent("jang_config.json"), atomically: true, encoding: .utf8)
        try "{\"model\":{\"type\":\"BPE\"}}".write(
            to: url.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )
        try """
        {"chat_template":"{% for m in messages %}{{m.content}}{% endfor %}","tokenizer_class":"Qwen2Tokenizer","eos_token":"<|im_end|>","pad_token":"<|endoftext|>","additional_special_tokens":["<|im_start|>","<|im_end|>","<|vision_start|>","<|video_pad|>"]}
        """.write(to: url.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        try "{\"metadata\":{\"format\":\"jangtq\",\"total_size\":0},\"weight_map\":{\"language_model.model.embed_tokens.weight\":\"model-00001-of-00001.safetensors\"}}".write(
            to: url.appendingPathComponent("model.safetensors.index.json"),
            atomically: true,
            encoding: .utf8
        )
        FileManager.default.createFile(
            atPath: url.appendingPathComponent("model-00001-of-00001.safetensors").path,
            contents: Data()
        )

        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: false, dtype: .bf16, totalBytes: 0, shardCount: 1)

        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)

        XCTAssertEqual(checks.first { $0.id == .jangConfigFormat }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .tokenizerFiles }?.status, .pass)
        let requiredFails = checks.filter { $0.required && $0.status == .fail }
        XCTAssertTrue(requiredFails.isEmpty, "unexpected required fails: \(requiredFails.map(\.id))")
    }

    func test_temporaryReviewRuntimeSoftensTokenizerAndChatTemplateGates() async throws {
        let url = try makeSparseJANGTQOutput()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let plan = ConversionPlan()
        plan.outputURL = url
        plan.sourceURL = url.deletingLastPathComponent().appendingPathComponent("source", isDirectory: true)
        plan.expertReviewIntent = .smartPrequantPrune
        plan.expertReviewBundleURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 64, isVL: false,
                              isVideoVL: false, hasGenerationConfig: false, dtype: .bf16, totalBytes: 0, shardCount: 1)

        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)

        XCTAssertEqual(checks.first { $0.id == .chatTemplate }?.status, .warn)
        XCTAssertEqual(checks.first { $0.id == .chatTemplate }?.required, false)
        XCTAssertEqual(checks.first { $0.id == .tokenizerFiles }?.status, .warn)
        XCTAssertEqual(checks.first { $0.id == .tokenizerFiles }?.required, false)
        let requiredFails = checks.filter { $0.required && $0.status == .fail }
        XCTAssertTrue(requiredFails.isEmpty, "temporary review runtime should still open Expert Lab: \(requiredFails.map(\.id))")
    }

    func test_finalOutputStillRequiresTokenizerAndChatTemplateGates() async throws {
        let url = try makeSparseJANGTQOutput()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 64, isVL: false,
                              isVideoVL: false, hasGenerationConfig: false, dtype: .bf16, totalBytes: 0, shardCount: 1)

        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)

        XCTAssertEqual(checks.first { $0.id == .chatTemplate }?.status, .fail)
        XCTAssertEqual(checks.first { $0.id == .chatTemplate }?.required, true)
        XCTAssertEqual(checks.first { $0.id == .tokenizerFiles }?.status, .fail)
        XCTAssertEqual(checks.first { $0.id == .tokenizerFiles }?.required, true)
    }

    func test_brokenOutputFlagsChatTemplateAndShardMismatch() async throws {
        let url = fixture("broken_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 2)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let failedIDs = checks.filter { $0.status == .fail }.map { $0.id }
        XCTAssertTrue(failedIDs.contains(.chatTemplate))
        XCTAssertTrue(failedIDs.contains(.shardsMatchIndex))
    }

    private func makeSparseJANGTQOutput() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-jangtq-sparse-\(UUID().uuidString)", isDirectory: true)
        let url = root.appendingPathComponent("review", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try """
        {"model_type":"qwen3_5_moe","num_hidden_layers":2,"num_experts":64,"torch_dtype":"bfloat16","quantization":{"group_size":64,"bits":3}}
        """.write(to: url.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try """
        {"version":2,"weight_format":"mxtq","profile":"JANGTQ3","source_model":{"name":"qwen-moe-bf16","architecture":"qwen3_5_moe"},"quantization":{"method":"affine+mxtq","group_size":64,"bits_default":3},"capabilities":{"reasoning_parser":"qwen3","tool_parser":"qwen","think_in_template":true,"supports_tools":true,"supports_thinking":true,"family":"qwen3_5_moe","modality":"text","cache_type":"hybrid"}}
        """.write(to: url.appendingPathComponent("jang_config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: url.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try #"{"tokenizer_class":"PreTrainedTokenizerFast","model_max_length":131072}"#
            .write(to: url.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        try #"{"metadata":{"format":"jangtq","total_size":0},"weight_map":{"language_model.model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#
            .write(to: url.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        FileManager.default.createFile(
            atPath: url.appendingPathComponent("model-00001-of-00001.safetensors").path,
            contents: Data()
        )
        return url
    }

    func test_reviewedPruneChecksRequireVerifiedPrunedSourceAndWriteFinalReport() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-verify-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")
        plan.expertReviewPruneReportURL = pruned.appendingPathComponent("expert_lab_prune_report.md")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)

        XCTAssertEqual(checks.first { $0.id == .reviewedPruneSource }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .reviewedPruneVerification }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .reviewedPruneSameSuite }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .reviewedPrunePlan }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .expertLabFinalReport }?.status, .pass)
        let report = output.appendingPathComponent("expert_lab_final_report.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: report.path))
        let text = try String(contentsOf: report, encoding: .utf8)
        XCTAssertTrue(text.contains("Expert Lab Final Conversion Report"))
        XCTAssertTrue(text.contains(pruned.path))
        XCTAssertTrue(text.contains("Mask comparison runs: 1"))
        XCTAssertTrue(text.contains("Pruned BF16/F16 same-suite generation: ready"))
        XCTAssertTrue(text.contains("expert_lab_pruned_generation_summary.json"))
        XCTAssertTrue(text.contains("Pruned-vs-reviewed masked comparison: 50 prompts"))
        XCTAssertTrue(text.contains("50 baseline-qualified"))
        XCTAssertTrue(text.contains("qualified pass 1.00"))
        XCTAssertTrue(text.contains("Pruned vMLX runtime source: \(pruned.path)"))
    }


    func test_intentPrunePlanChecksPassWithoutSemanticEvidenceRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("intent-prune-verify-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validIntentPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        XCTAssertEqual(checks.first { $0.id == .reviewedPrunePlan }?.status, .pass, checks.first { $0.id == .reviewedPrunePlan }?.hint ?? "")
        XCTAssertEqual(checks.first { $0.id == .reviewedPruneSource }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .reviewedPruneVerification }?.status, .pass)
        XCTAssertEqual(checks.first { $0.id == .reviewedPruneSameSuite }?.status, .pass)
    }

    func test_intentPrunePlanChecksRequireCrackPackWhenStanceIsCrack() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("intent-prune-crack-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validIntentPrunePlanJSON(safetyStance: "crack", includeCrackPack: false)
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let planCheck = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(planCheck.status, .fail)
        XCTAssertTrue(planCheck.hint?.contains("crack_pack") == true, planCheck.hint ?? "")
    }

    func test_reviewedPruneChecksRejectComparisonSafeDropMaskMismatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-verify-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_comparison_summary.json")) { json in
            json["safeDropCandidates"] = [["layer": 0, "expert": 0]]
        }

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = planURL

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)

        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(
            sameSuite.hint?.contains("safe-drop candidates do not match mask.json disabled experts") == true,
            sameSuite.hint ?? ""
        )
    }

    func test_reviewedPruneChecksRejectPlanDropsOutsideSameSuiteSafeDrops() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-verify-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try rewriteEmbeddedPlanDropExperts(planURL, experts: [0])

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = planURL

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)

        XCTAssertEqual(checks.first { $0.id == .reviewedPruneSameSuite }?.status, .pass)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(
            prunePlan.hint?.contains("drops experts outside same-suite safe-drop candidates") == true,
            prunePlan.hint ?? ""
        )
    }

    func test_reviewedPruneChecksRejectPrunePlanEvalIndexSidecarDrift() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-verify-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try reorderEmbeddedPlanPromptIDs(planURL)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = planURL
        plan.expertReviewPruneReportURL = pruned.appendingPathComponent("expert_lab_prune_report.md")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)

        XCTAssertEqual(checks.first { $0.id == .reviewedPruneSameSuite }?.status, .pass)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(
            prunePlan.hint?.contains("embedded eval_index does not match expert_lab_eval_index.json") == true,
            prunePlan.hint ?? ""
        )
        XCTAssertTrue(prunePlan.hint?.contains("prompt IDs differ") == true, prunePlan.hint ?? "")
    }

    func test_reviewedPruneChecksRejectMismatchedDecodeSettingsRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-decode-settings-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let evalRows = (0..<50)
            .map { index in
                #"{"promptID":"p\#(index)","domain":"general","semanticDomains":[\#(Self.semanticDomainsJSON(index: index))],"expectedKind":"exact","expected":"hello","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"baselineGenerationSettings":{"max_tokens":96,"temperature":0.0,"top_p":1.0,"top_k":0},"maskedGenerationSettings":{"max_tokens":97,"temperature":0.0,"top_p":1.0,"top_k":0},"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"#
            }
            .joined(separator: "\n")
            .appending("\n")
        try evalRows.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval.jsonl baseline/masked decode settings do not match") == true)
    }

    func test_reviewedPruneChecksRejectComparisonSummaryDriftFromEvalRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-comparison-drift-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try markFirstEvalRowHighRisk(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(
            sameSuite.hint?.contains("comparison summary high-risk domains do not match eval.jsonl") == true,
            sameSuite.hint ?? "missing hint"
        )
    }

    func test_reviewedPruneChecksRejectEvalIndexSuiteFingerprintDrift() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-eval-index-suite-fingerprint-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let originalSuite = try String(contentsOf: suite, encoding: .utf8)
        try originalSuite
            .replacingOccurrences(
                of: "Return only the number: 17 * 23.",
                with: "Return only the number: 19 * 29."
            )
            .write(to: suite, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("eval_index.json suite.jsonl fingerprint does not match suite.jsonl") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSuiteFingerprintDrift() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-suite-fingerprint-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
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

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source reviewed prompt suite fingerprint does not match reviewed suite") == true)
    }

    func test_reviewedPruneChecksRejectExternalReviewSidecarPaths() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-external-sidecars-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let externalEvidence = root.appendingPathComponent("external-evidence", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: externalEvidence, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: externalEvidence)
        try FileManager.default.copyItem(
            at: externalEvidence.appendingPathComponent("expert_lab_review_summary.json"),
            to: pruned.appendingPathComponent("expert_lab_review_summary.json")
        )

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")
        plan.expertReviewPruneReportURL = pruned.appendingPathComponent("expert_lab_prune_report.md")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(
            sameSuite.hint?.contains("review summary pruned source path does not match the selected pruned BF16/F16 source") == true
        )
    }

    func test_reviewedPruneChecksRejectReviewSummaryPrunedSourceMismatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-review-source-mismatch-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let otherPruned = root.appendingPathComponent("other-pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: otherPruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_review_summary.json")) { json in
            json["pruned_source"] = otherPruned.path
        }

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(
            sameSuite.hint?.contains("review summary pruned source path does not match the selected pruned BF16/F16 source") == true
        )
    }

    func test_reviewedPruneChecksRejectPrunedGenerationSummarySourceMismatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-generation-source-mismatch-\(UUID().uuidString)", isDirectory: true)
        let original = root.appendingPathComponent("original", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let otherPruned = root.appendingPathComponent("other-pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: otherPruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")) { json in
            json["pruned_source"] = otherPruned.path
        }

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = original
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(
            sameSuite.hint?.contains("pruned-source generation summary path does not match the selected pruned BF16/F16 source") == true
        )
    }

    func test_reviewedPruneChecksRejectSuiteWithoutRequiredSemanticProbes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-language-coverage-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned, includeRequiredSemanticProbes: false)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("suite.jsonl is missing required semantic prompt probes") == true)
    }

    func test_reviewedPruneChecksRequirePlanSidecarInPrunedSource() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-plan-sidecar-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        let reviewRun = root.appendingPathComponent("review-run", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: reviewRun, withIntermediateDirectories: true)
        let externalPlanURL = reviewRun.appendingPathComponent("prune_plan.json")
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: externalPlanURL, atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = externalPlanURL

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("missing prune_plan.json") == true)
    }

    func test_reviewedPruneChecksRejectPlanWithoutSemanticEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-semantic-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON(includeSemanticEvidence: false)
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("activation lift evidence") == true)
    }

    func test_reviewedPruneChecksRejectPlanWithEmptyPromptTags() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-prompt-tags-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON(includePromptTags: false)
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("prompt tags/examples") == true)
    }

    func test_reviewedPruneChecksRejectPlanWithoutMaskedImpactEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-masked-impact-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON(includeMaskedImpact: false)
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("masked-output impact evidence") == true)
    }

    func test_reviewedPruneChecksRejectPlanWithoutMaskedImpactScope() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-masked-impact-scope-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON(includeMaskedImpactScope: false)
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("masked-output impact scope evidence") == true)
    }

    func test_reviewedPruneChecksRejectMissingPrunedSourceSuiteVerification() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-missing-pruned-suite-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try """
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
            .write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned BF16/F16 same-suite vMLX verification has not run yet") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSourceGenerationPromptIDDrift() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-pruned-suite-id-drift-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let generations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let drifted = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"other\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"}}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try drifted.write(to: generations, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source generation missing suite prompt IDs") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSourceGenerationRowsMissingDecodeSettings() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-pruned-suite-row-decode-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")) { json in
            json["generation_defaults"] = [
                "max_tokens": 96,
                "temperature": 0.0
            ]
        }

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source generation row is missing decode settings evidence") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSourceGenerationRowsMissingRuntimeEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-pruned-suite-row-runtime-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let generations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let stripped = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try stripped.write(to: generations, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source generation is missing per-prompt runtime evidence") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSourceGenerationRowsMissingLayerStatsWhenLayerCountKnown() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-pruned-suite-row-layer-stats-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeKnownLayerHookEvidence(in: pruned, layerCount: 2)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source generation is missing per-prompt routed-layer stats") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSourceGenerationRowsMissingTokenTraceWhenLayerCountKnown() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-pruned-suite-row-token-trace-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeKnownLayerHookEvidence(in: pruned, layerCount: 1)
        try writePrunedGenerationsWithLayerStats(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source generation is missing per-prompt token_trace routing evidence") == true)
    }

    func test_reviewedPruneChecksRejectPartialPrunedSourceVMLXHookCoverage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-pruned-suite-hooks-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let comparison = pruned.appendingPathComponent("expert_lab_comparison_summary.json")
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTrace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndex = pruned.appendingPathComponent("expert_lab_eval_index.json")
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = pruned.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        let promptIDs = Self.promptIDJSON(count: 50)
        let suiteSHA256 = try Self.fileSHA256(suite)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_semantic_coverage":[],"validator_schema":"jang-expert-lab-validator-v1","validator_available_prompt_count":50,"prompt_classification_counts":{\(Self.classificationCountsJSON(count: 50))},"baseline_qualified_prompt_count":50,"baseline_qualified_prompt_ids":[\(promptIDs)],"baseline_invalid_prompt_ids":[],"inconclusive_prompt_ids":[],"preserved_prompt_ids":[\(promptIDs)],"degraded_prompt_ids":[],"baseline_qualified_masked_pass_rate":1.0,"baseline_qualified_semantic_coverage":[\(Self.requiredSemanticCoverageJSON())],"missing_baseline_qualified_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"generation_settings_checked":true,"suite_sha256":"\(suiteSHA256)","eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"hooked_moe_layers":40,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: evalIndex, atomically: true, encoding: .utf8)
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
          "hooked_moe_layers": 12,
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
        try writePrunedGenerationsWithLayerStats(in: pruned, layerCount: 40, includeTokenTrace: true)
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

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(
            sameSuite.hint?.contains("pruned-source generation vMLX hook coverage 12 of 40 routed layers") == true,
            sameSuite.hint ?? ""
        )
    }

    func test_reviewedPruneChecksRejectPlanSafetyBelowTopK() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-plan-safety-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON(keepExperts: 4, trainedTopK: 8)
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("trained top-k") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexPromptOrderMismatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-eval-order-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeEvalIndexPromptOrder(in: pruned, indices: Array((0..<50).reversed()))

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval_index.json prompt order does not match suite.jsonl") == true)
    }

    func test_reviewedPruneChecksRejectPrunedSourceGenerationPromptOrderMismatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-generation-order-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writePrunedGenerationPromptOrder(in: pruned, indices: [1, 0] + Array(2..<50))

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("pruned-source generation prompt order does not match reviewed suite") == true)
    }

    func test_reviewedPruneChecksRejectSafetyOnlyPlanWithoutEmbeddedSameSuiteEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-plan-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.safetyOnlyReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let prunePlan = try XCTUnwrap(checks.first { $0.id == .reviewedPrunePlan })
        XCTAssertEqual(prunePlan.status, .fail)
        XCTAssertTrue(prunePlan.hint?.contains("embedded same-suite comparison evidence") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexWithoutTokenDepthEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-token-depth-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"native_jangtq_review_bundle","runtime_backend":"jangtq","runtime_device":"Unit Metal","runtime_metal_enabled":true}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("generation-depth token evidence") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexWithoutRuntimeEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-runtime-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json"}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("runtime device evidence") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexWithoutPackageVersionEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-version-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":50,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("vMLX package version evidence") == true)
    }

    func test_reviewedPruneChecksRejectTopKOnlyEvalRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-topk-only-rows-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let evalRows = (0..<50)
            .map { #"{"promptID":"p\#($0)","baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"runtimeMode":"bf16_vmlx","runtimeBackend":"vmlx","runtimeDevice":"Unit Metal","runtimeMetalEnabled":true,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":0,"topKOverride":4,"risk":"none","regressionSeverity":"none"}"# }
            .joined(separator: "\n")
            .appending("\n")
        try evalRows.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("top-k-only comparisons cannot authorize hard pruning") == true)
    }

    func test_reviewedPruneChecksRejectEvalRowsMissingRuntimeEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-row-runtime-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let evalRows = (0..<50)
            .map { #"{"promptID":"p\#($0)","baselineText":"hello","maskedText":"hello","textDelta":0.0,"baselineTokenCount":12,"maskedTokenCount":12,"baselineRouteRecordCount":1,"maskedRouteRecordCount":1,"jangToolsVersion":"2.5.31","mlxVersion":"0.31.2","mlxLMVersion":"0.31.3","sourceModelPath":"/tmp/jang-unit-bf16-source","maskApplied":true,"disabledExpertCount":1,"risk":"none","regressionSeverity":"none"}"# }
            .joined(separator: "\n")
            .appending("\n")
        try evalRows.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval.jsonl is missing per-prompt runtime device evidence") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexWithoutRoutingEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-route-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("routing record evidence") == true)
    }

    func test_reviewedPruneChecksRejectEvalTraceRouteCountMismatch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-trace-count-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json["masked_route_record_count"] = 51
        }

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval_trace.jsonl has 50 masked routing records for 51 indexed masked route records") == true)
    }

    func test_reviewedPruneChecksRejectPartialRouteCoverage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-partial-route-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let promptIDs = Self.promptIDJSON(count: 50)
        try """
        {"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":[\(promptIDs)],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"min_baseline_tokens":12,"min_masked_tokens":12,"mean_baseline_tokens":12.0,"mean_masked_tokens":12.0,"baseline_route_record_count":50,"masked_route_record_count":49,"eval_jsonl":"expert_lab_eval.jsonl","eval_trace_jsonl":"expert_lab_eval_trace.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"bf16_vmlx","runtime_backend":"vmlx","runtime_device":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"/tmp/jang-unit-bf16-source","mask_applied":true,"disabled_expert_count":1}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("routing record evidence for every indexed prompt") == true)
    }

    func test_reviewedPruneChecksRejectPartialEvalIndexLayerStatsCoverage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-layer-stat-index-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try updateJSONFile(pruned.appendingPathComponent("expert_lab_eval_index.json")) { json in
            json["baseline_layer_stats_prompt_count"] = 50
            json["masked_layer_stats_prompt_count"] = 49
        }

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval_index.json layer-stat coverage is incomplete for indexed prompts") == true)
    }

    func test_reviewedPruneChecksRejectPartialEvalRowLayerStatsEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-layer-stat-row-evidence-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try writeEvalRowsWithPartialLayerStats(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval.jsonl layer-stat evidence is incomplete for baseline/masked prompts") == true)
    }

    func test_finalComparisonFailsWhenPostQuantSummaryPromptCountDoesNotMatchRows() throws {
        let root = try sizeSanityDir("expert-final-comparison-summary-count")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try """
        {"id":"p1","text":"2+2"}
        {"id":"p2","text":"Say hi"}

        """.write(to: suite, atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "2+2", ok: true, text: "4", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil),
                ExpertLabSmokeRecord(promptID: "p2", prompt: "Say hi", ok: true, text: "hi", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )
        try """
        {
          "schema": "jang-expert-lab-post-quant-smoke-v1",
          "source": "reviewed-suite",
          "suite_jsonl": "\(suite.path)",
          "prompt_count": 1,
          "passed_count": 2,
          "failed": [],
          "prompt_ids": ["p1", "p2"],
          "artifact": "\(output.appendingPathComponent("expert_lab_smoke.jsonl").path)"
        }
        """.write(to: output.appendingPathComponent("expert_lab_smoke_summary.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("summary records 1 prompts for 2 generated rows") == true)
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        let smoke = try XCTUnwrap(json["native_smoke"] as? [String: Any])
        XCTAssertEqual(smoke["summary_prompt_count"] as? Int, 1)
    }

    func test_finalComparisonFailsWhenPostQuantPromptOrderDoesNotMatchReviewedSuite() throws {
        let root = try sizeSanityDir("expert-final-comparison-prompt-order")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try """
        {"id":"p1","text":"2+2"}
        {"id":"p2","text":"Say hi"}

        """.write(to: suite, atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p2", prompt: "Say hi", ok: true, text: "hi", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil),
                ExpertLabSmokeRecord(promptID: "p1", prompt: "2+2", ok: true, text: "4", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("post-quant reviewed prompt suite order does not match reviewed suite") == true)
    }

    func test_finalComparisonFailsWhenPostQuantDecodeSettingsDoNotMatchReviewedSuite() throws {
        let root = try sizeSanityDir("expert-final-comparison-decode-settings")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try """
        {"id":"p1","text":"2+2","max_new_tokens":5,"temperature":0.2}

        """.write(to: suite, atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(
                    promptID: "p1",
                    prompt: "2+2",
                    generationSettings: ExpertLabSmokeGenerationSettings(maxTokens: 5, temperature: 0.0),
                    ok: true,
                    text: "4",
                    tokens: 1,
                    tokensPerSec: 20,
                    elapsedS: 0.1,
                    error: nil
                )
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("post-quant temperature for p1 does not match reviewed suite") == true)
    }

    func test_reviewedPruneChecksRejectContradictoryVerification() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-verify-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":false,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let verification = try XCTUnwrap(checks.first { $0.id == .reviewedPruneVerification })
        XCTAssertEqual(verification.status, .fail)
        XCTAssertTrue(verification.hint?.contains("router_rows_match") == true)
    }

    func test_reviewedPruneChecksRejectHighRiskSameSuiteComparison() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-risk-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(
            in: pruned,
            comparisonJSON: #"{"promptCount":50,"passRateBaseline":1.0,"passRateMasked":1.0,"meanTextDelta":0.7,"highRiskDomains":["math","safety"],"safeDropCandidates":[]}"#,
            evalRows: 50
        )

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("high-risk domains") == true)
    }

    func test_reviewedPruneChecksRejectIncompletePerPromptEvalRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-eval-gap-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned, evalRows: 12)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("eval.jsonl has 12 rows for 50 compared prompts") == true)
    }

    func test_reviewedPruneChecksRejectExtraSuiteRowsBeyondComparisonSummary() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-extra-suite-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try expandReviewSidecarsTo51Prompts(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.hint?.contains("suite.jsonl has 51 rows for 50 compared prompts") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexWithoutPromptIDCoverage() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-eval-index-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        try #"{"schema":"jang-expert-lab-eval-index-v1","prompt_count":50,"prompt_ids":["p0"],"risky_prompt_ids":[],"high_risk_domains":[],"semantic_coverage":["math","code","formatting","instruction_following","reasoning","safety_medical_legal_sensitive","chinese","non_english","multilingual","translation","english_dominant","unknown_language_role"],"missing_semantic_coverage":[],"eval_jsonl":"expert_lab_eval.jsonl","comparison_summary":"expert_lab_comparison_summary.json","mask":"mask.json","runtime_mode":"native_jangtq_review_bundle","runtime_backend":"jangtq","runtime_device":"Unit Metal","runtime_metal_enabled":true}"#
            .write(to: pruned.appendingPathComponent("expert_lab_eval_index.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("lists 1 prompt IDs for 50 indexed prompts") == true)
    }

    func test_reviewedPruneChecksRejectEvalIndexPromptIDsMissingFromEvalRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-eval-id-mismatch-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let mismatchedEvalText = (0..<50)
            .map { #"{"promptID":"q\#($0)","baselineText":"hello","maskedText":"hello"}"# }
            .joined(separator: "\n")
            .appending("\n")
        try mismatchedEvalText.write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("missing from eval.jsonl") == true)
    }

    func test_reviewedPruneChecksRejectEvalRowsOutsideIndexedSuite() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-extra-eval-row-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let eval = pruned.appendingPathComponent("expert_lab_eval.jsonl")
        let extra = #"{"promptID":"outside-suite","baselineText":"hello","maskedText":"hello","baselineRouteRecordCount":1,"maskedRouteRecordCount":1}"#
        var rows = try String(contentsOf: eval, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map(String.init)
        rows[49] = extra
        try rows.joined(separator: "\n")
            .appending("\n")
            .write(to: eval, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("eval.jsonl prompt IDs outside eval_index.json") == true)
    }

    func test_reviewedPruneChecksRejectTraceRowsOutsideIndexedSuite() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-extra-trace-row-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let trace = pruned.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let extra = #"{"promptID":"outside-suite","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"#
        try (String(contentsOf: trace, encoding: .utf8) + extra + "\n")
            .write(to: trace, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("eval_trace.jsonl prompt IDs outside eval_index.json") == true)
    }

    func test_reviewedPruneChecksRejectTraceMissingMaskedVariant() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-missing-masked-trace-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let baselineOnlyTrace = (0..<50)
            .map { #"{"promptID":"p\#($0)","domain":"general","variant":"baseline","record":{"tokenIndex":0,"layer":0,"selectedExperts":[0],"scores":[1.0],"disabledExperts":[],"effectiveTopK":1}}"# }
            .joined(separator: "\n")
            .appending("\n")
        try baselineOnlyTrace.write(to: pruned.appendingPathComponent("expert_lab_eval_trace.jsonl"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("missing masked routing records") == true)
    }

    func test_reviewedPruneChecksRejectMaskedTraceWithoutMaskEvidence() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-masked-trace-no-mask-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
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

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("masked routing records are missing mask evidence") == true)
    }

    func test_reviewedPruneChecksRejectMaskedTraceSelectingDisabledExperts() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-masked-trace-leak-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
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

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("selected disabled experts") == true)
    }

    func test_reviewedPruneChecksRejectMaskedTraceThatDoesNotMatchMaskJSON() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-prune-masked-trace-wrong-mask-\(UUID().uuidString)", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
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

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = PostConvertVerifier.reviewedPruneChecks(plan: plan, outputDir: output)
        let sameSuite = try XCTUnwrap(checks.first { $0.id == .reviewedPruneSameSuite })
        XCTAssertEqual(sameSuite.status, .fail)
        XCTAssertTrue(sameSuite.required)
        XCTAssertTrue(sameSuite.hint?.contains("mask.json evidence") == true)
    }

    func test_expertLabSmokeArtifactPassesOnlyWhenAllPromptsPass() throws {
        let dir = try sizeSanityDir("expert-smoke")
        let suite = dir.appendingPathComponent("expert_lab_suite.jsonl")
        defer { try? FileManager.default.removeItem(at: dir) }
        try """
        {"id":"a","domain":"math","text":"2+2","tags":["arithmetic"]}
        {"id":"b","domain":"general","text":"say hi","tags":["english_dominant"]}
        """
            .write(to: suite, atomically: true, encoding: .utf8)
        let passCheck = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(
                    promptID: "a",
                    prompt: "2+2",
                    generationSettings: ExpertLabSmokeGenerationSettings(maxTokens: 96, temperature: 0.2),
                    ok: true,
                    text: "4",
                    tokens: 1,
                    tokensPerSec: 12,
                    elapsedS: 0.1,
                    error: nil
                ),
                ExpertLabSmokeRecord(
                    promptID: "b",
                    prompt: "say hi",
                    ok: true,
                    text: "hi",
                    tokens: 1,
                    tokensPerSec: 10,
                    elapsedS: 0.1,
                    error: nil
                )
            ],
            outputDir: dir,
            source: "reviewed-suite",
            suiteURL: suite
        )
        XCTAssertEqual(passCheck.status, .pass)
        let artifact = dir.appendingPathComponent("expert_lab_smoke.jsonl")
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifact.path))
        let artifactLines = try String(contentsOf: artifact, encoding: .utf8).split(whereSeparator: \.isNewline)
        let firstRecord = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(artifactLines[0].utf8)) as? [String: Any])
        let generationSettings = try XCTUnwrap(firstRecord["generation_settings"] as? [String: Any])
        XCTAssertEqual(generationSettings["max_tokens"] as? Int, 96)
        XCTAssertEqual(generationSettings["temperature"] as? Double, 0.2)
        let summaryURL = dir.appendingPathComponent("expert_lab_smoke_summary.json")
        let summaryData = try Data(contentsOf: summaryURL)
        let summary = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summary["source"] as? String, "reviewed-suite")
        XCTAssertEqual(summary["suite_jsonl"] as? String, suite.path)
        XCTAssertEqual((summary["suite_sha256"] as? String)?.count, 64)
        XCTAssertEqual(summary["prompt_count"] as? Int, 2)
        XCTAssertEqual(summary["prompt_ids"] as? [String], ["a", "b"])
        let runtimeInfo = try XCTUnwrap(summary["runtime_info"] as? [String: Any])
        XCTAssertEqual(runtimeInfo["runtime_mode"] as? String, "post_quant_converted")
        XCTAssertEqual(runtimeInfo["model_path"] as? String, dir.path)
        XCTAssertEqual(runtimeInfo["output_path"] as? String, dir.path)

        let failCheck = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(
                    promptID: "bad",
                    prompt: "empty",
                    ok: false,
                    text: nil,
                    tokens: 0,
                    tokensPerSec: 0,
                    elapsedS: 0,
                    error: "empty generation"
                )
            ],
            outputDir: dir
        )
        XCTAssertEqual(failCheck.status, .fail)
        XCTAssertTrue(failCheck.hint?.contains("bad") ?? false)
    }

    func test_reviewedPruneNativeSmokeFailsWhenSuiteIsMissing() async throws {
        let dir = try sizeSanityDir("expert-smoke-missing-suite")
        defer { try? FileManager.default.removeItem(at: dir) }

        let check = await PostConvertVerifier.runExpertLabNativeSmoke(
            outputDir: dir,
            suiteURL: nil,
            requiresSuite: true
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.hint?.contains("requires the reviewed prompt suite") == true)
        let summaryURL = dir.appendingPathComponent("expert_lab_smoke_summary.json")
        let summaryData = try Data(contentsOf: summaryURL)
        let summary = try XCTUnwrap(JSONSerialization.jsonObject(with: summaryData) as? [String: Any])
        XCTAssertEqual(summary["source"] as? String, "missing-reviewed-suite")
        XCTAssertEqual(summary["prompt_count"] as? Int, 0)
    }

    func test_runFailsSkippedNativeSmokeCheckForReviewedPrune() async throws {
        let output = fixture("good_output")
        let pruned = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-pruned-source-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: pruned) }
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try """
        {"ok":true,"checks":{"config_parses":true,"index_parses":true,"index_covers_tensors":true,"router_rows_match":true,"expert_rows_match":true}}
        """
            .write(to: pruned.appendingPathComponent("verification.json"), atomically: true, encoding: .utf8)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)

        let plan = ConversionPlan()
        plan.sourceURL = pruned
        plan.outputURL = output
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        plan.expertReviewPrunedSourceURL = pruned
        plan.expertReviewPrunePlanURL = pruned.appendingPathComponent("prune_plan.json")

        let checks = await PostConvertVerifier().run(
            plan: plan,
            skipPythonValidate: true,
            skipNativeSmoke: true
        )

        let nativeSmoke = try XCTUnwrap(checks.first { $0.id == .expertLabNativeSmoke })
        XCTAssertEqual(nativeSmoke.status, .fail)
        XCTAssertTrue(nativeSmoke.required)
        XCTAssertTrue(nativeSmoke.hint?.contains("requires the reviewed prompt suite") == true)
    }

    func test_finalComparisonSummarizesReviewRunAndSmokeArtifact() throws {
        let root = try sizeSanityDir("expert-final-comparison")
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appendingPathComponent("run-a", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try """
        {
          "format":"jangtq",
          "format_version":"2.0",
          "profile":"JANGTQ4",
          "architecture":{"type":"qwen3_5_moe"},
          "capabilities":{"family":"qwen3_5_moe"},
          "quantization":{"bit_widths_used":[4],"block_size":64}
        }
        """.write(to: output.appendingPathComponent("jang_config.json"), atomically: true, encoding: .utf8)
        try #"{"runID":"run-a"}"#.write(to: runDir.appendingPathComponent("run.json"), atomically: true, encoding: .utf8)
        try #"{"prompt_id":"p1"}\#n{"prompt_id":"p2"}\#n"#.write(
            to: runDir.appendingPathComponent("generations.jsonl"),
            atomically: true,
            encoding: .utf8
        )
        try "{}".write(to: runDir.appendingPathComponent("atlas.json"), atomically: true, encoding: .utf8)
        try Data().write(to: runDir.appendingPathComponent("trace.sqlite"))
        let evalDir = runDir.appendingPathComponent("evals/2026-06-23T00-00-00Z", isDirectory: true)
        try FileManager.default.createDirectory(at: evalDir, withIntermediateDirectories: true)
        try #"{"promptCount":1,"meanTextDelta":0.1}"#.write(
            to: evalDir.appendingPathComponent("comparison_summary.json"),
            atomically: true,
            encoding: .utf8
        )
        let laterEvalDir = runDir.appendingPathComponent("evals/9999-12-31T23-59-59Z", isDirectory: true)
        try FileManager.default.createDirectory(at: laterEvalDir, withIntermediateDirectories: true)
        try #"{"promptCount":1,"meanTextDelta":0.9}"#.write(
            to: laterEvalDir.appendingPathComponent("comparison_summary.json"),
            atomically: true,
            encoding: .utf8
        )
        let planURL = runDir.appendingPathComponent("prune_plan.json")
        try """
        {
          "method":"prompt_trace_hits_mass_domain_lift_v1",
          "promptCount":2,
          "run_id":"run-a",
          "atlas_id":"atlas-a",
          "eval_artifact":"\(evalDir.path)",
          "layers":{
            "0":{
              "keep":[0,1],
              "drop":[2,3],
              "locked_keep":[1],
              "user_forced_drop":[3],
              "evidence":[
                {"expert":0,"label":"math-core","frequency":0.4,"router_mass":0.6,"reason":"kept by review","kept":true},
                {"expert":2,"label":"idle-format","frequency":0.01,"router_mass":0.02,"reason":"low activation across suite","kept":false},
                {"expert":3,"label":"manual-drop","frequency":0.03,"router_mass":0.04,"reason":"duplicate behavior","user_forced_drop":true,"kept":false}
              ]
            }
          }
        }
        """.write(to: planURL, atomically: true, encoding: .utf8)
        let prunedGenerations = runDir.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        let prunedSummary = runDir.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        try """
        {"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p1"},"result":{"text":"4","tokens":1,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\(runDir.path)"},"layer_stats":[{"layer":0,"token_count":1,"hit_counts":{"0":1},"probability_mass":{"0":1.0}}],"token_trace":[{"layer":0,"token_index":0,"selected_experts":[0],"scores":[1.0]}]}}
        {"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p2"},"result":{"text":"ok","tokens":1,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\(runDir.path)"},"layer_stats":[{"layer":0,"token_count":1,"hit_counts":{"0":1},"probability_mass":{"0":1.0}}],"token_trace":[{"layer":0,"token_index":0,"selected_experts":[0],"scores":[1.0]}]}}
        """
            .write(to: prunedGenerations, atomically: true, encoding: .utf8)
        try """
        {"id":"p1","domain":"math","text":"2+2","expected_kind":"exact","expected":"4","tags":["arithmetic"]}
        {"id":"p2","domain":"general","text":"Say ok","tags":["english_dominant"]}
        """
            .write(to: runDir.appendingPathComponent("expert_lab_suite.jsonl"), atomically: true, encoding: .utf8)
        let suiteSHA256 = try Self.fileSHA256(runDir.appendingPathComponent("expert_lab_suite.jsonl"))
        let evalJSONL = runDir.appendingPathComponent("expert_lab_eval.jsonl")
        try """
        {"promptID":"p1","domain":"math","semanticDomains":["math"],"expectedKind":"exact","expected":"4","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"4","maskedText":"4","textDelta":0.0}
        {"promptID":"p2","domain":"general","semanticDomains":["english_dominant"],"expectedKind":"exact","expected":"ok","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"ok","maskedText":"ok","textDelta":0.0}
        """
            .write(to: evalJSONL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-bf16-suite-v1",
          "ready":true,
          "pruned_source":"\(runDir.path)",
          "suite_sha256":"\(suiteSHA256)",
          "prompt_count":2,
          "generation_count":2,
          "runtime_mode":"bf16_vmlx",
          "runtime_backend":"vmlx",
          "runtime_device":"Unit Metal",
          "runtime_metal_enabled":true,
          "jang_tools_version":"2.5.31",
          "mlx_version":"0.31.2",
          "mlx_lm_version":"0.31.3",
          "runtime_source_model_path":"\(runDir.path)",
          "reviewed_masked_comparison_count":2,
          "reviewed_masked_mean_text_delta":0.0,
          "reviewed_masked_max_text_delta":0.0,
          "generations_jsonl":"\(prunedGenerations.path)"
        }
        """.write(to: prunedSummary, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "pruned_suite_verification_ready":true,
          "pruned_source":"\(runDir.path)",
          "pruned_suite_summary":"\(prunedSummary.path)",
          "pruned_suite_generations":"\(prunedGenerations.path)",
          "review_eval_directory":"\(evalDir.path)",
          "suite_jsonl":"\(runDir.appendingPathComponent("expert_lab_suite.jsonl").path)",
          "eval_jsonl":"\(evalJSONL.path)",
          "comparison_summary":"\(evalDir.appendingPathComponent("comparison_summary.json").path)"
        }
        """.write(to: runDir.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "2+2", ok: true, text: "4", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil),
                ExpertLabSmokeRecord(promptID: "p2", prompt: "Say ok", ok: true, text: "ok", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: runDir.appendingPathComponent("expert_lab_suite.jsonl")
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewOriginalSourceURL = root.appendingPathComponent("original")
        plan.expertReviewPrunedSourceURL = runDir

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(
            check.hint?.contains("baseline-qualified coverage is missing") == true,
            check.hint ?? "missing final comparison hint"
        )
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let review = try XCTUnwrap(json["review"] as? [String: Any])
        XCTAssertEqual(review["generation_count"] as? Int, 2)
        XCTAssertEqual(review["eval_count"] as? Int, 2)
        XCTAssertEqual(review["review_eval_directory"] as? String, evalDir.path)
        XCTAssertEqual(review["pruned_suite_verification_ready"] as? Bool, true)
        XCTAssertEqual(review["pruned_suite_summary"] as? String, prunedSummary.path)
        XCTAssertEqual(review["pruned_suite_generations"] as? String, prunedGenerations.path)
        let latestEval = try XCTUnwrap(review["latest_eval"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(latestEval["meanTextDelta"] as? Double), 0.1, accuracy: 0.0001)
        XCTAssertEqual(review["drop_count"] as? Int, 2)
        XCTAssertEqual(review["locked_keep_count"] as? Int, 1)
        XCTAssertEqual(review["user_forced_drop_count"] as? Int, 1)
        XCTAssertEqual(review["evidence_count"] as? Int, 3)
        let evidencePreview = try XCTUnwrap(review["evidence_preview"] as? [String])
        XCTAssertTrue(evidencePreview.contains { $0.contains("L0 E2: idle-format") })
        XCTAssertTrue(evidencePreview.contains { $0.contains("user-forced") })
        let smoke = try XCTUnwrap(json["native_smoke"] as? [String: Any])
        XCTAssertEqual(smoke["passed_count"] as? Int, 2)
        XCTAssertEqual(smoke["source"] as? String, "reviewed-suite")
        XCTAssertEqual(smoke["suite_jsonl"] as? String, runDir.appendingPathComponent("expert_lab_suite.jsonl").path)
        let postQuantRuntime = try XCTUnwrap(smoke["runtime_info"] as? [String: Any])
        XCTAssertEqual(postQuantRuntime["runtime_mode"] as? String, "post_quant_jangtq")
        XCTAssertEqual(postQuantRuntime["format"] as? String, "jangtq")
        XCTAssertEqual(postQuantRuntime["model_path"] as? String, output.path)
        XCTAssertEqual(postQuantRuntime["output_path"] as? String, output.path)
        let postQuantVsPruned = try XCTUnwrap(json["post_quant_vs_pruned_bf16"] as? [String: Any])
        XCTAssertEqual(postQuantVsPruned["compared_count"] as? Int, 2)
        XCTAssertEqual(try XCTUnwrap(postQuantVsPruned["max_text_delta"] as? Double), 0, accuracy: 0.0001)
        XCTAssertEqual(postQuantVsPruned["max_delta_prompt_id"] as? String, "p1")
        let perPrompt = try XCTUnwrap(postQuantVsPruned["per_prompt"] as? [[String: Any]])
        XCTAssertEqual(perPrompt.count, 2)
        XCTAssertEqual(perPrompt.first?["prompt_id"] as? String, "p1")
        XCTAssertEqual(perPrompt.first?["pruned_bf16_text"] as? String, "4")
        XCTAssertEqual(perPrompt.first?["post_quant_text"] as? String, "4")
        let semanticCoverage = try XCTUnwrap(json["post_quant_reviewed_suite_semantic_coverage"] as? [String: Any])
        XCTAssertEqual(semanticCoverage["ready"] as? Bool, false)
        XCTAssertEqual(semanticCoverage["prompt_count"] as? Int, 2)
        let missingSemantic = try XCTUnwrap(semanticCoverage["missing"] as? [String])
        XCTAssertTrue(missingSemantic.contains("chinese"))
        XCTAssertTrue(missingSemantic.contains("safety_medical_legal_sensitive"))
        let report = try String(contentsOf: output.appendingPathComponent("expert_lab_final_report.md"), encoding: .utf8)
        XCTAssertTrue(report.contains("Post-quant vs pruned BF16 reference: 2 prompts"))
        XCTAssertEqual(smoke["prompt_ids"] as? [String], ["p1", "p2"])
        XCTAssertEqual(json["post_quant_reviewed_suite"] as? String, runDir.appendingPathComponent("expert_lab_suite.jsonl").path)
        XCTAssertTrue(report.contains("Final Comparison"))
        XCTAssertTrue(report.contains("Reviewed eval artifact: \(evalDir.path)"))
        XCTAssertTrue(report.contains("Pruned BF16/F16 same-suite generation: ready"))
        XCTAssertTrue(report.contains("Pruned BF16/F16 generation summary: \(prunedSummary.path)"))
        XCTAssertTrue(report.contains("Pruned BF16/F16 generations: \(prunedGenerations.path)"))
        XCTAssertTrue(report.contains("Pruned-vs-reviewed masked comparison: 2 prompts; mean delta 0.0000; max delta 0.0000"))
        XCTAssertTrue(report.contains("Pruned vMLX runtime source: \(runDir.path)"))
        XCTAssertTrue(report.contains("Latest mask comparison: 1 prompts; mean text delta 0.1000"))
        XCTAssertTrue(report.contains("Locked keeps: 1"))
        XCTAssertTrue(report.contains("User-forced drops: 1"))
        XCTAssertTrue(report.contains("Prune evidence rows: 3"))
        XCTAssertTrue(report.contains("Post-quant reviewed-suite semantic coverage: missing"))
        XCTAssertTrue(report.contains("Post-quant prompt source: reviewed-suite"))
        XCTAssertTrue(report.contains("Post-quant prompt suite: \(runDir.appendingPathComponent("expert_lab_suite.jsonl").path)"))
        XCTAssertTrue(report.contains("Post-quant prompt IDs: p1, p2"))
        XCTAssertTrue(report.contains("Post-quant runtime: post_quant_jangtq; format jangtq"))
        XCTAssertTrue(report.contains("L0 E2: idle-format"))
        XCTAssertTrue(report.contains("Native smoke prompts: 2 / 2 passed"))
    }

    func test_finalComparisonFailsWhenPostQuantRowsMissingConvertedRuntimeEvidence() throws {
        let root = try sizeSanityDir("expert-final-comparison-postquant-runtime")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try #"{"id":"p1","text":"Return the stable reference word."}"#
            .appending("\n")
            .write(to: suite, atomically: true, encoding: .utf8)
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        try """
        {"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p1"},"result":{"text":"stable reference word","tokens":3,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\(pruned.path)"}}}
        """
            .write(to: prunedGenerations, atomically: true, encoding: .utf8)
        try """
        {"promptID":"p1","domain":"general","semanticDomains":[\(Self.requiredSemanticCoverageJSON())],"expectedKind":"exact","expected":"stable reference word","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"stable reference word","maskedText":"stable reference word","textDelta":0.0}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)",
          "pruned_suite_verification_ready":true,
          "pruned_source":"\(pruned.path)",
          "pruned_suite_generations":"\(prunedGenerations.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        try """
        {"prompt_id":"p1","prompt":"Return the stable reference word.","ok":true,"text":"stable reference word","tokens":3,"tokens_per_sec":20,"elapsed_s":0.1,"error":null}
        """
            .write(to: output.appendingPathComponent("expert_lab_smoke.jsonl"), atomically: true, encoding: .utf8)
        try """
        {
          "schema": "jang-expert-lab-post-quant-smoke-v1",
          "source": "reviewed-suite",
          "suite_jsonl": "\(suite.path)",
          "prompt_count": 1,
          "passed_count": 1,
          "failed": [],
          "prompt_ids": ["p1"],
          "artifact": "\(output.appendingPathComponent("expert_lab_smoke.jsonl").path)"
        }
        """.write(to: output.appendingPathComponent("expert_lab_smoke_summary.json"), atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.hint?.contains("missing converted runtime evidence") == true)
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        XCTAssertTrue((json["post_quant_same_suite_issue"] as? String)?.contains("converted runtime evidence") == true)
    }

    func test_finalComparisonFailsWhenPrunedBF16ReferenceRowsMissingRuntimeEvidence() throws {
        let root = try sizeSanityDir("expert-final-comparison-reference-runtime")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try #"{"id":"p1","text":"Return the stable reference word."}"#
            .appending("\n")
            .write(to: suite, atomically: true, encoding: .utf8)
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        try """
        {"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p1"},"result":{"text":"stable reference word","tokens":3}}
        """
            .write(to: prunedGenerations, atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)",
          "pruned_suite_verification_ready":true,
          "pruned_source":"\(pruned.path)",
          "pruned_suite_generations":"\(prunedGenerations.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "Return the stable reference word.", ok: true, text: "stable reference word", tokens: 3, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(
            check.hint?.contains("pruned-source generation is missing per-prompt runtime evidence") == true,
            check.hint ?? "missing final comparison hint"
        )
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        let postQuantVsPruned = try XCTUnwrap(json["post_quant_vs_pruned_bf16"] as? [String: Any])
        XCTAssertTrue(
            (postQuantVsPruned["issue"] as? String)?.contains("per-prompt runtime evidence") == true,
            (postQuantVsPruned["issue"] as? String) ?? "missing pruned reference issue"
        )
    }

    func test_finalComparisonRejectsReviewSummaryPrunedSourceMismatch() throws {
        let root = try sizeSanityDir("expert-final-comparison-reference-source")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let otherPruned = root.appendingPathComponent("other-pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: otherPruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try #"{"id":"p1","text":"Return the stable reference word."}"#
            .appending("\n")
            .write(to: suite, atomically: true, encoding: .utf8)
        let otherGenerations = otherPruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        try """
        {"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p1"},"result":{"text":"stable reference word","tokens":3,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\(otherPruned.path)"},"layer_stats":[{"layer":0,"token_count":1,"hit_counts":{"0":1},"probability_mass":{"0":1.0}}],"token_trace":[{"layer":0,"token_index":0,"selected_experts":[0],"scores":[1.0]}]}}
        """
            .write(to: otherGenerations, atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)",
          "pruned_suite_verification_ready":true,
          "pruned_source":"\(otherPruned.path)",
          "pruned_suite_generations":"\(otherGenerations.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "Return the stable reference word.", ok: true, text: "stable reference word", tokens: 3, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(
            check.hint?.contains("reference source path does not match the selected pruned BF16/F16 source") == true,
            check.hint ?? "missing final comparison hint"
        )
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        let postQuantVsPruned = try XCTUnwrap(json["post_quant_vs_pruned_bf16"] as? [String: Any])
        XCTAssertTrue(
            (postQuantVsPruned["issue"] as? String)?.contains("reference source path does not match the selected pruned BF16/F16 source") == true,
            (postQuantVsPruned["issue"] as? String) ?? "missing pruned reference issue"
        )
    }

    func test_finalComparisonFailsWhenPostQuantDivergesFromPrunedBF16Reference() throws {
        let root = try sizeSanityDir("expert-final-comparison-diverged-reference")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        try #"{"id":"p1","text":"Return the stable reference word."}"#
            .appending("\n")
            .write(to: suite, atomically: true, encoding: .utf8)
        let prunedGenerations = pruned.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        try """
        {"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p1"},"result":{"text":"stable reference word","tokens":3,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\(pruned.path)"},"layer_stats":[{"layer":0,"token_count":1,"hit_counts":{"0":1},"probability_mass":{"0":1.0}}],"token_trace":[{"layer":0,"token_index":0,"selected_experts":[0],"scores":[1.0]}]}}
        """
            .write(to: prunedGenerations, atomically: true, encoding: .utf8)
        try """
        {"promptID":"p1","domain":"general","semanticDomains":[\(Self.requiredSemanticCoverageJSON())],"expectedKind":"exact","expected":"stable reference word","validatorKind":"exact","validatorAvailable":true,"validatorSource":"suite_expected","baselinePassed":true,"maskedPassed":true,"baselineQualified":true,"promptClassification":"preserved","safeDropEvidenceEligible":true,"baselineText":"stable reference word","maskedText":"stable reference word","textDelta":0.0}
        """
            .write(to: pruned.appendingPathComponent("expert_lab_eval.jsonl"), atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)",
          "pruned_suite_verification_ready":true,
          "pruned_source":"\(pruned.path)",
          "pruned_suite_generations":"\(prunedGenerations.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "Return the stable reference word.", ok: true, text: "zzzzzzzzzzzz", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(
            check.hint?.contains("post-quant output failed baseline-qualified validators") == true,
            check.hint ?? "missing final comparison hint"
        )
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        let postQuantVsPruned = try XCTUnwrap(json["post_quant_vs_pruned_bf16"] as? [String: Any])
        XCTAssertEqual(postQuantVsPruned["compared_count"] as? Int, 1)
        XCTAssertTrue(
            (postQuantVsPruned["issue"] as? String)?.contains("failed baseline-qualified validators") == true,
            (postQuantVsPruned["issue"] as? String) ?? "missing pruned reference issue"
        )
    }

    func test_finalComparisonFailsWhenPostQuantSmokeDidNotUseReviewedSuite() throws {
        let root = try sizeSanityDir("expert-final-comparison-fallback")
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appendingPathComponent("run-a", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let planURL = runDir.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(runDir.appendingPathComponent("expert_lab_suite.jsonl").path)"
        }
        """.write(to: runDir.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "smoke-a", prompt: "2+2", ok: true, text: "4", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "built-in-smoke",
            suiteURL: nil
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = runDir

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("did not use the reviewed Expert Lab prompt suite") == true)
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        XCTAssertTrue((json["post_quant_same_suite_issue"] as? String)?.contains("reviewed Expert Lab prompt suite") == true)
        let smoke = try XCTUnwrap(json["native_smoke"] as? [String: Any])
        XCTAssertEqual(smoke["source"] as? String, "built-in-smoke")
        let report = try String(contentsOf: output.appendingPathComponent("expert_lab_final_report.md"), encoding: .utf8)
        XCTAssertTrue(report.contains("Post-quant same-suite evidence: missing or failed"))
    }

    func test_finalComparisonFailsWhenPostQuantSuiteCoverageIsPartial() throws {
        let root = try sizeSanityDir("expert-final-comparison-partial-suite")
        defer { try? FileManager.default.removeItem(at: root) }
        let runDir = root.appendingPathComponent("run-a", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let suite = runDir.appendingPathComponent("expert_lab_suite.jsonl")
        try """
        {"id":"p1","text":"2+2"}
        {"id":"p2","text":"Say hi"}

        """.write(to: suite, atomically: true, encoding: .utf8)
        let planURL = runDir.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(suite.path)"
        }
        """.write(to: runDir.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "2+2", ok: true, text: "4", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = runDir

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("missing prompt IDs") == true)
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
    }

    func test_finalComparisonFailsWhenPostQuantSuitePathDiffersFromReviewedSuite() throws {
        let root = try sizeSanityDir("expert-final-comparison-wrong-suite")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let reviewedSuite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let wrongSuite = root.appendingPathComponent("different_suite.jsonl")
        try #"{"id":"p1","text":"2+2"}"#
            .appending("\n")
            .write(to: reviewedSuite, atomically: true, encoding: .utf8)
        try #"{"id":"p1","text":"2+2"}"#
            .appending("\n")
            .write(to: wrongSuite, atomically: true, encoding: .utf8)
        let planURL = pruned.appendingPathComponent("prune_plan.json")
        try Self.validReviewedPrunePlanJSON()
            .write(to: planURL, atomically: true, encoding: .utf8)
        try """
        {
          "schema":"jang-expert-lab-pruned-source-review-v1",
          "same_suite_verification_ready":true,
          "suite_jsonl":"\(reviewedSuite.path)"
        }
        """.write(to: pruned.appendingPathComponent("expert_lab_review_summary.json"), atomically: true, encoding: .utf8)
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: [
                ExpertLabSmokeRecord(promptID: "p1", prompt: "2+2", ok: true, text: "4", tokens: 1, tokensPerSec: 20, elapsedS: 0.1, error: nil)
            ],
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: wrongSuite
        )

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: planURL
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("instead of reviewed suite") == true)
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        XCTAssertEqual(json["post_quant_reviewed_suite"] as? String, reviewedSuite.path)
        let smoke = try XCTUnwrap(json["native_smoke"] as? [String: Any])
        XCTAssertEqual(smoke["suite_jsonl"] as? String, wrongSuite.path)
    }

    func test_finalComparisonFailsWhenReviewedSuiteFingerprintChangesAfterPostQuantSmoke() throws {
        let root = try sizeSanityDir("expert-final-comparison-suite-fingerprint")
        defer { try? FileManager.default.removeItem(at: root) }
        let pruned = root.appendingPathComponent("pruned", isDirectory: true)
        let output = root.appendingPathComponent("converted", isDirectory: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Self.validReviewedPrunePlanJSON()
            .write(to: pruned.appendingPathComponent("prune_plan.json"), atomically: true, encoding: .utf8)
        try writeExpertLabReviewSidecars(in: pruned)
        let suite = pruned.appendingPathComponent("expert_lab_suite.jsonl")
        let records = (0..<50).map {
            ExpertLabSmokeRecord(
                promptID: "p\($0)",
                prompt: "Reviewed prompt \($0)",
                ok: true,
                text: "hello from pruned bf16",
                tokens: 12,
                tokensPerSec: 20,
                elapsedS: 0.1,
                error: nil
            )
        }
        _ = PostConvertVerifier.writeExpertLabSmokeArtifact(
            records: records,
            outputDir: output,
            source: "reviewed-suite",
            suiteURL: suite
        )
        let originalSuite = try String(contentsOf: suite, encoding: .utf8)
        try originalSuite
            .replacingOccurrences(
                of: "Return only the number: 17 * 23.",
                with: "Return only the number: 19 * 29."
            )
            .write(to: suite, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.outputURL = output
        plan.expertReviewPrunedSourceURL = pruned

        let check = PostConvertVerifier.writeExpertLabFinalComparison(
            plan: plan,
            outputDir: output,
            reviewedPlanURL: pruned.appendingPathComponent("prune_plan.json")
        )

        XCTAssertEqual(check.status, .fail)
        XCTAssertTrue(check.required)
        XCTAssertTrue(check.hint?.contains("fingerprint does not match reviewed suite") == true)
        let comparisonURL = output.appendingPathComponent("expert_lab_final_comparison.json")
        let data = try Data(contentsOf: comparisonURL)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["post_quant_same_suite_ready"] as? Bool, false)
        XCTAssertEqual(json["post_quant_same_suite_issue"] as? String, "post-quant reviewed prompt suite fingerprint does not match reviewed suite")
        XCTAssertEqual((json["post_quant_reviewed_suite_sha256"] as? String)?.count, 64)
        let smoke = try XCTUnwrap(json["native_smoke"] as? [String: Any])
        XCTAssertEqual((smoke["suite_sha256"] as? String)?.count, 64)
        XCTAssertNotEqual(smoke["suite_sha256"] as? String, json["post_quant_reviewed_suite_sha256"] as? String)
    }

    // MARK: - Iter 19: M42 — runJangValidate timeout + cancel-safe pattern

    func test_runJangValidate_defaultTimeoutIsReasonable() {
        // Validation is file-inspection only; a 60-second default is 10x
        // headroom over the ≤5s normal completion time. Pin this so a future
        // commit tightening it doesn't break long-running debug environments.
        XCTAssertGreaterThanOrEqual(PostConvertVerifier.defaultValidateTimeoutSeconds, 30,
                                    "validate timeout should leave headroom for slow dev machines")
        XCTAssertLessThanOrEqual(PostConvertVerifier.defaultValidateTimeoutSeconds, 300,
                                 "validate timeout shouldn't hide real hangs for too long")
    }

    func test_runJangValidate_returnsFalseOnNonexistentDir() async {
        // Bogus path — jang_tools.validate will exit non-zero, not hang.
        // The short-path is: process launches, exits quickly, returns false.
        // This exercises the terminationHandler branch of the continuation.
        let bogus = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID().uuidString)")
        let ok = await PostConvertVerifier.runJangValidate(outputDir: bogus, timeoutSeconds: 30)
        XCTAssertFalse(ok, "validate on a non-existent path must return false")
    }

    // MARK: - Iter 40: M116 disk-size sanity (feedback_model_checklist.md rule 2)

    private func sizeSanityDir(_ name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("disksize-\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func plantShard(in dir: URL, name: String, bytes: Int) throws {
        let url = dir.appendingPathComponent(name)
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(bytes))
    }

    private func writeExpertLabReviewSidecars(
        in pruned: URL,
        comparisonJSON: String? = nil,
        evalRows: Int = 50,
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
        let evalText = (0..<evalRows)
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

    private func writePrunedGenerationsWithLayerStats(
        in pruned: URL,
        layerCount: Int = 1,
        includeTokenTrace: Bool = false
    ) throws {
        let layerStats = (0..<layerCount)
            .map { #"{"layer":\#($0),"token_count":1,"hit_counts":{"0":1},"probability_mass":{"0":1.0}}"# }
            .joined(separator: ",")
        let tokenTrace = (0..<layerCount)
            .map { #"{"layer":\#($0),"token_index":0,"selected_experts":[0],"scores":[1.0]}"# }
            .joined(separator: ",")
        let tokenTraceField = includeTokenTrace ? #","token_trace":[\#(tokenTrace)]"# : ""
        let text = (0..<50)
            .map { #"{"schema":"jang-expert-lab-vmlx-generation-v1","prompt":{"id":"p\#($0)","text":"Say hello."},"result":{"text":"hello from pruned bf16","tokens":12,"runtime_info":{"runtime_mode":"bf16_vmlx","backend":"vmlx","device_name":"Unit Metal","runtime_metal_enabled":true,"jang_tools_version":"2.5.31","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","source_model_path":"\#(pruned.path)"},"layer_stats":[\#(layerStats)]\#(tokenTraceField)}}"# }
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

    private static func promptIDJSON(count: Int) -> String {
        (0..<count).map { #""p\#($0)""# }.joined(separator: ",")
    }

    private static func promptIDJSON(indices: [Int]) -> String {
        indices.map { #""p\#($0)""# }.joined(separator: ",")
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

    func test_diskSizeSanity_inRange_passes() throws {
        // 1 GB source @ 4 bits = 256 MB expected. Disk 250 MB → ratio 0.98×.
        let dir = try sizeSanityDir("inrange")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 250_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .pass, check.hint ?? "")
        XCTAssertEqual(check.id, .diskSizeSanity)
    }

    func test_diskSizeSanity_bloated_warns() throws {
        // 1 GB source @ 4 bits = 256 MB expected. Disk 1 GB (4×) → warn.
        // This is the M115 failure mode's safety net — orphan old shards
        // doubling disk size should trip the warn bucket.
        let dir = try sizeSanityDir("bloated")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 1_000_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .warn, check.hint ?? "")
        XCTAssertTrue(check.hint?.contains("disk=") ?? false)
        XCTAssertTrue(check.hint?.contains("expected") ?? false)
    }

    func test_diskSizeSanity_underrun_warns() throws {
        // 1 GB source @ 4 bits = 256 MB expected. Disk 50 MB (0.19×) → warn.
        // Incomplete convert detected.
        let dir = try sizeSanityDir("underrun")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 50_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .warn, check.hint ?? "")
    }

    func test_diskSizeSanity_excludes_imatrix() throws {
        // imatrix file should NOT count toward disk size — it's cache, not
        // weights. 250 MB real shard + 500 MB imatrix should still ratio
        // to 0.98× (expected 256 MB), not 2.9×.
        let dir = try sizeSanityDir("with-imatrix")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 250_000_000)
        try plantShard(in: dir, name: "jang_imatrix.safetensors", bytes: 500_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .pass, "imatrix must NOT count toward disk ratio: \(check.hint ?? "")")
    }

    func test_diskSizeSanity_missing_source_warns_with_hint() throws {
        // M175 (iter 102): pre-M175 this returned .pass with "couldn't
        // compute" hint — same visual state as a real pass, user couldn't
        // tell the audit hadn't run. Now .warn with an explicit "skipped"
        // marker. Updated from the pre-iter-102 test name to match.
        let dir = try sizeSanityDir("missing")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 100_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir, sourceBytes: 0, jangCfg: [:])
        XCTAssertEqual(check.status, .warn,
            "Missing inputs must surface as warn (not silent pass) so user sees the audit was skipped — M175")
        XCTAssertTrue(check.hint?.contains("couldn't compute") ?? false)
        XCTAssertTrue(check.hint?.contains("skipped") ?? false,
            "warn hint must mark the audit as skipped, not merely missing")
    }

    // MARK: - Iter 100 M174: diskSizeSanity must honor source dtype (FP8 / BF16)
    //
    // Same BF16-hardcoding bug iter-99 M173 fixed in
    // PreflightRunner.estimateOutputBytes. The sanity check uses the
    // same formula — 340 GB FP8 → JANG_4K produces ~178 GB output, but
    // pre-M174 `expected = source × bits / 16` gives 85 GB → ratio 2.09×
    // → false "bloat" warn on a correctly-sized output. User sees a
    // warning, worries, potentially re-runs convert for nothing.

    func test_diskSizeSanity_fp8_source_uses_8bit_divisor() throws {
        // 340 GB FP8 source × 4 bits = 170 GB expected. Disk 178 GB → ratio 1.05×.
        let dir = try sizeSanityDir("fp8-4bit")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 178_000_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 340_000_000_000,
            sourceDtype: .fp8,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .pass, check.hint ?? "")
    }

    func test_diskSizeSanity_bf16_source_preserves_pre_M174_behavior() throws {
        // Regression guard: BF16 source still uses /16 divisor. 1 GB × 4 bits = 256 MB.
        let dir = try sizeSanityDir("bf16-regression")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 250_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            sourceDtype: .bf16,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .pass, check.hint ?? "")
    }

    func test_diskSizeSanity_default_dtype_param_is_bf16() throws {
        // Backwards compat: callers that don't pass sourceDtype (pre-M174
        // signature) must get the BF16-assuming behavior. Avoids breaking
        // any existing test that passes {sourceBytes, jangCfg} only.
        let dir = try sizeSanityDir("default-dtype")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 250_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            jangCfg: ["quantization": ["actual_bits_per_weight": 4.0]])
        XCTAssertEqual(check.status, .pass, "default dtype must give same answer as bf16 explicit")
    }

    func test_diskSizeSanity_accepts_v1_bitsField_fallback() throws {
        // Some older jang_config.json used "actual_bits" (no _per_weight suffix).
        // Helper must accept both so v1 outputs don't get falsely warned.
        let dir = try sizeSanityDir("v1-bits")
        defer { try? FileManager.default.removeItem(at: dir) }
        try plantShard(in: dir, name: "model-00001-of-00001.safetensors", bytes: 250_000_000)
        let check = PostConvertVerifier.diskSizeSanityCheck(
            outputDir: dir,
            sourceBytes: 1_000_000_000,
            jangCfg: ["quantization": ["actual_bits": 4.0]])
        XCTAssertEqual(check.status, .pass)
    }

    // MARK: - Iter 81: M158 runJangValidate silent-failure regressions

    private func makeTempScript(_ body: String) throws -> URL {
        // Same pattern as PythonCLIInvokerTests (iter-77 M154). Isolated copy
        // because the two test suites are linked into the same bundle but
        // helpers aren't shared — keeping them local preserves the rule that
        // each test file is readable standalone.
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pcv-\(UUID().uuidString).sh")
        try "#!/bin/bash\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    func test_runJangValidate_does_not_hang_on_large_stderr_output() async throws {
        // Regression guard for M158 bug 1. Old code wired `Pipe()` to stdout
        // and stderr without ever reading them; macOS pipe buffer is 64 KB,
        // so any subprocess writing more than that before exit would block
        // on write(2) forever — and the validator would wait the full 60 s
        // timeout before returning false on a validate that actually passed.
        //
        // We push 200 KB to stderr, 200 KB to stdout, then exit 0. The fix
        // (FileHandle.nullDevice) lets the subprocess exit promptly; if the
        // bug regresses, the subprocess blocks on the first write past 64 KB
        // and we hit the 10 s timeout below → test fails with `ok == false`.
        let script = try makeTempScript(#"""
        dd if=/dev/urandom bs=1024 count=200 2>/dev/null | base64 >&2
        dd if=/dev/urandom bs=1024 count=200 2>/dev/null | base64
        exit 0
        """#)
        let start = Date()
        let ok = await PostConvertVerifier.runJangValidate(
            outputDir: URL(fileURLWithPath: "/tmp/irrelevant-\(UUID().uuidString)"),
            timeoutSeconds: 10,
            executableOverride: script
        )
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(ok, "subprocess exited 0 — pipe-fill regression if this reads false (hit timeout)")
        XCTAssertLessThan(elapsed, 5, "took \(elapsed)s — pipe-fill regression, should be <1s")
    }

    func test_runJangValidate_returns_true_on_immediate_zero_exit() async throws {
        // Regression guard for M158 bug 2. If terminationHandler is wired
        // AFTER proc.run(), a subprocess that exits in the microsecond window
        // between run() returning and the handler assignment will never fire
        // the handler → deadlock until the timeout → false. Fix ordering and
        // use a subprocess that exits as fast as possible so we maximize the
        // chance of triggering the race if it regresses.
        let script = try makeTempScript("exit 0")
        let start = Date()
        let ok = await PostConvertVerifier.runJangValidate(
            outputDir: URL(fileURLWithPath: "/tmp/irrelevant-\(UUID().uuidString)"),
            timeoutSeconds: 5,
            executableOverride: script
        )
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(ok, "fast-exit subprocess returned false — terminationHandler race regression")
        XCTAssertLessThan(elapsed, 3, "took \(elapsed)s — expected sub-second")
    }

    func test_runJangValidate_returns_false_on_nonzero_exit() async throws {
        // Symmetric coverage for the exit-0 test: make sure we correctly
        // report failure too, not just accidentally return true regardless.
        let script = try makeTempScript("exit 7")
        let ok = await PostConvertVerifier.runJangValidate(
            outputDir: URL(fileURLWithPath: "/tmp/irrelevant-\(UUID().uuidString)"),
            timeoutSeconds: 5,
            executableOverride: script
        )
        XCTAssertFalse(ok, "exit 7 must map to false")
    }

    func test_runJangValidate_timeoutFiresWithinTolerance() async {
        // Use an intentionally unreachable executable override so the child
        // subprocess just hangs waiting for stdin / never returns. The timeout
        // must kick in near the bound, not block indefinitely.
        // Strategy: shadow BundleResolver to return `/bin/cat` with no stdin —
        // that will block forever reading. We can't easily monkeypatch
        // BundleResolver from Swift tests, so instead we rely on the REAL
        // jang-tools validate on a REAL path that exits quickly (a1 happy path)
        // and just pin that the timeout parameter is respected.
        // TODO(M42-followup): proper hang test would need a test-only override.
        //
        // What we CAN test cheaply: a 0.1-second timeout against a real
        // subprocess start-up will never succeed — even a process that exits
        // in 200ms loses the race. So passing timeoutSeconds=0.1 should
        // return false due to timeout.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("jang-validate-timeout-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let start = Date()
        let ok = await PostConvertVerifier.runJangValidate(outputDir: tmpDir, timeoutSeconds: 0.1)
        let elapsed = Date().timeIntervalSince(start)
        // Either the subprocess exits <0.1s (unlikely on Python start-up) OR
        // the timeout fires. Either way `ok` is false (bogus dir → exit!=0, or
        // timeout → false). The key assertion is we DIDN'T wait the full
        // default 60s — elapsed must be well under that.
        XCTAssertFalse(ok)
        XCTAssertLessThan(elapsed, 10, "timeout must bound wall time near 0.1s, took \(elapsed)s")
        try? FileManager.default.removeItem(at: tmpDir)
    }
}
