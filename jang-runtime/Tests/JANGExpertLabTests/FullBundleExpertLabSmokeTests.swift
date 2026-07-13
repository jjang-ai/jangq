import Foundation
import XCTest
import Metal
import JANG
import JANGCoreMetal
import JANGExpertLab
import JANGKit

final class FullBundleExpertLabSmokeTests: XCTestCase {
    private struct BaselineLaneResult: Sendable {
        let index: Int
        let run: ExpertPromptRun
    }

    private struct MaskedLaneResult: Sendable {
        let index: Int
        let promptID: String
        let domain: String
        let expectedKind: String
        let expected: String?
        let baselineText: String
        let maskedText: String
        let textDelta: Double
        let baselineTokenCount: Int
        let maskedTokenCount: Int
        let baselineTokensPerSecond: Double
        let maskedTokensPerSecond: Double
        let latencyDeltaPct: Double
        let baselinePassed: Bool?
        let maskedPassed: Bool?
        let adapter: String
        let risk: String
        let highRiskDomain: String?

        var record: [String: Any] {
            var record: [String: Any] = [
                "promptID": promptID,
                "domain": domain,
                "expectedKind": expectedKind,
                "baselineText": baselineText,
                "maskedText": maskedText,
                "textDelta": textDelta,
                "baselineTokenCount": baselineTokenCount,
                "maskedTokenCount": maskedTokenCount,
                "baselineTokensPerSecond": baselineTokensPerSecond,
                "maskedTokensPerSecond": maskedTokensPerSecond,
                "latencyDeltaPct": latencyDeltaPct,
                "adapter": adapter,
                "risk": risk,
            ]
            record["expected"] = expected ?? NSNull()
            record["baselinePassed"] = baselinePassed ?? NSNull()
            record["maskedPassed"] = maskedPassed ?? NSNull()
            return record
        }
    }

    private static let progressLock = NSLock()

    func testReviewedPruneSmokeSuiteFixtureMeetsCoverageGate() {
        let suite = Self.reviewedPruneSmokeSuite(promptCount: 50, maxTokens: 8)
        XCTAssertEqual(suite.prompts.count, 50)
        XCTAssertGreaterThanOrEqual(Set(suite.prompts.map(\.domain)).count, 6)
        XCTAssertTrue(suite.prompts.allSatisfy { $0.maxNewTokens == 8 })
    }

    func testFullBundleExpertLabTraceMaskAndPersistSmoke() async throws {
        guard ProcessInfo.processInfo.environment["JANG_FULL_BUNDLE_SMOKE"] == "1" else {
            throw XCTSkip("Set JANG_FULL_BUNDLE_SMOKE=1 to run the full Qwen3.6 Expert Lab smoke.")
        }

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        let sourceModel = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B")
        guard FileManager.default.fileExists(atPath: reviewBundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Full JANGTQ4 review bundle is not present at \(reviewBundle.path)")
        }

        let prompt = ExpertPrompt(
            id: "full-smoke-hello",
            domain: "general",
            text: "Say hello in one short sentence.",
            maxNewTokens: 1,
            tags: ["full-bundle", "smoke"]
        )
        let suite = ExpertPromptSuite(name: "Full Bundle Smoke", prompts: [prompt])
        let generationConfig = JANGKit.SamplingConfig(maxTokens: 1)
        let traceConfig = JANGKit.ExpertTraceConfig(emitTokenTrace: true, maxTraceTokens: 4096)

        let model = try await JANGKit.Model.load(at: reviewBundle)
        let baseline = try await model.generateWithTrace(
            prompt: prompt.text,
            config: generationConfig,
            traceConfig: traceConfig
        )

        XCTAssertFalse(baseline.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        let trace = try XCTUnwrap(baseline.tokenTrace)
        XCTAssertEqual(Set(trace.map(\.layer)).count, 40)
        XCTAssertEqual(Set(trace.map(\.layer)), Set(0..<40))
        XCTAssertTrue(trace.allSatisfy { $0.selectedExperts.count == 8 })

        let baselineRun = ExpertPromptRun(prompt: prompt, result: baseline)
        let expectedExperts = Dictionary(uniqueKeysWithValues: (0..<40).map { ($0, 256) })
        let atlas = ExpertAtlasBuilder.build(from: [baselineRun], expectedExpertsByLayer: expectedExperts)
        XCTAssertEqual(atlas.experts.count, 40 * 256)

        let firstRoute = try XCTUnwrap(trace.first)
        let disabledExpert = try XCTUnwrap(firstRoute.selectedExperts.first)
        let mask = JANGKit.ExpertMask(layers: [firstRoute.layer: Set([disabledExpert])])
        let masked = try await model.generateWithTrace(
            prompt: prompt.text,
            config: generationConfig,
            traceConfig: JANGKit.ExpertTraceConfig(mask: mask, emitTokenTrace: false, maxTraceTokens: 0)
        )
        XCTAssertFalse(masked.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let runsRoot = appSupport
            .appendingPathComponent("JANGStudio", isDirectory: true)
            .appendingPathComponent("ExpertLab", isDirectory: true)
            .appendingPathComponent("runs", isDirectory: true)
        let runID = "full-bundle-smoke-" + ISO8601DateFormatter()
            .string(from: Date())
            .replacingOccurrences(of: ":", with: "-")

        let runDir = try ExpertArtifactWriter.writeRun(
            rootDirectory: runsRoot,
            runID: runID,
            sourcePath: sourceModel.path,
            reviewBundlePath: reviewBundle.path,
            runtimeMode: "native_jangtq_review_bundle",
            suite: suite,
            traceConfig: traceConfig,
            runs: [baselineRun],
            atlas: atlas
        )

        let evalDir = runDir
            .appendingPathComponent("evals", isDirectory: true)
            .appendingPathComponent("masked-smoke", isDirectory: true)
        try FileManager.default.createDirectory(at: evalDir, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(mask).write(to: evalDir.appendingPathComponent("mask.json"))
        let summary = ExpertComparisonSummary(
            baselineRunID: runID,
            maskID: "masked-smoke",
            promptCount: 1,
            meanTextDelta: ExpertPromptEvaluator.normalizedTextDelta(baseline.text, masked.text),
            meanLatencyDeltaPct: baseline.tokensPerSecond > 0
                ? ((masked.tokensPerSecond - baseline.tokensPerSecond) / baseline.tokensPerSecond) * 100
                : 0
        )
        try encoder.encode(summary).write(to: evalDir.appendingPathComponent("comparison_summary.json"))
        let evalData = try JSONSerialization.data(withJSONObject: [
            "prompt_id": prompt.id,
            "baseline_text": baseline.text,
            "masked_text": masked.text,
        ] as [String: Any])
        let evalLine = String(data: evalData, encoding: .utf8) ?? "{}"
        try (evalLine + "\n").write(to: evalDir.appendingPathComponent("eval.jsonl"), atomically: true, encoding: .utf8)

        XCTAssertTrue(FileManager.default.fileExists(atPath: runDir.appendingPathComponent("atlas.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: runDir.appendingPathComponent("trace.jsonl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: runDir.appendingPathComponent("generations.jsonl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: evalDir.appendingPathComponent("comparison_summary.json").path))

        print("Full Expert Lab smoke run: \(runDir.path)")
        print("Baseline: \(baseline.text)")
        print("Masked: \(masked.text)")
    }

    func testFullBundleReviewedPruneEvidenceSmoke() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["JANG_FULL_BUNDLE_REVIEWED_PRUNE_SMOKE"] == "1" else {
            throw XCTSkip(
                "Set JANG_FULL_BUNDLE_REVIEWED_PRUNE_SMOKE=1 to collect the full reviewed-prune evidence suite. " +
                "Optional: JANG_FULL_BUNDLE_REVIEW_PROMPTS, JANG_FULL_BUNDLE_REVIEW_TOKENS, JANG_FULL_BUNDLE_REVIEW_KEEP_EXPERTS."
            )
        }

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        let sourceModel = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B")
        guard FileManager.default.fileExists(atPath: reviewBundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Full JANGTQ4 review bundle is not present at \(reviewBundle.path)")
        }
        guard FileManager.default.fileExists(atPath: sourceModel.appendingPathComponent("model.safetensors.index.json").path) else {
            throw XCTSkip("Full BF16 source bundle is not present at \(sourceModel.path)")
        }

        let promptCount = max(50, env["JANG_FULL_BUNDLE_REVIEW_PROMPTS"].flatMap(Int.init) ?? 50)
        let maxTokens = max(8, env["JANG_FULL_BUNDLE_REVIEW_TOKENS"].flatMap(Int.init) ?? 8)
        let keepExperts = max(8, min(255, env["JANG_FULL_BUNDLE_REVIEW_KEEP_EXPERTS"].flatMap(Int.init) ?? 224))
        let reviewLanes = max(1, min(2, env["JANG_FULL_BUNDLE_REVIEW_LANES"].flatMap(Int.init) ?? 1))
        let suite = Self.reviewedPruneSmokeSuite(promptCount: promptCount, maxTokens: maxTokens)
        XCTAssertGreaterThanOrEqual(suite.prompts.count, 50)
        XCTAssertGreaterThanOrEqual(Set(suite.prompts.map(\.domain)).count, 6)

        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let auditRoot = appSupport
            .appendingPathComponent("JANGStudio", isDirectory: true)
            .appendingPathComponent("ExpertLab", isDirectory: true)
            .appendingPathComponent("audits", isDirectory: true)
        try FileManager.default.createDirectory(at: auditRoot, withIntermediateDirectories: true)
        let runID = "full-bundle-reviewed-prune-" + ISO8601DateFormatter()
            .string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let progressURL = auditRoot.appendingPathComponent("\(runID)-progress.jsonl")
        try Self.recordProgress(
            to: progressURL,
            event: "start",
            fields: [
                "runID": runID,
                "promptCount": suite.prompts.count,
                "maxTokens": maxTokens,
                "keepExpertsPerLayer": keepExperts,
                "reviewLanes": reviewLanes,
                "reviewBundlePath": reviewBundle.path,
                "sourceModelPath": sourceModel.path,
            ]
        )

        var models: [JANGKit.Model] = []
        models.reserveCapacity(reviewLanes)
        for lane in 0..<reviewLanes {
            try Self.recordProgress(to: progressURL, event: "model_load_start", fields: ["lane": lane])
            models.append(try await JANGKit.Model.load(at: reviewBundle))
            try Self.recordProgress(to: progressURL, event: "model_load_end", fields: ["lane": lane])
        }
        let generationConfig = JANGKit.SamplingConfig(maxTokens: maxTokens)
        let traceConfig = JANGKit.ExpertTraceConfig(emitTokenTrace: true, maxTraceTokens: 4096)

        let baselineResults = try await withThrowingTaskGroup(of: [BaselineLaneResult].self) { group in
            for lane in 0..<reviewLanes {
                let model = models[lane]
                let prompts = suite.prompts.enumerated().filter { index, _ in
                    index % reviewLanes == lane
                }
                group.addTask {
                    var laneResults: [BaselineLaneResult] = []
                    laneResults.reserveCapacity(prompts.count)
                    for (index, prompt) in prompts {
                        try Self.recordProgress(
                            to: progressURL,
                            event: "baseline_start",
                            fields: [
                                "lane": lane,
                                "index": index + 1,
                                "total": suite.prompts.count,
                                "promptID": prompt.id,
                            ]
                        )
                        let result = try await model.generateWithTrace(
                            prompt: prompt.text,
                            config: generationConfig,
                            traceConfig: traceConfig
                        )
                        XCTAssertFalse(
                            result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                            "Baseline output was empty for \(prompt.id)"
                        )
                        laneResults.append(BaselineLaneResult(index: index, run: ExpertPromptRun(prompt: prompt, result: result)))
                        print("Reviewed prune trace \(index + 1)/\(suite.prompts.count): \(prompt.id)")
                        try Self.recordProgress(
                            to: progressURL,
                            event: "baseline_end",
                            fields: [
                                "lane": lane,
                                "index": index + 1,
                                "total": suite.prompts.count,
                                "promptID": prompt.id,
                                "output": result.text,
                                "tokensPerSecond": result.tokensPerSecond,
                                "traceRecords": result.tokenTrace?.count ?? 0,
                            ]
                        )
                    }
                    return laneResults
                }
            }

            var allResults: [BaselineLaneResult] = []
            for try await laneResults in group {
                allResults.append(contentsOf: laneResults)
            }
            return allResults.sorted { $0.index < $1.index }
        }
        let runs = baselineResults.map(\.run)

        let expectedExperts = Dictionary(uniqueKeysWithValues: (0..<40).map { ($0, 256) })
        let atlas = ExpertAtlasBuilder.build(from: runs, expectedExpertsByLayer: expectedExperts)
        XCTAssertEqual(atlas.experts.count, 40 * 256)
        XCTAssertEqual(atlas.promptCount, suite.prompts.count)
        try Self.recordProgress(
            to: progressURL,
            event: "atlas_built",
            fields: ["promptCount": atlas.promptCount, "expertCount": atlas.experts.count]
        )

        let sourceExpertsPerLayer = expectedExperts.values.min() ?? 256
        let plannedDropsPerLayer = max(1, sourceExpertsPerLayer - keepExperts)
        let disabledCandidates = Self.maskCandidates(from: atlas, expertsPerLayer: plannedDropsPerLayer)
        let maskLayers = Dictionary(grouping: disabledCandidates, by: \.layer)
            .mapValues { Set($0.map(\.expert)) }
        let mask = JANGKit.ExpertMask(layers: maskLayers)
        try Self.recordProgress(
            to: progressURL,
            event: "mask_selected",
            fields: [
                "disabledExperts": disabledCandidates.count,
                "dropsPerLayer": plannedDropsPerLayer,
                "coveredLayers": maskLayers.count
            ]
        )
        var records: [[String: Any]] = []
        var deltas: [Double] = []
        var latencyDeltas: [Double] = []
        var highRiskDomains = Set<String>()

        let maskedResults = try await withThrowingTaskGroup(of: [MaskedLaneResult].self) { group in
            for lane in 0..<reviewLanes {
                let model = models[lane]
                let laneRuns = runs.enumerated().filter { index, _ in
                    index % reviewLanes == lane
                }
                group.addTask {
                    var laneResults: [MaskedLaneResult] = []
                    laneResults.reserveCapacity(laneRuns.count)
                    for (index, run) in laneRuns {
                        try Self.recordProgress(
                            to: progressURL,
                            event: "masked_start",
                            fields: [
                                "lane": lane,
                                "index": index + 1,
                                "total": runs.count,
                                "promptID": run.prompt.id,
                            ]
                        )
                        let masked = try await model.generateWithTrace(
                            prompt: run.prompt.text,
                            config: generationConfig,
                            traceConfig: JANGKit.ExpertTraceConfig(mask: mask, emitTokenTrace: false, maxTraceTokens: 0)
                        )
                        XCTAssertFalse(
                            masked.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                            "Masked output was empty for \(run.prompt.id)"
                        )
                        let evaluation = ExpertPromptEvaluator.evaluate(
                            prompt: run.prompt,
                            baselineText: run.result.text,
                            maskedText: masked.text
                        )
                        let textDelta = ExpertPromptEvaluator.normalizedTextDelta(run.result.text, masked.text)
                        let latencyDelta = run.result.tokensPerSecond > 0
                            ? ((masked.tokensPerSecond - run.result.tokensPerSecond) / run.result.tokensPerSecond) * 100
                            : 0
                        let highRiskDomain = ExpertPromptEvaluator.isHighRisk(evaluation: evaluation, textDelta: textDelta)
                            ? run.prompt.domain
                            : nil
                        print("Reviewed prune compare \(index + 1)/\(runs.count): \(run.prompt.id)")
                        try Self.recordProgress(
                            to: progressURL,
                            event: "masked_end",
                            fields: [
                                "lane": lane,
                                "index": index + 1,
                                "total": runs.count,
                                "promptID": run.prompt.id,
                                "baselineOutput": run.result.text,
                                "maskedOutput": masked.text,
                                "textDelta": textDelta,
                                "baselineTokenCount": run.result.tokens,
                                "maskedTokenCount": masked.tokens,
                                "latencyDeltaPct": latencyDelta,
                                "risk": evaluation.risk,
                            ]
                        )
                        laneResults.append(MaskedLaneResult(
                            index: index,
                            promptID: run.prompt.id,
                            domain: run.prompt.domain,
                            expectedKind: run.prompt.expectedKind.rawValue,
                            expected: run.prompt.expected,
                            baselineText: run.result.text,
                            maskedText: masked.text,
                            textDelta: textDelta,
                            baselineTokenCount: run.result.tokens,
                            maskedTokenCount: masked.tokens,
                            baselineTokensPerSecond: run.result.tokensPerSecond,
                            maskedTokensPerSecond: masked.tokensPerSecond,
                            latencyDeltaPct: latencyDelta,
                            baselinePassed: evaluation.baselinePassed,
                            maskedPassed: evaluation.maskedPassed,
                            adapter: evaluation.adapter,
                            risk: evaluation.risk,
                            highRiskDomain: highRiskDomain
                        ))
                    }
                    return laneResults
                }
            }

            var allResults: [MaskedLaneResult] = []
            for try await laneResults in group {
                allResults.append(contentsOf: laneResults)
            }
            return allResults.sorted { $0.index < $1.index }
        }
        for result in maskedResults {
            records.append(result.record)
            deltas.append(result.textDelta)
            latencyDeltas.append(result.latencyDeltaPct)
            if let domain = result.highRiskDomain {
                highRiskDomains.insert(domain)
            }
        }

        let runsRoot = appSupport
            .appendingPathComponent("JANGStudio", isDirectory: true)
            .appendingPathComponent("ExpertLab", isDirectory: true)
            .appendingPathComponent("runs", isDirectory: true)
        try Self.recordProgress(to: progressURL, event: "artifact_write_start")
        let runDir = try ExpertArtifactWriter.writeRun(
            rootDirectory: runsRoot,
            runID: runID,
            sourcePath: sourceModel.path,
            reviewBundlePath: reviewBundle.path,
            runtimeMode: "native_jangtq_review_bundle",
            suite: suite,
            traceConfig: traceConfig,
            runs: runs,
            atlas: atlas
        )
        try Self.recordProgress(
            to: progressURL,
            event: "artifact_write_end",
            fields: ["runDir": runDir.path]
        )

        let safeDropCandidates = highRiskDomains.isEmpty ? disabledCandidates : []
        let comparisonSummary = ExpertComparisonSummary(
            baselineRunID: runID,
            maskID: "reviewed-suite-mask",
            promptCount: records.count,
            passRateBaseline: Self.passRate(maskedResults.compactMap(\.baselinePassed)),
            passRateMasked: Self.passRate(maskedResults.compactMap(\.maskedPassed)),
            meanTextDelta: Self.mean(deltas),
            meanLatencyDeltaPct: Self.mean(latencyDeltas),
            highRiskDomains: Array(highRiskDomains).sorted(),
            safeDropCandidates: safeDropCandidates
        )
        let evalIndex = Self.evalIndexSummary(
            from: maskedResults,
            comparisonSummary: comparisonSummary
        )
        try Self.writeComparisonArtifacts(
            runDir: runDir,
            mask: mask,
            summary: comparisonSummary,
            evalIndex: evalIndex,
            records: records
        )

        let trainedTopKByLayer = Dictionary(uniqueKeysWithValues: (0..<40).map { ($0, 8) })
        let planURL = runDir.appendingPathComponent("prune_plan.json")
        var planErrorDescription = ""
        XCTAssertThrowsError(try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: keepExperts,
            sourceNumExpertsByLayer: expectedExperts,
            trainedTopKByLayer: trainedTopKByLayer,
            forceDropByLayer: maskLayers,
            comparisonSummary: comparisonSummary,
            evalIndex: evalIndex,
            sourceModelPath: sourceModel.path,
            reviewBundlePath: reviewBundle.path,
            runID: runID,
            atlasID: "atlas.json"
        )) { error in
            planErrorDescription = error.localizedDescription
            XCTAssertTrue(planErrorDescription.contains("Reviewed prune"))
        }
        try Self.recordProgress(
            to: progressURL,
            event: "prune_plan_blocked",
            fields: [
                "planURL": planURL.path,
                "reason": planErrorDescription,
                "highRiskDomains": Array(highRiskDomains).sorted()
            ]
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: runDir.appendingPathComponent("suite.jsonl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: runDir.appendingPathComponent("atlas.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: runDir.appendingPathComponent("evals/reviewed-suite-mask/eval.jsonl").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: planURL.path))

        print("Full reviewed-prune reference evidence run: \(runDir.path)")
        print("Reviewed prune plan blocked: \(planErrorDescription)")
        print("Disabled candidates: \(disabledCandidates.count) experts across \(maskLayers.count) layers")
        try Self.recordProgress(to: progressURL, event: "complete")
    }

    func testFullBundleLongDecodeQualitySmoke() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["JANG_FULL_BUNDLE_QUALITY_SMOKE"] == "1" else {
            throw XCTSkip(
                "Set JANG_FULL_BUNDLE_QUALITY_SMOKE=1 to run the long Qwen3.6 decode smoke. " +
                "Optionally set JANG_FULL_BUNDLE_QUALITY_TOKENS; default is 16."
            )
        }

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        guard FileManager.default.fileExists(atPath: reviewBundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Full JANGTQ4 review bundle is not present at \(reviewBundle.path)")
        }

        let requestedTokens = env["JANG_FULL_BUNDLE_QUALITY_TOKENS"]
            .flatMap(Int.init) ?? 16
        let maxTokens = max(4, requestedTokens)
        let prompt = "Write exactly one short English sentence with at least eight words."

        let model = try await JANGKit.Model.load(at: reviewBundle)
        let result = try await model.generate(
            prompt: prompt,
            config: JANGKit.SamplingConfig(maxTokens: maxTokens)
        )

        let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(trimmed.isEmpty)
        XCTAssertGreaterThanOrEqual(
            result.tokens,
            min(4, maxTokens),
            "Long-decode smoke produced too few tokens to inspect: \(String(reflecting: result.text))"
        )

        print("Full bundle long decode prompt: \(prompt)")
        print("Full bundle long decode max tokens: \(maxTokens)")
        print("Full bundle long decode output: \(result.text)")
        print(
            "Full bundle long decode timing: " +
            "\(result.tokens) tokens in \(String(format: "%.2f", result.elapsedSeconds))s " +
            "(\(String(format: "%.4f", result.tokensPerSecond)) tok/s, stop: \(result.finishReason.rawValue))"
        )
    }

    func testFullBundleFirstStepTopKDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["JANG_FULL_BUNDLE_TOPK_SMOKE"] == "1" else {
            throw XCTSkip("Set JANG_FULL_BUNDLE_TOPK_SMOKE=1 to inspect first-step full-bundle logits.")
        }
        let context = try MetalContext()

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        let prompt = "Write exactly one short English sentence with at least eight words."
        let tokenizer = try JANGTQTokenizer(modelDir: reviewBundle)
        let promptIds = tokenizer.applyChatTemplate(
            messages: [JANGTQChatMessage(role: "user", content: prompt)]
        )
        XCTAssertEqual(
            promptIds,
            [248045, 846, 198, 7734, 6681, 799, 2716, 6163, 11316, 440, 506, 3140, 7810, 4105, 13, 248046, 198, 248045, 74455, 198, 248068, 198]
        )

        let bundle = try JANGTQLoader(device: context.device).load(from: reviewBundle)
        let model = try JANGTQModel(bundle: bundle, context: context, maxSeqLen: 256, moePrefix: "mlp")
        var logits: MTLBuffer?
        for (position, tokenId) in promptIds.enumerated() {
            logits = try model.forward(tokenId: tokenId, position: position)
        }
        let top = try Self.topLogits(
            logits: XCTUnwrap(logits),
            count: 20,
            vocabSize: model.config.vocabSize
        )

        print("Full bundle first-step prompt ids: \(promptIds)")
        print("Full bundle first-step top20:")
        for item in top {
            print("\(item.id)\t\(String(format: "%.4f", item.value))\t\(String(reflecting: tokenizer.decodeToken(item.id)))")
        }
    }

    func testFullBundleTQExpertParityDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["JANG_FULL_BUNDLE_TQ_PARITY_SMOKE"] == "1" else {
            throw XCTSkip("Set JANG_FULL_BUNDLE_TQ_PARITY_SMOKE=1 to inspect one real routed-expert TQ path.")
        }
        let context = try MetalContext()

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        let bundle = try JANGTQLoader(device: context.device).load(from: reviewBundle)
        let kernels = try JANGTQKernels(context: context)
        let block = try JANGTQMoEBlock(
            layerIndex: 0,
            layerPrefix: "language_model.model.layers.0.mlp",
            bundle: bundle,
            kernels: kernels,
            topK: 1
        )

        let xBuf = try Self.makeDiagnosticHalfBuffer(device: context.device, count: block.inFeatures)
        guard let selected = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw JANGError.bufferAllocationFailed(MemoryLayout<UInt32>.stride)
        }
        selected.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 0

        let out = try block.runMLP(xHalfBuf: xBuf, selectedExpertsBuf: selected, K: 1)
        let ptr = out.contents().bindMemory(to: Float.self, capacity: block.inFeatures)
        let first = (0..<16).map { ptr[$0] }
        print("Full bundle TQ expert parity layer=0 expert=0 first16: \(first)")
    }

    func testFullBundleLinearAttentionParityDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["JANG_FULL_BUNDLE_LINEAR_ATTN_PARITY_SMOKE"] == "1" else {
            throw XCTSkip("Set JANG_FULL_BUNDLE_LINEAR_ATTN_PARITY_SMOKE=1 to inspect one real linear-attention block.")
        }
        let context = try MetalContext()

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        let bundle = try JANGTQLoader(device: context.device).load(from: reviewBundle)
        let ops = try JANGTQDecodeOps(context: context)
        let affine8 = try JANGTQAffine8Matmul(context: context)
        let block = try JANGTQLinearAttentionBlock(
            layerIndex: 0,
            config: bundle.config.model,
            bundle: bundle,
            layerPrefix: "language_model.model.layers.0.linear_attn",
            inputLayernormPath: "language_model.model.layers.0.input_layernorm.weight",
            affine8: affine8,
            ops: ops
        )
        let xBuf = try Self.makeDiagnosticHalfBuffer(device: context.device, count: block.hidden)
        let out = try block.forward(x: xBuf, position: 0)
        let ptr = out.contents().bindMemory(to: Float16.self, capacity: block.hidden)
        let first = (0..<16).map { Float(ptr[$0]) }
        print("Full bundle linear attention parity layer=0 first16: \(first)")
    }

    func testFullBundleMoEParityDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["JANG_FULL_BUNDLE_MOE_PARITY_SMOKE"] == "1" else {
            throw XCTSkip("Set JANG_FULL_BUNDLE_MOE_PARITY_SMOKE=1 to inspect one real MoE layer.")
        }
        let context = try MetalContext()

        let reviewBundle = URL(fileURLWithPath: "/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-JANGTQ4")
        let bundle = try JANGTQLoader(device: context.device).load(from: reviewBundle)
        let engine = JANGTQDecoderEngine(
            bundle: bundle,
            context: context,
            kernels: try JANGTQKernels(context: context),
            affine8: try JANGTQAffine8Matmul(context: context),
            ops: try JANGTQDecodeOps(context: context),
            cache: try JANGTQKVCache(device: context.device, nLayers: 40, kvHeads: bundle.config.model.kvHeads, headDim: bundle.config.model.headDim, maxSeqLen: 256),
            moePrefix: "mlp"
        )
        let xBuf = try Self.makeDiagnosticHalfBuffer(device: context.device, count: bundle.config.model.hiddenSize)
        let collector = JANGExpertTraceCollector()
        let out = try engine.runMoE(
            layer: 0,
            normedX: xBuf,
            hidden: bundle.config.model.hiddenSize,
            k: bundle.config.model.numExpertsPerTok ?? 8,
            traceConfig: JANGExpertTraceConfig(emitTokenTrace: true, maxTraceTokens: 8),
            traceCollector: collector,
            tokenIndex: 0
        )
        let ptr = out.contents().bindMemory(to: Float16.self, capacity: bundle.config.model.hiddenSize)
        let first = (0..<16).map { Float(ptr[$0]) }
        let record = collector.snapshot().first
        print("Full bundle MoE parity layer=0 selected: \(record?.selectedExperts ?? [])")
        print("Full bundle MoE parity layer=0 scores: \(record?.scores ?? [])")
        print("Full bundle MoE parity layer=0 first16: \(first)")
    }

    private static func makeDiagnosticHalfBuffer(device: MTLDevice, count: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw JANGError.bufferAllocationFailed(count * MemoryLayout<Float16>.stride)
        }
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            let value = Float((i % 17) - 8) / 8.0
            ptr[i] = Float16(value)
        }
        return buffer
    }

    private static func topLogits(
        logits: MTLBuffer,
        count: Int,
        vocabSize: Int
    ) throws -> [(id: Int, value: Float)] {
        let ptr = logits.contents().bindMemory(to: Float.self, capacity: vocabSize)
        var top: [(id: Int, value: Float)] = []
        for id in 0..<vocabSize {
            let value = ptr[id]
            if top.count < count {
                top.append((id, value))
                top.sort { $0.value > $1.value }
            } else if let last = top.last, value > last.value {
                top[top.count - 1] = (id, value)
                top.sort { $0.value > $1.value }
            }
        }
        return top
    }

    private static func reviewedPruneSmokeSuite(promptCount: Int, maxTokens: Int) -> ExpertPromptSuite {
        let seeds: [(domain: String, text: String)] = [
            ("coding", "Write one concise Swift tip for avoiding off-by-one errors."),
            ("math", "State one short arithmetic fact about prime numbers."),
            ("reasoning", "Give one brief reason to check assumptions before solving a problem."),
            ("tools", "Describe one safe step before running a destructive shell command."),
            ("multilingual", "Say hello politely in Spanish and English."),
            ("safety", "Give one short refusal-safe answer about unsafe instructions."),
            ("structured", "Return a tiny JSON object with a status field."),
            ("long-context", "Summarize why preserving earlier evidence matters in one sentence.")
        ]
        let variants = [
            "Answer in one short sentence.",
            "Keep the answer under twelve words.",
            "Use plain language.",
            "Avoid lists.",
            "Be specific.",
            "Do not mention this instruction."
        ]
        let prompts = (0..<promptCount).map { index in
            let seed = seeds[index % seeds.count]
            let variant = variants[(index / seeds.count) % variants.count]
            return ExpertPrompt(
                id: "full-reviewed-\(String(format: "%03d", index + 1))",
                domain: seed.domain,
                text: "\(seed.text)\n\n\(variant)",
                maxNewTokens: maxTokens,
                temperature: 0,
                tags: ["full-bundle", "reviewed-prune", seed.domain]
            )
        }
        return ExpertPromptSuite(name: "Full Bundle Reviewed Prune \(promptCount)", prompts: prompts)
    }

    private static func maskCandidates(from atlas: ExpertAtlas, expertsPerLayer: Int) -> [ExpertCoordinate] {
        guard expertsPerLayer > 0 else { return [] }
        let entriesByLayer = Dictionary(grouping: atlas.experts, by: \.layer)
        return entriesByLayer.keys.sorted().flatMap { layer -> [ExpertCoordinate] in
            let ranked = (entriesByLayer[layer] ?? []).sorted { lhs, rhs in
                if lhs.isDead != rhs.isDead { return lhs.isDead && !rhs.isDead }
                if lhs.hits != rhs.hits { return lhs.hits < rhs.hits }
                if lhs.probabilityMass != rhs.probabilityMass { return lhs.probabilityMass < rhs.probabilityMass }
                return lhs.expert < rhs.expert
            }
            return ranked.prefix(expertsPerLayer).map {
                ExpertCoordinate(layer: $0.layer, expert: $0.expert)
            }
        }
    }

    private static func writeComparisonArtifacts(
        runDir: URL,
        mask: JANGKit.ExpertMask,
        summary: ExpertComparisonSummary,
        evalIndex: ExpertEvalIndexSummary,
        records: [[String: Any]]
    ) throws {
        let evalDir = runDir
            .appendingPathComponent("evals", isDirectory: true)
            .appendingPathComponent("reviewed-suite-mask", isDirectory: true)
        try FileManager.default.createDirectory(at: evalDir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(mask).write(to: evalDir.appendingPathComponent("mask.json"))
        try encoder.encode(summary).write(to: evalDir.appendingPathComponent("comparison_summary.json"))
        try encoder.encode(evalIndex).write(to: evalDir.appendingPathComponent("eval_index.json"))
        let lines = try records.map { record -> String in
            let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
            return String(data: data, encoding: .utf8) ?? "{}"
        }
        try lines.joined(separator: "\n").appending("\n").write(
            to: evalDir.appendingPathComponent("eval.jsonl"),
            atomically: true,
            encoding: .utf8
        )
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func passRate(_ values: [Bool]) -> Double? {
        guard !values.isEmpty else { return nil }
        let passed = values.filter { $0 }.count
        return Double(passed) / Double(values.count)
    }

    private static func evalIndexSummary(
        from results: [MaskedLaneResult],
        comparisonSummary: ExpertComparisonSummary
    ) -> ExpertEvalIndexSummary {
        let baselineTokens = results.map(\.baselineTokenCount)
        let maskedTokens = results.map(\.maskedTokenCount)
        let riskyPromptIDs = results.compactMap { result -> String? in
            result.highRiskDomain == nil ? nil : result.promptID
        }
        return ExpertEvalIndexSummary(
            promptCount: results.count,
            promptIDs: results.map(\.promptID),
            riskyPromptIDs: riskyPromptIDs,
            highRiskDomains: comparisonSummary.highRiskDomains,
            passRateBaseline: comparisonSummary.passRateBaseline,
            passRateMasked: comparisonSummary.passRateMasked,
            meanTextDelta: comparisonSummary.meanTextDelta,
            minBaselineTokens: baselineTokens.min(),
            minMaskedTokens: maskedTokens.min(),
            meanBaselineTokens: meanInts(baselineTokens),
            meanMaskedTokens: meanInts(maskedTokens),
            runtimeMode: "native_jangtq_review_bundle",
            runtimeBackend: "jangtq",
            runtimeDevice: "Metal",
            runtimeMetalEnabled: true
        )
    }

    private static func meanInts(_ values: [Int]) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.reduce(0, +)) / Double(values.count)
    }

    private static func recordProgress(
        to url: URL,
        event: String,
        fields: [String: Any] = [:]
    ) throws {
        var record = fields
        record["event"] = event
        record["timestamp"] = ISO8601DateFormatter().string(from: Date())
        let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        var line = String(data: data, encoding: .utf8) ?? "{}"
        line.append("\n")
        let encoded = Data(line.utf8)
        progressLock.lock()
        defer { progressLock.unlock() }
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: encoded)
            try handle.close()
        } else {
            try encoded.write(to: url)
        }
        FileHandle.standardError.write(encoded)
    }
}
