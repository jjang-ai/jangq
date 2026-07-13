import XCTest
import CoreGraphics
import SQLite3
import JANGExpertLab
import JANGKit

final class ExpertAtlasTests: XCTestCase {
    func testSemanticTaxonomyCoversRequiredReviewLabels() {
        let prompts = [
            ExpertPrompt(id: "math", domain: "math", text: "solve"),
            ExpertPrompt(id: "code", domain: "coding", text: "write code", tags: ["swift"]),
            ExpertPrompt(id: "formatting", domain: "structured", text: "return JSON", tags: ["json"]),
            ExpertPrompt(id: "instruction", domain: "instruction", text: "follow hierarchy", tags: ["instruction-following"]),
            ExpertPrompt(id: "reasoning", domain: "reasoning", text: "reason"),
            ExpertPrompt(id: "sensitive", domain: "robustness", text: "safe answer", tags: ["safety", "medical", "legal"]),
            ExpertPrompt(id: "language", domain: "multilingual", text: "translate", subdomain: "chinese", tags: ["translation", "non_english"]),
            ExpertPrompt(id: "english", domain: "general", text: "classify English", tags: ["english_dominant"]),
            ExpertPrompt(id: "unknown", domain: "multilingual", text: "classify language role", subdomain: "unknown-language-role")
        ]
        let labels = Set(prompts.flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
        let required: Set<String> = [
            "math",
            "code",
            "formatting",
            "instruction_following",
            "reasoning",
            "safety_medical_legal_sensitive",
            "multilingual",
            "non_english",
            "chinese",
            "translation",
            "english_dominant",
            "unknown_language_role"
        ]

        XCTAssertEqual(ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains, required)
        XCTAssertTrue(required.isSubset(of: labels), "Missing labels: \(required.subtracting(labels))")
        XCTAssertEqual(ExpertDomainTaxonomy.displayName(for: "instruction_following"), "instruction following")
        XCTAssertEqual(
            ExpertDomainTaxonomy.displayName(for: "safety_medical_legal_sensitive"),
            "safety/medical/legal sensitive"
        )
    }

    func testPromptSuiteJSONLRoundTrip() throws {
        let suite = ExpertPromptSuite(
            name: "smoke",
            prompts: [
                ExpertPrompt(id: "code-1", domain: "code", text: "Write fizzbuzz."),
                ExpertPrompt(id: "math-1", domain: "math", text: "What is 2+2?", expected: "4"),
            ]
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("suite-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: url) }

        try suite.writeJSONL(to: url)
        let loaded = try ExpertPromptSuite.loadJSONL(name: "smoke", from: url)

        XCTAssertEqual(loaded, suite)
    }

    func testPromptSuiteLoadsPlanJSONLSchema() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("suite-plan-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: url) }
        try """
        {"id":"coding.python.001","domain":"coding","subdomain":"python","prompt":"Write a Python function.","expected_kind":"regex","expected":"def ","max_new_tokens":128,"temperature":0.0,"tags":["reasoning","syntax"],"weight":2.0}

        """.write(to: url, atomically: true, encoding: .utf8)

        let suite = try ExpertPromptSuite.loadJSONL(name: "custom", from: url)

        XCTAssertEqual(suite.prompts.first?.text, "Write a Python function.")
        XCTAssertEqual(suite.prompts.first?.expectedKind, .regex)
        XCTAssertEqual(suite.prompts.first?.subdomain, "python")
        XCTAssertEqual(suite.prompts.first?.maxNewTokens, 128)
        XCTAssertEqual(suite.prompts.first?.tags, ["reasoning", "syntax"])
        XCTAssertEqual(suite.prompts.first?.weight, 2.0)
    }

    func testPromptEvaluatorScoresExactRegexAndUnitTestAdapters() {
        let exact = ExpertPrompt(
            id: "exact",
            domain: "math",
            text: "2+2",
            expectedKind: .exact,
            expected: "4"
        )
        let exactOutcome = ExpertPromptEvaluator.evaluate(
            prompt: exact,
            baselineText: " 4\n",
            maskedText: "five"
        )
        XCTAssertEqual(exactOutcome.adapter, "normalized_exact")
        XCTAssertEqual(exactOutcome.baselinePassed, true)
        XCTAssertEqual(exactOutcome.maskedPassed, false)
        XCTAssertEqual(exactOutcome.risk, "regression")
        XCTAssertEqual(
            ExpertPromptEvaluator.regressionSeverity(evaluation: exactOutcome, textDelta: 0.10),
            ExpertPromptEvaluator.regressionSeverityCritical
        )

        let regex = ExpertPrompt(
            id: "regex",
            domain: "coding",
            text: "write code",
            expectedKind: .regex,
            expected: #"func\s+\w+"#
        )
        let regexOutcome = ExpertPromptEvaluator.evaluate(
            prompt: regex,
            baselineText: "func run() {}",
            maskedText: "run it"
        )
        XCTAssertEqual(regexOutcome.adapter, "regex")
        XCTAssertEqual(regexOutcome.baselinePassed, true)
        XCTAssertEqual(regexOutcome.maskedPassed, false)

        let unitTest = ExpertPrompt(
            id: "unit",
            domain: "coding",
            text: "emit passing marker",
            expectedKind: .unitTest,
            expected: #"PASS:\s+3/3"#
        )
        let unitOutcome = ExpertPromptEvaluator.evaluate(
            prompt: unitTest,
            baselineText: "PASS: 3/3",
            maskedText: "PASS: 2/3"
        )
        XCTAssertEqual(unitOutcome.adapter, "unit_test_expected_regex")
        XCTAssertEqual(unitOutcome.risk, "regression")
    }

    func testPromptEvaluatorLeavesFreeformUnscoredButTracksDeltaRisk() {
        let prompt = ExpertPrompt(id: "free", domain: "general", text: "explain")
        let outcome = ExpertPromptEvaluator.evaluate(
            prompt: prompt,
            baselineText: "short answer",
            maskedText: "different answer"
        )

        XCTAssertNil(outcome.baselinePassed)
        XCTAssertNil(outcome.maskedPassed)
        XCTAssertEqual(outcome.risk, "not_scored")
        XCTAssertEqual(
            ExpertPromptEvaluator.normalizedTextDelta("abc", "axc"),
            1.0 / 3.0,
            accuracy: 0.0001
        )
        XCTAssertFalse(ExpertPromptEvaluator.isHighRisk(evaluation: outcome, textDelta: 0.75))
        XCTAssertEqual(
            ExpertPromptEvaluator.regressionSeverity(evaluation: outcome, textDelta: 0.75),
            ExpertPromptEvaluator.regressionSeverityWatch
        )
        XCTAssertEqual(
            ExpertPromptEvaluator.regressionSeverity(evaluation: outcome, textDelta: 0.25),
            ExpertPromptEvaluator.regressionSeverityWatch
        )
    }

    func testAtlasBuilderAggregatesHitsAndLabelsSpecialists() {
        let code = ExpertPrompt(id: "code", domain: "code", text: "code")
        let math = ExpertPrompt(id: "math", domain: "math", text: "math")

        let run1 = ExpertPromptRun(
            prompt: code,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0, layer: 1,
                    selectedExperts: [2, 3], scores: [0.8, 0.2],
                    effectiveTopK: 2
                ),
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 1, layer: 1,
                    selectedExperts: [2, 4], scores: [0.7, 0.3],
                    effectiveTopK: 2
                ),
            ])
        )
        let run2 = ExpertPromptRun(
            prompt: math,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0, layer: 1,
                    selectedExperts: [5, 2], scores: [0.9, 0.1],
                    effectiveTopK: 2
                ),
            ])
        )

        let atlas = ExpertAtlasBuilder.build(from: [run1, run2])

        let expert2 = atlas.experts.first { $0.layer == 1 && $0.expert == 2 }
        XCTAssertEqual(expert2?.hits, 3)
        XCTAssertEqual(expert2?.domains["code"], 2)
        XCTAssertEqual(expert2?.domains["math"], 1)
        XCTAssertNil(expert2?.domains["coding"])
        XCTAssertEqual(expert2?.isHot, true)
        XCTAssertEqual(expert2?.meanTokenIndex ?? -1, 1.0 / 3.0, accuracy: 0.0001)
        XCTAssertEqual(expert2?.minTokenIndex, 0)
        XCTAssertEqual(expert2?.maxTokenIndex, 1)

        let expert5 = atlas.experts.first { $0.layer == 1 && $0.expert == 5 }
        XCTAssertEqual(expert5?.label, "math-specialist")
        XCTAssertGreaterThan(expert5?.domainLift["math"] ?? 0, 1.0)
    }

    func testAtlasEntryDecodesOlderJSONWithoutTokenDepthFields() throws {
        let json = """
        {
          "generatedAt": 0,
          "promptCount": 1,
          "experts": [
            {
              "layer": 0,
              "expert": 7,
              "hits": 1,
              "activationFrequency": 1.0,
              "probabilityMass": 0.9,
              "tokenCount": 1,
              "domains": {"code": 1},
              "domainLift": {"code": 1.5},
              "meanSelectedRank": 1.0,
              "entropyContribution": 0.0,
              "coactivationNeighbors": [],
              "topPrompts": ["p1"],
              "promptEvidence": [],
              "confidenceScore": 0.8,
              "label": "code-hot",
              "generatedLabel": "code-hot",
              "isDead": false,
              "isHot": true
            }
          ]
        }
        """

        let atlas = try JSONDecoder().decode(ExpertAtlas.self, from: Data(json.utf8))
        let entry = try XCTUnwrap(atlas.experts.first)
        XCTAssertNil(entry.meanTokenIndex)
        XCTAssertNil(entry.minTokenIndex)
        XCTAssertNil(entry.maxTokenIndex)
    }

    func testAtlasEntryDecodesLegacyLabelAsGeneratedLabel() throws {
        let json = """
        {
          "generatedAt": 0,
          "promptCount": 1,
          "experts": [
            {
              "layer": 0,
              "expert": 7,
              "hits": 1,
              "probabilityMass": 0.9,
              "tokenCount": 1,
              "domains": {"code": 1},
              "label": "code-hot",
              "isDead": false,
              "isHot": true
            }
          ]
        }
        """

        let atlas = try JSONDecoder().decode(ExpertAtlas.self, from: Data(json.utf8))
        let entry = try XCTUnwrap(atlas.experts.first)
        XCTAssertEqual(entry.generatedLabel, "code-hot")
        XCTAssertNil(entry.userLabel)
        XCTAssertNil(entry.userNotes)
        XCTAssertEqual(entry.domainLift, [:])
        XCTAssertEqual(entry.topPrompts, [])
    }

    func testAtlasBuilderUsesCanonicalDomainLiftForSemanticSignatures() throws {
        let code = ExpertPrompt(id: "code", domain: "code", text: "code")
        let safety = ExpertPrompt(id: "safety", domain: "robustness", text: "safety")
        let codingTrace = (0..<18).map { index in
            JANGKit.ExpertRouteRecord(
                tokenIndex: index,
                layer: 0,
                selectedExperts: index < 4 ? [7] : [0],
                scores: [1.0],
                effectiveTopK: 1
            )
        }
        let safetyTrace = (0..<2).map { index in
            JANGKit.ExpertRouteRecord(
                tokenIndex: index,
                layer: 0,
                selectedExperts: [7],
                scores: [1.0],
                effectiveTopK: 1
            )
        }
        let atlas = ExpertAtlasBuilder.build(from: [
            ExpertPromptRun(prompt: code, result: result(trace: codingTrace)),
            ExpertPromptRun(prompt: safety, result: result(trace: safetyTrace))
        ])

        let expert7 = try XCTUnwrap(atlas.experts.first { $0.layer == 0 && $0.expert == 7 })
        XCTAssertEqual(expert7.domains["code"], 4)
        XCTAssertEqual(expert7.domains["safety_medical_legal_sensitive"], 2)
        XCTAssertEqual(expert7.domains["safety_sensitive"], 2)
        XCTAssertNil(expert7.domains["coding"])
        XCTAssertNil(expert7.domains["robustness"])
        XCTAssertGreaterThan(expert7.domainLift["safety_sensitive"] ?? 0, expert7.domainLift["code"] ?? 0)
        XCTAssertEqual(
            ExpertDomainTaxonomy.dominantDomain(domains: expert7.domains, domainLift: expert7.domainLift),
            "safety_sensitive"
        )
        XCTAssertEqual(expert7.label, "safety_sensitive-specialist")
    }

    func testAtlasBuilderPreservesLanguageRoleSemanticFacets() throws {
        let prompt = ExpertPrompt(
            id: "zh",
            domain: "multilingual",
            text: "Translate this into Simplified Chinese.",
            subdomain: "chinese",
            tags: ["translation", "non_english"]
        )
        let trace = (0..<3).map { index in
            JANGKit.ExpertRouteRecord(
                tokenIndex: index,
                layer: 0,
                selectedExperts: [9],
                scores: [1.0],
                effectiveTopK: 1
            )
        }
        let atlas = ExpertAtlasBuilder.build(from: [
            ExpertPromptRun(prompt: prompt, result: result(trace: trace))
        ])

        let expert9 = try XCTUnwrap(atlas.experts.first { $0.layer == 0 && $0.expert == 9 })
        XCTAssertEqual(expert9.domains["multilingual"], 3)
        XCTAssertEqual(expert9.domains["non_english"], 3)
        XCTAssertEqual(expert9.domains["chinese"], 3)
        XCTAssertEqual(expert9.domains["translation"], 3)
        XCTAssertTrue(expert9.label.hasPrefix("chinese"))
        XCTAssertEqual(expert9.evidenceCount, 1)
        let evidence = try XCTUnwrap(expert9.promptEvidence?.first)
        XCTAssertEqual(evidence.promptID, "zh")
        XCTAssertEqual(evidence.domain, "multilingual")
        XCTAssertEqual(evidence.subdomain, "chinese")
        XCTAssertEqual(evidence.tags, ["translation", "non_english"])
        XCTAssertEqual(evidence.hits, 3)
        XCTAssertTrue(evidence.promptExcerpt.contains("Simplified Chinese"))
    }

    func testAtlasBuilderComputesCoactivationAndPromptAwareTokenCounts() {
        let promptA = ExpertPrompt(id: "a", domain: "code", text: "code")
        let promptB = ExpertPrompt(id: "b", domain: "code", text: "code again")
        let runs = [
            ExpertPromptRun(
                prompt: promptA,
                result: result(trace: [
                    JANGKit.ExpertRouteRecord(
                        tokenIndex: 0,
                        layer: 0,
                        selectedExperts: [1, 2],
                        scores: [0.6, 0.4],
                        effectiveTopK: 2,
                        entropy: 0.67
                    )
                ])
            ),
            ExpertPromptRun(
                prompt: promptB,
                result: result(trace: [
                    JANGKit.ExpertRouteRecord(
                        tokenIndex: 0,
                        layer: 0,
                        selectedExperts: [1, 2],
                        scores: [0.55, 0.45],
                        effectiveTopK: 2,
                        entropy: 0.69
                    )
                ])
            )
        ]

        let atlas = ExpertAtlasBuilder.build(from: runs)
        let expert1 = atlas.experts.first { $0.layer == 0 && $0.expert == 1 }

        XCTAssertEqual(expert1?.tokenCount, 2)
        XCTAssertEqual(expert1?.topPrompts, ["a", "b"])
        XCTAssertEqual(expert1?.coactivationNeighbors.first?.expert, 2)
        XCTAssertEqual(expert1?.coactivationNeighbors.first?.count, 2)
        XCTAssertGreaterThan(expert1?.entropyContribution ?? 0, 0)
    }

    func testAtlasBuilderFallsBackToLayerStatsWhenTokenTraceIsOmitted() {
        let prompt = ExpertPrompt(id: "compact", domain: "coding", text: "code")
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(
                trace: nil,
                layerStats: [
                    JANGKit.ExpertLayerStats(
                        layer: 2,
                        tokenCount: 4,
                        hitCounts: [1: 3, 7: 1],
                        probabilityMass: [1: 2.4, 7: 0.6]
                    )
                ]
            )
        )

        let atlas = ExpertAtlasBuilder.build(from: [run], expectedExpertsByLayer: [2: 8])

        let expert1 = atlas.experts.first { $0.layer == 2 && $0.expert == 1 }
        XCTAssertEqual(expert1?.hits, 3)
        XCTAssertEqual(expert1?.tokenCount, 4)
        XCTAssertEqual(expert1?.domains["code"], 3)
        XCTAssertEqual(expert1?.probabilityMass ?? 0, 2.4, accuracy: 1e-5)
        XCTAssertNil(expert1?.meanTokenIndex)
        XCTAssertNil(expert1?.minTokenIndex)
        XCTAssertNil(expert1?.maxTokenIndex)
        XCTAssertEqual(atlas.experts.filter(\.isDead).count, 6)
    }

    func testAtlasBuilderSanitizesNonFiniteLayerStatsBeforeEncoding() throws {
        let prompt = ExpertPrompt(id: "nan", domain: "coding", text: "code")
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(
                trace: nil,
                layerStats: [
                    JANGKit.ExpertLayerStats(
                        layer: 2,
                        tokenCount: 4,
                        hitCounts: [1: 3],
                        probabilityMass: [1: .nan]
                    )
                ]
            )
        )

        let atlas = ExpertAtlasBuilder.build(from: [run], expectedExpertsByLayer: [2: 2])
        let expert1 = try XCTUnwrap(atlas.experts.first { $0.layer == 2 && $0.expert == 1 })
        XCTAssertEqual(expert1.probabilityMass, 0)
        XCTAssertTrue(expert1.activationFrequency.isFinite)
        XCTAssertNoThrow(try JSONEncoder().encode(atlas))
    }

    func testAtlasBuilderMaterializesFullQwen36ExpertGrid() {
        let prompt = ExpertPrompt(id: "full-grid", domain: "general", text: "hello")
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0,
                    layer: 0,
                    selectedExperts: [7],
                    scores: [1.0],
                    effectiveTopK: 1
                )
            ])
        )
        let expected = Dictionary(uniqueKeysWithValues: (0..<40).map { ($0, 256) })

        let atlas = ExpertAtlasBuilder.build(from: [run], expectedExpertsByLayer: expected)

        XCTAssertEqual(atlas.experts.count, 40 * 256)
        XCTAssertEqual(atlas.sourceNumExpertsByLayer?.count, 40)
        XCTAssertEqual(atlas.sourceNumExpertsByLayer?["0"], 256)
        XCTAssertEqual(atlas.sourceNumExpertsByLayer?["39"], 256)
        XCTAssertEqual(Set(atlas.experts.map(\.layer)).count, 40)
        XCTAssertTrue((0..<40).allSatisfy { layer in
            atlas.experts.filter { $0.layer == layer }.count == 256
        })
        XCTAssertEqual(atlas.experts.first { $0.layer == 0 && $0.expert == 7 }?.hits, 1)
        XCTAssertEqual(atlas.experts.filter(\.isDead).count, (40 * 256) - 1)
    }

    func testMaskValidationBlocksTooFewAvailableExpertsAndWarnsHotDrops() {
        let mask = JANGKit.ExpertMask(layers: [0: [0, 1, 2]])
        let issues = ExpertMaskEngine.validate(
            mask: mask,
            sourceNumExpertsByLayer: [0: 4],
            trainedTopKByLayer: [0: 2],
            hotExperts: Set([ExpertCoordinate(layer: 0, expert: 0)])
        )

        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("fewer than top-k") })
        XCTAssertTrue(issues.contains { $0.severity == .warning && $0.message.contains("hot") })
    }

    func testArtifactWriterPersistsTraceSuiteAtlasAndGenerations() throws {
        let prompt = ExpertPrompt(id: "trace", domain: "code", text: "hello")
        let runtimeInfo = JANGKit.ModelRuntimeInfo(
            backend: "jangtq",
            runtimeMode: "native_jangtq_review_bundle",
            deviceName: "Unit Metal",
            metalEnabled: true,
            jangToolsVersion: "2.5.31",
            mlxVersion: "0.31.2",
            mlxLMVersion: "0.31.3",
            sourceModelPath: "/tmp/source",
            hookedMOELayers: 40,
            expectedMOELayers: 40,
            hookCoverageComplete: true
        )
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0,
                    layer: 0,
                    selectedExperts: [1],
                    scores: [1.0],
                    effectiveTopK: 1
                )
            ], layerStats: [
                JANGKit.ExpertLayerStats(
                    layer: 0,
                    tokenCount: 1,
                    hitCounts: [1: 1],
                    probabilityMass: [1: 1.0]
                )
            ], runtimeInfo: runtimeInfo)
        )
        let atlas = ExpertAtlasBuilder.build(from: [run], expectedExpertsByLayer: [0: 2])
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-artifacts-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let dir = try ExpertArtifactWriter.writeRun(
            rootDirectory: root,
            runID: "run_test",
            sourcePath: "/tmp/source",
            reviewBundlePath: "/tmp/review",
            runtimeMode: "unit",
            runtimeBackend: runtimeInfo.backend,
            runtimeDevice: runtimeInfo.deviceName,
            runtimeMetalEnabled: runtimeInfo.metalEnabled,
            suite: ExpertPromptSuite(name: "unit", prompts: [prompt]),
            traceConfig: JANGKit.ExpertTraceConfig(),
            runs: [run],
            atlas: atlas
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("run.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("suite.jsonl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("trace.jsonl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("trace.sqlite").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("generations.jsonl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("atlas.json").path))
        let atlasJSON = try String(contentsOf: dir.appendingPathComponent("atlas.json"), encoding: .utf8)
        XCTAssertTrue(atlasJSON.contains(#""sourceNumExpertsByLayer""#))
        XCTAssertTrue(atlasJSON.contains(#""0" : 2"#))
        let runJSON = try String(contentsOf: dir.appendingPathComponent("run.json"), encoding: .utf8)
        XCTAssertTrue(runJSON.contains(#""runtimeBackend" : "jangtq""#))
        XCTAssertTrue(runJSON.contains(#""runtimeDevice" : "Unit Metal""#))
        XCTAssertTrue(runJSON.contains(#""jangToolsVersion" : "2.5.31""#))
        XCTAssertTrue(runJSON.contains(#""mlxVersion" : "0.31.2""#))
        XCTAssertTrue(runJSON.contains(#""mlxLMVersion" : "0.31.3""#))
        let manifestDecoder = JSONDecoder()
        manifestDecoder.dateDecodingStrategy = .iso8601
        let manifest = try manifestDecoder.decode(
            ExpertRunManifest.self,
            from: Data(runJSON.utf8)
        )
        XCTAssertEqual(manifest.sourceModelPath, "/tmp/source")
        XCTAssertEqual(manifest.hookedMOELayers, 40)
        XCTAssertEqual(manifest.expectedMOELayers, 40)
        XCTAssertEqual(manifest.hookCoverageComplete, true)
        let summaries = try ExpertRunStore.listRuns(
            rootDirectory: root,
            matchingSourcePath: "/tmp/source",
            reviewBundlePath: "/tmp/review"
        )
        XCTAssertEqual(summaries.first?.sourceModelPath, "/tmp/source")
        XCTAssertEqual(summaries.first?.hookedMOELayers, 40)
        XCTAssertEqual(summaries.first?.expectedMOELayers, 40)
        XCTAssertEqual(summaries.first?.hookCoverageComplete, true)
        let generationText = try String(contentsOf: dir.appendingPathComponent("generations.jsonl"), encoding: .utf8)
        XCTAssertTrue(generationText.contains(#""runtimeDevice":"Unit Metal""#))
        XCTAssertTrue(generationText.contains(#""runtimeMetalEnabled":true"#))
        XCTAssertTrue(generationText.contains(#""jangToolsVersion":"2.5.31""#))
        XCTAssertTrue(generationText.contains(#""mlxVersion":"0.31.2""#))
        XCTAssertTrue(generationText.contains(#""mlxLMVersion":"0.31.3""#))
        let generationLine = try XCTUnwrap(generationText.split(whereSeparator: \.isNewline).first)
        let generationJSON = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(generationLine.utf8)
        ) as? [String: Any])
        XCTAssertEqual(generationJSON["sourceModelPath"] as? String, "/tmp/source")
        XCTAssertEqual(generationJSON["hookedMOELayers"] as? Int, 40)
        XCTAssertEqual(generationJSON["expectedMOELayers"] as? Int, 40)
        XCTAssertEqual(generationJSON["hookCoverageComplete"] as? Bool, true)
        let generationLayerStats = try XCTUnwrap(generationJSON["layerStats"] as? [[String: Any]])
        XCTAssertEqual(generationLayerStats.first?["layer"] as? Int, 0)
        XCTAssertEqual(generationLayerStats.first?["tokenCount"] as? Int, 1)
        let traceText = try String(contentsOf: dir.appendingPathComponent("trace.jsonl"), encoding: .utf8)
        XCTAssertTrue(traceText.contains(#""domain":"coding""#))
        XCTAssertEqual(try sqliteText(
            "SELECT domain FROM prompts WHERE prompt_id = 'trace';",
            in: dir.appendingPathComponent("trace.sqlite")
        ), "coding")
        XCTAssertEqual(try sqliteText(
            "SELECT domain FROM route_records LIMIT 1;",
            in: dir.appendingPathComponent("trace.sqlite")
        ), "coding")
        XCTAssertEqual(try sqliteInt(
            "SELECT COUNT(*) FROM route_records;",
            in: dir.appendingPathComponent("trace.sqlite")
        ), 1)
        XCTAssertEqual(try sqliteInt(
            "SELECT COUNT(*) FROM expert_events WHERE is_disabled = 0;",
            in: dir.appendingPathComponent("trace.sqlite")
        ), 1)
    }

    func testArtifactWriterPreservesManualAtlasAnnotations() throws {
        let prompt = ExpertPrompt(id: "trace", domain: "coding", text: "hello")
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0,
                    layer: 0,
                    selectedExperts: [1],
                    scores: [1.0],
                    effectiveTopK: 1
                )
            ])
        )
        let atlas = ExpertAtlas(
            promptCount: 1,
            experts: [
                ExpertAtlasEntry(
                    layer: 0,
                    expert: 1,
                    hits: 1,
                    probabilityMass: 1.0,
                    tokenCount: 1,
                    domains: ["code": 1],
                    label: "code-specialist",
                    generatedLabel: "code-specialist",
                    userLabel: "reviewed-coding-core",
                    userNotes: "Keep for Python trace stability.",
                    isDead: false,
                    isHot: true
                )
            ]
        )
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-manual-annotations-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let dir = try ExpertArtifactWriter.writeRun(
            rootDirectory: root,
            runID: "run_manual_annotations",
            sourcePath: "/tmp/source",
            reviewBundlePath: nil,
            runtimeMode: "bf16_vmlx",
            suite: ExpertPromptSuite(name: "unit", prompts: [prompt]),
            traceConfig: JANGKit.ExpertTraceConfig(),
            runs: [run],
            atlas: atlas
        )
        let atlasData = try Data(contentsOf: dir.appendingPathComponent("atlas.json"))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let saved = try decoder.decode(ExpertAtlas.self, from: atlasData)
        let entry = try XCTUnwrap(saved.experts.first)

        XCTAssertEqual(entry.generatedLabel, "code-specialist")
        XCTAssertEqual(entry.userLabel, "reviewed-coding-core")
        XCTAssertEqual(entry.userNotes, "Keep for Python trace stability.")
    }

    func testRunStoreListsMatchingRunsAndIgnoresOtherSources() throws {
        let prompt = ExpertPrompt(id: "trace", domain: "general", text: "hello")
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0,
                    layer: 0,
                    selectedExperts: [1],
                    scores: [1.0],
                    effectiveTopK: 1
                )
            ])
        )
        let atlas = ExpertAtlasBuilder.build(from: [run], expectedExpertsByLayer: [0: 2])
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-run-store-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        _ = try ExpertArtifactWriter.writeRun(
            rootDirectory: root,
            runID: "run_source_a",
            sourcePath: "/tmp/source-a",
            reviewBundlePath: "/tmp/review-a",
            runtimeMode: "unit",
            suite: ExpertPromptSuite(name: "unit-a", prompts: [prompt]),
            traceConfig: JANGKit.ExpertTraceConfig(),
            runs: [run],
            atlas: atlas,
            failureStage: "cancelled",
            failureMessage: "Trace cancelled after 1 of 3 prompts."
        )
        _ = try ExpertArtifactWriter.writeRun(
            rootDirectory: root,
            runID: "run_source_b",
            sourcePath: "/tmp/source-b",
            reviewBundlePath: "/tmp/review-b",
            runtimeMode: "unit",
            suite: ExpertPromptSuite(name: "unit-b", prompts: [prompt]),
            traceConfig: JANGKit.ExpertTraceConfig(),
            runs: [run],
            atlas: atlas
        )

        let matches = try ExpertRunStore.listRuns(
            rootDirectory: root,
            matchingSourcePath: "/tmp/source-a",
            reviewBundlePath: "/tmp/review-a"
        )

        XCTAssertEqual(matches.map(\.runID), ["run_source_a"])
        XCTAssertEqual(matches.first?.suiteID, "unit-a")
        XCTAssertEqual(matches.first?.promptCount, 1)
        XCTAssertEqual(matches.first?.failureStage, "cancelled")
        XCTAssertEqual(matches.first?.failureMessage, "Trace cancelled after 1 of 3 prompts.")

        let reviewBundleOnlyMatches = try ExpertRunStore.listRuns(
            rootDirectory: root,
            reviewBundlePath: "/tmp/review-a"
        )
        XCTAssertEqual(reviewBundleOnlyMatches.map(\.runID), ["run_source_a"])
    }

    func testRunStoreMatchesCanonicalAndSymlinkedModelPaths() throws {
        let prompt = ExpertPrompt(id: "trace", domain: "general", text: "hello")
        let run = ExpertPromptRun(
            prompt: prompt,
            result: result(trace: [
                JANGKit.ExpertRouteRecord(
                    tokenIndex: 0,
                    layer: 0,
                    selectedExperts: [1],
                    scores: [1.0],
                    effectiveTopK: 1
                )
            ])
        )
        let atlas = ExpertAtlasBuilder.build(from: [run], expectedExpertsByLayer: [0: 2])
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-run-store-symlink-\(UUID().uuidString)", isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let review = root.appendingPathComponent("review", isDirectory: true)
        let sourceLink = root.appendingPathComponent("source-link", isDirectory: true)
        let reviewLink = root.appendingPathComponent("review-link", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: review, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: sourceLink, withDestinationURL: source)
        try FileManager.default.createSymbolicLink(at: reviewLink, withDestinationURL: review)

        _ = try ExpertArtifactWriter.writeRun(
            rootDirectory: root,
            runID: "run_full_bundle",
            sourcePath: source.path,
            reviewBundlePath: review.path,
            runtimeMode: "native_jangtq_review_bundle",
            suite: ExpertPromptSuite(name: "Full Bundle Reviewed Prune 50", prompts: [prompt]),
            traceConfig: JANGKit.ExpertTraceConfig(),
            runs: [run],
            atlas: atlas
        )

        let matches = try ExpertRunStore.listRuns(
            rootDirectory: root,
            matchingSourcePath: sourceLink.path,
            reviewBundlePath: reviewLink.path
        )

        XCTAssertEqual(matches.map(\.runID), ["run_full_bundle"])
        XCTAssertEqual(matches.first?.suiteID, "Full Bundle Reviewed Prune 50")
    }

    func testMaskArtifactRoundTripsReviewState() throws {
        let artifact = ExpertLabMaskArtifact(
            disabledByLayer: [0: [1, 2]],
            dropCandidatesByLayer: [1: [3]],
            lockedKeepByLayer: [2: [4]],
            topKOverride: 3
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(artifact)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(ExpertLabMaskArtifact.self, from: data)

        XCTAssertEqual(decoded.disabledByLayer[0], [1, 2])
        XCTAssertEqual(decoded.dropCandidatesByLayer[1], [3])
        XCTAssertEqual(decoded.lockedKeepByLayer[2], [4])
        XCTAssertEqual(decoded.topKOverride, 3)
    }

    func testAtlasSelectionLassoIntersectsCellFrames() {
        let frames: [ExpertCoordinate: CGRect] = [
            ExpertCoordinate(layer: 0, expert: 0): CGRect(x: 0, y: 0, width: 10, height: 10),
            ExpertCoordinate(layer: 0, expert: 1): CGRect(x: 16, y: 0, width: 10, height: 10),
            ExpertCoordinate(layer: 1, expert: 0): CGRect(x: 0, y: 16, width: 10, height: 10),
        ]

        let selected = ExpertAtlasSelection.coordinates(
            intersecting: CGRect(x: 8, y: -2, width: 20, height: 14),
            cellFrames: frames
        )

        XCTAssertEqual(selected, [
            ExpertCoordinate(layer: 0, expert: 0),
            ExpertCoordinate(layer: 0, expert: 1),
        ])
        XCTAssertTrue(ExpertAtlasSelection.coordinates(
            intersecting: CGRect(x: 0, y: 0, width: 0, height: 10),
            cellFrames: frames
        ).isEmpty)
    }

    private func result(
        trace: [JANGKit.ExpertRouteRecord]?,
        layerStats: [JANGKit.ExpertLayerStats] = [],
        runtimeInfo: JANGKit.ModelRuntimeInfo? = nil
    ) -> JANGKit.ExpertRunResult {
        JANGKit.ExpertRunResult(
            text: "",
            tokens: 0,
            elapsedSeconds: 0,
            tokensPerSecond: 0,
            finishReason: .maxTokens,
            layerStats: layerStats,
            tokenTrace: trace,
            runtimeInfo: runtimeInfo
        )
    }

    private func sqliteInt(_ sql: String, in url: URL) throws -> Int {
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil), SQLITE_OK)
        defer { sqlite3_close(db) }
        var statement: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(db, sql, -1, &statement, nil), SQLITE_OK)
        defer { sqlite3_finalize(statement) }
        XCTAssertEqual(sqlite3_step(statement), SQLITE_ROW)
        return Int(sqlite3_column_int64(statement, 0))
    }

    private func sqliteText(_ sql: String, in url: URL) throws -> String {
        var db: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil), SQLITE_OK)
        defer { sqlite3_close(db) }
        var statement: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(db, sql, -1, &statement, nil), SQLITE_OK)
        defer { sqlite3_finalize(statement) }
        XCTAssertEqual(sqlite3_step(statement), SQLITE_ROW)
        guard let text = sqlite3_column_text(statement, 0) else { return "" }
        return String(cString: text)
    }
}
