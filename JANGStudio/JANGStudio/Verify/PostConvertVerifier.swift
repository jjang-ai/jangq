// JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift
import CryptoKit
import Foundation
import JANGExpertLab

struct ExpertLabConvertedRuntimeInfo: Codable, Equatable, Sendable {
    let runtimeMode: String?
    let backend: String?
    let modelPath: String?
    let outputPath: String?
    let configFound: Bool?
    let format: String?
    let formatVersion: String?
    let profile: String?
    let architecture: String?
    let capabilityFamily: String?
    let capabilityArch: String?
    let quantizationBits: [Int]?
    let quantizationBlockSize: Int?

    enum CodingKeys: String, CodingKey {
        case runtimeMode = "runtime_mode"
        case backend
        case modelPath = "model_path"
        case outputPath = "output_path"
        case configFound = "config_found"
        case format
        case formatVersion = "format_version"
        case profile
        case architecture
        case capabilityFamily = "capability_family"
        case capabilityArch = "capability_arch"
        case quantizationBits = "quantization_bits"
        case quantizationBlockSize = "quantization_block_size"
    }

    func payload() -> [String: Any] {
        [
            "runtime_mode": runtimeMode ?? NSNull(),
            "backend": backend ?? NSNull(),
            "model_path": modelPath ?? NSNull(),
            "output_path": outputPath ?? NSNull(),
            "config_found": configFound ?? NSNull(),
            "format": format ?? NSNull(),
            "format_version": formatVersion ?? NSNull(),
            "profile": profile ?? NSNull(),
            "architecture": architecture ?? NSNull(),
            "capability_family": capabilityFamily ?? NSNull(),
            "capability_arch": capabilityArch ?? NSNull(),
            "quantization_bits": quantizationBits ?? [],
            "quantization_block_size": quantizationBlockSize ?? NSNull()
        ]
    }

    func reportLine() -> String {
        var parts: [String] = []
        if let runtimeMode, !runtimeMode.isEmpty {
            parts.append(runtimeMode)
        }
        if let format, !format.isEmpty {
            parts.append("format \(format)")
        }
        if let profile, !profile.isEmpty {
            parts.append("profile \(profile)")
        }
        if let architecture, !architecture.isEmpty {
            parts.append("arch \(architecture)")
        } else if let capabilityFamily, !capabilityFamily.isEmpty {
            parts.append("family \(capabilityFamily)")
        } else if let capabilityArch, !capabilityArch.isEmpty {
            parts.append("arch \(capabilityArch)")
        }
        if let quantizationBits, !quantizationBits.isEmpty {
            parts.append("bits \(quantizationBits.map(String.init).joined(separator: "/"))")
        }
        if let modelPath, !modelPath.isEmpty {
            parts.append("model \(modelPath)")
        }
        return parts.isEmpty ? "not recorded" : parts.joined(separator: "; ")
    }
}

struct ExpertLabSmokeRecord: Codable, Equatable, Sendable {
    let promptID: String
    let prompt: String
    let generationSettings: ExpertLabSmokeGenerationSettings?
    let ok: Bool
    let text: String?
    let tokens: Int
    let tokensPerSec: Double
    let elapsedS: Double
    let error: String?
    let runtimeInfo: ExpertLabConvertedRuntimeInfo?

    init(
        promptID: String,
        prompt: String,
        generationSettings: ExpertLabSmokeGenerationSettings? = nil,
        ok: Bool,
        text: String?,
        tokens: Int,
        tokensPerSec: Double,
        elapsedS: Double,
        error: String?,
        runtimeInfo: ExpertLabConvertedRuntimeInfo? = nil
    ) {
        self.promptID = promptID
        self.prompt = prompt
        self.generationSettings = generationSettings
        self.ok = ok
        self.text = text
        self.tokens = tokens
        self.tokensPerSec = tokensPerSec
        self.elapsedS = elapsedS
        self.error = error
        self.runtimeInfo = runtimeInfo
    }

    enum CodingKeys: String, CodingKey {
        case promptID = "prompt_id"
        case prompt
        case generationSettings = "generation_settings"
        case ok
        case text
        case tokens
        case tokensPerSec = "tokens_per_sec"
        case elapsedS = "elapsed_s"
        case error
        case runtimeInfo = "runtime_info"
    }

    func withRuntimeInfo(_ runtimeInfo: ExpertLabConvertedRuntimeInfo?) -> ExpertLabSmokeRecord {
        ExpertLabSmokeRecord(
            promptID: promptID,
            prompt: prompt,
            generationSettings: generationSettings,
            ok: ok,
            text: text,
            tokens: tokens,
            tokensPerSec: tokensPerSec,
            elapsedS: elapsedS,
            error: error,
            runtimeInfo: runtimeInfo
        )
    }
}

struct ExpertLabSmokeGenerationSettings: Codable, Equatable, Sendable {
    let maxTokens: Int
    let temperature: Double

    enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case temperature
    }
}

private struct PostQuantPromptComparisonRow {
    let promptID: String
    let prunedBF16Text: String
    let postQuantText: String
    let prunedBF16Tokens: Int
    let postQuantTokens: Int
    let textDelta: Double
    let validatorKind: String?
    let baselineQualified: Bool
    let postQuantPassed: Bool?
    let promptClassification: String
    let safeDropEvidenceEligible: Bool
    let semanticDomains: [String]

    func payload() -> [String: Any] {
        [
            "prompt_id": promptID,
            "pruned_bf16_text": prunedBF16Text,
            "post_quant_text": postQuantText,
            "pruned_bf16_tokens": prunedBF16Tokens,
            "post_quant_tokens": postQuantTokens,
            "text_delta": textDelta,
            "validator_kind": validatorKind ?? NSNull(),
            "baseline_qualified": baselineQualified,
            "post_quant_passed": postQuantPassed ?? NSNull(),
            "prompt_classification": promptClassification,
            "safe_drop_evidence_eligible": safeDropEvidenceEligible,
            "semantic_domains": semanticDomains
        ]
    }
}

private struct ExpertLabSmokePrompt: Equatable, Sendable {
    let id: String
    let text: String
    let maxTokens: Int?
    let temperature: Double?
}

private struct ExpertLabSmokeSummary {
    let source: String
    let suiteURL: URL?
    let suiteSHA256: String?
    let promptCount: Int?
    let promptIDs: [String]
    let error: String?
}

private struct PostQuantPrunedBehaviorComparison {
    let referenceURL: URL?
    let comparedCount: Int
    let meanTextDelta: Double?
    let maxTextDelta: Double?
    let baselineQualifiedPromptCount: Int
    let postQuantBaselineQualifiedPassRate: Double?
    let classificationCounts: [String: Int]
    let baselineInvalidPromptIDs: [String]
    let inconclusivePromptIDs: [String]
    let preservedPromptIDs: [String]
    let degradedPromptIDs: [String]
    let baselineQualifiedSemanticCoverage: [String]
    let missingBaselineQualifiedSemanticCoverage: [String]
    let issue: String?
    let rows: [PostQuantPromptComparisonRow]

    init(
        referenceURL: URL?,
        comparedCount: Int,
        meanTextDelta: Double?,
        maxTextDelta: Double?,
        baselineQualifiedPromptCount: Int = 0,
        postQuantBaselineQualifiedPassRate: Double? = nil,
        classificationCounts: [String: Int] = [
            "baseline_invalid": 0,
            "preserved": 0,
            "degraded": 0,
            "inconclusive": 0
        ],
        baselineInvalidPromptIDs: [String] = [],
        inconclusivePromptIDs: [String] = [],
        preservedPromptIDs: [String] = [],
        degradedPromptIDs: [String] = [],
        baselineQualifiedSemanticCoverage: [String] = [],
        missingBaselineQualifiedSemanticCoverage: [String] = [],
        issue: String?,
        rows: [PostQuantPromptComparisonRow] = []
    ) {
        self.referenceURL = referenceURL
        self.comparedCount = comparedCount
        self.meanTextDelta = meanTextDelta
        self.maxTextDelta = maxTextDelta
        self.baselineQualifiedPromptCount = baselineQualifiedPromptCount
        self.postQuantBaselineQualifiedPassRate = postQuantBaselineQualifiedPassRate
        self.classificationCounts = classificationCounts
        self.baselineInvalidPromptIDs = baselineInvalidPromptIDs
        self.inconclusivePromptIDs = inconclusivePromptIDs
        self.preservedPromptIDs = preservedPromptIDs
        self.degradedPromptIDs = degradedPromptIDs
        self.baselineQualifiedSemanticCoverage = baselineQualifiedSemanticCoverage
        self.missingBaselineQualifiedSemanticCoverage = missingBaselineQualifiedSemanticCoverage
        self.issue = issue
        self.rows = rows
    }

    func payload(threshold: Double) -> [String: Any] {
        [
            "reference_generations": referenceURL?.path ?? NSNull(),
            "compared_count": comparedCount,
            "mean_text_delta": meanTextDelta ?? NSNull(),
            "max_text_delta": maxTextDelta ?? NSNull(),
            "max_delta_prompt_id": rows.max(by: { $0.textDelta < $1.textDelta })?.promptID ?? NSNull(),
            "max_allowed_text_delta": threshold,
            "baseline_qualified_prompt_count": baselineQualifiedPromptCount,
            "post_quant_baseline_qualified_pass_rate": postQuantBaselineQualifiedPassRate ?? NSNull(),
            "post_quant_classification_counts": classificationCounts,
            "baseline_invalid_prompt_ids": baselineInvalidPromptIDs,
            "inconclusive_prompt_ids": inconclusivePromptIDs,
            "post_quant_preserved_prompt_ids": preservedPromptIDs,
            "post_quant_degraded_prompt_ids": degradedPromptIDs,
            "baseline_qualified_semantic_coverage": baselineQualifiedSemanticCoverage,
            "missing_baseline_qualified_semantic_coverage": missingBaselineQualifiedSemanticCoverage,
            "per_prompt": rows.map { $0.payload() },
            "issue": issue ?? NSNull()
        ]
    }

    func reportLine() -> String {
        var parts = ["\(comparedCount) prompts"]
        parts.append("\(baselineQualifiedPromptCount) baseline-qualified")
        if let postQuantBaselineQualifiedPassRate {
            parts.append(String(format: "qualified pass %.2f", postQuantBaselineQualifiedPassRate))
        }
        if !classificationCounts.isEmpty {
            let classes = ["preserved", "degraded", "baseline_invalid", "inconclusive"].compactMap { key -> String? in
                guard let value = classificationCounts[key], value > 0 else { return nil }
                return "\(key) \(value)"
            }.joined(separator: ", ")
            if !classes.isEmpty {
                parts.append("classes \(classes)")
            }
        }
        if let meanTextDelta {
            parts.append(String(format: "mean delta %.4f", meanTextDelta))
        }
        if let maxTextDelta {
            parts.append(String(format: "max delta %.4f", maxTextDelta))
        }
        if !missingBaselineQualifiedSemanticCoverage.isEmpty {
            parts.append("missing qualified coverage \(missingBaselineQualifiedSemanticCoverage.joined(separator: ", "))")
        }
        if !degradedPromptIDs.isEmpty {
            parts.append("degraded \(degradedPromptIDs.prefix(5).joined(separator: ", "))")
        }
        if let worst = rows.max(by: { $0.textDelta < $1.textDelta }) {
            parts.append(String(format: "worst %@", worst.promptID))
        }
        if let issue, !issue.isEmpty {
            parts.append("issue \(issue)")
        }
        return parts.joined(separator: "; ")
    }
}

struct PostConvertVerifier {
    @MainActor func run(
        plan: ConversionPlan,
        capabilities: Capabilities = .frozen,
        skipPythonValidate: Bool = false,
        skipNativeSmoke: Bool = false
    ) async -> [VerifyCheck] {
        guard let out = plan.outputURL else {
            return [.init(id: .jangConfigExists, title: "jang_config.json exists",
                          status: .fail, required: true, hint: "No output dir")]
        }
        var checks: [VerifyCheck] = []
        let jangCfgURL = out.appendingPathComponent("jang_config.json")

        // #1 jang_config exists + JSON valid
        let jangCfg = (try? JSONSerialization.jsonObject(with: Data(contentsOf: jangCfgURL)) as? [String: Any]) ?? [:]
        checks.append(.init(id: .jangConfigExists, title: "jang_config.json exists",
                            status: jangCfg.isEmpty ? .fail : .pass, required: true,
                            hint: jangCfg.isEmpty ? "Missing or unparseable jang_config.json" : nil))

        // #2 format + format_version. Legacy JANG records format/version;
        // JANGTQ records weight_format/profile and the safetensors index marks
        // metadata.format=jangtq.
        let fmt = (jangCfg["format"] as? String) ?? ""
        let ver = (jangCfg["format_version"] as? String) ?? ""
        let weightFormat = (jangCfg["weight_format"] as? String) ?? ""
        let profile = (jangCfg["profile"] as? String) ?? ""
        let okLegacyFormat = fmt == "jang" && (ver.hasPrefix("2.") || ver.hasPrefix("3."))
        let okJANGTQFormat = weightFormat == "mxtq" && profile.hasPrefix("JANGTQ")
        let okFmt = okLegacyFormat || okJANGTQFormat
        checks.append(.init(id: .jangConfigFormat,
                            title: okJANGTQFormat ? "JANGTQ format" : "jang format v2+",
                            status: okFmt ? .pass : .fail,
                            required: true,
                            hint: okFmt ? nil : "format=\(fmt) version=\(ver) weight_format=\(weightFormat) profile=\(profile)"))

        // #3 schema via python (skipped in unit tests)
        if !skipPythonValidate {
            let ok = await Self.runJangValidate(outputDir: out)
            checks.append(.init(id: .schemaValid, title: "jang validate passes", status: ok ? .pass : .fail,
                                required: true, hint: ok ? nil : "Run `jang validate` for details"))
        } else {
            checks.append(.init(id: .schemaValid, title: "jang validate passes", status: .pass, required: true, hint: "skipped in test"))
        }

        // #4 capabilities
        let caps = (jangCfg["capabilities"] as? [String: Any]) ?? [:]
        checks.append(.init(id: .capabilitiesPresent, title: "capabilities stamp present",
                            status: caps.isEmpty ? .fail : .pass, required: true,
                            hint: caps.isEmpty ? "jang_config.capabilities missing" : nil))
        let isTemporaryReviewRuntime = Self.isTemporaryReviewRuntime(plan: plan, outputDir: out)

        // #5 chat template (inline | .jinja | .json all accepted)
        let hasJinja = FileManager.default.fileExists(atPath: out.appendingPathComponent("chat_template.jinja").path)
        let hasChatJSON = FileManager.default.fileExists(atPath: out.appendingPathComponent("chat_template.json").path)
        let tokCfgData = try? Data(contentsOf: out.appendingPathComponent("tokenizer_config.json"))
        let tokCfg = (tokCfgData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
        let hasInline = !((tokCfg["chat_template"] as? String) ?? "").isEmpty
        let hasChat = hasJinja || hasChatJSON || hasInline
        let chatMissingStatus: PreflightStatus = isTemporaryReviewRuntime ? .warn : .fail
        let chatMissingHint = isTemporaryReviewRuntime
            ? "Temporary Expert Review runtime can open; prompt execution still needs chat metadata from the source model"
            : "No chat_template inline / .jinja / .json file"
        checks.append(.init(id: .chatTemplate, title: "Chat template present",
                            status: hasChat ? .pass : chatMissingStatus,
                            required: !isTemporaryReviewRuntime,
                            hint: hasChat ? nil : chatMissingHint))

        // #6 tokenizer files
        let hasJSON = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer.json").path)
        let hasModel = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer.model").path)
        let hasCfg = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer_config.json").path)
        let hasSpecial = FileManager.default.fileExists(atPath: out.appendingPathComponent("special_tokens_map.json").path)
        let hasSpecialMetadata = hasSpecial || Self.tokenizerConfigDefinesSpecialTokens(tokCfg)
        let okTok = (hasJSON || hasModel) && hasCfg && hasSpecialMetadata
        let tokenizerMissingStatus: PreflightStatus = isTemporaryReviewRuntime ? .warn : .fail
        let tokenizerMissingHint = isTemporaryReviewRuntime
            ? "Temporary Expert Review runtime can open; prompt execution may fail until tokenizer metadata is restored"
            : "Missing tokenizer.json|.model, tokenizer_config, or special token metadata"
        checks.append(.init(id: .tokenizerFiles, title: "Tokenizer files complete",
                            status: okTok ? .pass : tokenizerMissingStatus,
                            required: !isTemporaryReviewRuntime,
                            hint: okTok ? nil : tokenizerMissingHint))

        // #7 shards match index
        let idxURL = out.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: idxURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = obj["weight_map"] as? [String: String] {
            let shards = Set(map.values)
            let onDisk = Set(shards.filter { FileManager.default.fileExists(atPath: out.appendingPathComponent($0).path) })
            let ok = shards == onDisk
            checks.append(.init(id: .shardsMatchIndex, title: "Shards match index",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Index references \(shards.count) shards, \(onDisk.count) on disk"))
        } else {
            checks.append(.init(id: .shardsMatchIndex, title: "Shards match index",
                                status: .fail, required: true, hint: "model.safetensors.index.json missing"))
        }

        // #8 VL preprocessors
        if plan.detected?.isVL == true {
            let ok = FileManager.default.fileExists(atPath: out.appendingPathComponent("preprocessor_config.json").path)
            checks.append(.init(id: .vlPreprocessors, title: "VL preprocessor configs",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Missing preprocessor_config.json for VL model"))
        }

        // #8b Video VL preprocessor — only required when detected.isVideoVL
        if plan.detected?.isVideoVL == true {
            let ok = FileManager.default.fileExists(atPath: out.appendingPathComponent("video_preprocessor_config.json").path)
            checks.append(.init(id: .videoPreprocessors, title: "Video VL preprocessor config",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Missing video_preprocessor_config.json for video-VL model"))
        }

        // #9 MiniMax custom .py
        if plan.detected?.modelType == "minimax_m2" {
            let files = (try? FileManager.default.contentsOfDirectory(atPath: out.path)) ?? []
            let hasPyModel = files.contains { $0.hasPrefix("modeling_") && $0.hasSuffix(".py") }
            let hasPyCfg = files.contains { $0.hasPrefix("configuration_") && $0.hasSuffix(".py") }
            let ok = hasPyModel && hasPyCfg
            checks.append(.init(id: .miniMaxCustomPy, title: "MiniMax modeling_*.py + configuration_*.py",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "HF trust_remote_code will fail without these"))
        }

        // #10 tokenizer class concrete.
        // Memory ref `feedback_jang_studio_audit_coverage.md` makes this a hard
        // requirement: swift-transformers (Osaurus, vmlx-swift-lm) throws
        // `unsupportedTokenizer("TokenizersBackend")` and the model won't load.
        // Upgraded from warn-only to required=true in Ralph iter 5; the Python
        // side (convert.py Osaurus fix) now auto-remaps, so this verifier row
        // catches sources that slipped past the remap (e.g. an unmapped
        // model_type would leave the blocklist value intact).
        let cls = (tokCfg["tokenizer_class"] as? String) ?? ""
        let classOK = !cls.isEmpty && !capabilities.tokenizerClassBlocklist.contains(cls)
        checks.append(.init(id: .tokenizerClassConcrete, title: "Tokenizer class concrete",
                            status: classOK ? .pass : .fail, required: true,
                            hint: classOK ? nil : "tokenizer_class=\(cls) is in blocklist — Osaurus/vmlx-swift-lm will fail to load. Re-run convert; if it persists, add your model_type to the Osaurus remap in convert.py."))

        // #11 generation_config.json — HF consumers expect this. Warn only (HF will fall back to defaults).
        let hasGenCfg = FileManager.default.fileExists(atPath: out.appendingPathComponent("generation_config.json").path)
        checks.append(.init(id: .generationConfig, title: "generation_config.json present",
                            status: hasGenCfg ? .pass : .warn, required: false,
                            hint: hasGenCfg ? nil : "HF downstream loaders may fall back to unexpected defaults"))

        // #12 layer count sanity — config.json must have num_hidden_layers > 0
        let cfgData = try? Data(contentsOf: out.appendingPathComponent("config.json"))
        let cfgObj = (cfgData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
        let layerCount = (cfgObj["num_hidden_layers"] as? Int)
            ?? ((cfgObj["text_config"] as? [String: Any])?["num_hidden_layers"] as? Int)
            ?? 0
        checks.append(.init(id: .layerCountSane, title: "num_hidden_layers > 0",
                            status: layerCount > 0 ? .pass : .fail, required: true,
                            hint: layerCount > 0 ? "\(layerCount) layers" : "config.json missing or has num_hidden_layers=0"))

        // #13 (M116, iter 40) disk-size sanity — `feedback_model_checklist.md`
        // rule 2: "disk size ≈ GPU RAM. No bloat." Compare the actual on-disk
        // shard bytes to what target-bits * source-size would predict.
        checks.append(Self.diskSizeSanityCheck(
            outputDir: out,
            sourceBytes: plan.detected?.totalBytes ?? 0,
            sourceDtype: plan.detected?.dtype ?? .unknown,
            jangCfg: jangCfg))
        checks.append(contentsOf: Self.reviewedPruneChecks(plan: plan, outputDir: out))
        if plan.expertReviewPrunedSourceURL != nil {
            let planURL = plan.expertReviewPrunePlanURL
                ?? plan.expertReviewPrunedSourceURL?.appendingPathComponent("prune_plan.json")
            if skipNativeSmoke {
                checks.append(.init(
                    id: .expertLabNativeSmoke,
                    title: "Expert Lab native smoke prompts",
                    status: .fail,
                    required: true,
                    hint: "post-quant Expert Lab verification requires the reviewed prompt suite; native smoke was skipped"
                ))
            } else {
                checks.append(await Self.runExpertLabNativeSmoke(
                    outputDir: out,
                    suiteURL: Self.reviewedPruneSuiteURL(plan: plan),
                    requiresSuite: true
                ))
            }
            if let planURL {
                checks.append(Self.writeExpertLabFinalComparison(
                    plan: plan,
                    outputDir: out,
                    reviewedPlanURL: planURL
                ))
            }
        }

        return checks
    }

    private static func isTemporaryReviewRuntime(plan: ConversionPlan, outputDir: URL) -> Bool {
        guard plan.expertReviewIntent == .smartPrequantPrune,
              plan.expertReviewPrunedSourceURL == nil else {
            return false
        }
        guard let reviewBundle = plan.expertReviewBundleURL else {
            return true
        }
        return normalizedPath(reviewBundle) == normalizedPath(outputDir)
    }

    private static func normalizedPath(_ url: URL) -> String {
        url.resolvingSymlinksInPath().standardizedFileURL.path
    }

    private static func normalizedPath(_ path: String) -> String {
        normalizedPath(URL(fileURLWithPath: path))
    }

    /// Shared helper so tests can pin the size-sanity ratios independently.
    /// Returns a VerifyCheck for the current size-vs-estimate comparison.
    ///
    /// Estimate model: quantized size ≈ source_bytes * (actual_bits / 16)
    /// (assuming bf16 source). Warn window is ≥2× bloat OR ≤0.5× underrun;
    /// wider than strict to avoid false-positives from overhead files
    /// (tokenizer, chat templates, etc. add ~5-50 MB which is negligible on
    /// 10s-of-GB models but significant on small ones).
    static func diskSizeSanityCheck(
        outputDir: URL,
        sourceBytes: Int64,
        sourceDtype: SourceDtype = .unknown,
        jangCfg: [String: Any]
    ) -> VerifyCheck {
        // Sum *.safetensors bytes in the output. Skip imatrix (iter 38 M114:
        // local cache, not part of the model weights).
        let files = (try? FileManager.default.contentsOfDirectory(atPath: outputDir.path)) ?? []
        var diskBytes: Int64 = 0
        for f in files where f.hasSuffix(".safetensors") && f != "jang_imatrix.safetensors" {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: outputDir.appendingPathComponent(f).path),
               let size = attrs[.size] as? Int64 {
                diskBytes += size
            }
        }

        // Extract avg bits from jang_config.json. Keys vary across versions:
        // `quantization.actual_bits_per_weight` (v2) or `quantization.actual_bits` (v1).
        let quant = (jangCfg["quantization"] as? [String: Any]) ?? [:]
        let avgBits = (quant["actual_bits_per_weight"] as? Double)
            ?? (quant["actual_bits"] as? Double)
            ?? 0.0

        // M175 (iter 102): pre-M175 "missing data" → `.pass` with a hint
        // noting we couldn't check. Same ambiguous-pass anti-pattern M05
        // / M175 sweep closed elsewhere — UX-wise identical to a real
        // pass. Since this is a post-convert verifier (not a preflight
        // gate), the user has already converted when this fires; they
        // want to know whether the output is correctly-sized. A silent
        // pass on missing inputs hides the fact that THIS audit couldn't
        // happen. Promote to `.warn` so it's visually distinct.
        if sourceBytes <= 0 || avgBits <= 0 {
            return .init(id: .diskSizeSanity, title: "Disk size within expected range",
                         status: .warn, required: false,
                         hint: "couldn't compute estimate (missing source size or avg bits — this audit skipped, not run)")
        }

        // M174 (iter 100): cross-boundary formula audit per iter-99 M173's
        // meta-lesson. Same `/ 16.0` hardcoding as the preflight estimator
        // — assumes BF16 source. For FP8 source the "expected" came out
        // HALF the truth → ratio was 2× → falsely warned that correctly-
        // sized FP8-converted outputs were "bloated." Reuse the
        // `PreflightRunner.sourceBytesPerWeight(_:)` helper so both formulas
        // stay aligned; drift between them would re-introduce the class of
        // bug.
        let bytesPerWeight = Double(PreflightRunner.sourceBytesPerWeight(sourceDtype))
        let expectedBytes = Double(sourceBytes) * avgBits / (8.0 * bytesPerWeight)
        let ratio = Double(diskBytes) / expectedBytes
        let diskGB = Double(diskBytes) / 1_000_000_000
        let expectedGB = expectedBytes / 1_000_000_000

        // Tolerant thresholds: rule-2 is "≈ GPU RAM", not strict equality.
        // <0.5× suggests the shards are incomplete or the estimate is way off
        // (e.g. lots of embeddings skipped). >2× suggests bloat (extra shards
        // orphaned — matches the M115 failure mode this check is a safety net
        // for). Anything in [0.5, 2.0] passes silently.
        if ratio < 0.5 || ratio > 2.0 {
            return .init(
                id: .diskSizeSanity,
                title: "Disk size within expected range",
                status: .warn, required: false,
                hint: String(format: "disk=%.2f GB, expected≈%.2f GB (source=%.2f GB @ %.2f bits). ratio=%.2f×",
                             diskGB, expectedGB,
                             Double(sourceBytes) / 1_000_000_000, avgBits, ratio))
        }
        return .init(id: .diskSizeSanity, title: "Disk size within expected range",
                     status: .pass, required: false,
                     hint: String(format: "disk=%.2f GB ≈ expected %.2f GB (ratio=%.2f×)",
                                  diskGB, expectedGB, ratio))
    }

    static func reviewedPruneChecks(plan: ConversionPlan, outputDir: URL) -> [VerifyCheck] {
        guard let prunedSource = plan.expertReviewPrunedSourceURL else { return [] }
        let fm = FileManager.default
        let sourceMatches = plan.sourceURL == prunedSource
        let verificationURL = prunedSource.appendingPathComponent("verification.json")
        let verification = Self.readVerification(at: verificationURL)
        let planURL = Self.reviewedPrunePlanURL(plan: plan, prunedSource: prunedSource)
        let planExists = fm.fileExists(atPath: planURL.path)
        let reviewedPlan = planExists ? readJSONObject(planURL) : nil
        let planIssue: String?
        if !planExists {
            planIssue = "missing prune_plan.json in pruned BF16/F16 source"
        } else if let issue = Self.reviewedPrunePlanIssue(reviewedPlan) {
            planIssue = issue
        } else if let reviewedPlan,
                  let issue = Self.reviewedPrunePlanSidecarConsistencyIssue(
                    plan: reviewedPlan,
                    prunedSource: prunedSource
                  ) {
            planIssue = issue
        } else {
            planIssue = nil
        }
        let sameSuite = Self.reviewedPruneSameSuiteCheck(prunedSource: prunedSource)
        let finalReport = Self.writeExpertLabFinalReport(
            plan: plan,
            outputDir: outputDir,
            pruneVerificationOK: verification.ok,
            reviewedPlanURL: planURL
        )

        return [
            .init(
                id: .reviewedPruneSource,
                title: "Converted source is reviewed pruned BF16/F16",
                status: sourceMatches ? .pass : .fail,
                required: true,
                hint: sourceMatches ? prunedSource.path : "conversion source does not match adopted pruned source"
            ),
            .init(
                id: .reviewedPruneVerification,
                title: "Pruned source verification passed",
                status: verification.ok ? .pass : .fail,
                required: true,
                hint: verification.ok ? verificationURL.path : verification.hint
            ),
            .init(
                id: .reviewedPrunePlan,
                title: "Reviewed prune plan evidence passed",
                status: planIssue == nil ? .pass : .fail,
                required: true,
                hint: planIssue == nil ? planURL.path : planIssue
            ),
            sameSuite,
            finalReport
        ]
    }

    private static func reviewedPrunePlanURL(plan _: ConversionPlan, prunedSource: URL) -> URL {
        return prunedSource.appendingPathComponent("prune_plan.json")
    }

    private static func reviewedPruneSuiteURL(plan: ConversionPlan) -> URL? {
        guard let prunedSource = plan.expertReviewPrunedSourceURL else { return nil }
        let summaryURL = prunedSource.appendingPathComponent("expert_lab_review_summary.json")
        if let summary = readJSONObject(summaryURL),
           let suiteURL = embeddedSidecarURL(
            summary["suite_jsonl"],
            prunedSource: prunedSource,
            fallbackName: "expert_lab_suite.jsonl"
           ),
           FileManager.default.isReadableFile(atPath: suiteURL.path) {
            return suiteURL
        }
        let sidecar = prunedSource.appendingPathComponent("expert_lab_suite.jsonl")
        return FileManager.default.isReadableFile(atPath: sidecar.path) ? sidecar : nil
    }

    private static let requiredPrunedSourceVerificationChecks = [
        "config_parses",
        "index_parses",
        "index_covers_tensors",
        "router_rows_match",
        "expert_rows_match"
    ]

    private static func reviewedPruneSameSuiteCheck(prunedSource: URL) -> VerifyCheck {
        let title = "Same-suite Expert Lab review passed"
        let summaryURL = prunedSource.appendingPathComponent("expert_lab_review_summary.json")
        let fm = FileManager.default
        guard let summary = readJSONObject(summaryURL) else {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: "missing or unreadable expert_lab_review_summary.json"
            )
        }
        guard summary["same_suite_verification_ready"] as? Bool == true else {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: "same-suite review sidecars are incomplete"
            )
        }
        guard let reviewPrunedSource = stringValue(summary["pruned_source"] ?? summary["prunedSource"]),
              !reviewPrunedSource.isEmpty else {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: "Expert Lab review summary is missing pruned BF16/F16 source path evidence"
            )
        }
        if canonicalPath(reviewPrunedSource) != canonicalPath(prunedSource.path) {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: "Expert Lab review summary pruned source path does not match the selected pruned BF16/F16 source"
            )
        }
        let embeddedSidecars = embeddedReviewSidecars(summary: summary, prunedSource: prunedSource)
        if let issue = embeddedSidecars.issue {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: issue
            )
        }
        let sidecars = embeddedSidecars.urls
        let requiredPaths: [(String, URL?)] = [
            ("suite_jsonl", sidecars["suite_jsonl"]),
            ("comparison_summary", sidecars["comparison_summary"]),
            ("eval_jsonl", sidecars["eval_jsonl"]),
            ("eval_trace_jsonl", sidecars["eval_trace_jsonl"]),
            ("eval_index", sidecars["eval_index"]),
            ("mask_json", sidecars["mask_json"]),
        ]
        let missing = requiredPaths.compactMap { key, url -> String? in
            guard let url,
                  fm.isReadableFile(atPath: url.path) else {
                return key
            }
            return nil
        }
        if !missing.isEmpty {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: "missing Expert Lab sidecars: \(missing.joined(separator: ", "))"
            )
        }
        let expectedLayerCount = expectedReviewedLayerCount(summary: summary)
        guard let comparisonURL = sidecars["comparison_summary"],
              let comparison = readJSONObject(comparisonURL) else {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: "comparison summary is unreadable"
            )
        }
        if let issue = reviewedPruneComparisonGateIssue(
            comparison: comparison,
            tracedPromptCount: intValue(summary["prompt_count"])
        ) {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: issue
            )
        }
        if let issue = comparisonSafeDropMaskIssue(
            comparison: comparison,
            maskURL: sidecars["mask_json"]
        ) {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: issue
            )
        }
        let promptCount = intValue(comparison["promptCount"] ?? comparison["prompt_count"]) ?? 0
        if let suiteURL = sidecars["suite_jsonl"] {
            if let issue = suiteSemanticCoverageIssue(suiteURL) {
                return .init(
                    id: .reviewedPruneSameSuite,
                    title: title,
                    status: .fail,
                    required: true,
                    hint: issue
                )
            }
            let suiteCount = lineCount(suiteURL)
            if suiteCount != promptCount {
                return .init(
                    id: .reviewedPruneSameSuite,
                    title: title,
                    status: .fail,
                    required: true,
                    hint: "suite.jsonl has \(suiteCount) rows for \(promptCount) compared prompts"
                )
            }
        }
        if let evalURL = sidecars["eval_jsonl"] {
            let evalCount = lineCount(evalURL)
            if evalCount != promptCount {
                return .init(
                    id: .reviewedPruneSameSuite,
                    title: title,
                    status: .fail,
                    required: true,
                    hint: "eval.jsonl has \(evalCount) rows for \(promptCount) compared prompts"
                )
            }
        }
        if let evalIndexURL = sidecars["eval_index"] {
            guard let index = readJSONObject(evalIndexURL) else {
                return .init(
                    id: .reviewedPruneSameSuite,
                    title: title,
                    status: .fail,
                    required: true,
                    hint: "eval_index.json is unreadable"
                )
            }
            if let issue = reviewedPruneEvalIndexIssue(
                index: index,
                comparedPromptCount: promptCount,
                tracedPromptCount: intValue(summary["prompt_count"]),
                comparison: comparison,
                suiteURL: sidecars["suite_jsonl"],
                evalURL: sidecars["eval_jsonl"],
                evalTraceURL: sidecars["eval_trace_jsonl"],
                maskURL: sidecars["mask_json"],
                sourceModelPath: stringValue(summary["source_model_path"] ?? summary["source_model"]),
                expectedLayerCount: expectedLayerCount
            ) {
                return .init(
                    id: .reviewedPruneSameSuite,
                    title: title,
                    status: .fail,
                    required: true,
                    hint: issue
                )
            }
        }
        if let issue = prunedSourceSuiteVerificationIssue(
            summary: summary,
            prunedSource: prunedSource,
            tracedPromptCount: intValue(summary["prompt_count"]),
            suiteURL: sidecars["suite_jsonl"],
            expectedLayerCount: expectedLayerCount
        ) {
            return .init(
                id: .reviewedPruneSameSuite,
                title: title,
                status: .fail,
                required: true,
                hint: issue
            )
        }
        return .init(
            id: .reviewedPruneSameSuite,
            title: title,
            status: .pass,
            required: true,
            hint: summaryURL.path
        )
    }

    private static let minimumReviewedPrunePromptCount = 50
    private static let minimumReviewedPruneMeanTokens: Double = 8

    private static func reviewedPrunePlanIssue(_ plan: [String: Any]?) -> String? {
        guard let plan else {
            return "Reviewed prune plan JSON is unreadable"
        }
        guard let safety = plan["safety"] as? [String: Any] else {
            return "prune_plan.json is missing top-k safety evidence"
        }
        guard boolValue(safety["passed"]) == true else {
            return "embedded safety block did not pass"
        }
        if let issues = safety["issues"] as? [String], !issues.isEmpty {
            return issues.joined(separator: " ")
        }
        guard let minimumActive = intValue(
            safety["minimum_active_experts_per_layer"] ?? safety["minimumActiveExpertsPerLayer"]
        ) else {
            return "safety block is missing minimum active experts"
        }
        guard let keep = declaredKeepExperts(in: plan) else {
            return "prune plan is missing keep experts per layer"
        }
        if minimumActive != keep {
            return "safety declares \(minimumActive) active experts but plan keeps \(keep)"
        }
        guard let trainedTopK = maxTrainedTopK(in: safety) else {
            return "safety block is missing trained top-k evidence"
        }
        if keep < trainedTopK {
            return "plan keeps \(keep) experts but trained top-k is \(trainedTopK)"
        }
        guard let comparison = plan["comparison_summary"] as? [String: Any] else {
            return "prune_plan.json is missing embedded same-suite comparison evidence"
        }
        if let issue = reviewedPruneComparisonGateIssue(
            comparison: comparison,
            tracedPromptCount: intValue(plan["promptCount"] ?? plan["prompt_count"])
        ) {
            return issue
        }
        if let issue = reviewedPrunePlanDropEvidenceIssue(plan: plan, comparison: comparison) {
            return issue
        }
        guard let evalIndex = plan["eval_index"] as? [String: Any] else {
            return "prune_plan.json is missing embedded per-prompt eval_index evidence"
        }
        if let issue = reviewedPruneEvalIndexIssue(
            index: evalIndex,
            comparedPromptCount: intValue(comparison["promptCount"] ?? comparison["prompt_count"]),
            tracedPromptCount: intValue(plan["promptCount"] ?? plan["prompt_count"]),
            comparison: comparison,
            sourceModelPath: stringValue(plan["source_model"] ?? plan["sourceModelPath"]),
            expectedLayerCount: expectedReviewedLayerCount(summary: plan)
        ) {
            return issue
        }
        if let issue = reviewedPruneSemanticEvidenceIssue(plan) {
            return issue
        }
        return nil
    }

    private static func reviewedPrunePlanSidecarConsistencyIssue(
        plan: [String: Any],
        prunedSource: URL
    ) -> String? {
        let summaryURL = prunedSource.appendingPathComponent("expert_lab_review_summary.json")
        guard let summary = readJSONObject(summaryURL) else {
            return "expert_lab_review_summary.json is unreadable for prune_plan.json consistency"
        }
        let embeddedSidecars = embeddedReviewSidecars(summary: summary, prunedSource: prunedSource)
        if let issue = embeddedSidecars.issue {
            return issue
        }
        let sidecars = embeddedSidecars.urls

        guard let planComparison = plan["comparison_summary"] as? [String: Any] else {
            return nil
        }
        guard let comparisonURL = sidecars["comparison_summary"],
              let sidecarComparison = readJSONObject(comparisonURL) else {
            return "expert_lab_comparison_summary.json is unreadable for prune_plan.json consistency"
        }
        if let issue = comparisonEvidenceConsistencyIssue(plan: planComparison, sidecar: sidecarComparison) {
            return "prune_plan.json embedded comparison_summary does not match expert_lab_comparison_summary.json: \(issue)"
        }

        guard let planEvalIndex = plan["eval_index"] as? [String: Any] else {
            return nil
        }
        guard let evalIndexURL = sidecars["eval_index"],
              let sidecarEvalIndex = readJSONObject(evalIndexURL) else {
            return "expert_lab_eval_index.json is unreadable for prune_plan.json consistency"
        }
        if let issue = evalIndexEvidenceConsistencyIssue(plan: planEvalIndex, sidecar: sidecarEvalIndex) {
            return "prune_plan.json embedded eval_index does not match expert_lab_eval_index.json: \(issue)"
        }
        return nil
    }

    private static func comparisonEvidenceConsistencyIssue(
        plan: [String: Any],
        sidecar: [String: Any]
    ) -> String? {
        if let issue = intConsistencyIssue(
            label: "prompt count",
            plan: plan["promptCount"] ?? plan["prompt_count"],
            sidecar: sidecar["promptCount"] ?? sidecar["prompt_count"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "baseline pass rate",
            plan: plan["passRateBaseline"] ?? plan["pass_rate_baseline"],
            sidecar: sidecar["passRateBaseline"] ?? sidecar["pass_rate_baseline"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "masked pass rate",
            plan: plan["passRateMasked"] ?? plan["pass_rate_masked"],
            sidecar: sidecar["passRateMasked"] ?? sidecar["pass_rate_masked"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "mean text delta",
            plan: plan["meanTextDelta"] ?? plan["mean_text_delta"],
            sidecar: sidecar["meanTextDelta"] ?? sidecar["mean_text_delta"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "mean latency delta",
            plan: plan["meanLatencyDeltaPct"] ?? plan["mean_latency_delta_pct"],
            sidecar: sidecar["meanLatencyDeltaPct"] ?? sidecar["mean_latency_delta_pct"]
        ) {
            return issue
        }
        if let issue = stringSetConsistencyIssue(
            label: "high-risk domains",
            plan: plan["highRiskDomains"] ?? plan["high_risk_domains"],
            sidecar: sidecar["highRiskDomains"] ?? sidecar["high_risk_domains"]
        ) {
            return issue
        }
        if let issue = intConsistencyIssue(
            label: "validator-available prompt count",
            plan: plan["validatorAvailablePromptCount"] ?? plan["validator_available_prompt_count"],
            sidecar: sidecar["validatorAvailablePromptCount"] ?? sidecar["validator_available_prompt_count"]
        ) {
            return issue
        }
        if let issue = intConsistencyIssue(
            label: "baseline-qualified prompt count",
            plan: plan["baselineQualifiedPromptCount"] ?? plan["baseline_qualified_prompt_count"],
            sidecar: sidecar["baselineQualifiedPromptCount"] ?? sidecar["baseline_qualified_prompt_count"]
        ) {
            return issue
        }
        if let issue = stringArrayExactConsistencyIssue(
            label: "baseline-qualified prompt IDs",
            plan: plan["baselineQualifiedPromptIDs"] ?? plan["baseline_qualified_prompt_ids"],
            sidecar: sidecar["baselineQualifiedPromptIDs"] ?? sidecar["baseline_qualified_prompt_ids"]
        ) {
            return issue
        }
        if let issue = stringArrayExactConsistencyIssue(
            label: "degraded prompt IDs",
            plan: plan["degradedPromptIDs"] ?? plan["degraded_prompt_ids"],
            sidecar: sidecar["degradedPromptIDs"] ?? sidecar["degraded_prompt_ids"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "baseline-qualified masked pass rate",
            plan: plan["baselineQualifiedMaskedPassRate"] ?? plan["baseline_qualified_masked_pass_rate"],
            sidecar: sidecar["baselineQualifiedMaskedPassRate"] ?? sidecar["baseline_qualified_masked_pass_rate"]
        ) {
            return issue
        }
        if let issue = stringSetConsistencyIssue(
            label: "baseline-qualified semantic coverage",
            plan: plan["baselineQualifiedSemanticCoverage"] ?? plan["baseline_qualified_semantic_coverage"],
            sidecar: sidecar["baselineQualifiedSemanticCoverage"] ?? sidecar["baseline_qualified_semantic_coverage"]
        ) {
            return issue
        }
        if let issue = stringSetConsistencyIssue(
            label: "missing baseline-qualified semantic coverage",
            plan: plan["missingBaselineQualifiedSemanticCoverage"] ?? plan["missing_baseline_qualified_semantic_coverage"],
            sidecar: sidecar["missingBaselineQualifiedSemanticCoverage"] ?? sidecar["missing_baseline_qualified_semantic_coverage"]
        ) {
            return issue
        }
        if let issue = stringConsistencyIssue(
            label: "regression severity",
            plan: plan["regressionSeverity"] ?? plan["regression_severity"],
            sidecar: sidecar["regressionSeverity"] ?? sidecar["regression_severity"]
        ) {
            return issue
        }
        return coordinateSetConsistencyIssue(
            label: "safe-drop candidates",
            plan: plan["safeDropCandidates"] ?? plan["safe_drop_candidates"],
            sidecar: sidecar["safeDropCandidates"] ?? sidecar["safe_drop_candidates"]
        )
    }

    private static func evalIndexEvidenceConsistencyIssue(
        plan: [String: Any],
        sidecar: [String: Any]
    ) -> String? {
        let checks: [String?] = [
            intConsistencyIssue(
                label: "prompt count",
                plan: plan["prompt_count"] ?? plan["promptCount"],
                sidecar: sidecar["prompt_count"] ?? sidecar["promptCount"]
            ),
            stringArrayExactConsistencyIssue(
                label: "prompt IDs",
                plan: plan["prompt_ids"] ?? plan["promptIDs"],
                sidecar: sidecar["prompt_ids"] ?? sidecar["promptIDs"]
            ),
            stringSetConsistencyIssue(
                label: "risky prompt IDs",
                plan: plan["risky_prompt_ids"] ?? plan["riskyPromptIDs"],
                sidecar: sidecar["risky_prompt_ids"] ?? sidecar["riskyPromptIDs"]
            ),
            stringSetConsistencyIssue(
                label: "high-risk domains",
                plan: plan["high_risk_domains"] ?? plan["highRiskDomains"],
                sidecar: sidecar["high_risk_domains"] ?? sidecar["highRiskDomains"]
            ),
            stringConsistencyIssue(
                label: "validator schema",
                plan: plan["validator_schema"] ?? plan["validatorSchema"],
                sidecar: sidecar["validator_schema"] ?? sidecar["validatorSchema"]
            ),
            intConsistencyIssue(
                label: "validator-available prompt count",
                plan: plan["validator_available_prompt_count"] ?? plan["validatorAvailablePromptCount"],
                sidecar: sidecar["validator_available_prompt_count"] ?? sidecar["validatorAvailablePromptCount"]
            ),
            intConsistencyIssue(
                label: "baseline-qualified prompt count",
                plan: plan["baseline_qualified_prompt_count"] ?? plan["baselineQualifiedPromptCount"],
                sidecar: sidecar["baseline_qualified_prompt_count"] ?? sidecar["baselineQualifiedPromptCount"]
            ),
            stringArrayExactConsistencyIssue(
                label: "baseline-qualified prompt IDs",
                plan: plan["baseline_qualified_prompt_ids"] ?? plan["baselineQualifiedPromptIDs"],
                sidecar: sidecar["baseline_qualified_prompt_ids"] ?? sidecar["baselineQualifiedPromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "baseline-invalid prompt IDs",
                plan: plan["baseline_invalid_prompt_ids"] ?? plan["baselineInvalidPromptIDs"],
                sidecar: sidecar["baseline_invalid_prompt_ids"] ?? sidecar["baselineInvalidPromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "inconclusive prompt IDs",
                plan: plan["inconclusive_prompt_ids"] ?? plan["inconclusivePromptIDs"],
                sidecar: sidecar["inconclusive_prompt_ids"] ?? sidecar["inconclusivePromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "preserved prompt IDs",
                plan: plan["preserved_prompt_ids"] ?? plan["preservedPromptIDs"],
                sidecar: sidecar["preserved_prompt_ids"] ?? sidecar["preservedPromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "degraded prompt IDs",
                plan: plan["degraded_prompt_ids"] ?? plan["degradedPromptIDs"],
                sidecar: sidecar["degraded_prompt_ids"] ?? sidecar["degradedPromptIDs"]
            ),
            doubleConsistencyIssue(
                label: "baseline-qualified masked pass rate",
                plan: plan["baseline_qualified_masked_pass_rate"] ?? plan["baselineQualifiedMaskedPassRate"],
                sidecar: sidecar["baseline_qualified_masked_pass_rate"] ?? sidecar["baselineQualifiedMaskedPassRate"]
            ),
            stringSetConsistencyIssue(
                label: "baseline-qualified semantic coverage",
                plan: plan["baseline_qualified_semantic_coverage"] ?? plan["baselineQualifiedSemanticCoverage"],
                sidecar: sidecar["baseline_qualified_semantic_coverage"] ?? sidecar["baselineQualifiedSemanticCoverage"]
            ),
            stringSetConsistencyIssue(
                label: "missing baseline-qualified semantic coverage",
                plan: plan["missing_baseline_qualified_semantic_coverage"] ?? plan["missingBaselineQualifiedSemanticCoverage"],
                sidecar: sidecar["missing_baseline_qualified_semantic_coverage"] ?? sidecar["missingBaselineQualifiedSemanticCoverage"]
            ),
            stringSetConsistencyIssue(
                label: "semantic coverage",
                plan: plan["semantic_coverage"] ?? plan["semanticCoverage"],
                sidecar: sidecar["semantic_coverage"] ?? sidecar["semanticCoverage"]
            ),
            stringSetConsistencyIssue(
                label: "missing semantic coverage",
                plan: plan["missing_semantic_coverage"] ?? plan["missingSemanticCoverage"],
                sidecar: sidecar["missing_semantic_coverage"] ?? sidecar["missingSemanticCoverage"]
            ),
            doubleConsistencyIssue(
                label: "minimum baseline tokens",
                plan: plan["min_baseline_tokens"] ?? plan["minBaselineTokens"],
                sidecar: sidecar["min_baseline_tokens"] ?? sidecar["minBaselineTokens"]
            ),
            doubleConsistencyIssue(
                label: "minimum masked tokens",
                plan: plan["min_masked_tokens"] ?? plan["minMaskedTokens"],
                sidecar: sidecar["min_masked_tokens"] ?? sidecar["minMaskedTokens"]
            ),
            doubleConsistencyIssue(
                label: "mean baseline tokens",
                plan: plan["mean_baseline_tokens"] ?? plan["meanBaselineTokens"],
                sidecar: sidecar["mean_baseline_tokens"] ?? sidecar["meanBaselineTokens"]
            ),
            doubleConsistencyIssue(
                label: "mean masked tokens",
                plan: plan["mean_masked_tokens"] ?? plan["meanMaskedTokens"],
                sidecar: sidecar["mean_masked_tokens"] ?? sidecar["meanMaskedTokens"]
            ),
            intConsistencyIssue(
                label: "baseline route record count",
                plan: plan["baseline_route_record_count"] ?? plan["baselineRouteRecordCount"],
                sidecar: sidecar["baseline_route_record_count"] ?? sidecar["baselineRouteRecordCount"]
            ),
            intConsistencyIssue(
                label: "masked route record count",
                plan: plan["masked_route_record_count"] ?? plan["maskedRouteRecordCount"],
                sidecar: sidecar["masked_route_record_count"] ?? sidecar["maskedRouteRecordCount"]
            ),
            boolConsistencyIssue(
                label: "generation settings checked",
                plan: plan["generation_settings_checked"] ?? plan["generationSettingsChecked"],
                sidecar: sidecar["generation_settings_checked"] ?? sidecar["generationSettingsChecked"]
            ),
            stringConsistencyIssue(
                label: "runtime mode",
                plan: plan["runtime_mode"] ?? plan["runtimeMode"],
                sidecar: sidecar["runtime_mode"] ?? sidecar["runtimeMode"]
            ),
            stringConsistencyIssue(
                label: "runtime backend",
                plan: plan["runtime_backend"] ?? plan["runtimeBackend"],
                sidecar: sidecar["runtime_backend"] ?? sidecar["runtimeBackend"]
            ),
            stringConsistencyIssue(
                label: "runtime device",
                plan: plan["runtime_device"] ?? plan["runtimeDevice"],
                sidecar: sidecar["runtime_device"] ?? sidecar["runtimeDevice"]
            ),
            boolConsistencyIssue(
                label: "runtime Metal flag",
                plan: plan["runtime_metal_enabled"] ?? plan["runtimeMetalEnabled"],
                sidecar: sidecar["runtime_metal_enabled"] ?? sidecar["runtimeMetalEnabled"]
            ),
            boolConsistencyIssue(
                label: "hook coverage flag",
                plan: plan["hook_coverage_complete"] ?? plan["hookCoverageComplete"],
                sidecar: sidecar["hook_coverage_complete"] ?? sidecar["hookCoverageComplete"]
            ),
            intConsistencyIssue(
                label: "hooked MoE layers",
                plan: plan["hooked_moe_layers"] ?? plan["hookedMOELayers"],
                sidecar: sidecar["hooked_moe_layers"] ?? sidecar["hookedMOELayers"]
            ),
            intConsistencyIssue(
                label: "expected MoE layers",
                plan: plan["expected_moe_layers"] ?? plan["expectedMOELayers"],
                sidecar: sidecar["expected_moe_layers"] ?? sidecar["expectedMOELayers"]
            ),
            stringConsistencyIssue(
                label: "JANG tools version",
                plan: plan["jang_tools_version"] ?? plan["jangToolsVersion"],
                sidecar: sidecar["jang_tools_version"] ?? sidecar["jangToolsVersion"]
            ),
            stringConsistencyIssue(
                label: "MLX version",
                plan: plan["mlx_version"] ?? plan["mlxVersion"],
                sidecar: sidecar["mlx_version"] ?? sidecar["mlxVersion"]
            ),
            stringConsistencyIssue(
                label: "MLX-LM version",
                plan: plan["mlx_lm_version"] ?? plan["mlxLMVersion"],
                sidecar: sidecar["mlx_lm_version"] ?? sidecar["mlxLMVersion"]
            ),
            normalizedPathConsistencyIssue(
                label: "source model path",
                plan: plan["source_model_path"] ?? plan["sourceModelPath"],
                sidecar: sidecar["source_model_path"] ?? sidecar["sourceModelPath"]
            ),
            boolConsistencyIssue(
                label: "mask applied flag",
                plan: plan["mask_applied"] ?? plan["maskApplied"],
                sidecar: sidecar["mask_applied"] ?? sidecar["maskApplied"]
            ),
            intConsistencyIssue(
                label: "disabled expert count",
                plan: plan["disabled_expert_count"] ?? plan["disabledExpertCount"],
                sidecar: sidecar["disabled_expert_count"] ?? sidecar["disabledExpertCount"]
            ),
            intConsistencyIssue(
                label: "top-k override",
                plan: plan["top_k_override"] ?? plan["topKOverride"],
                sidecar: sidecar["top_k_override"] ?? sidecar["topKOverride"]
            )
        ]
        return checks.first { $0 != nil } ?? nil
    }

    private static func intConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = intValue(plan),
              let sidecarValue = intValue(sidecar) else {
            return nil
        }
        return planValue == sidecarValue ? nil : "\(label) differs"
    }

    private static func doubleConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = doubleValue(plan),
              let sidecarValue = doubleValue(sidecar) else {
            return nil
        }
        return doubleEqual(planValue, sidecarValue) ? nil : "\(label) differs"
    }

    private static func boolConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = boolValue(plan),
              let sidecarValue = boolValue(sidecar) else {
            return nil
        }
        return planValue == sidecarValue ? nil : "\(label) differs"
    }

    private static func stringConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = nonEmptyString(stringValue(plan)),
              let sidecarValue = nonEmptyString(stringValue(sidecar)) else {
            return nil
        }
        return planValue == sidecarValue ? nil : "\(label) differs"
    }

    private static func normalizedPathConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = nonEmptyString(stringValue(plan)),
              let sidecarValue = nonEmptyString(stringValue(sidecar)) else {
            return nil
        }
        return normalizedPath(planValue) == normalizedPath(sidecarValue) ? nil : "\(label) differs"
    }

    private static func stringSetConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValues = stringArrayValue(plan),
              let sidecarValues = stringArrayValue(sidecar) else {
            return nil
        }
        return Set(planValues) == Set(sidecarValues) ? nil : "\(label) differ"
    }

    private static func stringArrayExactConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValues = stringArrayValue(plan),
              let sidecarValues = stringArrayValue(sidecar) else {
            return nil
        }
        return planValues == sidecarValues ? nil : "\(label) differ"
    }

    private static func coordinateSetConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planCoordinates = coordinateSet(plan),
              let sidecarCoordinates = coordinateSet(sidecar) else {
            return nil
        }
        return planCoordinates == sidecarCoordinates ? nil : "\(label) differ"
    }

    private static func coordinateSet(_ value: Any?) -> Set<String>? {
        guard let rows = value as? [Any] else { return nil }
        var coordinates = Set<String>()
        for row in rows {
            guard let object = row as? [String: Any],
                  let layer = intValue(object["layer"] ?? object["layer_id"] ?? object["layerID"]),
                  let expert = intValue(object["expert"] ?? object["expert_id"] ?? object["expertID"]) else {
                return nil
            }
            coordinates.insert("\(layer):\(expert)")
        }
        return coordinates
    }

    private static func comparisonSafeDropMaskIssue(
        comparison: [String: Any],
        maskURL: URL?
    ) -> String? {
        guard let safeDrops = coordinateSet(comparison["safeDropCandidates"] ?? comparison["safe_drop_candidates"]),
              !safeDrops.isEmpty else {
            return nil
        }
        guard let maskURL,
              let disabled = disabledCoordinateSet(fromMaskURL: maskURL) else {
            return "mask.json is unreadable"
        }
        guard !disabled.isEmpty else {
            return "mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning"
        }
        if safeDrops != disabled {
            return "comparison_summary.json safe-drop candidates do not match mask.json disabled experts: safe \(previewCoordinates(safeDrops)); mask \(previewCoordinates(disabled))"
        }
        return nil
    }

    private static func reviewedPrunePlanDropEvidenceIssue(
        plan: [String: Any],
        comparison: [String: Any]
    ) -> String? {
        guard let safeDrops = coordinateSet(comparison["safeDropCandidates"] ?? comparison["safe_drop_candidates"]) else {
            return nil
        }
        guard let plannedDrops = planDropCoordinateSet(plan) else {
            return "prune_plan.json planned drop list is unreadable"
        }
        let unsafeDrops = plannedDrops.subtracting(safeDrops)
        if !unsafeDrops.isEmpty {
            return "prune_plan.json drops experts outside same-suite safe-drop candidates: \(previewCoordinates(unsafeDrops))"
        }
        return nil
    }

    private static func disabledCoordinateSet(fromMaskURL url: URL) -> Set<String>? {
        guard let disabledByLayer = disabledExpertsByLayer(fromMaskURL: url) else { return nil }
        return Set(disabledByLayer.flatMap { layer, experts in
            experts.map { "\(layer):\($0)" }
        })
    }

    private static func planDropCoordinateSet(_ plan: [String: Any]) -> Set<String>? {
        guard let layers = plan["layers"] as? [String: Any] else { return nil }
        var coordinates = Set<String>()
        for key in layers.keys {
            guard let layer = layers[key] as? [String: Any],
                  let layerID = intValue(layer["layer"]) ?? intValue(key),
                  let drops = layer["drop"] as? [Any] else {
                return nil
            }
            for drop in drops {
                guard let expert = intValue(drop) else { return nil }
                coordinates.insert("\(layerID):\(expert)")
            }
        }
        return coordinates
    }

    private static func previewCoordinates(_ coordinates: Set<String>) -> String {
        let display = coordinates.sorted().map { coordinate -> String in
            let parts = coordinate.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { return coordinate }
            return "L\(parts[0]) E\(parts[1])"
        }
        let head = display.prefix(5).joined(separator: ", ")
        let remaining = max(0, display.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func reviewedPruneSemanticEvidenceIssue(_ plan: [String: Any]) -> String? {
        guard let layers = plan["layers"] as? [String: Any], !layers.isEmpty else {
            return "prune_plan.json is missing layer evidence rows"
        }
        var checkedRows = 0
        for layerKey in layers.keys.sorted() {
            guard let layer = layers[layerKey] as? [String: Any] else {
                return "prune_plan.json layer \(layerKey) is unreadable"
            }
            guard let evidenceRows = layer["evidence"] as? [[String: Any]], !evidenceRows.isEmpty else {
                return "prune_plan.json layer \(layerKey) is missing expert evidence rows"
            }
            for row in evidenceRows {
                let label = stringValue(row["label"]) ?? ""
                let normalizedLabel = label.lowercased()
                let hits = intValue(row["hits"]) ?? 0
                let domains = row["domains"] as? [String: Any] ?? [:]
                let isUnobserved = normalizedLabel.contains("unobserved") && hits == 0 && domains.isEmpty
                if isUnobserved { continue }

                checkedRows += 1
                let layerID = stringValue(layer["layer"]) ?? layerKey
                let expertID = stringValue(row["expert"]) ?? "?"
                let coordinate = "L\(layerID) E\(expertID)"

                guard doubleValue(row["router_mass"] ?? row["routerMass"] ?? row["probabilityMass"] ?? row["probability_mass"]) != nil else {
                    return "prune_plan.json evidence row \(coordinate) is missing gate mass evidence"
                }
                guard doubleValue(row["ablation_delta"] ?? row["ablationDelta"]) != nil else {
                    return "prune_plan.json evidence row \(coordinate) is missing masked-output impact evidence"
                }
                guard let impactScope = stringValue(row["masked_impact_scope"] ?? row["maskedImpactScope"]),
                      !impactScope.isEmpty else {
                    return "prune_plan.json evidence row \(coordinate) is missing masked-output impact scope evidence"
                }
                guard jsonBool(row["reviewed_mask_member"] ?? row["reviewedMaskMember"]) != nil else {
                    return "prune_plan.json evidence row \(coordinate) is missing reviewed mask membership evidence"
                }
                guard let domainLift = row["domain_lift"] as? [String: Any],
                      domainLift.contains(where: { doubleValue($0.value) != nil }) else {
                    return "prune_plan.json evidence row \(coordinate) is missing activation lift evidence"
                }
                guard let promptEvidence = row["prompt_evidence"] as? [[String: Any]], !promptEvidence.isEmpty else {
                    return "prune_plan.json evidence row \(coordinate) is missing prompt example evidence"
                }
                let hasPromptProof = promptEvidence.contains { prompt in
                    stringValue(prompt["promptID"] ?? prompt["prompt_id"]) != nil &&
                    stringValue(prompt["domain"]) != nil &&
                    stringValue(prompt["promptExcerpt"] ?? prompt["prompt_excerpt"]) != nil &&
                    (prompt["tags"] as? [Any])?.isEmpty == false &&
                    (intValue(prompt["hits"]) ?? 0) > 0
                }
                if !hasPromptProof {
                    return "prune_plan.json evidence row \(coordinate) has incomplete prompt tags/examples"
                }
            }
        }
        return checkedRows == 0 ? "prune_plan.json has no semantic expert evidence rows" : nil
    }

    private static func declaredKeepExperts(in plan: [String: Any]) -> Int? {
        if let keep = intValue(plan["keepExpertsPerLayer"] ?? plan["keep_experts_per_layer"]) {
            return keep
        }
        if let target = plan["target"] as? [String: Any],
           let keep = intValue(target["keep_experts_per_layer"] ?? target["keepExpertsPerLayer"]) {
            return keep
        }
        guard let layers = plan["layers"] as? [String: Any] else { return nil }
        let keepCounts = Set(layers.values.compactMap { value -> Int? in
            guard let layer = value as? [String: Any],
                  let keep = layer["keep"] as? [Any] else {
                return nil
            }
            return keep.count
        })
        return keepCounts.count == 1 ? keepCounts.first : nil
    }

    private static func maxTrainedTopK(in safety: [String: Any]) -> Int? {
        let raw = safety["trained_top_k_by_layer"] ?? safety["trainedTopKByLayer"]
        guard let topKByLayer = raw as? [String: Any] else { return nil }
        let values = topKByLayer.values.compactMap(intValue)
        return values.isEmpty ? nil : values.max()
    }

    private static func reviewedPruneComparisonGateIssue(
        comparison: [String: Any],
        tracedPromptCount: Int?
    ) -> String? {
        let promptCount = intValue(comparison["promptCount"] ?? comparison["prompt_count"]) ?? 0
        if promptCount < minimumReviewedPrunePromptCount {
            return "compare at least \(minimumReviewedPrunePromptCount) prompts before final quantization"
        }
        if let tracedPromptCount,
           tracedPromptCount > 0,
           promptCount != tracedPromptCount {
            return "rerun A/B compare for all \(tracedPromptCount) traced prompts"
        }
        if let highRiskDomains = comparison["highRiskDomains"] as? [String],
           !highRiskDomains.isEmpty {
            return "masked outputs regressed in high-risk domains: \(highRiskDomains.sorted().joined(separator: ", "))"
        }
        if let highRiskDomains = comparison["high_risk_domains"] as? [String],
           !highRiskDomains.isEmpty {
            return "masked outputs regressed in high-risk domains: \(highRiskDomains.sorted().joined(separator: ", "))"
        }
        if isBlockingRegressionSeverity(comparison["regressionSeverity"] ?? comparison["regression_severity"]) {
            return "masked comparison regression severity is high or critical"
        }
        let safeDropCandidates = comparison["safeDropCandidates"] ?? comparison["safe_drop_candidates"]
        guard let candidates = safeDropCandidates as? [Any] else {
            return "comparison summary is missing A/B-safe candidates"
        }
        if candidates.isEmpty {
            return "A/B comparison found no safe drop candidates"
        }
        if let issue = reviewedPruneComparisonValidatorIssue(comparison) {
            return issue
        }
        return nil
    }

    private static func reviewedPruneComparisonValidatorIssue(_ comparison: [String: Any]) -> String? {
        guard intValue(comparison["validatorAvailablePromptCount"] ?? comparison["validator_available_prompt_count"]) != nil,
              dictionaryValue(comparison["classificationCounts"] ?? comparison["prompt_classification_counts"]) != nil else {
            return "comparison summary is missing validator classification evidence"
        }
        guard let baselineQualified = intValue(
            comparison["baselineQualifiedPromptCount"] ?? comparison["baseline_qualified_prompt_count"]
        ),
              baselineQualified > 0 else {
            return "comparison summary has no baseline-qualified validator prompts"
        }
        let missingCoverage = stringArrayValue(
            comparison["missingBaselineQualifiedSemanticCoverage"]
                ?? comparison["missing_baseline_qualified_semantic_coverage"]
        ) ?? []
        if !missingCoverage.isEmpty {
            return "baseline-qualified prompts are missing semantic coverage: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        let degraded = stringArrayValue(comparison["degradedPromptIDs"] ?? comparison["degraded_prompt_ids"]) ?? []
        if !degraded.isEmpty {
            return "baseline-qualified prompts degraded after masking: \(previewIDs(Set(degraded)))"
        }
        guard let coverage = stringArrayValue(
            comparison["baselineQualifiedSemanticCoverage"]
                ?? comparison["baseline_qualified_semantic_coverage"]
        ),
              !coverage.isEmpty else {
            return "comparison summary is missing baseline-qualified semantic coverage evidence"
        }
        if let passRate = doubleValue(
            comparison["baselineQualifiedMaskedPassRate"] ?? comparison["baseline_qualified_masked_pass_rate"]
        ),
            passRate < 1.0 {
            return "masked validator pass rate is below 100% on baseline-qualified prompts"
        }
        return nil
    }

    private static func embeddedReviewSidecars(
        summary: [String: Any],
        prunedSource: URL
    ) -> (urls: [String: URL], issue: String?) {
        let specs: [(key: String, value: Any?, fileName: String)] = [
            ("suite_jsonl", summary["suite_jsonl"], "expert_lab_suite.jsonl"),
            ("comparison_summary", summary["comparison_summary"], "expert_lab_comparison_summary.json"),
            ("eval_jsonl", summary["eval_jsonl"], "expert_lab_eval.jsonl"),
            ("eval_trace_jsonl", summary["eval_trace_jsonl"], "expert_lab_eval_trace.jsonl"),
            ("eval_index", summary["eval_index"], "expert_lab_eval_index.json"),
            ("mask_json", summary["mask_json"] ?? summary["mask"] ?? summary["maskJSON"], "mask.json"),
        ]
        var urls: [String: URL] = [:]
        var outside: [String] = []
        for spec in specs {
            if let url = embeddedSidecarURL(spec.value, prunedSource: prunedSource, fallbackName: spec.fileName) {
                urls[spec.key] = url
            } else {
                outside.append(spec.key)
            }
        }
        if !outside.isEmpty {
            return (
                urls,
                "Expert Lab review summary sidecar paths must be embedded in the pruned BF16/F16 source: \(outside.joined(separator: ", "))"
            )
        }
        return (urls, nil)
    }

    private static func embeddedSidecarURL(_ value: Any?, prunedSource: URL, fallbackName: String) -> URL? {
        let expected = prunedSource.appendingPathComponent(fallbackName)
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return expected
        }
        let expanded = (raw as NSString).expandingTildeInPath
        let recorded = (expanded as NSString).isAbsolutePath
            ? URL(fileURLWithPath: expanded)
            : prunedSource.appendingPathComponent(expanded)
        return canonicalPath(recorded.path) == canonicalPath(expected.path) ? expected : nil
    }

    private static func reviewedPruneEvalIndexIssue(
        index: [String: Any],
        comparedPromptCount: Int?,
        tracedPromptCount: Int?,
        comparison: [String: Any]? = nil,
        suiteURL: URL? = nil,
        evalURL: URL? = nil,
        evalTraceURL: URL? = nil,
        maskURL: URL? = nil,
        sourceModelPath: String? = nil,
        expectedLayerCount: Int? = nil
    ) -> String? {
        let promptCount = intValue(index["prompt_count"] ?? index["promptCount"]) ?? 0
        let promptIDs = stringArrayValue(index["prompt_ids"] ?? index["promptIDs"]) ?? []
        if promptIDs.count != promptCount {
            return "eval_index.json lists \(promptIDs.count) prompt IDs for \(promptCount) indexed prompts"
        }
        if Set(promptIDs).count < promptIDs.count {
            return "eval_index.json contains duplicate prompt IDs"
        }
        if let issue = evalIndexSemanticCoverageIssue(index) {
            return issue
        }
        let indexedIDs = Set(promptIDs)
        var evalRowsForDecodeSettings: [[String: Any]]?
        if let suiteURL {
            guard let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]) else {
                return "suite.jsonl prompt IDs are unreadable"
            }
            if Set(suiteIDs).count < suiteIDs.count {
                return "suite.jsonl contains duplicate prompt IDs"
            }
            let suiteSet = Set(suiteIDs)
            let missing = suiteSet.subtracting(indexedIDs)
            if !missing.isEmpty {
                return "eval_index.json prompt IDs missing suite.jsonl prompts: \(previewIDs(missing))"
            }
            let unexpected = indexedIDs.subtracting(suiteSet)
            if !unexpected.isEmpty {
                return "eval_index.json prompt IDs outside suite.jsonl: \(previewIDs(unexpected))"
            }
            if promptIDs != suiteIDs {
                return "eval_index.json prompt order does not match suite.jsonl"
            }
        }
        if let evalURL {
            guard let evalIDs = jsonlStringIDs(evalURL, keys: ["promptID", "prompt_id", "id"]) else {
                return "eval.jsonl prompt IDs are unreadable"
            }
            if Set(evalIDs).count < evalIDs.count {
                return "eval.jsonl contains duplicate prompt IDs"
            }
            let evalSet = Set(evalIDs)
            let missing = indexedIDs.subtracting(evalSet)
            let unexpected = evalSet.subtracting(indexedIDs)
            if !missing.isEmpty, !unexpected.isEmpty {
                return "eval.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpected)); eval_index.json prompt IDs missing from eval.jsonl: \(previewIDs(missing))"
            }
            if !missing.isEmpty {
                return "eval_index.json prompt IDs missing from eval.jsonl: \(previewIDs(missing))"
            }
            if !unexpected.isEmpty {
                return "eval.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpected))"
            }
            if evalIDs != promptIDs {
                return "eval.jsonl prompt order does not match eval_index.json"
            }
            guard let evalRows = jsonlObjects(evalURL) else {
                return "eval.jsonl is unreadable"
            }
            evalRowsForDecodeSettings = evalRows
            if let issue = evalRowEvidenceIssue(
                rows: evalRows,
                expectedPromptIDs: promptIDs,
                sourceModelPath: sourceModelPath
            ) {
                return issue
            }
        }
        let expectedDisabledByLayer: [Int: Set<Int>]?
        if let maskURL {
            guard let mask = disabledExpertsByLayer(fromMaskURL: maskURL) else {
                return "mask.json is unreadable"
            }
            let disabledCount = mask.values.reduce(0) { $0 + $1.count }
            guard disabledCount > 0 else {
                return "mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning"
            }
            if let indexDisabledCount = intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
               indexDisabledCount != disabledCount {
                return "eval_index.json disabled expert count \(indexDisabledCount) does not match mask.json \(disabledCount)"
            }
            expectedDisabledByLayer = mask
        } else {
            expectedDisabledByLayer = nil
        }
        guard let baselineRouteRecordCount = intValue(index["baseline_route_record_count"] ?? index["baselineRouteRecordCount"]),
              let maskedRouteRecordCount = intValue(index["masked_route_record_count"] ?? index["maskedRouteRecordCount"]),
              baselineRouteRecordCount >= promptCount,
              maskedRouteRecordCount >= promptCount else {
            return "eval_index.json is missing routing record evidence for every indexed prompt"
        }
        if let evalTraceURL {
            guard let traceIDs = jsonlStringIDs(evalTraceURL, keys: ["promptID", "prompt_id", "id"]) else {
                return "eval_trace.jsonl prompt IDs are unreadable"
            }
            if traceIDs.isEmpty {
                return "eval_trace.jsonl has no routing records"
            }
            let traceSet = Set(traceIDs)
            let missing = indexedIDs.subtracting(traceSet)
            if !missing.isEmpty {
                return "eval_index.json prompt IDs missing from eval_trace.jsonl: \(previewIDs(missing))"
            }
            let unexpected = traceSet.subtracting(indexedIDs)
            if !unexpected.isEmpty {
                return "eval_trace.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpected))"
            }
            guard let traceRows = jsonlObjects(evalTraceURL) else {
                return "eval_trace.jsonl is unreadable"
            }
            if let issue = evalTraceVariantIssue(
                rows: traceRows,
                expectedPromptIDs: promptIDs,
                disabledExpertCount: intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
                topKOverride: intValue(index["top_k_override"] ?? index["topKOverride"]),
                expectedDisabledByLayer: expectedDisabledByLayer,
                expectedBaselineRouteRecordCount: baselineRouteRecordCount,
                expectedMaskedRouteRecordCount: maskedRouteRecordCount
            ) {
                return issue
            }
        }
        if let comparedPromptCount, promptCount != comparedPromptCount {
            return "eval_index.json covers \(promptCount) of \(comparedPromptCount) compared prompts"
        }
        if let tracedPromptCount, tracedPromptCount > 0, promptCount != tracedPromptCount {
            return "eval_index.json covers \(promptCount) of \(tracedPromptCount) traced prompts"
        }
        if let risky = index["risky_prompt_ids"] as? [Any], !risky.isEmpty {
            return "eval_index.json still has risky prompt IDs"
        }
        if let risky = index["riskyPromptIDs"] as? [Any], !risky.isEmpty {
            return "eval_index.json still has risky prompt IDs"
        }
        if isBlockingRegressionSeverity(index["regression_severity"] ?? index["regressionSeverity"]) {
            return "eval_index.json regression severity is high or critical"
        }
        if let highRiskDomains = index["high_risk_domains"] as? [Any], !highRiskDomains.isEmpty {
            return "eval_index.json still has high-risk domains"
        }
        if let highRiskDomains = index["highRiskDomains"] as? [Any], !highRiskDomains.isEmpty {
            return "eval_index.json still has high-risk domains"
        }
        if let issue = evalComparisonConsistencyIssue(
            comparison: comparison,
            index: index,
            evalRows: evalRowsForDecodeSettings
        ) {
            return issue
        }
        guard let meanBaselineTokens = doubleValue(index["mean_baseline_tokens"] ?? index["meanBaselineTokens"]),
              let meanMaskedTokens = doubleValue(index["mean_masked_tokens"] ?? index["meanMaskedTokens"]) else {
            return "eval_index.json is missing generation-depth token evidence"
        }
        let shallow = min(meanBaselineTokens, meanMaskedTokens)
        if shallow < minimumReviewedPruneMeanTokens {
            return String(
                format: "eval_index.json average generated depth %.1f tokens is below %.0f",
                shallow,
                minimumReviewedPruneMeanTokens
            )
        }
        if let issue = evalIndexLayerStatsCoverageIssue(index: index, promptCount: promptCount) {
            return issue
        }
        guard stringValue(index["eval_trace_jsonl"] ?? index["evalTraceJSONL"]) != nil else {
            return "eval_index.json is missing eval_trace.jsonl evidence"
        }
        guard let runtimeMode = stringValue(index["runtime_mode"] ?? index["runtimeMode"]),
              !runtimeMode.isEmpty,
              let runtimeDevice = stringValue(index["runtime_device"] ?? index["runtimeDevice"]),
              !runtimeDevice.isEmpty,
              let runtimeMetalEnabled = boolValue(index["runtime_metal_enabled"] ?? index["runtimeMetalEnabled"]) else {
            return "eval_index.json is missing runtime device evidence"
        }
        if runtimeMetalEnabled != true {
            return "eval_index.json did not record a Metal runtime"
        }
        if runtimeMode != "bf16_vmlx" {
            return "eval_index.json did not record BF16/vMLX runtime evidence"
        }
        if stringValue(index["runtime_backend"] ?? index["runtimeBackend"]) != "vmlx" {
            return "eval_index.json did not record vMLX backend evidence"
        }
        if boolValue(index["hook_coverage_complete"] ?? index["hookCoverageComplete"]) == false {
            return "eval_index.json recorded incomplete vMLX routed-layer hook coverage"
        }
        if let expectedLayerCount {
            guard let hookedLayers = intValue(index["hooked_moe_layers"] ?? index["hookedMOELayers"]) else {
                return "eval_index.json is missing vMLX routed-layer hook evidence"
            }
            if hookedLayers < expectedLayerCount {
                return "eval_index.json vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers"
            }
        }
        if let expectedMOELayers = intValue(index["expected_moe_layers"] ?? index["expectedMOELayers"]),
           let hookedLayers = intValue(index["hooked_moe_layers"] ?? index["hookedMOELayers"]),
           hookedLayers < expectedMOELayers {
            return "eval_index.json vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers"
        }
        guard let jangToolsVersion = stringValue(index["jang_tools_version"] ?? index["jangToolsVersion"]),
              !jangToolsVersion.isEmpty,
              let mlxVersion = stringValue(index["mlx_version"] ?? index["mlxVersion"]),
              !mlxVersion.isEmpty,
              let mlxLMVersion = stringValue(index["mlx_lm_version"] ?? index["mlxLMVersion"]),
              !mlxLMVersion.isEmpty else {
            return "eval_index.json is missing vMLX package version evidence"
        }
        guard let evalSourcePath = stringValue(index["source_model_path"] ?? index["sourceModelPath"]) else {
            return "eval_index.json is missing source model path evidence"
        }
        if let sourceModelPath,
           normalizedPath(evalSourcePath) != normalizedPath(sourceModelPath) {
            return "eval_index.json source model path does not match reviewed source"
        }
        guard boolValue(index["mask_applied"] ?? index["maskApplied"]) == true else {
            return "eval_index.json did not record an applied BF16/vMLX mask"
        }
        guard let disabledExpertCount = intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
              disabledExpertCount > 0 else {
            return "eval_index.json did not record disabled expert evidence; top-k-only comparisons cannot authorize hard pruning"
        }
        if let evalRowsForDecodeSettings,
           let issue = evalDecodeSettingsIssue(index: index, rows: evalRowsForDecodeSettings) {
            return issue
        }
        if let issue = evalIndexValidatorEvidenceIssue(index) {
            return issue
        }
        if let suiteURL {
            guard let expectedSuiteSHA256 = fileSHA256(suiteURL) else {
                return "eval_index.json suite.jsonl fingerprint could not be computed"
            }
            guard let recordedSuiteSHA256 = stringValue(index["suite_sha256"] ?? index["suiteSHA256"]),
                  !recordedSuiteSHA256.isEmpty else {
                return "eval_index.json is missing suite.jsonl fingerprint evidence"
            }
            if recordedSuiteSHA256 != expectedSuiteSHA256 {
                return "eval_index.json suite.jsonl fingerprint does not match suite.jsonl"
            }
        }
        return nil
    }

    private static func evalRowEvidenceIssue(
        rows: [[String: Any]],
        expectedPromptIDs: [String],
        sourceModelPath: String?
    ) -> String? {
        if rows.count != expectedPromptIDs.count {
            return "eval.jsonl has \(rows.count) rows for \(expectedPromptIDs.count) indexed prompts"
        }
        let rowIDs = rows.compactMap { promptID(in: $0) }
        if rowIDs.count != rows.count {
            return "eval.jsonl prompt IDs are unreadable"
        }
        if rowIDs != expectedPromptIDs {
            return "eval.jsonl prompt order does not match eval_index.json"
        }
        if rows.contains(where: {
            trimmedString($0["baselineText"] ?? $0["baseline_text"]) == nil
                || trimmedString($0["maskedText"] ?? $0["masked_text"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt baseline/masked output text"
        }
        if rows.contains(where: {
            guard let textDelta = doubleValue($0["textDelta"] ?? $0["text_delta"]),
                  textDelta.isFinite else {
                return true
            }
            return false
        }) {
            return "eval.jsonl is missing per-prompt text delta evidence"
        }
        if rows.contains(where: {
            (intValue($0["baselineTokenCount"] ?? $0["baseline_token_count"]) ?? 0) <= 0
                || (intValue($0["maskedTokenCount"] ?? $0["masked_token_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt token count evidence"
        }
        if rows.contains(where: {
            (intValue($0["baselineRouteRecordCount"] ?? $0["baseline_route_record_count"]) ?? 0) <= 0
                || (intValue($0["maskedRouteRecordCount"] ?? $0["masked_route_record_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt routing record evidence"
        }
        if let issue = evalRowLayerStatsCoverageIssue(rows: rows) {
            return issue
        }
        if rows.contains(where: {
            trimmedString($0["runtimeMode"] ?? $0["runtime_mode"]) == nil
                || trimmedString($0["runtimeDevice"] ?? $0["runtime_device"]) == nil
                || boolValue($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt runtime device evidence"
        }
        if rows.contains(where: {
            boolValue($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) != true
        }) {
            return "eval.jsonl did not record a Metal runtime"
        }
        if rows.contains(where: {
            trimmedString($0["runtimeMode"] ?? $0["runtime_mode"]) != "bf16_vmlx"
        }) {
            return "eval.jsonl did not record BF16/vMLX runtime evidence"
        }
        if rows.contains(where: {
            trimmedString($0["runtimeBackend"] ?? $0["runtime_backend"]) != "vmlx"
        }) {
            return "eval.jsonl did not record per-prompt vMLX backend evidence"
        }
        if rows.contains(where: {
            trimmedString($0["jangToolsVersion"] ?? $0["jang_tools_version"]) == nil
                || trimmedString($0["mlxVersion"] ?? $0["mlx_version"]) == nil
                || trimmedString($0["mlxLMVersion"] ?? $0["mlx_lm_version"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt vMLX package version evidence"
        }
        if rows.contains(where: {
            trimmedString($0["sourceModelPath"] ?? $0["source_model_path"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt source model path evidence"
        }
        if let sourceModelPath {
            let expectedSourcePath = normalizedPath(sourceModelPath)
            if rows.contains(where: {
                normalizedPath(trimmedString($0["sourceModelPath"] ?? $0["source_model_path"]) ?? "") != expectedSourcePath
            }) {
                return "eval.jsonl source model path does not match reviewed source"
            }
        }
        if rows.contains(where: {
            boolValue($0["maskApplied"] ?? $0["mask_applied"]) != true
        }) {
            return "eval.jsonl did not record an applied BF16/vMLX mask"
        }
        if rows.contains(where: {
            boolValue($0["maskApplied"] ?? $0["mask_applied"]) == true
                && (intValue($0["disabledExpertCount"] ?? $0["disabled_expert_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt disabled expert evidence; top-k-only comparisons cannot authorize hard pruning"
        }
        if rows.contains(where: {
            trimmedString($0["risk"]) == nil
                || trimmedString($0["regressionSeverity"] ?? $0["regression_severity"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt regression flag evidence"
        }
        if let issue = evalRowValidatorClassificationIssue(rows: rows) {
            return issue
        }
        return nil
    }

    private static func evalRowValidatorClassificationIssue(rows: [[String: Any]]) -> String? {
        for row in rows {
            guard trimmedString(row["validatorKind"] ?? row["validator_kind"]) != nil,
                  boolValue(row["validatorAvailable"] ?? row["validator_available"]) != nil,
                  trimmedString(row["promptClassification"] ?? row["prompt_classification"]) != nil,
                  boolValue(row["baselineQualified"] ?? row["baseline_qualified"]) != nil,
                  boolValue(row["safeDropEvidenceEligible"] ?? row["safe_drop_evidence_eligible"]) != nil else {
                return "eval.jsonl is missing per-prompt validator classification evidence"
            }
            let classification = promptClassification(row)
            let baselinePassed = boolValue(row["baselinePassed"] ?? row["baseline_passed"])
            let maskedPassed = boolValue(row["maskedPassed"] ?? row["masked_passed"])
            switch classification {
            case "baseline_invalid":
                if baselinePassed != false {
                    return "eval.jsonl baseline_invalid row has inconsistent baseline validator result"
                }
            case "preserved":
                if baselinePassed != true || maskedPassed != true {
                    return "eval.jsonl preserved row has inconsistent validator results"
                }
            case "degraded":
                if baselinePassed != true || maskedPassed != false {
                    return "eval.jsonl degraded row has inconsistent validator results"
                }
            case "inconclusive":
                break
            default:
                return "eval.jsonl has unknown prompt classification \(classification)"
            }
        }
        return nil
    }

    private static func evalComparisonConsistencyIssue(
        comparison: [String: Any]?,
        index: [String: Any],
        evalRows: [[String: Any]]?
    ) -> String? {
        guard let comparison else { return nil }
        let comparisonPromptCount = intValue(comparison["promptCount"] ?? comparison["prompt_count"]) ?? 0
        if let indexPromptCount = intValue(index["prompt_count"] ?? index["promptCount"]),
           comparisonPromptCount != indexPromptCount {
            return "comparison summary prompt count does not match eval_index.json"
        }

        let comparisonHighRisk = stringArrayValue(comparison["highRiskDomains"] ?? comparison["high_risk_domains"]) ?? []
        let indexHighRisk = stringArrayValue(index["high_risk_domains"] ?? index["highRiskDomains"]) ?? []
        if Set(comparisonHighRisk) != Set(indexHighRisk) {
            return "comparison summary high-risk domains do not match eval_index.json"
        }

        if let comparisonSeverity = trimmedString(comparison["regressionSeverity"] ?? comparison["regression_severity"]),
           let indexSeverity = trimmedString(index["regression_severity"] ?? index["regressionSeverity"]),
           comparisonSeverity != indexSeverity {
            return "comparison summary regression severity does not match eval_index.json"
        }

        guard let evalRows else { return nil }
        if evalRows.count != comparisonPromptCount {
            return "comparison summary covers \(comparisonPromptCount) of \(evalRows.count) eval.jsonl rows"
        }
        let rowHighRisk = evalRowsHighRiskDomains(evalRows)
        if Set(comparisonHighRisk) != Set(rowHighRisk) {
            return "comparison summary high-risk domains do not match eval.jsonl"
        }
        let rowSeverity = evalRowsRegressionSeverity(evalRows)
        if rowSeverity != "none" {
            guard let comparisonSeverity = trimmedString(comparison["regressionSeverity"] ?? comparison["regression_severity"]) else {
                return "comparison summary is missing regression severity evidence"
            }
            if comparisonSeverity != rowSeverity {
                return "comparison summary regression severity does not match eval.jsonl"
            }
        }
        let rowTextDeltas = evalRows.compactMap {
            doubleValue($0["textDelta"] ?? $0["text_delta"])
        }
        if rowTextDeltas.count == evalRows.count,
           let summaryMean = doubleValue(comparison["meanTextDelta"] ?? comparison["mean_text_delta"]),
           !doubleEqual(summaryMean, mean(rowTextDeltas)) {
            return "comparison summary mean text delta does not match eval.jsonl"
        }
        let rowLatencyDeltas = evalRows.compactMap {
            doubleValue($0["latencyDeltaPct"] ?? $0["latency_delta_pct"])
        }
        if rowLatencyDeltas.count == evalRows.count,
           let summaryMean = doubleValue(comparison["meanLatencyDeltaPct"] ?? comparison["mean_latency_delta_pct"]),
           !doubleEqual(summaryMean, mean(rowLatencyDeltas)) {
            return "comparison summary mean latency delta does not match eval.jsonl"
        }
        let baselinePassRate = passRate(evalRows.map {
            boolValue($0["baselinePassed"] ?? $0["baseline_passed"])
        })
        if let baselinePassRate,
           let summaryPassRate = doubleValue(comparison["passRateBaseline"] ?? comparison["pass_rate_baseline"]),
           !doubleEqual(summaryPassRate, baselinePassRate) {
            return "comparison summary baseline pass rate does not match eval.jsonl"
        }
        let maskedPassRate = passRate(evalRows.map {
            boolValue($0["maskedPassed"] ?? $0["masked_passed"])
        })
        if let maskedPassRate,
           let summaryPassRate = doubleValue(comparison["passRateMasked"] ?? comparison["pass_rate_masked"]),
           !doubleEqual(summaryPassRate, maskedPassRate) {
            return "comparison summary masked pass rate does not match eval.jsonl"
        }
        let rowClassifications = evalRowsPromptClassifications(evalRows)
        if let summaryCounts = intDictionaryValue(
            comparison["classificationCounts"] ?? comparison["prompt_classification_counts"]
        ),
            summaryCounts != classificationCounts(rowClassifications) {
            return "comparison summary prompt classification counts do not match eval.jsonl"
        }
        let rowBaselineQualified = evalRows.filter { boolValue($0["baselineQualified"] ?? $0["baseline_qualified"]) == true }
        if let summaryBaselineQualified = intValue(
            comparison["baselineQualifiedPromptCount"] ?? comparison["baseline_qualified_prompt_count"]
        ),
            summaryBaselineQualified != rowBaselineQualified.count {
            return "comparison summary baseline-qualified prompt count does not match eval.jsonl"
        }
        let rowDegradedIDs = evalRows.compactMap { row -> String? in
            promptClassification(row) == "degraded" ? promptID(in: row) : nil
        }
        if let summaryDegradedIDs = stringArrayValue(comparison["degradedPromptIDs"] ?? comparison["degraded_prompt_ids"]),
           summaryDegradedIDs != rowDegradedIDs {
            return "comparison summary degraded prompt IDs do not match eval.jsonl"
        }
        let rowBaselineQualifiedCoverage = evalRowsBaselineQualifiedSemanticCoverage(evalRows)
        if let summaryCoverage = stringArrayValue(
            comparison["baselineQualifiedSemanticCoverage"] ?? comparison["baseline_qualified_semantic_coverage"]
        ),
            Set(summaryCoverage) != Set(rowBaselineQualifiedCoverage) {
            return "comparison summary baseline-qualified semantic coverage does not match eval.jsonl"
        }
        let rowMissingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(rowBaselineQualifiedCoverage))
            .sorted()
        if let summaryMissingCoverage = stringArrayValue(
            comparison["missingBaselineQualifiedSemanticCoverage"]
                ?? comparison["missing_baseline_qualified_semantic_coverage"]
        ),
            Set(summaryMissingCoverage) != Set(rowMissingCoverage) {
            return "comparison summary missing baseline-qualified semantic coverage does not match eval.jsonl"
        }
        return nil
    }

    private static func evalRowsHighRiskDomains(_ rows: [[String: Any]]) -> [String] {
        Array(Set(rows.filter(rowIsBlockingRegression).flatMap { row -> [String] in
            if let domains = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]),
               !domains.isEmpty {
                return domains
            }
            return [trimmedString(row["domain"]) ?? "unknown"]
        })).sorted()
    }

    private static func evalRowsPromptClassifications(_ rows: [[String: Any]]) -> [String] {
        rows.map(promptClassification)
    }

    private static func classificationCounts(_ classifications: [String]) -> [String: Int] {
        var counts = [
            "baseline_invalid": 0,
            "preserved": 0,
            "degraded": 0,
            "inconclusive": 0
        ]
        for classification in classifications {
            counts[classification, default: 0] += 1
        }
        return counts
    }

    private static func evalRowsBaselineQualifiedSemanticCoverage(_ rows: [[String: Any]]) -> [String] {
        let domains = rows.filter {
            boolValue($0["baselineQualified"] ?? $0["baseline_qualified"]) == true
        }.flatMap { row -> [String] in
            semanticDomains(in: row)
        }
        return Array(Set(domains)).filter { $0 != "general" }.sorted()
    }

    private static func semanticDomains(in row: [String: Any]) -> [String] {
        if let semantic = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]),
           !semantic.isEmpty {
            return semantic.map(ExpertDomainTaxonomy.canonicalSemanticDomain)
        }
        if let domain = trimmedString(row["domain"]) {
            return [ExpertDomainTaxonomy.canonicalSemanticDomain(domain)]
        }
        return []
    }

    private static func rowIsBlockingRegression(_ row: [String: Any]) -> Bool {
        if promptClassification(row) == "degraded" {
            return true
        }
        if ["baseline_invalid", "inconclusive", "preserved"].contains(promptClassification(row)) {
            return false
        }
        let severity = rowRegressionSeverity(row)
        return severity == "high" || severity == "critical"
    }

    private static func evalRowsRegressionSeverity(_ rows: [[String: Any]]) -> String {
        rows.map(rowRegressionSeverity).max {
            severityRank($0) < severityRank($1)
        } ?? "none"
    }

    private static func rowRegressionSeverity(_ row: [String: Any]) -> String {
        let classification = promptClassification(row)
        if classification == "degraded" {
            return "critical"
        }
        if ["baseline_invalid", "inconclusive", "preserved"].contains(classification) {
            if let severity = trimmedString(row["regressionSeverity"] ?? row["regression_severity"]),
               severity == "watch" {
                return severity
            }
            return "none"
        }
        if let severity = trimmedString(row["regressionSeverity"] ?? row["regression_severity"]) {
            return severity
        }
        if trimmedString(row["risk"]) == "regression" {
            return "critical"
        }
        if boolValue(row["maskedPassed"] ?? row["masked_passed"]) == false {
            return "high"
        }
        if let textDelta = doubleValue(row["textDelta"] ?? row["text_delta"]) {
            if textDelta > 0.50 { return "high" }
            if textDelta > 0.20 { return "watch" }
        }
        return "none"
    }

    private static func promptClassification(_ row: [String: Any]) -> String {
        if let classification = trimmedString(row["promptClassification"] ?? row["prompt_classification"]) {
            return classification
        }
        guard let baselinePassed = boolValue(row["baselinePassed"] ?? row["baseline_passed"]),
              let maskedPassed = boolValue(row["maskedPassed"] ?? row["masked_passed"]) else {
            return "inconclusive"
        }
        if !baselinePassed { return "baseline_invalid" }
        return maskedPassed ? "preserved" : "degraded"
    }

    private static func validatorPassed(text: String, row: [String: Any]) -> Bool? {
        let kind = (trimmedString(row["validatorKind"] ?? row["validator_kind"])
            ?? trimmedString(row["expectedKind"] ?? row["expected_kind"])
            ?? "freeform")
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        let normalizedKind: String
        switch kind {
        case "normalized_exact", "equals", "match":
            normalizedKind = "exact"
        case "regexp":
            normalizedKind = "regex"
        case "unit_test_expected_regex":
            normalizedKind = "unit_test"
        case "substring":
            normalizedKind = "contains"
        default:
            normalizedKind = kind
        }
        let expected = trimmedString(row["expected"])
        switch normalizedKind {
        case "exact":
            guard let expected else { return nil }
            return text.trimmingCharacters(in: .whitespacesAndNewlines) == expected.trimmingCharacters(in: .whitespacesAndNewlines)
        case "regex", "unit_test":
            guard let expected,
                  let regex = try? NSRegularExpression(pattern: expected) else { return nil }
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            return regex.firstMatch(in: text, options: [], range: range) != nil
        case "contains":
            guard let expected else { return nil }
            return text.contains(expected)
        case "json":
            guard jsonObject(in: text) != nil else { return false }
            guard let expected else { return true }
            guard let expectedData = expected.data(using: .utf8),
                  let expectedObject = try? JSONSerialization.jsonObject(with: expectedData) else {
                return text.contains(expected)
            }
            guard let expectedDictionary = expectedObject as? [String: Any],
                  let required = expectedDictionary["required"] as? [Any] else {
                return true
            }
            guard let parsedDictionary = jsonObject(in: text) as? [String: Any] else { return false }
            return required.allSatisfy { key in
                guard let key = key as? String else { return false }
                return parsedDictionary[key] != nil
            }
        default:
            return nil
        }
    }

    private static func jsonObject(in text: String) -> Any? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidates: [String]
        if trimmed.hasPrefix("{"), trimmed.hasSuffix("}") {
            candidates = [trimmed]
        } else if let start = trimmed.firstIndex(of: "{"),
                  let end = trimmed.lastIndex(of: "}"),
                  start < end {
            candidates = [String(trimmed[start...end])]
        } else {
            candidates = []
        }
        for candidate in candidates {
            if let data = candidate.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: data) {
                return object
            }
        }
        return nil
    }

    private static func severityRank(_ severity: String) -> Int {
        switch severity {
        case "critical":
            return 3
        case "high":
            return 2
        case "watch":
            return 1
        default:
            return 0
        }
    }

    private static func passRate(_ values: [Bool?]) -> Double? {
        let scored = values.compactMap { $0 }
        guard !scored.isEmpty else { return nil }
        return Double(scored.filter { $0 }.count) / Double(scored.count)
    }

    private static func doubleEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) <= 0.000_001
    }

    private struct DecodeSettings {
        let maxTokens: Int
        let temperature: Double
        let topP: Double
        let topK: Int
    }

    private static func evalDecodeSettingsIssue(index: [String: Any], rows: [[String: Any]]) -> String? {
        guard boolValue(index["generation_settings_checked"] ?? index["generationSettingsChecked"]) == true else {
            return "eval_index.json is missing decode settings evidence"
        }
        for row in rows {
            let baselineValue = row["baselineGenerationSettings"] ?? row["baseline_generation_settings"]
            let maskedValue = row["maskedGenerationSettings"] ?? row["masked_generation_settings"]
            guard baselineValue != nil, maskedValue != nil else {
                return "eval.jsonl is missing baseline/masked decode settings evidence"
            }
            guard let baseline = decodeSettings(baselineValue),
                  let masked = decodeSettings(maskedValue) else {
                return "eval.jsonl has unreadable baseline/masked decode settings evidence"
            }
            if baseline.maxTokens != masked.maxTokens
                || abs(baseline.temperature - masked.temperature) > 0.000_001
                || abs(baseline.topP - masked.topP) > 0.000_001
                || baseline.topK != masked.topK {
                return "eval.jsonl baseline/masked decode settings do not match"
            }
        }
        return nil
    }

    private static func decodeSettings(_ value: Any?) -> DecodeSettings? {
        guard let object = value as? [String: Any],
              let maxTokens = intValue(object["max_tokens"] ?? object["maxTokens"]),
              maxTokens > 0,
              let temperature = doubleValue(object["temperature"]),
              temperature.isFinite,
              let topP = doubleValue(object["top_p"] ?? object["topP"]),
              topP.isFinite,
              let topK = intValue(object["top_k"] ?? object["topK"]),
              topK >= 0 else {
            return nil
        }
        return DecodeSettings(maxTokens: maxTokens, temperature: temperature, topP: topP, topK: topK)
    }

    private static func evalIndexLayerStatsCoverageIssue(index: [String: Any], promptCount: Int) -> String? {
        let baselineCount = intValue(index["baseline_layer_stats_prompt_count"] ?? index["baselineLayerStatsPromptCount"])
        let maskedCount = intValue(index["masked_layer_stats_prompt_count"] ?? index["maskedLayerStatsPromptCount"])
        guard baselineCount != nil || maskedCount != nil else { return nil }
        guard baselineCount == promptCount, maskedCount == promptCount else {
            return "eval_index.json layer-stat coverage is incomplete for indexed prompts"
        }
        return nil
    }

    private static func evalRowLayerStatsCoverageIssue(rows: [[String: Any]]) -> String? {
        let baselineRows = rows.filter { nonEmptyJSONArray($0["baselineLayerStats"] ?? $0["baseline_layer_stats"]) }.count
        let maskedRows = rows.filter { nonEmptyJSONArray($0["maskedLayerStats"] ?? $0["masked_layer_stats"]) }.count
        guard baselineRows > 0 || maskedRows > 0 else { return nil }
        guard baselineRows == rows.count, maskedRows == rows.count else {
            return "eval.jsonl layer-stat evidence is incomplete for baseline/masked prompts"
        }
        return nil
    }

    private static func evalTraceVariantIssue(
        rows: [[String: Any]],
        expectedPromptIDs: [String],
        disabledExpertCount: Int?,
        topKOverride: Int?,
        expectedDisabledByLayer: [Int: Set<Int>]? = nil,
        expectedBaselineRouteRecordCount: Int? = nil,
        expectedMaskedRouteRecordCount: Int? = nil
    ) -> String? {
        let expected = Set(expectedPromptIDs)
        let expectedMaskLayers = Set((expectedDisabledByLayer ?? [:]).filter { !$0.value.isEmpty }.keys)
        var baselinePromptIDs = Set<String>()
        var maskedPromptIDs = Set<String>()
        var maskedPromptIDsWithMaskEvidence = Set<String>()
        var maskedPromptExpectedMaskLayers: [String: Set<Int>] = [:]
        var baselineTraceCount = 0
        var maskedTraceCount = 0

        for row in rows {
            guard let id = promptID(in: row) else {
                return "eval_trace.jsonl prompt IDs are unreadable"
            }
            let variant = trimmedString(row["variant"])?.lowercased()
            switch variant {
            case "baseline":
                baselineTraceCount += 1
                baselinePromptIDs.insert(id)
            case "masked":
                maskedTraceCount += 1
                maskedPromptIDs.insert(id)
                if let issue = traceRowDisabledSelectionIssue(row, promptID: id) {
                    return issue
                }
                if let layer = traceRowExpectedMaskEvidenceLayer(
                    row,
                    expectedDisabledByLayer: expectedDisabledByLayer
                ) {
                    maskedPromptExpectedMaskLayers[id, default: []].insert(layer)
                }
                if traceRowHasMaskEvidence(
                    row,
                    disabledExpertCount: disabledExpertCount,
                    topKOverride: topKOverride
                ) {
                    maskedPromptIDsWithMaskEvidence.insert(id)
                }
            default:
                continue
            }
        }

        let missingBaseline = expected.subtracting(baselinePromptIDs)
        if !missingBaseline.isEmpty {
            return "eval_trace.jsonl missing baseline routing records for prompt IDs: \(previewIDs(missingBaseline))"
        }
        let missingMasked = expected.subtracting(maskedPromptIDs)
        if !missingMasked.isEmpty {
            return "eval_trace.jsonl missing masked routing records for prompt IDs: \(previewIDs(missingMasked))"
        }
        if let expectedBaselineRouteRecordCount,
           baselineTraceCount != expectedBaselineRouteRecordCount {
            return "eval_trace.jsonl has \(baselineTraceCount) baseline routing records for \(expectedBaselineRouteRecordCount) indexed baseline route records"
        }
        if let expectedMaskedRouteRecordCount,
           maskedTraceCount != expectedMaskedRouteRecordCount {
            return "eval_trace.jsonl has \(maskedTraceCount) masked routing records for \(expectedMaskedRouteRecordCount) indexed masked route records"
        }
        if (disabledExpertCount ?? 0) > 0 || topKOverride != nil {
            let missingMaskEvidence = expected.subtracting(maskedPromptIDsWithMaskEvidence)
            if !missingMaskEvidence.isEmpty {
                return "eval_trace.jsonl masked routing records are missing mask evidence for prompt IDs: \(previewIDs(missingMaskEvidence))"
            }
        }
        if !expectedMaskLayers.isEmpty {
            for id in expectedPromptIDs {
                let missingLayers = expectedMaskLayers.subtracting(maskedPromptExpectedMaskLayers[id] ?? [])
                if !missingLayers.isEmpty {
                    return "eval_trace.jsonl masked routing records are missing mask.json evidence for prompt \(id) layers: \(previewInts(missingLayers))"
                }
            }
        }
        return nil
    }

    private static func traceRowExpectedMaskEvidenceLayer(
        _ row: [String: Any],
        expectedDisabledByLayer: [Int: Set<Int>]?
    ) -> Int? {
        guard let expectedDisabledByLayer,
              let record = row["record"] as? [String: Any],
              let layer = intValue(record["layer"] ?? record["layerIndex"] ?? record["layer_index"]),
              let expectedDisabled = expectedDisabledByLayer[layer],
              !expectedDisabled.isEmpty else {
            return nil
        }
        let disabled = Set(arrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap { intValue($0) })
        return expectedDisabled.isSubset(of: disabled) ? layer : nil
    }

    private static func traceRowDisabledSelectionIssue(_ row: [String: Any], promptID: String) -> String? {
        guard let record = row["record"] as? [String: Any] else { return nil }
        let disabled = Set(arrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap { intValue($0) })
        guard !disabled.isEmpty else { return nil }
        let selected = Set(arrayValue(record["selectedExperts"] ?? record["selected_experts"]).compactMap { intValue($0) })
        let leaked = selected.intersection(disabled)
        guard !leaked.isEmpty else { return nil }
        let sorted = leaked.sorted()
        let head = sorted.prefix(5).map { String($0) }.joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        let preview = remaining == 0 ? head : "\(head), +\(remaining) more"
        return "eval_trace.jsonl masked routing records selected disabled experts for prompt \(promptID): \(preview)"
    }

    private static func traceRowHasMaskEvidence(
        _ row: [String: Any],
        disabledExpertCount: Int?,
        topKOverride: Int?
    ) -> Bool {
        guard let record = row["record"] as? [String: Any] else { return false }
        if (disabledExpertCount ?? 0) > 0 {
            let disabledExperts = arrayValue(record["disabledExperts"] ?? record["disabled_experts"])
            if !disabledExperts.isEmpty { return true }
            if (intValue(record["disabledExpertCount"] ?? record["disabled_expert_count"]) ?? 0) > 0 {
                return true
            }
            return false
        }
        if topKOverride != nil {
            return intValue(record["effectiveTopK"] ?? record["effective_top_k"] ?? record["topK"] ?? record["top_k"]) != nil
        }
        return true
    }

    private static func disabledExpertsByLayer(fromMaskURL url: URL) -> [Int: Set<Int>]? {
        guard let mask = readJSONObject(url) else { return nil }
        for key in ["disabled_by_layer", "layers", "disabledExpertsByLayer"] {
            if let map = intSetMap(mask[key]) {
                return map.filter { !$0.value.isEmpty }
            }
        }
        return [:]
    }

    private static func intSetMap(_ value: Any?) -> [Int: Set<Int>]? {
        guard let dictionary = value as? [String: Any] else { return nil }
        var result: [Int: Set<Int>] = [:]
        for (key, rawExperts) in dictionary {
            guard let layer = intValue(key) else { return nil }
            result[layer] = Set(arrayValue(rawExperts).compactMap { intValue($0) })
        }
        return result
    }

    private static func prunedSourceSuiteVerificationIssue(
        summary: [String: Any],
        prunedSource: URL,
        tracedPromptCount: Int?,
        suiteURL: URL?,
        expectedLayerCount: Int? = nil
    ) -> String? {
        guard boolValue(summary["pruned_suite_verification_ready"]) == true else {
            if let issue = stringValue(summary["pruned_suite_verification_issue"]),
               !issue.isEmpty {
                return issue
            }
            return "missing pruned-source same-suite vMLX generation evidence"
        }
        guard let summaryURL = embeddedSidecarURL(
            summary["pruned_suite_summary"],
            prunedSource: prunedSource,
            fallbackName: "expert_lab_pruned_generation_summary.json"
        ),
            let generationURL = embeddedSidecarURL(
                summary["pruned_suite_generations"],
                prunedSource: prunedSource,
                fallbackName: "expert_lab_pruned_generations.jsonl"
            ) else {
            return "pruned-source generation sidecar paths must be embedded in the pruned BF16/F16 source"
        }
        guard FileManager.default.isReadableFile(atPath: summaryURL.path),
              FileManager.default.isReadableFile(atPath: generationURL.path) else {
            return "missing pruned-source generation sidecar paths"
        }
        guard let prunedSummary = readJSONObject(summaryURL) else {
            return "pruned-source generation summary is unreadable"
        }
        guard boolValue(prunedSummary["ready"]) == true else {
            if let issue = stringValue(prunedSummary["issue"]), !issue.isEmpty {
                return issue
            }
            return "pruned-source generation summary did not pass"
        }
        let promptCount = intValue(prunedSummary["prompt_count"]) ?? 0
        if let tracedPromptCount, tracedPromptCount > 0, promptCount != tracedPromptCount {
            return "pruned-source generation covers \(promptCount) of \(tracedPromptCount) reviewed prompts"
        }
        if let suiteURL {
            let suiteCount = lineCount(suiteURL)
            if suiteCount != promptCount {
                return "pruned-source generation covers \(promptCount) of \(suiteCount) suite prompts"
            }
            guard let expectedSuiteSHA256 = fileSHA256(suiteURL) else {
                return "pruned-source reviewed prompt suite fingerprint could not be computed"
            }
            guard let recordedSuiteSHA256 = stringValue(prunedSummary["suite_sha256"] ?? prunedSummary["suiteSHA256"]),
                  !recordedSuiteSHA256.isEmpty else {
                return "pruned-source reviewed prompt suite fingerprint was not recorded"
            }
            if recordedSuiteSHA256 != expectedSuiteSHA256 {
                return "pruned-source reviewed prompt suite fingerprint does not match reviewed suite"
            }
        }
        if let recordedGenerationCount = intValue(prunedSummary["generation_count"]),
           recordedGenerationCount != promptCount {
            return "pruned-source generation summary records \(recordedGenerationCount) rows for \(promptCount) prompts"
        }
        let generationCount = lineCount(generationURL)
        if generationCount != promptCount {
            return "pruned-source generation JSONL has \(generationCount) rows for \(promptCount) prompts"
        }
        if let suiteURL,
           let issue = sameSuitePromptIDIssue(suiteURL: suiteURL, generationURL: generationURL) {
            return issue
        }
        guard let generationRows = jsonlObjects(generationURL) else {
            return "pruned-source generation JSONL is unreadable"
        }
        if let issue = prunedGenerationSettingsIssue(
            rows: generationRows,
            suiteURL: suiteURL,
            generationDefaults: prunedSummary["generation_defaults"] as? [String: Any]
        ) {
            return issue
        }
        guard let prunedSourcePath = stringValue(prunedSummary["pruned_source"] ?? summary["pruned_source"]) else {
            return "pruned-source generation summary is missing pruned source path evidence"
        }
        if canonicalPath(prunedSourcePath) != canonicalPath(prunedSource.path) {
            return "pruned-source generation summary path does not match the selected pruned BF16/F16 source"
        }
        if let issue = prunedGenerationRowEvidenceIssue(
            rows: generationRows,
            prunedSourcePath: prunedSourcePath,
            expectedLayerCount: expectedLayerCount
        ) {
            return issue
        }
        let requiredComparisonCount = max(promptCount, tracedPromptCount ?? 0, suiteURL.map(lineCount) ?? 0)
        guard let comparedCount = intValue(
            prunedSummary["reviewed_ab_comparison_count"] ?? prunedSummary["reviewed_masked_comparison_count"]
        ),
              comparedCount >= requiredComparisonCount else {
            let compared = intValue(
                prunedSummary["reviewed_ab_comparison_count"] ?? prunedSummary["reviewed_masked_comparison_count"]
            ) ?? 0
            return "pruned-source reviewed-output comparison covers \(compared) of \(requiredComparisonCount) prompts"
        }
        guard boolValue(prunedSummary["pruned_validator_outcomes_checked"]) == true else {
            return "pruned-source generation is missing reviewed validator outcome evidence"
        }
        guard let baselineQualified = intValue(prunedSummary["baseline_qualified_prompt_count"]),
              baselineQualified > 0 else {
            return "pruned-source generation has no baseline-qualified validator prompts"
        }
        guard let classificationCounts = intDictionaryValue(prunedSummary["pruned_classification_counts"]),
              classificationCounts.values.reduce(0, +) >= baselineQualified else {
            return "pruned-source generation is missing pruned prompt classification evidence"
        }
        let missingQualifiedCoverage = stringArrayValue(
            prunedSummary["missing_baseline_qualified_semantic_coverage"]
                ?? prunedSummary["missingBaselineQualifiedSemanticCoverage"]
        ) ?? []
        if !missingQualifiedCoverage.isEmpty {
            return "pruned-source generation baseline-qualified coverage is missing: \(missingQualifiedCoverage.sorted().joined(separator: ", "))"
        }
        let degraded = stringArrayValue(
            prunedSummary["pruned_degraded_prompt_ids"] ?? prunedSummary["degradedPromptIDs"]
        ) ?? []
        if !degraded.isEmpty {
            return "pruned-source generation degraded baseline-qualified prompts: \(previewIDs(Set(degraded)))"
        }
        if let passRate = doubleValue(prunedSummary["pruned_baseline_qualified_pass_rate"]),
           passRate < 1.0 {
            return "pruned-source validator pass rate is below 100% on baseline-qualified prompts"
        }
        guard let runtimeMode = stringValue(prunedSummary["runtime_mode"]),
              !runtimeMode.isEmpty,
              let runtimeDevice = stringValue(prunedSummary["runtime_device"]),
              !runtimeDevice.isEmpty,
              boolValue(prunedSummary["runtime_metal_enabled"]) == true else {
            return "pruned-source generation is missing vMLX Metal runtime evidence"
        }
        if runtimeMode != "bf16_vmlx" {
            return "pruned-source generation did not record BF16/vMLX runtime evidence"
        }
        if stringValue(prunedSummary["runtime_backend"] ?? prunedSummary["runtimeBackend"]) != "vmlx" {
            return "pruned-source generation did not record vMLX backend evidence"
        }
        if boolValue(prunedSummary["hook_coverage_complete"] ?? prunedSummary["hookCoverageComplete"]) == false {
            return "pruned-source generation recorded incomplete vMLX routed-layer hook coverage"
        }
        if let expectedLayerCount {
            guard let hookedLayers = intValue(prunedSummary["hooked_moe_layers"] ?? prunedSummary["hookedMOELayers"]) else {
                return "pruned-source generation is missing vMLX routed-layer hook evidence"
            }
            if hookedLayers < expectedLayerCount {
                return "pruned-source generation vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers"
            }
        }
        if let expectedMOELayers = intValue(prunedSummary["expected_moe_layers"] ?? prunedSummary["expectedMOELayers"]),
           let hookedLayers = intValue(prunedSummary["hooked_moe_layers"] ?? prunedSummary["hookedMOELayers"]),
           hookedLayers < expectedMOELayers {
            return "pruned-source generation vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers"
        }
        guard let jangToolsVersion = stringValue(prunedSummary["jang_tools_version"] ?? prunedSummary["jangToolsVersion"]),
              !jangToolsVersion.isEmpty,
              let mlxVersion = stringValue(prunedSummary["mlx_version"] ?? prunedSummary["mlxVersion"]),
              !mlxVersion.isEmpty,
              let mlxLMVersion = stringValue(prunedSummary["mlx_lm_version"] ?? prunedSummary["mlxLMVersion"]),
              !mlxLMVersion.isEmpty else {
            return "pruned-source generation is missing vMLX package version evidence"
        }
        guard let runtimeSourcePath = stringValue(prunedSummary["runtime_source_model_path"]) else {
            return "pruned-source generation is missing runtime source path evidence"
        }
        if canonicalPath(runtimeSourcePath) != canonicalPath(prunedSource.path) {
            return "pruned-source generation source path does not match the pruned BF16/F16 source"
        }
        return nil
    }

    private static func prunedGenerationRowEvidenceIssue(
        rows: [[String: Any]],
        prunedSourcePath: String?,
        expectedLayerCount: Int? = nil
    ) -> String? {
        for row in rows {
            guard let result = row["result"] as? [String: Any] else {
                return "pruned-source generation row is missing result"
            }
            let text = trimmedString(result["text"]) ?? ""
            let tokens = intValue(result["tokens"]) ?? 0
            if text.isEmpty || tokens <= 0 {
                return "pruned-source generation produced an empty prompt output"
            }
            guard let runtime = result["runtime_info"] as? [String: Any] else {
                return "pruned-source generation is missing per-prompt runtime evidence"
            }
            guard let runtimeMode = trimmedString(runtime["runtime_mode"] ?? runtime["runtimeMode"]),
                  !runtimeMode.isEmpty,
                  let runtimeDevice = trimmedString(runtime["device_name"] ?? runtime["runtime_device"] ?? runtime["runtimeDevice"]),
                  !runtimeDevice.isEmpty,
                  boolValue(runtime["runtime_metal_enabled"] ?? runtime["runtimeMetalEnabled"]) == true else {
                return "pruned-source generation is missing per-prompt vMLX Metal runtime evidence"
            }
            if runtimeMode != "bf16_vmlx" {
                return "pruned-source generation did not record per-prompt BF16/vMLX runtime evidence"
            }
            if trimmedString(runtime["backend"] ?? runtime["runtime_backend"] ?? runtime["runtimeBackend"]) != "vmlx" {
                return "pruned-source generation did not record per-prompt vMLX backend evidence"
            }
            if trimmedString(runtime["jang_tools_version"] ?? runtime["jangToolsVersion"]) == nil
                || trimmedString(runtime["mlx_version"] ?? runtime["mlxVersion"]) == nil
                || trimmedString(runtime["mlx_lm_version"] ?? runtime["mlxLMVersion"]) == nil {
                return "pruned-source generation is missing per-prompt vMLX package version evidence"
            }
            guard let runtimeSourcePath = trimmedString(runtime["source_model_path"] ?? runtime["sourceModelPath"]) else {
                return "pruned-source generation is missing per-prompt source path evidence"
            }
            if let prunedSourcePath,
               normalizedPath(runtimeSourcePath) != normalizedPath(prunedSourcePath) {
                return "pruned-source generation per-prompt source path does not match the pruned BF16/F16 source"
            }
            if let issue = prunedGenerationLayerStatsIssue(
                result: result,
                expectedLayerCount: expectedLayerCount
            ) {
                return issue
            }
            if let issue = prunedGenerationTokenTraceIssue(
                result: result,
                expectedLayerCount: expectedLayerCount
            ) {
                return issue
            }
        }
        return nil
    }

    private static func prunedGenerationLayerStatsIssue(
        result: [String: Any],
        expectedLayerCount: Int?
    ) -> String? {
        guard expectedLayerCount != nil else { return nil }
        guard let rows = result["layer_stats"] as? [[String: Any]], !rows.isEmpty else {
            return "pruned-source generation is missing per-prompt routed-layer stats"
        }
        let layerIDs = rows.compactMap { intValue($0["layer"] ?? $0["layer_id"] ?? $0["layerID"]) }
        if layerIDs.count != rows.count {
            return "pruned-source generation routed-layer stats have unreadable layer IDs"
        }
        if Set(layerIDs).count < layerIDs.count {
            return "pruned-source generation routed-layer stats contain duplicate layers"
        }
        if let expectedLayerCount {
            if rows.count < expectedLayerCount {
                return "pruned-source generation routed-layer stats cover \(rows.count) of \(expectedLayerCount) layers"
            }
        }
        if rows.contains(where: { (intValue($0["token_count"] ?? $0["tokenCount"]) ?? 0) <= 0 }) {
            return "pruned-source generation routed-layer stats are missing token-position depth"
        }
        if rows.contains(where: { layerStatMap($0["hit_counts"] ?? $0["hitCounts"]).isEmpty }) {
            return "pruned-source generation routed-layer stats are missing expert hit counts"
        }
        if rows.contains(where: { layerStatMap($0["probability_mass"] ?? $0["probabilityMass"]).isEmpty }) {
            return "pruned-source generation routed-layer stats are missing expert gate-mass evidence"
        }
        return nil
    }

    private static func prunedGenerationTokenTraceIssue(
        result: [String: Any],
        expectedLayerCount: Int?
    ) -> String? {
        guard expectedLayerCount != nil else { return nil }
        guard let layerStats = result["layer_stats"] as? [[String: Any]], !layerStats.isEmpty else {
            return nil
        }
        let expectedRoutes = layerStats.reduce(0) {
            $0 + (intValue($1["token_count"] ?? $1["tokenCount"]) ?? 0)
        }
        guard expectedRoutes > 0 else {
            return "pruned-source generation is missing per-prompt routed layer-token records"
        }
        guard let trace = result["token_trace"] as? [[String: Any]], !trace.isEmpty else {
            return "pruned-source generation is missing per-prompt token_trace routing evidence"
        }
        if trace.count != expectedRoutes {
            return "pruned-source generation token_trace has \(trace.count) rows for \(expectedRoutes) routed layer-token records"
        }
        for row in trace {
            guard intValue(row["layer"]) != nil,
                  intValue(row["token_index"] ?? row["tokenIndex"]) != nil else {
                return "pruned-source generation token_trace is missing layer/token evidence"
            }
            if arrayValue(row["selected_experts"] ?? row["selectedExperts"]).isEmpty {
                return "pruned-source generation token_trace is missing selected expert evidence"
            }
        }
        return nil
    }

    private static func layerStatMap(_ value: Any?) -> [String: Any] {
        if let value = value as? [String: Any] { return value }
        return [:]
    }

    private static func sameSuitePromptIDIssue(suiteURL: URL, generationURL: URL) -> String? {
        guard let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]) else {
            return "suite.jsonl prompt IDs are unreadable"
        }
        guard let generationIDs = jsonlPromptIDs(generationURL) else {
            return "pruned-source generation prompt IDs are unreadable"
        }
        let suiteSet = Set(suiteIDs)
        if suiteSet.count != suiteIDs.count {
            return "suite.jsonl contains duplicate prompt IDs"
        }
        let generationSet = Set(generationIDs)
        if generationSet.count != generationIDs.count {
            return "pruned-source generation JSONL contains duplicate prompt IDs"
        }
        let missing = suiteSet.subtracting(generationSet)
        if !missing.isEmpty {
            return "pruned-source generation missing suite prompt IDs: \(previewIDs(missing))"
        }
        let unexpected = generationSet.subtracting(suiteSet)
        if !unexpected.isEmpty {
            return "pruned-source generation has prompt IDs outside reviewed suite: \(previewIDs(unexpected))"
        }
        if generationIDs != suiteIDs {
            return "pruned-source generation prompt order does not match reviewed suite"
        }
        return nil
    }

    private static func prunedGenerationSettingsIssue(
        rows: [[String: Any]],
        suiteURL: URL?,
        generationDefaults: [String: Any]?
    ) -> String? {
        guard let generationDefaults else { return nil }
        let defaultMaxTokens = intValue(generationDefaults["max_tokens"] ?? generationDefaults["maxTokens"])
        let defaultTemperature = doubleValue(generationDefaults["temperature"])
        guard defaultMaxTokens != nil || defaultTemperature != nil else {
            return "pruned-source generation defaults are missing decode settings"
        }
        let suiteSettings = suiteURL.flatMap(suiteGenerationSettings)
        for row in rows {
            guard let prompt = row["prompt"] as? [String: Any],
                  let promptID = promptID(in: prompt) else {
                return "pruned-source generation row is missing prompt settings evidence"
            }
            guard let result = row["result"] as? [String: Any],
                  let settings = result["generation_settings"] as? [String: Any] else {
                return "pruned-source generation row is missing decode settings evidence"
            }
            guard let recordedMaxTokens = intValue(settings["max_tokens"] ?? settings["maxTokens"]),
                  recordedMaxTokens > 0,
                  let recordedTemperature = doubleValue(settings["temperature"]) else {
                return "pruned-source generation row has unreadable decode settings"
            }
            let expected = suiteSettings?[promptID]
            if let expectedMaxTokens = expected?.maxTokens ?? defaultMaxTokens,
               recordedMaxTokens != expectedMaxTokens {
                return "pruned-source generation max_tokens for \(promptID) does not match reviewed suite"
            }
            if let expectedTemperature = expected?.temperature ?? defaultTemperature,
               abs(recordedTemperature - expectedTemperature) > 0.000_001 {
                return "pruned-source generation temperature for \(promptID) does not match reviewed suite"
            }
        }
        return nil
    }

    private static func suiteGenerationSettings(_ url: URL) -> [String: (maxTokens: Int?, temperature: Double?)]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var settings: [String: (maxTokens: Int?, temperature: Double?)] = [:]
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let promptID = promptID(in: object) else {
                return nil
            }
            settings[promptID] = (
                intValue(object["max_new_tokens"] ?? object["maxTokens"]),
                doubleValue(object["temperature"])
            )
        }
        return settings
    }

    private static func stringArrayValue(_ value: Any?) -> [String]? {
        switch value {
        case let values as [String]:
            return values
        case let values as [Any]:
            return values.compactMap { $0 as? String }
        default:
            return nil
        }
    }

    private static func arrayValue(_ value: Any?) -> [Any] {
        switch value {
        case let values as [Any]:
            return values
        default:
            return []
        }
    }

    private static func urlValue(_ value: Any?) -> URL? {
        guard let path = value as? String, !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    private static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }

    private static func jsonlStringIDs(_ url: URL, keys: [String]) -> [String]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var ids: [String] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            if let id = keys.lazy.compactMap({ object[$0] as? String }).first {
                ids.append(id)
            }
        }
        return ids
    }

    private static func jsonlPromptIDs(_ url: URL) -> [String]? {
        guard let rows = jsonlObjects(url) else { return nil }
        var ids: [String] = []
        for row in rows {
            if let id = promptID(in: row) {
                ids.append(id)
                continue
            }
            if let prompt = row["prompt"] as? [String: Any],
               let id = promptID(in: prompt) {
                ids.append(id)
                continue
            }
            return nil
        }
        return ids
    }

    private static func jsonlObjects(_ url: URL) -> [[String: Any]]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var objects: [[String: Any]] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            objects.append(object)
        }
        return objects
    }

    private static func trimmedString(_ value: Any?) -> String? {
        stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func nonEmptyString(_ value: String?) -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty else {
            return nil
        }
        return value
    }

    private static func nonEmptyJSONArray(_ value: Any?) -> Bool {
        guard let array = value as? [Any] else { return false }
        return !array.isEmpty
    }

    private static func dictionaryValue(_ value: Any?) -> [String: Any]? {
        value as? [String: Any]
    }

    private static func intDictionaryValue(_ value: Any?) -> [String: Int]? {
        guard let dictionary = dictionaryValue(value) else { return nil }
        var result: [String: Int] = [:]
        for (key, raw) in dictionary {
            guard let value = intValue(raw) else { return nil }
            result[key] = value
        }
        return result
    }

    private static func promptID(in row: [String: Any]) -> String? {
        ["promptID", "prompt_id", "id"].lazy.compactMap { key in
            trimmedString(row[key])
        }.first { !$0.isEmpty }
    }

    private static func suiteSemanticCoverageIssue(_ suiteURL: URL) -> String? {
        guard let suite = try? ExpertPromptSuite.loadJSONL(
            name: suiteURL.deletingPathExtension().lastPathComponent,
            from: suiteURL
        ) else {
            return "suite.jsonl semantic prompt coverage is unreadable"
        }
        let semanticDomains = Set(suite.prompts.flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
        let missing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.subtracting(semanticDomains).sorted()
        return missing.isEmpty
            ? nil
            : "suite.jsonl is missing required semantic prompt probes: \(missing.joined(separator: ", "))"
    }

    private static func evalIndexSemanticCoverageIssue(_ index: [String: Any]) -> String? {
        guard let semanticCoverage = stringArrayValue(index["semantic_coverage"] ?? index["semanticCoverage"]),
              !semanticCoverage.isEmpty else {
            return "eval_index.json is missing semantic coverage evidence"
        }
        let coverage = Set(
            semanticCoverage
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let missingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(coverage)
            .sorted()
        if !missingCoverage.isEmpty {
            return "eval_index.json semantic coverage is missing required probes: \(missingCoverage.joined(separator: ", "))"
        }
        guard let recordedMissing = stringArrayValue(index["missing_semantic_coverage"] ?? index["missingSemanticCoverage"]) else {
            return "eval_index.json is missing missing-semantic-coverage evidence"
        }
        let missing = Set(
            recordedMissing
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        if !missing.isEmpty {
            return "eval_index.json records missing semantic prompt probes: \(missing.sorted().joined(separator: ", "))"
        }
        return nil
    }

    private static func evalIndexValidatorEvidenceIssue(_ index: [String: Any]) -> String? {
        guard nonEmptyString(stringValue(index["validator_schema"] ?? index["validatorSchema"])) != nil,
              intValue(index["validator_available_prompt_count"] ?? index["validatorAvailablePromptCount"]) != nil,
              dictionaryValue(index["prompt_classification_counts"] ?? index["promptClassificationCounts"]) != nil else {
            return "eval_index.json is missing validator classification evidence"
        }
        guard let promptCount = intValue(index["prompt_count"] ?? index["promptCount"]) else {
            return "eval_index.json is missing prompt count evidence"
        }
        guard let baselineQualified = intValue(
            index["baseline_qualified_prompt_count"] ?? index["baselineQualifiedPromptCount"]
        ),
              baselineQualified > 0 else {
            return "eval_index.json has no baseline-qualified validator prompts"
        }
        guard let baselineQualifiedIDs = stringArrayValue(
            index["baseline_qualified_prompt_ids"] ?? index["baselineQualifiedPromptIDs"]
        ),
              let baselineInvalidIDs = stringArrayValue(
                index["baseline_invalid_prompt_ids"] ?? index["baselineInvalidPromptIDs"]
              ),
              let inconclusiveIDs = stringArrayValue(
                index["inconclusive_prompt_ids"] ?? index["inconclusivePromptIDs"]
              ),
              let preservedIDs = stringArrayValue(
                index["preserved_prompt_ids"] ?? index["preservedPromptIDs"]
              ),
              let degradedIDs = stringArrayValue(
                index["degraded_prompt_ids"] ?? index["degradedPromptIDs"]
              ) else {
            return "eval_index.json is missing prompt classification ID lists"
        }
        if baselineQualifiedIDs.count != baselineQualified {
            return "eval_index.json baseline-qualified prompt IDs do not match the baseline-qualified count"
        }
        let classified = baselineInvalidIDs.count + inconclusiveIDs.count + preservedIDs.count + degradedIDs.count
        if classified != promptCount {
            return "eval_index.json prompt classifications cover \(classified) of \(promptCount) prompts"
        }
        if !degradedIDs.isEmpty {
            return "eval_index.json has baseline-qualified prompt regressions: \(previewIDs(Set(degradedIDs)))"
        }
        let missingCoverage = stringArrayValue(
            index["missing_baseline_qualified_semantic_coverage"]
                ?? index["missingBaselineQualifiedSemanticCoverage"]
        ) ?? []
        if !missingCoverage.isEmpty {
            return "eval_index.json baseline-qualified semantic coverage is missing: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        guard let coverage = stringArrayValue(
            index["baseline_qualified_semantic_coverage"]
                ?? index["baselineQualifiedSemanticCoverage"]
        ),
              !coverage.isEmpty else {
            return "eval_index.json is missing baseline-qualified semantic coverage evidence"
        }
        if let passRate = doubleValue(
            index["baseline_qualified_masked_pass_rate"] ?? index["baselineQualifiedMaskedPassRate"]
        ),
            passRate < 1.0 {
            return "eval_index.json masked validator pass rate is below 100% on baseline-qualified prompts"
        }
        return nil
    }

    private static func reviewedSuiteSemanticCoverage(_ suiteURL: URL?) -> [String: Any] {
        guard let suiteURL else {
            return [
                "ready": false,
                "domains": [],
                "missing": ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted(),
                "prompt_count": 0,
                "issue": "reviewed prompt suite path missing"
            ]
        }
        guard let suite = try? ExpertPromptSuite.loadJSONL(
            name: suiteURL.deletingPathExtension().lastPathComponent,
            from: suiteURL
        ) else {
            return [
                "ready": false,
                "domains": [],
                "missing": ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.sorted(),
                "prompt_count": 0,
                "issue": "reviewed prompt suite semantic coverage unreadable"
            ]
        }
        let domains = Set(suite.prompts.flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
        let missing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.subtracting(domains).sorted()
        var payload: [String: Any] = [
            "ready": missing.isEmpty,
            "domains": domains.sorted(),
            "missing": missing,
            "prompt_count": suite.prompts.count
        ]
        if !missing.isEmpty {
            payload["issue"] = "missing required semantic prompt probes: \(missing.joined(separator: ", "))"
        }
        return payload
    }

    private static func previewIDs(_ ids: Set<String>) -> String {
        let sorted = ids.sorted()
        let head = sorted.prefix(5).joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func previewInts(_ values: Set<Int>) -> String {
        let sorted = values.sorted()
        let head = sorted.prefix(5).map { String($0) }.joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func readVerification(at url: URL) -> (ok: Bool, hint: String) {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (false, "missing or unreadable verification.json in pruned BF16/F16 source")
        }
        guard jsonBool(json["ok"]) == true else {
            let errors = (json["errors"] as? [String])?.joined(separator: "; ")
            return (false, errors?.isEmpty == false ? errors! : "Pruned source verification did not pass")
        }
        let checks = (json["checks"] as? [String: Any])?.compactMapValues(jsonBool) ?? [:]
        let missing = requiredPrunedSourceVerificationChecks.filter { checks[$0] == nil }
        let failed = checks
            .filter { !$0.value }
            .map(\.key)
            .sorted()
        if !missing.isEmpty {
            return (false, "Missing required verification checks: \(missing.joined(separator: ", "))")
        }
        if !failed.isEmpty {
            return (false, "Failed verification checks: \(failed.joined(separator: ", "))")
        }
        return (true, url.path)
    }

    private static func jsonBool(_ value: Any?) -> Bool? {
        if let value = value as? Bool {
            return value
        }
        if let value = value as? NSNumber {
            return value.boolValue
        }
        return nil
    }

    private static func tokenizerConfigDefinesSpecialTokens(_ tokCfg: [String: Any]) -> Bool {
        let scalarKeys = ["bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"]
        if scalarKeys.contains(where: { key in
            guard let value = tokCfg[key] as? String else { return false }
            return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) {
            return true
        }
        if let additional = tokCfg["additional_special_tokens"] as? [Any] {
            return additional.contains { value in
                guard let string = value as? String else { return false }
                return !string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
        }
        return false
    }

    private static func writeExpertLabFinalReport(
        plan: ConversionPlan,
        outputDir: URL,
        pruneVerificationOK: Bool,
        reviewedPlanURL: URL
    ) -> VerifyCheck {
        let url = outputDir.appendingPathComponent("expert_lab_final_report.md")
        let generated = ISO8601DateFormatter().string(from: Date())
        let originalSource = plan.expertReviewOriginalSourceURL?.path ?? "not recorded"
        let prunedSource = plan.expertReviewPrunedSourceURL?.path ?? "not recorded"
        let pruneReport = plan.expertReviewPruneReportURL?.path ?? prunedSource + "/expert_lab_prune_report.md"
        let output = plan.outputURL?.path ?? outputDir.path
        let reviewSummary = readReviewSummary(reviewedPlanURL: reviewedPlanURL)
        let sameSuiteReady = boolValue(reviewSummary["same_suite_verification_ready"]) ?? false
        let prunedSuiteReady = boolValue(reviewSummary["pruned_suite_verification_ready"]) ?? false
        let prunedSuiteSummary = stringValue(reviewSummary["pruned_suite_summary"]) ?? "not recorded"
        let prunedSuiteGenerations = stringValue(reviewSummary["pruned_suite_generations"]) ?? "not recorded"
        let prunedMaskedComparison = prunedMaskedComparisonLine(reviewSummary)
        let prunedRuntimeSource = prunedSuiteRuntimeSourceLine(reviewSummary)
        let evalCount = intValue(reviewSummary["eval_count"]) ?? 0
        let promptCount = intValue(reviewSummary["prompt_count"]) ?? 0
        let text = """
        # Expert Lab Final Conversion Report

        Generated: \(generated)

        Original source: \(originalSource)
        Reviewed prune plan: \(reviewedPlanURL.path)
        Pruned BF16/F16 source: \(prunedSource)
        Pruned-source verification: \(pruneVerificationOK ? "passed" : "failed or missing")
        Same-suite review evidence: \(sameSuiteReady ? "ready" : "missing")
        Pruned BF16/F16 same-suite generation: \(prunedSuiteReady ? "ready" : "missing")
        Pruned BF16/F16 generation summary: \(prunedSuiteSummary)
        Pruned BF16/F16 generations: \(prunedSuiteGenerations)
        Pruned-vs-reviewed masked comparison: \(prunedMaskedComparison)
        Pruned vMLX runtime source: \(prunedRuntimeSource)
        Prompt evidence: \(promptCount)
        Mask comparison runs: \(evalCount)
        Prune report: \(pruneReport)
        Converted output: \(output)
        Profile: \(plan.profile)
        Family: \(plan.family.rawValue)
        Quantization method: \(plan.method.rawValue)

        This conversion used the Expert Lab path: prompt-suite trace review, reviewed keep/drop plan, verified BF16/F16 hard prune, then final conversion.
        """
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
            return .init(
                id: .expertLabFinalReport,
                title: "Expert Lab final report saved",
                status: .pass,
                required: false,
                hint: url.path
            )
        } catch {
            return .init(
                id: .expertLabFinalReport,
                title: "Expert Lab final report saved",
                status: .warn,
                required: false,
                hint: error.localizedDescription
            )
        }
    }

    static let expertLabSmokeTimeoutSeconds: Double = 120

    private static let expertLabSmokePrompts: [ExpertLabSmokePrompt] = [
        ExpertLabSmokePrompt(id: "smoke-arithmetic", text: "Answer with only the number: 2 + 2 =", maxTokens: nil, temperature: nil),
        ExpertLabSmokePrompt(id: "smoke-explain", text: "In one short sentence, explain what a router expert is.", maxTokens: nil, temperature: nil),
        ExpertLabSmokePrompt(id: "smoke-format", text: "Write exactly three comma-separated colors.", maxTokens: nil, temperature: nil)
    ]

    static func runExpertLabNativeSmoke(
        outputDir: URL,
        suiteURL: URL? = nil,
        timeoutSeconds: Double = expertLabSmokeTimeoutSeconds,
        requiresSuite: Bool = false
    ) async -> VerifyCheck {
        let runner = InferenceRunner(modelPath: outputDir)
        let suitePrompts = suiteURL.flatMap(loadExpertLabSuitePrompts)
        if requiresSuite, suitePrompts == nil {
            return writeExpertLabSmokeArtifact(
                records: [],
                outputDir: outputDir,
                source: "missing-reviewed-suite",
                suiteURL: suiteURL,
                error: "post-quant Expert Lab verification requires the reviewed prompt suite"
            )
        }
        let prompts = suitePrompts ?? expertLabSmokePrompts
        let source = suitePrompts == nil ? "built-in-smoke" : "reviewed-suite"
        var records: [ExpertLabSmokeRecord] = []
        for prompt in prompts {
            let generationSettings = ExpertLabSmokeGenerationSettings(
                maxTokens: prompt.maxTokens ?? 24,
                temperature: prompt.temperature ?? 0
            )
            do {
                let result = try await generateSmoke(
                    runner: runner,
                    prompt: prompt.text,
                    maxTokens: generationSettings.maxTokens,
                    temperature: generationSettings.temperature,
                    timeoutSeconds: timeoutSeconds
                )
                let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                records.append(ExpertLabSmokeRecord(
                    promptID: prompt.id,
                    prompt: prompt.text,
                    generationSettings: generationSettings,
                    ok: !trimmed.isEmpty && result.tokens > 0,
                    text: trimmed,
                    tokens: result.tokens,
                    tokensPerSec: result.tokensPerSec,
                    elapsedS: result.elapsedS,
                    error: trimmed.isEmpty ? "empty generation" : nil,
                    runtimeInfo: convertedRuntimeInfo(
                        outputDir: outputDir,
                        modelPath: result.model
                    )
                ))
            } catch {
                records.append(ExpertLabSmokeRecord(
                    promptID: prompt.id,
                    prompt: prompt.text,
                    generationSettings: generationSettings,
                    ok: false,
                    text: nil,
                    tokens: 0,
                    tokensPerSec: 0,
                    elapsedS: 0,
                    error: String(describing: error)
                ))
            }
        }
        return writeExpertLabSmokeArtifact(
            records: records,
            outputDir: outputDir,
            source: source,
            suiteURL: suitePrompts == nil ? nil : suiteURL,
            error: nil
        )
    }

    static func writeExpertLabSmokeArtifact(
        records: [ExpertLabSmokeRecord],
        outputDir: URL,
        source: String = "unknown",
        suiteURL: URL? = nil,
        error: String? = nil
    ) -> VerifyCheck {
        let url = outputDir.appendingPathComponent("expert_lab_smoke.jsonl")
        let summaryURL = outputDir.appendingPathComponent("expert_lab_smoke_summary.json")
        do {
            let records = enrichedSmokeRecords(records, outputDir: outputDir)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let lines = try records.map { record -> String in
                let data = try encoder.encode(record)
                return String(data: data, encoding: .utf8) ?? "{}"
            }
            try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
            let failed = records.filter { !$0.ok }.map(\.promptID)
            let runtimeInfos = records.filter(\.ok).compactMap(\.runtimeInfo)
            let summary: [String: Any] = [
                "schema": "jang-expert-lab-post-quant-smoke-v1",
                "source": source,
                "suite_jsonl": suiteURL?.path ?? NSNull(),
                "suite_sha256": suiteURL.flatMap(fileSHA256) ?? NSNull(),
                "prompt_count": records.count,
                "passed_count": records.filter(\.ok).count,
                "failed": failed,
                "prompt_ids": records.map(\.promptID),
                "runtime_info": postQuantSmokeRuntimeSummary(runtimeInfos) ?? NSNull(),
                "error": error ?? NSNull(),
                "artifact": url.path
            ]
            let summaryData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            try summaryData.write(to: summaryURL)
            let passed = error == nil && !records.isEmpty && records.allSatisfy(\.ok)
            let failureHint: String
            if let error {
                failureHint = "\(error); see \(summaryURL.path)"
            } else if records.isEmpty {
                failureHint = "no smoke prompts ran; see \(summaryURL.path)"
            } else {
                failureHint = "failed \(failed.joined(separator: ", ")); see \(url.path)"
            }
            return .init(
                id: .expertLabNativeSmoke,
                title: source == "reviewed-suite" ? "Expert Lab same-suite post-quant prompts" : "Expert Lab native smoke prompts",
                status: passed ? .pass : .fail,
                required: true,
                hint: passed ? "\(records.count) prompts from \(suiteURL?.path ?? source); \(url.path)" : failureHint
            )
        } catch {
            return .init(
                id: .expertLabNativeSmoke,
                title: "Expert Lab native smoke prompts",
                status: .fail,
                required: true,
                hint: "could not write smoke artifact: \(error.localizedDescription)"
            )
        }
    }

    private static func enrichedSmokeRecords(
        _ records: [ExpertLabSmokeRecord],
        outputDir: URL
    ) -> [ExpertLabSmokeRecord] {
        records.map { record in
            guard record.ok, record.runtimeInfo == nil else { return record }
            return record.withRuntimeInfo(
                convertedRuntimeInfo(outputDir: outputDir, modelPath: outputDir.path)
            )
        }
    }

    private static func convertedRuntimeInfo(
        outputDir: URL,
        modelPath: String
    ) -> ExpertLabConvertedRuntimeInfo {
        let config = readJSONObject(outputDir.appendingPathComponent("jang_config.json")) ?? [:]
        let quantization = config["quantization"] as? [String: Any] ?? [:]
        let architecture = config["architecture"] as? [String: Any] ?? [:]
        let capabilities = config["capabilities"] as? [String: Any] ?? [:]
        let sourceModel = config["source_model"] as? [String: Any] ?? [:]
        let profile = trimmedString(config["profile"])
        let explicitFormat = trimmedString(config["format"] ?? config["weight_format"] ?? quantization["format"])
        let inferredFormat: String?
        if let explicitFormat {
            inferredFormat = explicitFormat
        } else if profile?.range(of: "jangtq", options: .caseInsensitive) != nil {
            inferredFormat = "jangtq"
        } else {
            inferredFormat = nil
        }
        let normalizedFormat = inferredFormat?.lowercased()
        let architectureName = trimmedString(
            architecture["type"]
                ?? sourceModel["architecture"]
                ?? capabilities["family"]
                ?? capabilities["arch"]
                ?? config["model_type"]
        )
        return ExpertLabConvertedRuntimeInfo(
            runtimeMode: normalizedFormat.map { "post_quant_\($0)" } ?? "post_quant_converted",
            backend: "jang_tools inference",
            modelPath: modelPath,
            outputPath: outputDir.path,
            configFound: !config.isEmpty,
            format: normalizedFormat,
            formatVersion: trimmedString(config["format_version"] ?? config["version"]),
            profile: profile,
            architecture: architectureName,
            capabilityFamily: trimmedString(capabilities["family"]),
            capabilityArch: trimmedString(capabilities["arch"]),
            quantizationBits: quantizationBitValues(quantization),
            quantizationBlockSize: intValue(
                quantization["block_size"]
                    ?? quantization["group_size"]
                    ?? quantization["blockSize"]
                    ?? quantization["groupSize"]
            )
        )
    }

    private static func quantizationBitValues(_ quantization: [String: Any]) -> [Int] {
        let arrayBits = intArray(quantization["bit_widths_used"] ?? quantization["bitWidthsUsed"])
        if !arrayBits.isEmpty {
            return arrayBits
        }
        return [
            quantization["bits"],
            quantization["bits_default"],
            quantization["actual_bits_per_weight"],
            quantization["actual_bits"],
            quantization["mxtq_bits"]
        ].compactMap(intValue)
    }

    private static func postQuantSmokeRuntimeSummary(
        _ runtimeInfos: [ExpertLabConvertedRuntimeInfo]
    ) -> [String: Any]? {
        guard let first = runtimeInfos.first else { return nil }
        var payload = first.payload()
        payload["record_count"] = runtimeInfos.count
        payload["runtime_modes"] = Array(Set(runtimeInfos.compactMap(\.runtimeMode))).sorted()
        payload["model_paths"] = Array(Set(runtimeInfos.compactMap(\.modelPath))).sorted()
        payload["formats"] = Array(Set(runtimeInfos.compactMap(\.format))).sorted()
        payload["config_found_all"] = runtimeInfos.allSatisfy { $0.configFound == true }
        return payload
    }

    static func writeExpertLabFinalComparison(
        plan: ConversionPlan,
        outputDir: URL,
        reviewedPlanURL: URL
    ) -> VerifyCheck {
        let smokeURL = outputDir.appendingPathComponent("expert_lab_smoke.jsonl")
        let smokeRecords = readSmokeRecords(from: smokeURL)
        let smokeSummary = readSmokeSummary(from: outputDir.appendingPathComponent("expert_lab_smoke_summary.json"))
        let reviewSummary = readReviewSummary(reviewedPlanURL: reviewedPlanURL)
        let expectedSuiteURL = reviewedPruneSuiteURL(plan: plan)
        let semanticCoverage = reviewedSuiteSemanticCoverage(expectedSuiteURL)
        let semanticCoverageIssue = expectedSuiteURL.flatMap { suiteSemanticCoverageIssue($0) }
        let sameSuiteIssue = postQuantSameSuiteIssue(
            smokeRecords: smokeRecords,
            smokeSummary: smokeSummary,
            expectedSuiteURL: expectedSuiteURL,
            convertedOutputURL: plan.outputURL ?? outputDir
        )
        let prunedBehaviorComparison = postQuantPrunedBehaviorComparison(
            smokeRecords: smokeRecords,
            reviewSummary: reviewSummary,
            expectedPrunedSourceURL: plan.expertReviewPrunedSourceURL
        )
        let postQuantIssue = sameSuiteIssue ?? prunedBehaviorComparison.issue ?? semanticCoverageIssue
        let payload: [String: Any] = [
            "schema": "jang-expert-lab-final-comparison-v1",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "original_source": plan.expertReviewOriginalSourceURL?.path ?? NSNull(),
            "pruned_source": plan.expertReviewPrunedSourceURL?.path ?? NSNull(),
            "converted_output": (plan.outputURL ?? outputDir).path,
            "reviewed_prune_plan": reviewedPlanURL.path,
            "review": reviewSummary,
            "post_quant_same_suite_ready": postQuantIssue == nil,
            "post_quant_same_suite_issue": postQuantIssue ?? NSNull(),
            "post_quant_reviewed_suite": expectedSuiteURL?.path ?? NSNull(),
            "post_quant_reviewed_suite_sha256": expectedSuiteURL.flatMap(fileSHA256) ?? NSNull(),
            "post_quant_reviewed_suite_semantic_coverage": semanticCoverage,
            "native_smoke": [
                "artifact": smokeURL.path,
                "prompt_count": smokeRecords.count,
                "summary_prompt_count": smokeSummary.promptCount ?? NSNull(),
                "passed_count": smokeRecords.filter(\.ok).count,
                "failed": smokeRecords.filter { !$0.ok }.map(\.promptID),
                "tokens_per_sec_mean": mean(smokeRecords.map(\.tokensPerSec)),
                "source": smokeSummary.source,
                "suite_jsonl": smokeSummary.suiteURL?.path ?? NSNull(),
                "suite_sha256": smokeSummary.suiteSHA256 ?? NSNull(),
                "prompt_ids": smokeSummary.promptIDs,
                "runtime_info": postQuantSmokeRuntimeSummary(smokeRecords.filter(\.ok).compactMap(\.runtimeInfo)) ?? NSNull(),
                "error": smokeSummary.error ?? NSNull(),
            ] as [String: Any],
            "post_quant_vs_pruned_bf16": prunedBehaviorComparison.payload(
                threshold: maximumPostQuantPrunedTextDelta
            )
        ]
        let comparisonURL = outputDir.appendingPathComponent("expert_lab_final_comparison.json")
        do {
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: comparisonURL)
            appendFinalComparisonSection(
                outputDir: outputDir,
                comparisonURL: comparisonURL,
                reviewSummary: reviewSummary,
                smokeRecords: smokeRecords,
                smokeSummary: smokeSummary,
                semanticCoverage: semanticCoverage,
                prunedBehaviorComparison: prunedBehaviorComparison,
                postQuantIssue: postQuantIssue
            )
            return .init(
                id: .expertLabFinalComparison,
                title: "Expert Lab final comparison saved",
                status: postQuantIssue == nil ? .pass : .fail,
                required: true,
                hint: postQuantIssue ?? comparisonURL.path
            )
        } catch {
            return .init(
                id: .expertLabFinalComparison,
                title: "Expert Lab final comparison saved",
                status: .fail,
                required: true,
                hint: error.localizedDescription
            )
        }
    }

    private static func postQuantSameSuiteIssue(
        smokeRecords: [ExpertLabSmokeRecord],
        smokeSummary: ExpertLabSmokeSummary,
        expectedSuiteURL: URL?,
        convertedOutputURL: URL
    ) -> String? {
        guard smokeSummary.source == "reviewed-suite" else {
            return "post-quant verification did not use the reviewed Expert Lab prompt suite"
        }
        guard let suiteURL = smokeSummary.suiteURL else {
            return "post-quant reviewed prompt suite path was not recorded"
        }
        guard let expectedSuiteURL else {
            return "post-quant reviewed prompt suite path is missing from pruned-source sidecars"
        }
        guard canonicalPath(suiteURL.path) == canonicalPath(expectedSuiteURL.path) else {
            return "post-quant verification used \(suiteURL.path) instead of reviewed suite \(expectedSuiteURL.path)"
        }
        guard let suitePromptIDs = jsonlStringIDs(expectedSuiteURL, keys: ["id", "prompt_id", "promptID"]),
              !suitePromptIDs.isEmpty else {
            return "post-quant reviewed prompt suite IDs are unreadable"
        }
        guard !smokeRecords.isEmpty else {
            return "post-quant reviewed prompt suite produced no records"
        }
        let recordIDs = smokeRecords.map(\.promptID)
        if let summaryPromptCount = smokeSummary.promptCount,
           summaryPromptCount != recordIDs.count {
            return "post-quant reviewed prompt suite summary records \(summaryPromptCount) prompts for \(recordIDs.count) generated rows"
        }
        if Set(recordIDs).count < recordIDs.count {
            return "post-quant reviewed prompt suite produced duplicate prompt IDs"
        }
        if Set(smokeSummary.promptIDs).count < smokeSummary.promptIDs.count {
            return "post-quant reviewed prompt suite summary has duplicate prompt IDs"
        }
        let expected = Set(suitePromptIDs)
        let actual = Set(recordIDs)
        let missing = expected.subtracting(actual)
        if !missing.isEmpty {
            return "post-quant reviewed prompt suite missing prompt IDs: \(previewIDs(missing))"
        }
        let unexpected = actual.subtracting(expected)
        if !unexpected.isEmpty {
            return "post-quant reviewed prompt suite has unexpected prompt IDs: \(previewIDs(unexpected))"
        }
        if recordIDs != suitePromptIDs {
            return "post-quant reviewed prompt suite order does not match reviewed suite"
        }
        let summarized = Set(smokeSummary.promptIDs)
        if summarized != actual || smokeSummary.promptIDs.count != recordIDs.count {
            return "post-quant reviewed prompt suite summary does not match generated rows"
        }
        if smokeSummary.promptIDs != recordIDs {
            return "post-quant reviewed prompt suite summary prompt order does not match generated rows"
        }
        if let generationSettingsIssue = postQuantGenerationSettingsIssue(
            smokeRecords: smokeRecords,
            suiteURL: expectedSuiteURL
        ) {
            return generationSettingsIssue
        }
        let failed = smokeRecords.filter { !$0.ok }.map(\.promptID)
        guard failed.isEmpty else {
            return "post-quant reviewed prompt suite failed: \(failed.joined(separator: ", "))"
        }
        if let runtimeIssue = postQuantSmokeRuntimeIssue(
            smokeRecords: smokeRecords,
            convertedOutputURL: convertedOutputURL
        ) {
            return runtimeIssue
        }
        guard let expectedSuiteSHA256 = fileSHA256(expectedSuiteURL) else {
            return "post-quant reviewed prompt suite fingerprint could not be computed"
        }
        guard let smokeSuiteSHA256 = smokeSummary.suiteSHA256, !smokeSuiteSHA256.isEmpty else {
            return "post-quant reviewed prompt suite fingerprint was not recorded"
        }
        guard smokeSuiteSHA256 == expectedSuiteSHA256 else {
            return "post-quant reviewed prompt suite fingerprint does not match reviewed suite"
        }
        return nil
    }

    private static func postQuantGenerationSettingsIssue(
        smokeRecords: [ExpertLabSmokeRecord],
        suiteURL: URL
    ) -> String? {
        let suiteSettings = suiteGenerationSettings(suiteURL) ?? [:]
        let suiteRequiresSettings = suiteSettings.values.contains {
            $0.maxTokens != nil || $0.temperature != nil
        }
        let recordsHaveSettings = smokeRecords.contains { $0.generationSettings != nil }
        guard suiteRequiresSettings || recordsHaveSettings else { return nil }
        for record in smokeRecords {
            guard let settings = record.generationSettings else {
                return "post-quant reviewed prompt suite is missing decode settings evidence"
            }
            guard settings.maxTokens > 0, settings.temperature.isFinite else {
                return "post-quant reviewed prompt suite has unreadable decode settings"
            }
            let expectedMaxTokens = suiteSettings[record.promptID]?.maxTokens ?? 24
            if settings.maxTokens != expectedMaxTokens {
                return "post-quant max_tokens for \(record.promptID) does not match reviewed suite"
            }
            let expectedTemperature = suiteSettings[record.promptID]?.temperature ?? 0
            if abs(settings.temperature - expectedTemperature) > 0.000_001 {
                return "post-quant temperature for \(record.promptID) does not match reviewed suite"
            }
        }
        return nil
    }

    private static func postQuantSmokeRuntimeIssue(
        smokeRecords: [ExpertLabSmokeRecord],
        convertedOutputURL: URL
    ) -> String? {
        for record in smokeRecords where record.ok {
            guard record.tokens > 0 else {
                return "post-quant generation row is missing token evidence"
            }
            guard let runtime = record.runtimeInfo else {
                return "post-quant generation row is missing converted runtime evidence"
            }
            guard let runtimeMode = nonEmptyString(runtime.runtimeMode),
                  runtimeMode.hasPrefix("post_quant_") else {
                return "post-quant generation row did not record a post-quant runtime mode"
            }
            guard nonEmptyString(runtime.backend) != nil else {
                return "post-quant generation row is missing converted backend evidence"
            }
            guard let modelPath = nonEmptyString(runtime.modelPath) else {
                return "post-quant generation row is missing converted model path evidence"
            }
            guard let outputPath = nonEmptyString(runtime.outputPath) else {
                return "post-quant generation row is missing converted output path evidence"
            }
            let convertedPath = canonicalPath(convertedOutputURL.path)
            if canonicalPath(modelPath) != convertedPath || canonicalPath(outputPath) != convertedPath {
                return "post-quant generation row runtime path does not match the converted output"
            }
        }
        return nil
    }

    private static let maximumPostQuantPrunedTextDelta = 0.75

    private static func postQuantPrunedBehaviorComparison(
        smokeRecords: [ExpertLabSmokeRecord],
        reviewSummary: [String: Any],
        expectedPrunedSourceURL: URL?
    ) -> PostQuantPrunedBehaviorComparison {
        guard let expectedPrunedSourceURL else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: nil,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant verification is missing the selected pruned BF16 source"
            )
        }
        guard let reviewPrunedSourcePath = stringValue(reviewSummary["pruned_source"] ?? reviewSummary["prunedSource"]) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: nil,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant verification is missing the pruned BF16 source path"
            )
        }
        if canonicalPath(reviewPrunedSourcePath) != canonicalPath(expectedPrunedSourceURL.path) {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: nil,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant pruned BF16 reference source path does not match the selected pruned BF16/F16 source"
            )
        }
        guard let referenceURL = embeddedSidecarURL(
            reviewSummary["pruned_suite_generations"],
            prunedSource: expectedPrunedSourceURL,
            fallbackName: "expert_lab_pruned_generations.jsonl"
        ) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: nil,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant pruned BF16 reference generations must be embedded in the pruned source"
            )
        }
        guard FileManager.default.isReadableFile(atPath: referenceURL.path) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: nil,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant verification is missing pruned BF16 reference generations"
            )
        }
        guard let rows = jsonlObjects(referenceURL) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "pruned BF16 reference generations are unreadable"
            )
        }
        if let issue = prunedGenerationRowEvidenceIssue(
            rows: rows,
            prunedSourcePath: expectedPrunedSourceURL.path,
            expectedLayerCount: expectedReviewedLayerCount(summary: reviewSummary)
        ) {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: issue
            )
        }
        guard let evalURL = embeddedSidecarURL(
            reviewSummary["eval_jsonl"] ?? reviewSummary["evalJSONL"],
            prunedSource: expectedPrunedSourceURL,
            fallbackName: "expert_lab_eval.jsonl"
        ) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant review eval.jsonl must be embedded in the pruned source"
            )
        }
        guard FileManager.default.isReadableFile(atPath: evalURL.path) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant verification is missing review eval.jsonl validator evidence"
            )
        }
        guard let evalRows = jsonlObjects(evalURL) else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "review eval.jsonl validator evidence is unreadable"
            )
        }
        if let issue = evalRowValidatorClassificationIssue(rows: evalRows) {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: issue
            )
        }
        var evalByPromptID: [String: [String: Any]] = [:]
        for row in evalRows {
            guard let id = promptID(in: row) else {
                return PostQuantPrunedBehaviorComparison(
                    referenceURL: referenceURL,
                    comparedCount: 0,
                    meanTextDelta: nil,
                    maxTextDelta: nil,
                    issue: "review eval.jsonl prompt IDs are unreadable"
                )
            }
            if evalByPromptID[id] != nil {
                return PostQuantPrunedBehaviorComparison(
                    referenceURL: referenceURL,
                    comparedCount: 0,
                    meanTextDelta: nil,
                    maxTextDelta: nil,
                    issue: "review eval.jsonl contains duplicate prompt IDs"
                )
            }
            evalByPromptID[id] = row
        }
        let baselineQualifiedEvalRows = evalRows.filter {
            boolValue($0["baselineQualified"] ?? $0["baseline_qualified"]) == true
        }
        guard !baselineQualifiedEvalRows.isEmpty else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant comparison has no baseline-qualified validator prompts"
            )
        }
        let baselineQualifiedSemanticCoverage = evalRowsBaselineQualifiedSemanticCoverage(evalRows)
        let missingBaselineQualifiedCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(baselineQualifiedSemanticCoverage))
            .sorted()
        var prunedByPromptID: [String: (text: String, tokens: Int)] = [:]
        var referenceOrder: [String] = []
        for row in rows {
            guard let prompt = row["prompt"] as? [String: Any],
                  let id = promptID(in: prompt),
                  let result = row["result"] as? [String: Any],
                  let text = trimmedString(result["text"]),
                  !text.isEmpty else {
                return PostQuantPrunedBehaviorComparison(
                    referenceURL: referenceURL,
                    comparedCount: 0,
                    meanTextDelta: nil,
                    maxTextDelta: nil,
                    issue: "pruned BF16 reference generations are missing prompt IDs or text"
                )
            }
            if prunedByPromptID[id] != nil {
                return PostQuantPrunedBehaviorComparison(
                    referenceURL: referenceURL,
                    comparedCount: 0,
                    meanTextDelta: nil,
                    maxTextDelta: nil,
                    issue: "pruned BF16 reference generations contain duplicate prompt IDs"
                )
            }
            prunedByPromptID[id] = (text, intValue(result["tokens"]) ?? 0)
            referenceOrder.append(id)
        }
        guard !prunedByPromptID.isEmpty else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "pruned BF16 reference generations are empty"
            )
        }
        let evalIDs = Set(evalByPromptID.keys)
        let referenceIDs = Set(prunedByPromptID.keys)
        let missingEvalIDs = referenceIDs.subtracting(evalIDs)
        if !missingEvalIDs.isEmpty {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "review eval.jsonl is missing pruned BF16 prompt IDs: \(previewIDs(missingEvalIDs))"
            )
        }
        let unexpectedEvalIDs = evalIDs.subtracting(referenceIDs)
        if !unexpectedEvalIDs.isEmpty {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "review eval.jsonl has prompt IDs outside pruned BF16 reference: \(previewIDs(unexpectedEvalIDs))"
            )
        }

        var postQuantByPromptID: [String: ExpertLabSmokeRecord] = [:]
        for record in smokeRecords {
            guard record.ok,
                  let text = record.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { continue }
            postQuantByPromptID[record.promptID] = record.withRuntimeInfo(record.runtimeInfo)
        }
        let expected = referenceIDs
        let actual = Set(postQuantByPromptID.keys)
        let missing = expected.subtracting(actual)
        if !missing.isEmpty {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: postQuantByPromptID.count,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant output missing pruned BF16 prompt IDs: \(previewIDs(missing))"
            )
        }
        let unexpected = actual.subtracting(expected)
        if !unexpected.isEmpty {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: postQuantByPromptID.count,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant output has prompt IDs outside pruned BF16 reference: \(previewIDs(unexpected))"
            )
        }

        let comparisonRows = referenceOrder.compactMap { id -> PostQuantPromptComparisonRow? in
            guard let pruned = prunedByPromptID[id],
                  let postQuant = postQuantByPromptID[id],
                  let postQuantText = postQuant.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !postQuantText.isEmpty,
                  let evalRow = evalByPromptID[id] else { return nil }
            let baselineQualified = boolValue(evalRow["baselineQualified"] ?? evalRow["baseline_qualified"]) == true
            let postQuantPassed = baselineQualified
                ? validatorPassed(text: postQuantText, row: evalRow)
                : nil
            let reviewClassification = promptClassification(evalRow)
            let classification: String
            if baselineQualified {
                classification = postQuantPassed == true ? "preserved" : "degraded"
            } else if reviewClassification == "baseline_invalid" {
                classification = "baseline_invalid"
            } else {
                classification = "inconclusive"
            }
            return PostQuantPromptComparisonRow(
                promptID: id,
                prunedBF16Text: pruned.text,
                postQuantText: postQuantText,
                prunedBF16Tokens: pruned.tokens,
                postQuantTokens: postQuant.tokens,
                textDelta: ExpertPromptEvaluator.normalizedTextDelta(pruned.text, postQuantText),
                validatorKind: trimmedString(evalRow["validatorKind"] ?? evalRow["validator_kind"]),
                baselineQualified: baselineQualified,
                postQuantPassed: postQuantPassed,
                promptClassification: classification,
                safeDropEvidenceEligible: baselineQualified && postQuantPassed == true,
                semanticDomains: semanticDomains(in: evalRow)
            )
        }
        guard !comparisonRows.isEmpty else {
            return PostQuantPrunedBehaviorComparison(
                referenceURL: referenceURL,
                comparedCount: 0,
                meanTextDelta: nil,
                maxTextDelta: nil,
                issue: "post-quant comparison has no overlapping pruned BF16 prompt IDs"
            )
        }
        let deltas = comparisonRows.map(\.textDelta)
        let meanDelta = mean(deltas)
        let maxDelta = deltas.max() ?? 0
        let classifications = comparisonRows.map(\.promptClassification)
        let counts = classificationCounts(classifications)
        let baselineInvalidIDs = comparisonRows
            .filter { $0.promptClassification == "baseline_invalid" }
            .map(\.promptID)
        let inconclusiveIDs = comparisonRows
            .filter { $0.promptClassification == "inconclusive" }
            .map(\.promptID)
        let preservedIDs = comparisonRows
            .filter { $0.promptClassification == "preserved" }
            .map(\.promptID)
        let degradedIDs = comparisonRows
            .filter { $0.promptClassification == "degraded" }
            .map(\.promptID)
        let baselineQualifiedRows = comparisonRows.filter(\.baselineQualified)
        let postQuantPassRate = passRate(baselineQualifiedRows.map(\.postQuantPassed))
        let issue: String?
        if baselineQualifiedRows.isEmpty {
            issue = "post-quant comparison has no baseline-qualified validator prompts"
        } else if !missingBaselineQualifiedCoverage.isEmpty {
            issue = "post-quant baseline-qualified coverage is missing: \(missingBaselineQualifiedCoverage.joined(separator: ", "))"
        } else if !degradedIDs.isEmpty {
            issue = "post-quant output failed baseline-qualified validators: \(previewIDs(Set(degradedIDs)))"
        } else if let postQuantPassRate, postQuantPassRate < 1.0 {
            issue = "post-quant validator pass rate is below 100% on baseline-qualified prompts"
        } else if postQuantPassRate == nil {
            issue = "post-quant comparison is missing baseline-qualified validator outcomes"
        } else {
            issue = nil
        }
        return PostQuantPrunedBehaviorComparison(
            referenceURL: referenceURL,
            comparedCount: deltas.count,
            meanTextDelta: meanDelta,
            maxTextDelta: maxDelta,
            baselineQualifiedPromptCount: baselineQualifiedRows.count,
            postQuantBaselineQualifiedPassRate: postQuantPassRate,
            classificationCounts: counts,
            baselineInvalidPromptIDs: baselineInvalidIDs,
            inconclusivePromptIDs: inconclusiveIDs,
            preservedPromptIDs: preservedIDs,
            degradedPromptIDs: degradedIDs,
            baselineQualifiedSemanticCoverage: baselineQualifiedSemanticCoverage,
            missingBaselineQualifiedSemanticCoverage: missingBaselineQualifiedCoverage,
            issue: issue,
            rows: comparisonRows
        )
    }

    private static func readReviewSummary(reviewedPlanURL: URL) -> [String: Any] {
        let fm = FileManager.default
        var summary: [String: Any] = [
            "reviewed_prune_plan": reviewedPlanURL.path
        ]
        if let json = readJSONObject(reviewedPlanURL) {
            summary["method"] = json["method"]
            summary["prompt_count"] = json["promptCount"]
            summary["run_id"] = json["run_id"]
            summary["atlas_id"] = json["atlas_id"]
            summary["source_model"] = json["source_model"]
            summary["review_bundle"] = json["review_bundle"]
            summary["eval_artifact"] = json["eval_artifact"]
            summary["keep_experts_per_layer"] = json["keepExpertsPerLayer"]
            if let comparison = json["comparison_summary"] as? [String: Any] {
                summary["latest_eval"] = comparison
                summary["eval_count"] = max(intValue(summary["eval_count"]) ?? 0, 1)
            }
            if let layers = json["layers"] as? [String: Any] {
                var layerCount = 0
                var keepCount = 0
                var dropCount = 0
                var lockedKeepCount = 0
                var forcedDropCount = 0
                var evidenceCount = 0
                var evidencePreview: [String] = []
                var layerSummaries: [[String: Any]] = []
                for key in sortedLayerKeys(layers.keys) {
                    guard let value = layers[key] else { continue }
                    guard let layer = value as? [String: Any] else { continue }
                    let keep = intArray(layer["keep"])
                    let drop = intArray(layer["drop"])
                    let lockedKeep = intArray(layer["locked_keep"])
                    let forcedDrop = intArray(layer["user_forced_drop"])
                    let evidenceRows = (layer["evidence"] as? [[String: Any]])
                        ?? (layer["evidence"] as? [Any])?.compactMap { $0 as? [String: Any] }
                        ?? []
                    layerCount += 1
                    keepCount += keep.count
                    dropCount += drop.count
                    lockedKeepCount += lockedKeep.count
                    forcedDropCount += forcedDrop.count
                    evidenceCount += evidenceRows.count
                    if layerSummaries.count < 12 {
                        layerSummaries.append([
                            "layer": intValue(key) ?? key,
                            "keep_count": keep.count,
                            "drop_count": drop.count,
                            "locked_keep_count": lockedKeep.count,
                            "user_forced_drop_count": forcedDrop.count,
                            "drop_preview": Array(drop.prefix(8))
                        ])
                    }
                    for evidence in evidenceRows where evidencePreview.count < 6 {
                        guard boolValue(evidence["kept"]) != true else { continue }
                        evidencePreview.append(evidencePreviewLine(layer: key, evidence: evidence))
                    }
                }
                summary["layer_count"] = layerCount
                summary["keep_count"] = keepCount
                summary["drop_count"] = dropCount
                summary["locked_keep_count"] = lockedKeepCount
                summary["user_forced_drop_count"] = forcedDropCount
                summary["evidence_count"] = evidenceCount
                summary["evidence_preview"] = evidencePreview
                summary["layer_summaries"] = layerSummaries
            }
        }

        let prunedReviewSummaryURL = reviewedPlanURL
            .deletingLastPathComponent()
            .appendingPathComponent("expert_lab_review_summary.json")
        if let prunedReview = readJSONObject(prunedReviewSummaryURL) {
            summary["pruned_review_summary"] = prunedReview
            summary["pruned_source"] = prunedReview["pruned_source"]
            summary["same_suite_verification_ready"] = prunedReview["same_suite_verification_ready"]
            summary["review_run_directory"] = prunedReview["review_run_directory"]
            summary["review_eval_directory"] = prunedReview["review_eval_directory"]
            summary["suite_jsonl"] = prunedReview["suite_jsonl"]
            summary["eval_jsonl"] = prunedReview["eval_jsonl"]
            summary["eval_index"] = prunedReview["eval_index"]
            summary["pruned_suite_verification_ready"] = prunedReview["pruned_suite_verification_ready"]
            summary["pruned_suite_verification_issue"] = prunedReview["pruned_suite_verification_issue"]
            summary["pruned_suite_summary"] = prunedReview["pruned_suite_summary"]
            summary["pruned_suite_generations"] = prunedReview["pruned_suite_generations"]
            if summary["run_id"] == nil { summary["run_id"] = prunedReview["run_id"] }
            if summary["atlas_id"] == nil { summary["atlas_id"] = prunedReview["atlas_id"] }
            if intValue(summary["prompt_count"]) == nil {
                summary["prompt_count"] = prunedReview["prompt_count"]
            }
            if let evalPath = prunedReview["eval_jsonl"] as? String,
               fm.isReadableFile(atPath: evalPath) {
                summary["eval_count"] = max(intValue(summary["eval_count"]) ?? 0, 1)
            }
            if let comparisonPath = prunedReview["comparison_summary"] as? String,
               let comparison = readJSONObject(URL(fileURLWithPath: comparisonPath)) {
                summary["latest_eval"] = comparison
            }
        }

        let runDir = reviewedPlanURL.deletingLastPathComponent()
        if fm.fileExists(atPath: runDir.appendingPathComponent("run.json").path) {
            summary["review_run_directory"] = runDir.path
            summary["has_atlas"] = fm.fileExists(atPath: runDir.appendingPathComponent("atlas.json").path)
            summary["has_trace_sqlite"] = fm.fileExists(atPath: runDir.appendingPathComponent("trace.sqlite").path)
            summary["generation_count"] = lineCount(runDir.appendingPathComponent("generations.jsonl"))
            let evals = runDir.appendingPathComponent("evals", isDirectory: true)
            let evalDirs = ((try? fm.contentsOfDirectory(
                at: evals,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []).filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }
            summary["eval_count"] = evalDirs.count
            if summary["latest_eval"] == nil,
               let latest = evalDirs.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }).first,
               let evalSummary = readJSONObject(latest.appendingPathComponent("comparison_summary.json")) {
                summary["latest_eval"] = evalSummary
            }
        }
        return summary
    }

    private static func readSmokeRecords(from url: URL) -> [ExpertLabSmokeRecord] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(whereSeparator: \.isNewline).compactMap { line in
            try? decoder.decode(ExpertLabSmokeRecord.self, from: Data(line.utf8))
        }
    }

    private static func readSmokeSummary(from url: URL) -> ExpertLabSmokeSummary {
        guard let json = readJSONObject(url) else {
            return ExpertLabSmokeSummary(source: "unknown", suiteURL: nil, suiteSHA256: nil, promptCount: nil, promptIDs: [], error: nil)
        }
        let suiteURL = stringValue(json["suite_jsonl"]).map { URL(fileURLWithPath: $0) }
        let promptIDs = (json["prompt_ids"] as? [Any])?.compactMap(stringValue) ?? []
        return ExpertLabSmokeSummary(
            source: stringValue(json["source"]) ?? "unknown",
            suiteURL: suiteURL,
            suiteSHA256: stringValue(json["suite_sha256"] ?? json["suiteSHA256"]),
            promptCount: intValue(json["prompt_count"] ?? json["promptCount"]),
            promptIDs: promptIDs,
            error: stringValue(json["error"])
        )
    }

    private static func appendFinalComparisonSection(
        outputDir: URL,
        comparisonURL: URL,
        reviewSummary: [String: Any],
        smokeRecords: [ExpertLabSmokeRecord],
        smokeSummary: ExpertLabSmokeSummary,
        semanticCoverage: [String: Any],
        prunedBehaviorComparison: PostQuantPrunedBehaviorComparison,
        postQuantIssue: String?
    ) {
        let reportURL = outputDir.appendingPathComponent("expert_lab_final_report.md")
        let existing = (try? String(contentsOf: reportURL, encoding: .utf8)) ?? "# Expert Lab Final Conversion Report\n"
        let smokePass = smokeRecords.filter(\.ok).count
        let smokeTotal = smokeRecords.count
        let evalCount = intValue(reviewSummary["eval_count"]) ?? 0
        let generationCount = intValue(reviewSummary["generation_count"]) ?? 0
        let promptCount = intValue(reviewSummary["prompt_count"]) ?? 0
        let layerCount = intValue(reviewSummary["layer_count"]) ?? 0
        let dropCount = intValue(reviewSummary["drop_count"]) ?? 0
        let lockedKeepCount = intValue(reviewSummary["locked_keep_count"]) ?? 0
        let forcedDropCount = intValue(reviewSummary["user_forced_drop_count"]) ?? 0
        let evidenceCount = intValue(reviewSummary["evidence_count"]) ?? 0
        let method = stringValue(reviewSummary["method"]) ?? "not recorded"
        let runID = stringValue(reviewSummary["run_id"]) ?? "not recorded"
        let atlasID = stringValue(reviewSummary["atlas_id"]) ?? "not recorded"
        let reviewEvalDirectory = stringValue(reviewSummary["review_eval_directory"])
            ?? stringValue(reviewSummary["eval_artifact"])
            ?? "not recorded"
        let prunedSuiteReady = boolValue(reviewSummary["pruned_suite_verification_ready"]) ?? false
        let prunedSuiteSummary = stringValue(reviewSummary["pruned_suite_summary"]) ?? "not recorded"
        let prunedSuiteGenerations = stringValue(reviewSummary["pruned_suite_generations"]) ?? "not recorded"
        let prunedMaskedComparison = prunedMaskedComparisonLine(reviewSummary)
        let prunedRuntimeSource = prunedSuiteRuntimeSourceLine(reviewSummary)
        let postQuantRuntime = smokeRecords.filter(\.ok).compactMap(\.runtimeInfo).first?.reportLine() ?? "not recorded"
        let evalLine = latestEvalLine(reviewSummary["latest_eval"] as? [String: Any])
        let evidencePreview = reviewSummary["evidence_preview"] as? [String] ?? []

        var lines = [
            "",
            "## Final Comparison",
            "",
            "Comparison artifact: \(comparisonURL.path)",
            "Review run: \(runID)",
            "Atlas: \(atlasID)",
            "Prune method: \(method)",
            "Prompt evidence: \(promptCount)",
            "Review generations: \(generationCount)",
            "Atlas layers reviewed: \(layerCount)",
            "Mask comparison runs: \(evalCount)",
            "Reviewed eval artifact: \(reviewEvalDirectory)",
            "Pruned BF16/F16 same-suite generation: \(prunedSuiteReady ? "ready" : "missing")",
            "Pruned BF16/F16 generation summary: \(prunedSuiteSummary)",
            "Pruned BF16/F16 generations: \(prunedSuiteGenerations)",
            "Pruned-vs-reviewed masked comparison: \(prunedMaskedComparison)",
            "Pruned vMLX runtime source: \(prunedRuntimeSource)",
            "Pruned vMLX runtime versions: \(prunedSuiteRuntimeVersionLine(reviewSummary))",
            "Latest mask comparison: \(evalLine)",
            "Planned expert drops: \(dropCount)",
            "Locked keeps: \(lockedKeepCount)",
            "User-forced drops: \(forcedDropCount)",
            "Prune evidence rows: \(evidenceCount)",
            "Post-quant vs pruned BF16 reference: \(prunedBehaviorComparison.reportLine())",
            "Post-quant same-suite evidence: \(postQuantIssue == nil ? "passed" : "missing or failed")",
            "Post-quant reviewed-suite semantic coverage: \(semanticCoverageReportLine(semanticCoverage))",
            "Post-quant prompt source: \(smokeSummary.source)",
            "Post-quant prompt suite: \(smokeSummary.suiteURL?.path ?? "not recorded")",
            "Post-quant prompt suite SHA256: \(smokeSummary.suiteSHA256 ?? "not recorded")",
            "Post-quant runtime: \(postQuantRuntime)",
            "Native smoke prompts: \(smokePass) / \(smokeTotal) passed"
        ]
        if !smokeSummary.promptIDs.isEmpty {
            let preview = smokeSummary.promptIDs.prefix(8).joined(separator: ", ")
            lines.append("Post-quant prompt IDs: \(preview)\(smokeSummary.promptIDs.count > 8 ? ", ..." : "")")
        }
        if let error = smokeSummary.error {
            lines.append("Post-quant prompt error: \(error)")
        }
        if !evidencePreview.isEmpty {
            lines.append("")
            lines.append("Evidence preview:")
            lines.append(contentsOf: evidencePreview.map { "- \($0)" })
        }
        let section = "\n" + lines.joined(separator: "\n") + "\n"
        try? (existing + section).write(to: reportURL, atomically: true, encoding: .utf8)
    }

    private static func semanticCoverageReportLine(_ coverage: [String: Any]) -> String {
        let ready = boolValue(coverage["ready"]) == true
        let promptCount = intValue(coverage["prompt_count"]) ?? 0
        let domains = coverage["domains"] as? [String] ?? []
        let missing = coverage["missing"] as? [String] ?? []
        let status = ready ? "ready" : "missing"
        let domainPreview = domains.prefix(12).joined(separator: ", ")
        let domainTail = domains.count > 12 ? ", ..." : ""
        let missingText = missing.isEmpty ? "none" : missing.joined(separator: ", ")
        return "\(status); \(promptCount) prompts; domains \(domainPreview)\(domainTail); missing \(missingText)"
    }

    private static func sortedLayerKeys(_ keys: Dictionary<String, Any>.Keys) -> [String] {
        keys.sorted { lhs, rhs in
            let left = intValue(lhs)
            let right = intValue(rhs)
            if let left, let right, left != right { return left < right }
            if left != nil { return true }
            if right != nil { return false }
            return lhs < rhs
        }
    }

    private static func intArray(_ value: Any?) -> [Int] {
        guard let values = value as? [Any] else { return [] }
        return values.compactMap(intValue)
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        case let value as String:
            return Int(value)
        default:
            return nil
        }
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            return value
        case let value as NSNumber:
            return value.boolValue
        case let value as String:
            if value.caseInsensitiveCompare("true") == .orderedSame { return true }
            if value.caseInsensitiveCompare("false") == .orderedSame { return false }
            return nil
        default:
            return nil
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }

    private static func stringValue(_ value: Any?) -> String? {
        guard let value else { return nil }
        if value is NSNull { return nil }
        if let string = value as? String {
            return string.isEmpty ? nil : string
        }
        return String(describing: value)
    }

    private static func isBlockingRegressionSeverity(_ value: Any?) -> Bool {
        guard let severity = stringValue(value) else { return false }
        return severity == "high" || severity == "critical"
    }

    private static func evidencePreviewLine(layer: String, evidence: [String: Any]) -> String {
        let expert = intValue(evidence["expert"]).map(String.init) ?? "?"
        let label = stringValue(evidence["label"]) ?? "unlabeled"
        let forced = boolValue(evidence["user_forced_drop"]) == true ? "user-forced; " : ""
        let frequency = fixed4(evidence["frequency"])
        let routerMass = fixed4(evidence["router_mass"] ?? evidence["routerMass"] ?? evidence["probabilityMass"])
        let impactScope = stringValue(evidence["masked_impact_scope"] ?? evidence["maskedImpactScope"])
            .map { ", impact \($0)" } ?? ""
        let reviewedMask = jsonBool(evidence["reviewed_mask_member"] ?? evidence["reviewedMaskMember"])
            .map { $0 ? ", reviewed mask member" : ", not in reviewed mask" } ?? ""
        let reason = stringValue(evidence["reason"]).map { " - \($0)" } ?? ""
        return "L\(layer) E\(expert): \(label) (\(forced)freq \(frequency), mass \(routerMass)\(impactScope)\(reviewedMask))\(reason)"
    }

    private static func fixed4(_ value: Any?) -> String {
        guard let number = doubleValue(value) else { return "n/a" }
        return String(format: "%.4f", number)
    }

    private static func latestEvalLine(_ latestEval: [String: Any]?) -> String {
        guard let latestEval else { return "not recorded" }
        var parts: [String] = []
        if let promptCount = intValue(latestEval["promptCount"] ?? latestEval["prompt_count"]) {
            parts.append("\(promptCount) prompts")
        }
        if let passRateBaseline = doubleValue(latestEval["passRateBaseline"] ?? latestEval["pass_rate_baseline"]) {
            parts.append(String(format: "baseline pass %.2f", passRateBaseline))
        }
        if let passRateMasked = doubleValue(latestEval["passRateMasked"] ?? latestEval["pass_rate_masked"]) {
            parts.append(String(format: "masked pass %.2f", passRateMasked))
        }
        if let meanTextDelta = doubleValue(latestEval["meanTextDelta"] ?? latestEval["mean_text_delta"]) {
            parts.append(String(format: "mean text delta %.4f", meanTextDelta))
        }
        if let severity = stringValue(latestEval["regressionSeverity"] ?? latestEval["regression_severity"]) {
            parts.append("regression severity \(severity)")
        }
        if let latencyDelta = doubleValue(latestEval["meanLatencyDeltaPct"] ?? latestEval["mean_latency_delta_pct"]) {
            parts.append(String(format: "latency delta %.2f%%", latencyDelta))
        }
        if let highRiskDomains = latestEval["highRiskDomains"] as? [String], !highRiskDomains.isEmpty {
            parts.append("high-risk domains \(highRiskDomains.joined(separator: ", "))")
        } else if let highRiskDomains = latestEval["high_risk_domains"] as? [String], !highRiskDomains.isEmpty {
            parts.append("high-risk domains \(highRiskDomains.joined(separator: ", "))")
        }
        return parts.isEmpty ? "available" : parts.joined(separator: "; ")
    }

    private static func prunedMaskedComparisonLine(_ reviewSummary: [String: Any]) -> String {
        guard let summary = prunedSuiteSummaryObject(reviewSummary) else { return "not recorded" }
        var parts: [String] = []
        if let comparedCount = intValue(
            summary["reviewed_ab_comparison_count"] ?? summary["reviewed_masked_comparison_count"]
        ) {
            parts.append("\(comparedCount) prompts")
        }
        if let baselineQualified = intValue(summary["baseline_qualified_prompt_count"]) {
            parts.append("\(baselineQualified) baseline-qualified")
        }
        if let passRate = doubleValue(summary["pruned_baseline_qualified_pass_rate"]) {
            parts.append(String(format: "qualified pass %.2f", passRate))
        }
        if let counts = intDictionaryValue(summary["pruned_classification_counts"]) {
            let classes = ["preserved", "degraded", "baseline_invalid", "inconclusive"].compactMap { key -> String? in
                guard let value = counts[key], value > 0 else { return nil }
                return "\(key) \(value)"
            }.joined(separator: ", ")
            if !classes.isEmpty {
                parts.append("classes \(classes)")
            }
        }
        if let meanDelta = doubleValue(
            summary["reviewed_ab_mean_text_delta"] ?? summary["reviewed_masked_mean_text_delta"]
        ) {
            parts.append(String(format: "mean delta %.4f", meanDelta))
        }
        if let maxDelta = doubleValue(
            summary["reviewed_ab_max_text_delta"] ?? summary["reviewed_masked_max_text_delta"]
        ) {
            parts.append(String(format: "max delta %.4f", maxDelta))
        }
        let missingCoverage = stringArrayValue(
            summary["missing_baseline_qualified_semantic_coverage"]
                ?? summary["missingBaselineQualifiedSemanticCoverage"]
        ) ?? []
        if !missingCoverage.isEmpty {
            parts.append("missing qualified coverage \(missingCoverage.joined(separator: ", "))")
        }
        let degraded = stringArrayValue(summary["pruned_degraded_prompt_ids"] ?? summary["degradedPromptIDs"]) ?? []
        if !degraded.isEmpty {
            parts.append("degraded \(degraded.prefix(5).joined(separator: ", "))")
        }
        if let issue = stringValue(summary["issue"]), !issue.isEmpty {
            parts.append("issue \(issue)")
        }
        return parts.isEmpty ? "not recorded" : parts.joined(separator: "; ")
    }

    private static func prunedSuiteRuntimeSourceLine(_ reviewSummary: [String: Any]) -> String {
        guard let summary = prunedSuiteSummaryObject(reviewSummary) else { return "not recorded" }
        return stringValue(summary["runtime_source_model_path"]) ?? "not recorded"
    }

    private static func prunedSuiteRuntimeVersionLine(_ reviewSummary: [String: Any]) -> String {
        guard let summary = prunedSuiteSummaryObject(reviewSummary) else { return "not recorded" }
        let parts = [
            ("jang", stringValue(summary["jang_tools_version"] ?? summary["jangToolsVersion"])),
            ("mlx", stringValue(summary["mlx_version"] ?? summary["mlxVersion"])),
            ("mlx-lm", stringValue(summary["mlx_lm_version"] ?? summary["mlxLMVersion"]))
        ].compactMap { label, value -> String? in
            guard let value, !value.isEmpty else { return nil }
            return "\(label) \(value)"
        }
        return parts.isEmpty ? "not recorded" : parts.joined(separator: ", ")
    }

    private static func prunedSuiteSummaryObject(_ reviewSummary: [String: Any]) -> [String: Any]? {
        guard let summaryPath = stringValue(reviewSummary["pruned_suite_summary"]) else {
            return nil
        }
        return readJSONObject(URL(fileURLWithPath: summaryPath))
    }

    private static func readJSONObject(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }

    private static func fileSHA256(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(whereSeparator: \.isNewline).count
    }

    private static func expectedReviewedLayerCount(summary: [String: Any]) -> Int? {
        if let layerCount = intValue(summary["layer_count"] ?? summary["layerCount"]),
           layerCount > 0 {
            return layerCount
        }
        if let layers = summary["layers"] as? [String: Any], !layers.isEmpty {
            return layers.count
        }
        let candidatePaths = [
            stringValue(summary["source_model_path"] ?? summary["sourceModelPath"]),
            stringValue(summary["source_model"] ?? summary["sourceModel"]),
            stringValue(summary["pruned_source"] ?? summary["prunedSource"])
        ].compactMap { $0 }
        for path in candidatePaths {
            if let layerCount = configLayerCount(modelPath: path) {
                return layerCount
            }
        }
        return nil
    }

    private static func configLayerCount(modelPath: String) -> Int? {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let config = readJSONObject(configURL) else { return nil }
        let textConfig = config["text_config"] as? [String: Any] ?? config
        let layerCount = intValue(
            textConfig["num_hidden_layers"] ?? textConfig["n_layer"] ?? textConfig["num_layers"]
        )
        guard let layerCount, layerCount > 0 else { return nil }
        return layerCount
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func loadExpertLabSuitePrompts(from url: URL) -> [ExpertLabSmokePrompt]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let prompts = text.split(whereSeparator: \.isNewline).compactMap { line -> ExpertLabSmokePrompt? in
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = stringValue(json["id"] ?? json["prompt_id"]),
                  let promptText = stringValue(json["text"] ?? json["prompt"]),
                  !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return nil
            }
            let maxTokens = intValue(json["max_new_tokens"] ?? json["maxTokens"])
            let temperature = doubleValue(json["temperature"])
            return ExpertLabSmokePrompt(id: id, text: promptText, maxTokens: maxTokens, temperature: temperature)
        }
        return prompts.isEmpty ? nil : prompts
    }

    private static func generateSmoke(
        runner: InferenceRunner,
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        timeoutSeconds: Double
    ) async throws -> InferenceResult {
        try await withThrowingTaskGroup(of: InferenceResult.self) { group in
            group.addTask {
                try await runner.generate(
                    prompt: prompt,
                    maxTokens: maxTokens,
                    temperature: temperature,
                    noThinking: true
                )
            }
            group.addTask {
                try await Task.sleep(for: .seconds(timeoutSeconds))
                await runner.cancel()
                throw InferenceError(message: "smoke prompt timed out after \(Int(timeoutSeconds))s", code: -3)
            }
            guard let result = try await group.next() else {
                throw InferenceError(message: "smoke prompt did not run", code: -4)
            }
            group.cancelAll()
            return result
        }
    }

    /// Default wall-time budget for `jang validate`. Validation is file-inspection
    /// only (no model load, no inference) — it should complete in ≤5 seconds
    /// on any reasonable machine. 60s is a 10× safety margin that still caps
    /// the worst case so a hung Python subprocess can't stall VerifyStep
    /// indefinitely if the user leaves the wizard open in the background.
    /// Exposed as a parameter for tests; prod callers use the default.
    static let defaultValidateTimeoutSeconds: Double = 60

    /// Run `jang validate` and return whether it exited 0 within the timeout.
    /// M42 (iter 19): previously used `proc.waitUntilExit()` which blocks the
    /// calling thread indefinitely if the subprocess hangs. Now uses the same
    /// actor-friendly pattern as PythonRunner/InferenceRunner (iter 3): a
    /// CheckedContinuation tied to `terminationHandler`, plus a Task.sleep
    /// timeout race that SIGTERMs on expiry.
    ///
    /// M158 (iter 81): two silent-failure bugs audited + fixed:
    ///
    ///   1. Pipe-fill hang. The old code wired `Pipe()` to stdout + stderr
    ///      without ever reading them. macOS pipe buffers are ~64 KB; if the
    ///      subprocess writes more than that before exiting, the write blocks,
    ///      the process never exits, and we report validation as failed even
    ///      though the validator never got to say yes or no. `jang validate`
    ///      usually stays well under 64 KB, but a traceback on top of a deep
    ///      shard listing is easy to push over. Swapped to
    ///      `FileHandle.nullDevice` — we don't surface subprocess output
    ///      anywhere, so discarding is the right primitive.
    ///
    ///   2. terminationHandler wired AFTER run(). A fast-exiting subprocess
    ///      could terminate in the microsecond window between `try proc.run()`
    ///      returning and the handler assignment, and Foundation would never
    ///      invoke the handler on a process that already terminated. Result:
    ///      continuation deadlocks until the 60 s timeout → false. Reordered
    ///      so the handler is wired before `run()`, which closes the window
    ///      entirely.
    ///
    /// - Parameter executableOverride: test-only hook, mirrors the
    ///   InferenceRunner / PythonCLIInvoker pattern (iter-32 M100 / iter-76
    ///   M153). Production passes nil (defaults to BundleResolver.pythonExecutable);
    ///   tests supply a shell script that exercises the pipe-fill and
    ///   fast-exit paths without spinning up a real Python runtime.
    static func runJangValidate(outputDir: URL,
                                timeoutSeconds: Double = defaultValidateTimeoutSeconds,
                                executableOverride: URL? = nil) async -> Bool {
        let proc = Process()
        proc.executableURL = executableOverride ?? BundleResolver.pythonExecutable
        proc.arguments = ["-m", "jang_tools", "validate", outputDir.path]
        // M158: discard subprocess output. Capturing into Pipe() with no
        // reader deadlocks the subprocess once the 64 KB kernel buffer fills.
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice

        // Race the natural exit against a timeout sleep. First winner resolves
        // the continuation; if the timeout wins, SIGTERM the subprocess + a
        // 3-second SIGKILL escalation so a truly deadlocked child still dies.
        //
        // M101 (iter 33): wrap in withTaskCancellationHandler so a cancelled
        // consumer Task (e.g., user navigating away from VerifyStep mid-run)
        // also tears down the subprocess instead of waiting for the 60s
        // default timeout. See iter-32 cross-layer cancel sweep.
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
                let resolver = BoolContinuationResolver()

                // M158: wire terminationHandler BEFORE run() so a subprocess
                // that exits immediately still fires the handler. Setting it
                // post-run() races the process's own termination.
                proc.terminationHandler = { p in
                    resolver.resumeOnce(cont, returning: p.terminationStatus == 0)
                }

                do {
                    try proc.run()
                } catch {
                    resolver.resumeOnce(cont, returning: false)
                    return
                }

                Task.detached {
                    try? await Task.sleep(for: .seconds(timeoutSeconds))
                    resolver.resumeOnce(cont, returning: false) {
                        // SIGTERM + 3s SIGKILL escalation, same pattern as PythonRunner.
                        if proc.isRunning { proc.terminate() }
                        Task.detached {
                            try? await Task.sleep(for: .seconds(3))
                            if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
                        }
                    }
                }
            }
        } onCancel: {
            // On consumer-Task cancel, SIGTERM the subprocess. The
            // terminationHandler will then resolve the continuation with
            // the terminated exit code (→ returns false, which is fine:
            // cancelled verifications are treated as "did not succeed").
            if proc.isRunning { proc.terminate() }
        }
    }
}

private final class BoolContinuationResolver: @unchecked Sendable {
    private let lock = DispatchQueue(label: "PostConvertVerifier.BoolContinuationResolver")
    private var resolved = false

    func resumeOnce(
        _ continuation: CheckedContinuation<Bool, Never>,
        returning value: Bool,
        beforeResume: (() -> Void)? = nil
    ) {
        lock.sync {
            guard !resolved else { return }
            resolved = true
            beforeResume?()
            continuation.resume(returning: value)
        }
    }
}
