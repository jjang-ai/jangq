// JANGStudio/Tests/JANGStudioTests/CoverageMatrixTests.swift
import XCTest
@testable import JANGStudio

/// Full cartesian audit of (profile x architecture class x dtype x VL variant)
/// through preflight, CLIArgsBuilder, and PostConvertVerifier.
@MainActor
final class CoverageMatrixTests: XCTestCase {
    nonisolated(unsafe) private var tmp: URL!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cov-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    // 12 arch classes covering: dense, MoE (8/256 experts), VL image, VL video, every dtype,
    // JANGTQ-whitelisted + non-whitelisted, MiniMax-class (custom .py required).
    private static let archClasses: [(model: String, experts: Int, isVL: Bool, isVideoVL: Bool,
                                      dtype: SourceDtype, label: String)] = [
        ("llama",           0,   false, false, .bf16, "dense llama BF16"),
        ("llama",           0,   false, false, .fp16, "dense llama FP16"),
        ("qwen3_5_moe",     8,   false, false, .bf16, "qwen 8 experts BF16"),
        ("qwen3_5_moe",     256, false, false, .bf16, "qwen 256 experts BF16"),
        ("qwen3_5_moe",     256, true,  false, .bf16, "qwen image-VL BF16"),
        ("qwen3_5_moe",     256, true,  true,  .bf16, "qwen video-VL BF16"),
        ("qwen3_5_moe",     256, false, false, .fp16, "qwen 256 experts FP16"),
        ("qwen3_5_moe",     256, false, false, .fp8,  "qwen 256 experts FP8"),
        ("qwen3_5_moe_text", 256, false, false, .bf16, "qwen text 256 experts BF16"),
        ("minimax_m2",      256, false, false, .fp8,  "minimax FP8"),
        ("minimax_m2",      256, false, false, .bf16, "minimax BF16"),
        ("glm_moe_dsa",     256, false, false, .fp8,  "glm FP8 (JANGTQ blocked in v1)"),
        ("deepseek_v32",    256, false, false, .bf16, "deepseek BF16"),
    ]

    private func makePlan(_ c: (model: String, experts: Int, isVL: Bool, isVideoVL: Bool,
                                dtype: SourceDtype, label: String),
                          family: Family, profile: String) throws -> ConversionPlan {
        let src = tmp.appendingPathComponent("src-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try "{\"model_type\":\"\(c.model)\",\"num_hidden_layers\":4}".write(
            to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let p = ConversionPlan()
        p.sourceURL = src
        p.outputURL = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        p.family = family
        p.profile = profile
        p.detected = .init(modelType: c.model, isMoE: c.experts > 1, numExperts: c.experts,
                            isVL: c.isVL, isVideoVL: c.isVideoVL, dtype: c.dtype,
                            totalBytes: 0, shardCount: 1)
        return p
    }

    // MARK: - Preflight matrix

    func test_preflight_JANG_family_passesArchSupportCheckForEveryArch() throws {
        let profiles = ["JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
                        "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
                        "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
                        "JANG_5K", "JANG_6K", "JANG_6M"]
        for arch in Self.archClasses {
            for prof in profiles {
                let p = try makePlan(arch, family: .jang, profile: prof)
                let checks = PreflightRunner().run(plan: p, capabilities: .frozen)
                let archCheck = checks.first { $0.id == .jangtqArchSupported }!
                XCTAssertEqual(archCheck.status, .pass,
                    "JANG on \(arch.label) profile=\(prof) should not be rejected by jangtq-arch check")
            }
        }
    }

    func test_preflight_JANGTQ_rejectsNonWhitelistedArchs() throws {
        let nonWhitelisted = Self.archClasses.filter {
            $0.model != "qwen3_5_moe"
                && $0.model != "qwen3_5_moe_text"
                && $0.model != "minimax_m2"
        }
        for arch in nonWhitelisted {
            for prof in ["JANGTQ2", "JANGTQ3", "JANGTQ4"] {
                let p = try makePlan(arch, family: .jangtq, profile: prof)
                let checks = PreflightRunner().run(plan: p, capabilities: .frozen)
                let archCheck = checks.first { $0.id == .jangtqArchSupported }!
                XCTAssertEqual(archCheck.status, .fail,
                    "JANGTQ on \(arch.label) profile=\(prof) must be blocked by preflight")
            }
        }
    }

    func test_preflight_JANGTQ_acceptsWhitelistedWithBF16FP16orFP8() throws {
        let whitelisted = Self.archClasses.filter {
            ($0.model == "qwen3_5_moe" || $0.model == "qwen3_5_moe_text" || $0.model == "minimax_m2")
                && ($0.dtype == .bf16 || $0.dtype == .fp16 || $0.dtype == .fp8)
        }
        for arch in whitelisted {
            for prof in ["JANGTQ2", "JANGTQ3", "JANGTQ4"] {
                let p = try makePlan(arch, family: .jangtq, profile: prof)
                let checks = PreflightRunner().run(plan: p, capabilities: .frozen)
                XCTAssertEqual(checks.first { $0.id == .jangtqArchSupported }!.status, .pass,
                    "\(arch.label)/\(prof) arch check should pass")
                XCTAssertEqual(checks.first { $0.id == .jangtqSourceDtype }!.status, .pass,
                    "\(arch.label)/\(prof) dtype check should pass")
            }
        }
    }

    // MARK: - CLI args routing

    func test_cliArgs_everyProfileEveryArchProducesValidInvocation() throws {
        let jang = ["JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
                    "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
                    "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
                    "JANG_5K", "JANG_6K", "JANG_6M"]
        let jangtq = ["JANGTQ2", "JANGTQ3", "JANGTQ4"]
        for arch in Self.archClasses {
            for prof in jang {
                let p = try makePlan(arch, family: .jang, profile: prof)
                let args = CLIArgsBuilder.args(for: p)
                XCTAssertEqual(Array(args.prefix(5)), ["-m", "jang_tools", "--progress=json", "--quiet-text", "convert"],
                    "\(arch.label)/\(prof): wrong prefix")
                XCTAssertTrue(args.contains(prof), "\(arch.label)/\(prof): profile not in args")
                XCTAssertTrue(args.contains("--progress=json"), "\(arch.label)/\(prof): missing --progress=json")
            }
            if arch.model == "qwen3_5_moe" || arch.model == "qwen3_5_moe_text" || arch.model == "minimax_m2" {
                for prof in jangtq {
                    let p = try makePlan(arch, family: .jangtq, profile: prof)
                    let args = CLIArgsBuilder.args(for: p)
                    XCTAssertTrue(args[1].contains("jangtq"),
                        "\(arch.label) profile=\(prof) should route to a JANGTQ converter, got \(args[1])")
                    XCTAssertTrue(args.contains(prof), "\(arch.label)/\(prof): profile not in args")
                }
            }
        }
    }

    // MARK: - Verifier: missing-file matrix

    func test_verifier_flagsEachMissingRequiredFile() async throws {
        let scenarios: [(remove: String, expect: VerifyID, label: String)] = [
            ("jang_config.json",             .jangConfigExists,   "missing jang_config"),
            ("tokenizer.json",               .tokenizerFiles,     "missing tokenizer.json"),
            ("tokenizer_config.json",        .tokenizerFiles,     "missing tokenizer_config.json"),
            ("special_tokens_map.json",      .tokenizerFiles,     "missing special_tokens_map.json"),
            ("model.safetensors.index.json", .shardsMatchIndex,   "missing shard index"),
        ]
        for s in scenarios {
            let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
            try Self.writeGoodFixture(at: out)
            try FileManager.default.removeItem(at: out.appendingPathComponent(s.remove))
            let plan = ConversionPlan()
            plan.outputURL = out
            plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                                  dtype: .bf16, totalBytes: 0, shardCount: 1)
            let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
            XCTAssertTrue(checks.contains { $0.id == s.expect && $0.status == .fail },
                          "scenario '\(s.label)' should trigger \(s.expect)")
        }
    }

    func test_verifier_chatTemplate_inline_OR_jinja_OR_json_allSatisfy() async throws {
        // A. neither present -> fail
        let noneURL = tmp.appendingPathComponent("out-none-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: noneURL)
        try "{\"tokenizer_class\":\"Qwen2Tokenizer\"}".write(
            to: noneURL.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        var planA = ConversionPlan()
        planA.outputURL = noneURL
        planA.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                               dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checksA = await PostConvertVerifier().run(plan: planA, skipPythonValidate: true)
        XCTAssertTrue(checksA.contains { $0.id == .chatTemplate && $0.status == .fail },
                      "no chat template should fail")

        // B. .jinja alone -> pass
        let jinjaURL = tmp.appendingPathComponent("out-jinja-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: jinjaURL)
        try "{\"tokenizer_class\":\"Qwen2Tokenizer\"}".write(
            to: jinjaURL.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        try "{% for m in messages %}{{m.content}}{% endfor %}".write(
            to: jinjaURL.appendingPathComponent("chat_template.jinja"),
            atomically: true, encoding: .utf8)
        var planB = ConversionPlan()
        planB.outputURL = jinjaURL
        planB.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                               dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checksB = await PostConvertVerifier().run(plan: planB, skipPythonValidate: true)
        XCTAssertTrue(checksB.contains { $0.id == .chatTemplate && $0.status == .pass },
                      ".jinja file should satisfy chatTemplate")

        // C. chat_template.json alone -> pass
        let jsonURL = tmp.appendingPathComponent("out-json-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: jsonURL)
        try "{\"tokenizer_class\":\"Qwen2Tokenizer\"}".write(
            to: jsonURL.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        try "{\"chat_template\":\"{% for m in messages %}{{m.content}}{% endfor %}\"}".write(
            to: jsonURL.appendingPathComponent("chat_template.json"),
            atomically: true, encoding: .utf8)
        var planC = ConversionPlan()
        planC.outputURL = jsonURL
        planC.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                               dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checksC = await PostConvertVerifier().run(plan: planC, skipPythonValidate: true)
        XCTAssertTrue(checksC.contains { $0.id == .chatTemplate && $0.status == .pass },
                      "chat_template.json alone should satisfy chatTemplate")
    }

    func test_verifier_imageVL_requiresPreprocessor() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        let plan = ConversionPlan()
        plan.outputURL = out
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: true,
                              dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        XCTAssertTrue(checks.contains { $0.id == .vlPreprocessors && $0.status == .fail },
                      "VL model without preprocessor_config.json should fail vlPreprocessors")
    }

    func test_verifier_videoVL_requiresVideoPreprocessor() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        // Write image preprocessor but NOT video preprocessor
        try "{\"image_size\":224}".write(to: out.appendingPathComponent("preprocessor_config.json"),
                                         atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.outputURL = out
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: true,
                              isVideoVL: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        XCTAssertEqual(checks.first { $0.id == .vlPreprocessors }!.status, .pass,
                       "image preprocessor present -> vlPreprocessors should pass")
        XCTAssertEqual(checks.first { $0.id == .videoPreprocessors }!.status, .fail,
                       "video preprocessor absent -> videoPreprocessors should fail")
    }

    func test_verifier_layerCountSanityFailsWhenConfigLacksLayers() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        try "{\"model_type\":\"qwen3_5_moe\"}".write(
            to: out.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.outputURL = out
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        XCTAssertEqual(checks.first { $0.id == .layerCountSane }!.status, .fail,
                       "missing num_hidden_layers should fail layerCountSane")
    }

    func test_verifier_generationConfig_isWarnNotFail() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        let plan = ConversionPlan()
        plan.outputURL = out
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let gen = checks.first { $0.id == .generationConfig }!
        XCTAssertEqual(gen.status, .warn, "missing generation_config.json should be warn not fail")
        XCTAssertFalse(gen.required, "generationConfig must be required=false")
    }

    func test_verifier_minimaxCustomPyGatedByModelType() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        let plan = ConversionPlan()
        plan.outputURL = out
        plan.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256, isVL: false,
                              dtype: .fp8, totalBytes: 0, shardCount: 1)
        let checks1 = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        XCTAssertTrue(checks1.contains { $0.id == .miniMaxCustomPy && $0.status == .fail },
                      "minimax without .py files should fail miniMaxCustomPy")

        try "# stub".write(to: out.appendingPathComponent("modeling_minimax.py"),
                          atomically: true, encoding: .utf8)
        try "# stub".write(to: out.appendingPathComponent("configuration_minimax.py"),
                          atomically: true, encoding: .utf8)
        let checks2 = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        XCTAssertTrue(checks2.contains { $0.id == .miniMaxCustomPy && $0.status == .pass },
                      "minimax with .py files should pass miniMaxCustomPy")
    }

    func test_verifier_tokenizerClassConcreteIsBlockingFailure() async throws {
        // Memory cross-ref (feedback_jang_studio_audit_coverage.md):
        // TokenizersBackend breaks Osaurus / vmlx-swift-lm hard. Upgraded in
        // Ralph iter 5 from warn-only to required=true fail so the wizard
        // blocks users from shipping a broken model.
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        try "{\"chat_template\":\"{% for m in messages %}{{m.content}}{% endfor %}\",\"tokenizer_class\":\"TokenizersBackend\"}".write(
            to: out.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.outputURL = out
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let cls = checks.first { $0.id == .tokenizerClassConcrete }!
        XCTAssertEqual(cls.status, .fail, "TokenizersBackend must fail hard (Osaurus cannot load)")
        XCTAssertTrue(cls.required, "tokenizerClassConcrete must be a required blocker")
    }

    // MARK: - Helpers

    /// Writes a minimal valid JANG output directory (config includes num_hidden_layers
    /// so the layer-count check passes by default). Mirrors Fixtures/good_output/.
    /// Deliberately omits preprocessor_config.json so VL tests can assert its absence.
    /// Deliberately omits generation_config.json so the generationConfig warn tests work.
    private static func writeGoodFixture(at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try "{\"model_type\":\"qwen3_5_moe\",\"torch_dtype\":\"bfloat16\",\"num_hidden_layers\":4,\"hidden_size\":128}".write(
            to: url.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{\"format\":\"jang\",\"format_version\":\"2.0\",\"capabilities\":{\"arch\":\"qwen3_5_moe\"},\"quantization\":{\"bit_widths_used\":[4],\"block_size\":64}}".write(
            to: url.appendingPathComponent("jang_config.json"), atomically: true, encoding: .utf8)
        try "{\"model\":{\"type\":\"BPE\"}}".write(
            to: url.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try "{\"chat_template\":\"{% for m in messages %}{{m.content}}{% endfor %}\",\"tokenizer_class\":\"Qwen2Tokenizer\"}".write(
            to: url.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        try "{\"bos_token\":\"<s>\",\"eos_token\":\"</s>\"}".write(
            to: url.appendingPathComponent("special_tokens_map.json"), atomically: true, encoding: .utf8)
        try "{\"weight_map\":{\"a\":\"model-00001-of-00001.safetensors\"}}".write(
            to: url.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        FileManager.default.createFile(atPath: url.appendingPathComponent("model-00001-of-00001.safetensors").path,
                                       contents: Data())
    }
}
