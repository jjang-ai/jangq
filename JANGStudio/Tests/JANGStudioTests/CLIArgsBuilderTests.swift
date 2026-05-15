// JANGStudio/Tests/JANGStudioTests/CLIArgsBuilderTests.swift
import XCTest
@testable import JANGStudio

@MainActor
final class CLIArgsBuilderTests: XCTestCase {
    private func plan(src: String = "/tmp/src",
                      out: String = "/tmp/out",
                      family: Family = .jang,
                      profile: String = "JANG_4K",
                      method: QuantMethod = .mse,
                      hadamard: Bool = false,
                      modelType: String = "qwen3_5_moe") -> ConversionPlan {
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: src)
        p.outputURL = URL(fileURLWithPath: out)
        p.family = family
        p.profile = profile
        p.method = method
        p.hadamard = hadamard
        p.detected = .init(modelType: modelType, isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        return p
    }

    func test_emptyPlanReturnsNoArgs() {
        let p = ConversionPlan()   // no URLs
        XCTAssertEqual(CLIArgsBuilder.args(for: p), [])
    }

    func test_everyJANGProfileRoutesToJangConvert() {
        let profiles = ["JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
                        "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
                        "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
                        "JANG_5K", "JANG_6K", "JANG_6M"]
        for prof in profiles {
            let args = CLIArgsBuilder.args(for: plan(profile: prof))
            XCTAssertEqual(Array(args.prefix(2)), ["-m", "jang_tools"], "profile=\(prof) should route to jang_tools, got \(args.prefix(2))")
            XCTAssertEqual(args[2], "convert", "profile=\(prof) should use `convert` subcommand")
            XCTAssertTrue(args.contains("-p"), "profile=\(prof) missing -p flag")
            XCTAssertEqual(args[args.firstIndex(of: "-p")! + 1], prof,
                           "profile=\(prof) flag value should match input")
            XCTAssertTrue(args.contains("--progress=json"), "profile=\(prof) missing --progress=json")
            XCTAssertTrue(args.contains("--quiet-text"), "profile=\(prof) missing --quiet-text")
        }
    }

    func test_everyJANGTQProfileRoutesCorrectlyByArch() {
        let qwenExpected = "jang_tools.convert_qwen35_jangtq"
        let mmExpected = "jang_tools.convert_minimax_jangtq"
        for prof in ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"] {
            let qwenArgs = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: prof, modelType: "qwen3_5_moe"))
            XCTAssertEqual(qwenArgs[1], qwenExpected, "qwen JANGTQ \(prof) routes wrong")
            XCTAssertTrue(qwenArgs.contains(prof))

            let mmArgs = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: prof, modelType: "minimax_m2"))
            XCTAssertEqual(mmArgs[1], mmExpected, "minimax JANGTQ \(prof) routes wrong")
            XCTAssertTrue(mmArgs.contains(prof))
        }
    }

    func test_newJANGTQFamiliesRouteToTheirRealConverters() {
        let cases: [(String, String, String)] = [
            ("hy_v3", "JANGTQ_K", "jang_tools.convert_hy3_jangtq"),
            ("bailing_hybrid", "JANGTQ2", "jang_tools.convert_ling_jangtq"),
            ("nemotron_h", "JANGTQ4", "jang_tools.convert_nemotron_jangtq"),
            ("zaya", "JANGTQ_K", "jang_tools.convert_zaya_jangtq"),
            ("zaya1_vl", "JANGTQ4", "jang_tools.convert_zaya1_vl_jangtq"),
            ("laguna", "JANGTQ2", "jang_tools.convert_laguna_jangtq"),
            ("mistral3", "JANGTQ3", "jang_tools.convert_mistral3_jangtq"),
        ]
        for (modelType, profile, module) in cases {
            let args = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: profile, modelType: modelType))
            XCTAssertEqual(args[1], module, "\(modelType)/\(profile) routes wrong: \(args)")
            XCTAssertTrue(args.contains("--progress=json"), "\(modelType) should keep JSON progress")
            XCTAssertTrue(args.contains(profile), "\(modelType)/\(profile): profile not in args")
        }
    }

    func test_flagStyleJANGTQConvertersRouteCorrectly() {
        let dsv4 = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: "JANGTQ_K", modelType: "deepseek_v4"))
        XCTAssertEqual(dsv4[1], "jang_tools.dsv4.convert_dsv4_jangtq")
        XCTAssertTrue(dsv4.contains("--src"))
        XCTAssertTrue(dsv4.contains("--dst"))
        XCTAssertEqual(dsv4[dsv4.firstIndex(of: "--profile")! + 1], "2")
        XCTAssertEqual(dsv4[dsv4.firstIndex(of: "--variant")! + 1], "K")

        let kimi = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: "JANGTQ_K", modelType: "kimi_k25"))
        XCTAssertEqual(kimi[1], "jang_tools.kimi_prune.convert_kimi_jangtq")
        XCTAssertTrue(kimi.contains("--src"))
        XCTAssertTrue(kimi.contains("--dst"))
        XCTAssertEqual(kimi[kimi.firstIndex(of: "--profile")! + 1], "JANGTQ_K")
    }

    func test_hadamardFlagOnlyAddedForJANGWhenRequested() {
        let on = CLIArgsBuilder.args(for: plan(profile: "JANG_4K", hadamard: true))
        XCTAssertTrue(on.contains("--hadamard"), "hadamard=true should add --hadamard")
        let off = CLIArgsBuilder.args(for: plan(profile: "JANG_4K", hadamard: false))
        XCTAssertFalse(off.contains("--hadamard"), "hadamard=false should NOT add --hadamard")
        let jangtqOn = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: "JANGTQ2", hadamard: true))
        XCTAssertFalse(jangtqOn.contains("--hadamard"), "JANGTQ never passes --hadamard (converter has no such flag)")
    }

    func test_methodFlagPropagatesToJANGOnly() {
        for m in [QuantMethod.mse, .rtn, .mseAll] {
            let args = CLIArgsBuilder.args(for: plan(method: m))
            XCTAssertTrue(args.contains("-m"), "method=\(m) missing -m flag")
            let idxs = args.enumerated().compactMap { $0.element == "-m" ? $0.offset : nil }
            // There are TWO -m flags in the jang convert args: one for python's module invocation
            // (`-m jang_tools`), one for the convert method (`-m mse|rtn|mseAll`). The second one
            // should carry the method rawValue.
            XCTAssertEqual(idxs.count, 2, "expected two -m flags in jang args, got \(idxs.count)")
            XCTAssertEqual(args[idxs[1] + 1], m.rawValue)
        }
    }

    func test_jangtqDoesNotPassMethodOrHadamard() {
        let args = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: "JANGTQ3",
                                                 method: .rtn, hadamard: true,
                                                 modelType: "minimax_m2"))
        XCTAssertFalse(args.contains("--hadamard"))
        // The only "-m" in a JANGTQ invocation is the python module flag; no second -m for method.
        let mFlags = args.filter { $0 == "-m" }
        XCTAssertEqual(mFlags.count, 1)
    }

    func test_sourceAndOutputPathsFlowThrough() {
        let p = plan(src: "/Volumes/LLMs/Qwen3.6", out: "/Volumes/Out/Qwen3.6-JANG_2L")
        let args = CLIArgsBuilder.args(for: p)
        XCTAssertTrue(args.contains("/Volumes/LLMs/Qwen3.6"))
        XCTAssertTrue(args.contains("/Volumes/Out/Qwen3.6-JANG_2L"))
    }

    func test_unknownJANGTQArchReturnsNoArgs() {
        // If preflight somehow lets an unsupported model reach Run, do not run
        // a random converter against it.
        let args = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: "JANGTQ2", modelType: "some_other_moe"))
        XCTAssertEqual(args, [])
    }

    // MARK: - Advanced overrides propagate to the JANG CLI

    func test_forceBlockSizeFlagsAppendWhenSet() {
        let p = plan(profile: "JANG_4K")
        p.overrides.forceBlockSize = 128
        let args = CLIArgsBuilder.args(for: p)
        XCTAssertTrue(args.contains("-b"), "expected -b flag when forceBlockSize is set")
        let idx = args.firstIndex(of: "-b")!
        XCTAssertEqual(args[idx + 1], "128")
    }

    func test_forceBlockSizeOmittedWhenNilOrZero() {
        let args = CLIArgsBuilder.args(for: plan(profile: "JANG_4K"))
        XCTAssertFalse(args.contains("-b"), "no -b flag when forceBlockSize is nil")
    }

    func test_forceDtypeFlagPropagates() {
        for (dtype, alias) in [(SourceDtype.bf16, "bf16"),
                               (SourceDtype.fp16, "fp16"),
                               (SourceDtype.fp8, "fp8")] {
            let p = plan(profile: "JANG_2L")
            p.overrides.forceDtype = dtype
            let args = CLIArgsBuilder.args(for: p)
            XCTAssertTrue(args.contains("--force-dtype"),
                          "expected --force-dtype flag when forceDtype=\(dtype)")
            let idx = args.firstIndex(of: "--force-dtype")!
            XCTAssertEqual(args[idx + 1], alias)
        }
    }

    func test_forceDtypeUnknownAndJangV2OmitFlag() {
        for dtype in [SourceDtype.unknown, .jangV2] {
            let p = plan()
            p.overrides.forceDtype = dtype
            let args = CLIArgsBuilder.args(for: p)
            XCTAssertFalse(args.contains("--force-dtype"),
                           "forceDtype=\(dtype) should not emit a flag")
        }
    }

    func test_jangtqIgnoresAdvancedOverridesForNow() {
        let p = plan(family: .jangtq, profile: "JANGTQ2", modelType: "minimax_m2")
        p.overrides.forceBlockSize = 128
        p.overrides.forceDtype = .fp8
        let args = CLIArgsBuilder.args(for: p)
        // JANGTQ convert scripts take positional args only (SRC OUT PROFILE);
        // they don't accept -b or --force-dtype yet. Extending those scripts
        // is a separate change.
        XCTAssertFalse(args.contains("-b"))
        XCTAssertFalse(args.contains("--force-dtype"))
    }
}
