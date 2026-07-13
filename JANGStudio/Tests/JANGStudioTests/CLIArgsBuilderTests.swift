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
            XCTAssertEqual(args[4], "convert", "profile=\(prof) should use `convert` subcommand after global flags")
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
        for prof in ["JANGTQ2", "JANGTQ3", "JANGTQ4"] {
            let qwenArgs = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: prof, modelType: "qwen3_5_moe"))
            XCTAssertEqual(qwenArgs[1], qwenExpected, "qwen JANGTQ \(prof) routes wrong")
            XCTAssertTrue(qwenArgs.contains(prof))

            let qwenTextArgs = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: prof, modelType: "qwen3_5_moe_text"))
            XCTAssertEqual(qwenTextArgs[1], qwenExpected, "qwen text JANGTQ \(prof) routes wrong")
            XCTAssertTrue(qwenTextArgs.contains(prof))

            let mmArgs = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: prof, modelType: "minimax_m2"))
            XCTAssertEqual(mmArgs[1], mmExpected, "minimax JANGTQ \(prof) routes wrong")
            XCTAssertTrue(mmArgs.contains(prof))
        }
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

    func test_jangGlobalProgressFlagsPrecedeConvertSubcommand() {
        let args = CLIArgsBuilder.args(for: plan())
        XCTAssertEqual(Array(args.prefix(5)), ["-m", "jang_tools", "--progress=json", "--quiet-text", "convert"])
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

    func test_unknownJANGTQArchReturnsEmptyArgs() {
        // Hard-fail: unmapped architectures must not silently route to the Qwen
        // converter (pre-PR1 default). Preflight + RunStep refuse to start.
        let args = CLIArgsBuilder.args(for: plan(family: .jangtq, profile: "JANGTQ2", modelType: "some_other_moe"))
        XCTAssertEqual(args, [], "unmapped JANGTQ arch must return empty argv, got \(args)")
        XCTAssertNil(CLIArgsBuilder.jangtqModule(for: plan(family: .jangtq, profile: "JANGTQ2", modelType: "some_other_moe")))
    }

    func test_unknownModelType_nilTextModelType_returnsEmptyArgs() {
        let p = plan(family: .jangtq, profile: "JANGTQ2", modelType: "some_other_moe")
        // Explicit: no textModelType fallback available.
        p.detected = .init(modelType: "some_other_moe", isMoE: true, numExperts: 64, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1,
                           textModelType: nil)
        XCTAssertEqual(CLIArgsBuilder.args(for: p), [])
        XCTAssertNil(CLIArgsBuilder.jangtqModule(for: p))
    }

    func test_jangtqModule_usesTextModelTypeForWrapper() {
        // VL wrapper: outer modelType is not on the module map, but textModelType is.
        // Same fixture shape as ConversionPlanTests wrapper allow-list case.
        let p = plan(family: .jangtq, profile: "JANGTQ3", modelType: "qwen3_5_moe_wrapper")
        p.detected = .init(modelType: "qwen3_5_moe_wrapper", isMoE: true, numExperts: 256, isVL: true,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1,
                           textModelType: "qwen3_5_moe_text")
        let args = CLIArgsBuilder.args(for: p)
        XCTAssertFalse(args.isEmpty, "wrapper + textModelType should produce non-empty argv")
        XCTAssertEqual(args[1], "jang_tools.convert_qwen35_jangtq")
        XCTAssertEqual(CLIArgsBuilder.jangtqModule(for: p), "jang_tools.convert_qwen35_jangtq")
    }

    func test_jangtqModule_nilWhenDetectedMissing() {
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: "/tmp/src")
        p.outputURL = URL(fileURLWithPath: "/tmp/out")
        p.family = .jangtq
        p.profile = "JANGTQ2"
        p.detected = nil
        XCTAssertNil(CLIArgsBuilder.jangtqModule(for: p))
        XCTAssertEqual(CLIArgsBuilder.args(for: p), [])
    }

    func test_failureReason_distinguishesMissingURLsVsUnmappedJANGTQ() {
        let missingURLs = ConversionPlan()
        missingURLs.family = .jangtq
        let missingReason = CLIArgsBuilder.failureReason(for: missingURLs)
        XCTAssertNotNil(missingReason)
        XCTAssertTrue(missingReason!.contains("sourceURL and outputURL"),
                      "missing URLs reason should mention URLs, got: \(missingReason!)")

        let unmapped = plan(family: .jangtq, profile: "JANGTQ2", modelType: "some_other_moe")
        let unmappedReason = CLIArgsBuilder.failureReason(for: unmapped)
        XCTAssertNotNil(unmappedReason)
        XCTAssertTrue(unmappedReason!.contains("no module mapping"),
                      "unmapped reason should mention module mapping, got: \(unmappedReason!)")
        XCTAssertTrue(unmappedReason!.contains("some_other_moe"))

        let ok = plan(family: .jangtq, profile: "JANGTQ2", modelType: "qwen3_5_moe")
        XCTAssertNil(CLIArgsBuilder.failureReason(for: ok),
                     "mapped JANGTQ with URLs should have no failure reason")

        let jangOK = plan(family: .jang, profile: "JANG_4K", modelType: "llama")
        XCTAssertNil(CLIArgsBuilder.failureReason(for: jangOK),
                     "JANG family never fails on module map")
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
