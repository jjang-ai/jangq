// JANGStudio/Tests/JANGStudioTests/ConversionPlanTests.swift
import XCTest
@testable import JANGStudio

final class ConversionPlanTests: XCTestCase {
    func test_defaults() {
        let plan = ConversionPlan()
        XCTAssertNil(plan.sourceURL)
        XCTAssertEqual(plan.family, .jang)
        XCTAssertEqual(plan.profile, "JANG_4K")
        XCTAssertEqual(plan.method, .mse)
        XCTAssertFalse(plan.hadamard)
        XCTAssertEqual(plan.run, .idle)
    }

    func test_isStep1Complete_requiresSourceAndDetection() {
        let p = ConversionPlan()
        XCTAssertFalse(p.isStep1Complete)
        p.sourceURL = URL(fileURLWithPath: "/tmp/x")
        XCTAssertFalse(p.isStep1Complete)
        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isStep1Complete)
    }

    func test_isJANGTQAllowed_matrix() {
        let p = ConversionPlan()
        let whitelist = ["qwen3_5_moe", "qwen3_5_moe_text", "minimax_m2"]

        p.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "qwen3_5_moe_text", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "qwen3_5_moe_wrapper", isMoE: true, numExperts: 256, isVL: true,
                           isVideoVL: true, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1,
                           textModelType: "qwen3_5_moe_text")
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        // GLM deferred to v1.1
        p.detected = .init(modelType: "glm_moe_dsa", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed(for: whitelist))
    }

    // MARK: - Iter 10: M62 — applyDefaults(from:) wires settings to wizard init

    @MainActor
    func test_applyDefaults_seeds_profile_family_method_hadamard() {
        let s = AppSettings()
        s.defaultProfile = "JANG_2L"
        s.defaultFamily = "jangtq"
        s.defaultMethod = "rtn"
        s.defaultHadamardEnabled = true
        let p = ConversionPlan()
        p.applyDefaults(from: s)
        XCTAssertEqual(p.profile, "JANG_2L")
        XCTAssertEqual(p.family, .jangtq)
        XCTAssertEqual(p.method, .rtn)
        XCTAssertTrue(p.hadamard)
    }

    @MainActor
    func test_applyDefaults_ignores_empty_profile() {
        let s = AppSettings()
        s.defaultProfile = ""    // corrupted/first-launch state
        s.defaultFamily = "jang"
        s.defaultMethod = "mse"
        let p = ConversionPlan()
        p.profile = "JANG_4K"
        p.applyDefaults(from: s)
        // Empty settings value must NOT overwrite — user would lose their
        // in-memory profile selection.
        XCTAssertEqual(p.profile, "JANG_4K")
    }

    @MainActor
    func test_applyDefaults_ignores_unknown_method() {
        let s = AppSettings()
        s.defaultMethod = "bogus-method-xyz"
        let p = ConversionPlan()
        p.method = .mse    // starting point
        p.applyDefaults(from: s)
        // Unknown method string must NOT coerce to something else; current
        // method is preserved.
        XCTAssertEqual(p.method, .mse)
    }

    @MainActor
    func test_applyDefaults_accepts_mse_all_aliases() {
        // Swift enum is `mseAll`; settings may save "mse-all" or "mseall".
        // Both aliases must map to the same enum case.
        let p = ConversionPlan()
        let s1 = AppSettings(); s1.defaultMethod = "mse-all"
        p.applyDefaults(from: s1)
        XCTAssertEqual(p.method, .mseAll)

        let s2 = AppSettings(); s2.defaultMethod = "mseall"
        p.applyDefaults(from: s2)
        XCTAssertEqual(p.method, .mseAll)

        let s3 = AppSettings(); s3.defaultMethod = "mse_all"
        p.applyDefaults(from: s3)
        XCTAssertEqual(p.method, .mseAll)
    }

    @MainActor
    func test_applyDefaults_preserves_per_conversion_state() {
        // The fields that represent per-conversion STATE (not defaults)
        // must not be clobbered by applyDefaults. If someone accidentally
        // extended applyDefaults to touch sourceURL / detected / outputURL /
        // run, the user would lose in-flight work on "Convert another".
        let s = AppSettings()
        s.defaultProfile = "JANG_2L"
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: "/src")
        p.outputURL = URL(fileURLWithPath: "/out")
        p.run = .running
        p.applyDefaults(from: s)
        XCTAssertEqual(p.sourceURL?.path, "/src")
        XCTAssertEqual(p.outputURL?.path, "/out")
        XCTAssertEqual(p.run, .running)
    }

    func test_adoptReviewedPrunedSource_prefersPrunedSourcePlanSidecar() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversion-plan-adopt-\(UUID().uuidString)", isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let reviewBundle = root.appendingPathComponent("review-bundle", isDirectory: true)
        let reviewRun = root.appendingPathComponent("review-run", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned-source", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: reviewRun, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        let runPlan = reviewRun.appendingPathComponent("prune_plan.json")
        let prunedPlan = pruned.appendingPathComponent("prune_plan.json")
        try #"{"schema":"run-plan"}"#.write(to: runPlan, atomically: true, encoding: .utf8)
        try #"{"schema":"pruned-source-plan"}"#.write(to: prunedPlan, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.sourceURL = source
        plan.outputURL = reviewBundle
        plan.expertReviewIntent = .smartPrequantPrune
        plan.expertReviewSourceURL = source
        plan.expertReviewBundleURL = reviewBundle
        plan.expertReviewPlanURL = runPlan

        plan.workflowMode = .expertLab
        plan.adoptReviewedPrunedSource(pruned)

        XCTAssertEqual(plan.sourceURL, pruned)
        XCTAssertEqual(plan.expertReviewOriginalSourceURL, source)
        XCTAssertEqual(plan.expertReviewPrunedSourceURL, pruned)
        XCTAssertEqual(plan.expertReviewPrunePlanURL, prunedPlan)
        XCTAssertEqual(plan.expertReviewBundleURL, reviewBundle)
        XCTAssertNil(plan.expertReviewPlanURL)
        XCTAssertNil(plan.outputURL)
        XCTAssertEqual(plan.run, .idle)
        XCTAssertEqual(plan.workflowMode, .convert, "post-adopt final quantize is Convert path")
    }

    func test_adoptReviewedPrunedSourceRequiresPrunedSourcePlanSidecar() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversion-plan-adopt-pruned-sidecar-\(UUID().uuidString)", isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let reviewRun = root.appendingPathComponent("review-run", isDirectory: true)
        let pruned = root.appendingPathComponent("pruned-source", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: reviewRun, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pruned, withIntermediateDirectories: true)
        let runPlan = reviewRun.appendingPathComponent("prune_plan.json")
        try #"{"schema":"run-plan"}"#.write(to: runPlan, atomically: true, encoding: .utf8)

        let plan = ConversionPlan()
        plan.expertReviewSourceURL = source
        plan.expertReviewPlanURL = runPlan

        plan.adoptReviewedPrunedSource(pruned)

        XCTAssertEqual(plan.expertReviewPrunePlanURL, pruned.appendingPathComponent("prune_plan.json"))
        XCTAssertNotEqual(plan.expertReviewPrunePlanURL, runPlan)
        XCTAssertEqual(plan.expertReviewPrunedSourceURL, pruned)
    }

    func test_persistRestore_roundtrip() throws {
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: "/tmp/model")
        p.outputURL = URL(fileURLWithPath: "/tmp/out")
        p.profile = "JANG_2L"
        p.family = .jang
        p.workflowMode = .expertLab
        let data = try p.encodeForDefaults()
        let r = try ConversionPlan.decodeFromDefaults(data)
        XCTAssertEqual(r.sourceURL, p.sourceURL)
        XCTAssertEqual(r.outputURL, p.outputURL)
        XCTAssertEqual(r.profile, "JANG_2L")
        XCTAssertEqual(r.workflowMode, .expertLab)
    }

    func test_workflowMode_migration_inProgressSession_upgradesToExpertLab() throws {
        // Missing workflowMode key + smartPrequantPrune → expertLab.
        // Omit `overrides` entirely so ArchitectureOverrides uses its default init.
        let json = """
        {
          "expertReviewIntent": "smartPrequantPrune",
          "expertReviewSourceURL": "file:///tmp/src",
          "profile": "JANG_4K",
          "family": "jang",
          "method": "mse",
          "hadamard": false
        }
        """
        let plan = try ConversionPlan.decodeFromDefaults(Data(json.utf8))
        XCTAssertEqual(plan.workflowMode, .expertLab)
    }

    func test_workflowMode_migration_postAdoptPrunedOnly_staysConvert() throws {
        // Only pruned-source rails, no in-progress session → convert.
        let json = """
        {
          "expertReviewPrunedSourceURL": "file:///tmp/pruned",
          "profile": "JANG_4K",
          "family": "jang",
          "method": "mse",
          "hadamard": false
        }
        """
        let plan = try ConversionPlan.decodeFromDefaults(Data(json.utf8))
        XCTAssertEqual(plan.workflowMode, .convert)
    }

    func test_workflowMode_heal_explicitConvertWithInProgressSession() throws {
        // Explicit convert + in-progress session fields → heal to expertLab.
        let json = """
        {
          "workflowMode": "convert",
          "expertReviewIntent": "smartPrequantPrune",
          "expertReviewSourceURL": "file:///tmp/src",
          "profile": "JANG_4K",
          "family": "jang",
          "method": "mse",
          "hadamard": false
        }
        """
        let plan = try ConversionPlan.decodeFromDefaults(Data(json.utf8))
        XCTAssertEqual(plan.workflowMode, .expertLab)
    }

    func test_workflowMode_explicitConvertWithoutSession_honored() throws {
        let json = """
        {
          "workflowMode": "convert",
          "profile": "JANG_4K",
          "family": "jang",
          "method": "mse",
          "hadamard": false
        }
        """
        let plan = try ConversionPlan.decodeFromDefaults(Data(json.utf8))
        XCTAssertEqual(plan.workflowMode, .convert)
    }
}
