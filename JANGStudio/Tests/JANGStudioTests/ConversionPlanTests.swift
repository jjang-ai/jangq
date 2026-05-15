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
        let whitelist = Capabilities.frozen.jangtqWhitelist

        p.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        p.detected = .init(modelType: "deepseek_v4", isMoE: true, numExperts: 256, isVL: false,
                           isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed(for: whitelist))

        // GLM remains blocked until there is a proven GLM 5.1 runtime path.
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

    func test_persistRestore_roundtrip() throws {
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: "/tmp/model")
        p.outputURL = URL(fileURLWithPath: "/tmp/out")
        p.profile = "JANG_2L"
        p.family = .jang
        let data = try p.encodeForDefaults()
        let r = try ConversionPlan.decodeFromDefaults(data)
        XCTAssertEqual(r.sourceURL, p.sourceURL)
        XCTAssertEqual(r.outputURL, p.outputURL)
        XCTAssertEqual(r.profile, "JANG_2L")
    }
}
