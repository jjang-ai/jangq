// JANGStudio/Tests/JANGStudioTests/PreflightRunnerTests.swift
import XCTest
@testable import JANGStudio

final class PreflightRunnerTests: XCTestCase {
    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    func test_missingSourceDirFails() {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/tmp/definitely_missing_xyz")
        plan.outputURL = tmp
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .sourceReadable && $0.status == .fail })
    }

    func test_jangtqOnLlamaFails() throws {
        let src = tmp.appendingPathComponent("src"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .jangtqArchSupported && $0.status == .fail })
    }

    func test_jangtqProfileUnsupportedForFamilyFails() throws {
        let src = tmp.appendingPathComponent("src-zaya")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"zaya"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "zaya", isMoE: true, numExperts: 16, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        plan.family = .jangtq
        plan.profile = "JANGTQ3"
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .jangtqProfileSupported && $0.status == .fail })
    }

    func test_outputSameAsSourceFails() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src   // same!
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .outputUsable && $0.status == .fail })
    }

    // MARK: - M139 (iter 61): reject nested src/dst
    //
    // Pre-iter-61 outputUsable only rejected `dst == src`. A user picking
    // output INSIDE the source tree (e.g., src=/models/foo, dst=/models/foo/out)
    // passed preflight, let convert write shards into a subdir of the source.
    // Confusing + risky if any future cleanup pass rglobs.

    func test_outputInsideSourceFails() throws {
        let src = tmp.appendingPathComponent("foo")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src.appendingPathComponent("out")   // nested inside src
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let failure = checks.first { $0.id == .outputUsable && $0.status == .fail }
        XCTAssertNotNil(failure, "outputUsable should fail when output is inside source")
        XCTAssertEqual(failure?.hint, "Output cannot be inside the source folder")
    }

    func test_sourceInsideOutputFails() throws {
        // Symmetric: user picks output as the parent of source.
        let parent = tmp.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let src = parent.appendingPathComponent("hf-model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = parent   // source nested inside output
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let failure = checks.first { $0.id == .outputUsable && $0.status == .fail }
        XCTAssertNotNil(failure, "outputUsable should fail when source is inside output")
        XCTAssertEqual(failure?.hint, "Source cannot be inside the output folder")
    }

    func test_siblingPrefixPathsDoNotTrigger() throws {
        // Regression: `/a/b` must NOT be rejected as "inside /a/bc". The
        // check uses path + "/" specifically to prevent this.
        let srcParent = tmp.appendingPathComponent("abc")
        try FileManager.default.createDirectory(at: srcParent, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: srcParent.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = srcParent
        plan.outputURL = tmp.appendingPathComponent("abcd")   // sibling with shared prefix
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let found = checks.first { $0.id == .outputUsable }
        XCTAssertEqual(found?.status, .pass,
            "Sibling directories with a shared string prefix (abc vs abcd) must not trigger the nested-path check. Got \(found?.hint ?? "nil")")
    }

    // MARK: - M140 (iter 62): symmetric to M131 — preflight must detect
    // 512+ expert MoE dynamically, not just via a hardcoded name list.
    //
    // Pre-iter-62 check passed on ANY model_type not in the capabilities
    // `knownExpert512Types` hardcoded list. A future 512-expert
    // qwen3_5_moe or deepseek_v3 variant would skip the warning while the
    // underlying recommend.py-side already forces bfloat16 for the same
    // dynamic reason. Symmetric bug across the boundary; fix the preflight
    // to match.

    func test_bf16Warning_fires_on_dynamic_512_experts() throws {
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        // 512-expert qwen3_5_moe — NOT in the default .frozen whitelist
        // `knownExpert512Types: ["minimax_m2", "glm_moe_dsa"]`, but the
        // num_experts count should trigger the dynamic check.
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 512, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16   // user forced fp16 — should warn
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .warn,
            "512-expert qwen3_5_moe + forced fp16 must warn about bfloat16. Got status=\(String(describing: bf?.status)).")
        XCTAssertNotNil(bf?.hint)
        XCTAssertTrue(bf?.hint?.contains("512 experts") ?? false,
            "Hint must include the dynamic expert count so the user understands what triggered the warning. Got hint=\(bf?.hint ?? "nil")")
    }

    func test_bf16Warning_still_fires_for_named_whitelist_types() throws {
        // Regression guard: the named-family path must keep working after
        // iter-62's dynamic extension. minimax_m2 is in the frozen
        // whitelist — should warn on fp16 override.
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"minimax_m2"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        // Named whitelist entry, but with numExperts unknown (0).
        plan.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 0, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .fp8, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .warn)
    }

    func test_bf16Warning_skips_small_moe() throws {
        // Regression guard: 256-expert qwen3_5_moe must stay passing — no
        // over-warn on smaller MoEs.
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .pass,
            "256-expert qwen3_5_moe must not fire the 512+ warning.")
    }

    // MARK: - M141 (iter 63): diskSpace preflight actually gates
    //
    // Pre-iter-63 PreflightRunner.run always passed `estimated: 0` to the
    // diskSpace check, which short-circuited to `.pass` unconditionally.
    // User with near-full disk got no warning — convert started, filled
    // the disk, crashed mid-shard. Iter 63 wires profile-aware estimation
    // via `estimateOutputBytes(plan:, profiles:)` using the same formula
    // as jang_tools/estimate_model.predict.

    func test_estimateOutputBytes_scales_by_profile_avgBits() {
        // 100 GB bf16 source × (4/16) × 1.05 = 26.25 GB for JANG_4K.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        // 100 GB * 4/16 * 1.05 = 26.25 GB
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000,
                       "JANG_4K on 100 GB bf16 source should predict ~26 GB output")
    }

    func test_estimateOutputBytes_uses_real_avgBits_for_JANG_2L() {
        // JANG_2L is 2.9 bits/weight avg → 100 GB × 2.9/16 × 1.05 = 19.03 GB
        let plan = ConversionPlan()
        plan.profile = "JANG_2L"
        plan.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertGreaterThan(est, 15_000_000_000)
        XCTAssertLessThan(est, 25_000_000_000)
    }

    func test_estimateOutputBytes_returns_zero_before_source_inspected() {
        // Regression: pre-inspection state (detected=nil) must return 0 so
        // the disk-space check falls back to its `estimated <= 0` .pass
        // short-circuit — we can't gate until we know the source size.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        // detected stays nil
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 0,
            "Without a detected source size, the estimator must return 0 so preflight doesn't falsely fail.")
    }

    // MARK: - Iter 102 M175: ramAdequate sibling of M05 (ambiguous-pass sweep)

    func test_ramAdequate_pre_inspection_warns_about_uncheckable_state() {
        // Pre-M175 this returned .pass with nil hint when totalBytes was
        // unknown — user saw ✓ but the check hadn't actually evaluated.
        // OOM mid-convert is arguably worse than disk-full since the OS
        // may kill the subprocess before it can surface a clean error.
        let src = tmp.appendingPathComponent("src-ram-uninspected")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-ram-uninspected")
        plan.profile = "JANG_4K"
        // detected stays nil → totalBytes unknown
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        guard let ram = checks.first(where: { $0.id == .ramAdequate }) else {
            XCTFail("no ramAdequate check in preflight output")
            return
        }
        XCTAssertEqual(ram.status, .warn,
            "pre-inspection ramAdequate must warn, not silently pass (M175)")
        XCTAssertTrue(ram.hint?.contains("no estimate") == true,
            "warn hint must explain the uncheckable state. Got: \(ram.hint ?? "nil")")
    }

    func test_ramAdequate_post_inspection_with_room_still_passes() {
        // Regression guard for the real-check branch.
        let src = tmp.appendingPathComponent("src-ram-ok")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-ram-ok")
        plan.profile = "JANG_4K"
        // 10 MB source → needed ~15 MB → any Mac has this much RAM
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 10_000_000, shardCount: 1)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        let ram = checks.first(where: { $0.id == .ramAdequate })
        XCTAssertEqual(ram?.status, .pass,
            "post-inspection with ample RAM must still pass (regression guard)")
    }

    // MARK: - Iter 101 M05: diskSpace disambiguates "no estimate" from "enough room"

    func test_diskSpace_pre_inspection_warns_about_uncheckable_state() {
        // Plan with output set but no detected source → estimator returns 0.
        // Pre-M05 this produced `.pass` which visually matched a real passing
        // check. M05 changes to `.warn` with "(no estimate yet)" hint so the
        // user knows the system didn't actually verify sufficient space.
        let src = tmp.appendingPathComponent("src-uninspected")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-uninspected")
        plan.profile = "JANG_4K"
        // detected stays nil → totalBytes unknown → estimator returns 0
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        guard let disk = checks.first(where: { $0.id == .diskSpace }) else {
            XCTFail("no diskSpace check in preflight output")
            return
        }
        XCTAssertEqual(disk.status, .warn,
            "pre-inspection diskSpace must warn, not silently pass — user needs to see 'no estimate' state")
        XCTAssertTrue(disk.hint?.contains("no estimate") == true,
            "warn hint must explain WHY the check is uncheckable. Got: \(disk.hint ?? "nil")")
    }

    func test_diskSpace_post_inspection_with_room_still_passes() {
        // Regression guard: with a real detected source size + sufficient free
        // space, the check must still produce `.pass` (M05 only changed the
        // uncheckable branch).
        let src = tmp.appendingPathComponent("src-with-shards")
        try? FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try? #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out-with-shards")
        plan.profile = "JANG_4K"
        // Tiny model — estimate will be comfortably under free space
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 10_000_000, shardCount: 1)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen, profiles: .frozen)
        let disk = checks.first(where: { $0.id == .diskSpace })
        XCTAssertEqual(disk?.status, .pass,
            "with a real estimate + enough free space, diskSpace must still pass (regression guard)")
    }

    // MARK: - Iter 99 M173: estimate honors source dtype (FP8 / BF16 / FP32)

    func test_estimateOutputBytes_fp8_source_uses_8bit_divisor() {
        // For FP8 source, src_bytes = weights × 1 (not × 2). Output at 4 bits
        // avg = weights × 0.5. So the formula needs src_bytes × (4/8) × 1.05
        // = 0.525 × src_bytes, NOT the BF16-assuming 0.26 × src_bytes. Pre-
        // M173 the user got a prediction half the real need → "plenty of
        // disk" green check → convert fails mid-way on disk-full.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "deepseek_v3", isMoE: true, numExperts: 256,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .fp8, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        // 100 GB FP8 × 4/8 × 1.05 = 52.5 GB
        XCTAssertEqual(est, 52_500_000_000, accuracy: 1_000_000_000,
            "FP8 source must use /8 divisor — pre-M173 this returned ~26 GB (half the real need)")
    }

    func test_estimateOutputBytes_bf16_source_matches_pre_M173_behavior() {
        // Regression: BF16 source (the case the original formula was written
        // for) must still produce the same answer post-M173.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000)
    }

    func test_estimateOutputBytes_fp16_source_same_as_bf16() {
        // FP16 is also 2 bytes/weight — same /16 divisor as BF16.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .fp16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000)
    }

    func test_estimateOutputBytes_unknown_dtype_falls_back_to_16bit_assumption() {
        // Safety default: if source dtype is unknown (older models, detection
        // drift), assume 16-bit — the common case historically. Conservative:
        // over-estimate is better than under-estimate for the disk-space gate
        // (better to refuse a safe convert than to permit a disk-full crash).
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .unknown, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 26_250_000_000, accuracy: 500_000_000,
            "Unknown dtype must fall back to /16 (BF16/FP16 assumption) for safety")
    }

    func test_estimateOutputBytes_returns_zero_for_unknown_profile() {
        // Regression: unknown profile must also produce 0 rather than
        // guessing a bits value — prevents false positives from typos.
        let plan = ConversionPlan()
        plan.profile = "JANG_UNKNOWN_99X"
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0,
                              isVL: false, isVideoVL: false, hasGenerationConfig: true,
                              dtype: .bf16, totalBytes: 100_000_000_000, shardCount: 1)
        let est = PreflightRunner.estimateOutputBytes(plan: plan, profiles: .frozen)
        XCTAssertEqual(est, 0,
            "Unknown profile must return 0 (caller falls back to pass). Don't guess a bit-width.")
    }

    // MARK: - M142 (iter 64): hadamardVsLowBits uses compress bits, not substring
    //
    // Pre-iter-64 the check was:
    //   let is2bit = plan.profile.contains("_2")
    //              || plan.profile == "JANG_1L"
    //              || plan.profile == "JANGTQ2"
    // Brittle: a future "JANG_20" (20-bit) would trip as 2-bit; a future
    // "JANG_0S" (<1-bit) wouldn't be flagged. Fix: look up compressBits
    // from ProfilesService — single source of truth with Python-side
    // allocate.JANG_PROFILES.

    func test_compressBitsForProfile_JANG_2L() {
        // JANG_2L is (8, 6, 2) → compress=2.
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_2L", profiles: .frozen), 2)
    }

    func test_compressBitsForProfile_JANG_1L() {
        // JANG_1L is (8, 8, 2) → compress=2 (low-bit flagged).
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_1L", profiles: .frozen), 2)
    }

    func test_compressBitsForProfile_JANG_4M() {
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_4M", profiles: .frozen), 4)
    }

    func test_compressBitsForProfile_K_quant() {
        // JANG_4K has compressBits=nil in the schema; derive from avgBits=4.0.
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANG_4K", profiles: .frozen), 4)
    }

    func test_compressBitsForProfile_JANGTQ2() {
        XCTAssertEqual(PreflightRunner.compressBitsForProfile("JANGTQ2", profiles: .frozen), 2)
    }

    func test_compressBitsForProfile_unknown_returns_nil() {
        // Iter-54 M132 fix: typo defense. Unknown profile must return nil
        // so the caller can distinguish "unknown" from "known and high-bit".
        XCTAssertNil(PreflightRunner.compressBitsForProfile("JANG_BOGUS_99Y", profiles: .frozen))
    }

    func test_hadamardAt2bitWarns() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.profile = "JANG_2S"
        plan.hadamard = true
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .hadamardVsLowBits && $0.status == .warn })
    }
}
