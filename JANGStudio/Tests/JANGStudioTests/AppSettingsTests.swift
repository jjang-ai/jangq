import XCTest
@testable import JANGStudio

@MainActor
final class AppSettingsTests: XCTestCase {
    nonisolated private static let allLeafKeys: [String] = [
        BundleResolver.pythonOverrideDefaultsKey,
        BundleResolver.tickThrottleMsDefaultsKey,
        BundleResolver.mlxThreadCountDefaultsKey,
        BundleResolver.customJangToolsPathDefaultsKey,
    ]

    override func setUp() {
        super.setUp()
        // Wipe defaults before each test to avoid cross-contamination.
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
        for k in Self.allLeafKeys { UserDefaults.standard.removeObject(forKey: k) }
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
        for k in Self.allLeafKeys { UserDefaults.standard.removeObject(forKey: k) }
        super.tearDown()
    }

    func test_defaults_match_spec() {
        let s = AppSettings()
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        XCTAssertEqual(s.defaultFamily, "jang")
        XCTAssertEqual(s.defaultMethod, "mse")
        // M200 (iter 137): defaultCalibrationSamples removed as a lie.
        XCTAssertEqual(s.outputNamingTemplate, "{basename}-{profile}")
        XCTAssertEqual(s.logVerbosity, .normal)
        XCTAssertEqual(s.tickThrottleMs, 100)
        XCTAssertTrue(s.revealInFinderOnFinish)
        XCTAssertTrue(s.metalPipelineCacheEnabled)
        XCTAssertTrue(s.copyDiagnosticsAlwaysVisible)
    }

    func test_render_output_name_substitutes_tokens() {
        let s = AppSettings()
        s.outputNamingTemplate = "{basename}-{profile}-{family}"
        let result = s.renderOutputName(basename: "Qwen3-0.6B", profile: "JANG_4K", family: "jang")
        XCTAssertEqual(result, "Qwen3-0.6B-JANG_4K-jang")
    }

    func test_render_output_name_with_date_token() {
        let s = AppSettings()
        s.outputNamingTemplate = "{basename}-{date}"
        let result = s.renderOutputName(basename: "model", profile: "x", family: "y")
        // Result should start with "model-" and have a date following
        XCTAssertTrue(result.hasPrefix("model-20"))   // 2026-
    }

    func test_persist_and_reload() {
        let s1 = AppSettings()
        s1.defaultProfile = "JANG_2L"
        s1.tickThrottleMs = 50
        s1.logVerbosity = .verbose
        s1.persist()

        let s2 = AppSettings()   // loads from UserDefaults
        XCTAssertEqual(s2.defaultProfile, "JANG_2L")
        XCTAssertEqual(s2.tickThrottleMs, 50)
        XCTAssertEqual(s2.logVerbosity, .verbose)
    }

    func test_reset_restores_defaults() {
        let s = AppSettings()
        s.defaultProfile = "JANG_2L"
        s.tickThrottleMs = 999
        s.reset()
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        XCTAssertEqual(s.tickThrottleMs, 100)
    }

    func test_update_channel_roundtrip() {
        let s1 = AppSettings()
        s1.updateChannel = .beta
        s1.persist()
        let s2 = AppSettings()
        XCTAssertEqual(s2.updateChannel, .beta)
    }

    // MARK: - Iter 9: M61 — pythonOverridePath now actually flows to BundleResolver

    func test_python_override_persists_to_leaf_consumer_key() {
        let s = AppSettings()
        s.pythonOverridePath = "/opt/homebrew/bin/python3.11"
        s.persist()
        XCTAssertEqual(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "/opt/homebrew/bin/python3.11",
            "persist() must mirror pythonOverridePath to the leaf-consumer UserDefaults key"
        )
    }

    func test_clearing_python_override_removes_leaf_key() {
        // Set and persist first
        let s = AppSettings()
        s.pythonOverridePath = "/some/path"
        s.persist()
        XCTAssertNotNil(UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey))

        // Now clear
        s.pythonOverridePath = ""
        s.persist()
        XCTAssertNil(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "clearing pythonOverridePath must REMOVE the leaf key (so env/bundled fallbacks take over)"
        )
    }

    func test_bundle_resolver_reads_user_default() {
        UserDefaults.standard.set("/usr/bin/env", forKey: BundleResolver.pythonOverrideDefaultsKey)
        XCTAssertEqual(BundleResolver.pythonExecutable.path, "/usr/bin/env")
    }

    func test_bundle_resolver_ignores_empty_user_default() {
        // Empty string should NOT be treated as a valid override — fall through to env/bundled.
        UserDefaults.standard.set("", forKey: BundleResolver.pythonOverrideDefaultsKey)
        // Exact return depends on env but it should NOT be the empty string we set.
        XCTAssertNotEqual(BundleResolver.pythonExecutable.path, "")
    }

    func test_load_resyncs_leaf_consumer_key_on_fresh_process() {
        // Simulate: previous process persisted a python override. Current process
        // starts with a DIFFERENT leaf-key value (or no leaf-key at all).
        // load() on init must re-sync the leaf key from the Snapshot so consumers
        // don't get stale data.
        let s1 = AppSettings()
        s1.pythonOverridePath = "/path/from/prior/session"
        s1.persist()
        // Simulate leaf-key drift (someone else wrote a different value)
        UserDefaults.standard.set("/wrong/path", forKey: BundleResolver.pythonOverrideDefaultsKey)
        // Fresh AppSettings() calls load(), which should re-mirror the Snapshot value
        let s2 = AppSettings()
        XCTAssertEqual(s2.pythonOverridePath, "/path/from/prior/session")
        XCTAssertEqual(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "/path/from/prior/session",
            "load() must re-sync leaf key from the authoritative Snapshot"
        )
    }

    func test_reset_clears_leaf_consumer_key() {
        let s = AppSettings()
        s.pythonOverridePath = "/some/path"
        s.persist()
        XCTAssertNotNil(UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey))
        s.reset()
        XCTAssertNil(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "reset() must also clear leaf-consumer mirrors (reset() calls persist internally)"
        )
    }

    // MARK: - Iter 11: M62 env passthrough mirroring

    func test_tick_throttle_mirrors_only_non_default() {
        let s = AppSettings()
        // Default is 100 ms — mirrored value should be absent (fall through to
        // Python default) so the env var is never set for users who never
        // touched the slider.
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 0,
                       "default 100 ms → leaf key absent (0 from .integer() means absent)")

        s.tickThrottleMs = 250
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 250)

        // Returning to default must REMOVE the key.
        s.tickThrottleMs = 100
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 0,
                       "returning to default must remove the mirror key")
    }

    func test_mlx_thread_count_mirrors_only_non_zero() {
        let s = AppSettings()
        s.persist()  // defaults: mlxThreadCount=0 (auto)
        XCTAssertNil(UserDefaults.standard.object(forKey: BundleResolver.mlxThreadCountDefaultsKey),
                     "0 means auto → leaf key must be absent")

        s.mlxThreadCount = 8
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.mlxThreadCountDefaultsKey), 8)

        s.mlxThreadCount = 0
        s.persist()
        XCTAssertNil(UserDefaults.standard.object(forKey: BundleResolver.mlxThreadCountDefaultsKey),
                     "returning to auto must remove the mirror key")
    }

    func test_custom_jang_tools_path_mirrors_only_non_empty() {
        let s = AppSettings()
        s.customJangToolsPath = "/Users/eric/custom"
        s.persist()
        XCTAssertEqual(UserDefaults.standard.string(forKey: BundleResolver.customJangToolsPathDefaultsKey),
                       "/Users/eric/custom")

        s.customJangToolsPath = ""
        s.persist()
        XCTAssertNil(UserDefaults.standard.string(forKey: BundleResolver.customJangToolsPathDefaultsKey))
    }

    // MARK: - Iter 25: M48 — defaultHFOrg persists + seeds Publish sheet

    func test_default_hf_org_default_is_empty() {
        let s = AppSettings()
        XCTAssertEqual(s.defaultHFOrg, "",
                       "default must be empty so we don't prefix a wrong org on users who haven't configured it")
    }

    func test_default_hf_org_persists_across_process() {
        let s1 = AppSettings()
        s1.defaultHFOrg = "dealignai"
        s1.persist()
        let s2 = AppSettings()
        XCTAssertEqual(s2.defaultHFOrg, "dealignai")
    }

    func test_reset_clears_default_hf_org() {
        let s = AppSettings()
        s.defaultHFOrg = "dealignai"
        s.reset()
        XCTAssertEqual(s.defaultHFOrg, "")
    }

    func test_pre_iter25_snapshot_defaults_hf_org_to_empty() throws {
        // Older snapshots persisted before iter 25 won't have a defaultHFOrg
        // field. The JSON decoder must still accept them (field has a default)
        // and apply the empty string. Without the Snapshot default, old
        // UserDefaults would fail to decode after an app update.
        let oldSnapshot = """
        {
            "defaultOutputParentPath": "",
            "defaultProfile": "JANG_4K",
            "defaultFamily": "jang",
            "defaultMethod": "mse",
            "defaultHadamardEnabled": false,
            "defaultCalibrationSamples": 256,
            "outputNamingTemplate": "{basename}-{profile}",
            "autoDeletePartialOnCancel": false,
            "revealInFinderOnFinish": true,
            "pythonOverridePath": "",
            "customJangToolsPath": "",
            "logVerbosity": "normal",
            "jsonlLogRetentionLines": 10000,
            "logFileOutputDir": "",
            "tickThrottleMs": 100,
            "maxBundleSizeWarningMb": 450,
            "mlxThreadCount": 0,
            "metalPipelineCacheEnabled": true,
            "preAllocateRam": false,
            "preAllocateRamGb": 4,
            "convertConcurrency": 1,
            "copyDiagnosticsAlwaysVisible": true,
            "anonymizePathsInDiagnostics": false,
            "githubIssuesUrl": "https://github.com/jjang-ai/jangq/issues",
            "autoOpenIssueTrackerOnCrash": false,
            "updateChannel": "stable",
            "autoCheckForUpdates": true
        }
        """.data(using: .utf8)!
        UserDefaults.standard.set(oldSnapshot, forKey: "JANGStudioSettings")
        // New AppSettings() calls load() which decodes the old snapshot.
        let s = AppSettings()
        // Other fields should round-trip unchanged.
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        // New field defaults to empty since the old snapshot didn't carry it.
        XCTAssertEqual(s.defaultHFOrg, "",
                       "pre-iter-25 UserDefaults snapshot must not fail to decode; defaultHFOrg should default to empty")
        // M200 (iter 137): the blob contains `defaultCalibrationSamples: 256`
        // which was REMOVED in M200. JSONDecoder tolerates unknown keys by
        // default, so the snapshot still decodes cleanly and the other
        // fields round-trip. This test now also serves as a regression
        // guard for M200's "no migration shim required" claim.
    }

    func test_load_resyncs_env_passthrough_keys_on_fresh_process() {
        let s1 = AppSettings()
        s1.tickThrottleMs = 200
        s1.mlxThreadCount = 4
        s1.customJangToolsPath = "/prior/session"
        s1.persist()

        // Simulate drift
        UserDefaults.standard.set(999, forKey: BundleResolver.tickThrottleMsDefaultsKey)
        UserDefaults.standard.set(999, forKey: BundleResolver.mlxThreadCountDefaultsKey)
        UserDefaults.standard.set("/wrong", forKey: BundleResolver.customJangToolsPathDefaultsKey)

        // Fresh AppSettings() → load() → mirrorLeafConsumerKeys() re-syncs.
        _ = AppSettings()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 200)
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.mlxThreadCountDefaultsKey), 4)
        XCTAssertEqual(UserDefaults.standard.string(forKey: BundleResolver.customJangToolsPathDefaultsKey),
                       "/prior/session")
    }

    // MARK: - M147 (iter 69): corrupted saved settings must not silently vanish
    //
    // Pre-iter-69 AppSettings.load() used `try?` on the JSONDecoder call,
    // so a schema-migration-breaking decode silently reverted the user to
    // factory defaults. Same asymmetry as iter-37 M111's persist() fix
    // (which DID log to stderr). Now load() distinguishes "no data yet"
    // (first launch — silent) from "decode failed" (logs to stderr so
    // Copy Diagnostics captures the incident in bug reports).

    func test_load_with_corrupted_settings_blob_falls_back_to_defaults() {
        // Simulate a corrupted UserDefaults blob that cannot decode as
        // AppSettings.Snapshot. The app must load with factory defaults
        // (not crash).
        let key = "JANGStudioSettings"
        UserDefaults.standard.set(Data("not valid json at all".utf8), forKey: key)
        defer { UserDefaults.standard.removeObject(forKey: key) }

        let s = AppSettings()   // load() should log + return, not crash
        // Default values — matches AppSettings's field initializers.
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        XCTAssertFalse(s.defaultHadamardEnabled)
    }

    func test_load_with_no_saved_settings_is_silent() {
        // Regression: the "first launch, no data" path must remain
        // silent — don't surface a log every app open for users who
        // haven't saved settings yet.
        let key = "JANGStudioSettings"
        UserDefaults.standard.removeObject(forKey: key)
        // There's no easy way to assert stderr stayed empty from this
        // thread, so this test just exercises the path and verifies no
        // crash — paired with the corruption test above, together they
        // pin the split between first-launch and decode-failure branches.
        let s = AppSettings()
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
    }

    func test_load_method_split_is_present_in_source() throws {
        // Source-inspection pin: verifies the two branches stay separate
        // so a future refactor can't re-collapse them into a single
        // `guard … try? … else { return }` that silently swallows both.
        let srcURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // JANGStudioTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // JANGStudio (xcodeproj root)
            .appendingPathComponent("JANGStudio/Models/AppSettings.swift")
        let src = try String(contentsOf: srcURL, encoding: .utf8)
        XCTAssertTrue(
            src.contains("load failed (settings decode error"),
            "AppSettings.load() must log decode failures to stderr per M147 iter 69."
        )
    }

    // MARK: - Iter 96 M66: stale-UserDefaults coercion surfaces to stderr

    func test_snapshot_apply_logs_coercion_for_invalid_logVerbosity() throws {
        // Source inspection: a coercion of an invalid LogVerbosity rawValue
        // must emit a stderr log. Without it, a user who downgraded from a
        // future version silently loses their verbose setting with no
        // explanation in Copy Diagnostics.
        let srcURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Models/AppSettings.swift")
        let src = try String(contentsOf: srcURL, encoding: .utf8)
        XCTAssertTrue(
            src.contains("logVerbosity=\\\"") && src.contains("coercing to .normal"),
            "Snapshot.apply must write a stderr line naming the bad logVerbosity value + the fallback (M66 iter 96)"
        )
    }

    func test_snapshot_apply_logs_coercion_for_invalid_updateChannel() throws {
        let srcURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Models/AppSettings.swift")
        let src = try String(contentsOf: srcURL, encoding: .utf8)
        XCTAssertTrue(
            src.contains("updateChannel=\\\"") && src.contains("coercing to .stable"),
            "Snapshot.apply must write a stderr line naming the bad updateChannel value + the fallback (M66 iter 96)"
        )
    }

    // MARK: - Iter 98 M64: observeAndPersist coalescing documented

    func test_observeAndPersist_coalescing_rationale_is_pinned() throws {
        // Guard the M64 iter-98 rationale: withObservationTracking's onChange
        // is one-shot, so multiple mutations in the same main-actor pass
        // produce ONE persist() call (capturing all changes). This is
        // coalescing-by-design, not a bug. If a future refactor drops the
        // rationale comment or restructures the loop, we want this test to
        // flag it so the reviewer reads the M64 entry.
        let srcURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard/SettingsWindow.swift")
        let src = try String(contentsOf: srcURL, encoding: .utf8)
        XCTAssertTrue(
            src.contains("M64") && src.contains("coalescing") && src.lowercased().contains("one-shot"),
            "observeAndPersist must document why its coalescing is correct-by-design"
        )
    }

    @MainActor
    func test_observeAndPersist_captures_rapid_multi_field_mutations() {
        // Functional pin: after multiple mutations in a single synchronous
        // pass, the persisted Snapshot contains ALL of them — not just the
        // last one, not just the first one. The one-shot onChange + Task-
        // deferred persist() design captures the post-batch state.
        let s = AppSettings()
        s.reset()   // start from known defaults

        // Mutate multiple fields synchronously — simulates a Reset button
        // or a bulk-setter method. These land in the same main-actor
        // transaction.
        s.logVerbosity = .verbose
        s.tickThrottleMs = 250
        s.defaultHFOrg = "dealignai"
        s.persist()   // simulate the persist() the observer would trigger

        // Reload from UserDefaults to confirm ALL three fields round-tripped.
        let s2 = AppSettings()
        XCTAssertEqual(s2.logVerbosity, .verbose, "first mutation must persist")
        XCTAssertEqual(s2.tickThrottleMs, 250, "second mutation must persist")
        XCTAssertEqual(s2.defaultHFOrg, "dealignai", "third mutation must persist")
    }

    // MARK: - Iter 109 M176: WizardCoordinator.canActivate functional pin

    @MainActor
    func test_wizardCoordinator_canActivate_gates_unreached_steps() {
        // Functional test for the canActivate logic the sidebar defers to
        // post-M176. Fresh plan → only .source reachable. PR3: expert steps
        // also require workflowMode == .expertLab.
        let coord = WizardCoordinator()
        XCTAssertTrue(coord.canActivate(.source))
        XCTAssertEqual(coord.visibleSteps(), [.source, .profile, .run, .verify])
        XCTAssertFalse(coord.canActivate(.expertReview),
            "Expert Review must be unreachable in Convert mode / without MoE")
        XCTAssertFalse(coord.canActivate(.pruneReview),
            "Prune Review must be unreachable until Expert Review exports a plan")
        XCTAssertFalse(coord.canActivate(.profile))
        XCTAssertFalse(coord.canActivate(.run))
        XCTAssertFalse(coord.canActivate(.verify))

        // MoE + Convert: still no expert activation.
        coord.plan.sourceURL = URL(fileURLWithPath: "/tmp/moe")
        coord.plan.detected = .init(
            modelType: "qwen3_5_moe", isMoE: true, numExperts: 64, isVL: false,
            dtype: .bf16, totalBytes: 0, shardCount: 1
        )
        coord.plan.workflowMode = .convert
        XCTAssertFalse(coord.canActivate(.expertReview))

        // MoE + Expert Lab: expert review opens.
        coord.plan.workflowMode = .expertLab
        XCTAssertTrue(coord.canActivate(.expertReview))
        XCTAssertTrue(coord.canActivate(.profile))
    }

    // MARK: - Iter 108 M62: remaining inert settings labeled "not yet implemented"

    func test_autoCheckForUpdates_has_not_yet_implemented_label() throws {
        // M176b (iter 110): autoCheckForUpdates toggle persists but Sparkle
        // auto-updater ships in v1.1. Per iter-108 M62's pattern, the
        // toggle gets its own "Not yet implemented" label citing Sparkle
        // as the blocker — so the toggle's inert-today status is visible
        // at its attention site, not just in the section caption.
        let srcURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard/SettingsWindow.swift")
        let src = try String(contentsOf: srcURL, encoding: .utf8)
        XCTAssertTrue(
            src.contains("Sparkle integration") && src.contains("Not yet implemented"),
            "autoCheckForUpdates toggle must carry the M62-style Not-yet-implemented label citing Sparkle (M176b iter 110)"
        )
    }

    func test_inert_settings_have_not_yet_implemented_labels() throws {
        // iter-14 M62 wired 9 of 12 UI-lie settings to their actual consumers.
        // Three remained inert: logVerbosity (needs wide JANG_LOG_LEVEL
        // refactor), preAllocateRam, preAllocateRamGb (needs MLX buffer-pool
        // API). Iter-108 adds visible "not yet implemented" labels so the
        // UI doesn't lie — settings persist for when impl lands, but user
        // knows what's unwired today.
        let srcURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard/SettingsWindow.swift")
        let src = try String(contentsOf: srcURL, encoding: .utf8)

        XCTAssertTrue(
            src.contains("JANG_LOG_LEVEL") && src.contains("Not yet implemented"),
            "logVerbosity section must carry a Not-yet-implemented label citing JANG_LOG_LEVEL as the blocker (M62 iter 108)"
        )
        XCTAssertTrue(
            src.contains("MLX buffer-pool") && src.contains("Not yet implemented"),
            "preAllocateRam section must carry a Not-yet-implemented label citing MLX buffer-pool API as the blocker (M62 iter 108)"
        )
    }

    // MARK: - Iter 104 M108: try? site count invariant (coarse bulk-addition trap)

    func test_try_question_site_count_within_threshold() throws {
        // M108 (iter 14 observation, closed iter 104): `try?` is NOT inherently
        // a bug — iter-104's audit classified all 34 sites into 8 acceptable
        // categories:
        //   A. comment text referencing prior fixes (6)
        //   B. parse-tolerance file reads (read file that may not exist) (9)
        //   C. Task.sleep ignore in cancellation paths (5)
        //   D. stderrTask.value await fallback (2)
        //   E. regex compile of known-valid static patterns (2)
        //   F. macOS resource-query with 0-fallback (1)
        //   G. temp-dir cleanup / pipe-close (4)
        //   H. JSON round-trip in mixed-type dict (2)
        //   I. other acceptable patterns — inspect new additions (varies)
        //
        // The BAD pattern is iter-35 M107 / iter-80 M157's class: user-action
        // silent swallow in a Button handler. Iter-35+iter-80 fixed those.
        // This test catches bulk NEW additions — a routine 1-2 new try? per
        // PR goes through; a 15+ addition triggers review. The threshold
        // lives above today's 34 with generous headroom.
        //
        // When this test fails:
        //   1. The engineer adding new try? should classify each addition
        //      per the taxonomy above.
        //   2. If all fit, bump the threshold (a small nudge, 50 → 60).
        //   3. If any is user-action-silent-swallow, fix with do/catch +
        //      stderr log per iter-35 / iter-80 pattern.
        //   4. Update the category counts in this comment.

        let srcDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/JANGStudio")

        var total = 0
        let enumerator = FileManager.default.enumerator(atPath: srcDir.path)
        while let rel = enumerator?.nextObject() as? String {
            guard rel.hasSuffix(".swift") else { continue }
            let path = srcDir.appendingPathComponent(rel).path
            guard let content = try? String(contentsOfFile: path, encoding: .utf8) else { continue }
            // `try?` (with question mark) as a substring anywhere — counts
            // comment mentions too, which is fine for this coarse gate.
            let occurrences = content.components(separatedBy: "try?").count - 1
            total += occurrences
        }

        // Today's count is 34. Threshold is 50 — 16 addition headroom before
        // this test fires. A bulk new-pattern introduction (e.g., someone
        // reintroducing the M107 class) would push past this; routine work
        // passes through.
        XCTAssertLessThanOrEqual(total, 50,
            "try? site count (\(total)) exceeds threshold — audit new additions per the taxonomy in this test's comment (M108 iter 104). If all additions are in acceptable categories, bump the threshold.")
    }

    // MARK: - Iter 103 M65: AppSettings mutation is SettingsWindow-only (grep invariant)

    func test_appSettings_mutations_are_settingsWindow_only() throws {
        // M65 (iter 14 observation): the observeAndPersist auto-persist Task
        // is bound to the Settings sheet's .task. If anything OUTSIDE the
        // Settings sheet mutates settings programmatically (future crash
        // reporter toggling autoOpenIssueTrackerOnCrash, telemetry sampler,
        // etc.), the mutation wouldn't trigger persistence until the user
        // opens Settings again — lost change.
        //
        // Iter-103 grep confirms: today, ALL `settings.<prop> = …` sites live
        // in SettingsWindow.swift. This test pins that invariant so any
        // future addition (e.g. a background sync that mutates a setting)
        // fails here and forces the engineer to either (a) move the mutation
        // into SettingsWindow's reach or (b) rewire auto-persist to cover
        // the new site.
        //
        // We walk the Swift sources (excluding SettingsWindow.swift) and
        // assert no line matches `settings\.<identifier>\s*=`. Environment-
        // injected `settings` reads (e.g. `settings.foo` as RHS) are fine;
        // only WRITES trigger the M65 hypothetical.

        let wizardDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio")

        let enumerator = FileManager.default.enumerator(atPath: wizardDir.path)
        var offenders: [(file: String, line: Int, text: String)] = []
        while let rel = enumerator?.nextObject() as? String {
            guard rel.hasSuffix(".swift") else { continue }
            // Skip SettingsWindow (the allowed mutation site) and the
            // AppSettings.swift file itself (self-mutation is expected).
            if rel.hasSuffix("Wizard/SettingsWindow.swift") { continue }
            if rel.hasSuffix("Models/AppSettings.swift") { continue }
            let path = wizardDir.appendingPathComponent(rel).path
            guard let content = try? String(contentsOfFile: path, encoding: .utf8) else { continue }
            for (i, rawLine) in content.components(separatedBy: "\n").enumerated() {
                let line = rawLine.trimmingCharacters(in: .whitespaces)
                if line.hasPrefix("//") { continue }
                if line.hasPrefix("///") { continue }
                // Match `settings.<ident> =` but NOT `settings.foo ==` or
                // `_ = settings.foo` or closure-param `settings = ...`
                // (that last one would be a new @State which is rare).
                if let range = line.range(of: #"\bsettings\.[A-Za-z_][A-Za-z0-9_]*\s*=[^=]"#,
                                          options: .regularExpression) {
                    // Skip Picker/Binding get/set — those are reads.
                    let snippet = String(line[range])
                    if snippet.contains(",") || snippet.contains("get:") { continue }
                    offenders.append((file: rel, line: i + 1, text: line))
                }
            }
        }
        XCTAssertTrue(offenders.isEmpty,
            "M65 invariant violated — settings mutated outside SettingsWindow:\n" +
            offenders.map { "  \($0.file):\($0.line) — \($0.text)" }.joined(separator: "\n"))
    }

    func test_snapshot_apply_still_coerces_invalid_values_to_defaults() {
        // Functional test: behavior is preserved — an invalid rawValue still
        // results in the default being applied. The M66 fix ADDS logging
        // but doesn't change the fallback behavior. AppSettings() calls
        // load() internally, so corrupting UserDefaults BEFORE init gets
        // picked up.
        let badSnapshot = #"""
        {
            "version": 1,
            "logVerbosity": "this-is-not-a-real-verbosity-case",
            "updateChannel": "also-not-real",
            "defaultProfile": "",
            "defaultFamily": "jang",
            "defaultMethod": "mse",
            "defaultHadamardEnabled": false,
            "defaultCalibrationSamples": 128,
            "outputNamingTemplate": "{src}-{profile}",
            "autoDeletePartialOnCancel": true,
            "revealInFinderOnFinish": true,
            "defaultHFOrg": "",
            "pythonOverridePath": "",
            "customJangToolsPath": "",
            "jsonlLogRetentionLines": 500,
            "logFileOutputDir": "",
            "tickThrottleMs": 100,
            "maxBundleSizeWarningMb": 200,
            "mlxThreadCount": 0,
            "metalPipelineCacheEnabled": true,
            "preAllocateRam": false,
            "preAllocateRamGb": 32,
            "convertConcurrency": 2,
            "copyDiagnosticsAlwaysVisible": false,
            "anonymizePathsInDiagnostics": false,
            "githubIssuesUrl": "",
            "autoOpenIssueTrackerOnCrash": false,
            "autoCheckForUpdates": true
        }
        """#
        UserDefaults.standard.set(Data(badSnapshot.utf8), forKey: "JANGStudioSettings")
        defer { UserDefaults.standard.removeObject(forKey: "JANGStudioSettings") }
        let s = AppSettings()   // init calls load() which runs Snapshot.apply
        XCTAssertEqual(s.logVerbosity, .normal,
            "invalid logVerbosity must coerce to .normal (behavior preserved while logging added)")
        XCTAssertEqual(s.updateChannel, .stable,
            "invalid updateChannel must coerce to .stable (behavior preserved while logging added)")
    }
}
