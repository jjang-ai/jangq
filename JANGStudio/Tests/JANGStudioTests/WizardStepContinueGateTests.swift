// JANGStudio/Tests/JANGStudioTests/WizardStepContinueGateTests.swift
//
// Peer-helper parity on wizard step "Continue" buttons.
//
// Each wizard step has a forward-navigation button (e.g. "Continue →",
// "Start Conversion"). ArchitectureStep was deleted in PR2; Advanced
// overrides now live on ProfileStep.
//
// We test code-SHAPE rather than runtime behavior because SwiftUI's
// `.disabled` state isn't trivially inspectable outside a ViewInspector /
// XCUITest harness.

import XCTest
@testable import JANGStudio

final class WizardStepContinueGateTests: XCTestCase {

    private static let stepsRoot: URL = {
        // Test bundle is at .../DerivedData/.../Build/Products/Debug/JANGStudio.app/Contents/PlugIns/JANGStudioTests.xctest
        // Source tree lives in the repo — we look it up relative to the test runner's working directory.
        // SRCROOT isn't exposed to XCTest at runtime, so fall back to the common repo path.
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // JANGStudioTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // JANGStudio (xcodeproj root)
            .appendingPathComponent("JANGStudio/Wizard/Steps")
    }()

    private func stepSource(_ stepFile: String) throws -> String {
        let url = Self.stepsRoot.appendingPathComponent(stepFile)
        return try String(contentsOf: url, encoding: .utf8)
    }

    func test_profileStep_hostsAdvancedOverridesDisclosure() throws {
        // PR2: ArchitectureStep deleted; force dtype / block size live on Profile.
        let src = try stepSource("ProfileStep.swift")
        XCTAssertTrue(
            src.contains("DisclosureGroup(\"Advanced overrides\""),
            "ProfileStep must host Advanced overrides DisclosureGroup after Architecture cleanup"
        )
        XCTAssertTrue(
            src.contains("forceDtype") && src.contains("forceBlockSize"),
            "ProfileStep Advanced overrides must bind forceDtype and forceBlockSize"
        )
        XCTAssertTrue(
            src.contains("coord.plan.family == .jang"),
            "Advanced overrides should be gated to JANG family (JANGTQ ignores them)"
        )
    }

    func test_sourceStep_continue_is_gated() throws {
        let src = try stepSource("SourceStep.swift")
        // SourceStep uses the `if coord.plan.isStep1Complete { Button… }` style.
        XCTAssertTrue(
            src.contains("if coord.plan.isStep1Complete"),
            "SourceStep Continue button must remain gated via `if isStep1Complete`."
        )
    }

    func test_profileStep_continue_is_gated() throws {
        let src = try stepSource("ProfileStep.swift")
        XCTAssertTrue(
            src.contains(".disabled(!allMandatoryPass())"),
            "ProfileStep Start Conversion button must remain gated via preflight."
        )
        XCTAssertTrue(
            src.contains("guard !preflight.isEmpty else { return false }"),
            "ProfileStep must keep conversion disabled until preflight checks have actually loaded."
        )
        XCTAssertTrue(
            src.contains("if isFinalConversionFromReviewedPrune"),
            "ProfileStep must apply a stricter gate for final quantization from a reviewed BF16/F16 prune."
        )
        XCTAssertTrue(
            src.contains("$0.id == .reviewedPruneVerified && $0.status == .pass"),
            "ProfileStep final quantization must require the reviewed-prune preflight check to pass."
        )
    }

    func test_runStep_continue_is_gated() throws {
        let src = try stepSource("RunStep.swift")
        XCTAssertTrue(
            src.contains("if coord.plan.run == .succeeded"),
            "RunStep Continue → Verify button must remain gated by run status."
        )
    }

    // MARK: - M135 (iter 57): SourceStep stale-task cancellation
    //
    // When the user picks folder A, a detection task starts. If they
    // immediately pick folder B before A's task finishes, A must be
    // cancelled — otherwise A's result eventually stomps B's detected
    // state and the user sees A's metadata while sourceURL points at B.

    func test_sourceStep_tracks_detection_task_handle() throws {
        let src = try stepSource("SourceStep.swift")
        XCTAssertTrue(
            src.contains("@State private var detectionTask: Task<Void, Never>?"),
            """
            SourceStep must store the detection Task handle so it can be
            cancelled when the user picks a new folder. Without this,
            concurrent detections race and the slower one stomps state.
            """
        )
    }

    func test_sourceStep_pickFolder_cancels_previous_task() throws {
        let src = try stepSource("SourceStep.swift")
        // The pickFolder body must call `detectionTask?.cancel()` BEFORE
        // starting the new task. Locate the panel.runModal() block and
        // check the cancel call is present AND precedes the new Task.
        guard let pickRange = src.range(of: "if panel.runModal() == .OK, let url = panel.url {") else {
            XCTFail("could not locate pickFolder's panel.runModal block")
            return
        }
        let body = String(src[pickRange.upperBound...].prefix(800))
        XCTAssertTrue(
            body.contains("detectionTask?.cancel()"),
            """
            pickFolder must call detectionTask?.cancel() to tear down any
            previous detection. Body after panel.runModal:
            \(body)
            """
        )
        // Cancel must come before the new Task assignment.
        if let cancelIdx = body.range(of: "detectionTask?.cancel()")?.lowerBound,
           let newTaskIdx = body.range(of: "detectionTask = Task")?.lowerBound {
            XCTAssertLessThan(cancelIdx, newTaskIdx,
                "cancel() must precede the new Task assignment")
        }
    }

    // MARK: - M161 (iter 84): SourceStep URL-match guard on write-back
    //
    // iter-57 M135's `detectionTask?.cancel()` handles the case where the
    // user picks a new folder within the SAME SourceStep view instance.
    // But `detectionTask` is `@State private` — scoped to the view. When
    // the user sidebar-jumps to Architecture and back to Source, the old
    // SourceStep is destroyed; the detection Task it spawned keeps
    // running (Task independence from creation scope) but the NEW
    // SourceStep instance has `detectionTask = nil` so it can't cancel
    // the orphan. The orphan then completes and stomps
    // `coord.plan.detected` with stale data.
    //
    // M161's defense-in-depth: every MainActor.run write-back in
    // detectAndRecommend now checks `coord.plan.sourceURL == url`. If
    // sourceURL has moved on, the write is stale regardless of cancel
    // state and is discarded.

    func test_sourceStep_detectAndRecommend_guards_writes_by_url_match() throws {
        let src = try stepSource("SourceStep.swift")
        // There are 5 MainActor.run sites in detectAndRecommend (detect
        // success, detect error, isDetecting=false, rec success, rec error).
        // Each must contain a `guard coord.plan.sourceURL == url else { return }`
        // to block stale writes from orphaned tasks.
        let guardCount = src.components(
            separatedBy: "guard coord.plan.sourceURL == url else { return }"
        ).count - 1
        XCTAssertGreaterThanOrEqual(guardCount, 5,
            """
            detectAndRecommend must have sourceURL-match guards at all 5
            MainActor.run write-back sites (detect success, detect error,
            isDetecting=false, rec success, rec error). Found \(guardCount)
            — M161 regression.
            """
        )
    }

    func test_sourceStep_M161_rationale_is_present() throws {
        let src = try stepSource("SourceStep.swift")
        // Pin the M161 rationale comment so a future "simplification" that
        // strips it also flags to the reviewer why those guards are there.
        XCTAssertTrue(
            src.contains("M161") && src.contains("orphaned"),
            "M161 rationale comment must remain — explains why URL-match is the authoritative stale-write defense"
        )
    }

    // MARK: - M162 (iter 85): sheet dismissal must cancel in-flight subprocesses

    func test_publishSheet_cancels_publishTask_onDisappear() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("PublishToHuggingFaceSheet.swift"), encoding: .utf8)
        XCTAssertTrue(
            src.contains(".onDisappear") && src.contains("publishTask?.cancel()"),
            """
            PublishToHuggingFaceSheet must wire `.onDisappear { publishTask?.cancel() }`
            so sheet dismissal tears down the active upload. Without it, user
            who clicks Close mid-publish continues uploading to HF in the
            background with no UI — accidental data exfiltration vector.
            """
        )
    }

    func test_publishSheet_M162_rationale_pinned() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("PublishToHuggingFaceSheet.swift"), encoding: .utf8)
        XCTAssertTrue(
            src.contains("M162") && src.contains("data-exfiltration"),
            "M162 rationale must remain — describes why the onDisappear cancel is security-critical"
        )
    }

    func test_testInferenceSheet_cancels_vm_onDisappear() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("TestInferenceSheet.swift"), encoding: .utf8)
        XCTAssertTrue(
            src.contains(".onDisappear") && src.contains("vm.cancel()"),
            """
            TestInferenceSheet must wire `.onDisappear { Task { await vm.cancel() } }`
            so dismissing mid-generate tears down the inference subprocess —
            freeing the GPU + memory for subsequent runs.
            """
        )
    }

    // MARK: - M23 (iter 97): Delete partial output distinguishes "already gone" from real failure

    func test_runStep_delete_partial_output_distinguishes_already_gone() throws {
        let src = try stepSource("RunStep.swift")
        XCTAssertTrue(
            src.contains("catch CocoaError.fileNoSuchFile"),
            "Delete-partial-output must catch CocoaError.fileNoSuchFile separately so 'already gone' doesn't surface as 'FAILED' (M23 iter 97)"
        )
        XCTAssertTrue(
            src.contains("already gone"),
            "The success-but-already-gone log line must contain 'already gone' so the user understands the goal state is achieved"
        )
    }

    // MARK: - M176 (iter 109): sidebar navigation honors canActivate

    func test_sidebar_selection_binding_gates_on_canActivate() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("WizardCoordinator.swift"), encoding: .utf8)
        // Pre-M176: `set: { coord.active = $0 ?? .source }` (no gate).
        // Post-M176: set: closure calls coord.canActivate(step) before assigning.
        XCTAssertTrue(
            src.contains("coord.canActivate(step)") && src.contains("set:"),
            "WizardView sidebar set: binding must check canActivate before assigning — M176 iter 109"
        )
        XCTAssertTrue(
            src.contains("visibleSteps()"),
            "WizardView sidebar must filter by visibleSteps (PR3 mode switch)"
        )
        XCTAssertTrue(
            src.contains("ensureActiveIsVisible"),
            "WizardView must keep active selection visible after mode changes"
        )
    }

    func test_runStepNavigationRequiresReviewedPruneVerificationForFinalQuant() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("WizardCoordinator.swift"), encoding: .utf8)

        XCTAssertTrue(src.contains("PreflightRunner.reviewedPruneVerifiedCheck(plan: plan)?.status != .pass"))
        XCTAssertTrue(src.contains("plan.expertReviewPrunedSourceURL != nil"))
        XCTAssertTrue(src.contains("return plan.isStep3Complete"))
    }

    func test_expertLab_primary_workflow_steps_are_visible_in_wizard() throws {
        // PR3: name-only enum titles; numbered labels via displayTitle under mode.
        XCTAssertEqual(
            WizardStep.allCases.map(\.title),
            [
                "Source Model",
                "Expert Review",
                "Prune Review",
                "Conversion Profile",
                "Build / Convert",
                "Verify",
            ]
        )
        let coord = WizardCoordinator()
        XCTAssertEqual(
            coord.visibleSteps().map { coord.displayTitle(for: $0) },
            ["1 · Source Model", "2 · Conversion Profile", "3 · Build / Convert", "4 · Verify"]
        )
        coord.plan.detected = .init(
            modelType: "qwen3_5_moe", isMoE: true, numExperts: 64, isVL: false,
            dtype: .bf16, totalBytes: 0, shardCount: 1
        )
        coord.plan.workflowMode = .expertLab
        XCTAssertEqual(
            coord.visibleSteps().map { coord.displayTitle(for: $0) },
            [
                "1 · Source Model",
                "2 · Expert Review",
                "3 · Prune Review",
                "4 · Conversion Profile",
                "5 · Build / Convert",
                "6 · Verify",
            ]
        )
    }

    // MARK: - M172 (iter 95): dead progressLog state removed from PublishSheet

    func test_publishSheet_has_no_dead_progressLog_state() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("PublishToHuggingFaceSheet.swift"), encoding: .utf8)
        // Split into lines and scan only non-comment lines — the M172
        // rationale comments mention "progressLog" by name, which is fine.
        // But no non-comment line should declare or append to the dead state.
        let nonCommentLines = src.split(separator: "\n").filter {
            !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//")
        }
        let joined = nonCommentLines.joined(separator: "\n")
        XCTAssertFalse(
            joined.contains("@State private var progressLog"),
            "progressLog @State must stay removed (M172 iter 95) — no UI reads it; resurrection means dead code or a half-wired log pane"
        )
        XCTAssertFalse(
            joined.contains("progressLog.append"),
            "progressLog.append calls must stay removed — there's nothing reading the array"
        )
    }

    // MARK: - M171 (iter 94): .onAppear-Task sweep — SourceStep + dryRun gaps

    func test_sourceStep_cancels_detectionTask_onDisappear() throws {
        let src = try stepSource("SourceStep.swift")
        // Must have .onDisappear { detectionTask?.cancel() } so sidebar-jump
        // or window-close during detection tears down the Python subprocess
        // promptly. iter-57 M135 handled concurrent picks; iter-84 M161
        // handled orphan state-corruption; M171 closes the subprocess-
        // teardown gap.
        XCTAssertTrue(
            src.contains(".onDisappear") && src.contains("detectionTask?.cancel()"),
            "SourceStep must wire `.onDisappear { detectionTask?.cancel() }` for subprocess teardown on view unmount"
        )
    }

    func test_publishSheet_cancels_dryRunTask_onDisappear() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("PublishToHuggingFaceSheet.swift"), encoding: .utf8)
        XCTAssertTrue(
            src.contains("dryRunTask"),
            "PublishToHuggingFaceSheet must track the Preview-button Task handle (M171)"
        )
        XCTAssertTrue(
            src.contains("dryRunTask?.cancel()"),
            "PublishToHuggingFaceSheet's .onDisappear must cancel dryRunTask so Preview orphans get torn down"
        )
        XCTAssertTrue(
            src.contains("dryRunTask = Task { await runDryRun() }"),
            "Preview button must assign the Task to dryRunTask instead of discarding it"
        )
    }

    // MARK: - M170 (iter 93): RunStep main-window orphan subprocess on quit/close

    func test_runStep_cancels_runTask_onDisappear() throws {
        let src = try stepSource("RunStep.swift")
        XCTAssertTrue(
            src.contains("@State private var runTask: Task<Void, Never>?"),
            "RunStep must track the active conversion Task so .onDisappear can cancel it on window close / app quit"
        )
        XCTAssertTrue(
            src.contains(".onDisappear") && src.contains("runTask?.cancel()"),
            """
            RunStep needs `.onDisappear { runTask?.cancel() }` so closing
            the main window (red-X, cmd-Q) tears down the convert subprocess.
            Without it, Python keeps running for 30 more minutes as an
            orphaned child of launchd with no UI to cancel from — user sees
            Mac at 100% CPU after "quitting."
            """
        )
    }

    func test_runStep_retry_buttons_use_runTask_handle() throws {
        let src = try stepSource("RunStep.swift")
        // Both Retry paths (after cancelled, after failed) must use the
        // handle so a subsequent window close can cancel the retry too.
        let retrySpawns = src.components(separatedBy: "runTask = Task { await start() }").count - 1
        XCTAssertGreaterThanOrEqual(retrySpawns, 2,
            "Both Retry buttons (after cancelled + after failed) must store the retry Task in runTask; found \(retrySpawns) spawns via runTask")
    }

    // MARK: - M163 (iter 86): Retry-button Task orphan sweep across read-only sheets

    func test_generateModelCardSheet_retry_task_cancelled_onDisappear() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("GenerateModelCardSheet.swift"), encoding: .utf8)
        XCTAssertTrue(
            src.contains("retryTask") && src.contains(".onDisappear") && src.contains("retryTask?.cancel()"),
            "GenerateModelCardSheet must track retryTask + cancel on .onDisappear (M163)"
        )
    }

    func test_usageExamplesSheet_retry_task_cancelled_onDisappear() throws {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("JANGStudio/Wizard")
        let src = try String(contentsOf: dir.appendingPathComponent("UsageExamplesSheet.swift"), encoding: .utf8)
        XCTAssertTrue(
            src.contains("retryTask") && src.contains(".onDisappear") && src.contains("retryTask?.cancel()"),
            "UsageExamplesSheet must track retryTask + cancel on .onDisappear (M163)"
        )
    }

    // MARK: - M136 (iter 58): RunStep auto-start must only fire on .idle
    //
    // Pre-iter-58 RunStep's .onAppear was:
    //     .onAppear { Task { await start() } }
    // and `start()` only guarded `run != .running`. A user nav-backing
    // from VerifyStep via the sidebar (to inspect logs of a finished
    // conversion) would re-trigger start(), wipe logs, overwrite the
    // already-written output folder, and start a fresh conversion the
    // user didn't ask for. The fix gates the onAppear on run == .idle.
    // Retry buttons still call start() directly — they're after a failure/
    // cancel so the weaker `!= .running` guard there is correct.

    func test_runStep_onAppear_only_auto_starts_when_idle() throws {
        let src = try stepSource("RunStep.swift")
        // The onAppear block must check `coord.plan.run == .idle` before
        // calling start(). A bare `.onAppear { Task { await start() } }`
        // is the bug.
        //
        // Strip comment lines first — `.onAppear` appears in the iter-58
        // M136 rationale comment, which would make a naive "first .onAppear
        // + look nearby" test miss the actual modifier.
        let codeOnly = src.split(separator: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")

        guard codeOnly.contains(".onAppear") else {
            XCTFail("could not locate RunStep onAppear in code")
            return
        }
        // Pattern that must appear in code: the onAppear body must gate on
        // `coord.plan.run == .idle`.
        XCTAssertTrue(
            codeOnly.contains("coord.plan.run == .idle"),
            """
            RunStep .onAppear must check `coord.plan.run == .idle` before
            calling start(). Pre-iter-58 a nav-back from VerifyStep would
            restart a completed conversion, wiping logs and overwriting
            the already-written output folder.
            """
        )
    }

    // MARK: - M157 (iter 80): SettingsWindow.openLogsDirectory surfaces failures
    //
    // Pre-M157 used `try? FileManager.default.createDirectory(...)` — silent
    // swallow. NSWorkspace.open against a nonexistent dir is a no-op, so
    // user clicking "Open logs directory" got ZERO feedback on a
    // permission-denied / read-only-volume / disk-full path. Classic iter-35
    // M107 silent-failure-on-user-action bug; just with a different verb.

    func test_settingsWindow_openLogs_surfaces_createDirectory_failures() throws {
        let settingsURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // JANGStudioTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // xcodeproj root
            .appendingPathComponent("JANGStudio/Wizard/SettingsWindow.swift")
        let src = try String(contentsOf: settingsURL, encoding: .utf8)
        // The silent try? must be gone.
        XCTAssertFalse(
            src.contains("try? FileManager.default.createDirectory(at: dir"),
            """
            SettingsWindow.openLogsDirectory regressed to silent
            try? FileManager.default.createDirectory(at:). Pre-M157 this
            swallowed permission-denied / disk-full errors and
            NSWorkspace.open against a nonexistent dir was a no-op —
            user saw nothing when clicking the button. Must use
            do/catch with stderr logging + parent-dir fallback.
            See M157 iter 80.
            """
        )
        // Check the fallback path exists in source.
        XCTAssertTrue(
            src.contains("dir.deletingLastPathComponent()"),
            "openLogsDirectory must fall back to opening the parent dir on createDirectory failure."
        )
        XCTAssertTrue(
            src.contains("[SettingsWindow] could not create"),
            "openLogsDirectory must log createDirectory failures to stderr (Copy Diagnostics pipeline)."
        )
    }

    // MARK: - M143 (iter 65): SourceStep.applyRecommendation must respect
    // settings.defaultProfile, not a hardcoded "JANG_4K".
    //
    // Pre-iter-65: `if plan.profile == "JANG_4K"` gated the overwrite.
    // Users with Settings → defaultProfile = JANG_2L never got the
    // per-source recommendation applied because their plan.profile
    // started at JANG_2L after applyDefaults.

    // MARK: - M146 (iter 68): ProfileStep auto-outputURL follows profile changes
    //
    // Pre-M146 ProfileStep auto-filled `coord.plan.outputURL` once (on
    // .onAppear when nil). If the user then switched profile in the
    // Picker, the already-set outputURL kept the OLD profile in its
    // folder name. Result: convert writes to `src-JANG_4K` but user
    // converted as JANG_2L. Wrong label on every downstream artifact.
    // Fix: .onChange(of: profile) regenerates outputURL only when it
    // matches the auto-pattern for the old profile (user-picked paths
    // via pickOutput don't match and are preserved).

    func test_profileStep_auto_outputURL_follows_profile_change() throws {
        let src = try stepSource("ProfileStep.swift")
        // The onChange(of: coord.plan.profile) block must include a
        // regeneration branch that compares current outputURL to the
        // auto-pattern for the OLD profile.
        XCTAssertTrue(
            src.contains(".onChange(of: coord.plan.profile)"),
            "ProfileStep must react to profile changes. See M146 iter 68."
        )
        XCTAssertTrue(
            src.contains("autoOutputURL(for: src, profile: newProfile)"),
            """
            ProfileStep's profile-change handler must regenerate outputURL
            using the NEW profile name when the current URL matches the
            auto-pattern. See M146 iter 68. Post-M210 the regeneration
            goes through autoOutputURL(for:profile:) which honors the
            settings.outputNamingTemplate + family slot, replacing the
            earlier literal "\\(src.lastPathComponent)-\\(newProfile)"
            concatenation.
            """
        )
        XCTAssertTrue(
            src.contains("if cur == autoOld"),
            """
            The regeneration MUST be gated on the outputURL matching the
            auto-pattern for the old profile — otherwise user-picked
            custom output folders get silently rewritten too.
            """
        )
    }

    // MARK: - M144 (iter 66): family + profile must stay coupled
    //
    // Pre-M144 applyRecommendation overwrote `family` unconditionally
    // every time a source was picked, even if the profile was preserved
    // (because iter-65 M143's seed-default check kept it user-chosen).
    // Result: user picks source → goes to ProfileStep → switches to
    // JANGTQ2 (family=.jangtq) → re-picks source → applyRecommendation
    // preserves profile=JANGTQ2 but overwrites family=.jang →
    // inconsistent pair, invalid state.

    func test_sourceStep_applyRecommendation_does_not_unconditionally_set_family() throws {
        let src = try stepSource("SourceStep.swift")
        // The bare `plan.family = (rec.recommended.family == "jangtq") ? .jangtq : .jang`
        // that unconditionally overwrote family should be gone. Search
        // outside of comments.
        let codeOnly = src.split(separator: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        XCTAssertFalse(
            codeOnly.contains(#"plan.family = (rec.recommended.family == "jangtq")"#),
            """
            SourceStep.applyRecommendation must not unconditionally
            overwrite plan.family. When profile is preserved, family must
            stay coupled to profile or the pair becomes inconsistent
            (e.g. family=.jang + profile=JANGTQ2). See M144 iter 66.
            """
        )
    }

    func test_sourceStep_applyRecommendation_derives_family_from_profile() throws {
        let src = try stepSource("SourceStep.swift")
        // The fix derives family from profile after overwriting profile:
        //   plan.family = plan.profile.hasPrefix("JANGTQ") ? .jangtq : .jang
        XCTAssertTrue(
            src.contains(#"plan.profile.hasPrefix("JANGTQ")"#),
            """
            After overwriting plan.profile, applyRecommendation must derive
            plan.family from the profile name so the two fields can never
            disagree. Expected `plan.profile.hasPrefix("JANGTQ")` check.
            See M144 iter 66.
            """
        )
    }

    // MARK: - M145 (iter 67): hadamard / method / forceDtype preservation
    //
    // Pre-M145 all three were unconditionally overwritten on every
    // re-pick — silently wiping user's ProfileStep adjustments. Extend
    // iter-66 M144's pattern to all fields that applyDefaults seeds.

    func test_sourceStep_method_preservation_uses_settings_default_method() throws {
        let src = try stepSource("SourceStep.swift")
        // method overwrite must now be guarded by comparing plan.method
        // to what settings.defaultMethod would have seeded.
        XCTAssertTrue(
            src.contains("if plan.method == seedMethod"),
            "applyRecommendation must preserve method when user manually changed it. See M145 iter 67."
        )
    }

    func test_sourceStep_hadamard_preservation_uses_settings_default() throws {
        let src = try stepSource("SourceStep.swift")
        XCTAssertTrue(
            src.contains("if plan.hadamard == settings.defaultHadamardEnabled"),
            "applyRecommendation must preserve hadamard when user manually toggled it. See M145 iter 67."
        )
    }

    func test_sourceStep_forceDtype_preservation_via_nil_guard() throws {
        let src = try stepSource("SourceStep.swift")
        // forceDtype was previously overwritten when rec had one. Now:
        // only set if user hasn't already chosen an override (nil).
        XCTAssertTrue(
            src.contains("if plan.overrides.forceDtype == nil"),
            "applyRecommendation must preserve forceDtype when user set an override. See M145 iter 67."
        )
    }

    func test_sourceStep_applyRecommendation_uses_settings_default() throws {
        let src = try stepSource("SourceStep.swift")
        // The literal hardcoded comparison must be gone.
        XCTAssertFalse(
            src.contains(#"plan.profile == "JANG_4K""#),
            """
            SourceStep.applyRecommendation must NOT compare plan.profile
            against a hardcoded "JANG_4K" literal. Users with a different
            settings.defaultProfile never got the per-source recommendation
            applied. Use the settings.defaultProfile (with "JANG_4K"
            fallback) as the seed-default comparison. See M143 iter 65.
            """
        )
        // The replacement pattern must be present — compares against the
        // settings.defaultProfile (or the empty fallback).
        XCTAssertTrue(
            src.contains("settings.defaultProfile"),
            "SourceStep must reference settings.defaultProfile for the 'user hasn't touched' heuristic."
        )
    }

    // MARK: - M138 (iter 60): RunStep late-Cancel on successful conversion
    //
    // Sibling of M137 in RunStep. Pre-iter-60 code used:
    //     coord.plan.run = cancelRequested ? .cancelled : .succeeded
    // after the for-await exited normally. But PythonRunner treats a
    // cancelled subprocess AND a successful one the same way:
    // continuation.finish() clean, no throw. So a user clicking Cancel
    // at the same microsecond the conversion completed with exit 0 got
    // run=.cancelled — and with autoDeletePartialOnCancel=true, the
    // successful output FOLDER was deleted. Higher data-loss stakes
    // than M137 (which only mis-labeled an already-uploaded HF repo).
    //
    // iter 60 tracks `sawSuccessfulDone` from the final .done(ok: true)
    // event — that's the authoritative "completed successfully" signal.

    func test_runStep_tracks_successful_done_event() throws {
        let src = try stepSource("RunStep.swift")
        // The @State flag must exist.
        XCTAssertTrue(
            src.contains("@State private var sawSuccessfulDone"),
            "RunStep must track sawSuccessfulDone to distinguish cancel vs late-cancel-after-success. See M138 iter 60."
        )
        // The flag must be set when the .done event arrives with ok=true.
        XCTAssertTrue(
            src.contains("sawSuccessfulDone = true"),
            "sawSuccessfulDone must be assigned true in the .done(ok:) case."
        )
        // The flag must be checked before falling back to cancelRequested.
        XCTAssertTrue(
            src.contains("if sawSuccessfulDone"),
            "The stream-complete branch must check sawSuccessfulDone BEFORE the cancelRequested fallback."
        )
    }

    // MARK: - M137 (iter 59): Publish sheet race — late-Cancel shouldn't show
    // "cancelled" on an already-completed upload.
    //
    // Pre-iter-59: runPublish() checked `if wasCancelled { error } else
    // { success }` AFTER the for-await loop exited naturally. If the user
    // clicked Cancel microseconds after the final upload event landed,
    // wasCancelled got set to true, loop exited normally, user saw
    // "Upload cancelled" despite the HF repo having the complete files.
    // Post-iter-59: CancellationError catch is the authoritative cancelled
    // branch. Natural loop exit is always treated as success.

    func test_publishSheet_treats_natural_completion_as_success() throws {
        // Reach into the sheet source file (peer to the step files).
        let parent = Self.stepsRoot.deletingLastPathComponent()
        let url = parent.appendingPathComponent("PublishToHuggingFaceSheet.swift")
        let src = try String(contentsOf: url, encoding: .utf8)
        // After the for-await loop, the code must NOT branch on
        // `if wasCancelled` before setting publishResult. The
        // authoritative cancel signal is `catch is CancellationError`.
        XCTAssertTrue(
            src.contains("catch is CancellationError"),
            """
            PublishToHuggingFaceSheet.runPublish must catch CancellationError
            specifically to signal user-initiated cancel. Without this, a
            late-cancel click (post-completion) would set wasCancelled=true
            and the user would see "cancelled" despite the HF repo having
            the files. See M137 in iter 59.
            """
        )
        // Verify the buggy pre-iter-59 pattern is gone: there must NOT be
        // an `if wasCancelled {` immediately followed by errorMessage =
        // "Upload cancelled" inside the do-block (pre-for-await-loop-exit
        // natural branch).
        let hasBuggyPattern = src.contains(#"if wasCancelled {"#) &&
            src.range(of: #"if wasCancelled \{\s*\n\s*//[^\n]*\n\s*errorMessage"#,
                      options: .regularExpression) != nil
        XCTAssertFalse(
            hasBuggyPattern,
            "pre-iter-59 pattern `if wasCancelled { errorMessage = ... }` on the natural-completion path re-introduced."
        )
    }

    func test_sourceStep_guards_mutations_with_isCancelled() throws {
        let src = try stepSource("SourceStep.swift")
        // detectAndRecommend must have `guard !Task.isCancelled else { return }`
        // before mutating coord.plan.detected / recommendation / errorText.
        // Count the guards — expect at least 3 (one per mutation site after
        // a suspension point).
        let guardCount = src.components(separatedBy: "guard !Task.isCancelled else { return }").count - 1
        XCTAssertGreaterThanOrEqual(
            guardCount, 3,
            """
            detectAndRecommend must guard state mutations with
            `guard !Task.isCancelled else { return }` after every await
            suspension point. Found \(guardCount) guards; need at least 3
            (Step A success, Step A error, Step B success/error).
            """
        )
    }
}
