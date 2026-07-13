// JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift
import SwiftUI

struct RunStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(AppSettings.self) private var settings
    // M204 (iter 139): inject ProfilesService so start() can re-estimate
    // output size for the final disk-space re-check. Matches the
    // ProfileStep injection pattern (same service, same usage).
    @Environment(ProfilesService.self) private var profilesSvc
    @State private var phase: (n: Int, total: Int, name: String) = (0, 5, "idle")
    @State private var tick: (done: Int, total: Int, label: String)? = nil
    @State private var logs: [String] = []
    @State private var runner: PythonRunner?
    @State private var startedAt: Date?
    /// M221: elapsed-time display, updated by a 1s Timer.publish while running.
    @State private var elapsedTime: TimeInterval = 0
    @State private var cancelRequested: Bool = false
    /// M138 (iter 60): authoritative success marker. Pre-iter-60, RunStep
    /// used `cancelRequested ? .cancelled : .succeeded` to decide outcome
    /// after the stream exited. But PythonRunner treats a cancelled AND a
    /// successful subprocess THE SAME WAY: `continuation.finish()` clean,
    /// no throw. So a user hitting Cancel at the same microsecond the
    /// conversion completed with exit 0 would get run=.cancelled even
    /// though the output is fully written. Worse: if the user had
    /// "Auto-delete partial output on cancel" enabled, the successful
    /// output folder was DELETED. Same race class as iter-59 M137
    /// (Publish), but with data-loss stakes.
    ///
    /// Track whether we received a final `.done(ok: true, …)` event —
    /// THAT is the authoritative "conversion completed successfully"
    /// signal. A cancel that preempted the final write won't emit
    /// ok=true, so `sawSuccessfulDone` stays false and we correctly
    /// report .cancelled.
    @State private var sawSuccessfulDone: Bool = false
    /// M170 (iter 93): handle to the active conversion Task. Pre-M170,
    /// .onAppear spawned `Task { await start() }` with no handle — when the
    /// user closed the main window (red-X) or quit the app (cmd-Q) mid-
    /// convert, SwiftUI unmounted RunStep, the `runner` @State was lost,
    /// but this Task kept running and the Python convert subprocess
    /// continued writing to disk for up to 30 more minutes as an orphaned
    /// child of launchd. Mac stayed at 100% CPU with no UI to cancel from.
    /// Same class as iter-85 M162's sheet-dismiss orphan but for the main-
    /// window view. The .onDisappear hook cancels this handle, which
    /// propagates through iter-32 M100's withTaskCancellationHandler →
    /// PythonRunner.cancel() → SIGTERM + 3 s SIGKILL escalation.
    @State private var runTask: Task<Void, Never>?

    var body: some View {
        Form {
            Section(runStatusSectionTitle) {
            HStack {
                Text(phaseDisplayTitle).font(.headline)
                Spacer()
                if coord.plan.run == .running {
                    Button("Cancel", role: .destructive) {
                        cancelRequested = true
                        Task { await runner?.cancel() }
                    }
                    .disabled(cancelRequested)
                }
            }
            ProgressView(value: Double(phase.n), total: Double(phase.total))
            // M221: elapsed time display with ETA estimate for tick-based progress.
            if startedAt != nil, coord.plan.run == .running {
                HStack(spacing: 6) {
                    Image(systemName: "clock")
                        .foregroundStyle(.secondary)
                        .font(.caption2)
                    Text(elapsedTimeString)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    if let t = tick, t.done > 3, t.total > t.done, elapsedTime > 5 {
                        let estimated = (elapsedTime / Double(t.done)) * Double(t.total - t.done)
                        Text("· ~\(etaString(estimated)) remaining")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .monospacedDigit()
                    }
                }
            }
            if let t = tick {
                ProgressView(value: Double(t.done), total: Double(t.total)) {
                    Text(t.label)
                        .font(.caption)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
            }
            }

            if coord.plan.run == .succeeded {
                Section(successSectionTitle) {
                    Button("Continue → Verify") { coord.active = .verify }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                }
            } else if coord.plan.run == .cancelled {
                Section("Recovery") {
                    Label(cancelledSummary, systemImage: "stop.circle.fill")
                        .foregroundStyle(.orange)
                    HStack {
                        Button("Retry") {
                            cancelRequested = false
                            runTask?.cancel()
                            runTask = Task { await start() }
                        }
                            .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                        Button("Delete partial output", role: .destructive) {
                            if let out = coord.plan.outputURL {
                                // M107 (iter 35): surface delete failures via the
                                // existing log pane so the user doesn't assume the
                                // delete succeeded when it didn't (permission
                                // denied, file in use by another process, already
                                // gone from disk). Pre-iter-35 the `try?` silently
                                // swallowed every error; user walked away thinking
                                // the output was cleaned up when it wasn't.
                                //
                                // M23 (iter 97): distinguish "already gone" from
                                // real failure. Pre-M23 the catch branch reported
                                // "delete FAILED: No such file or directory" when
                                // the folder had already been cleaned up externally
                                // (manual rm, auto-delete-on-cancel ran, prior
                                // click already succeeded). User sees "FAILED" but
                                // the goal state is achieved — misleading. Detect
                                // NSFileNoSuchFileError → "already gone" success
                                // message. Real failures still show the error.
                                do {
                                    try FileManager.default.removeItem(at: out)
                                    logs.append("[cleanup] deleted \(out.path)")
                                } catch CocoaError.fileNoSuchFile {
                                    logs.append("[cleanup] \(out.path) — already gone (nothing to delete)")
                                } catch {
                                    logs.append("[cleanup] delete FAILED: \(error.localizedDescription)")
                                }
                            }
                        }
                        .disabled(coord.plan.outputURL == nil)
                    }
                }
            } else if coord.plan.run == .failed {
                Section("Recovery") {
                    ExpertLabConsoleCard(accent: ExpertLabVisual.danger) {
                        VStack(alignment: .leading, spacing: 10) {
                            ExpertLabKicker(text: coord.plan.expertReviewIntent == .smartPrequantPrune ? "Legacy review bundle failed" : "Conversion failed",
                                            color: ExpertLabVisual.danger)
                            Label(failureTitle, systemImage: "xmark.octagon.fill")
                                .font(.headline)
                                .foregroundStyle(ExpertLabVisual.danger)
                            Text(failureSummary)
                                .font(.caption)
                                .foregroundStyle(Color.white.opacity(0.72))
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                            HStack {
                                Button {
                                    cancelRequested = false
                                    runTask?.cancel()
                                    runTask = Task { await start() }
                                } label: {
                                    Label("Retry", systemImage: "arrow.clockwise")
                                }
                                .buttonStyle(.borderedProminent)
                                Button {
                                    writeDiagnostics()
                                } label: {
                                    Label("Copy Diagnostics", systemImage: "doc.on.doc")
                                }
                            }
                        }
                    }
                }
            }

            Section("Log") {
                ScrollView {
                    Text(logs.suffix(500).joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(Color.white.opacity(0.84))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxWidth: .infinity, minHeight: coord.plan.run == .failed ? 180 : 300)
                .padding(10)
                .background(Color.black.opacity(0.42))
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(ExpertLabVisual.line, lineWidth: 1)
                }
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(ExpertLabVisual.canvas)
        .padding()
        .environment(\.colorScheme, .dark)
        // M136 (iter 58): only auto-start on first entry (.idle). Without
        // the run-state check, SwiftUI's .onAppear fires every time the view
        // reappears — e.g., when the user nav-backs from VerifyStep via the
        // sidebar to inspect logs. `start()`'s only guard was
        // `run != .running`, so a completed / failed / cancelled run got
        // restarted on nav-back, wiping logs + overwriting the finished
        // output folder. Retry buttons below still call `start()` directly
        // (they rely on the weaker guard inside start()); only the
        // auto-start path needs the tighter gate.
        .onAppear {
            if coord.plan.run == .idle {
                runTask = Task { await start() }
            }
        }
        .onDisappear {
            // M170 (iter 93): cancel the conversion when RunStep unmounts.
            // SwiftUI fires .onDisappear on window close (red-X), tab switch
            // away, and app quit (cmd-Q). Without this hook, the Python
            // convert subprocess kept running for up to 30 more minutes as
            // an orphaned child of launchd — user saw Mac pegged at 100%
            // CPU with no UI to cancel from. The cancel propagates through
            // iter-32 M100's withTaskCancellationHandler → runner.cancel()
            // → SIGTERM + 3 s SIGKILL. Matches iter-85 M162 sheet pattern
            // but for the main-window view.
            runTask?.cancel()
        }
        // M221: 1-second timer to update elapsed-time display while running.
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
            guard coord.plan.run == .running, let s = startedAt else { return }
            elapsedTime = Date().timeIntervalSince(s)
        }
    }

    private func start() async {
        guard coord.plan.run != .running else { return }
        if isFinalQuantFromReviewedPrune,
           let check = PreflightRunner.reviewedPruneVerifiedCheck(plan: coord.plan),
           check.status != .pass {
            coord.plan.run = .failed
            logs.removeAll()
            phase = (0, 5, "blocked")
            logs.append("[preflight] Reviewed BF16/F16 prune verification blocked final quantization.")
            if let hint = check.hint, !hint.isEmpty {
                logs.append("[preflight] \(hint)")
            }
            return
        }
        // M204 (iter 139): cheap final disk-space re-check BEFORE spawning
        // the Python subprocess. The Step-3 preflight can be minutes stale
        // (user reads the preview card, disk fills via a concurrent
        // download or Time Machine snapshot). Pre-M204, a user whose disk
        // filled between preflight-green and Start ran convert for 20+
        // minutes before ENOSPC mid-shard, then had to clean up partial
        // output. The re-check reuses PreflightRunner's estimator so
        // "we have N GB / need M GB" math is consistent with Step 3.
        if let dst = coord.plan.outputURL {
            let parent = dst.deletingLastPathComponent()
            let rv = try? parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            let free = Int64(rv?.volumeAvailableCapacityForImportantUsage ?? 0)
            let estimated = PreflightRunner.estimateOutputBytes(
                plan: coord.plan,
                profiles: profilesSvc.profiles
            )
            if estimated > 0 && free < estimated {
                coord.plan.run = .failed
                logs.append(
                    "[preflight] Disk space dropped below the estimated "
                    + "output size since preflight. Need ~\(estimated / 1_000_000_000) GB, "
                    + "have \(free / 1_000_000_000) GB free on the output volume. "
                    + "Free up space or pick a different output folder, then retry."
                )
                return
            }
        }
        // PR1 (hard-fail unknown JANGTQ module): refuse empty argv BEFORE
        // setting run=.running. Preflight should catch whitelist/module gaps
        // at Profile, but VL wrapper skew or a forced family can still reach
        // Run. Checking here avoids a flash of "running" and prevents
        // PythonRunner from spawning with empty arguments.
        if let reason = CLIArgsBuilder.failureReason(for: coord.plan) {
            coord.plan.run = .failed
            logs.removeAll()
            phase = (0, 5, "blocked")
            logs.append("[error] \(reason)")
            return
        }
        let args = buildArgs()
        coord.plan.run = .running
        cancelRequested = false
        sawSuccessfulDone = false   // M138: reset for the new run.
        logs.removeAll()
        startedAt = Date()
        let r = PythonRunner(extraArgs: args)
        runner = r
        do {
            for try await ev in r.run() {
                await MainActor.run { apply(ev) }
            }
            // Stream finished without throwing. M138 (iter 60): use the
            // authoritative success signal (final `.done(ok: true, …)`
            // event) rather than the user's cancel-intent flag. A late
            // cancel click that landed AFTER the subprocess naturally
            // completed would otherwise set run=.cancelled and —
            // catastrophically, when autoDeletePartialOnCancel=true —
            // delete the successfully-written output folder.
            await MainActor.run {
                if sawSuccessfulDone {
                    coord.plan.run = .succeeded
                    if cancelRequested {
                        // Document the race outcome so the user who hit
                        // Cancel understands the subprocess beat them.
                        logs.append("[note] Cancel click landed after the final write — output is complete.")
                    }
                } else {
                    coord.plan.run = cancelRequested ? .cancelled : .succeeded
                    if cancelRequested {
                        logs.append("[cancelled] SIGTERM acknowledged, process exited")
                        // M62: honor Settings → General → Behavior →
                        // "Auto-delete partial output on cancel". Previously inert.
                        if settings.autoDeletePartialOnCancel, let out = coord.plan.outputURL {
                            do {
                                try FileManager.default.removeItem(at: out)
                                logs.append("[cancelled] deleted partial output at \(out.path) (auto-delete setting on)")
                            } catch {
                                logs.append("[cancelled] auto-delete failed: \(error.localizedDescription)")
                            }
                        }
                    }
                }
            }
        } catch {
            await MainActor.run {
                coord.plan.run = .failed
                // M169 (iter 92): use localizedDescription so the log shows
                // the tiered remediation (exit code + stderr + actionable
                // next-step) instead of the raw `ProcessError(code:…, stderr:…)`
                // struct print. For non-ProcessError types, localizedDescription
                // falls back to the platform default — still better than `\(error)`.
                logs.append("[ERROR] \(error.localizedDescription)")
            }
        }
    }

    private var isReviewRuntimeBuild: Bool {
        coord.plan.expertReviewIntent == .smartPrequantPrune
    }

    private var isFinalQuantFromReviewedPrune: Bool {
        coord.plan.expertReviewPrunedSourceURL != nil
    }

    private var runStatusSectionTitle: String {
        if isReviewRuntimeBuild { return "Legacy review bundle build" }
        if isFinalQuantFromReviewedPrune { return "Quantize with \(coord.plan.family == .jangtq ? "JANGTQ" : "JANG")" }
        return "Run status"
    }

    private var successSectionTitle: String {
        if isReviewRuntimeBuild { return "Legacy review bundle ready" }
        if isFinalQuantFromReviewedPrune { return "Post-quant same-suite verification ready" }
        return "Ready to verify"
    }

    private var phaseDisplayTitle: String {
        let name = isReviewRuntimeBuild ? reviewRuntimePhaseName(phase.name) : finalQuantPhaseName(phase.name)
        return "Phase \(phase.n)/\(phase.total) · \(name)"
    }

    private var cancelledSummary: String {
        isReviewRuntimeBuild
            ? "Cancelled — partial legacy review bundle left on disk at the output folder."
            : "Cancelled — partial output left on disk at output folder."
    }

    private func reviewRuntimePhaseName(_ name: String) -> String {
        switch name.lowercased() {
        case "scan": return "scan source"
        case "convert": return "build legacy review bundle"
        case "write", "finalize": return "write legacy review artifacts"
        default: return name
        }
    }

    private func finalQuantPhaseName(_ name: String) -> String {
        guard isFinalQuantFromReviewedPrune else { return name }
        switch name.lowercased() {
        case "scan": return "scan verified pruned source"
        case "convert": return "quantize pruned source"
        case "validate": return "validate post-quant bundle"
        case "write", "finalize": return "write final JANG artifacts"
        default: return name
        }
    }

    private var failureTitle: String {
        coord.plan.expertReviewIntent == .smartPrequantPrune
            ? "Legacy review bundle build failed"
            : "Conversion failed"
    }

    private var failureSummary: String {
        let interesting = failureIssueLine
        let prefix = coord.plan.expertReviewIntent == .smartPrequantPrune
            ? "The legacy review bundle was not written. BF16/vMLX Expert Review can still run from the original source; this compatibility build did not finish."
            : "The output was not marked complete."
        return "\(prefix) Last reported issue: \(interesting)"
    }

    private var failureIssueLine: String {
        let reversedLogs = logs.reversed()
        let raw = reversedLogs.first { $0.contains("[done] error=") }
            ?? reversedLogs.first { ($0.contains("ValueError") || $0.contains("RuntimeError")) && !$0.contains("[ERROR]") }
            ?? reversedLogs.first { $0.contains("[ERROR]") }
            ?? logs.last
            ?? "No failure detail was reported."
        return raw.components(separatedBy: "\n").first ?? raw
    }

    private func writeDiagnostics() {
        // M109 (iter 36): `.first!` would crash the app in sandboxed /
        // MDM environments where `.desktopDirectory` isn't available.
        // Fall back to the home directory so Copy Diagnostics always
        // works — the user can move the zip afterward.
        let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory())
        let events = logs.filter { $0.hasPrefix("{") }
        // M62-anonymize: honor Settings → Diagnostics →
        // "Anonymize paths in diagnostics". Otherwise a bug report
        // zip leaks the user's filesystem layout.
        // M107 (iter 35): surface write failure via the existing log
        // pane instead of silently dismissing.
        // M106 (iter 42): switched to writeAsync so the ditto
        // subprocess runs off MainActor. Pre-fix, a large diag
        // bundle (50+ MB of tick events + stderr) could beach-ball
        // the UI for several seconds during zip creation.
        Task {
            do {
                let url = try await DiagnosticsBundle.writeAsync(
                    plan: coord.plan, logLines: logs, eventLines: events,
                    verify: [], to: desktop,
                    anonymizePaths: settings.anonymizePathsInDiagnostics)
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } catch {
                logs.append("[diagnostics] FAILED to write zip: \(error.localizedDescription)")
            }
        }
    }

    private func apply(_ ev: ProgressEvent) {
        switch ev.payload {
        case .phase(let n, let total, let name):
            phase = (n, total, name); tick = nil
            logs.append("[\(n)/\(total)] \(name)")
        case .tick(let done, let total, let label):
            tick = (done, total, label ?? "")
        case .message(let level, let text):
            logs.append("[\(level)] \(text)")
        case .done(let ok, _, let err):
            if ok {
                // M138 (iter 60): authoritative success marker. Python emits
                // exactly one .done event at end-of-run; ok=true means the
                // subprocess completed its final write without error.
                sawSuccessfulDone = true
            } else if let err {
                logs.append("[done] error=\(err)")
            }
        case .versionMismatch(let v): logs.append("[error] protocol version \(v) unsupported")
        case .parseError(let s): logs.append("[parse-err] \(s)")
        }
    }

    private func buildArgs() -> [String] { CLIArgsBuilder.args(for: coord.plan) }

    // M221: formatted elapsed time HH:MM:SS or MM:SS.
    private var elapsedTimeString: String {
        let t = Int(elapsedTime)
        if t >= 3600 {
            return String(format: "%d:%02d:%02d elapsed", t / 3600, (t % 3600) / 60, t % 60)
        }
        return String(format: "%d:%02d elapsed", t / 60, t % 60)
    }

    // M221: format seconds into a concise remaining-time string.
    private func etaString(_ seconds: TimeInterval) -> String {
        let t = Int(seconds)
        if t >= 3600 {
            return String(format: "~%dh %dm", t / 3600, (t % 3600) / 60)
        } else if t >= 60 {
            return String(format: "~%dm %ds", t / 60, t % 60)
        }
        return "<1m"
    }
}
