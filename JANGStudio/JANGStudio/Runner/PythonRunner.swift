// JANGStudio/JANGStudio/Runner/PythonRunner.swift
import Foundation

struct ProcessError: Error, Equatable, LocalizedError {
    let code: Int32
    let lastStderr: String

    /// M169 (iter 92): applies iter-90 M167 / iter-91 M168's "surface
    /// remediation, not just symptom" meta-lesson to the convert-subprocess
    /// error path. Pre-M169, `RunStep.swift:187` stringified ProcessError
    /// directly (`logs.append("[ERROR] \(error)")` → ugly
    /// `ProcessError(code: 1, lastStderr: "...")` print format), leaving the
    /// user with raw stderr and no next-action guidance. Now the log line
    /// reads "jang-tools convert exited X: <stderr>\n→ <remediation>"
    /// with a tiered hint system that covers the four most common convert
    /// failure modes plus a generic fallback.
    var errorDescription: String? {
        let trimmed = lastStderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = trimmed.isEmpty
            ? "jang-tools convert exited \(code)"
            : "jang-tools convert exited \(code): \(trimmed)"
        return "\(body)\n→ \(Self.remediation(code: code, stderr: trimmed))"
    }

    /// Tiered remediation matching iter-91 M168's substring + fallback
    /// design. Substring (not regex) + case-insensitive survives upstream
    /// error-message tweaks across MLX / HF / Python versions.
    nonisolated static func remediation(code: Int32, stderr: String) -> String {
        let lower = stderr.lowercased()

        // OOM / allocation failures — MLX says "Failed to allocate N bytes",
        // CPython says "MemoryError", macOS OOM-killer uses SIGKILL (exit 137).
        let oomSignals = ["failed to allocate", "memoryerror", "cannot allocate memory", "out of memory"]
        if oomSignals.contains(where: { lower.contains($0) }) || code == 137 || lower.contains("killed") {
            return "Convert ran out of memory. Try a smaller profile (e.g., JANG_2L or JANG_3L instead of JANG_4K), close other apps to free RAM, or run on a larger Mac (128+ GB recommended for 256+ expert models)."
        }

        // Disk full — Python OSError 28 / POSIX ENOSPC.
        if lower.contains("no space left") || lower.contains("[errno 28]") || lower.contains("disk quota") {
            return "Out of disk space. Convert output needs roughly source-size × (avg-bits / 16). Free up space, or pick a different output folder on a larger volume."
        }

        // Missing trust_remote_code modules — MiniMax, Cascade, custom architectures.
        // Python surfaces this as `ModuleNotFoundError: No module named 'modeling_X'`.
        if lower.contains("no module named 'modeling_")
            || lower.contains("trust_remote_code")
            || (lower.contains("modulenotfounderror") && lower.contains("modeling_")) {
            return "Model uses custom code (trust_remote_code) but modeling_*.py is missing from the source folder. Re-download INCLUDING .py files: `huggingface-cli download <repo> --include '*.py' --include '*.safetensors' --include '*.json'`."
        }

        // Corrupt safetensors shard — user needs to re-download (possibly a
        // cache corruption or interrupted transfer).
        if lower.contains("safetensorserror") || lower.contains("header too big")
            || lower.contains("invalid header") || lower.contains("corrupt") {
            return "Shard file appears corrupt. Re-download the source model (`huggingface-cli download <repo>` or `git clone`), then retry."
        }

        return "Check the log pane above for details, or click Copy Diagnostics to bundle logs for a bug report. Retrying with a smaller profile (JANG_2L / JANG_3L) often helps if the root cause is memory pressure."
    }
}

actor PythonRunner {
    nonisolated let executable: URL
    nonisolated let extraArgs: [String]
    private var process: Process?
    private var cancelled = false

    init(executableOverride: URL? = nil, extraArgs: [String]) {
        self.executable = executableOverride ?? BundleResolver.pythonExecutable
        self.extraArgs = extraArgs
    }

    // `nonisolated` — can be called from any context; the detached Task
    // inside hops back to the actor via `await self.launch(...)`.
    //
    // M98 (iter 31): wires `continuation.onTermination` → `cancel()` so
    // consumer-Task cancellation OR stream abandonment propagates to the
    // subprocess. Without this, iter-3's cancel() pattern only covered the
    // EXPLICIT `await runner.cancel()` path; a SwiftUI view dismount or
    // abandoned iterator would leave the subprocess orphaned + running.
    // Same class of bug iter-30 (M96) caught in PublishService.
    nonisolated func run() -> AsyncThrowingStream<ProgressEvent, Error> {
        AsyncThrowingStream { continuation in
            // M98 (iter 31): onTermination is registered INSIDE launch() after
            // the continuation has a live producer. Registering it in the
            // outer build-closure was unreliable — Swift 6 fires the
            // termination callback with reason=.cancelled on stream
            // construction under some isolation contexts (observed on XCTest
            // Task.detached consumers). Moving the registration into launch()
            // defers it until spawn is complete, which avoids that race.
            Task.detached {
                await self.launch(continuation: continuation)
            }
        }
    }

    private func launch(
        continuation: AsyncThrowingStream<ProgressEvent, Error>.Continuation
    ) async {
        let proc = Process()
        proc.executableURL = executable
        proc.arguments = extraArgs
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        // Merge user-settings-driven env vars (tick throttle, thread count,
        // PYTHONPATH prepend). These flow from AppSettings → UserDefaults
        // leaf keys → BundleResolver.childProcessEnvAdditions. See M62 in
        // the Ralph audit log.
        for (k, v) in BundleResolver.childProcessEnvAdditions(inherited: env) {
            env[k] = v
        }
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        self.process = proc

        // M98 (iter 31): register termination handler AFTER the subprocess
        // is constructed, so stream abandonment / consumer cancel mid-run
        // propagates SIGTERM via runner.cancel(). Registering in run()'s
        // outer build-closure fired spuriously with reason=.cancelled under
        // some Task-isolation contexts observed in XCTest.
        continuation.onTermination = { _ in
            Task.detached { await self.cancel() }
        }

        // M221: surface stdout as ProgressEvents instead of silently draining.
        // Non-JSON lines (raw Python prints, jang-tools log lines that aren't
        // JSONL progress events) are yielded as `info`-level messages so the
        // UI can show them in the log pane or as status text even when the
        // JSONL tick stream stalls. JSONL lines are parsed and yielded as
        // structured events (phase/tick/message/done) so they work the same
        // as stderr JSONL for tools that emit to either fd.
        let stdoutTask = Task.detached {
            let parser = JSONLProgressParser()
            for try await line in outPipe.fileHandleForReading.bytes.lines {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                if let ev = parser.parse(line: line) {
                    continuation.yield(ev)
                } else {
                    // Non-JSON line — figure the severity from prefix heuristics
                    // so the UI renders it at the right level.
                    let lower = trimmed.lowercased()
                    let level: String
                    if lower.hasPrefix("error") || lower.hasPrefix("[error")
                        || lower.contains("traceback") {
                        level = "error"
                    } else if lower.hasPrefix("warn") || lower.hasPrefix("[warn")
                        || lower.contains("warning") {
                        level = "warn"
                    } else {
                        level = "info"
                    }
                    continuation.yield(ProgressEvent(
                        ts: Date().timeIntervalSince1970,
                        type: level == "error" ? .error : (level == "warn" ? .warn : .info),
                        payload: .message(level: level, text: trimmed)
                    ))
                }
            }
        }

        // Drain stderr — parser + lastErrTail are owned exclusively by this task
        // (avoids the Swift 6 shared-mutable-state error).
        let stderrTask = Task.detached {
            let parser = JSONLProgressParser()
            var lastErrTail = ""
            for try await line in errPipe.fileHandleForReading.bytes.lines {
                lastErrTail = String(line.suffix(256))
                if let ev = parser.parse(line: line) {
                    continuation.yield(ev)
                }
            }
            return lastErrTail
        }

        do {
            try proc.run()
        } catch {
            try? outPipe.fileHandleForWriting.close()
            try? errPipe.fileHandleForWriting.close()
            _ = await stdoutTask.result
            _ = await stderrTask.result
            continuation.finish(throwing: error)
            return
        }
        try? outPipe.fileHandleForWriting.close()
        try? errPipe.fileHandleForWriting.close()

        if cancelled, proc.isRunning {
            Self.terminateWithEscalation(proc)
        }

        // Wait for termination off the actor thread so cancel() can acquire
        // the actor and call proc.terminate() while we are waiting.
        await Task.detached { proc.waitUntilExit() }.value

        if cancelled {
            try? outPipe.fileHandleForReading.close()
            try? errPipe.fileHandleForReading.close()
        }

        _ = await stdoutTask.result
        let lastErrTail = (try? await stderrTask.value) ?? ""

        if proc.terminationStatus == 0 {
            continuation.finish()
        } else if cancelled {
            continuation.finish()    // cancellation is a clean exit
        } else {
            continuation.finish(
                throwing: ProcessError(
                    code: proc.terminationStatus,
                    lastStderr: lastErrTail
                )
            )
        }
    }

    func cancel() {
        cancelled = true
        guard let proc = process, proc.isRunning else { return }
        Self.terminateWithEscalation(proc)
    }

    private nonisolated static func terminateWithEscalation(_ proc: Process) {
        proc.terminate()   // SIGTERM
        Task.detached {
            try? await Task.sleep(for: .seconds(3))
            if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        }
    }
}
