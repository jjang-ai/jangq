// JANGStudio/JANGStudio/Runner/InferenceRunner.swift
import Foundation

struct InferenceResult: Codable, Equatable, Sendable {
    let text: String
    let tokens: Int
    let tokensPerSec: Double
    let elapsedS: Double
    let loadTimeS: Double?
    let peakRssMb: Double
    let model: String

    enum CodingKeys: String, CodingKey {
        case text, tokens, model
        case tokensPerSec = "tokens_per_sec"
        case elapsedS = "elapsed_s"
        case loadTimeS = "load_time_s"
        case peakRssMb = "peak_rss_mb"
    }
}

struct InferenceError: Error, Equatable, Sendable {
    let message: String
    let code: Int32

    /// Sentinel code for user-requested cancellation. Callers can filter on
    /// `error.code == InferenceError.cancelledCode` to avoid surfacing cancel
    /// as a failure in the UI.
    static let cancelledCode: Int32 = -2

    var wasCancelled: Bool { code == Self.cancelledCode }
}

/// Wraps `python -m jang_tools inference` as a one-shot subprocess.
/// Not streaming in v1 — returns the full response at once. Each call spawns
/// a fresh Python process that loads the model, generates, exits.
///
/// **Cancellation:** `generate` suspends on a `CheckedContinuation` tied to
/// `proc.terminationHandler` so the actor is NOT held during the wait.
/// `cancel()` can then acquire the actor, set `cancelled = true`, and fire
/// SIGTERM. This is the same pattern we applied to PythonRunner in commit
/// `6270214` — without it, the actor deadlocks and the cancel button in
/// TestInferenceSheet would never take effect.
actor InferenceRunner {
    nonisolated let modelPath: URL
    /// Executable to run. Production uses `BundleResolver.pythonExecutable`;
    /// tests pass a short-sleep shell script to exercise the cancel / timeout
    /// paths without needing an actual Python + model to load. Mirrors the
    /// `executableOverride` pattern on PythonRunner.
    nonisolated let executableOverride: URL?
    private var currentProcess: Process?
    private var cancelled: Bool = false

    init(modelPath: URL, executableOverride: URL? = nil) {
        self.modelPath = modelPath
        self.executableOverride = executableOverride
    }

    func generate(prompt: String,
                  maxTokens: Int = 100,
                  temperature: Double = 0.0,
                  imagePath: URL? = nil,
                  videoPath: URL? = nil,
                  noThinking: Bool = false) async throws -> InferenceResult {
        // Reset cancel flag for this call — each generate is independent.
        cancelled = false

        var args = [
            "-m", "jang_tools", "inference",
            "--model", modelPath.path,
            "--prompt", prompt,
            "--max-tokens", String(maxTokens),
            "--temperature", String(temperature),
            "--json",
        ]
        if let image = imagePath {
            args += ["--image", image.path]
        }
        if let video = videoPath {
            args += ["--video", video.path]
        }
        // M121 (iter 45): opt-in short-answer mode for reasoning models.
        // When the user ticks "Skip thinking" in TestInferenceSheet, we
        // pass --no-thinking so the chat template doesn't wrap the prompt
        // with a <think>…</think> block that would eat the 150-token
        // smoke-test budget on GLM-5.1 / Qwen3.6 / MiniMax M2.7.
        if noThinking {
            args.append("--no-thinking")
        }

        let proc = Process()
        proc.executableURL = executableOverride ?? BundleResolver.pythonExecutable
        proc.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        // M62 env-passthrough: user settings → child env.
        for (k, v) in BundleResolver.childProcessEnvAdditions(inherited: env) {
            env[k] = v
        }
        proc.environment = env

        let out = Pipe()
        let err = Pipe()
        proc.standardOutput = out
        proc.standardError = err
        self.currentProcess = proc

        // M159 (iter 82): drain both pipes in detached tasks BEFORE run().
        // Previously both streams were read AFTER terminationHandler fired,
        // via synchronous `readDataToEndOfFile()`. That's a classic
        // cross-process deadlock: the subprocess can't exit until its
        // write(2) calls unblock, and its write(2) calls can't unblock until
        // we drain the 64 KB pipe buffer, and we don't drain until it exits.
        // MLX inference can easily emit >64 KB of stderr logs on chatty
        // models (GLM-5.1 per-layer warnings, MiniMax weight-load chatter),
        // so this hit users in practice as a Test Inference sheet that
        // spins forever and only unsticks when they hit Cancel.
        //
        // Matches PythonRunner's pattern: read tasks start immediately,
        // Task.detached runs on a separate thread so the blocking
        // `readDataToEndOfFile()` doesn't hold the actor. When the
        // subprocess finally closes the pipe's write end (either by exit
        // or by explicit close below on run() failure), the read returns
        // and the task completes.
        let stdoutTask = Task.detached { out.fileHandleForReading.readDataToEndOfFile() }
        let stderrTask = Task.detached { err.fileHandleForReading.readDataToEndOfFile() }

        do {
            try proc.run()
        } catch {
            // If run() throws (e.g., executable missing), the pipes never
            // get a writer — we have to close the write ends explicitly so
            // the detached reads can see EOF and drain. Otherwise both
            // tasks leak until the pipes go out of scope via ARC, which
            // can be deferred unpredictably under Swift's cooperative
            // executor.
            try? out.fileHandleForWriting.close()
            try? err.fileHandleForWriting.close()
            _ = await stdoutTask.value
            _ = await stderrTask.value
            throw error
        }

        // Wait for termination OFF the actor thread so cancel() can acquire
        // the actor and call proc.terminate() while we are waiting.
        // Without this, the actor is held inside waitUntilExit() and cancel()
        // queues indefinitely, defeating the Cancel button.
        //
        // M100 (iter 32): wrap the CheckedContinuation in a
        // withTaskCancellationHandler so consumer-Task cancellation ALSO
        // terminates the subprocess. Iter 3's M19 fix handled explicit
        // `await runner.cancel()`, but the Task-cancel path was never wired.
        // Without this wrap, cancelling the consuming Task leaves the
        // subprocess running to natural completion (wasted load, stale
        // inference, orphaned GPU allocation). Same pattern iter 30/31
        // applied to publish + PythonRunner.
        await withTaskCancellationHandler {
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                proc.terminationHandler = { _ in done.resume() }
            }
        } onCancel: {
            // `onCancel` is a Sendable, nonisolated closure. We can't await
            // the actor's `cancel()` directly — hop via Task.detached.
            Task.detached { await self.cancel() }
        }

        let stdout = await stdoutTask.value
        let stderr = await stderrTask.value
        let errorText = String(data: stderr, encoding: .utf8) ?? ""

        // If the user cancelled, surface a specific error so UI can distinguish
        // cancel from a real failure.
        if cancelled {
            throw InferenceError(
                message: "generation cancelled by user",
                code: InferenceError.cancelledCode
            )
        }

        guard proc.terminationStatus == 0 else {
            throw InferenceError(
                message: errorText.isEmpty
                    ? "inference exited with code \(proc.terminationStatus)"
                    : errorText.trimmingCharacters(in: .whitespacesAndNewlines),
                code: proc.terminationStatus
            )
        }

        // jang_tools inference emits a SINGLE JSON line at the end. Strip anything before.
        let raw = String(data: stdout, encoding: .utf8) ?? ""
        guard let jsonLine = raw.split(whereSeparator: \.isNewline).last(where: { $0.hasPrefix("{") }) else {
            throw InferenceError(
                message: "inference output did not include a JSON line: \(raw.suffix(200))",
                code: -1
            )
        }
        do {
            let data = Data(jsonLine.utf8)
            // Check for error shape first
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let errorStr = obj["error"] as? String {
                throw InferenceError(message: errorStr, code: -1)
            }
            return try JSONDecoder().decode(InferenceResult.self, from: data)
        } catch let e as InferenceError {
            throw e
        } catch {
            throw InferenceError(message: "inference JSON decode failed: \(error)", code: -1)
        }
    }

    func cancel() {
        cancelled = true
        guard let p = currentProcess, p.isRunning else { return }
        p.terminate()   // SIGTERM
        // Escalate to SIGKILL after 3s if the process ignores SIGTERM.
        // Capture `p` strongly so even if a new generate() is started,
        // this timer targets the correct (old) process.
        Task.detached {
            try? await Task.sleep(for: .seconds(3))
            if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        }
    }
}
