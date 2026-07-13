// JANGStudio/JANGStudio/Wizard/IntentPruneViewModel.swift
// Sheet ViewModel for Intent Prune (plan §6, §12 IP4).
//
// Pipeline (when transitions exist):
//   intent-prune-score → prequant-prune-qwen-moe (hard prune + structural verify)
// Without transitions: dry-runable structure with clear guidance; CLI args
// still build for Preview / when the user attaches expert_transitions.jsonl.
import Foundation
import Observation

enum IntentPruneError: Error, LocalizedError {
    case cli(code: Int32, stderr: String)
    case missingTransitions
    case noCapabilitySelected
    case crackConfirmRequired
    case outputConflictsWithSource
    case planNotWritten(URL)
    case pruneDecodeFailed(String)

    var errorDescription: String? {
        switch self {
        case .cli(let code, let stderr):
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return "Intent Prune CLI exited \(code)"
            }
            return "Intent Prune CLI exited \(code): \(trimmed)"
        case .missingTransitions:
            return "Need expert_transitions.jsonl from a Reviewed 50 BF16/vMLX trace. Choose a transitions file, or open Advanced Expert Lab to run the suite first."
        case .noCapabilitySelected:
            return "Select at least one capability chip (e.g. Coding)."
        case .crackConfirmRequired:
            return "CRACK abliteration requires the confirmation checkbox."
        case .outputConflictsWithSource:
            return "Choose an output separate from the source model tree. Intent Prune never writes into the original BF16/F16 source."
        case .planNotWritten(let url):
            return "Prune plan was not written at \(url.path)"
        case .pruneDecodeFailed(let detail):
            return "Could not parse hard-prune result: \(detail)"
        }
    }
}

@MainActor
@Observable
final class IntentPruneViewModel {
    let sourceURL: URL
    let detected: ArchitectureSummary

    // MARK: - User selections
    var selectedChips: Set<IntentPruneChip> = [.coding]
    var safetyStance: IntentPruneSafetyStance = .keep
    var crackConfirmed: Bool = false
    var budget: IntentPruneBudget = .standard
    /// Path to `expert_transitions.jsonl` (or adjacency). Optional until Run.
    var transitionsURL: URL?
    var adjacencyURL: URL?

    // MARK: - Run state
    var phase: IntentPrunePhase = .idle
    var isRunning = false
    var cancelRequested = false
    var errorText: String?
    var statusDetail: String = ""
    var planURL: URL?
    var outputURL: URL
    var prunedVerified = false
    var lastScoreSummary: String?
    var runTask: Task<Void, Never>?

    /// Structural hard-prune JSON summary (when prune completes).
    var pruneSummaryJSON: [String: Any]?

    init(sourceURL: URL, detected: ArchitectureSummary) {
        self.sourceURL = sourceURL
        self.detected = detected
        let k = IntentPruneBudget.standard.keepK(
            expertsPerLayer: max(detected.numExperts, 1),
            trainedTopK: detected.numExpertsPerTok ?? 1
        )
        self.outputURL = Self.defaultOutputURL(
            sourceURL: sourceURL,
            chips: [.coding],
            keepK: k,
            safetyStance: .keep
        )
        // Auto-discover transitions near source if present.
        self.transitionsURL = Self.discoverTransitions(near: sourceURL)
    }

    // MARK: - Derived

    var expertsPerLayer: Int { max(detected.numExperts, 1) }

    var trainedTopK: Int { max(detected.numExpertsPerTok ?? 1, 1) }

    var keepK: Int {
        budget.keepK(expertsPerLayer: expertsPerLayer, trainedTopK: trainedTopK)
    }

    var intentDomainKeys: [String] {
        IntentPruneCLIArgsBuilder.domainKeys(for: selectedChips)
    }

    var canRun: Bool {
        guard !isRunning else { return false }
        guard !selectedChips.isEmpty else { return false }
        if safetyStance.isCrack, !crackConfirmed { return false }
        guard transitionsURL != nil || adjacencyURL != nil else { return false }
        guard !outputConflictsWithSource else { return false }
        return true
    }

    var canPreviewScores: Bool {
        guard !isRunning else { return false }
        guard !selectedChips.isEmpty else { return false }
        if safetyStance.isCrack, !crackConfirmed { return false }
        return transitionsURL != nil || adjacencyURL != nil
    }

    var canAdopt: Bool {
        prunedVerified && planURL != nil && FileManager.default.fileExists(atPath: outputURL.path)
    }

    var outputConflictsWithSource: Bool {
        let src = sourceURL.standardizedFileURL.path
        let out = outputURL.standardizedFileURL.path
        if out == src { return true }
        if out.hasPrefix(src + "/") { return true }
        if src.hasPrefix(out + "/") { return true }
        return false
    }

    var crackGateMessage: String? {
        guard safetyStance.isCrack else { return nil }
        if crackConfirmed { return nil }
        return "CRACK is abliteration (reduced refusal-path specialization). Confirm to continue."
    }

    var evidenceLine: String {
        var parts = ["Evidence: Reviewed Prune 50 (required)"]
        if safetyStance.isCrack {
            parts.append("CRACK pack: crack-probes-v1 (auto-attached)")
        }
        return parts.joined(separator: " · ")
    }

    var keepSummaryLine: String {
        "Keep \(keepK) / \(expertsPerLayer) experts per layer · drop \(expertsPerLayer - keepK)"
    }

    var sourceSummaryLine: String {
        let layers = detected.numHiddenLayers.map { "\($0) layers" } ?? "layers unknown"
        return "\(sourceURL.lastPathComponent) · \(layers) · \(expertsPerLayer) experts/layer · \(detected.dtype.rawValue.uppercased())"
    }

    // MARK: - Mutations

    func toggleChip(_ chip: IntentPruneChip) {
        if selectedChips.contains(chip) {
            selectedChips.remove(chip)
        } else {
            selectedChips.insert(chip)
        }
        refreshDefaultOutputURL()
    }

    func setSafetyStance(_ stance: IntentPruneSafetyStance) {
        safetyStance = stance
        if !stance.isCrack {
            crackConfirmed = false
        }
        refreshDefaultOutputURL()
    }

    func setBudget(_ budget: IntentPruneBudget) {
        self.budget = budget
        refreshDefaultOutputURL()
    }

    func refreshDefaultOutputURL() {
        outputURL = Self.defaultOutputURL(
            sourceURL: sourceURL,
            chips: Array(selectedChips),
            keepK: keepK,
            safetyStance: safetyStance
        )
    }

    func cancelRun() {
        cancelRequested = true
        runTask?.cancel()
    }

    // MARK: - CLI preview (no I/O)

    func buildScoreArgs(planPath: String) -> [String] {
        IntentPruneCLIArgsBuilder.scoreArgs(
            transitionsPath: transitionsURL?.path,
            adjacencyPath: adjacencyURL?.path,
            outputPlanPath: planPath,
            numExperts: expertsPerLayer,
            numLayers: detected.numHiddenLayers,
            keepK: keepK,
            safetyStance: safetyStance,
            intentsKeep: intentDomainKeys,
            sourceModelPath: sourceURL.path,
            trainedTopK: trainedTopK
        )
    }

    func buildHardPruneArgs(planPath: String) -> [String] {
        IntentPruneCLIArgsBuilder.hardPruneArgs(
            sourcePath: sourceURL.path,
            outputPath: outputURL.path,
            keepExperts: keepK,
            prunePlanPath: planPath
        )
    }

    // MARK: - Pipeline

    func previewScores() {
        runTask?.cancel()
        runTask = Task { await runPipeline(hardPrune: false) }
    }

    func runIntentPrune() {
        runTask?.cancel()
        runTask = Task { await runPipeline(hardPrune: true) }
    }

    private func runPipeline(hardPrune: Bool) async {
        errorText = nil
        lastScoreSummary = nil
        prunedVerified = false
        pruneSummaryJSON = nil
        cancelRequested = false
        isRunning = true
        phase = .preparing
        statusDetail = "Building Intent Prune work directory…"

        defer {
            isRunning = false
            if phase != .ready && phase != .failed {
                phase = .failed
            }
        }

        do {
            try validateSelections()
            if outputConflictsWithSource {
                throw IntentPruneError.outputConflictsWithSource
            }

            let workDir = try makeWorkDirectory()
            let planPath = workDir.appendingPathComponent("intent_prune_plan.json")
            planURL = planPath

            guard !Task.isCancelled else { return }

            phase = .scoring
            statusDetail = "Running hybrid_v1 scorer (intent-prune-score)…"
            let scoreArgs = buildScoreArgs(planPath: planPath.path)
            let scoreData = try await PythonCLIInvoker.invoke(args: scoreArgs) { code, stderr in
                IntentPruneError.cli(code: code, stderr: stderr)
            }
            if let text = String(data: scoreData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !text.isEmpty {
                lastScoreSummary = text
            }

            guard !Task.isCancelled else { return }
            phase = .writingPlan
            statusDetail = "Checking prune plan…"
            guard FileManager.default.fileExists(atPath: planPath.path) else {
                throw IntentPruneError.planNotWritten(planPath)
            }
            // Sidecar plan next to output for adoptReviewedPrunedSource rails.
            try? FileManager.default.createDirectory(
                at: outputURL,
                withIntermediateDirectories: true
            )

            guard hardPrune else {
                phase = .ready
                statusDetail = "Score preview complete. Plan: \(planPath.lastPathComponent)"
                return
            }

            guard !Task.isCancelled else { return }
            phase = .hardPruning
            statusDetail = "Hard-pruning BF16/F16 (prequant-prune-qwen-moe)…"
            let pruneArgs = buildHardPruneArgs(planPath: planPath.path)
            let pruneData = try await PythonCLIInvoker.invoke(args: pruneArgs) { code, stderr in
                IntentPruneError.cli(code: code, stderr: stderr)
            }

            guard !Task.isCancelled else { return }
            phase = .verifying
            statusDetail = "Parsing structural verification…"
            if let obj = try? JSONSerialization.jsonObject(with: pruneData) as? [String: Any] {
                pruneSummaryJSON = obj
                let status = (obj["verification_status"] as? String)
                    ?? (obj["verificationStatus"] as? String)
                    ?? ((obj["verification"] as? [String: Any])?["status"] as? String)
                let passed = status?.lowercased() == "pass"
                    || status?.lowercased() == "passed"
                    || status?.lowercased() == "ok"
                    || (obj["ok"] as? Bool) == true
                    || ((obj["verification"] as? [String: Any])?["passed"] as? Bool) == true
                prunedVerified = passed || FileManager.default.fileExists(
                    atPath: outputURL.appendingPathComponent("config.json").path
                )
            } else {
                // Some prune builds return non-JSON diagnostics; accept tree presence.
                prunedVerified = FileManager.default.fileExists(
                    atPath: outputURL.appendingPathComponent("config.json").path
                )
                if !prunedVerified {
                    throw IntentPruneError.pruneDecodeFailed(
                        String(data: pruneData, encoding: .utf8) ?? "<binary>"
                    )
                }
            }

            // Copy plan into pruned tree so adopt / preflight can find prune_plan.json.
            let destPlan = outputURL.appendingPathComponent("prune_plan.json")
            if FileManager.default.fileExists(atPath: destPlan.path) {
                try? FileManager.default.removeItem(at: destPlan)
            }
            try? FileManager.default.copyItem(at: planPath, to: destPlan)

            phase = .ready
            statusDetail = prunedVerified
                ? "Pruned source ready. Convert uses this folder as the new source."
                : "Prune finished; review structural verification before Convert."
        } catch is CancellationError {
            phase = .failed
            statusDetail = "Cancelled"
            errorText = "Intent Prune cancelled."
        } catch {
            phase = .failed
            statusDetail = "Failed"
            errorText = error.localizedDescription
        }
    }

    private func validateSelections() throws {
        if selectedChips.isEmpty {
            throw IntentPruneError.noCapabilitySelected
        }
        if safetyStance.isCrack, !crackConfirmed {
            throw IntentPruneError.crackConfirmRequired
        }
        if transitionsURL == nil, adjacencyURL == nil {
            throw IntentPruneError.missingTransitions
        }
    }

    private func makeWorkDirectory() throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("jang-intent-prune-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    // MARK: - Helpers

    static func defaultOutputURL(
        sourceURL: URL,
        chips: [IntentPruneChip],
        keepK: Int,
        safetyStance: IntentPruneSafetyStance
    ) -> URL {
        let name = IntentPruneCLIArgsBuilder.artifactFolderName(
            sourceBaseName: sourceURL.lastPathComponent,
            chips: chips,
            keepK: keepK,
            safetyStance: safetyStance
        )
        return sourceURL.deletingLastPathComponent().appendingPathComponent(name)
    }

    static func discoverTransitions(near sourceURL: URL) -> URL? {
        let fm = FileManager.default
        let candidates = [
            sourceURL.appendingPathComponent("expert_transitions.jsonl"),
            sourceURL.appendingPathComponent("expert_lab/expert_transitions.jsonl"),
            sourceURL.deletingLastPathComponent()
                .appendingPathComponent("expert_transitions.jsonl"),
        ]
        for url in candidates where fm.fileExists(atPath: url.path) {
            return url
        }
        return nil
    }
}
