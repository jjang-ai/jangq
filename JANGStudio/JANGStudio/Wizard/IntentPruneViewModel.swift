// JANGStudio/JANGStudio/Wizard/IntentPruneViewModel.swift
// Sheet ViewModel for Intent Prune quality loop:
//   Evidence → Shape (keep/drop) → Score + Hard Prune → Quality note → Convert
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
            return "Need expert_transitions.jsonl from a real-domain BF16/vMLX trace. Attach a file or open Advanced Expert Lab to generate evidence first."
        case .noCapabilitySelected:
            return "Select at least one Keep capability (green panel)."
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

    // MARK: - Workflow
    var stage: IntentPruneStage = .shape

    // MARK: - User selections
    var selectedChips: Set<IntentPruneChip> = [.coding, .math]
    var dropChips: Set<IntentPruneDropChip> = []
    var safetyStance: IntentPruneSafetyStance = .keep
    var crackConfirmed: Bool = false
    var budget: IntentPruneBudget = .standard
    /// Path to `expert_transitions.jsonl` (or adjacency). Required for score/prune.
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
    var pruneSummaryJSON: [String: Any]?
    /// User acknowledged holdout quality step (full auto holdout suite is CLI-side today).
    var qualityAcknowledged = false

    init(sourceURL: URL, detected: ArchitectureSummary) {
        self.sourceURL = sourceURL
        self.detected = detected
        let k = IntentPruneBudget.standard.keepK(
            expertsPerLayer: max(detected.numExperts, 1),
            trainedTopK: detected.numExpertsPerTok ?? 1
        )
        self.outputURL = Self.defaultOutputURL(
            sourceURL: sourceURL,
            chips: [.coding, .math],
            dropChips: [],
            keepK: k,
            safetyStance: .keep
        )
        self.transitionsURL = Self.discoverTransitions(near: sourceURL)
        if transitionsURL != nil {
            // Evidence already present — stay on Shape for selections.
            stage = .shape
        }
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

    var dropDomainKeys: [String] {
        IntentPruneCLIArgsBuilder.dropDomainKeys(for: dropChips)
    }

    var hasEvidence: Bool {
        transitionsURL != nil || adjacencyURL != nil
    }

    var shapeComplete: Bool {
        !selectedChips.isEmpty && (!safetyStance.isCrack || crackConfirmed)
    }

    var canEnterStage: (IntentPruneStage) -> Bool {
        { stage in
            switch stage {
            case .evidence, .shape:
                return true
            case .prune:
                return self.hasEvidence && self.shapeComplete
            case .quality:
                return self.prunedVerified
            case .convert:
                return self.canAdopt && self.qualityAcknowledged
            }
        }
    }

    var canRun: Bool {
        guard !isRunning else { return false }
        guard shapeComplete else { return false }
        guard hasEvidence else { return false }
        guard !outputConflictsWithSource else { return false }
        return true
    }

    var canPreviewScores: Bool {
        guard !isRunning else { return false }
        guard shapeComplete else { return false }
        return hasEvidence
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
        if hasEvidence {
            let name = transitionsURL?.lastPathComponent ?? adjacencyURL?.lastPathComponent ?? "evidence"
            var parts = ["Evidence: \(name)"]
            if safetyStance.isCrack {
                parts.append("CRACK pack auto-attached on score")
            }
            return parts.joined(separator: " · ")
        }
        return "Evidence: attach expert_transitions.jsonl (real-domain preferred)"
    }

    var keepSummaryLine: String {
        "Keep \(keepK) / \(expertsPerLayer) experts per layer · drop \(expertsPerLayer - keepK)"
    }

    var sourceSummaryLine: String {
        let layers = detected.numHiddenLayers.map { "\($0) layers" } ?? "layers unknown"
        return "\(sourceURL.lastPathComponent) · \(layers) · \(expertsPerLayer) experts/layer · \(detected.dtype.rawValue.uppercased())"
    }

    var planSummaryLine: String {
        let keep = selectedChips.map(\.title).sorted().joined(separator: ", ")
        let drop = dropChips.filter { !$0.switchesToCrackStance }.map(\.title).sorted().joined(separator: ", ")
        var parts = ["Keep: \(keep.isEmpty ? "—" : keep)"]
        if !drop.isEmpty { parts.append("Drop: \(drop)") }
        parts.append("K=\(keepK)")
        parts.append(safetyStance.title)
        return parts.joined(separator: " · ")
    }

    // MARK: - Mutations

    func goToStage(_ next: IntentPruneStage) {
        guard canEnterStage(next) || next == .evidence || next == .shape else { return }
        stage = next
    }

    func toggleChip(_ chip: IntentPruneChip) {
        if selectedChips.contains(chip) {
            selectedChips.remove(chip)
        } else {
            selectedChips.insert(chip)
            // Keep wins over drop for the same semantic area when possible.
            if chip == .multilingual { dropChips.remove(.multilingual) }
            if chip == .tools { dropChips.remove(.tools) }
            if chip == .longContext { dropChips.remove(.longContext) }
        }
        qualityAcknowledged = false
        refreshDefaultOutputURL()
    }

    func toggleDropChip(_ chip: IntentPruneDropChip) {
        if chip.switchesToCrackStance {
            setSafetyStance(.crack)
            return
        }
        if dropChips.contains(chip) {
            dropChips.remove(chip)
        } else {
            dropChips.insert(chip)
            if chip == .multilingual { selectedChips.remove(.multilingual) }
            if chip == .tools { selectedChips.remove(.tools) }
            if chip == .longContext { selectedChips.remove(.longContext) }
            if chip == .chinese { selectedChips.remove(.multilingual) }
        }
        qualityAcknowledged = false
        refreshDefaultOutputURL()
    }

    func applyPreset(_ preset: IntentPrunePreset) {
        selectedChips = preset.keep
        dropChips = preset.drop.filter { !$0.switchesToCrackStance }
        qualityAcknowledged = false
        refreshDefaultOutputURL()
    }

    func clearSelections() {
        selectedChips = []
        dropChips = []
        qualityAcknowledged = false
        refreshDefaultOutputURL()
    }

    func setSafetyStance(_ stance: IntentPruneSafetyStance) {
        safetyStance = stance
        if !stance.isCrack {
            crackConfirmed = false
            dropChips.remove(.safetyHeavy)
        } else {
            dropChips.insert(.safetyHeavy)
        }
        qualityAcknowledged = false
        refreshDefaultOutputURL()
    }

    func setBudget(_ budget: IntentPruneBudget) {
        self.budget = budget
        qualityAcknowledged = false
        refreshDefaultOutputURL()
    }

    func refreshDefaultOutputURL() {
        outputURL = Self.defaultOutputURL(
            sourceURL: sourceURL,
            chips: Array(selectedChips),
            dropChips: Array(dropChips),
            keepK: keepK,
            safetyStance: safetyStance
        )
    }

    func acknowledgeQuality() {
        qualityAcknowledged = true
        if canAdopt {
            stage = .convert
        }
    }

    func cancelRun() {
        cancelRequested = true
        runTask?.cancel()
    }

    // MARK: - CLI

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
            intentsDrop: dropDomainKeys,
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
        stage = .prune
        runTask?.cancel()
        runTask = Task { await runPipeline(hardPrune: false) }
    }

    func runIntentPrune() {
        stage = .prune
        runTask?.cancel()
        runTask = Task { await runPipeline(hardPrune: true) }
    }

    private func runPipeline(hardPrune: Bool) async {
        errorText = nil
        lastScoreSummary = nil
        prunedVerified = false
        pruneSummaryJSON = nil
        qualityAcknowledged = false
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
                prunedVerified = FileManager.default.fileExists(
                    atPath: outputURL.appendingPathComponent("config.json").path
                )
                if !prunedVerified {
                    throw IntentPruneError.pruneDecodeFailed(
                        String(data: pruneData, encoding: .utf8) ?? "<binary>"
                    )
                }
            }

            let destPlan = outputURL.appendingPathComponent("prune_plan.json")
            if FileManager.default.fileExists(atPath: destPlan.path) {
                try? FileManager.default.removeItem(at: destPlan)
            }
            try? FileManager.default.copyItem(at: planPath, to: destPlan)

            phase = .ready
            stage = .quality
            statusDetail = prunedVerified
                ? "Structural prune OK. Review quality checklist before Convert."
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
        dropChips: [IntentPruneDropChip],
        keepK: Int,
        safetyStance: IntentPruneSafetyStance
    ) -> URL {
        let name = IntentPruneCLIArgsBuilder.artifactFolderName(
            sourceBaseName: sourceURL.lastPathComponent,
            chips: chips,
            keepK: keepK,
            safetyStance: safetyStance,
            dropChips: dropChips
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
