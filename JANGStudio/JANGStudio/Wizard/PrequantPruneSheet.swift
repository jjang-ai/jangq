import AppKit
import CryptoKit
import JANGExpertLab
import SwiftUI
import UniformTypeIdentifiers

struct PrequantPruneSheet: View {
    let sourceURL: URL
    let detected: ArchitectureSummary
    let onAdoptPrunedSource: (URL) -> Void
    let showsCloseButton: Bool

    @Environment(\.dismiss) private var dismiss
    @State private var keepExperts: Int
    @State private var outputURL: URL
    @State private var isRunning = false
    /// M221: elapsed time display for the running indicator.
    @State private var prunStartedAt: Date?
    @State private var prunElapsedTime: TimeInterval = 0
    /// M221: human-readable current operation phase label.
    @State private var prunePhaseText = "Pruning BF16/F16 source…"
    @State private var errorText: String?
    @State private var summary: PrequantPruneSummary?
    @State private var reportURL: URL?
    @State private var prunePlanURL: URL?
    @State private var prunePlanSummary: ImportedPrunePlanSummary?
    @State private var reviewEvidenceReady = false
    @State private var reviewEvidenceIssue: String?
    @State private var prunedSuiteEvidenceReady = false
    @State private var prunedSuiteEvidenceIssue: String?
    @State private var prunedSuiteSummaryURL: URL?
    @State private var prunedSuiteGenerationsURL: URL?

    init(
        sourceURL: URL,
        detected: ArchitectureSummary,
        initialPrunePlanURL: URL? = nil,
        showsCloseButton: Bool = true,
        onAdoptPrunedSource: @escaping (URL) -> Void
    ) {
        self.sourceURL = sourceURL
        self.detected = detected
        self.onAdoptPrunedSource = onAdoptPrunedSource
        self.showsCloseButton = showsCloseButton
        let maxKeep = max(detected.numExperts - 1, 1)
        let minKeep = min(max(8, detected.numExperts / 2), maxKeep)
        var defaultKeep = min(max(detected.numExperts - 32, minKeep), maxKeep)
        var initialPlanSummary: ImportedPrunePlanSummary?
        var resolvedInitialPlanURL: URL?
        if let initialPrunePlanURL,
           let summary = try? Self.readPrunePlanSummary(
            url: initialPrunePlanURL,
            sourceURL: sourceURL,
            sourceExperts: detected.numExperts
           ) {
            defaultKeep = summary.keepExperts
            initialPlanSummary = summary
            resolvedInitialPlanURL = initialPrunePlanURL
        }
        self._keepExperts = State(initialValue: defaultKeep)
        self._outputURL = State(initialValue: Self.defaultOutputURL(
            sourceURL: sourceURL,
            keepExperts: defaultKeep,
            isSmartPlan: resolvedInitialPlanURL != nil
        ))
        self._prunePlanURL = State(initialValue: resolvedInitialPlanURL)
        self._prunePlanSummary = State(initialValue: initialPlanSummary)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Form {
            Section("Source") {
                LabeledContent("Model", value: sourceURL.lastPathComponent)
                LabeledContent("Experts", value: "\(detected.numExperts) per layer")
                LabeledContent("Stage", value: "BF16/F16 source before conversion")
                Label("The original source directory is never modified.", systemImage: "lock.shield")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Prune Plan") {
                    Stepper(value: $keepExperts, in: minKeep...maxKeep, step: 8) {
                        LabeledContent("Keep experts", value: "\(keepExperts) / \(detected.numExperts)")
                    }
                    .disabled(prunePlanURL != nil)
                    .onChange(of: keepExperts) { _, newValue in
                        outputURL = Self.defaultOutputURL(
                            sourceURL: sourceURL,
                            keepExperts: newValue,
                            isSmartPlan: prunePlanURL != nil
                        )
                    }
                    LabeledContent("Drop per layer", value: "\(detected.numExperts - keepExperts)")
                    if let prunePlanURL, let prunePlanSummary {
                        LabeledContent("Selection", value: "Expert Lab trace plan")
                        LabeledContent("Plan", value: prunePlanURL.lastPathComponent)
                        LabeledContent("Method", value: prunePlanSummary.method)
                        LabeledContent("Layers", value: "\(prunePlanSummary.layerCount)")
                        LabeledContent("Prompt evidence", value: "\(prunePlanSummary.promptCount) prompts")
                        LabeledContent("Locked keep", value: "\(prunePlanSummary.lockedKeepCount) experts")
                        LabeledContent("User-forced drop", value: "\(prunePlanSummary.userForcedDropCount) experts")
                        LabeledContent("Evidence rows", value: "\(prunePlanSummary.evidenceCount)")
                        LabeledContent("Safety", value: prunePlanSummary.safetyDescription)
                        if let intent = prunePlanSummary.intentMeta, intent.scorer != nil || intent.schema == "jang-intent-prune-plan-v1" {
                            if let scorer = intent.scorer {
                                LabeledContent("Scorer", value: scorer)
                            }
                            if let suiteName = intent.suiteName {
                                LabeledContent("Suite", value: suiteName)
                            }
                            if let crack = intent.crackPackName {
                                LabeledContent("CRACK pack", value: crack)
                            }
                        }
                        if let comparison = prunePlanSummary.comparisonSummary {
                            LabeledContent("A/B comparison", value: "\(comparison.promptCount) prompts")
                            LabeledContent("Masked pass rate", value: comparison.maskedPassRateDescription)
                            if let baselineQualified = comparison.baselineQualifiedPromptCount {
                                LabeledContent("Baseline-qualified", value: "\(baselineQualified) prompts")
                            }
                            if let qualifiedPass = comparison.baselineQualifiedMaskedPassRate {
                                LabeledContent("Qualified masked pass", value: ImportedComparisonSummary.passRateDescription(qualifiedPass))
                            }
                            if let classes = comparison.classificationCounts {
                                LabeledContent("Prompt classes", value: ImportedComparisonSummary.classificationDescription(classes))
                            }
                            LabeledContent("Mean text delta", value: comparison.meanTextDeltaDescription)
                            if let severity = comparison.regressionSeverity {
                                LabeledContent("Regression severity", value: severity)
                            }
                            if !comparison.missingBaselineQualifiedSemanticCoverage.isEmpty {
                                LabeledContent("Missing qualified coverage", value: comparison.missingBaselineQualifiedSemanticCoverage.joined(separator: ", "))
                            }
                            if !comparison.degradedPromptIDs.isEmpty {
                                LabeledContent("Degraded prompts", value: comparison.degradedPromptIDs.prefix(6).joined(separator: ", "))
                            }
                            if !comparison.highRiskDomains.isEmpty {
                                LabeledContent("High-risk domains", value: comparison.highRiskDomains.joined(separator: ", "))
                            }
                            LabeledContent("A/B-safe candidates", value: "\(comparison.safeDropCandidateCount)")
                        }
                        if let runID = prunePlanSummary.runID {
                            LabeledContent("Review run", value: runID)
                        }
                        if let atlasID = prunePlanSummary.atlasID {
                            LabeledContent("Atlas", value: atlasID)
                        }
                        if let evalArtifactPath = prunePlanSummary.evalArtifactPath {
                            LabeledContent("Eval evidence", value: URL(fileURLWithPath: evalArtifactPath).lastPathComponent)
                        }
                        if !prunePlanSummary.evidencePreview.isEmpty {
                            DisclosureGroup("Evidence Preview") {
                                VStack(alignment: .leading, spacing: 6) {
                                    ForEach(prunePlanSummary.evidencePreview, id: \.self) { line in
                                        Text(line)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                            .textSelection(.enabled)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        }
                        Text("This reviewed path uses prompt-suite trace evidence, A/B comparison where available, atlas labels, locked keeps, and user keep/drop decisions. Hard pruning writes a new BF16/F16 source and then verifies its structure before conversion.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Label("Fallback selection: router-row strength only. This is inspectable only; it does not test what experts do on prompts or unlock final quantization.",
                              systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                        Text("For smart pruning: open the original BF16/F16 source in Expert Lab, run BF16/vMLX prompt-suite traces, export a Smart Prune Plan, then choose it here.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Button {
                            choosePrunePlan()
                        } label: {
                            Label("Choose Expert Lab Plan...", systemImage: "doc.badge.gearshape")
                        }
                        Button("Clear Plan") {
                            prunePlanURL = nil
                            prunePlanSummary = nil
                            outputURL = Self.defaultOutputURL(sourceURL: sourceURL, keepExperts: keepExperts)
                        }
                        .disabled(prunePlanURL == nil)
                    }
                }

                Section("Output") {
                    HStack {
                        Text(outputURL.path)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        Spacer()
                        Button("Choose...") { chooseOutput() }
                    }
                    Text("The original source directory is never modified. The pruned BF16/F16 source can be converted with JANG or JANGTQ after this step.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if outputConflictsWithSource {
                        Label("Choose an output separate from the source model tree.", systemImage: "xmark.octagon.fill")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                if isRunning {
                    Section {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(prunePhaseText)
                                        .foregroundStyle(.secondary)
                                        .font(.caption.weight(.medium))
                                    // M221: elapsed time while pruning.
                                    if prunStartedAt != nil {
                                        HStack(spacing: 6) {
                                            Image(systemName: "clock")
                                                .foregroundStyle(.tertiary)
                                                .font(.caption2)
                                            Text(prunElapsedTimeString)
                                                .font(.caption)
                                                .foregroundStyle(.tertiary)
                                                .monospacedDigit()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let summary {
                    Section("Verified Pruned Source") {
                        LabeledContent("Output", value: outputURL.path)
                        LabeledContent("Kept", value: "\(summary.numExperts) experts")
                        LabeledContent("Source experts", value: "\(summary.sourceNumExperts)")
                        LabeledContent("Method", value: summary.method)
                        LabeledContent("Verification", value: summary.verificationStatus)
                        if let verification = summary.verification {
                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(verification.requiredCheckRows) { row in
                                    Label(row.name, systemImage: row.passed ? "checkmark.circle" : "xmark.octagon")
                                        .foregroundStyle(row.passed ? Color.secondary : Color.red)
                                }
                                if let reason = verification.strictFailureReason {
                                    Text(reason)
                                        .font(.caption)
                                        .foregroundStyle(.red)
                                }
                            }
                        }
                        if let reportURL {
                            LabeledContent("Report", value: reportURL.lastPathComponent)
                            Button {
                                NSWorkspace.shared.activateFileViewerSelecting([reportURL])
                            } label: {
                                Label("Reveal Report", systemImage: "doc.text.magnifyingglass")
                            }
                        }
                        if prunePlanURL != nil {
                            LabeledContent("Same-suite evidence", value: reviewEvidenceReady ? "ready" : "missing")
                            LabeledContent("Pruned BF16/vMLX generation", value: prunedSuiteEvidenceReady ? "ready" : "missing")
                            if let prunedSuiteArtifactDescription {
                                Text("Pruned artifacts: \(prunedSuiteArtifactDescription)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                            if let reviewEvidenceIssue, !reviewEvidenceIssue.isEmpty {
                                Text(reviewEvidenceIssue)
                                    .font(.caption)
                                    .foregroundStyle(reviewEvidenceReady ? Color.secondary : Color.red)
                                    .textSelection(.enabled)
                            }
                            if let prunedSuiteEvidenceIssue,
                               prunedSuiteEvidenceIssue != reviewEvidenceIssue,
                               !prunedSuiteEvidenceIssue.isEmpty {
                                Text(prunedSuiteEvidenceIssue)
                                    .font(.caption)
                                    .foregroundStyle(prunedSuiteEvidenceReady ? Color.secondary : Color.red)
                                    .textSelection(.enabled)
                            }
                        }
                        Text(prunedSourceAdoptionMessage(summary: summary))
                            .font(.caption)
                            .foregroundStyle(canAdoptPrunedSource ? Color.secondary : Color.red)
                    }
                }

                if let errorText {
                    Section {
                        Label(errorText, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                    }
                }
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
            .background(ExpertLabVisual.canvas)
            Divider()
            footer
        }
        .background(ExpertLabVisual.canvas)
        .frame(minWidth: 680, minHeight: 560)
        // M221: 1-second timer to update elapsed-time display while pruning.
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
            guard isRunning, let s = prunStartedAt else { return }
            prunElapsedTime = Date().timeIntervalSince(s)
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            ExpertLabConsoleCard(accent: ExpertLabVisual.warm) {
                VStack(alignment: .leading, spacing: 10) {
                    ExpertLabKicker(text: "BF16/F16 hard prune", color: ExpertLabVisual.warm)
                    Text("Prune Before Conversion")
                        .font(.title3.weight(.semibold))
                    Text("Review the keep/drop plan, hard-prune a new BF16/F16 source, verify every structural check, then convert the smaller model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ExpertLabWorkflowStrip(
                        steps: ["Reviewed Plan", "BF16/F16 Prune", "Verify", "Convert"],
                        activeIndex: canAdoptPrunedSource ? 2 : 1
                    )
                }
            }
            Spacer()
            if showsCloseButton {
                Button("Close") { dismiss() }
                    .disabled(isRunning)
            }
        }
        .padding(14)
    }

    private var footer: some View {
        HStack {
            if summary != nil {
                Button {
                    onAdoptPrunedSource(outputURL)
                    if showsCloseButton {
                        dismiss()
                    }
                } label: {
                    Label("Use Verified Pruned Source", systemImage: "checkmark.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canAdoptPrunedSource)
            }
            Spacer()
            Button {
                Task { await runPrune() }
            } label: {
                Label("Run BF16/F16 Prune", systemImage: "scissors")
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning || summary != nil || outputConflictsWithSource)
        }
        .padding(12)
    }

    private var minKeep: Int {
        min(max(8, detected.numExperts / 2), maxKeep)
    }

    private var maxKeep: Int {
        max(detected.numExperts - 1, 1)
    }

    private var canAdoptPrunedSource: Bool {
        guard summary?.isVerified == true else { return false }
        return prunePlanURL != nil && reviewEvidenceReady && prunedSuiteEvidenceReady
    }

    private var prunedSuiteArtifactDescription: String? {
        guard prunedSuiteSummaryURL != nil || prunedSuiteGenerationsURL != nil else { return nil }
        return [
            prunedSuiteSummaryURL?.lastPathComponent,
            prunedSuiteGenerationsURL?.lastPathComponent
        ]
        .compactMap { $0 }
        .joined(separator: ", ")
    }

    private var outputConflictsWithSource: Bool {
        Self.prunedOutputConflictsWithSource(sourceURL: sourceURL, outputURL: outputURL)
    }

    // M221: formatted elapsed time while pruning.
    private var prunElapsedTimeString: String {
        let t = Int(prunElapsedTime)
        if t >= 3600 {
            return String(format: "%d:%02d:%02d", t / 3600, (t % 3600) / 60, t % 60)
        }
        return String(format: "%d:%02d", t / 60, t % 60)
    }

    nonisolated static func prunedOutputConflictsWithSource(sourceURL: URL, outputURL: URL) -> Bool {
        let sourcePath = sourceURL.standardizedFileURL.resolvingSymlinksInPath().path
        let outputPath = outputURL.standardizedFileURL.resolvingSymlinksInPath().path
        return path(outputPath, isInsideOrEqualTo: sourcePath)
            || path(sourcePath, isInsideOrEqualTo: outputPath)
    }

    private nonisolated static func path(_ child: String, isInsideOrEqualTo parent: String) -> Bool {
        if child == parent { return true }
        let parentPrefix = parent == "/" ? "/" : parent + "/"
        return child.hasPrefix(parentPrefix)
    }

    private func prunedSourceAdoptionMessage(summary: PrequantPruneSummary?) -> String {
        guard summary?.isVerified == true else {
            return "Verification did not pass, so Studio will not adopt this source for conversion."
        }
        if prunePlanURL == nil {
            return "Router-only fallback output is inspectable only. Choose an Expert Lab plan and rerun BF16/vMLX same-suite verification before final conversion."
        }
        if prunePlanURL != nil && !reviewEvidenceReady {
            let issue = reviewEvidenceIssue.map { " \($0)" } ?? ""
            return "Structural verification passed, but same-suite Expert Lab evidence is not ready, so Studio will not adopt this reviewed source for final quantization.\(issue)"
        }
        if prunePlanURL != nil && !prunedSuiteEvidenceReady {
            let issue = prunedSuiteEvidenceIssue.map { " \($0)" } ?? ""
            return "Structural verification passed, but pruned BF16/F16 same-suite vMLX generation is not ready, so Studio will not adopt this reviewed source for final quantization.\(issue)"
        }
        return "This pruned BF16/F16 source is now eligible for final JANG/JANGTQ conversion. The original source remains unchanged."
    }

    private func chooseOutput() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = outputURL.lastPathComponent
        panel.directoryURL = outputURL.deletingLastPathComponent()
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            outputURL = url
        }
    }

    private func choosePrunePlan() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let summary = try Self.readPrunePlanSummary(
                    url: url,
                    sourceURL: sourceURL,
                    sourceExperts: detected.numExperts
                )
                prunePlanURL = url
                prunePlanSummary = summary
                keepExperts = summary.keepExperts
                outputURL = Self.defaultOutputURL(
                    sourceURL: sourceURL,
                    keepExperts: summary.keepExperts,
                    isSmartPlan: true
                )
                errorText = nil
            } catch {
                errorText = "Could not read prune plan: \(error.localizedDescription)"
            }
        }
    }

    private static let intentPrunePlanSchema = "jang-intent-prune-plan-v1"

    /// Normalize `jang-intent-prune-plan-v1` (flat keep lists, scorer, scalar top-k)
    /// into the Expert Lab import shape so existing decoders/gates stay shared.
    private static func normalizeImportedPrunePlanDictionary(_ plan: [String: Any]) -> [String: Any] {
        var out = plan
        let schema = (out["schema"] as? String) ?? ""
        let isIntent = schema == intentPrunePlanSchema
            || ((out["scorer"] as? String)?.isEmpty == false && out["method"] == nil)

        if isIntent || out["method"] == nil {
            if let scorer = out["scorer"] as? String, !scorer.isEmpty, out["method"] == nil {
                out["method"] = scorer
            }
        }
        if out["method"] == nil {
            out["method"] = isIntent ? "hybrid_v1" : "external_keep_map"
        }

        if out["keepExpertsPerLayer"] == nil, let keep = out["keep_experts_per_layer"] {
            out["keepExpertsPerLayer"] = keep
        }
        if out["promptCount"] == nil {
            if let pc = out["prompt_count"] {
                out["promptCount"] = pc
            } else if let suite = out["suite"] as? [String: Any],
                      let pc = suite["prompt_count"] ?? suite["promptCount"] {
                out["promptCount"] = pc
            }
        }

        if var safety = out["safety"] as? [String: Any] {
            let hasByLayer = safety["trained_top_k_by_layer"] != nil || safety["trainedTopKByLayer"] != nil
            if !hasByLayer, let topK = safety["trained_top_k"] ?? safety["trainedTopK"] {
                var byLayer: [String: Any] = [:]
                if let layers = out["layers"] as? [String: Any] {
                    for key in layers.keys {
                        byLayer[key] = topK
                    }
                }
                if byLayer.isEmpty {
                    byLayer["0"] = topK
                }
                safety["trained_top_k_by_layer"] = byLayer
            }
            out["safety"] = safety
        }

        if let layers = out["layers"] as? [String: Any] {
            var normalized: [String: Any] = [:]
            let numExperts = (out["num_experts_source"] as? Int)
                ?? (out["numExpertsSource"] as? Int)
                ?? (out["sourceNumExperts"] as? Int)
            for (key, value) in layers {
                if let arr = value as? [Any] {
                    let keepInts = arr.compactMap { ($0 as? Int) ?? ($0 as? NSNumber)?.intValue }
                    var layerObj: [String: Any] = [
                        "keep": keepInts,
                        "drop": [] as [Int],
                        "evidence": [] as [[String: Any]],
                    ]
                    if let numExperts, numExperts > 0 {
                        let keepSet = Set(keepInts)
                        layerObj["drop"] = (0..<numExperts).filter { !keepSet.contains($0) }
                        layerObj["num_source_experts"] = numExperts
                    }
                    normalized[key] = layerObj
                } else {
                    normalized[key] = value
                }
            }
            out["layers"] = normalized
        }

        if isIntent {
            out["_intent_prune"] = true
        }
        return out
    }

    private static func isIntentPrunePlanDictionary(_ plan: [String: Any]) -> Bool {
        if let schema = plan["schema"] as? String, schema == intentPrunePlanSchema {
            return true
        }
        if plan["_intent_prune"] as? Bool == true {
            return true
        }
        if let scorer = plan["scorer"] as? String, !scorer.isEmpty, plan["method"] == nil {
            return true
        }
        return false
    }

    private static func readPrunePlanSummary(
        url: URL,
        sourceURL: URL,
        sourceExperts: Int
    ) throws -> ImportedPrunePlanSummary {
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dict = object as? [String: Any] else {
            throw ImportedPrunePlanError.emptyLayers
        }
        let isIntent = isIntentPrunePlanDictionary(dict)
        let normalized = normalizeImportedPrunePlanDictionary(dict)
        let normalizedData = try JSONSerialization.data(withJSONObject: normalized)
        let plan = try JSONDecoder().decode(ImportedPrunePlan.self, from: normalizedData)
        return try ImportedPrunePlanSummary(
            plan: plan,
            sourceURL: sourceURL,
            sourceExperts: sourceExperts,
            skipSemanticEvidence: isIntent,
            intentMeta: IntentPrunePlanMeta(from: dict)
        )
    }

    private func runPrune() async {
        guard !outputConflictsWithSource else {
            await MainActor.run {
                errorText = "Choose an output separate from the source model tree. Expert pruning never writes into or above the original BF16/F16 source."
            }
            return
        }
        await MainActor.run {
            isRunning = true
            prunStartedAt = Date()
            prunElapsedTime = 0
            errorText = nil
            summary = nil
            reportURL = nil
            reviewEvidenceReady = false
            reviewEvidenceIssue = nil
            prunedSuiteEvidenceReady = false
            prunedSuiteEvidenceIssue = nil
            prunedSuiteSummaryURL = nil
            prunedSuiteGenerationsURL = nil
        }
        do {
            prunePhaseText = "Pruning BF16/F16 source…"
            let data = try await PythonCLIInvoker.invoke(args: [
                "-m", "jang_tools",
                "--quiet-text",
                "prequant-prune-qwen-moe",
                sourceURL.path,
                outputURL.path,
            ] + pruneArgs()) { code, stderr in
                PrequantPruneError.cli(code: code, stderr: stderr)
            }
            prunePhaseText = "Verifying pruned source…"
            let result = try JSONDecoder().decode(PrequantPruneSummary.self, from: data)
            try copyReviewedPlanSidecarIfNeeded()
            prunePhaseText = "Persisting review evidence…"
            let copiedEvidence = try persistReviewEvidenceSidecarsIfNeeded()
            prunePhaseText = "Verifying pruned suite…"
            let prunedSuite = try await persistPrunedSourceSuiteVerificationIfNeeded(
                suiteURL: copiedEvidence.suiteURL,
                evalURL: copiedEvidence.evalURL,
                evalTraceURL: copiedEvidence.evalTraceURL
            )
            prunePhaseText = "Saving report…"
            _ = try updateReviewSummaryWithPrunedSuiteVerification(
                copiedEvidence: copiedEvidence,
                prunedSuite: prunedSuite
            )
            let report = try writeReport(summary: result)
            await MainActor.run {
                summary = result
                reportURL = report
                reviewEvidenceReady = copiedEvidence.ready
                reviewEvidenceIssue = copiedEvidence.issue
                prunedSuiteEvidenceReady = prunedSuite.ready
                prunedSuiteEvidenceIssue = prunedSuite.issue
                prunedSuiteSummaryURL = prunedSuite.summaryURL
                prunedSuiteGenerationsURL = prunedSuite.generationsURL
                isRunning = false
            }
        } catch {
            await MainActor.run {
                errorText = error.localizedDescription
                isRunning = false
            }
        }
    }

    private static func defaultOutputURL(sourceURL: URL, keepExperts: Int) -> URL {
        defaultOutputURL(sourceURL: sourceURL, keepExperts: keepExperts, isSmartPlan: false)
    }

    private static func defaultOutputURL(sourceURL: URL, keepExperts: Int, isSmartPlan: Bool) -> URL {
        let suffix = isSmartPlan ? "smartpruned" : "prepruned"
        return sourceURL
            .deletingLastPathComponent()
            .appendingPathComponent("\(sourceURL.lastPathComponent)-\(suffix)-\(keepExperts)e")
    }

    private func pruneArgs() -> [String] {
        var args: [String] = [
            "--keep-experts", "\(keepExperts)",
            "--json",
        ]
        if let prunePlanURL {
            args.append(contentsOf: ["--keep-map", prunePlanURL.path, "--require-reviewed-comparison"])
        }
        return args
    }

    private struct ReviewEvidenceSidecars {
        let ready: Bool
        let issue: String?
        let suiteURL: URL?
        let evalURL: URL?
        let evalTraceURL: URL?
        let maskURL: URL?
    }

    private struct PrunedSourceSuiteVerification {
        let ready: Bool
        let issue: String?
        let summaryURL: URL?
        let generationsURL: URL?
    }

    private struct PrunedMaskedBehaviorComparison {
        let issue: String?
        let comparedCount: Int
        let meanTextDelta: Double?
        let maxTextDelta: Double?
        let baselineQualifiedPromptCount: Int
        let prunedBaselineQualifiedPassRate: Double?
        let classificationCounts: [String: Int]
        let baselineInvalidPromptIDs: [String]
        let inconclusivePromptIDs: [String]
        let preservedPromptIDs: [String]
        let degradedPromptIDs: [String]
        let baselineQualifiedSemanticCoverage: [String]
        let missingBaselineQualifiedSemanticCoverage: [String]
    }

    private func copyReviewedPlanSidecarIfNeeded() throws {
        guard let prunePlanURL else { return }
        let sidecarURL = outputURL.appendingPathComponent("prune_plan.json")
        if prunePlanURL.standardizedFileURL == sidecarURL.standardizedFileURL {
            return
        }
        try FileManager.default.createDirectory(
            at: outputURL,
            withIntermediateDirectories: true,
            attributes: nil
        )
        if Self.materializedReviewedPrunePlanExists(at: sidecarURL) {
            return
        }
        if FileManager.default.fileExists(atPath: sidecarURL.path) {
            try FileManager.default.removeItem(at: sidecarURL)
        }
        try FileManager.default.copyItem(at: prunePlanURL, to: sidecarURL)
    }

    private static func materializedReviewedPrunePlanExists(at url: URL) -> Bool {
        guard let plan = readJSONObject(url) else { return false }
        guard materializedSidecarName(
            plan["suite_jsonl"] ?? plan["suiteJSONL"],
            equals: "expert_lab_suite.jsonl"
        ) else {
            return false
        }
        guard let evalIndex = plan["eval_index"] as? [String: Any] else { return false }
        guard stringValue(evalIndex["suite_sha256"] ?? evalIndex["suiteSHA256"])?.isEmpty == false else {
            return false
        }
        let evalIndexMaskName = materializedMaskSidecarName(
            evalIndex["mask_json"] ?? evalIndex["maskJSON"] ?? evalIndex["mask"]
        )
        guard materializedSidecarName(
            evalIndex["comparison_summary"] ?? evalIndex["comparisonSummary"],
            equals: "expert_lab_comparison_summary.json"
        ),
              materializedSidecarName(
                evalIndex["eval_jsonl"] ?? evalIndex["evalJSONL"],
                equals: "expert_lab_eval.jsonl"
              ),
              materializedSidecarName(
                evalIndex["eval_trace_jsonl"] ?? evalIndex["evalTraceJSONL"],
                equals: "expert_lab_eval_trace.jsonl"
              ),
              evalIndexMaskName != nil else {
            return false
        }
        guard let sidecars = plan["reviewed_evidence_sidecars"] as? [String: Any] else { return false }
        let sidecarMaskName = materializedMaskSidecarName(
            sidecars["mask_json"] ?? sidecars["maskJSON"] ?? sidecars["mask"]
        )
        return materializedSidecarName(sidecars["suite_jsonl"] ?? sidecars["suiteJSONL"], equals: "expert_lab_suite.jsonl")
            && materializedSidecarName(sidecars["comparison_summary"] ?? sidecars["comparisonSummary"], equals: "expert_lab_comparison_summary.json")
            && materializedSidecarName(sidecars["eval_jsonl"] ?? sidecars["evalJSONL"], equals: "expert_lab_eval.jsonl")
            && materializedSidecarName(sidecars["eval_trace_jsonl"] ?? sidecars["evalTraceJSONL"], equals: "expert_lab_eval_trace.jsonl")
            && materializedSidecarName(sidecars["eval_index"] ?? sidecars["evalIndex"], equals: "expert_lab_eval_index.json")
            && sidecarMaskName == evalIndexMaskName
    }

    private static func materializedSidecarName(_ value: Any?, equals expected: String) -> Bool {
        guard let name = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return false
        }
        return name == expected
    }

    private static func materializedMaskSidecarName(_ value: Any?) -> String? {
        guard let name = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              name == "mask.json" || name == "expert_lab_mask.json" else {
            return nil
        }
        return name
    }

    private static func materializeReviewedPrunePlanSidecar(
        planURL: URL,
        suiteURL: URL,
        comparisonURL: URL,
        evalURL: URL,
        evalTraceURL: URL,
        evalIndexURL: URL,
        maskURL: URL
    ) throws {
        guard var plan = readJSONObject(planURL) else {
            throw PrequantPruneError.materialization("copied prune_plan.json is unreadable")
        }
        guard var evalIndex = readJSONObject(evalIndexURL) ?? plan["eval_index"] as? [String: Any] else {
            throw PrequantPruneError.materialization("reviewed prune plan is missing eval_index evidence")
        }

        let suiteName = suiteURL.lastPathComponent
        let comparisonName = comparisonURL.lastPathComponent
        let evalName = evalURL.lastPathComponent
        let evalTraceName = evalTraceURL.lastPathComponent
        let evalIndexName = evalIndexURL.lastPathComponent
        let maskName = maskURL.lastPathComponent
        guard let suiteSHA256 = fileSHA256(suiteURL) else {
            throw PrequantPruneError.materialization("reviewed prompt suite fingerprint could not be computed")
        }

        evalIndex["suite_jsonl"] = suiteName
        evalIndex["suite_sha256"] = suiteSHA256
        evalIndex["comparison_summary"] = comparisonName
        evalIndex["eval_jsonl"] = evalName
        evalIndex["eval_trace_jsonl"] = evalTraceName
        evalIndex["mask"] = maskName
        evalIndex["mask_json"] = maskName
        let evalIndexData = try JSONSerialization.data(withJSONObject: evalIndex, options: [.prettyPrinted, .sortedKeys])
        try evalIndexData.write(to: evalIndexURL)

        plan["suite_jsonl"] = suiteName
        plan["suite_sha256"] = suiteSHA256
        plan["eval_index"] = evalIndex
        plan["reviewed_evidence_sidecars"] = [
            "suite_jsonl": suiteName,
            "comparison_summary": comparisonName,
            "eval_jsonl": evalName,
            "eval_trace_jsonl": evalTraceName,
            "eval_index": evalIndexName,
            "mask": maskName,
            "mask_json": maskName,
        ]
        let planData = try JSONSerialization.data(withJSONObject: plan, options: [.prettyPrinted, .sortedKeys])
        try planData.write(to: planURL)
    }

    private func persistReviewEvidenceSidecarsIfNeeded() throws -> ReviewEvidenceSidecars {
        guard let prunePlanURL else {
            return ReviewEvidenceSidecars(ready: true, issue: nil, suiteURL: nil, evalURL: nil, evalTraceURL: nil, maskURL: nil)
        }
        let fm = FileManager.default
        try fm.createDirectory(at: outputURL, withIntermediateDirectories: true, attributes: nil)
        if let materialized = Self.materializedReviewEvidenceSidecars(
            outputURL: outputURL,
            sourceURL: sourceURL,
            expectedLayerCount: prunePlanSummary?.layerCount ?? Self.configLayerCount(modelPath: sourceURL.path)
        ) {
            return materialized
        }

        let reviewRunDirectory = prunePlanURL.deletingLastPathComponent()
        let suiteSource = reviewRunDirectory.appendingPathComponent("suite.jsonl")
        let suiteSidecar = outputURL.appendingPathComponent("expert_lab_suite.jsonl")
        let copiedSuite = try copySidecarIfPresent(from: suiteSource, to: suiteSidecar)

        let planEvalPath = prunePlanSummary?.evalArtifactPath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let evalResolution: (directory: URL?, issue: String?) = {
            if let planEvalPath, !planEvalPath.isEmpty {
                guard let directory = Self.planEvalDirectory(from: planEvalPath) else {
                    return (
                        nil,
                        "recorded eval_artifact is missing required same-suite files: \(planEvalPath)"
                    )
                }
                return (directory, nil)
            }
            return (
                Self.latestComparisonDirectory(in: reviewRunDirectory.appendingPathComponent("evals", isDirectory: true)),
                nil
            )
        }()
        let reviewEvalDirectory = evalResolution.directory
        let comparisonSidecar = outputURL.appendingPathComponent("expert_lab_comparison_summary.json")
        let evalSidecar = outputURL.appendingPathComponent("expert_lab_eval.jsonl")
        let evalTraceSidecar = outputURL.appendingPathComponent("expert_lab_eval_trace.jsonl")
        let evalIndexSidecar = outputURL.appendingPathComponent("expert_lab_eval_index.json")
        let maskSidecar = outputURL.appendingPathComponent("mask.json")
        let copiedComparison = try reviewEvalDirectory.map {
            try copySidecarIfPresent(from: $0.appendingPathComponent("comparison_summary.json"), to: comparisonSidecar)
        } ?? false
        let copiedEval = try reviewEvalDirectory.map {
            try copySidecarIfPresent(from: $0.appendingPathComponent("eval.jsonl"), to: evalSidecar)
        } ?? false
        let copiedEvalTrace = try reviewEvalDirectory.map {
            try copySidecarIfPresent(from: $0.appendingPathComponent("eval_trace.jsonl"), to: evalTraceSidecar)
        } ?? false
        var copiedEvalIndex = try reviewEvalDirectory.map {
            try copySidecarIfPresent(from: $0.appendingPathComponent("eval_index.json"), to: evalIndexSidecar)
        } ?? false
        let copiedMask = try reviewEvalDirectory.map {
            try copySidecarIfPresent(from: $0.appendingPathComponent("mask.json"), to: maskSidecar)
        } ?? false
        if !copiedEvalIndex, copiedSuite, copiedComparison, copiedEval, copiedEvalTrace, copiedMask {
            copiedEvalIndex = try Self.writeRecoveredEvalIndexIfPossible(
                reviewRunDirectory: reviewRunDirectory,
                evalDirectory: reviewEvalDirectory,
                suiteURL: suiteSidecar,
                evalURL: evalSidecar,
                evalTraceURL: evalTraceSidecar,
                comparisonURL: comparisonSidecar,
                destination: evalIndexSidecar
            )
        }
        let copiedAllEvidence = copiedSuite && copiedComparison && copiedEval && copiedEvalTrace && copiedEvalIndex && copiedMask
        if copiedAllEvidence {
            try Self.materializeReviewedPrunePlanSidecar(
                planURL: outputURL.appendingPathComponent("prune_plan.json"),
                suiteURL: suiteSidecar,
                comparisonURL: comparisonSidecar,
                evalURL: evalSidecar,
                evalTraceURL: evalTraceSidecar,
                evalIndexURL: evalIndexSidecar,
                maskURL: maskSidecar
            )
        }
        let expectedLayerCount = prunePlanSummary?.layerCount ?? Self.configLayerCount(modelPath: sourceURL.path)
        let comparedPromptCount = Self.readJSONObject(comparisonSidecar).flatMap {
            Self.intValue($0["promptCount"] ?? $0["prompt_count"])
        }
        let evidenceIssue = copiedAllEvidence
            ? Self.sameSuiteEvidenceIssue(
                suiteURL: suiteSidecar,
                evalURL: evalSidecar,
                evalTraceURL: evalTraceSidecar,
                evalIndexURL: evalIndexSidecar,
                maskURL: maskSidecar,
                sourceModelPath: sourceURL.path,
                expectedLayerCount: expectedLayerCount,
                comparedPromptCount: comparedPromptCount
            )
            : evalResolution.issue ?? "same-suite Expert Lab sidecars were not all copied"
        let evidenceReady = copiedAllEvidence && evidenceIssue == nil
        let initialSuiteIssue: Any = evidenceReady
            ? "pruned BF16/F16 same-suite vMLX verification has not run yet"
            : (evidenceIssue.map { $0 as Any } ?? NSNull())

        let reviewSummaryURL = outputURL.appendingPathComponent("expert_lab_review_summary.json")
        let payload: [String: Any] = [
            "schema": "jang-expert-lab-pruned-source-review-v1",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "source_model": sourceURL.path,
            "source_model_path": sourceURL.path,
            "pruned_source": outputURL.path,
            "reviewed_prune_plan": outputURL.appendingPathComponent("prune_plan.json").path,
            "original_reviewed_prune_plan": prunePlanURL.path,
            "review_run_directory": fm.fileExists(atPath: reviewRunDirectory.appendingPathComponent("run.json").path)
                ? reviewRunDirectory.path
                : NSNull(),
            "review_eval_directory": reviewEvalDirectory?.path ?? NSNull(),
            "run_id": prunePlanSummary?.runID ?? NSNull(),
            "atlas_id": prunePlanSummary?.atlasID ?? NSNull(),
            "prompt_count": prunePlanSummary?.promptCount ?? 0,
            "layer_count": prunePlanSummary?.layerCount ?? 0,
            "keep_experts_per_layer": prunePlanSummary?.keepExperts ?? keepExperts,
            "suite_jsonl": copiedSuite ? suiteSidecar.path : NSNull(),
            "comparison_summary": copiedComparison ? comparisonSidecar.path : NSNull(),
            "eval_jsonl": copiedEval ? evalSidecar.path : NSNull(),
            "eval_trace_jsonl": copiedEvalTrace ? evalTraceSidecar.path : NSNull(),
            "eval_index": copiedEvalIndex ? evalIndexSidecar.path : NSNull(),
            "mask_json": copiedMask ? maskSidecar.path : NSNull(),
            "mask": copiedMask ? maskSidecar.path : NSNull(),
            "same_suite_verification_ready": false,
            "same_suite_verification_issue": initialSuiteIssue,
            "review_sidecars_ready": evidenceReady,
            "review_sidecars_issue": evidenceIssue ?? NSNull(),
            "pruned_suite_verification_ready": false,
            "pruned_suite_verification_issue": "pruned BF16/F16 same-suite vMLX verification has not run yet"
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: reviewSummaryURL)
        return ReviewEvidenceSidecars(
            ready: evidenceReady,
            issue: evidenceIssue,
            suiteURL: copiedSuite ? suiteSidecar : nil,
            evalURL: copiedEval ? evalSidecar : nil,
            evalTraceURL: copiedEvalTrace ? evalTraceSidecar : nil,
            maskURL: copiedMask ? maskSidecar : nil
        )
    }

    private static func materializedReviewEvidenceSidecars(
        outputURL: URL,
        sourceURL: URL,
        expectedLayerCount: Int?
    ) -> ReviewEvidenceSidecars? {
        let planURL = outputURL.appendingPathComponent("prune_plan.json")
        let summaryURL = outputURL.appendingPathComponent("expert_lab_review_summary.json")
        guard materializedReviewedPrunePlanExists(at: planURL),
              let summary = readJSONObject(summaryURL) else {
            return nil
        }

        let suiteURL = materializedSidecarURL(
            summary["suite_jsonl"],
            outputURL: outputURL,
            fallbackName: "expert_lab_suite.jsonl"
        )
        let comparisonURL = materializedSidecarURL(
            summary["comparison_summary"],
            outputURL: outputURL,
            fallbackName: "expert_lab_comparison_summary.json"
        )
        let evalURL = materializedSidecarURL(
            summary["eval_jsonl"],
            outputURL: outputURL,
            fallbackName: "expert_lab_eval.jsonl"
        )
        let evalTraceURL = materializedSidecarURL(
            summary["eval_trace_jsonl"],
            outputURL: outputURL,
            fallbackName: "expert_lab_eval_trace.jsonl"
        )
        let evalIndexURL = materializedSidecarURL(
            summary["eval_index"],
            outputURL: outputURL,
            fallbackName: "expert_lab_eval_index.json"
        )
        let maskURL = materializedSidecarURL(
            summary["mask_json"] ?? summary["mask"],
            outputURL: outputURL,
            fallbackName: "mask.json"
        )
        let fm = FileManager.default
        let required: [(String, URL)] = [
            ("suite_jsonl", suiteURL),
            ("comparison_summary", comparisonURL),
            ("eval_jsonl", evalURL),
            ("eval_trace_jsonl", evalTraceURL),
            ("eval_index", evalIndexURL),
            ("mask_json", maskURL),
        ]
        let missing = required.compactMap { key, url -> String? in
            fm.isReadableFile(atPath: url.path) ? nil : key
        }
        if !missing.isEmpty {
            return ReviewEvidenceSidecars(
                ready: false,
                issue: "materialized Expert Lab sidecars are missing: \(missing.joined(separator: ", "))",
                suiteURL: fm.isReadableFile(atPath: suiteURL.path) ? suiteURL : nil,
                evalURL: fm.isReadableFile(atPath: evalURL.path) ? evalURL : nil,
                evalTraceURL: fm.isReadableFile(atPath: evalTraceURL.path) ? evalTraceURL : nil,
                maskURL: fm.isReadableFile(atPath: maskURL.path) ? maskURL : nil
            )
        }
        if let embeddedIssue = materializedSidecarsEmbeddedIssue(required, outputURL: outputURL) {
            return ReviewEvidenceSidecars(
                ready: false,
                issue: embeddedIssue,
                suiteURL: embeddedSidecarIfInside(suiteURL, outputURL: outputURL),
                evalURL: embeddedSidecarIfInside(evalURL, outputURL: outputURL),
                evalTraceURL: embeddedSidecarIfInside(evalTraceURL, outputURL: outputURL),
                maskURL: embeddedSidecarIfInside(maskURL, outputURL: outputURL)
            )
        }
        let comparedPromptCount = readJSONObject(comparisonURL).flatMap {
            intValue($0["promptCount"] ?? $0["prompt_count"])
        }
        let evidenceIssue = sameSuiteEvidenceIssue(
            suiteURL: suiteURL,
            evalURL: evalURL,
            evalTraceURL: evalTraceURL,
            evalIndexURL: evalIndexURL,
            maskURL: maskURL,
            sourceModelPath: sourceURL.path,
            expectedLayerCount: expectedLayerCount,
            comparedPromptCount: comparedPromptCount
        )
        return ReviewEvidenceSidecars(
            ready: evidenceIssue == nil,
            issue: evidenceIssue,
            suiteURL: suiteURL,
            evalURL: evalURL,
            evalTraceURL: evalTraceURL,
            maskURL: maskURL
        )
    }

    private static func materializedSidecarsEmbeddedIssue(
        _ required: [(String, URL)],
        outputURL: URL
    ) -> String? {
        let outside = required.compactMap { key, url -> String? in
            embeddedSidecarIfInside(url, outputURL: outputURL) == nil ? key : nil
        }
        guard !outside.isEmpty else { return nil }
        return "materialized Expert Lab sidecar paths must be embedded in the pruned BF16/F16 source: \(outside.joined(separator: ", "))"
    }

    private static func embeddedSidecarIfInside(_ url: URL, outputURL: URL) -> URL? {
        let outputPath = canonicalPath(outputURL.path)
        let sidecarPath = canonicalPath(url.path)
        return path(sidecarPath, isInsideOrEqualTo: outputPath) ? url : nil
    }

    private static func materializedSidecarURL(
        _ value: Any?,
        outputURL: URL,
        fallbackName: String
    ) -> URL {
        guard let path = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty else {
            return outputURL.appendingPathComponent(fallbackName)
        }
        let expanded = (path as NSString).expandingTildeInPath
        if (expanded as NSString).isAbsolutePath {
            return URL(fileURLWithPath: expanded)
        }
        return outputURL.appendingPathComponent(expanded)
    }

    private func persistPrunedSourceSuiteVerificationIfNeeded(
        suiteURL: URL?,
        evalURL: URL?,
        evalTraceURL: URL?
    ) async throws -> PrunedSourceSuiteVerification {
        guard prunePlanURL != nil else {
            return PrunedSourceSuiteVerification(ready: true, issue: nil, summaryURL: nil, generationsURL: nil)
        }
        guard let suiteURL else {
            return PrunedSourceSuiteVerification(
                ready: false,
                issue: "reviewed prompt suite sidecar is missing; cannot verify pruned BF16/F16 behavior",
                summaryURL: nil,
                generationsURL: nil
            )
        }

        let fm = FileManager.default
        let runDir = outputURL.appendingPathComponent("expert_lab_pruned_vmlx_run", isDirectory: true)
        if fm.fileExists(atPath: runDir.path) {
            try fm.removeItem(at: runDir)
        }
        try fm.createDirectory(at: runDir, withIntermediateDirectories: true, attributes: nil)

        _ = try await PythonCLIInvoker.invoke(args: [
            "-m", "jang_tools",
            "--quiet-text",
            "expert-lab-vmlx",
            outputURL.path,
            "--suite", suiteURL.path,
            "--output", runDir.path,
            "--max-tokens", "64",
            "--emit-token-trace",
            "--max-trace-tokens", "32768",
        ]) { code, stderr in
            PrequantPruneError.cli(code: code, stderr: stderr)
        }

        let rawSummaryURL = runDir.appendingPathComponent("summary.json")
        let rawGenerationsURL = runDir.appendingPathComponent("generations.jsonl")
        let summaryURL = outputURL.appendingPathComponent("expert_lab_pruned_generation_summary.json")
        let generationsURL = outputURL.appendingPathComponent("expert_lab_pruned_generations.jsonl")
        _ = try copySidecarIfPresent(from: rawGenerationsURL, to: generationsURL)

        let structuralIssue = Self.prunedSourceSuiteIssue(
            suiteURL: suiteURL,
            rawSummaryURL: rawSummaryURL,
            generationsURL: generationsURL,
            prunedSourcePath: outputURL.path,
            expectedLayerCount: prunePlanSummary?.layerCount ?? Self.configLayerCount(modelPath: outputURL.path)
        )
        let reviewedBehaviorComparison = Self.prunedReviewedBehaviorComparison(
            evalURL: evalURL,
            generationsURL: generationsURL
        )
        let rawSummary = Self.readJSONObject(rawSummaryURL) ?? [:]
        let runtimeInfo = rawSummary["runtime_info"] as? [String: Any] ?? [:]
        let runtimeSourceIssue: String? = {
            guard Self.stringValue(runtimeInfo["runtime_mode"]) == "bf16_vmlx" else {
                return "pruned BF16/F16 vMLX summary did not record BF16/vMLX runtime evidence"
            }
            guard Self.stringValue(runtimeInfo["backend"]) == "vmlx" else {
                return "pruned BF16/F16 vMLX summary did not record vMLX backend evidence"
            }
            if Self.boolValue(runtimeInfo["hook_coverage_complete"] ?? runtimeInfo["hookCoverageComplete"]) == false {
                return "pruned BF16/F16 vMLX summary recorded incomplete routed-layer hook coverage"
            }
            guard let runtimeSourcePath = Self.stringValue(runtimeInfo["source_model_path"]) else {
                return "pruned BF16/F16 vMLX summary is missing source model path evidence"
            }
            if Self.canonicalPath(runtimeSourcePath) != Self.canonicalPath(outputURL.path) {
                return "pruned BF16/F16 vMLX summary source path does not match the pruned source"
            }
            guard Self.stringValue(runtimeInfo["jang_tools_version"])?.isEmpty == false,
                  Self.stringValue(runtimeInfo["mlx_version"])?.isEmpty == false,
                  Self.stringValue(runtimeInfo["mlx_lm_version"])?.isEmpty == false else {
                return "pruned BF16/F16 vMLX summary is missing package version evidence"
            }
            return nil
        }()
        let issue = structuralIssue ?? runtimeSourceIssue ?? reviewedBehaviorComparison.issue
        let payload: [String: Any] = [
            "schema": "jang-expert-lab-pruned-bf16-suite-v1",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "source_model": sourceURL.path,
            "pruned_source": outputURL.path,
            "suite_jsonl": suiteURL.path,
            "suite_sha256": Self.fileSHA256(suiteURL) ?? NSNull(),
            "generation_defaults": rawSummary["generation_defaults"] ?? NSNull(),
            "raw_summary_json": rawSummaryURL.path,
            "generations_jsonl": generationsURL.path,
            "prompt_count": Self.lineCount(suiteURL),
            "generation_count": Self.lineCount(generationsURL),
            "runtime_mode": runtimeInfo["runtime_mode"] ?? NSNull(),
            "runtime_backend": runtimeInfo["backend"] ?? NSNull(),
            "runtime_device": runtimeInfo["device_name"] ?? NSNull(),
            "runtime_metal_enabled": runtimeInfo["runtime_metal_enabled"] ?? NSNull(),
            "jang_tools_version": runtimeInfo["jang_tools_version"] ?? NSNull(),
            "mlx_version": runtimeInfo["mlx_version"] ?? NSNull(),
            "mlx_lm_version": runtimeInfo["mlx_lm_version"] ?? NSNull(),
            "mlx_vlm_version": runtimeInfo["mlx_vlm_version"] ?? NSNull(),
            "runtime_source_model_path": runtimeInfo["source_model_path"] ?? NSNull(),
            "hooked_moe_layers": runtimeInfo["hooked_moe_layers"] ?? NSNull(),
            "expected_moe_layers": runtimeInfo["expected_moe_layers"] ?? NSNull(),
            "hook_coverage_complete": runtimeInfo["hook_coverage_complete"] ?? NSNull(),
            "reviewed_masked_eval_jsonl": evalURL?.path ?? NSNull(),
            "reviewed_masked_eval_trace_jsonl": evalTraceURL?.path ?? NSNull(),
            "reviewed_masked_comparison_count": reviewedBehaviorComparison.comparedCount,
            "reviewed_masked_mean_text_delta": reviewedBehaviorComparison.meanTextDelta ?? NSNull(),
            "reviewed_masked_max_text_delta": reviewedBehaviorComparison.maxTextDelta ?? NSNull(),
            "reviewed_ab_eval_jsonl": evalURL?.path ?? NSNull(),
            "reviewed_ab_comparison_count": reviewedBehaviorComparison.comparedCount,
            "reviewed_ab_mean_text_delta": reviewedBehaviorComparison.meanTextDelta ?? NSNull(),
            "reviewed_ab_max_text_delta": reviewedBehaviorComparison.maxTextDelta ?? NSNull(),
            "reviewed_behavior_reference": "baseline_or_masked",
            "pruned_validator_outcomes_checked": true,
            "baseline_qualified_prompt_count": reviewedBehaviorComparison.baselineQualifiedPromptCount,
            "pruned_baseline_qualified_pass_rate": reviewedBehaviorComparison.prunedBaselineQualifiedPassRate ?? NSNull(),
            "pruned_classification_counts": reviewedBehaviorComparison.classificationCounts,
            "baseline_invalid_prompt_ids": reviewedBehaviorComparison.baselineInvalidPromptIDs,
            "inconclusive_prompt_ids": reviewedBehaviorComparison.inconclusivePromptIDs,
            "pruned_preserved_prompt_ids": reviewedBehaviorComparison.preservedPromptIDs,
            "pruned_degraded_prompt_ids": reviewedBehaviorComparison.degradedPromptIDs,
            "baseline_qualified_semantic_coverage": reviewedBehaviorComparison.baselineQualifiedSemanticCoverage,
            "missing_baseline_qualified_semantic_coverage": reviewedBehaviorComparison.missingBaselineQualifiedSemanticCoverage,
            "ready": issue == nil,
            "issue": issue ?? NSNull()
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: summaryURL)
        return PrunedSourceSuiteVerification(
            ready: issue == nil,
            issue: issue,
            summaryURL: summaryURL,
            generationsURL: generationsURL
        )
    }

    private func updateReviewSummaryWithPrunedSuiteVerification(
        copiedEvidence: ReviewEvidenceSidecars,
        prunedSuite: PrunedSourceSuiteVerification
    ) throws -> (ready: Bool, issue: String?) {
        guard prunePlanURL != nil else { return (true, nil) }
        let reviewSummaryURL = outputURL.appendingPathComponent("expert_lab_review_summary.json")
        var payload = Self.readJSONObject(reviewSummaryURL) ?? [:]
        payload["review_sidecars_ready"] = copiedEvidence.ready
        payload["review_sidecars_issue"] = copiedEvidence.issue ?? NSNull()
        payload["pruned_suite_verification_ready"] = prunedSuite.ready
        payload["pruned_suite_verification_issue"] = prunedSuite.issue ?? NSNull()
        payload["pruned_suite_summary"] = prunedSuite.summaryURL?.path ?? NSNull()
        payload["pruned_suite_generations"] = prunedSuite.generationsURL?.path ?? NSNull()
        let ready = copiedEvidence.ready && prunedSuite.ready
        let issue = copiedEvidence.issue ?? prunedSuite.issue
        payload["same_suite_verification_ready"] = ready
        payload["same_suite_verification_issue"] = issue ?? NSNull()
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: reviewSummaryURL)
        return (ready, issue)
    }

    private static func planEvalDirectory(from path: String?) -> URL? {
        guard let path, !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path)
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.appendingPathComponent("comparison_summary.json").path),
              fm.fileExists(atPath: url.appendingPathComponent("eval.jsonl").path),
              fm.fileExists(atPath: url.appendingPathComponent("eval_trace.jsonl").path) else {
            return nil
        }
        return url
    }

    @discardableResult
    private func copySidecarIfPresent(from source: URL, to destination: URL) throws -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: source.path) else { return false }
        if source.standardizedFileURL == destination.standardizedFileURL {
            return true
        }
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.copyItem(at: source, to: destination)
        return true
    }

    private static func latestComparisonDirectory(in evalsDirectory: URL) -> URL? {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(
            at: evalsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }
        let candidates = urls.compactMap { url -> (url: URL, date: Date)? in
            guard (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else {
                return nil
            }
            guard fm.fileExists(atPath: url.appendingPathComponent("comparison_summary.json").path) else {
                return nil
            }
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return (url, date)
        }
        return candidates.sorted {
            if $0.date != $1.date { return $0.date > $1.date }
            return $0.url.lastPathComponent > $1.url.lastPathComponent
        }.first?.url
    }

    private static func sameSuiteEvidenceIssue(
        suiteURL: URL,
        evalURL: URL,
        evalTraceURL: URL,
        evalIndexURL: URL,
        maskURL: URL? = nil,
        sourceModelPath: String? = nil,
        expectedLayerCount: Int? = nil,
        comparedPromptCount: Int? = nil
    ) -> String? {
        guard let index = readJSONObject(evalIndexURL) else {
            return "eval_index.json is unreadable"
        }
        let promptCount = intValue(index["prompt_count"] ?? index["promptCount"]) ?? 0
        let promptIDs = stringArrayValue(index["prompt_ids"] ?? index["promptIDs"]) ?? []
        if promptIDs.count != promptCount {
            return "eval_index.json lists \(promptIDs.count) prompt IDs for \(promptCount) indexed prompts"
        }
        if Set(promptIDs).count < promptIDs.count {
            return "eval_index.json contains duplicate prompt IDs"
        }
        if let issue = evalIndexSemanticCoverageIssue(index) {
            return issue
        }
        if let comparedPromptCount, promptCount != comparedPromptCount {
            return "eval_index.json covers \(promptCount) of \(comparedPromptCount) compared prompts"
        }
        guard let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]) else {
            return "suite.jsonl prompt IDs are unreadable"
        }
        if Set(suiteIDs).count < suiteIDs.count {
            return "suite.jsonl contains duplicate prompt IDs"
        }
        if let issue = suiteSemanticCoverageIssue(suiteURL) {
            return issue
        }
        let indexedIDs = Set(promptIDs)
        var evalRowsForDecodeSettings: [[String: Any]]?
        let suiteSet = Set(suiteIDs)
        let missingSuiteIDs = suiteSet.subtracting(indexedIDs)
        if !missingSuiteIDs.isEmpty {
            return "eval_index.json prompt IDs missing suite.jsonl prompts: \(previewIDs(missingSuiteIDs))"
        }
        let unexpectedSuiteIDs = indexedIDs.subtracting(suiteSet)
        if !unexpectedSuiteIDs.isEmpty {
            return "eval_index.json prompt IDs outside suite.jsonl: \(previewIDs(unexpectedSuiteIDs))"
        }
        if promptIDs != suiteIDs {
            return "eval_index.json prompt order does not match suite.jsonl"
        }
        guard let evalIDs = jsonlStringIDs(evalURL, keys: ["promptID", "prompt_id", "id"]) else {
            return "eval.jsonl prompt IDs are unreadable"
        }
        if Set(evalIDs).count < evalIDs.count {
            return "eval.jsonl contains duplicate prompt IDs"
        }
        let evalSet = Set(evalIDs)
        let missingEvalIDs = indexedIDs.subtracting(evalSet)
        if !missingEvalIDs.isEmpty {
            return "eval_index.json prompt IDs missing from eval.jsonl: \(previewIDs(missingEvalIDs))"
        }
        let unexpectedEvalIDs = evalSet.subtracting(indexedIDs)
        if !unexpectedEvalIDs.isEmpty {
            return "eval.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpectedEvalIDs))"
        }
        if evalIDs != promptIDs {
            return "eval.jsonl prompt order does not match eval_index.json"
        }
        guard let evalRows = jsonlObjects(evalURL) else {
            return "eval.jsonl is unreadable"
        }
        evalRowsForDecodeSettings = evalRows
        if let issue = evalRowEvidenceIssue(
            rows: evalRows,
            expectedPromptIDs: promptIDs,
            sourceModelPath: sourceModelPath
        ) {
            return issue
        }
        let expectedDisabledByLayer: [Int: Set<Int>]?
        if let maskURL {
            guard let mask = disabledExpertsByLayer(fromMaskURL: maskURL) else {
                return "mask.json is unreadable"
            }
            let disabledCount = mask.values.reduce(0) { $0 + $1.count }
            guard disabledCount > 0 else {
                return "mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning"
            }
            if let indexDisabledCount = intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
               indexDisabledCount != disabledCount {
                return "eval_index.json disabled expert count \(indexDisabledCount) does not match mask.json \(disabledCount)"
            }
            expectedDisabledByLayer = mask
        } else {
            expectedDisabledByLayer = nil
        }
        guard let traceIDs = jsonlStringIDs(evalTraceURL, keys: ["promptID", "prompt_id", "id"]) else {
            return "eval_trace.jsonl prompt IDs are unreadable"
        }
        guard !traceIDs.isEmpty else {
            return "eval_trace.jsonl has no routing records"
        }
        let traceSet = Set(traceIDs)
        let missingTraceIDs = indexedIDs.subtracting(traceSet)
        if !missingTraceIDs.isEmpty {
            return "eval_index.json prompt IDs missing from eval_trace.jsonl: \(previewIDs(missingTraceIDs))"
        }
        let unexpectedTraceIDs = traceSet.subtracting(indexedIDs)
        if !unexpectedTraceIDs.isEmpty {
            return "eval_trace.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpectedTraceIDs))"
        }
        guard let traceRows = jsonlObjects(evalTraceURL) else {
            return "eval_trace.jsonl is unreadable"
        }
        guard let baselineRouteRecordCount = intValue(index["baseline_route_record_count"] ?? index["baselineRouteRecordCount"]),
              let maskedRouteRecordCount = intValue(index["masked_route_record_count"] ?? index["maskedRouteRecordCount"]),
              baselineRouteRecordCount >= promptIDs.count,
              maskedRouteRecordCount >= promptIDs.count else {
            return "eval_index.json is missing routing record evidence for every indexed prompt"
        }
        if let issue = evalTraceVariantIssue(
            rows: traceRows,
            expectedPromptIDs: promptIDs,
            disabledExpertCount: intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
            topKOverride: intValue(index["top_k_override"] ?? index["topKOverride"]),
            expectedDisabledByLayer: expectedDisabledByLayer,
            expectedBaselineRouteRecordCount: baselineRouteRecordCount,
            expectedMaskedRouteRecordCount: maskedRouteRecordCount
        ) {
            return issue
        }
        let highRiskDomains = stringArrayValue(index["high_risk_domains"] ?? index["highRiskDomains"]) ?? []
        if !highRiskDomains.isEmpty {
            return "eval_index.json still has high-risk domains: \(highRiskDomains.sorted().joined(separator: ", "))"
        }
        let meanBaselineTokens = doubleValue(index["mean_baseline_tokens"] ?? index["meanBaselineTokens"])
        let meanMaskedTokens = doubleValue(index["mean_masked_tokens"] ?? index["meanMaskedTokens"])
        guard let meanBaselineTokens, let meanMaskedTokens else {
            return "eval_index.json is missing generation-depth token evidence"
        }
        let shallow = min(meanBaselineTokens, meanMaskedTokens)
        if shallow < 8 {
            return String(format: "eval_index.json average generated depth %.1f tokens is too shallow", shallow)
        }
        if let issue = evalIndexLayerStatsCoverageIssue(index: index, promptCount: promptIDs.count) {
            return issue
        }
        guard stringValue(index["eval_trace_jsonl"] ?? index["evalTraceJSONL"]) != nil else {
            return "eval_index.json is missing eval_trace.jsonl evidence"
        }
        guard let runtimeMode = stringValue(index["runtime_mode"] ?? index["runtimeMode"]),
              !runtimeMode.isEmpty,
              let runtimeDevice = stringValue(index["runtime_device"] ?? index["runtimeDevice"]),
              !runtimeDevice.isEmpty,
              let runtimeMetalEnabled = boolValue(index["runtime_metal_enabled"] ?? index["runtimeMetalEnabled"]) else {
            return "eval_index.json is missing runtime device evidence"
        }
        if runtimeMetalEnabled != true {
            return "eval_index.json did not record a Metal runtime"
        }
        if runtimeMode != "bf16_vmlx" {
            return "eval_index.json did not record BF16/vMLX runtime evidence"
        }
        if stringValue(index["runtime_backend"] ?? index["runtimeBackend"]) != "vmlx" {
            return "eval_index.json did not record vMLX backend evidence"
        }
        if boolValue(index["hook_coverage_complete"] ?? index["hookCoverageComplete"]) == false {
            return "eval_index.json recorded incomplete vMLX routed-layer hook coverage"
        }
        if let expectedLayerCount {
            guard let hookedLayers = intValue(index["hooked_moe_layers"] ?? index["hookedMOELayers"]) else {
                return "eval_index.json is missing vMLX routed-layer hook evidence"
            }
            if hookedLayers < expectedLayerCount {
                return "eval_index.json vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers"
            }
        }
        if let expectedMOELayers = intValue(index["expected_moe_layers"] ?? index["expectedMOELayers"]),
           let hookedLayers = intValue(index["hooked_moe_layers"] ?? index["hookedMOELayers"]),
           hookedLayers < expectedMOELayers {
            return "eval_index.json vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers"
        }
        guard let evalSourcePath = stringValue(index["source_model_path"] ?? index["sourceModelPath"]) else {
            return "eval_index.json is missing source model path evidence"
        }
        if let sourceModelPath,
           canonicalPath(evalSourcePath) != canonicalPath(sourceModelPath) {
            return "eval_index.json source model path does not match reviewed source"
        }
        guard boolValue(index["mask_applied"] ?? index["maskApplied"]) == true else {
            return "eval_index.json did not record an applied BF16/vMLX mask"
        }
        guard let disabledExpertCount = intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
              disabledExpertCount > 0 else {
            return "eval_index.json did not record disabled expert evidence; top-k-only comparisons cannot authorize hard pruning"
        }
        if let evalRowsForDecodeSettings,
           let issue = evalDecodeSettingsIssue(index: index, rows: evalRowsForDecodeSettings) {
            return issue
        }
        guard let expectedSuiteSHA256 = fileSHA256(suiteURL) else {
            return "eval_index.json suite.jsonl fingerprint could not be computed"
        }
        guard let recordedSuiteSHA256 = stringValue(index["suite_sha256"] ?? index["suiteSHA256"]),
              !recordedSuiteSHA256.isEmpty else {
            return "eval_index.json is missing suite.jsonl fingerprint evidence"
        }
        if recordedSuiteSHA256 != expectedSuiteSHA256 {
            return "eval_index.json suite.jsonl fingerprint does not match suite.jsonl"
        }
        return nil
    }

    private static func evalRowEvidenceIssue(
        rows: [[String: Any]],
        expectedPromptIDs: [String],
        sourceModelPath: String?
    ) -> String? {
        if rows.count != expectedPromptIDs.count {
            return "eval.jsonl has \(rows.count) rows for \(expectedPromptIDs.count) indexed prompts"
        }
        let rowIDs = rows.compactMap { promptID(in: $0) }
        if rowIDs.count != rows.count {
            return "eval.jsonl prompt IDs are unreadable"
        }
        if rowIDs != expectedPromptIDs {
            return "eval.jsonl prompt order does not match eval_index.json"
        }
        if rows.contains(where: {
            trimmedString($0["baselineText"] ?? $0["baseline_text"]) == nil
                || trimmedString($0["maskedText"] ?? $0["masked_text"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt baseline/masked output text"
        }
        if rows.contains(where: {
            guard let textDelta = doubleValue($0["textDelta"] ?? $0["text_delta"]),
                  textDelta.isFinite else {
                return true
            }
            return false
        }) {
            return "eval.jsonl is missing per-prompt text delta evidence"
        }
        if rows.contains(where: {
            (intValue($0["baselineTokenCount"] ?? $0["baseline_token_count"]) ?? 0) <= 0
                || (intValue($0["maskedTokenCount"] ?? $0["masked_token_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt token count evidence"
        }
        if rows.contains(where: {
            (intValue($0["baselineRouteRecordCount"] ?? $0["baseline_route_record_count"]) ?? 0) <= 0
                || (intValue($0["maskedRouteRecordCount"] ?? $0["masked_route_record_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt routing record evidence"
        }
        if let issue = evalRowLayerStatsCoverageIssue(rows: rows) {
            return issue
        }
        if rows.contains(where: {
            trimmedString($0["runtimeMode"] ?? $0["runtime_mode"]) == nil
                || trimmedString($0["runtimeDevice"] ?? $0["runtime_device"]) == nil
                || boolValue($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt runtime device evidence"
        }
        if rows.contains(where: {
            boolValue($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) != true
        }) {
            return "eval.jsonl did not record a Metal runtime"
        }
        if rows.contains(where: {
            trimmedString($0["runtimeMode"] ?? $0["runtime_mode"]) != "bf16_vmlx"
        }) {
            return "eval.jsonl did not record BF16/vMLX runtime evidence"
        }
        if rows.contains(where: {
            trimmedString($0["runtimeBackend"] ?? $0["runtime_backend"]) != "vmlx"
        }) {
            return "eval.jsonl did not record per-prompt vMLX backend evidence"
        }
        if rows.contains(where: {
            trimmedString($0["jangToolsVersion"] ?? $0["jang_tools_version"]) == nil
                || trimmedString($0["mlxVersion"] ?? $0["mlx_version"]) == nil
                || trimmedString($0["mlxLMVersion"] ?? $0["mlx_lm_version"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt vMLX package version evidence"
        }
        if rows.contains(where: {
            trimmedString($0["sourceModelPath"] ?? $0["source_model_path"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt source model path evidence"
        }
        if let sourceModelPath {
            let expectedSourcePath = canonicalPath(sourceModelPath)
            if rows.contains(where: {
                canonicalPath(trimmedString($0["sourceModelPath"] ?? $0["source_model_path"]) ?? "") != expectedSourcePath
            }) {
                return "eval.jsonl source model path does not match reviewed source"
            }
        }
        if rows.contains(where: {
            boolValue($0["maskApplied"] ?? $0["mask_applied"]) != true
        }) {
            return "eval.jsonl did not record an applied BF16/vMLX mask"
        }
        if rows.contains(where: {
            boolValue($0["maskApplied"] ?? $0["mask_applied"]) == true
                && (intValue($0["disabledExpertCount"] ?? $0["disabled_expert_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt disabled expert evidence; top-k-only comparisons cannot authorize hard pruning"
        }
        if rows.contains(where: {
            trimmedString($0["risk"]) == nil
                || trimmedString($0["regressionSeverity"] ?? $0["regression_severity"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt regression flag evidence"
        }
        return nil
    }

    private struct DecodeSettings {
        let maxTokens: Int
        let temperature: Double
        let topP: Double
        let topK: Int
    }

    private static func evalDecodeSettingsIssue(index: [String: Any], rows: [[String: Any]]) -> String? {
        guard boolValue(index["generation_settings_checked"] ?? index["generationSettingsChecked"]) == true else {
            return "eval_index.json is missing decode settings evidence"
        }
        for row in rows {
            let baselineValue = row["baselineGenerationSettings"] ?? row["baseline_generation_settings"]
            let maskedValue = row["maskedGenerationSettings"] ?? row["masked_generation_settings"]
            guard baselineValue != nil, maskedValue != nil else {
                return "eval.jsonl is missing baseline/masked decode settings evidence"
            }
            guard let baseline = decodeSettings(baselineValue),
                  let masked = decodeSettings(maskedValue) else {
                return "eval.jsonl has unreadable baseline/masked decode settings evidence"
            }
            if baseline.maxTokens != masked.maxTokens
                || abs(baseline.temperature - masked.temperature) > 0.000_001
                || abs(baseline.topP - masked.topP) > 0.000_001
                || baseline.topK != masked.topK {
                return "eval.jsonl baseline/masked decode settings do not match"
            }
        }
        return nil
    }

    private static func decodeSettings(_ value: Any?) -> DecodeSettings? {
        guard let object = value as? [String: Any],
              let maxTokens = intValue(object["max_tokens"] ?? object["maxTokens"]),
              maxTokens > 0,
              let temperature = doubleValue(object["temperature"]),
              temperature.isFinite,
              let topP = doubleValue(object["top_p"] ?? object["topP"]),
              topP.isFinite,
              let topK = intValue(object["top_k"] ?? object["topK"]),
              topK >= 0 else {
            return nil
        }
        return DecodeSettings(maxTokens: maxTokens, temperature: temperature, topP: topP, topK: topK)
    }

    private static func evalIndexLayerStatsCoverageIssue(index: [String: Any], promptCount: Int) -> String? {
        let baselineCount = intValue(index["baseline_layer_stats_prompt_count"] ?? index["baselineLayerStatsPromptCount"])
        let maskedCount = intValue(index["masked_layer_stats_prompt_count"] ?? index["maskedLayerStatsPromptCount"])
        guard baselineCount != nil || maskedCount != nil else { return nil }
        guard baselineCount == promptCount, maskedCount == promptCount else {
            return "eval_index.json layer-stat coverage is incomplete for indexed prompts"
        }
        return nil
    }

    private static func evalRowLayerStatsCoverageIssue(rows: [[String: Any]]) -> String? {
        let baselineRows = rows.filter { nonEmptyJSONArray($0["baselineLayerStats"] ?? $0["baseline_layer_stats"]) }.count
        let maskedRows = rows.filter { nonEmptyJSONArray($0["maskedLayerStats"] ?? $0["masked_layer_stats"]) }.count
        guard baselineRows > 0 || maskedRows > 0 else { return nil }
        guard baselineRows == rows.count, maskedRows == rows.count else {
            return "eval.jsonl layer-stat evidence is incomplete for baseline/masked prompts"
        }
        return nil
    }

    private static func evalTraceVariantIssue(
        rows: [[String: Any]],
        expectedPromptIDs: [String],
        disabledExpertCount: Int?,
        topKOverride: Int?,
        expectedDisabledByLayer: [Int: Set<Int>]? = nil,
        expectedBaselineRouteRecordCount: Int? = nil,
        expectedMaskedRouteRecordCount: Int? = nil
    ) -> String? {
        let expected = Set(expectedPromptIDs)
        let expectedMaskLayers = Set((expectedDisabledByLayer ?? [:]).filter { !$0.value.isEmpty }.keys)
        var baselinePromptIDs = Set<String>()
        var maskedPromptIDs = Set<String>()
        var maskedPromptIDsWithMaskEvidence = Set<String>()
        var maskedPromptExpectedMaskLayers: [String: Set<Int>] = [:]
        var baselineTraceCount = 0
        var maskedTraceCount = 0

        for row in rows {
            guard let id = promptID(in: row) else {
                return "eval_trace.jsonl prompt IDs are unreadable"
            }
            let variant = trimmedString(row["variant"])?.lowercased()
            switch variant {
            case "baseline":
                baselineTraceCount += 1
                baselinePromptIDs.insert(id)
            case "masked":
                maskedTraceCount += 1
                maskedPromptIDs.insert(id)
                if let issue = traceRowDisabledSelectionIssue(row, promptID: id) {
                    return issue
                }
                if let layer = traceRowExpectedMaskEvidenceLayer(
                    row,
                    expectedDisabledByLayer: expectedDisabledByLayer
                ) {
                    maskedPromptExpectedMaskLayers[id, default: []].insert(layer)
                }
                if traceRowHasMaskEvidence(
                    row,
                    disabledExpertCount: disabledExpertCount,
                    topKOverride: topKOverride
                ) {
                    maskedPromptIDsWithMaskEvidence.insert(id)
                }
            default:
                continue
            }
        }

        let missingBaseline = expected.subtracting(baselinePromptIDs)
        if !missingBaseline.isEmpty {
            return "eval_trace.jsonl missing baseline routing records for prompt IDs: \(previewIDs(missingBaseline))"
        }
        let missingMasked = expected.subtracting(maskedPromptIDs)
        if !missingMasked.isEmpty {
            return "eval_trace.jsonl missing masked routing records for prompt IDs: \(previewIDs(missingMasked))"
        }
        if let expectedBaselineRouteRecordCount,
           baselineTraceCount != expectedBaselineRouteRecordCount {
            return "eval_trace.jsonl has \(baselineTraceCount) baseline routing records for \(expectedBaselineRouteRecordCount) indexed baseline route records"
        }
        if let expectedMaskedRouteRecordCount,
           maskedTraceCount != expectedMaskedRouteRecordCount {
            return "eval_trace.jsonl has \(maskedTraceCount) masked routing records for \(expectedMaskedRouteRecordCount) indexed masked route records"
        }
        if (disabledExpertCount ?? 0) > 0 || topKOverride != nil {
            let missingMaskEvidence = expected.subtracting(maskedPromptIDsWithMaskEvidence)
            if !missingMaskEvidence.isEmpty {
                return "eval_trace.jsonl masked routing records are missing mask evidence for prompt IDs: \(previewIDs(missingMaskEvidence))"
            }
        }
        if !expectedMaskLayers.isEmpty {
            for id in expectedPromptIDs {
                let missingLayers = expectedMaskLayers.subtracting(maskedPromptExpectedMaskLayers[id] ?? [])
                if !missingLayers.isEmpty {
                    return "eval_trace.jsonl masked routing records are missing mask.json evidence for prompt \(id) layers: \(previewInts(missingLayers))"
                }
            }
        }
        return nil
    }

    private static func traceRowExpectedMaskEvidenceLayer(
        _ row: [String: Any],
        expectedDisabledByLayer: [Int: Set<Int>]?
    ) -> Int? {
        guard let expectedDisabledByLayer,
              let record = row["record"] as? [String: Any],
              let layer = intValue(record["layer"] ?? record["layerIndex"] ?? record["layer_index"]),
              let expectedDisabled = expectedDisabledByLayer[layer],
              !expectedDisabled.isEmpty else {
            return nil
        }
        let disabled = Set(arrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap { intValue($0) })
        return expectedDisabled.isSubset(of: disabled) ? layer : nil
    }

    private static func traceRowDisabledSelectionIssue(_ row: [String: Any], promptID: String) -> String? {
        guard let record = row["record"] as? [String: Any] else { return nil }
        let disabled = Set(arrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap { intValue($0) })
        guard !disabled.isEmpty else { return nil }
        let selected = Set(arrayValue(record["selectedExperts"] ?? record["selected_experts"]).compactMap { intValue($0) })
        let leaked = selected.intersection(disabled)
        guard !leaked.isEmpty else { return nil }
        let sorted = leaked.sorted()
        let head = sorted.prefix(5).map { String($0) }.joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        let preview = remaining == 0 ? head : "\(head), +\(remaining) more"
        return "eval_trace.jsonl masked routing records selected disabled experts for prompt \(promptID): \(preview)"
    }

    private static func traceRowHasMaskEvidence(
        _ row: [String: Any],
        disabledExpertCount: Int?,
        topKOverride: Int?
    ) -> Bool {
        guard let record = row["record"] as? [String: Any] else { return false }
        if (disabledExpertCount ?? 0) > 0 {
            let disabledExperts = arrayValue(record["disabledExperts"] ?? record["disabled_experts"])
            if !disabledExperts.isEmpty { return true }
            if (intValue(record["disabledExpertCount"] ?? record["disabled_expert_count"]) ?? 0) > 0 {
                return true
            }
            return false
        }
        if topKOverride != nil {
            return intValue(record["effectiveTopK"] ?? record["effective_top_k"] ?? record["topK"] ?? record["top_k"]) != nil
        }
        return true
    }

    private static func disabledExpertsByLayer(fromMaskURL url: URL) -> [Int: Set<Int>]? {
        guard let mask = readJSONObject(url) else { return nil }
        for key in ["disabled_by_layer", "layers", "disabledExpertsByLayer"] {
            if let map = intSetMap(mask[key]) {
                return map.filter { !$0.value.isEmpty }
            }
        }
        return [:]
    }

    private static func intSetMap(_ value: Any?) -> [Int: Set<Int>]? {
        guard let dictionary = value as? [String: Any] else { return nil }
        var result: [Int: Set<Int>] = [:]
        for (key, rawExperts) in dictionary {
            guard let layer = intValue(key) else { return nil }
            result[layer] = Set(arrayValue(rawExperts).compactMap { intValue($0) })
        }
        return result
    }

    private static func prunedSourceSuiteIssue(
        suiteURL: URL,
        rawSummaryURL: URL,
        generationsURL: URL,
        prunedSourcePath: String? = nil,
        expectedLayerCount: Int? = nil
    ) -> String? {
        guard let summary = readJSONObject(rawSummaryURL) else {
            return "pruned BF16/F16 vMLX summary is unreadable"
        }
        guard boolValue(summary["ok"]) == true else {
            return "pruned BF16/F16 vMLX run did not report ok"
        }
        guard let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]),
              !suiteIDs.isEmpty else {
            return "reviewed prompt suite IDs are unreadable"
        }
        if Set(suiteIDs).count < suiteIDs.count {
            return "reviewed prompt suite contains duplicate prompt IDs"
        }
        let expectedCount = suiteIDs.count
        let summaryCount = intValue(summary["prompt_count"]) ?? 0
        if summaryCount != expectedCount {
            return "pruned BF16/F16 vMLX summary covers \(summaryCount) of \(expectedCount) suite prompts"
        }
        guard let rows = jsonlObjects(generationsURL) else {
            return "pruned BF16/F16 generation JSONL is unreadable"
        }
        if rows.count != expectedCount {
            return "pruned BF16/F16 generation JSONL has \(rows.count) rows for \(expectedCount) suite prompts"
        }
        let generatedIDs = rows.compactMap { row -> String? in
            guard let prompt = row["prompt"] as? [String: Any] else { return nil }
            return promptID(in: prompt)
        }
        if generatedIDs.count != rows.count {
            return "pruned BF16/F16 generation JSONL is missing prompt IDs"
        }
        if Set(generatedIDs).count < generatedIDs.count {
            return "pruned BF16/F16 generation JSONL contains duplicate prompt IDs"
        }
        let expectedIDs = Set(suiteIDs)
        let actualIDs = Set(generatedIDs)
        let missing = expectedIDs.subtracting(actualIDs)
        if !missing.isEmpty {
            return "pruned BF16/F16 generation JSONL is missing suite prompt IDs: \(previewIDs(missing))"
        }
        let unexpected = actualIDs.subtracting(expectedIDs)
        if !unexpected.isEmpty {
            return "pruned BF16/F16 generation JSONL has prompt IDs outside the reviewed suite: \(previewIDs(unexpected))"
        }
        if generatedIDs != suiteIDs {
            return "pruned BF16/F16 generation prompt order does not match reviewed suite"
        }
        for row in rows {
            guard let result = row["result"] as? [String: Any] else {
                return "pruned BF16/F16 generation row is missing result"
            }
            let text = stringValue(result["text"])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let tokens = intValue(result["tokens"]) ?? 0
            if text.isEmpty || tokens <= 0 {
                return "pruned BF16/F16 generation produced an empty prompt output"
            }
            guard let runtime = result["runtime_info"] as? [String: Any],
                  let runtimeMode = stringValue(runtime["runtime_mode"]),
                  !runtimeMode.isEmpty,
                  let runtimeDevice = stringValue(runtime["device_name"]),
                  !runtimeDevice.isEmpty,
                  boolValue(runtime["runtime_metal_enabled"]) == true else {
                return "pruned BF16/F16 generation is missing vMLX Metal runtime evidence"
            }
            if runtimeMode != "bf16_vmlx" {
                return "pruned BF16/F16 generation did not record BF16/vMLX runtime evidence"
            }
            if stringValue(runtime["backend"] ?? runtime["runtime_backend"] ?? runtime["runtimeBackend"]) != "vmlx" {
                return "pruned BF16/F16 generation did not record vMLX backend evidence"
            }
            if boolValue(runtime["hook_coverage_complete"] ?? runtime["hookCoverageComplete"]) == false {
                return "pruned BF16/F16 generation recorded incomplete vMLX routed-layer hook coverage"
            }
            if let expectedLayerCount {
                guard let hookedLayers = intValue(runtime["hooked_moe_layers"] ?? runtime["hookedMOELayers"]) else {
                    return "pruned BF16/F16 generation is missing vMLX routed-layer hook evidence"
                }
                if hookedLayers < expectedLayerCount {
                    return "pruned BF16/F16 generation vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers"
                }
            }
            if let expectedMOELayers = intValue(runtime["expected_moe_layers"] ?? runtime["expectedMOELayers"]),
               let hookedLayers = intValue(runtime["hooked_moe_layers"] ?? runtime["hookedMOELayers"]),
               hookedLayers < expectedMOELayers {
                return "pruned BF16/F16 generation vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers"
            }
            guard let runtimeSourcePath = stringValue(runtime["source_model_path"]) else {
                return "pruned BF16/F16 generation is missing source model path evidence"
            }
            if let prunedSourcePath,
               canonicalPath(runtimeSourcePath) != canonicalPath(prunedSourcePath) {
                return "pruned BF16/F16 generation source path does not match the pruned source"
            }
            guard stringValue(runtime["jang_tools_version"] ?? runtime["jangToolsVersion"])?.isEmpty == false,
                  stringValue(runtime["mlx_version"] ?? runtime["mlxVersion"])?.isEmpty == false,
                  stringValue(runtime["mlx_lm_version"] ?? runtime["mlxLMVersion"])?.isEmpty == false else {
                return "pruned BF16/F16 generation is missing per-prompt vMLX package version evidence"
            }
            if let issue = prunedGenerationLayerStatsIssue(
                result: result,
                expectedLayerCount: expectedLayerCount
            ) {
                return issue
            }
            if let issue = prunedGenerationTokenTraceIssue(
                result: result,
                expectedLayerCount: expectedLayerCount
            ) {
                return issue
            }
        }
        return nil
    }

    private static func prunedGenerationLayerStatsIssue(
        result: [String: Any],
        expectedLayerCount: Int?
    ) -> String? {
        guard expectedLayerCount != nil else { return nil }
        guard let rows = result["layer_stats"] as? [[String: Any]], !rows.isEmpty else {
            return "pruned BF16/F16 generation is missing per-prompt routed-layer stats"
        }
        let layerIDs = rows.compactMap { intValue($0["layer"] ?? $0["layer_id"] ?? $0["layerID"]) }
        if layerIDs.count != rows.count {
            return "pruned BF16/F16 generation routed-layer stats have unreadable layer IDs"
        }
        if Set(layerIDs).count < layerIDs.count {
            return "pruned BF16/F16 generation routed-layer stats contain duplicate layers"
        }
        if let expectedLayerCount {
            if rows.count < expectedLayerCount {
                return "pruned BF16/F16 generation routed-layer stats cover \(rows.count) of \(expectedLayerCount) layers"
            }
        }
        if rows.contains(where: { (intValue($0["token_count"] ?? $0["tokenCount"]) ?? 0) <= 0 }) {
            return "pruned BF16/F16 generation routed-layer stats are missing token-position depth"
        }
        if rows.contains(where: { layerStatMap($0["hit_counts"] ?? $0["hitCounts"]).isEmpty }) {
            return "pruned BF16/F16 generation routed-layer stats are missing expert hit counts"
        }
        if rows.contains(where: { layerStatMap($0["probability_mass"] ?? $0["probabilityMass"]).isEmpty }) {
            return "pruned BF16/F16 generation routed-layer stats are missing expert gate-mass evidence"
        }
        return nil
    }

    private static func prunedGenerationTokenTraceIssue(
        result: [String: Any],
        expectedLayerCount: Int?
    ) -> String? {
        guard expectedLayerCount != nil else { return nil }
        guard let layerStats = result["layer_stats"] as? [[String: Any]], !layerStats.isEmpty else {
            return nil
        }
        let expectedRoutes = layerStats.reduce(0) {
            $0 + (intValue($1["token_count"] ?? $1["tokenCount"]) ?? 0)
        }
        guard expectedRoutes > 0 else {
            return "pruned BF16/F16 generation is missing per-prompt routed layer-token records"
        }
        guard let trace = result["token_trace"] as? [[String: Any]], !trace.isEmpty else {
            return "pruned BF16/F16 generation is missing per-prompt token_trace routing evidence"
        }
        if trace.count != expectedRoutes {
            return "pruned BF16/F16 generation token_trace has \(trace.count) rows for \(expectedRoutes) routed layer-token records"
        }
        for row in trace {
            guard intValue(row["layer"]) != nil,
                  intValue(row["token_index"] ?? row["tokenIndex"]) != nil else {
                return "pruned BF16/F16 generation token_trace is missing layer/token evidence"
            }
            if arrayValue(row["selected_experts"] ?? row["selectedExperts"]).isEmpty {
                return "pruned BF16/F16 generation token_trace is missing selected expert evidence"
            }
        }
        return nil
    }

    private static func layerStatMap(_ value: Any?) -> [String: Any] {
        if let value = value as? [String: Any] { return value }
        return [:]
    }

    private static let maximumPrunedReviewedTextDelta = 0.50

    private static func prunedReviewedBehaviorComparison(
        evalURL: URL?,
        generationsURL: URL
    ) -> PrunedMaskedBehaviorComparison {
        func failed(_ issue: String, comparedCount: Int = 0) -> PrunedMaskedBehaviorComparison {
            PrunedMaskedBehaviorComparison(
                issue: issue,
                comparedCount: comparedCount,
                meanTextDelta: nil,
                maxTextDelta: nil,
                baselineQualifiedPromptCount: 0,
                prunedBaselineQualifiedPassRate: nil,
                classificationCounts: classificationCounts([]),
                baselineInvalidPromptIDs: [],
                inconclusivePromptIDs: [],
                preservedPromptIDs: [],
                degradedPromptIDs: [],
                baselineQualifiedSemanticCoverage: [],
                missingBaselineQualifiedSemanticCoverage: Array(ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains).sorted()
            )
        }

        guard let evalURL else {
            return failed("reviewed eval sidecar is missing; cannot compare pruned BF16/F16 behavior")
        }
        guard let evalRows = jsonlObjects(evalURL) else {
            return failed("reviewed masked eval JSONL is unreadable")
        }
        guard let generationRows = jsonlObjects(generationsURL) else {
            return failed("pruned BF16/F16 generation JSONL is unreadable")
        }

        var seenEvalPromptIDs = Set<String>()
        var duplicateEvalPromptIDs = Set<String>()
        var reviewedTextsByPromptID: [String: [String]] = [:]
        var reviewedRowsByPromptID: [String: [String: Any]] = [:]
        var reviewedOrder: [String] = []
        for row in evalRows {
            guard let id = promptID(in: row) else {
                continue
            }
            if !seenEvalPromptIDs.insert(id).inserted {
                duplicateEvalPromptIDs.insert(id)
                continue
            }
            reviewedRowsByPromptID[id] = row
            reviewedOrder.append(id)
            let baselineText = trimmedString(row["baselineText"] ?? row["baseline_text"])
            let maskedText = trimmedString(row["maskedText"] ?? row["masked_text"])
            let texts = [baselineText, maskedText].compactMap { text -> String? in
                guard let text, !text.isEmpty else { return nil }
                return text
            }
            if !texts.isEmpty { reviewedTextsByPromptID[id] = texts }
        }
        if !duplicateEvalPromptIDs.isEmpty {
            return failed(
                "reviewed eval JSONL contains duplicate prompt IDs: \(previewIDs(duplicateEvalPromptIDs))",
                comparedCount: reviewedTextsByPromptID.count
            )
        }
        guard !reviewedTextsByPromptID.isEmpty else {
            return failed("reviewed eval JSONL has no baseline or masked output text")
        }

        var seenPrunedPromptIDs = Set<String>()
        var duplicatePrunedPromptIDs = Set<String>()
        var prunedByPromptID: [String: String] = [:]
        for row in generationRows {
            guard let prompt = row["prompt"] as? [String: Any],
                  let id = promptID(in: prompt) else {
                continue
            }
            if !seenPrunedPromptIDs.insert(id).inserted {
                duplicatePrunedPromptIDs.insert(id)
                continue
            }
            guard
                  let result = row["result"] as? [String: Any],
                  let text = trimmedString(result["text"]),
                  !text.isEmpty else {
                continue
            }
            prunedByPromptID[id] = text
        }
        if !duplicatePrunedPromptIDs.isEmpty {
            return failed(
                "pruned BF16/F16 generation JSONL contains duplicate prompt IDs: \(previewIDs(duplicatePrunedPromptIDs))",
                comparedCount: prunedByPromptID.count
            )
        }

        let expectedIDs = Set(reviewedRowsByPromptID.keys)
        let generatedIDs = Set(prunedByPromptID.keys)
        let missing = expectedIDs.subtracting(generatedIDs)
        if !missing.isEmpty {
            return failed(
                "pruned BF16/F16 generation is missing reviewed eval prompt IDs: \(previewIDs(missing))",
                comparedCount: prunedByPromptID.count
            )
        }
        let unexpected = generatedIDs.subtracting(expectedIDs)
        if !unexpected.isEmpty {
            return failed(
                "pruned BF16/F16 generation has prompt IDs missing from reviewed eval: \(previewIDs(unexpected))",
                comparedCount: prunedByPromptID.count
            )
        }

        let deltas = reviewedOrder.compactMap { id -> (id: String, delta: Double)? in
            guard let reviewedTexts = reviewedTextsByPromptID[id],
                  let pruned = prunedByPromptID[id] else {
                return nil
            }
            let bestDelta = reviewedTexts
                .map { normalizedTextDelta($0, pruned) }
                .min() ?? 1
            return (id, bestDelta)
        }
        guard !deltas.isEmpty else {
            return failed("pruned BF16/F16 behavior comparison has no overlapping prompt IDs")
        }
        let values = deltas.map(\.delta)
        let meanDelta = values.reduce(0, +) / Double(values.count)
        let maxDelta = values.max() ?? 0

        var classifications: [String] = []
        var baselineInvalidPromptIDs: [String] = []
        var inconclusivePromptIDs: [String] = []
        var preservedPromptIDs: [String] = []
        var degradedPromptIDs: [String] = []
        var baselineQualifiedRows: [[String: Any]] = []
        var prunedPasses: [Bool] = []

        for id in reviewedOrder {
            guard let row = reviewedRowsByPromptID[id],
                  let pruned = prunedByPromptID[id] else { continue }
            let baselineQualified = boolValue(row["baselineQualified"] ?? row["baseline_qualified"])
                ?? (boolValue(row["baselinePassed"] ?? row["baseline_passed"]) == true)
            let reviewedClassification = promptClassification(row)
            let prunedPassed = validatorPassed(text: pruned, row: row)
            let classification: String
            if baselineQualified {
                baselineQualifiedRows.append(row)
                if prunedPassed == true {
                    classification = "preserved"
                    prunedPasses.append(true)
                } else {
                    classification = "degraded"
                    prunedPasses.append(false)
                }
            } else if reviewedClassification == "baseline_invalid" {
                classification = "baseline_invalid"
            } else {
                classification = "inconclusive"
            }
            classifications.append(classification)
            switch classification {
            case "baseline_invalid": baselineInvalidPromptIDs.append(id)
            case "preserved": preservedPromptIDs.append(id)
            case "degraded": degradedPromptIDs.append(id)
            default: inconclusivePromptIDs.append(id)
            }
        }

        let baselineQualifiedCoverage = semanticCoverage(fromEvalRows: baselineQualifiedRows)
        let missingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(baselineQualifiedCoverage))
            .sorted()
        let passRate = passRate(prunedPasses)
        let issue: String?
        if baselineQualifiedRows.isEmpty {
            issue = "reviewed eval JSONL has no baseline-qualified validator prompts"
        } else if !missingCoverage.isEmpty {
            issue = "baseline-qualified prompts are missing semantic coverage: \(missingCoverage.joined(separator: ", "))"
        } else if !degradedPromptIDs.isEmpty {
            issue = "pruned BF16/F16 generation failed validators for baseline-qualified prompts: \(previewIDs(Set(degradedPromptIDs)))"
        } else if let passRate, passRate < 1.0 {
            issue = "pruned BF16/F16 validator pass rate is below 100% on baseline-qualified prompts"
        } else {
            issue = nil
        }
        return PrunedMaskedBehaviorComparison(
            issue: issue,
            comparedCount: deltas.count,
            meanTextDelta: meanDelta,
            maxTextDelta: maxDelta,
            baselineQualifiedPromptCount: baselineQualifiedRows.count,
            prunedBaselineQualifiedPassRate: passRate,
            classificationCounts: classificationCounts(classifications),
            baselineInvalidPromptIDs: baselineInvalidPromptIDs,
            inconclusivePromptIDs: inconclusivePromptIDs,
            preservedPromptIDs: preservedPromptIDs,
            degradedPromptIDs: degradedPromptIDs,
            baselineQualifiedSemanticCoverage: baselineQualifiedCoverage,
            missingBaselineQualifiedSemanticCoverage: missingCoverage
        )
    }

    private static func promptClassification(_ row: [String: Any]) -> String {
        if let classification = trimmedString(row["promptClassification"] ?? row["prompt_classification"]) {
            return classification
        }
        guard let baselinePassed = boolValue(row["baselinePassed"] ?? row["baseline_passed"]),
              let maskedPassed = boolValue(row["maskedPassed"] ?? row["masked_passed"]) else {
            return "inconclusive"
        }
        if !baselinePassed { return "baseline_invalid" }
        return maskedPassed ? "preserved" : "degraded"
    }

    private static func classificationCounts(_ classifications: [String]) -> [String: Int] {
        var counts = [
            "baseline_invalid": 0,
            "preserved": 0,
            "degraded": 0,
            "inconclusive": 0
        ]
        for classification in classifications {
            counts[classification, default: 0] += 1
        }
        return counts
    }

    private static func semanticCoverage(fromEvalRows rows: [[String: Any]]) -> [String] {
        let domains = rows.flatMap { row -> [String] in
            if let semantic = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]),
               !semantic.isEmpty {
                return semantic.map(ExpertDomainTaxonomy.canonicalSemanticDomain)
            }
            if let domain = trimmedString(row["domain"]) {
                return [ExpertDomainTaxonomy.canonicalSemanticDomain(domain)]
            }
            return []
        }
        return Array(Set(domains)).filter { $0 != "general" }.sorted()
    }

    private static func passRate(_ values: [Bool]) -> Double? {
        guard !values.isEmpty else { return nil }
        return Double(values.filter { $0 }.count) / Double(values.count)
    }

    private static func validatorPassed(text: String, row: [String: Any]) -> Bool? {
        let kind = (trimmedString(row["validatorKind"] ?? row["validator_kind"])
            ?? trimmedString(row["expectedKind"] ?? row["expected_kind"])
            ?? "freeform")
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        let normalizedKind: String
        switch kind {
        case "normalized_exact", "equals", "match":
            normalizedKind = "exact"
        case "regexp":
            normalizedKind = "regex"
        case "unit_test_expected_regex":
            normalizedKind = "unit_test"
        case "substring":
            normalizedKind = "contains"
        default:
            normalizedKind = kind
        }
        let expected = trimmedString(row["expected"])
        switch normalizedKind {
        case "exact":
            guard let expected else { return nil }
            return text.trimmingCharacters(in: .whitespacesAndNewlines) == expected.trimmingCharacters(in: .whitespacesAndNewlines)
        case "regex", "unit_test":
            guard let expected,
                  let regex = try? NSRegularExpression(pattern: expected) else { return nil }
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            return regex.firstMatch(in: text, options: [], range: range) != nil
        case "contains":
            guard let expected else { return nil }
            return text.contains(expected)
        case "json":
            return jsonObject(in: text) != nil
        default:
            return nil
        }
    }

    private static func jsonObject(in text: String) -> Any? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidates: [String]
        if trimmed.hasPrefix("{"), trimmed.hasSuffix("}") {
            candidates = [trimmed]
        } else if let start = trimmed.firstIndex(of: "{"),
                  let end = trimmed.lastIndex(of: "}"),
                  start < end {
            candidates = [String(trimmed[start...end])]
        } else {
            candidates = []
        }
        for candidate in candidates {
            if let data = candidate.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: data) {
                return object
            }
        }
        return nil
    }

    static func writeRecoveredEvalIndexIfPossible(
        reviewRunDirectory: URL,
        evalDirectory: URL?,
        suiteURL: URL,
        evalURL: URL,
        evalTraceURL: URL,
        comparisonURL: URL,
        destination: URL
    ) throws -> Bool {
        guard let evalDirectory,
              let comparison = readJSONObject(comparisonURL),
              let promptIDs = jsonlStringIDs(evalURL, keys: ["promptID", "prompt_id", "id"]),
              !promptIDs.isEmpty else {
            return false
        }
        let promptCount = intValue(comparison["promptCount"] ?? comparison["prompt_count"]) ?? promptIDs.count
        guard promptIDs.count == promptCount,
              Set(promptIDs).count == promptIDs.count else {
            return false
        }
        let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]) ?? []
        guard Set(suiteIDs).count == suiteIDs.count,
              promptIDs == suiteIDs else {
            return false
        }
        guard let traceIDs = jsonlStringIDs(evalTraceURL, keys: ["promptID", "prompt_id", "id"]),
              !traceIDs.isEmpty,
              Set(traceIDs) == Set(promptIDs) else {
            return false
        }
        let evalRows = jsonlObjects(evalURL) ?? []
        let baselineTokenCounts = evalRows.compactMap {
            intValue($0["baselineTokenCount"] ?? $0["baseline_token_count"])
        }
        let maskedTokenCounts = evalRows.compactMap {
            intValue($0["maskedTokenCount"] ?? $0["masked_token_count"])
        }
        let hasCompleteTokenCounts = baselineTokenCounts.count == promptIDs.count
            && maskedTokenCounts.count == promptIDs.count
        let baselineRouteRecordCounts = evalRows.compactMap {
            intValue($0["baselineRouteRecordCount"] ?? $0["baseline_route_record_count"])
        }
        let maskedRouteRecordCounts = evalRows.compactMap {
            intValue($0["maskedRouteRecordCount"] ?? $0["masked_route_record_count"])
        }
        let hasCompleteRouteCounts = baselineRouteRecordCounts.count == promptIDs.count
            && maskedRouteRecordCounts.count == promptIDs.count
        let generationSettingsChecked = recoveredGenerationSettingsChecked(
            rows: evalRows,
            promptCount: promptIDs.count
        )
        let baselineLayerStatsPromptCount = evalRows.filter {
            nonEmptyJSONArray($0["baselineLayerStats"] ?? $0["baseline_layer_stats"])
        }.count
        let maskedLayerStatsPromptCount = evalRows.filter {
            nonEmptyJSONArray($0["maskedLayerStats"] ?? $0["masked_layer_stats"])
        }.count
        let riskyPromptIDs = evalRows.compactMap { row -> String? in
            guard let id = ["promptID", "prompt_id", "id"].lazy.compactMap({ row[$0] as? String }).first else {
                return nil
            }
            let classification = promptClassification(row)
            if classification == "degraded" {
                return id
            }
            if ["baseline_invalid", "inconclusive", "preserved"].contains(classification) {
                return nil
            }
            let risk = row["risk"] as? String
            let textDelta = doubleValue(row["textDelta"] ?? row["text_delta"]) ?? 0
            let maskedPassed = boolValue(row["maskedPassed"] ?? row["masked_passed"])
            return risk == "regression" || maskedPassed == false || textDelta > 0.50 ? id : nil
        }
        let rowSeverities = evalRows.map { row -> String in
            let classification = promptClassification(row)
            if classification == "degraded" {
                return "critical"
            }
            if ["baseline_invalid", "inconclusive", "preserved"].contains(classification) {
                if let severity = stringValue(row["regressionSeverity"] ?? row["regression_severity"]),
                   severity == "watch" {
                    return severity
                }
                return "none"
            }
            if let severity = stringValue(row["regressionSeverity"] ?? row["regression_severity"]),
               !severity.isEmpty {
                return severity
            }
            let risk = row["risk"] as? String
            let textDelta = doubleValue(row["textDelta"] ?? row["text_delta"]) ?? 0
            let maskedPassed = boolValue(row["maskedPassed"] ?? row["masked_passed"])
            if risk == "regression" { return "critical" }
            if maskedPassed == false || textDelta > 0.50 { return "high" }
            if textDelta > 0.20 || risk == "masked_improved" || risk == "failed_baseline" { return "watch" }
            return "none"
        }
        let regressionSeverity = stringValue(comparison["regressionSeverity"] ?? comparison["regression_severity"])
            ?? maxRegressionSeverity(rowSeverities)
        let highRiskDomains = recoveredHighRiskDomains(comparison: comparison, evalRows: evalRows)
        let recoveredCoverage = recoveredSemanticCoverage(from: evalRows)
        let missingSemanticCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(recoveredCoverage))
            .sorted()
        let classifications = evalRows.map(promptClassification)
        let baselineQualifiedRows = evalRows.filter {
            boolValue($0["baselineQualified"] ?? $0["baseline_qualified"])
                ?? (boolValue($0["baselinePassed"] ?? $0["baseline_passed"]) == true)
        }
        let baselineQualifiedCoverage = semanticCoverage(fromEvalRows: baselineQualifiedRows)
        let missingBaselineQualifiedCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(baselineQualifiedCoverage))
            .sorted()
        let degradedPromptIDs = evalRows.compactMap { row -> String? in
            promptClassification(row) == "degraded" ? promptID(in: row) : nil
        }
        let runtimeRecord = evalRows.first {
            stringValue($0["runtimeMode"] ?? $0["runtime_mode"]) != nil
                || stringValue($0["runtimeBackend"] ?? $0["runtime_backend"]) != nil
                || stringValue($0["runtimeDevice"] ?? $0["runtime_device"]) != nil
                || boolValue($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) != nil
                || stringValue($0["sourceModelPath"] ?? $0["source_model_path"]) != nil
                || stringValue($0["jangToolsVersion"] ?? $0["jang_tools_version"]) != nil
                || stringValue($0["mlxVersion"] ?? $0["mlx_version"]) != nil
                || stringValue($0["mlxLMVersion"] ?? $0["mlx_lm_version"]) != nil
                || boolValue($0["maskApplied"] ?? $0["mask_applied"]) != nil
        }
        var payload: [String: Any] = [
            "schema": "jang-expert-lab-eval-index-v1",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "run_id": reviewRunDirectory.lastPathComponent,
            "mask_id": evalDirectory.lastPathComponent,
            "prompt_count": promptIDs.count,
            "risky_prompt_ids": riskyPromptIDs,
            "prompt_ids": promptIDs,
            "high_risk_domains": highRiskDomains,
            "pass_rate_baseline": comparison["passRateBaseline"] ?? comparison["pass_rate_baseline"] ?? NSNull(),
            "pass_rate_masked": comparison["passRateMasked"] ?? comparison["pass_rate_masked"] ?? NSNull(),
            "validator_schema": "jang-expert-lab-validator-v1",
            "validator_available_prompt_count": evalRows.filter {
                boolValue($0["validatorAvailable"] ?? $0["validator_available"]) == true
            }.count,
            "prompt_classification_counts": classificationCounts(classifications),
            "baseline_qualified_prompt_count": baselineQualifiedRows.count,
            "baseline_qualified_prompt_ids": baselineQualifiedRows.compactMap(promptID),
            "baseline_invalid_prompt_ids": evalRows.compactMap { row in
                promptClassification(row) == "baseline_invalid" ? promptID(in: row) : nil
            },
            "inconclusive_prompt_ids": evalRows.compactMap { row in
                promptClassification(row) == "inconclusive" ? promptID(in: row) : nil
            },
            "preserved_prompt_ids": evalRows.compactMap { row in
                promptClassification(row) == "preserved" ? promptID(in: row) : nil
            },
            "degraded_prompt_ids": degradedPromptIDs,
            "baseline_qualified_masked_pass_rate": passRate(
                baselineQualifiedRows.compactMap {
                    boolValue($0["maskedPassed"] ?? $0["masked_passed"])
                }
            ) ?? NSNull(),
            "baseline_qualified_semantic_coverage": baselineQualifiedCoverage,
            "missing_baseline_qualified_semantic_coverage": missingBaselineQualifiedCoverage,
            "mean_text_delta": comparison["meanTextDelta"] ?? comparison["mean_text_delta"] ?? 0,
            "regression_severity": regressionSeverity,
            "semantic_coverage": recoveredCoverage,
            "missing_semantic_coverage": missingSemanticCoverage,
            "min_baseline_tokens": hasCompleteTokenCounts ? (baselineTokenCounts.min() ?? 0) : NSNull(),
            "min_masked_tokens": hasCompleteTokenCounts ? (maskedTokenCounts.min() ?? 0) : NSNull(),
            "mean_baseline_tokens": hasCompleteTokenCounts ? meanInts(baselineTokenCounts) : NSNull(),
            "mean_masked_tokens": hasCompleteTokenCounts ? meanInts(maskedTokenCounts) : NSNull(),
            "baseline_route_record_count": hasCompleteRouteCounts ? baselineRouteRecordCounts.reduce(0, +) : NSNull(),
            "masked_route_record_count": hasCompleteRouteCounts ? maskedRouteRecordCounts.reduce(0, +) : NSNull(),
            "generation_settings_checked": generationSettingsChecked,
            "eval_jsonl": "eval.jsonl",
            "eval_trace_jsonl": FileManager.default.fileExists(atPath: evalTraceURL.path) ? "eval_trace.jsonl" : NSNull(),
            "comparison_summary": "comparison_summary.json",
            "mask": "mask.json"
        ]
        if baselineLayerStatsPromptCount > 0 || maskedLayerStatsPromptCount > 0 {
            payload["baseline_layer_stats_prompt_count"] = baselineLayerStatsPromptCount
            payload["masked_layer_stats_prompt_count"] = maskedLayerStatsPromptCount
        }
        if let runtimeMode = stringValue(runtimeRecord?["runtimeMode"] ?? runtimeRecord?["runtime_mode"]) {
            payload["runtime_mode"] = runtimeMode
        }
        if let runtimeBackend = stringValue(runtimeRecord?["runtimeBackend"] ?? runtimeRecord?["runtime_backend"]) {
            payload["runtime_backend"] = runtimeBackend
        }
        if let runtimeDevice = stringValue(runtimeRecord?["runtimeDevice"] ?? runtimeRecord?["runtime_device"]) {
            payload["runtime_device"] = runtimeDevice
        }
        if let runtimeMetalEnabled = boolValue(runtimeRecord?["runtimeMetalEnabled"] ?? runtimeRecord?["runtime_metal_enabled"]) {
            payload["runtime_metal_enabled"] = runtimeMetalEnabled
        }
        if let jangToolsVersion = stringValue(runtimeRecord?["jangToolsVersion"] ?? runtimeRecord?["jang_tools_version"]) {
            payload["jang_tools_version"] = jangToolsVersion
        }
        if let mlxVersion = stringValue(runtimeRecord?["mlxVersion"] ?? runtimeRecord?["mlx_version"]) {
            payload["mlx_version"] = mlxVersion
        }
        if let mlxLMVersion = stringValue(runtimeRecord?["mlxLMVersion"] ?? runtimeRecord?["mlx_lm_version"]) {
            payload["mlx_lm_version"] = mlxLMVersion
        }
        if let mlxVLMVersion = stringValue(runtimeRecord?["mlxVLMVersion"] ?? runtimeRecord?["mlx_vlm_version"]) {
            payload["mlx_vlm_version"] = mlxVLMVersion
        }
        if let sourceModelPath = stringValue(runtimeRecord?["sourceModelPath"] ?? runtimeRecord?["source_model_path"]) {
            payload["source_model_path"] = sourceModelPath
        }
        if let hookedMOELayers = intValue(runtimeRecord?["hookedMOELayers"] ?? runtimeRecord?["hooked_moe_layers"]) {
            payload["hooked_moe_layers"] = hookedMOELayers
        }
        if let expectedMOELayers = intValue(runtimeRecord?["expectedMOELayers"] ?? runtimeRecord?["expected_moe_layers"]) {
            payload["expected_moe_layers"] = expectedMOELayers
        }
        if let hookCoverageComplete = boolValue(runtimeRecord?["hookCoverageComplete"] ?? runtimeRecord?["hook_coverage_complete"]) {
            payload["hook_coverage_complete"] = hookCoverageComplete
        }
        if let maskApplied = boolValue(runtimeRecord?["maskApplied"] ?? runtimeRecord?["mask_applied"]) {
            payload["mask_applied"] = maskApplied
        }
        if let disabledExpertCount = intValue(runtimeRecord?["disabledExpertCount"] ?? runtimeRecord?["disabled_expert_count"]) {
            payload["disabled_expert_count"] = disabledExpertCount
        }
        if let topKOverride = intValue(runtimeRecord?["topKOverride"] ?? runtimeRecord?["top_k_override"]) {
            payload["top_k_override"] = topKOverride
        }
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try data.write(to: destination)
        return true
    }

    private static func recoveredSemanticCoverage(from rows: [[String: Any]]) -> [String] {
        let domains = rows.flatMap { row -> [String] in
            let semanticDomains = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]) ?? []
            if !semanticDomains.isEmpty {
                return semanticDomains
            }
            if let domain = trimmedString(row["domain"]) {
                return [domain]
            }
            return []
        }
        return Array(Set(domains.map(ExpertDomainTaxonomy.canonicalSemanticDomain)))
            .filter { $0 != "general" }
            .sorted()
    }

    private static func recoveredGenerationSettingsChecked(
        rows: [[String: Any]],
        promptCount: Int
    ) -> Bool {
        guard promptCount > 0, rows.count == promptCount else { return false }
        for row in rows {
            let baselineValue = row["baselineGenerationSettings"] ?? row["baseline_generation_settings"]
            let maskedValue = row["maskedGenerationSettings"] ?? row["masked_generation_settings"]
            guard let baseline = decodeSettings(baselineValue),
                  let masked = decodeSettings(maskedValue),
                  baseline.maxTokens == masked.maxTokens,
                  abs(baseline.temperature - masked.temperature) <= 0.000_001,
                  abs(baseline.topP - masked.topP) <= 0.000_001,
                  baseline.topK == masked.topK else {
                return false
            }
        }
        return true
    }

    private static func readJSONObject(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }

    private static func fileSHA256(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func jsonlStringIDs(_ url: URL, keys: [String]) -> [String]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var ids: [String] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            if let id = keys.lazy.compactMap({ object[$0] as? String }).first {
                ids.append(id)
            }
        }
        return ids
    }

    private static func jsonlObjects(_ url: URL) -> [[String: Any]]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var rows: [[String: Any]] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            rows.append(object)
        }
        return rows
    }

    private static func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(whereSeparator: \.isNewline).count
    }

    private static func configLayerCount(modelPath: String) -> Int? {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let config = readJSONObject(configURL) else { return nil }
        let textConfig = config["text_config"] as? [String: Any] ?? config
        let layerCount = intValue(
            textConfig["num_hidden_layers"] ?? textConfig["n_layer"] ?? textConfig["num_layers"]
        )
        guard let layerCount, layerCount > 0 else { return nil }
        return layerCount
    }

    private static func suiteSemanticCoverageIssue(_ suiteURL: URL) -> String? {
        guard let suite = try? ExpertPromptSuite.loadJSONL(
            name: suiteURL.deletingPathExtension().lastPathComponent,
            from: suiteURL
        ) else {
            return "suite.jsonl semantic prompt coverage is unreadable"
        }
        let semanticDomains = Set(suite.prompts.flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
        let missing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.subtracting(semanticDomains).sorted()
        return missing.isEmpty
            ? nil
            : "suite.jsonl is missing required semantic prompt probes: \(missing.joined(separator: ", "))"
    }

    private static func evalIndexSemanticCoverageIssue(_ index: [String: Any]) -> String? {
        guard let semanticCoverage = stringArrayValue(index["semantic_coverage"] ?? index["semanticCoverage"]),
              !semanticCoverage.isEmpty else {
            return "eval_index.json is missing semantic coverage evidence"
        }
        let coverage = Set(
            semanticCoverage
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let missingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(coverage)
            .sorted()
        if !missingCoverage.isEmpty {
            return "eval_index.json semantic coverage is missing required probes: \(missingCoverage.joined(separator: ", "))"
        }
        guard let recordedMissing = stringArrayValue(index["missing_semantic_coverage"] ?? index["missingSemanticCoverage"]) else {
            return "eval_index.json is missing missing-semantic-coverage evidence"
        }
        let missing = Set(
            recordedMissing
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        if !missing.isEmpty {
            return "eval_index.json records missing semantic prompt probes: \(missing.sorted().joined(separator: ", "))"
        }
        return nil
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        case let value as String:
            return Int(value)
        default:
            return nil
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }

    private static func stringValue(_ value: Any?) -> String? {
        switch value {
        case let value as String:
            return value
        case let value as NSNumber:
            return value.stringValue
        default:
            return nil
        }
    }

    private static func arrayValue(_ value: Any?) -> [Any] {
        switch value {
        case let values as [Any]:
            return values
        default:
            return []
        }
    }

    private static func trimmedString(_ value: Any?) -> String? {
        stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func nonEmptyJSONArray(_ value: Any?) -> Bool {
        guard let array = value as? [Any] else { return false }
        return !array.isEmpty
    }

    private static func promptID(in row: [String: Any]) -> String? {
        ["promptID", "prompt_id", "id"].lazy.compactMap { key in
            trimmedString(row[key])
        }.first { !$0.isEmpty }
    }

    private static func normalizedTextDelta(_ lhs: String, _ rhs: String) -> Double {
        let maxLen = max(lhs.count, rhs.count, 1)
        let common = zip(lhs, rhs).filter { $0.0 == $0.1 }.count
        return Double(maxLen - common) / Double(maxLen)
    }

    private static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }

    private static func meanInts(_ values: [Int]) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.reduce(0, +)) / Double(values.count)
    }

    private static func maxRegressionSeverity(_ severities: [String]) -> String {
        severities.max { severityRank($0) < severityRank($1) } ?? "none"
    }

    private static func severityRank(_ severity: String) -> Int {
        switch severity {
        case "critical":
            return 3
        case "high":
            return 2
        case "watch":
            return 1
        default:
            return 0
        }
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            return value
        case let value as NSNumber:
            return value.boolValue
        case let value as String:
            return Bool(value)
        default:
            return nil
        }
    }

    private static func stringArrayValue(_ value: Any?) -> [String]? {
        switch value {
        case let values as [String]:
            return values
        case let values as [Any]:
            return values.compactMap { $0 as? String }
        default:
            return nil
        }
    }

    private static func recoveredHighRiskDomains(
        comparison: [String: Any],
        evalRows: [[String: Any]]
    ) -> [String] {
        var domains = Set((stringArrayValue(comparison["highRiskDomains"] ?? comparison["high_risk_domains"]) ?? [])
            .map(ExpertDomainTaxonomy.canonicalSemanticDomain))
        for row in evalRows where isHighRiskEvalRow(row) {
            let semanticDomains = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]) ?? []
            if semanticDomains.isEmpty {
                if let domain = trimmedString(row["domain"]) {
                    domains.insert(ExpertDomainTaxonomy.canonicalSemanticDomain(domain))
                }
            } else {
                domains.formUnion(semanticDomains.map(ExpertDomainTaxonomy.canonicalSemanticDomain))
            }
        }
        return domains.sorted()
    }

    private static func isHighRiskEvalRow(_ row: [String: Any]) -> Bool {
        let classification = promptClassification(row)
        if classification == "degraded" {
            return true
        }
        if ["baseline_invalid", "inconclusive", "preserved"].contains(classification) {
            return false
        }
        if let severity = trimmedString(row["regressionSeverity"] ?? row["regression_severity"]) {
            return severity == "high" || severity == "critical"
        }
        let risk = trimmedString(row["risk"])
        let textDelta = doubleValue(row["textDelta"] ?? row["text_delta"]) ?? 0
        let maskedPassed = boolValue(row["maskedPassed"] ?? row["masked_passed"])
        return risk == "regression" || maskedPassed == false || textDelta > 0.50
    }

    private static func previewIDs(_ ids: Set<String>) -> String {
        let sorted = ids.sorted()
        let head = sorted.prefix(5).joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func previewInts(_ values: Set<Int>) -> String {
        let sorted = values.sorted()
        let head = sorted.prefix(5).map { String($0) }.joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private func writeReport(summary: PrequantPruneSummary) throws -> URL {
        try FileManager.default.createDirectory(
            at: outputURL,
            withIntermediateDirectories: true,
            attributes: nil
        )
        let url = outputURL.appendingPathComponent("expert_lab_prune_report.md")
        let generated = ISO8601DateFormatter().string(from: Date())
        let plan = prunePlanURL?.path ?? "router-row fallback"
        let planSummaryLines: String
        if let prunePlanSummary {
            let comparisonLines: String
            if let comparison = prunePlanSummary.comparisonSummary {
                let riskDomains = comparison.highRiskDomains.isEmpty
                    ? "none"
                    : comparison.highRiskDomains.joined(separator: ", ")
                comparisonLines = """
                A/B comparison prompts: \(comparison.promptCount)
                Masked pass rate: \(comparison.maskedPassRateDescription)
                Baseline pass rate: \(comparison.baselinePassRateDescription)
                Baseline-qualified prompts: \(comparison.baselineQualifiedPromptCount.map(String.init) ?? "not recorded")
                Qualified masked pass rate: \(ImportedComparisonSummary.passRateDescription(comparison.baselineQualifiedMaskedPassRate))
                Prompt classifications: \(comparison.classificationCounts.map(ImportedComparisonSummary.classificationDescription) ?? "not recorded")
                Missing baseline-qualified coverage: \(comparison.missingBaselineQualifiedSemanticCoverage.isEmpty ? "none" : comparison.missingBaselineQualifiedSemanticCoverage.joined(separator: ", "))
                Degraded baseline-qualified prompts: \(comparison.degradedPromptIDs.isEmpty ? "none" : comparison.degradedPromptIDs.prefix(8).joined(separator: ", "))
                Mean text delta: \(comparison.meanTextDeltaDescription)
                Regression severity: \(comparison.regressionSeverity ?? "unknown")
                Mean latency delta: \(comparison.meanLatencyDeltaDescription)
                High-risk domains: \(riskDomains)
                A/B-safe candidates: \(comparison.safeDropCandidateCount)
                """
            } else {
                comparisonLines = "A/B comparison: not embedded in plan"
            }
            let preview = prunePlanSummary.evidencePreview.isEmpty
                ? "- no evidence preview"
                : prunePlanSummary.evidencePreview.map { "- \($0)" }.joined(separator: "\n")
            planSummaryLines = """
            Review run: \(prunePlanSummary.runID ?? "unknown")
            Atlas: \(prunePlanSummary.atlasID ?? "unknown")
            Prompt evidence: \(prunePlanSummary.promptCount)
            Layers: \(prunePlanSummary.layerCount)
            Locked keeps: \(prunePlanSummary.lockedKeepCount)
            User-forced drops: \(prunePlanSummary.userForcedDropCount)
            Evidence rows: \(prunePlanSummary.evidenceCount)
            \(comparisonLines)

            ## Reviewed Plan Evidence Preview

            \(preview)
            """
        } else {
            planSummaryLines = """
            Review run: router-row fallback
            Prompt evidence: 0
            """
        }
        let checks = summary.verification?.checks?.sorted(by: { $0.key < $1.key }) ?? []
        let checkLines = checks.map { item in "- \(item.key): \(item.value ? "passed" : "failed")" }.joined(separator: "\n")
        let requiredCheckLines = summary.verification?.requiredCheckRows
            .map { "- \($0.name): \($0.passed ? "passed" : "failed")" }
            .joined(separator: "\n") ?? "- not reported"
        let gateFailure = summary.verification?.strictFailureReason ?? "none"
        let errors = summary.verification?.errors ?? []
        let errorLines = errors.isEmpty
            ? "- none"
            : errors.map { "- \($0)" }.joined(separator: "\n")
        let sidecars = [
            "prune_plan.json",
            "prune_manifest.json",
            "expert_prune_manifest.json",
            "source_fingerprint.json",
            "verification.json",
            "expert_lab_review_summary.json",
            "expert_lab_suite.jsonl",
            "expert_lab_comparison_summary.json",
            "expert_lab_eval.jsonl",
            "expert_lab_eval_trace.jsonl",
            "expert_lab_eval_index.json",
            "expert_lab_pruned_generation_summary.json",
            "expert_lab_pruned_generations.jsonl"
        ].map { "- \($0)" }.joined(separator: "\n")
        let text = """
        # Expert Lab Prune Report

        Generated: \(generated)

        Source: \(sourceURL.path)
        Output: \(outputURL.path)
        Reviewed plan: \(plan)
        Method: \(summary.method)
        Source experts per layer: \(summary.sourceNumExperts)
        Kept experts per layer: \(summary.numExperts)
        Verification: \(summary.verificationStatus)

        ## Reviewed Plan Summary

        \(planSummaryLines)

        ## Verification Checks

        \(checkLines.isEmpty ? "- not reported" : checkLines)

        ## Required Verification Gate

        \(requiredCheckLines)

        Gate failure reason: \(gateFailure)

        ## Verification Errors

        \(errorLines)

        ## Sidecars

        \(sidecars)
        """
        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}

private struct PrequantPruneSummary: Decodable {
    let stage: String
    let sourceNumExperts: Int
    let numExperts: Int
    let method: String
    let verification: PrequantPruneVerification?

    enum CodingKeys: String, CodingKey {
        case stage
        case sourceNumExperts = "source_num_experts"
        case numExperts = "num_experts"
        case method
        case verification
    }

    var verificationStatus: String {
        guard let verification else { return "not reported" }
        return verification.isStrictlyVerified ? "passed" : "failed"
    }

    var isVerified: Bool {
        verification?.isStrictlyVerified == true
    }
}

private struct PrequantPruneVerification: Decodable {
    static let requiredChecks = [
        "config_parses",
        "index_parses",
        "index_covers_tensors",
        "router_rows_match",
        "expert_rows_match"
    ]

    let ok: Bool
    let checks: [String: Bool]?
    let errors: [String]?

    var isStrictlyVerified: Bool {
        ok && missingRequiredChecks.isEmpty && failedCheckNames.isEmpty
    }

    var requiredCheckRows: [PrequantVerificationCheckRow] {
        Self.requiredChecks.map { key in
            PrequantVerificationCheckRow(key: key, passed: checks?[key] == true)
        }
    }

    var strictFailureReason: String? {
        if !ok, let errors, !errors.isEmpty {
            return errors.prefix(3).joined(separator: "; ")
        }
        if !missingRequiredChecks.isEmpty {
            return "Missing required verification checks: \(missingRequiredChecks.joined(separator: ", "))"
        }
        if !failedCheckNames.isEmpty {
            return "Failed verification checks: \(failedCheckNames.joined(separator: ", "))"
        }
        if !ok {
            return "Pruned source verification did not pass"
        }
        return nil
    }

    private var missingRequiredChecks: [String] {
        guard let checks else { return Self.requiredChecks }
        return Self.requiredChecks.filter { checks[$0] == nil }
    }

    private var failedCheckNames: [String] {
        (checks ?? [:])
            .filter { !$0.value }
            .map(\.key)
            .sorted()
    }
}

private struct PrequantVerificationCheckRow: Identifiable {
    let key: String
    let passed: Bool

    var id: String { key }

    var name: String {
        switch key {
        case "config_parses": return "Config parses"
        case "index_parses": return "Index parses"
        case "index_covers_tensors": return "Index covers tensors"
        case "router_rows_match": return "Router rows match"
        case "expert_rows_match": return "Expert rows match"
        default: return key.replacingOccurrences(of: "_", with: " ")
        }
    }
}

private struct ImportedPrunePlan: Decodable {
    let method: String
    let keepExpertsPerLayer: Int?
    let runID: String?
    let atlasID: String?
    let promptCount: Int?
    let evalArtifact: String?
    let sourceModel: String?
    let reviewBundle: String?
    let comparisonSummary: ImportedComparisonSummary?
    let evalIndex: ImportedEvalIndexSummary?
    let safety: ImportedPrunePlanSafety?
    let layers: [String: ImportedPrunePlanLayer]

    enum CodingKeys: String, CodingKey {
        case method
        case keepExpertsPerLayer
        case runID = "run_id"
        case atlasID = "atlas_id"
        case promptCount
        case evalArtifact = "eval_artifact"
        case sourceModel = "source_model"
        case reviewBundle = "review_bundle"
        case comparisonSummary = "comparison_summary"
        case evalIndex = "eval_index"
        case safety
        case layers
    }
}

private struct ImportedPrunePlanLayer: Decodable {
    let keep: [Int]
    let drop: [Int]
    let lockedKeep: [Int]
    let userForcedDrop: [Int]
    let evidence: [ImportedPrunePlanEvidence]

    enum CodingKeys: String, CodingKey {
        case keep
        case drop
        case lockedKeep = "locked_keep"
        case userForcedDrop = "user_forced_drop"
        case evidence
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        keep = try c.decode([Int].self, forKey: .keep)
        drop = try c.decodeIfPresent([Int].self, forKey: .drop) ?? []
        lockedKeep = try c.decodeIfPresent([Int].self, forKey: .lockedKeep) ?? []
        userForcedDrop = try c.decodeIfPresent([Int].self, forKey: .userForcedDrop) ?? []
        evidence = try c.decodeIfPresent([ImportedPrunePlanEvidence].self, forKey: .evidence) ?? []
    }
}

private struct ImportedPrunePlanEvidence: Decodable {
    let expert: Int
    let hits: Int?
    let domains: [String: Int]?
    let label: String?
    let reason: String?
    let frequency: Double?
    let routerMass: Double?
    let ablationDelta: Double?
    let maskedImpactScope: String?
    let reviewedMaskMember: Bool?
    let domainLift: [String: Double]?
    let promptEvidence: [ImportedPromptEvidence]?
    let userNotes: String?
    let userForcedDrop: Bool?
    let kept: Bool?

    enum CodingKeys: String, CodingKey {
        case expert
        case hits
        case domains
        case label
        case reason
        case frequency
        case routerMass = "router_mass"
        case ablationDelta = "ablation_delta"
        case maskedImpactScope = "masked_impact_scope"
        case reviewedMaskMember = "reviewed_mask_member"
        case domainLift = "domain_lift"
        case promptEvidence = "prompt_evidence"
        case userNotes = "user_notes"
        case userForcedDrop = "user_forced_drop"
        case kept
    }

    var semanticProofDescription: String? {
        var parts: [String] = []
        if let lift = domainLift?
            .filter({ $0.value.isFinite })
            .sorted(by: { lhs, rhs in
                if lhs.value != rhs.value { return lhs.value > rhs.value }
                return lhs.key < rhs.key
            })
            .first {
            parts.append("lift \(lift.key) \(String(format: "%.4f", lift.value))")
        }
        if let prompt = promptEvidence?.first?.previewDescription {
            parts.append(prompt)
        }
        let note = userNotes?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !note.isEmpty {
            parts.append("note \(String(note.prefix(80)))")
        }
        if let scope = maskedImpactScope, !scope.isEmpty {
            parts.append("impact \(scope)")
        }
        if let reviewedMaskMember {
            parts.append(reviewedMaskMember ? "reviewed mask member" : "not in reviewed mask")
        }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }
}

private struct ImportedPromptEvidence: Decodable {
    let promptID: String?
    let domain: String?
    let subdomain: String?
    let tags: [String]?
    let promptExcerpt: String?
    let hits: Int?

    enum CodingKeys: String, CodingKey {
        case promptID
        case promptIDSnake = "prompt_id"
        case domain
        case subdomain
        case tags
        case promptExcerpt
        case promptExcerptSnake = "prompt_excerpt"
        case hits
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        promptID = try c.decodeIfPresent(String.self, forKey: .promptID)
            ?? c.decodeIfPresent(String.self, forKey: .promptIDSnake)
        domain = try c.decodeIfPresent(String.self, forKey: .domain)
        subdomain = try c.decodeIfPresent(String.self, forKey: .subdomain)
        tags = try c.decodeIfPresent([String].self, forKey: .tags)
        promptExcerpt = try c.decodeIfPresent(String.self, forKey: .promptExcerpt)
            ?? c.decodeIfPresent(String.self, forKey: .promptExcerptSnake)
        hits = try c.decodeIfPresent(Int.self, forKey: .hits)
    }

    var previewDescription: String? {
        let id = promptID?.isEmpty == false ? promptID! : nil
        let tagText = tags?.isEmpty == false ? " [\(tags!.joined(separator: ", "))]" : ""
        let excerpt = promptExcerpt.map { Self.clipped($0) }
        if let id, let excerpt {
            return "prompt \(id)\(tagText): \(excerpt)"
        }
        if let id {
            return "prompt \(id)\(tagText)"
        }
        return excerpt.map { "prompt example\(tagText): \($0)" }
    }

    private static func clipped(_ text: String, limit: Int = 80) -> String {
        let trimmed = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > limit else { return trimmed }
        let end = trimmed.index(trimmed.startIndex, offsetBy: limit)
        return String(trimmed[..<end]) + "..."
    }
}

private struct ImportedComparisonSummary: Decodable {
    let baselineRunID: String?
    let maskID: String?
    let promptCount: Int
    let passRateBaseline: Double?
    let passRateMasked: Double?
    let baselineQualifiedPromptCount: Int?
    let baselineQualifiedMaskedPassRate: Double?
    let validatorAvailablePromptCount: Int?
    let classificationCounts: [String: Int]?
    let missingBaselineQualifiedSemanticCoverage: [String]
    let degradedPromptIDs: [String]
    let meanTextDelta: Double
    let meanLatencyDeltaPct: Double
    let regressionSeverity: String?
    let highRiskDomains: [String]
    let safeDropCandidateCount: Int

    enum CodingKeys: String, CodingKey {
        case baselineRunID
        case baselineRunIDSnake = "baseline_run_id"
        case maskID
        case maskIDSnake = "mask_id"
        case promptCount
        case promptCountSnake = "prompt_count"
        case passRateBaseline
        case passRateBaselineSnake = "pass_rate_baseline"
        case passRateMasked
        case passRateMaskedSnake = "pass_rate_masked"
        case baselineQualifiedPromptCount
        case baselineQualifiedPromptCountSnake = "baseline_qualified_prompt_count"
        case baselineQualifiedMaskedPassRate
        case baselineQualifiedMaskedPassRateSnake = "baseline_qualified_masked_pass_rate"
        case validatorAvailablePromptCount
        case validatorAvailablePromptCountSnake = "validator_available_prompt_count"
        case classificationCounts
        case classificationCountsSnake = "prompt_classification_counts"
        case missingBaselineQualifiedSemanticCoverage
        case missingBaselineQualifiedSemanticCoverageSnake = "missing_baseline_qualified_semantic_coverage"
        case degradedPromptIDs
        case degradedPromptIDsSnake = "degraded_prompt_ids"
        case meanTextDelta
        case meanTextDeltaSnake = "mean_text_delta"
        case meanLatencyDeltaPct
        case meanLatencyDeltaPctSnake = "mean_latency_delta_pct"
        case regressionSeverity
        case regressionSeveritySnake = "regression_severity"
        case highRiskDomains
        case highRiskDomainsSnake = "high_risk_domains"
        case safeDropCandidates
        case safeDropCandidatesSnake = "safe_drop_candidates"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        baselineRunID = try c.decodeIfPresent(String.self, forKey: .baselineRunID)
            ?? c.decodeIfPresent(String.self, forKey: .baselineRunIDSnake)
        maskID = try c.decodeIfPresent(String.self, forKey: .maskID)
            ?? c.decodeIfPresent(String.self, forKey: .maskIDSnake)
        promptCount = try c.decodeIfPresent(Int.self, forKey: .promptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .promptCountSnake)
            ?? 0
        passRateBaseline = try c.decodeIfPresent(Double.self, forKey: .passRateBaseline)
            ?? c.decodeIfPresent(Double.self, forKey: .passRateBaselineSnake)
        passRateMasked = try c.decodeIfPresent(Double.self, forKey: .passRateMasked)
            ?? c.decodeIfPresent(Double.self, forKey: .passRateMaskedSnake)
        baselineQualifiedPromptCount = try c.decodeIfPresent(Int.self, forKey: .baselineQualifiedPromptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .baselineQualifiedPromptCountSnake)
        baselineQualifiedMaskedPassRate = try c.decodeIfPresent(Double.self, forKey: .baselineQualifiedMaskedPassRate)
            ?? c.decodeIfPresent(Double.self, forKey: .baselineQualifiedMaskedPassRateSnake)
        validatorAvailablePromptCount = try c.decodeIfPresent(Int.self, forKey: .validatorAvailablePromptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .validatorAvailablePromptCountSnake)
        classificationCounts = try c.decodeIfPresent([String: Int].self, forKey: .classificationCounts)
            ?? c.decodeIfPresent([String: Int].self, forKey: .classificationCountsSnake)
        missingBaselineQualifiedSemanticCoverage = try c.decodeIfPresent(
            [String].self,
            forKey: .missingBaselineQualifiedSemanticCoverage
        )
            ?? c.decodeIfPresent([String].self, forKey: .missingBaselineQualifiedSemanticCoverageSnake)
            ?? []
        degradedPromptIDs = try c.decodeIfPresent([String].self, forKey: .degradedPromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .degradedPromptIDsSnake)
            ?? []
        meanTextDelta = try c.decodeIfPresent(Double.self, forKey: .meanTextDelta)
            ?? c.decodeIfPresent(Double.self, forKey: .meanTextDeltaSnake)
            ?? 0
        meanLatencyDeltaPct = try c.decodeIfPresent(Double.self, forKey: .meanLatencyDeltaPct)
            ?? c.decodeIfPresent(Double.self, forKey: .meanLatencyDeltaPctSnake)
            ?? 0
        regressionSeverity = try c.decodeIfPresent(String.self, forKey: .regressionSeverity)
            ?? c.decodeIfPresent(String.self, forKey: .regressionSeveritySnake)
        highRiskDomains = try c.decodeIfPresent([String].self, forKey: .highRiskDomains)
            ?? c.decodeIfPresent([String].self, forKey: .highRiskDomainsSnake)
            ?? []
        let camelSafeDropCandidates = try c.decodeIfPresent(
            [ImportedComparisonCoordinate].self,
            forKey: .safeDropCandidates
        )
        let snakeSafeDropCandidates = try c.decodeIfPresent(
            [ImportedComparisonCoordinate].self,
            forKey: .safeDropCandidatesSnake
        )
        let safeDropCandidates = camelSafeDropCandidates ?? snakeSafeDropCandidates ?? []
        safeDropCandidateCount = safeDropCandidates.count
    }

    var baselinePassRateDescription: String {
        Self.passRateDescription(passRateBaseline)
    }

    var maskedPassRateDescription: String {
        Self.passRateDescription(passRateMasked)
    }

    var meanTextDeltaDescription: String {
        String(format: "%.4f", meanTextDelta)
    }

    var meanLatencyDeltaDescription: String {
        String(format: "%.2f%%", meanLatencyDeltaPct)
    }

    static func passRateDescription(_ value: Double?) -> String {
        guard let value else { return "unscored" }
        return String(format: "%.0f%%", value * 100)
    }

    static func classificationDescription(_ counts: [String: Int]) -> String {
        ["preserved", "degraded", "baseline_invalid", "inconclusive"]
            .compactMap { key -> String? in
                guard let value = counts[key], value > 0 else { return nil }
                return "\(key) \(value)"
            }
            .joined(separator: ", ")
    }
}

private struct ImportedComparisonCoordinate: Decodable {
    let layer: Int
    let expert: Int
}

private struct ImportedPrunePlanSafety: Decodable {
    let passed: Bool
    let minimumActiveExpertsPerLayer: Int?
    let trainedTopKByLayer: [String: Int]
    let trainedTopK: Int?
    let issues: [String]

    enum CodingKeys: String, CodingKey {
        case passed
        case minimumActiveExpertsPerLayer
        case minimumActiveExpertsPerLayerSnake = "minimum_active_experts_per_layer"
        case trainedTopKByLayer
        case trainedTopKByLayerSnake = "trained_top_k_by_layer"
        case trainedTopK
        case trainedTopKSnake = "trained_top_k"
        case issues
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        passed = try c.decodeIfPresent(Bool.self, forKey: .passed) ?? false
        minimumActiveExpertsPerLayer = try c.decodeIfPresent(Int.self, forKey: .minimumActiveExpertsPerLayer)
            ?? c.decodeIfPresent(Int.self, forKey: .minimumActiveExpertsPerLayerSnake)
        trainedTopKByLayer = try c.decodeIfPresent([String: Int].self, forKey: .trainedTopKByLayer)
            ?? c.decodeIfPresent([String: Int].self, forKey: .trainedTopKByLayerSnake)
            ?? [:]
        trainedTopK = try c.decodeIfPresent(Int.self, forKey: .trainedTopK)
            ?? c.decodeIfPresent(Int.self, forKey: .trainedTopKSnake)
        issues = try c.decodeIfPresent([String].self, forKey: .issues) ?? []
    }

    var maxTrainedTopK: Int? {
        if let trainedTopK { return trainedTopK }
        return trainedTopKByLayer.values.max()
    }
}

private struct ImportedEvalIndexSummary: Decodable {
    let promptCount: Int
    let promptIDs: [String]
    let riskyPromptIDs: [String]
    let highRiskDomains: [String]
    let regressionSeverity: String?
    let meanBaselineTokens: Double?
    let meanMaskedTokens: Double?
    let baselineRouteRecordCount: Int?
    let maskedRouteRecordCount: Int?
    let baselineLayerStatsPromptCount: Int?
    let maskedLayerStatsPromptCount: Int?
    let generationSettingsChecked: Bool?
    let suiteJSONL: String?
    let suiteSHA256: String?
    let evalJSONL: String?
    let evalTraceJSONL: String?
    let comparisonSummary: String?
    let semanticCoverage: [String]
    let missingSemanticCoverage: [String]?
    let validatorSchema: String?
    let validatorAvailablePromptCount: Int?
    let promptClassificationCounts: [String: Int]?
    let baselineQualifiedPromptCount: Int?
    let baselineQualifiedPromptIDs: [String]
    let baselineInvalidPromptIDs: [String]
    let inconclusivePromptIDs: [String]
    let preservedPromptIDs: [String]
    let degradedPromptIDs: [String]
    let baselineQualifiedMaskedPassRate: Double?
    let baselineQualifiedSemanticCoverage: [String]
    let missingBaselineQualifiedSemanticCoverage: [String]
    let mask: String?
    let maskJSON: String?
    let runtimeMode: String?
    let runtimeBackend: String?
    let runtimeDevice: String?
    let runtimeMetalEnabled: Bool?
    let jangToolsVersion: String?
    let mlxVersion: String?
    let mlxLMVersion: String?
    let sourceModelPath: String?
    let hookedMOELayers: Int?
    let expectedMOELayers: Int?
    let hookCoverageComplete: Bool?
    let maskApplied: Bool?
    let disabledExpertCount: Int?
    let topKOverride: Int?

    enum CodingKeys: String, CodingKey {
        case promptCount
        case promptCountSnake = "prompt_count"
        case promptIDs
        case promptIDsSnake = "prompt_ids"
        case riskyPromptIDs
        case riskyPromptIDsSnake = "risky_prompt_ids"
        case highRiskDomains
        case highRiskDomainsSnake = "high_risk_domains"
        case regressionSeverity
        case regressionSeveritySnake = "regression_severity"
        case meanBaselineTokens
        case meanBaselineTokensSnake = "mean_baseline_tokens"
        case meanMaskedTokens
        case meanMaskedTokensSnake = "mean_masked_tokens"
        case baselineRouteRecordCount
        case baselineRouteRecordCountSnake = "baseline_route_record_count"
        case maskedRouteRecordCount
        case maskedRouteRecordCountSnake = "masked_route_record_count"
        case baselineLayerStatsPromptCount
        case baselineLayerStatsPromptCountSnake = "baseline_layer_stats_prompt_count"
        case maskedLayerStatsPromptCount
        case maskedLayerStatsPromptCountSnake = "masked_layer_stats_prompt_count"
        case generationSettingsChecked
        case generationSettingsCheckedSnake = "generation_settings_checked"
        case suiteJSONL
        case suiteJSONLSnake = "suite_jsonl"
        case suiteSHA256
        case suiteSHA256Snake = "suite_sha256"
        case evalJSONL
        case evalJSONLSnake = "eval_jsonl"
        case evalTraceJSONL
        case evalTraceJSONLSnake = "eval_trace_jsonl"
        case comparisonSummary
        case comparisonSummarySnake = "comparison_summary"
        case semanticCoverage
        case semanticCoverageSnake = "semantic_coverage"
        case missingSemanticCoverage
        case missingSemanticCoverageSnake = "missing_semantic_coverage"
        case validatorSchema
        case validatorSchemaSnake = "validator_schema"
        case validatorAvailablePromptCount
        case validatorAvailablePromptCountSnake = "validator_available_prompt_count"
        case promptClassificationCounts
        case promptClassificationCountsSnake = "prompt_classification_counts"
        case baselineQualifiedPromptCount
        case baselineQualifiedPromptCountSnake = "baseline_qualified_prompt_count"
        case baselineQualifiedPromptIDs
        case baselineQualifiedPromptIDsSnake = "baseline_qualified_prompt_ids"
        case baselineInvalidPromptIDs
        case baselineInvalidPromptIDsSnake = "baseline_invalid_prompt_ids"
        case inconclusivePromptIDs
        case inconclusivePromptIDsSnake = "inconclusive_prompt_ids"
        case preservedPromptIDs
        case preservedPromptIDsSnake = "preserved_prompt_ids"
        case degradedPromptIDs
        case degradedPromptIDsSnake = "degraded_prompt_ids"
        case baselineQualifiedMaskedPassRate
        case baselineQualifiedMaskedPassRateSnake = "baseline_qualified_masked_pass_rate"
        case baselineQualifiedSemanticCoverage
        case baselineQualifiedSemanticCoverageSnake = "baseline_qualified_semantic_coverage"
        case missingBaselineQualifiedSemanticCoverage
        case missingBaselineQualifiedSemanticCoverageSnake = "missing_baseline_qualified_semantic_coverage"
        case mask
        case maskJSON
        case maskJSONSnake = "mask_json"
        case runtimeMode
        case runtimeModeSnake = "runtime_mode"
        case runtimeBackend
        case runtimeBackendSnake = "runtime_backend"
        case runtimeDevice
        case runtimeDeviceSnake = "runtime_device"
        case runtimeMetalEnabled
        case runtimeMetalEnabledSnake = "runtime_metal_enabled"
        case jangToolsVersion
        case jangToolsVersionSnake = "jang_tools_version"
        case mlxVersion
        case mlxVersionSnake = "mlx_version"
        case mlxLMVersion
        case mlxLMVersionSnake = "mlx_lm_version"
        case sourceModelPath
        case sourceModelPathSnake = "source_model_path"
        case hookedMOELayers
        case hookedMOELayersSnake = "hooked_moe_layers"
        case expectedMOELayers
        case expectedMOELayersSnake = "expected_moe_layers"
        case hookCoverageComplete
        case hookCoverageCompleteSnake = "hook_coverage_complete"
        case maskApplied
        case maskAppliedSnake = "mask_applied"
        case disabledExpertCount
        case disabledExpertCountSnake = "disabled_expert_count"
        case topKOverride
        case topKOverrideSnake = "top_k_override"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        promptCount = try c.decodeIfPresent(Int.self, forKey: .promptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .promptCountSnake)
            ?? 0
        promptIDs = try c.decodeIfPresent([String].self, forKey: .promptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .promptIDsSnake)
            ?? []
        riskyPromptIDs = try c.decodeIfPresent([String].self, forKey: .riskyPromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .riskyPromptIDsSnake)
            ?? []
        highRiskDomains = try c.decodeIfPresent([String].self, forKey: .highRiskDomains)
            ?? c.decodeIfPresent([String].self, forKey: .highRiskDomainsSnake)
            ?? []
        regressionSeverity = try c.decodeIfPresent(String.self, forKey: .regressionSeverity)
            ?? c.decodeIfPresent(String.self, forKey: .regressionSeveritySnake)
        meanBaselineTokens = try c.decodeIfPresent(Double.self, forKey: .meanBaselineTokens)
            ?? c.decodeIfPresent(Double.self, forKey: .meanBaselineTokensSnake)
        meanMaskedTokens = try c.decodeIfPresent(Double.self, forKey: .meanMaskedTokens)
            ?? c.decodeIfPresent(Double.self, forKey: .meanMaskedTokensSnake)
        baselineRouteRecordCount = try c.decodeIfPresent(Int.self, forKey: .baselineRouteRecordCount)
            ?? c.decodeIfPresent(Int.self, forKey: .baselineRouteRecordCountSnake)
        maskedRouteRecordCount = try c.decodeIfPresent(Int.self, forKey: .maskedRouteRecordCount)
            ?? c.decodeIfPresent(Int.self, forKey: .maskedRouteRecordCountSnake)
        baselineLayerStatsPromptCount = try c.decodeIfPresent(Int.self, forKey: .baselineLayerStatsPromptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .baselineLayerStatsPromptCountSnake)
        maskedLayerStatsPromptCount = try c.decodeIfPresent(Int.self, forKey: .maskedLayerStatsPromptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .maskedLayerStatsPromptCountSnake)
        generationSettingsChecked = try c.decodeIfPresent(Bool.self, forKey: .generationSettingsChecked)
            ?? c.decodeIfPresent(Bool.self, forKey: .generationSettingsCheckedSnake)
        suiteJSONL = try c.decodeIfPresent(String.self, forKey: .suiteJSONL)
            ?? c.decodeIfPresent(String.self, forKey: .suiteJSONLSnake)
        suiteSHA256 = try c.decodeIfPresent(String.self, forKey: .suiteSHA256)
            ?? c.decodeIfPresent(String.self, forKey: .suiteSHA256Snake)
        evalJSONL = try c.decodeIfPresent(String.self, forKey: .evalJSONL)
            ?? c.decodeIfPresent(String.self, forKey: .evalJSONLSnake)
        evalTraceJSONL = try c.decodeIfPresent(String.self, forKey: .evalTraceJSONL)
            ?? c.decodeIfPresent(String.self, forKey: .evalTraceJSONLSnake)
        comparisonSummary = try c.decodeIfPresent(String.self, forKey: .comparisonSummary)
            ?? c.decodeIfPresent(String.self, forKey: .comparisonSummarySnake)
        semanticCoverage = try c.decodeIfPresent([String].self, forKey: .semanticCoverage)
            ?? c.decodeIfPresent([String].self, forKey: .semanticCoverageSnake)
            ?? []
        missingSemanticCoverage = try c.decodeIfPresent([String].self, forKey: .missingSemanticCoverage)
            ?? c.decodeIfPresent([String].self, forKey: .missingSemanticCoverageSnake)
        validatorSchema = try c.decodeIfPresent(String.self, forKey: .validatorSchema)
            ?? c.decodeIfPresent(String.self, forKey: .validatorSchemaSnake)
        validatorAvailablePromptCount = try c.decodeIfPresent(Int.self, forKey: .validatorAvailablePromptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .validatorAvailablePromptCountSnake)
        promptClassificationCounts = try c.decodeIfPresent([String: Int].self, forKey: .promptClassificationCounts)
            ?? c.decodeIfPresent([String: Int].self, forKey: .promptClassificationCountsSnake)
        baselineQualifiedPromptCount = try c.decodeIfPresent(Int.self, forKey: .baselineQualifiedPromptCount)
            ?? c.decodeIfPresent(Int.self, forKey: .baselineQualifiedPromptCountSnake)
        baselineQualifiedPromptIDs = try c.decodeIfPresent([String].self, forKey: .baselineQualifiedPromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .baselineQualifiedPromptIDsSnake)
            ?? []
        baselineInvalidPromptIDs = try c.decodeIfPresent([String].self, forKey: .baselineInvalidPromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .baselineInvalidPromptIDsSnake)
            ?? []
        inconclusivePromptIDs = try c.decodeIfPresent([String].self, forKey: .inconclusivePromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .inconclusivePromptIDsSnake)
            ?? []
        preservedPromptIDs = try c.decodeIfPresent([String].self, forKey: .preservedPromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .preservedPromptIDsSnake)
            ?? []
        degradedPromptIDs = try c.decodeIfPresent([String].self, forKey: .degradedPromptIDs)
            ?? c.decodeIfPresent([String].self, forKey: .degradedPromptIDsSnake)
            ?? []
        baselineQualifiedMaskedPassRate = try c.decodeIfPresent(Double.self, forKey: .baselineQualifiedMaskedPassRate)
            ?? c.decodeIfPresent(Double.self, forKey: .baselineQualifiedMaskedPassRateSnake)
        baselineQualifiedSemanticCoverage = try c.decodeIfPresent(
            [String].self,
            forKey: .baselineQualifiedSemanticCoverage
        )
            ?? c.decodeIfPresent([String].self, forKey: .baselineQualifiedSemanticCoverageSnake)
            ?? []
        missingBaselineQualifiedSemanticCoverage = try c.decodeIfPresent(
            [String].self,
            forKey: .missingBaselineQualifiedSemanticCoverage
        )
            ?? c.decodeIfPresent([String].self, forKey: .missingBaselineQualifiedSemanticCoverageSnake)
            ?? []
        mask = try c.decodeIfPresent(String.self, forKey: .mask)
        maskJSON = try c.decodeIfPresent(String.self, forKey: .maskJSON)
            ?? c.decodeIfPresent(String.self, forKey: .maskJSONSnake)
        runtimeMode = try c.decodeIfPresent(String.self, forKey: .runtimeMode)
            ?? c.decodeIfPresent(String.self, forKey: .runtimeModeSnake)
        runtimeBackend = try c.decodeIfPresent(String.self, forKey: .runtimeBackend)
            ?? c.decodeIfPresent(String.self, forKey: .runtimeBackendSnake)
        runtimeDevice = try c.decodeIfPresent(String.self, forKey: .runtimeDevice)
            ?? c.decodeIfPresent(String.self, forKey: .runtimeDeviceSnake)
        runtimeMetalEnabled = try c.decodeIfPresent(Bool.self, forKey: .runtimeMetalEnabled)
            ?? c.decodeIfPresent(Bool.self, forKey: .runtimeMetalEnabledSnake)
        jangToolsVersion = try c.decodeIfPresent(String.self, forKey: .jangToolsVersion)
            ?? c.decodeIfPresent(String.self, forKey: .jangToolsVersionSnake)
        mlxVersion = try c.decodeIfPresent(String.self, forKey: .mlxVersion)
            ?? c.decodeIfPresent(String.self, forKey: .mlxVersionSnake)
        mlxLMVersion = try c.decodeIfPresent(String.self, forKey: .mlxLMVersion)
            ?? c.decodeIfPresent(String.self, forKey: .mlxLMVersionSnake)
        sourceModelPath = try c.decodeIfPresent(String.self, forKey: .sourceModelPath)
            ?? c.decodeIfPresent(String.self, forKey: .sourceModelPathSnake)
        hookedMOELayers = try c.decodeIfPresent(Int.self, forKey: .hookedMOELayers)
            ?? c.decodeIfPresent(Int.self, forKey: .hookedMOELayersSnake)
        expectedMOELayers = try c.decodeIfPresent(Int.self, forKey: .expectedMOELayers)
            ?? c.decodeIfPresent(Int.self, forKey: .expectedMOELayersSnake)
        hookCoverageComplete = try c.decodeIfPresent(Bool.self, forKey: .hookCoverageComplete)
            ?? c.decodeIfPresent(Bool.self, forKey: .hookCoverageCompleteSnake)
        maskApplied = try c.decodeIfPresent(Bool.self, forKey: .maskApplied)
            ?? c.decodeIfPresent(Bool.self, forKey: .maskAppliedSnake)
        disabledExpertCount = try c.decodeIfPresent(Int.self, forKey: .disabledExpertCount)
            ?? c.decodeIfPresent(Int.self, forKey: .disabledExpertCountSnake)
        topKOverride = try c.decodeIfPresent(Int.self, forKey: .topKOverride)
            ?? c.decodeIfPresent(Int.self, forKey: .topKOverrideSnake)
    }
}

private struct IntentPrunePlanMeta {
    let schema: String?
    let scorer: String?
    let safetyStance: String?
    let intentsKeep: [String]
    let suiteName: String?
    let suiteSHA256: String?
    let suitePromptCount: Int?
    let crackPackName: String?
    let crackPackSHA256: String?
    let crackPackPromptCount: Int?

    init(from plan: [String: Any]) {
        schema = plan["schema"] as? String
        scorer = plan["scorer"] as? String
        safetyStance = (plan["safety_stance"] as? String) ?? (plan["safetyStance"] as? String)
        intentsKeep = (plan["intents_keep"] as? [String])
            ?? (plan["intentsKeep"] as? [String])
            ?? []
        let suite = plan["suite"] as? [String: Any]
        suiteName = suite?["name"] as? String
        suiteSHA256 = suite?["sha256"] as? String
        suitePromptCount = (suite?["prompt_count"] as? Int)
            ?? (suite?["promptCount"] as? Int)
            ?? (suite?["prompt_count"] as? NSNumber)?.intValue
        let crack = plan["crack_pack"] as? [String: Any]
        crackPackName = crack?["name"] as? String
        crackPackSHA256 = crack?["sha256"] as? String
        crackPackPromptCount = (crack?["prompt_count"] as? Int)
            ?? (crack?["promptCount"] as? Int)
            ?? (crack?["prompt_count"] as? NSNumber)?.intValue
    }
}

private struct ImportedPrunePlanSummary {
    private static let minimumReviewedPrunePromptCount = 50
    private static let minimumReviewedPruneMeanTokens: Double = 8

    let method: String
    let keepExperts: Int
    let layerCount: Int
    let runID: String?
    let atlasID: String?
    let evalArtifactPath: String?
    let promptCount: Int
    let lockedKeepCount: Int
    let userForcedDropCount: Int
    let evidenceCount: Int
    let comparisonSummary: ImportedComparisonSummary?
    let safetyDescription: String
    let evidencePreview: [String]
    let intentMeta: IntentPrunePlanMeta?

    init(
        plan: ImportedPrunePlan,
        sourceURL: URL,
        sourceExperts: Int,
        skipSemanticEvidence: Bool = false,
        intentMeta: IntentPrunePlanMeta? = nil
    ) throws {
        guard !plan.layers.isEmpty else {
            throw ImportedPrunePlanError.emptyLayers
        }
        if let planSource = plan.sourceModel, !planSource.isEmpty {
            let selectedSource = sourceURL.standardizedFileURL.resolvingSymlinksInPath().path
            let embeddedSource = URL(fileURLWithPath: planSource).standardizedFileURL.resolvingSymlinksInPath().path
            guard selectedSource == embeddedSource else {
                throw ImportedPrunePlanError.sourceMismatch(planSource: embeddedSource, selectedSource: selectedSource)
            }
        }
        let keepCounts = Set(plan.layers.values.map { $0.keep.count })
        guard keepCounts.count == 1, let keepExperts = keepCounts.first else {
            throw ImportedPrunePlanError.mixedKeepCounts
        }
        if let declared = plan.keepExpertsPerLayer, declared != keepExperts {
            throw ImportedPrunePlanError.declaredKeepMismatch(declared: declared, actual: keepExperts)
        }
        guard keepExperts > 0 else {
            throw ImportedPrunePlanError.emptyKeep
        }
        guard keepExperts < sourceExperts else {
            throw ImportedPrunePlanError.keepTooHigh(keep: keepExperts, sourceExperts: sourceExperts)
        }
        guard let safety = plan.safety else {
            throw ImportedPrunePlanError.safetyRejected("plan is missing top-k safety evidence.")
        }
        if !safety.passed {
            throw ImportedPrunePlanError.safetyRejected("embedded plan safety did not pass.")
        }
        if !safety.issues.isEmpty {
            throw ImportedPrunePlanError.safetyRejected(safety.issues.joined(separator: " "))
        }
        guard let minimumActive = safety.minimumActiveExpertsPerLayer else {
            throw ImportedPrunePlanError.safetyRejected("safety block is missing minimum active experts.")
        }
        if minimumActive != keepExperts {
            throw ImportedPrunePlanError.safetyRejected(
                "safety declares \(minimumActive) minimum active experts but plan keeps \(keepExperts)."
            )
        }
        guard let trainedTopK = safety.maxTrainedTopK else {
            throw ImportedPrunePlanError.safetyRejected("safety block is missing trained top-k evidence.")
        }
        if keepExperts < trainedTopK {
            throw ImportedPrunePlanError.safetyRejected(
                "plan keeps \(keepExperts) experts but embedded trained top-k is \(trainedTopK)."
            )
        }
        guard let comparison = plan.comparisonSummary else {
            throw ImportedPrunePlanError.missingComparison
        }
        if let issue = Self.comparisonGateIssue(
            comparison: comparison,
            evalIndex: plan.evalIndex,
            sourceModelPath: sourceURL.path,
            tracedPromptCount: plan.promptCount ?? 0,
            expectedLayerCount: plan.layers.count
        ) {
            throw ImportedPrunePlanError.comparisonRejected(issue)
        }
        if !skipSemanticEvidence {
            if let issue = Self.semanticEvidenceIssue(plan.layers) {
                throw ImportedPrunePlanError.semanticEvidenceRejected(issue)
            }
        }
        self.method = plan.method
        self.keepExperts = keepExperts
        self.layerCount = plan.layers.count
        self.runID = plan.runID
        self.atlasID = plan.atlasID
        self.evalArtifactPath = plan.evalArtifact
        self.promptCount = plan.promptCount ?? 0
        self.lockedKeepCount = plan.layers.values.reduce(0) { $0 + $1.lockedKeep.count }
        self.userForcedDropCount = plan.layers.values.reduce(0) { $0 + $1.userForcedDrop.count }
        self.evidenceCount = plan.layers.values.reduce(0) { $0 + $1.evidence.count }
        self.comparisonSummary = comparison
        self.intentMeta = intentMeta
        if let intentMeta, intentMeta.schema == "jang-intent-prune-plan-v1" || intentMeta.scorer != nil {
            var parts = ["passed; min active \(minimumActive), trained top-k \(trainedTopK)"]
            if let stance = intentMeta.safetyStance, !stance.isEmpty {
                parts.append("stance \(stance)")
            }
            if !intentMeta.intentsKeep.isEmpty {
                parts.append("intents \(intentMeta.intentsKeep.joined(separator: ","))")
            }
            if let crack = intentMeta.crackPackName {
                parts.append("CRACK \(crack)")
            }
            self.safetyDescription = parts.joined(separator: "; ")
        } else {
            self.safetyDescription = "passed; min active \(minimumActive), trained top-k \(trainedTopK)"
        }
        let layerPreviewLines = plan.layers.keys
            .sorted { (Int($0) ?? Int.max, $0) < (Int($1) ?? Int.max, $1) }
            .flatMap { layer -> [String] in
                guard let layerPlan = plan.layers[layer] else { return [] }
                return Array(layerPlan.evidence.filter { $0.kept != true }.prefix(2))
                    .map { evidence -> String in
                        let label = evidence.label ?? "unlabeled"
                        let forced = evidence.userForcedDrop == true ? "user-forced; " : ""
                        let reason = evidence.reason?.isEmpty == false ? " — \(evidence.reason!)" : ""
                        let frequency = evidence.frequency.map { String(format: "%.4f", $0) } ?? "n/a"
                        let routerMass = evidence.routerMass.map { String(format: "%.4f", $0) } ?? "n/a"
                        let ablation = evidence.ablationDelta.map {
                            ", A/B delta " + String(format: "%.4f", $0)
                        } ?? ""
                        let proof = evidence.semanticProofDescription.map { ", \($0)" } ?? ""
                        return "L\(layer) E\(evidence.expert): \(label) (\(forced)freq \(frequency), mass \(routerMass)\(ablation)\(proof))\(reason)"
                    }
            }
        self.evidencePreview = Array(layerPreviewLines.prefix(6))
    }

    private static func semanticEvidenceIssue(_ layers: [String: ImportedPrunePlanLayer]) -> String? {
        var checkedRows = 0
        for layer in layers.keys.sorted(by: { (Int($0) ?? Int.max, $0) < (Int($1) ?? Int.max, $1) }) {
            guard let layerPlan = layers[layer] else { continue }
            guard !layerPlan.evidence.isEmpty else {
                return "layer \(layer) is missing expert evidence rows."
            }
            for evidence in layerPlan.evidence {
                let label = evidence.label?.lowercased() ?? ""
                let isUnobserved = label.contains("unobserved")
                    && (evidence.hits ?? 0) == 0
                    && (evidence.domains?.isEmpty ?? true)
                if isUnobserved { continue }

                checkedRows += 1
                let coordinate = "L\(layer) E\(evidence.expert)"
                guard evidence.routerMass != nil else {
                    return "\(coordinate) is missing gate mass evidence."
                }
                guard evidence.ablationDelta != nil else {
                    return "\(coordinate) is missing masked-output impact evidence."
                }
                guard evidence.maskedImpactScope?.isEmpty == false else {
                    return "\(coordinate) is missing masked-output impact scope evidence."
                }
                guard evidence.reviewedMaskMember != nil else {
                    return "\(coordinate) is missing reviewed mask membership evidence."
                }
                guard let domainLift = evidence.domainLift,
                      domainLift.contains(where: { $0.value.isFinite }) else {
                    return "\(coordinate) is missing activation lift evidence."
                }
                guard let promptEvidence = evidence.promptEvidence, !promptEvidence.isEmpty else {
                    return "\(coordinate) is missing prompt example evidence."
                }
                let hasPromptProof = promptEvidence.contains { prompt in
                    prompt.promptID?.isEmpty == false &&
                    prompt.domain?.isEmpty == false &&
                    prompt.promptExcerpt?.isEmpty == false &&
                    prompt.tags?.isEmpty == false &&
                    (prompt.hits ?? 0) > 0
                }
                if !hasPromptProof {
                    return "\(coordinate) has incomplete prompt tags/examples."
                }
            }
        }
        return checkedRows == 0 ? "plan has no semantic expert evidence rows." : nil
    }

    private static func comparisonGateIssue(
        comparison: ImportedComparisonSummary,
        evalIndex: ImportedEvalIndexSummary?,
        sourceModelPath: String?,
        tracedPromptCount: Int,
        expectedLayerCount: Int
    ) -> String? {
        if comparison.promptCount < minimumReviewedPrunePromptCount {
            return "compare at least \(minimumReviewedPrunePromptCount) prompts before BF16/F16 prune."
        }
        if tracedPromptCount > 0, comparison.promptCount != tracedPromptCount {
            return "rerun A/B compare for all \(tracedPromptCount) traced prompts before BF16/F16 prune."
        }
        guard comparison.validatorAvailablePromptCount != nil,
              comparison.classificationCounts != nil else {
            return "comparison summary is missing validator classification evidence."
        }
        guard let baselineQualified = comparison.baselineQualifiedPromptCount,
              baselineQualified > 0 else {
            return "comparison summary has no baseline-qualified validator prompts."
        }
        if !comparison.missingBaselineQualifiedSemanticCoverage.isEmpty {
            return "baseline-qualified prompts are missing semantic coverage: \(comparison.missingBaselineQualifiedSemanticCoverage.sorted().joined(separator: ", "))."
        }
        if !comparison.degradedPromptIDs.isEmpty {
            return "baseline-qualified prompts degraded after masking: \(comparison.degradedPromptIDs.prefix(5).joined(separator: ", "))."
        }
        if let passRate = comparison.baselineQualifiedMaskedPassRate,
           passRate < 1.0 {
            return "masked validator pass rate is below 100% on baseline-qualified prompts."
        }
        if !comparison.highRiskDomains.isEmpty {
            return "masked outputs regressed in high-risk domains: \(comparison.highRiskDomains.sorted().joined(separator: ", "))."
        }
        if isBlockingRegressionSeverity(comparison.regressionSeverity) {
            return "masked comparison severity \(comparison.regressionSeverity ?? "unknown") must be reviewed before BF16/F16 prune."
        }
        if comparison.safeDropCandidateCount == 0 {
            return "A/B comparison found no safe drop candidates."
        }
        guard let evalIndex else {
            return "plan is missing per-prompt eval_index evidence."
        }
        guard evalIndex.suiteJSONL?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            return "eval_index is missing suite.jsonl evidence."
        }
        guard evalIndex.suiteSHA256?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            return "eval_index is missing suite.jsonl fingerprint evidence."
        }
        if let issue = evalIndexSemanticCoverageIssue(evalIndex) {
            return issue
        }
        if let issue = evalIndexValidatorEvidenceIssue(evalIndex) {
            return issue
        }
        if evalIndex.promptIDs.count != evalIndex.promptCount {
            return "eval_index lists \(evalIndex.promptIDs.count) prompt IDs for \(evalIndex.promptCount) indexed prompts."
        }
        if Set(evalIndex.promptIDs).count < evalIndex.promptIDs.count {
            return "eval_index contains duplicate prompt IDs."
        }
        if evalIndex.promptCount != comparison.promptCount {
            return "eval_index covers \(evalIndex.promptCount) of \(comparison.promptCount) compared prompts."
        }
        if tracedPromptCount > 0, evalIndex.promptCount != tracedPromptCount {
            return "eval_index covers \(evalIndex.promptCount) of \(tracedPromptCount) traced prompts."
        }
        if !evalIndex.riskyPromptIDs.isEmpty {
            return "eval_index still has risky prompt IDs."
        }
        if isBlockingRegressionSeverity(evalIndex.regressionSeverity) {
            return "eval_index regression severity \(evalIndex.regressionSeverity ?? "unknown") must be reviewed before BF16/F16 prune."
        }
        if !evalIndex.highRiskDomains.isEmpty {
            return "eval_index still has high-risk domains: \(evalIndex.highRiskDomains.sorted().joined(separator: ", "))."
        }
        guard let meanBaselineTokens = evalIndex.meanBaselineTokens,
              let meanMaskedTokens = evalIndex.meanMaskedTokens else {
            return "eval_index is missing generation-depth token evidence."
        }
        let shallow = min(meanBaselineTokens, meanMaskedTokens)
        if shallow < minimumReviewedPruneMeanTokens {
            return String(
                format: "eval_index average generated depth %.1f tokens is below %.0f.",
                shallow,
                minimumReviewedPruneMeanTokens
            )
        }
        guard let baselineRouteRecordCount = evalIndex.baselineRouteRecordCount,
              let maskedRouteRecordCount = evalIndex.maskedRouteRecordCount,
              baselineRouteRecordCount >= evalIndex.promptCount,
              maskedRouteRecordCount >= evalIndex.promptCount else {
            return "eval_index is missing routing record evidence for every indexed prompt."
        }
        if let issue = evalIndexLayerStatsCoverageIssue(evalIndex) {
            return issue
        }
        guard evalIndex.comparisonSummary?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            return "eval_index is missing comparison_summary evidence."
        }
        guard evalIndex.evalJSONL?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            return "eval_index is missing eval.jsonl evidence."
        }
        guard evalIndex.evalTraceJSONL?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            return "eval_index is missing eval_trace.jsonl evidence."
        }
        guard evalIndex.runtimeMode?.isEmpty == false,
              evalIndex.runtimeDevice?.isEmpty == false,
              evalIndex.runtimeMetalEnabled != nil else {
            return "eval_index is missing runtime device evidence."
        }
        if evalIndex.runtimeMetalEnabled != true {
            return "eval_index did not record a Metal runtime."
        }
        if evalIndex.runtimeMode != "bf16_vmlx" {
            return "eval_index did not record BF16/vMLX runtime evidence."
        }
        if evalIndex.runtimeBackend != "vmlx" {
            return "eval_index did not record vMLX backend evidence."
        }
        if evalIndex.hookCoverageComplete == false {
            return "eval_index recorded incomplete vMLX routed-layer hook coverage."
        }
        if expectedLayerCount > 0, evalIndex.hookedMOELayers == nil {
            return "eval_index is missing vMLX routed-layer hook evidence."
        }
        if let hookedLayers = evalIndex.hookedMOELayers, hookedLayers < expectedLayerCount {
            return "eval_index vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers."
        }
        if let expectedMOELayers = evalIndex.expectedMOELayers,
           let hookedLayers = evalIndex.hookedMOELayers,
           hookedLayers < expectedMOELayers {
            return "eval_index vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers."
        }
        guard evalIndex.jangToolsVersion?.isEmpty == false,
              evalIndex.mlxVersion?.isEmpty == false,
              evalIndex.mlxLMVersion?.isEmpty == false else {
            return "eval_index is missing vMLX package version evidence."
        }
        guard let evalSourcePath = evalIndex.sourceModelPath, !evalSourcePath.isEmpty else {
            return "eval_index is missing source model path evidence."
        }
        if let sourceModelPath,
           canonicalPath(evalSourcePath) != canonicalPath(sourceModelPath) {
            return "eval_index source model path does not match the selected BF16/F16 source."
        }
        if evalIndex.maskApplied != true {
            return "eval_index did not record an applied BF16/vMLX mask."
        }
        let maskJSON = evalIndex.maskJSON ?? evalIndex.mask
        guard maskJSON?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            return "eval_index is missing mask.json evidence."
        }
        if (evalIndex.disabledExpertCount ?? 0) <= 0 {
            return "eval_index did not record disabled expert evidence; top-k-only comparisons cannot authorize hard pruning."
        }
        if evalIndex.generationSettingsChecked != true {
            return "eval_index is missing decode settings evidence."
        }
        return nil
    }

    private static func evalIndexSemanticCoverageIssue(_ evalIndex: ImportedEvalIndexSummary) -> String? {
        guard !evalIndex.semanticCoverage.isEmpty else {
            return "eval_index is missing semantic coverage evidence."
        }
        let coverage = Set(
            evalIndex.semanticCoverage
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let missingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(coverage)
            .sorted()
        if !missingCoverage.isEmpty {
            return "eval_index semantic coverage is missing required probes: \(missingCoverage.joined(separator: ", "))."
        }
        guard let recordedMissing = evalIndex.missingSemanticCoverage else {
            return "eval_index is missing missing-semantic-coverage evidence."
        }
        let missing = Set(
            recordedMissing
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        if !missing.isEmpty {
            return "eval_index records missing semantic prompt probes: \(missing.sorted().joined(separator: ", "))."
        }
        return nil
    }

    private static func evalIndexValidatorEvidenceIssue(_ evalIndex: ImportedEvalIndexSummary) -> String? {
        guard evalIndex.validatorSchema?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false,
              evalIndex.validatorAvailablePromptCount != nil,
              evalIndex.promptClassificationCounts != nil else {
            return "eval_index is missing validator classification evidence."
        }
        guard let baselineQualified = evalIndex.baselineQualifiedPromptCount,
              baselineQualified > 0 else {
            return "eval_index has no baseline-qualified validator prompts."
        }
        if evalIndex.baselineQualifiedPromptIDs.count != baselineQualified {
            return "eval_index baseline-qualified prompt IDs do not match the baseline-qualified count."
        }
        let classified = evalIndex.baselineInvalidPromptIDs.count
            + evalIndex.inconclusivePromptIDs.count
            + evalIndex.preservedPromptIDs.count
            + evalIndex.degradedPromptIDs.count
        if classified != evalIndex.promptCount {
            return "eval_index prompt classifications cover \(classified) of \(evalIndex.promptCount) prompts."
        }
        if !evalIndex.degradedPromptIDs.isEmpty {
            let preview = evalIndex.degradedPromptIDs.prefix(5).joined(separator: ", ")
            return "eval_index has baseline-qualified prompt regressions: \(preview)."
        }
        if !evalIndex.missingBaselineQualifiedSemanticCoverage.isEmpty {
            return "eval_index baseline-qualified semantic coverage is missing: \(evalIndex.missingBaselineQualifiedSemanticCoverage.sorted().joined(separator: ", "))."
        }
        if evalIndex.baselineQualifiedSemanticCoverage.isEmpty {
            return "eval_index is missing baseline-qualified semantic coverage evidence."
        }
        if let passRate = evalIndex.baselineQualifiedMaskedPassRate,
           passRate < 1.0 {
            return "eval_index masked validator pass rate is below 100% on baseline-qualified prompts."
        }
        return nil
    }

    private static func evalIndexLayerStatsCoverageIssue(_ evalIndex: ImportedEvalIndexSummary) -> String? {
        let baselineCount = evalIndex.baselineLayerStatsPromptCount
        let maskedCount = evalIndex.maskedLayerStatsPromptCount
        guard baselineCount != nil || maskedCount != nil else { return nil }
        guard baselineCount == evalIndex.promptCount, maskedCount == evalIndex.promptCount else {
            return "eval_index layer-stat coverage is incomplete for indexed prompts."
        }
        return nil
    }

    private static func isBlockingRegressionSeverity(_ severity: String?) -> Bool {
        severity == "high" || severity == "critical"
    }

    private static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }
}

private enum ImportedPrunePlanError: Error, LocalizedError {
    case emptyLayers
    case mixedKeepCounts
    case declaredKeepMismatch(declared: Int, actual: Int)
    case emptyKeep
    case keepTooHigh(keep: Int, sourceExperts: Int)
    case sourceMismatch(planSource: String, selectedSource: String)
    case missingComparison
    case safetyRejected(String)
    case comparisonRejected(String)
    case semanticEvidenceRejected(String)

    var errorDescription: String? {
        switch self {
        case .emptyLayers:
            return "plan has no layers."
        case .mixedKeepCounts:
            return "plan uses different keep counts across layers."
        case .declaredKeepMismatch(let declared, let actual):
            return "plan declares \(declared) experts per layer but contains \(actual)."
        case .emptyKeep:
            return "plan keep lists are empty."
        case .keepTooHigh(let keep, let sourceExperts):
            return "plan keeps \(keep) experts but source only has \(sourceExperts)."
        case .sourceMismatch(let planSource, let selectedSource):
            return "plan source_model does not match the selected BF16/F16 source. Plan: \(planSource). Selected: \(selectedSource)."
        case .missingComparison:
            return "plan is missing same-suite A/B comparison evidence."
        case .safetyRejected(let issue):
            return "plan failed the top-k safety gate: \(issue)"
        case .comparisonRejected(let issue):
            return "plan failed the same-suite A/B comparison gate: \(issue)"
        case .semanticEvidenceRejected(let issue):
            return "plan failed the semantic evidence gate: \(issue)"
        }
    }
}

private enum PrequantPruneError: Error, LocalizedError {
    case cli(code: Int32, stderr: String)
    case materialization(String)

    var errorDescription: String? {
        switch self {
        case .cli(let code, let stderr):
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty
                ? "prequant prune exited \(code)"
                : "prequant prune exited \(code): \(trimmed)"
        case .materialization(let issue):
            return "reviewed prune evidence materialization failed: \(issue)"
        }
    }
}
