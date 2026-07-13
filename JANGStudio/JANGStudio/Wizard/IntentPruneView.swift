// JANGStudio/JANGStudio/Wizard/IntentPruneView.swift
// Intent Prune sheet — intents + Keep/Balanced/CRACK + budget (plan §6).
import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct IntentPruneView: View {
    let sourceURL: URL
    let detected: ArchitectureSummary
    let onAdoptPrunedSource: (URL) -> Void
    var onOpenAdvancedExpertLab: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var vm: IntentPruneViewModel

    init(
        sourceURL: URL,
        detected: ArchitectureSummary,
        onAdoptPrunedSource: @escaping (URL) -> Void,
        onOpenAdvancedExpertLab: (() -> Void)? = nil
    ) {
        self.sourceURL = sourceURL
        self.detected = detected
        self.onAdoptPrunedSource = onAdoptPrunedSource
        self.onOpenAdvancedExpertLab = onOpenAdvancedExpertLab
        self._vm = State(
            initialValue: IntentPruneViewModel(sourceURL: sourceURL, detected: detected)
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Form {
                sourceSection
                intentsSection
                safetySection
                budgetSection
                evidenceSection
                transitionsSection
                progressSection
                resultSection
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
            .background(ExpertLabVisual.canvas)
            Divider()
            footer
        }
        .background(ExpertLabVisual.canvas)
        .frame(minWidth: 720, minHeight: 640)
        .disabled(vm.isRunning && vm.cancelRequested)
        .alert(
            "Intent Prune",
            isPresented: Binding(
                get: { vm.errorText != nil && !vm.isRunning },
                set: { if !$0 { vm.errorText = nil } }
            )
        ) {
            Button("OK") { vm.errorText = nil }
        } message: {
            Text(vm.errorText ?? "")
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            ExpertLabConsoleCard(accent: ExpertLabVisual.accent) {
                VStack(alignment: .leading, spacing: 10) {
                    ExpertLabKicker(text: "Intent Prune")
                    Text("Shape model by capability")
                        .font(.title3.weight(.semibold))
                    Text("Pick what this MoE should stay good at, choose a safety stance, set a size budget, then score + hard-prune a new BF16/F16 source. Original source stays immutable.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    ExpertLabWorkflowStrip(
                        steps: ["Intents", "Score", "Hard Prune", "Verify", "Convert"],
                        activeIndex: stripIndex
                    )
                    Text(vm.sourceSummaryLine)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 8) {
                if let onOpenAdvancedExpertLab {
                    Button {
                        onOpenAdvancedExpertLab()
                        dismiss()
                    } label: {
                        Label("Advanced Expert Lab", systemImage: "point.3.connected.trianglepath.dotted")
                    }
                    .disabled(vm.isRunning)
                }
                Button("Close") {
                    if vm.isRunning { vm.cancelRun() }
                    dismiss()
                }
                .disabled(vm.isRunning && !vm.cancelRequested)
            }
        }
        .padding(14)
    }

    private var stripIndex: Int {
        switch vm.phase {
        case .idle: return 0
        case .preparing, .scoring, .writingPlan: return 1
        case .hardPruning: return 2
        case .verifying: return 3
        case .ready: return 4
        case .failed: return 0
        }
    }

    // MARK: - Sections

    private var sourceSection: some View {
        Section("Source") {
            LabeledContent("Model", value: sourceURL.lastPathComponent)
            LabeledContent("Experts", value: "\(detected.numExperts) per layer")
            if let layers = detected.numHiddenLayers {
                LabeledContent("Layers", value: "\(layers)")
            }
            LabeledContent("Dtype", value: detected.dtype.rawValue.uppercased())
            Label("The original source directory is never modified.", systemImage: "lock.shield")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var intentsSection: some View {
        Section {
            Text("What should this model be good at? (multi-select)")
                .font(.caption)
                .foregroundStyle(.secondary)
            FlowChipGrid(chips: IntentPruneChip.allCases, selected: vm.selectedChips) { chip in
                vm.toggleChip(chip)
            }
            if vm.selectedChips.isEmpty {
                Label("Select at least one capability chip.", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(ExpertLabVisual.warm)
            } else {
                Text("Domains: \(vm.intentDomainKeys.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        } header: {
            Text("Intents")
        }
    }

    private var safetySection: some View {
        Section {
            Picker("Stance", selection: Binding(
                get: { vm.safetyStance },
                set: { vm.setSafetyStance($0) }
            )) {
                ForEach(IntentPruneSafetyStance.allCases) { stance in
                    Text(stance.title).tag(stance)
                }
            }
            .pickerStyle(.segmented)
            .disabled(vm.isRunning)

            Text(vm.safetyStance.subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)

            if vm.safetyStance.isCrack {
                ExpertLabConsoleCard(accent: ExpertLabVisual.danger) {
                    VStack(alignment: .leading, spacing: 8) {
                        ExpertLabKicker(text: "CRACK — abliteration", color: ExpertLabVisual.danger)
                        Text("CRACK down-ranks experts that specialize in safety/refusal paths while protecting your keep-intents. Output folders include a -CRACK suffix. This is not a silent default.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Toggle(isOn: $vm.crackConfirmed) {
                            Text("I understand CRACK is abliteration and want to proceed")
                                .font(.callout)
                        }
                        .disabled(vm.isRunning)
                        if let msg = vm.crackGateMessage {
                            Label(msg, systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundStyle(ExpertLabVisual.warm)
                        }
                    }
                }
            }
        } header: {
            Text("Safety stance")
        }
    }

    private var budgetSection: some View {
        Section {
            Picker("Budget", selection: Binding(
                get: { vm.budget },
                set: { vm.setBudget($0) }
            )) {
                ForEach(IntentPruneBudget.allCases) { b in
                    Text(b.title).tag(b)
                }
            }
            .pickerStyle(.segmented)
            .disabled(vm.isRunning)

            LabeledContent("Keep-K", value: vm.keepSummaryLine)
            Text("Light ≈ 90% · Standard ≈ 75% · Aggressive ≈ 60% of experts per layer (clamped to trained top-k).")
                .font(.caption)
                .foregroundStyle(.secondary)
        } header: {
            Text("Size budget")
        }
    }

    private var evidenceSection: some View {
        Section("Evidence") {
            Label(vm.evidenceLine, systemImage: "checkmark.seal")
                .font(.caption)
            Text("Reviewed 50 is the authority suite for Intent Prune. Smoke suites cannot unlock final quant.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var transitionsSection: some View {
        Section {
            HStack {
                if let url = vm.transitionsURL {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(url.lastPathComponent).font(.callout.weight(.medium))
                        Text(url.path)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                } else {
                    Text("No expert_transitions.jsonl selected")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Choose…") { chooseTransitions() }
                    .disabled(vm.isRunning)
                if vm.transitionsURL != nil {
                    Button("Clear") {
                        vm.transitionsURL = nil
                    }
                    .disabled(vm.isRunning)
                }
            }
            if vm.transitionsURL == nil {
                Label(
                    "Attach transitions from a Reviewed 50 BF16/vMLX run, or open Advanced Expert Lab to generate them. Score + hard-prune wire real CLI when this file is present.",
                    systemImage: "info.circle"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Text("Output")
                Spacer()
                Text(vm.outputURL.path)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Button("Choose…") { chooseOutput() }
                    .disabled(vm.isRunning)
            }
            if vm.outputConflictsWithSource {
                Label("Output must be outside the source tree.", systemImage: "xmark.octagon.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        } header: {
            Text("Transitions & output")
        }
    }

    @ViewBuilder
    private var progressSection: some View {
        if vm.isRunning || vm.phase == .ready || vm.phase == .failed {
            Section("Progress") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        if vm.isRunning {
                            ProgressView().controlSize(.small)
                        }
                        Text(vm.phase.label)
                            .font(.callout.weight(.medium))
                        Spacer()
                        if vm.isRunning {
                            Button("Cancel", role: .destructive) {
                                vm.cancelRun()
                            }
                        }
                    }
                    ProgressView(value: vm.phase.progress)
                    if !vm.statusDetail.isEmpty {
                        Text(vm.statusDetail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    if let summary = vm.lastScoreSummary, !summary.isEmpty {
                        DisclosureGroup("Scorer output") {
                            Text(summary)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        if vm.phase == .ready {
            Section("Result") {
                if let plan = vm.planURL {
                    LabeledContent("Plan", value: plan.lastPathComponent)
                }
                LabeledContent("Keep-K", value: "\(vm.keepK)")
                LabeledContent("Stance", value: vm.safetyStance.title)
                LabeledContent(
                    "Intents",
                    value: vm.selectedChips.map(\.title).sorted().joined(separator: ", ")
                )
                LabeledContent("Output", value: vm.outputURL.path)
                    .textSelection(.enabled)
                if vm.canAdopt {
                    Label("Structural prune complete. Adopt to continue Convert on the pruned source.",
                          systemImage: "checkmark.seal.fill")
                        .font(.caption)
                        .foregroundStyle(ExpertLabVisual.good)
                } else if vm.planURL != nil, !vm.prunedVerified {
                    Label("Score/plan ready. Run full Intent Prune to hard-prune, or adopt after prune verifies.",
                          systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 10) {
            if vm.canAdopt {
                Button {
                    onAdoptPrunedSource(vm.outputURL)
                    dismiss()
                } label: {
                    Label("Convert pruned model", systemImage: "arrow.right.circle.fill")
                }
                .buttonStyle(.borderedProminent)
            }
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([vm.outputURL])
            } label: {
                Label("Reveal folder", systemImage: "folder")
            }
            .disabled(!FileManager.default.fileExists(atPath: vm.outputURL.path))

            Spacer()

            Button {
                vm.previewScores()
            } label: {
                Label("Preview scores", systemImage: "chart.bar.doc.horizontal")
            }
            .disabled(!vm.canPreviewScores)

            Button {
                vm.runIntentPrune()
            } label: {
                Label("Run Intent Prune", systemImage: "scissors")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!vm.canRun)
        }
        .padding(12)
    }

    // MARK: - Pickers

    private func chooseTransitions() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Select expert_transitions.jsonl from a Reviewed 50 BF16/vMLX run"
        panel.allowedContentTypes = [
            UTType(filenameExtension: "jsonl") ?? .json,
            .json,
            .data,
        ]
        if panel.runModal() == .OK, let url = panel.url {
            vm.transitionsURL = url
        }
    }

    private func chooseOutput() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Folder for the pruned BF16/F16 source (created if needed)"
        if panel.runModal() == .OK, let url = panel.url {
            // If user picks a parent, nest the default artifact name under it.
            if url.lastPathComponent == sourceURL.deletingLastPathComponent().lastPathComponent
                || !url.lastPathComponent.contains("intent") {
                vm.outputURL = url.appendingPathComponent(
                    IntentPruneCLIArgsBuilder.artifactFolderName(
                        sourceBaseName: sourceURL.lastPathComponent,
                        chips: Array(vm.selectedChips),
                        keepK: vm.keepK,
                        safetyStance: vm.safetyStance
                    )
                )
            } else {
                vm.outputURL = url
            }
        }
    }
}

// MARK: - Chip grid

/// Simple wrapping chip row without UIKit FlowLayout dependency.
private struct FlowChipGrid: View {
    let chips: [IntentPruneChip]
    let selected: Set<IntentPruneChip>
    let onToggle: (IntentPruneChip) -> Void

    var body: some View {
        // Fixed multi-row layout — readable on macOS Form without custom layout.
        VStack(alignment: .leading, spacing: 8) {
            chipRow(Array(chips.prefix(4)))
            chipRow(Array(chips.dropFirst(4)))
        }
    }

    private func chipRow(_ items: [IntentPruneChip]) -> some View {
        HStack(spacing: 8) {
            ForEach(items) { chip in
                let isOn = selected.contains(chip)
                Button {
                    onToggle(chip)
                } label: {
                    Text(chip.title)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            isOn
                                ? ExpertLabVisual.accent.opacity(0.22)
                                : ExpertLabVisual.panelRaised
                        )
                        .foregroundStyle(isOn ? ExpertLabVisual.accent : .secondary)
                        .overlay {
                            Capsule()
                                .stroke(
                                    isOn ? ExpertLabVisual.accent.opacity(0.55) : ExpertLabVisual.line,
                                    lineWidth: 1
                                )
                        }
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)
            }
            Spacer(minLength: 0)
        }
    }
}
