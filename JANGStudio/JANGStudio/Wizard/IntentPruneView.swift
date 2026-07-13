// JANGStudio/JANGStudio/Wizard/IntentPruneView.swift
// Intent Prune sheet — design-b quality loop:
// Evidence → Shape (keep/drop surface) → Prune → Quality → Convert
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
            workflowPills
            Divider().overlay(ExpertLabVisual.line)
            HStack(spacing: 0) {
                stageDock
                Divider().overlay(ExpertLabVisual.line)
                stageContent
            }
            Divider().overlay(ExpertLabVisual.line)
            footer
        }
        .background(ExpertLabVisual.canvas)
        .frame(minWidth: 920, minHeight: 680)
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
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                ExpertLabKicker(text: "Intent Prune")
                Text(vm.stage.headline)
                    .font(.title3.weight(.semibold))
                Text(vm.sourceSummaryLine)
                    .font(.caption2)
                    .foregroundStyle(ExpertLabVisual.textDim)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 8)
            HStack(spacing: 6) {
                badge(detected.modelType, color: ExpertLabVisual.accent)
                badge("\(detected.numExperts)e", color: ExpertLabVisual.warm)
                if vm.prunedVerified {
                    badge("pruned K=\(vm.keepK)", color: ExpertLabVisual.good)
                }
            }
            if let onOpenAdvancedExpertLab {
                Button {
                    onOpenAdvancedExpertLab()
                    dismiss()
                } label: {
                    Label("Atlas", systemImage: "point.3.connected.trianglepath.dotted")
                }
                .disabled(vm.isRunning)
            }
            Button("Close") {
                if vm.isRunning { vm.cancelRun() }
                dismiss()
            }
            .disabled(vm.isRunning && !vm.cancelRequested)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(ExpertLabVisual.panel)
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .foregroundStyle(color)
            .background(color.opacity(0.1))
            .overlay(
                RoundedRectangle(cornerRadius: 3)
                    .stroke(color.opacity(0.3), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 3))
    }

    // MARK: - Workflow pills

    private var workflowPills: some View {
        HStack(spacing: 4) {
            ForEach(IntentPruneStage.allCases) { stage in
                let active = vm.stage == stage
                let done = stageDone(stage) && !active
                let blocked = !vm.canEnterStage(stage) && !active && stage != .evidence && stage != .shape
                Button {
                    vm.goToStage(stage)
                } label: {
                    HStack(spacing: 4) {
                        Text(String(format: "%02d", stage.rawValue + 1))
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                        Text(stage.title)
                            .font(.system(size: 10, weight: .medium))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .foregroundStyle(active ? ExpertLabVisual.accent : (done ? ExpertLabVisual.good : ExpertLabVisual.textFaint))
                    .background(
                        active
                            ? ExpertLabVisual.accent.opacity(0.08)
                            : (done ? ExpertLabVisual.good.opacity(0.06) : Color.white.opacity(0.02))
                    )
                    .overlay(
                        Capsule()
                            .stroke(
                                active
                                    ? ExpertLabVisual.accent
                                    : (done ? ExpertLabVisual.good.opacity(0.35) : ExpertLabVisual.line),
                                lineWidth: 1
                            )
                    )
                    .clipShape(Capsule())
                    .opacity(blocked ? 0.4 : 1)
                }
                .buttonStyle(.plain)
                .disabled(blocked)
            }
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(ExpertLabVisual.panel)
    }

    private func stageDone(_ stage: IntentPruneStage) -> Bool {
        switch stage {
        case .evidence: return vm.hasEvidence
        case .shape: return vm.shapeComplete
        case .prune: return vm.prunedVerified
        case .quality: return vm.qualityAcknowledged
        case .convert: return vm.canAdopt && vm.qualityAcknowledged
        }
    }

    // MARK: - Dock

    private var stageDock: some View {
        VStack(spacing: 2) {
            ForEach(IntentPruneStage.allCases) { stage in
                let active = vm.stage == stage
                let done = stageDone(stage) && !active
                Button {
                    vm.goToStage(stage)
                } label: {
                    VStack(spacing: 2) {
                        Image(systemName: stage.dockSymbol)
                            .font(.system(size: 14, weight: .medium))
                        Text(stage.title)
                            .font(.system(size: 7, weight: .medium))
                    }
                    .frame(width: 52)
                    .padding(.vertical, 8)
                    .foregroundStyle(active ? ExpertLabVisual.accent : (done ? ExpertLabVisual.good : ExpertLabVisual.textFaint))
                    .background(active ? ExpertLabVisual.accent.opacity(0.04) : Color.clear)
                    .overlay(alignment: .leading) {
                        Rectangle()
                            .fill(active ? ExpertLabVisual.accent : Color.clear)
                            .frame(width: 2)
                    }
                }
                .buttonStyle(.plain)
            }
            Spacer()
        }
        .frame(width: 52)
        .background(ExpertLabVisual.panel)
    }

    // MARK: - Stage content

    @ViewBuilder
    private var stageContent: some View {
        switch vm.stage {
        case .evidence:
            evidenceStage
        case .shape:
            shapeStage
        case .prune:
            pruneStage
        case .quality:
            qualityStage
        case .convert:
            convertStage
        }
    }

    // MARK: Shape — primary selection surface

    private var shapeStage: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("This is the prune decision surface. Green = keep (protect). Red = drop (deprioritize when freeing experts).")
                        .font(.system(size: 11))
                        .foregroundStyle(ExpertLabVisual.textDim)
                        .fixedSize(horizontal: false, vertical: true)

                    // Presets
                    Text("QUICK PRESETS")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                        .tracking(0.6)
                    FlowLayout(spacing: 6) {
                        ForEach(IntentPrunePreset.all) { preset in
                            Button(preset.title) {
                                vm.applyPreset(preset)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        Button("Clear") { vm.clearSelections() }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                    }

                    // Dual keep / drop panels
                    HStack(alignment: .top, spacing: 10) {
                        selectionPanel(
                            title: "Keep",
                            subtitle: "Protect experts used for these domains. Select at least one.",
                            count: vm.selectedChips.count,
                            accent: ExpertLabVisual.good,
                            isDrop: false
                        ) {
                            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 6) {
                                ForEach(IntentPruneChip.allCases) { chip in
                                    capabilityTile(
                                        title: chip.title,
                                        detail: chip.detail,
                                        domains: chip.domainKeys.isEmpty ? "backbone floor" : chip.domainKeys.joined(separator: " · "),
                                        color: chip.accentColor,
                                        isOn: vm.selectedChips.contains(chip),
                                        isDrop: false
                                    ) {
                                        vm.toggleChip(chip)
                                    }
                                }
                            }
                        }

                        selectionPanel(
                            title: "Drop / deprioritize",
                            subtitle: "Optional. Frees slots for keep intents. Safety-heavy switches CRACK stance.",
                            count: vm.dropChips.filter { !$0.switchesToCrackStance }.count
                                + (vm.safetyStance.isCrack ? 1 : 0),
                            accent: ExpertLabVisual.danger,
                            isDrop: true
                        ) {
                            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 6) {
                                ForEach(IntentPruneDropChip.allCases) { chip in
                                    let isOn = chip.switchesToCrackStance
                                        ? vm.safetyStance.isCrack
                                        : vm.dropChips.contains(chip)
                                    capabilityTile(
                                        title: chip.title,
                                        detail: chip.detail,
                                        domains: chip.domainKeys.isEmpty ? "stance: CRACK" : chip.domainKeys.joined(separator: " · "),
                                        color: chip.accentColor,
                                        isOn: isOn,
                                        isDrop: true
                                    ) {
                                        vm.toggleDropChip(chip)
                                    }
                                }
                            }
                        }
                    }

                    // Stance + budget
                    HStack(alignment: .top, spacing: 10) {
                        ExpertLabConsoleCard {
                            VStack(alignment: .leading, spacing: 8) {
                                ExpertLabSectionHeader(title: "Safety stance")
                                Picker("Stance", selection: Binding(
                                    get: { vm.safetyStance },
                                    set: { vm.setSafetyStance($0) }
                                )) {
                                    ForEach(IntentPruneSafetyStance.allCases) { s in
                                        Text(s.title).tag(s)
                                    }
                                }
                                .pickerStyle(.segmented)
                                .disabled(vm.isRunning)
                                Text(vm.safetyStance.subtitle)
                                    .font(.caption2)
                                    .foregroundStyle(ExpertLabVisual.textDim)
                                if vm.safetyStance.isCrack {
                                    Toggle(isOn: $vm.crackConfirmed) {
                                        Text("I understand CRACK is abliteration and want to proceed")
                                            .font(.caption)
                                    }
                                    .disabled(vm.isRunning)
                                    if let msg = vm.crackGateMessage {
                                        Label(msg, systemImage: "exclamationmark.triangle.fill")
                                            .font(.caption2)
                                            .foregroundStyle(ExpertLabVisual.warm)
                                    }
                                }
                            }
                        }

                        ExpertLabConsoleCard {
                            VStack(alignment: .leading, spacing: 8) {
                                ExpertLabSectionHeader(title: "Size budget → keep-K")
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
                                ExpertLabStatRow(label: "Keep-K", value: "\(vm.keepK) / \(vm.expertsPerLayer)")
                                ExpertLabStatRow(label: "Drop slots/layer", value: "\(vm.expertsPerLayer - vm.keepK)")
                                Text("Light ≈ 90% · Standard ≈ 75% · Aggressive ≈ 60%. If quality math fails later, raise to Light.")
                                    .font(.caption2)
                                    .foregroundStyle(ExpertLabVisual.textDim)
                            }
                        }
                    }

                    // Live plan summary
                    ExpertLabConsoleCard(accent: ExpertLabVisual.warm) {
                        VStack(alignment: .leading, spacing: 6) {
                            ExpertLabSectionHeader(title: "Live prune plan", systemImage: "list.bullet.rectangle")
                            Text(vm.planSummaryLine)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(ExpertLabVisual.accent)
                                .textSelection(.enabled)
                            Text(vm.keepSummaryLine)
                                .font(.caption2)
                                .foregroundStyle(ExpertLabVisual.textDim)
                            Text("Hybrid scorer ranks experts; budget cuts the bottom. Drop intents push those domains down the rank.")
                                .font(.caption2)
                                .foregroundStyle(ExpertLabVisual.textFaint)
                        }
                    }
                }
                .padding(14)
            }

            Divider().overlay(ExpertLabVisual.line)
            shapeRightRail
        }
    }

    private var shapeRightRail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                processChecklist
                ExpertLabConsoleCard(accent: ExpertLabVisual.warm) {
                    VStack(alignment: .leading, spacing: 8) {
                        ExpertLabSectionHeader(title: "Primary action", systemImage: "arrow.right.circle")
                        if !vm.hasEvidence {
                            Button {
                                vm.goToStage(.evidence)
                            } label: {
                                Label("Collect evidence first", systemImage: "waveform.path.ecg")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            Text("You can keep shaping; prune stays locked until transitions exist.")
                                .font(.caption2)
                                .foregroundStyle(ExpertLabVisual.textDim)
                        } else if vm.shapeComplete {
                            Button {
                                vm.goToStage(.prune)
                            } label: {
                                Label("Next · Review & Prune", systemImage: "scissors")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(ExpertLabVisual.warm)
                        } else if vm.selectedChips.isEmpty {
                            Text("Select ≥1 Keep capability.")
                                .font(.caption)
                                .foregroundStyle(ExpertLabVisual.warm)
                        } else if vm.safetyStance.isCrack, !vm.crackConfirmed {
                            Text("Confirm CRACK abliteration to continue.")
                                .font(.caption)
                                .foregroundStyle(ExpertLabVisual.warm)
                        }
                    }
                }
            }
            .padding(10)
        }
        .frame(width: 260)
        .background(ExpertLabVisual.panel)
    }

    // MARK: Evidence

    private var evidenceStage: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Scoring needs domain-tagged expert transitions from a real BF16/vMLX suite. Marker echo suites are routing-only.")
                        .font(.system(size: 11))
                        .foregroundStyle(ExpertLabVisual.textDim)

                    ExpertLabConsoleCard {
                        VStack(alignment: .leading, spacing: 10) {
                            ExpertLabSectionHeader(title: "Transitions file", systemImage: "doc.text")
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
                                    Button("Clear") { vm.transitionsURL = nil }
                                        .disabled(vm.isRunning)
                                }
                            }
                            Text(vm.evidenceLine)
                                .font(.caption2)
                                .foregroundStyle(ExpertLabVisual.textDim)
                        }
                    }

                    ExpertLabConsoleCard {
                        VStack(alignment: .leading, spacing: 8) {
                            ExpertLabSectionHeader(title: "Generate evidence")
                            Text("Open Advanced Expert Lab to run a real-domain suite with token paths / transitions emission, then attach the resulting expert_transitions.jsonl here.")
                                .font(.caption)
                                .foregroundStyle(ExpertLabVisual.textDim)
                            if let onOpenAdvancedExpertLab {
                                Button {
                                    onOpenAdvancedExpertLab()
                                    dismiss()
                                } label: {
                                    Label("Open Advanced Expert Lab", systemImage: "point.3.connected.trianglepath.dotted")
                                }
                            }
                        }
                    }

                    if vm.hasEvidence {
                        Label("Evidence ready — continue to Shape to choose keep/drop.", systemImage: "checkmark.seal.fill")
                            .font(.caption)
                            .foregroundStyle(ExpertLabVisual.good)
                    }
                }
                .padding(14)
            }
            Divider().overlay(ExpertLabVisual.line)
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    processChecklist
                    Button {
                        vm.goToStage(.shape)
                    } label: {
                        Label("Next · Shape selections", systemImage: "circle.grid.cross")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(10)
            }
            .frame(width: 260)
            .background(ExpertLabVisual.panel)
        }
    }

    // MARK: Prune

    private var pruneStage: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ExpertLabConsoleCard(accent: ExpertLabVisual.warm) {
                        VStack(alignment: .leading, spacing: 8) {
                            ExpertLabSectionHeader(title: "Pruning with this plan", systemImage: "scissors")
                            Text(vm.planSummaryLine)
                                .font(.system(size: 11, design: .monospaced))
                                .textSelection(.enabled)
                            ExpertLabStatRow(label: "Keep-K", value: "\(vm.keepK) / \(vm.expertsPerLayer)")
                            ExpertLabStatRow(label: "Output", value: vm.outputURL.lastPathComponent)
                            HStack {
                                Button("Edit keep / drop") { vm.goToStage(.shape) }
                                    .disabled(vm.isRunning)
                                Button("Choose output…") { chooseOutput() }
                                    .disabled(vm.isRunning)
                            }
                            if vm.outputConflictsWithSource {
                                Label("Output must be outside the source tree.", systemImage: "xmark.octagon.fill")
                                    .font(.caption)
                                    .foregroundStyle(.red)
                            }
                        }
                    }

                    if vm.isRunning || vm.phase == .ready || vm.phase == .failed {
                        ExpertLabConsoleCard {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    if vm.isRunning { ProgressView().controlSize(.small) }
                                    Text(vm.phase.label).font(.callout.weight(.medium))
                                    Spacer()
                                    if vm.isRunning {
                                        Button("Cancel", role: .destructive) { vm.cancelRun() }
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

                    if vm.prunedVerified {
                        Label("Structural verify OK. Quality checklist is next — structural OK ≠ quality OK.", systemImage: "checkmark.seal.fill")
                            .font(.caption)
                            .foregroundStyle(ExpertLabVisual.good)
                    }

                    Label("Original BF16/F16 source is never modified.", systemImage: "lock.shield")
                        .font(.caption)
                        .foregroundStyle(ExpertLabVisual.textDim)
                }
                .padding(14)
            }
            Divider().overlay(ExpertLabVisual.line)
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    processChecklist
                    Button {
                        vm.previewScores()
                    } label: {
                        Label("Preview scores", systemImage: "chart.bar.doc.horizontal")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(!vm.canPreviewScores)

                    Button {
                        vm.runIntentPrune()
                    } label: {
                        Label("Run Hard Prune", systemImage: "scissors")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(ExpertLabVisual.warm)
                    .disabled(!vm.canRun)

                    if vm.prunedVerified {
                        Button {
                            vm.goToStage(.quality)
                        } label: {
                            Label("Next · Quality", systemImage: "checkmark.seal")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                .padding(10)
            }
            .frame(width: 260)
            .background(ExpertLabVisual.panel)
        }
    }

    // MARK: Quality

    private var qualityStage: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Structural prune is done. Before Convert, confirm capability holdouts (math/code/keep-lang). Full auto holdout scoring runs via CLI/Expert Lab today; this gate makes the quality step explicit.")
                        .font(.system(size: 11))
                        .foregroundStyle(ExpertLabVisual.textDim)

                    HStack(spacing: 8) {
                        gateCard(title: "Structural", value: vm.prunedVerified ? "OK" : "—", pass: vm.prunedVerified)
                        gateCard(title: "Keep-K", value: "\(vm.keepK)", pass: true)
                        gateCard(title: "Stance", value: vm.safetyStance.title, pass: !vm.safetyStance.isCrack || vm.crackConfirmed)
                        gateCard(title: "Drops", value: "\(vm.dropDomainKeys.count)", pass: true)
                    }

                    ExpertLabConsoleCard {
                        VStack(alignment: .leading, spacing: 8) {
                            ExpertLabSectionHeader(title: "Recommended holdouts", systemImage: "checklist")
                            Text("• Math: exact answers (e.g. √14641 = 121, compound %)\n• Code: fizzbuzz / merge / binary search\n• Keep language purity (CJK ratio if not dropping Chinese)\n• If CRACK: still-refuse anchors + dual-use comply rate")
                                .font(.caption)
                                .foregroundStyle(ExpertLabVisual.textDim)
                            Text(vm.planSummaryLine)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(ExpertLabVisual.accent)
                        }
                    }

                    if vm.qualityAcknowledged {
                        Label("Quality step acknowledged. Convert unlocked.", systemImage: "checkmark.seal.fill")
                            .font(.caption)
                            .foregroundStyle(ExpertLabVisual.good)
                    }
                }
                .padding(14)
            }
            Divider().overlay(ExpertLabVisual.line)
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    processChecklist
                    Button {
                        vm.acknowledgeQuality()
                    } label: {
                        Label(
                            vm.qualityAcknowledged ? "Quality acknowledged" : "I verified holdouts · Unlock Convert",
                            systemImage: "checkmark.seal"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!vm.prunedVerified || vm.qualityAcknowledged)

                    if vm.qualityAcknowledged {
                        Button {
                            vm.goToStage(.convert)
                        } label: {
                            Label("Next · Convert", systemImage: "arrow.right.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(ExpertLabVisual.good)
                    }

                    if let onOpenAdvancedExpertLab {
                        Button {
                            onOpenAdvancedExpertLab()
                            dismiss()
                        } label: {
                            Label("Run holdouts in Expert Lab", systemImage: "point.3.connected.trianglepath.dotted")
                                .frame(maxWidth: .infinity)
                        }
                    }
                }
                .padding(10)
            }
            .frame(width: 280)
            .background(ExpertLabVisual.panel)
        }
    }

    // MARK: Convert

    private var convertStage: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if vm.canAdopt && vm.qualityAcknowledged {
                        Label("Pruned source ready. Adopt to continue Profile → Build → Verify on the new BF16 tree.", systemImage: "checkmark.seal.fill")
                            .font(.callout)
                            .foregroundStyle(ExpertLabVisual.good)
                    } else {
                        Text("Convert unlocks after structural prune and quality acknowledgement.")
                            .font(.callout)
                            .foregroundStyle(ExpertLabVisual.textDim)
                    }
                    ExpertLabConsoleCard {
                        VStack(alignment: .leading, spacing: 6) {
                            ExpertLabStatRow(label: "Plan", value: vm.planURL?.lastPathComponent ?? "—")
                            ExpertLabStatRow(label: "Keep-K", value: "\(vm.keepK)")
                            ExpertLabStatRow(label: "Stance", value: vm.safetyStance.title)
                            ExpertLabStatRow(label: "Output", value: vm.outputURL.path)
                            Text(vm.planSummaryLine)
                                .font(.caption2)
                                .foregroundStyle(ExpertLabVisual.textDim)
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding(14)
            }
            Divider().overlay(ExpertLabVisual.line)
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    processChecklist
                    if vm.canAdopt && vm.qualityAcknowledged {
                        Button {
                            onAdoptPrunedSource(vm.outputURL)
                            dismiss()
                        } label: {
                            Label("Convert pruned model", systemImage: "arrow.right.circle.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(ExpertLabVisual.good)
                    }
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([vm.outputURL])
                    } label: {
                        Label("Reveal folder", systemImage: "folder")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(!FileManager.default.fileExists(atPath: vm.outputURL.path))
                }
                .padding(10)
            }
            .frame(width: 260)
            .background(ExpertLabVisual.panel)
        }
    }

    // MARK: Shared pieces

    private var processChecklist: some View {
        ExpertLabConsoleCard {
            VStack(alignment: .leading, spacing: 4) {
                ExpertLabSectionHeader(title: "Process")
                checklistRow("Evidence", done: vm.hasEvidence)
                checklistRow("Shape (selections)", done: vm.shapeComplete)
                checklistRow("Prune", done: vm.prunedVerified)
                checklistRow("Quality", done: vm.qualityAcknowledged)
                checklistRow("Convert", done: vm.canAdopt && vm.qualityAcknowledged)
            }
        }
    }

    private func checklistRow(_ title: String, done: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: done ? "checkmark.square.fill" : "square")
                .foregroundStyle(done ? ExpertLabVisual.good : ExpertLabVisual.textFaint)
                .font(.system(size: 11))
            Text(title)
                .font(.system(size: 10))
                .foregroundStyle(done ? ExpertLabVisual.accent : ExpertLabVisual.textDim)
            Spacer()
        }
    }

    private func gateCard(title: String, value: String, pass: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased())
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(ExpertLabVisual.textDim)
                .tracking(0.5)
            Text(value)
                .font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(pass ? ExpertLabVisual.good : ExpertLabVisual.warm)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ExpertLabVisual.panelRaised)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(pass ? ExpertLabVisual.good.opacity(0.35) : ExpertLabVisual.line, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func selectionPanel<Content: View>(
        title: String,
        subtitle: String,
        count: Int,
        accent: Color,
        isDrop: Bool,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(accent)
                Spacer()
                Text("\(count)")
                    .font(.system(size: 10, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .foregroundStyle(accent)
                    .overlay(Capsule().stroke(accent.opacity(0.35), lineWidth: 1))
            }
            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(ExpertLabVisual.textDim)
            content()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ExpertLabVisual.panelRaised)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(accent.opacity(0.28), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func capabilityTile(
        title: String,
        detail: String,
        domains: String,
        color: Color,
        isOn: Bool,
        isDrop: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 7) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(color)
                    .frame(width: 10, height: 10)
                    .padding(.top, 2)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(isOn ? .primary : .secondary)
                        .multilineTextAlignment(.leading)
                    Text(detail)
                        .font(.system(size: 9))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(domains)
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                        .lineLimit(1)
                }
                Spacer(minLength: 2)
                Text(isOn ? (isDrop ? "×" : "✓") : "")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(isDrop ? ExpertLabVisual.danger : ExpertLabVisual.good)
            }
            .padding(8)
            .background(
                isOn
                    ? (isDrop ? ExpertLabVisual.danger.opacity(0.08) : ExpertLabVisual.good.opacity(0.08))
                    : Color.white.opacity(0.02)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(
                        isOn
                            ? (isDrop ? ExpertLabVisual.danger.opacity(0.45) : ExpertLabVisual.good.opacity(0.45))
                            : ExpertLabVisual.line,
                        lineWidth: 1
                    )
            )
            .clipShape(RoundedRectangle(cornerRadius: 5))
        }
        .buttonStyle(.plain)
        .disabled(vm.isRunning)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 10) {
            if vm.canAdopt && vm.qualityAcknowledged {
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

            if vm.stage == .shape || vm.stage == .prune {
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
        }
        .padding(12)
        .background(ExpertLabVisual.panel)
    }

    // MARK: - Pickers

    private func chooseTransitions() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Select expert_transitions.jsonl from a real-domain BF16/vMLX run (preferred over marker-only suites)"
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
            if url.lastPathComponent == sourceURL.deletingLastPathComponent().lastPathComponent
                || !url.lastPathComponent.contains("intent") {
                vm.outputURL = url.appendingPathComponent(
                    IntentPruneCLIArgsBuilder.artifactFolderName(
                        sourceBaseName: sourceURL.lastPathComponent,
                        chips: Array(vm.selectedChips),
                        keepK: vm.keepK,
                        safetyStance: vm.safetyStance,
                        dropChips: Array(vm.dropChips)
                    )
                )
            } else {
                vm.outputURL = url
            }
        }
    }
}

// MARK: - Simple flow layout for presets

private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrange(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for (index, frame) in result.frames.enumerated() {
            subviews[index].place(
                at: CGPoint(x: bounds.minX + frame.minX, y: bounds.minY + frame.minY),
                proposal: ProposedViewSize(frame.size)
            )
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, frames: [CGRect]) {
        let maxWidth = proposal.width ?? .infinity
        var frames: [CGRect] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var width: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            frames.append(CGRect(origin: CGPoint(x: x, y: y), size: size))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
            width = max(width, x - spacing)
        }
        return (CGSize(width: width, height: y + rowHeight), frames)
    }
}
