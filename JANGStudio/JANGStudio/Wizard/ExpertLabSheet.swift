import AppKit
import CryptoKit
import Foundation
import JANGExpertLab
import JANGKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

enum ExpertLabPresentation {
    case standalone
    case embeddedInWizard
}

struct ExpertLabSheet: View {
    let modelPath: URL
    let modelType: String
    let profile: String
    let sizeGb: Double
    let reviewMode: Bool
    let showsCloseButton: Bool
    let presentation: ExpertLabPresentation
    let onPrunePlanReady: ((URL) -> Void)?

    @Environment(\.dismiss) private var dismiss
    @State private var vm: ExpertLabViewModel
    @State private var atlasDisplayMode: ExpertAtlasDisplayMode = .condensedCards
    @State private var lassoStart: CGPoint?
    @State private var lassoEnd: CGPoint?
    @State private var showsAdvancedTraceSettings = false
    @State private var showsAtlasLegend = false
    // M224: condensed-cards atlas coloring mode and minimum-hits threshold.
    @State private var condensedAtlasColorMode: CondensedAtlasColorMode = .domain
    @State private var condensedAtlasHitsThreshold: Double = 0
    // M221: debounced atlas filter text — prevents grid redraw per keystroke.
    // TextField binds to these @State vars; onChange debounces before writing to
    // the ViewModel's @Observable properties. Without this, every character triggers
    // filteredAtlasEntries → atlasGridRows re-evaluation on the main thread.
    @State private var debouncedLayerFilterText = ""
    @State private var debouncedExpertFilterText = ""
    @State private var debouncedDomainFilterText = ""
    @State private var debouncedPromptFilterText = ""
    /// M221: single debounce task — coalesces all four filter-text updates into
    /// one 150 ms tick so atlas entries aren't re-filtered on every keystroke.
    @State private var atlasFilterDebounceTask: Task<Void, Never>?
    /// M223: condensed-card hover tooltip state.
    @State private var hoveredCoordinate: ExpertCoordinate?
    @State private var hoverLocation: CGPoint?
    /// M225: detail-panel preview on hover without persisting selection.
    @State private var hoveredEntry: ExpertAtlasEntry?

    private let atlasGridCellSize: CGFloat = 20
    private let atlasGridCellSpacing: CGFloat = 2
    private let atlasGridRowSpacing: CGFloat = 3
    private let atlasGridLayerLabelWidth: CGFloat = 24
    private let atlasGridPadding: CGFloat = 8
    private let atlasTableDisplayLimit = 600

    init(
        modelPath: URL,
        modelType: String = "unknown",
        profile: String = "JANG_4K",
        sizeGb: Double = 0,
        sourceModelPath: URL? = nil,
        reviewMode: Bool = false,
        showsCloseButton: Bool = true,
        presentation: ExpertLabPresentation = .standalone,
        onPrunePlanReady: ((URL) -> Void)? = nil
    ) {
        self.modelPath = modelPath
        self.modelType = modelType
        self.profile = profile
        self.sizeGb = sizeGb
        self.reviewMode = reviewMode
        self.showsCloseButton = showsCloseButton
        self.presentation = presentation
        self.onPrunePlanReady = onPrunePlanReady
        self._vm = State(initialValue: ExpertLabViewModel(modelPath: modelPath, sourceModelPath: sourceModelPath))
    }

    var body: some View {
        labChrome
            .background(presentation == .standalone ? ExpertLabVisual.canvas : Color.clear)
            .alert(
                "Expert Lab",
                isPresented: Binding(get: { vm.lastError != nil }, set: { if !$0 { vm.lastError = nil } })
            ) {
                Button("OK") { vm.lastError = nil }
            } message: {
                Text(vm.lastError ?? "")
            }
    }

    private var labChrome: some View {
        VSplitView {
            VStack(spacing: 0) {
                if presentation == .standalone {
                    topBar
                } else {
                    embeddedHeader
                }
                HSplitView {
                    if presentation == .standalone {
                        dock
                    }
                    leftPanel
                        .frame(minWidth: 220, idealWidth: 240, maxWidth: 320)
                    atlasPanel
                    rightPanel
                        .frame(minWidth: 260, idealWidth: 280, maxWidth: 380)
                }
            }
            if shouldShowCompareTray {
                compareTray
                    .frame(minHeight: 120, idealHeight: 170, maxHeight: 340)
            }
        }
        .padding(presentation == .embeddedInWizard ? 14 : 0)
        // M221: blocking progress overlay when running a long operation.
        .overlay {
            if vm.isRunning && (vm.statusText == "Loading model" || vm.runProgressTotal > 5) {
                loadingOverlay
            }
        }
    }

    // M221: full-sheet blocking overlay for long-running operations (model load, trace).
    private var loadingOverlay: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
            VStack(spacing: 16) {
                ProgressView()
                    .controlSize(.large)
                Text(vm.statusText)
                    .font(.headline)
                    .foregroundStyle(.white)
                if vm.runProgressTotal > 0 {
                    ProgressView(value: Double(vm.runProgress), total: Double(vm.runProgressTotal))
                        .frame(width: 200)
                }
                if !vm.cancelRequested {
                    Button("Cancel", role: .destructive) {
                        vm.cancelRun()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                } else {
                    Text("Cancelling…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(32)
            .background {
                RoundedRectangle(cornerRadius: 12)
                    .fill(.ultraThinMaterial)
            }
        }
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            Text(reviewMode ? "Expert Review" : "Expert Lab")
                .font(.system(size: 13, weight: .semibold))
            ExpertLabBadge(text: "\(modelType) · \(profile)", color: ExpertLabVisual.accent)
            ExpertLabBadge(
                text: vm.capability.summary,
                color: vm.capability.isTraceSupported ? ExpertLabVisual.good : ExpertLabVisual.warm
            )
            compactWorkflow
            Spacer(minLength: 10)
            Text((modelPath.path as NSString).abbreviatingWithTildeInPath)
                .font(.system(size: 9, weight: .regular, design: .monospaced))
                .foregroundStyle(ExpertLabVisual.textFaint)
                .lineLimit(1)
                .truncationMode(.middle)
            if vm.isRunning {
                ProgressView()
                    .controlSize(.small)
                Text(vm.statusText)
                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                    .foregroundStyle(ExpertLabVisual.textDim)
                Button("Cancel") { vm.cancelRun() }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .foregroundStyle(ExpertLabVisual.warm)
                    .disabled(vm.cancelRequested)
            }
            if showsCloseButton {
                Button("Close") { dismiss() }
                    .buttonStyle(.borderless)
                    .font(.caption)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(height: 1)
        }
    }

    private var embeddedHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Label(reviewMode ? "Expert Review" : "Expert Lab",
                      systemImage: "point.3.connected.trianglepath.dotted")
                    .font(.system(size: 18, weight: .semibold))
                ExpertLabBadge(text: "\(modelType) · \(profile)", color: ExpertLabVisual.accent)
                ExpertLabBadge(
                    text: vm.capability.summary,
                    color: vm.capability.isTraceSupported ? ExpertLabVisual.good : ExpertLabVisual.warm
                )
                Spacer(minLength: 10)
                if vm.isRunning {
                    ProgressView()
                        .controlSize(.small)
                    Text(vm.statusText)
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(ExpertLabVisual.textDim)
                    Button("Cancel") { vm.cancelRun() }
                        .buttonStyle(.borderless)
                        .font(.caption)
                        .foregroundStyle(ExpertLabVisual.warm)
                        .disabled(vm.cancelRequested)
                }
            }
            HStack(spacing: 12) {
                compactWorkflow
                Spacer(minLength: 10)
                Text((modelPath.path as NSString).abbreviatingWithTildeInPath)
                    .font(.system(size: 9, weight: .regular, design: .monospaced))
                    .foregroundStyle(ExpertLabVisual.textFaint)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.horizontal, 4)
        .padding(.bottom, 12)
    }

    private var compactWorkflow: some View {
        HStack(spacing: 5) {
            let steps = ["Trace", "Atlas", "Mask", "Prune", "Convert"]
            ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                let completed = workflowCompletedIndices.contains(index)
                let active = workflowActiveIndex == index
                HStack(spacing: 2) {
                    if completed && !active {
                        Text("✓")
                            .foregroundStyle(ExpertLabVisual.good)
                    }
                    Text(String(format: "%02d %@", index + 1, step))
                        .foregroundStyle(active ? ExpertLabVisual.accent : completed ? ExpertLabVisual.good : ExpertLabVisual.textFaint)
                }
                .font(.system(size: 10, weight: active ? .semibold : .regular, design: .monospaced))
                .padding(.vertical, 2)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(active ? ExpertLabVisual.accent : Color.clear)
                        .frame(height: 2)
                }
                if index < steps.count - 1 {
                    Text("/")
                        .font(.system(size: 8, weight: .regular, design: .monospaced))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                }
            }
        }
    }

    private var workflowActiveIndex: Int {
        if vm.canGenerateReviewedPrunePlan { return 3 }
        if vm.hasComparisonSummary { return 2 }
        if vm.atlas != nil { return 1 }
        return 0
    }

    private var workflowCompletedIndices: Set<Int> {
        var indices: Set<Int> = []
        if vm.atlas != nil {
            indices.formUnion([0, 1])
        }
        if vm.hasComparisonSummary {
            indices.insert(2)
        }
        if vm.canGenerateReviewedPrunePlan {
            indices.insert(3)
        }
        return indices
    }

    private var shouldShowCompareTray: Bool {
        vm.hasComparisonSummary
            || !vm.lastEvalSummary.isEmpty
            || vm.baselineOutputReady
            || vm.maskedOutputReady
    }

    private var shouldShowRightWorkflowControls: Bool {
        vm.selectedEntry != nil
            || vm.groupSelectionCount > 0
            || vm.hasAnyReviewState
            || vm.hasComparisonSummary
            || vm.canGenerateReviewedPrunePlan
    }

    private var dock: some View {
        VStack(spacing: 2) {
            dockButton(systemImage: "point.3.connected.trianglepath.dotted", active: true, help: "Trace and atlas")
            dockButton(systemImage: "arrow.clockwise", active: false, help: "Run history")
            Spacer()
            dockButton(systemImage: "questionmark.circle", active: false, help: "Legend")
        }
        .padding(.vertical, 4)
        .frame(width: 40)
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(width: 1)
        }
    }

    private func dockButton(systemImage: String, active: Bool, help: String) -> some View {
        Image(systemName: systemImage)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(active ? ExpertLabVisual.accent : ExpertLabVisual.textFaint)
            .frame(width: 32, height: 32)
            .background(active ? ExpertLabVisual.accent.opacity(0.06) : Color.clear)
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .stroke(active ? ExpertLabVisual.accent.opacity(0.15) : Color.clear, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .help(help)
    }

    private var leftPanel: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 6) {
                ExpertLabSubPanel(title: "Trace", systemImage: "waveform.path.ecg") {
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("Suite", selection: $vm.selectedSuiteName) {
                            ForEach(vm.suites, id: \.name) { suite in
                                Text("\(suite.name) (\(suite.prompts.count))").tag(suite.name)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .onChange(of: vm.selectedSuiteName) { _, _ in
                            vm.syncSelectedPrompt()
                        }
                        Text(vm.selectedSuiteEvidenceSummary)
                            .font(.caption2)
                            .foregroundStyle(ExpertLabVisual.textDim)
                            .lineLimit(2)
                        Text(vm.reviewRuntimeTargetSummary)
                            .font(.caption2)
                            .foregroundStyle(ExpertLabVisual.textDim)
                            .lineLimit(3)
                            .textSelection(.enabled)
                        Button {
                            Task { await vm.runTrace() }
                        } label: {
                            Label(reviewMode ? "Run Trace" : "Run Trace", systemImage: "waveform.path.ecg")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(ExpertLabCompactButtonStyle(primary: true))
                        .disabled(!vm.capability.isTraceSupported || vm.isRunning)
                        DisclosureGroup(isExpanded: $showsAdvancedTraceSettings) {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(spacing: 6) {
                                    Stepper(value: $vm.maxTokens, in: 8...512, step: 8) {
                                        LabeledContent("Tokens", value: "\(vm.maxTokens)")
                                    }
                                    Stepper(value: $vm.maxTraceTokens, in: 64...65536, step: 512) {
                                        LabeledContent("Cap", value: "\(vm.maxTraceTokens)")
                                    }
                                }
                                Toggle("Token trace", isOn: $vm.emitTokenTrace)
                                Picker("Prompt", selection: $vm.selectedPromptID) {
                                    ForEach(vm.selectedSuite.prompts) { prompt in
                                        Text(prompt.id).tag(prompt.id)
                                    }
                                }
                                .labelsHidden()
                                .pickerStyle(.menu)
                                HStack {
                                    Button { vm.importSuite() } label: {
                                        Label("Import", systemImage: "square.and.arrow.down")
                                    }
                                    Button { vm.exportSelectedSuite() } label: {
                                        Label("Export", systemImage: "square.and.arrow.up")
                                    }
                                }
                                .buttonStyle(.borderless)
                            }
                            .font(.system(size: 10))
                            .padding(.top, 3)
                        } label: {
                            Text("More settings")
                                .font(.system(size: 9, weight: .medium))
                                .foregroundStyle(ExpertLabVisual.textDim)
                        }
                        if vm.isRunning {
                            Button {
                                vm.cancelRun()
                            } label: {
                                Label("Cancel", systemImage: "stop.circle")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                        }
                        if vm.runProgressTotal > 0 {
                            ProgressView(value: Double(vm.runProgress), total: Double(vm.runProgressTotal))
                            Text("\(vm.runProgress) / \(vm.runProgressTotal) prompts")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if !vm.runHistory.isEmpty {
                    ExpertLabSubPanel(title: "History", systemImage: "clock.arrow.circlepath") {
                        VStack(alignment: .leading, spacing: 6) {
	                            Picker("Run", selection: $vm.selectedRunID) {
	                                ForEach(vm.runHistory) { run in
	                                    Text(vm.title(for: run)).tag(run.runID)
	                                }
                            }
                            .labelsHidden()
                            .pickerStyle(.menu)
                            .onChange(of: vm.selectedRunID) { _, _ in
                                vm.refreshSelectedRunEvidence()
                            }
                            if !vm.selectedRunEvidenceSummary.isEmpty {
                                Text(vm.selectedRunEvidenceSummary)
                                    .font(.caption2)
                                    .foregroundStyle(ExpertLabVisual.textDim)
                                    .lineLimit(3)
                                    .textSelection(.enabled)
                            }
                            if !vm.selectedRunArtifactSummary.isEmpty {
                                Text(vm.selectedRunArtifactSummary)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(4)
                                    .textSelection(.enabled)
                            }
                            if !vm.selectedRunEvidenceWarning.isEmpty {
                                Label(vm.selectedRunEvidenceWarning, systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption2)
                                    .foregroundStyle(ExpertLabVisual.warm)
                                    .lineLimit(3)
                                    .textSelection(.enabled)
                            }
                            HStack {
                                Button { Task { await vm.loadSelectedRun() } } label: {
                                    Label("Load", systemImage: "arrow.clockwise")
                                }
                                .disabled(vm.selectedRunSummary == nil)
                                Button {
                                    if let s = vm.selectedRunSummary {
                                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: s.directoryPath)])
                                    }
                                } label: {
                                    Label("Reveal", systemImage: "folder")
                                }
                                .disabled(vm.selectedRunSummary == nil)
                                Button { vm.reloadRunHistory() } label: {
                                    Label("Refresh", systemImage: "arrow.triangle.2.circlepath")
                                }
                            }
                            .buttonStyle(.borderless)
                            .font(.caption)
                            if let dir = vm.lastRunDirectory {
                                Text("Active: \(dir.lastPathComponent)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                if vm.hasRecoveryInfo {
                    ExpertLabSubPanel(title: "Recovery", systemImage: "exclamationmark.triangle") {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(vm.recoverySummary)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(4)
                                .textSelection(.enabled)
                            HStack {
                                Button { Task { await vm.runTrace() } } label: {
                                    Label("Retry", systemImage: "arrow.clockwise")
                                }
                                .disabled(!vm.capability.isTraceSupported || vm.isRunning)
                                Button { vm.openRecoveryFolder() } label: {
                                    Label("Folder", systemImage: "folder")
                                }
                                Button { vm.copyRecoveryDiagnostics() } label: {
                                    Label("Copy", systemImage: "doc.on.doc")
                                }
                                Button(role: .destructive) { vm.cleanPartialRun() } label: {
                                    Label("Clean", systemImage: "trash")
                                }
                                .disabled(!vm.canCleanPartialRun)
                            }
                            .buttonStyle(.borderless)
                            .font(.caption)
                        }
                    }
                }

                // MARK: M224 — Condensed-card atlas visual controls
                if vm.atlas != nil {
                    ExpertLabSubPanel(title: "Cards", systemImage: "rectangle.grid.2x2") {
                        VStack(alignment: .leading, spacing: 10) {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Color by")
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundStyle(ExpertLabVisual.textDim)
                                VStack(alignment: .leading, spacing: 4) {
                                    colorModeRadio(.domain)
                                    colorModeRadio(.frequency)
                                    colorModeRadio(.dropRisk)
                                }
                            }

                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("Minimum hits")
                                        .font(.system(size: 10, weight: .medium))
                                        .foregroundStyle(ExpertLabVisual.textDim)
                                    Spacer(minLength: 0)
                                    Text("\(Int(condensedAtlasHitsThreshold))")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(ExpertLabVisual.textFaint)
                                }
                                Slider(value: $condensedAtlasHitsThreshold, in: 0...80, step: 1)
                                    .controlSize(.small)
                            }
                        }
                    }
                }
            }
            .padding(8)
        }
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(width: 1)
        }
                .frame(minWidth: 220, idealWidth: 240, maxWidth: 320)
    }

    private var atlasPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Atlas")
                        .font(.system(size: 12, weight: .semibold))
                    Text(vm.atlasSummary)
                        .font(.system(size: 10))
                        .foregroundStyle(ExpertLabVisual.textDim)
                }
                Spacer()
                Button {
                    showsAtlasLegend.toggle()
                } label: {
                    Label("Semantic colors", systemImage: "info.circle")
                        .font(.system(size: 9))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                }
                .buttonStyle(.borderless)
                .popover(isPresented: $showsAtlasLegend, arrowEdge: .top) {
                    atlasLegendPopover
                }
                Picker("Atlas view", selection: $atlasDisplayMode) {
                    ForEach(ExpertAtlasDisplayMode.allCases) { mode in
                        Label(mode.title, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 152)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background(ExpertLabVisual.canvas)
            .overlay(alignment: .bottom) {
                Rectangle()
                    .fill(ExpertLabVisual.line)
                    .frame(height: 1)
            }

            quickProbeBar
            if vm.hasLivePromptOutput {
                livePromptOutputPanel
            }

            // M225: condensed-card specific toolbar actions.
            if atlasDisplayMode == .condensedCards {
                HStack(spacing: 10) {
                    Button {
                        vm.applySelectionMask()
                    } label: {
                        Label("Mask selected", systemImage: "slash.circle")
                            .font(.system(size: 10))
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(vm.groupSelectionCount == 0)
                    .help("Temporarily mask all selected experts")

                    if vm.groupSelectionCount > 0 {
                        Text(vm.groupSelectionSummary)
                            .font(.system(size: 10))
                            .foregroundStyle(ExpertLabVisual.textDim)
                            .lineLimit(1)
                    }

                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .background(ExpertLabVisual.panel)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(ExpertLabVisual.line)
                        .frame(height: 1)
                }
            }

            HStack(spacing: 1) {
                ForEach(ExpertAtlasFilter.allCases) { filter in
                    atlasFilterButton(filter)
                }
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 3)
            .background(ExpertLabVisual.panel)
            .overlay(alignment: .bottom) {
                Rectangle()
                    .fill(ExpertLabVisual.line)
                    .frame(height: 1)
            }

            atlasQueryBar
            if atlasDisplayMode == .table {
                atlasTablePanel
            } else if atlasDisplayMode == .condensedCards {
                atlasCondensedCardsPanel(rows: vm.atlasGridRows, metricScales: vm.atlasLayerMetricScales)
            } else {
                atlasMapPanel(rows: vm.atlasGridRows, metricScales: vm.atlasLayerMetricScales)
            }
        }
        .background(ExpertLabVisual.canvas)
        .frame(minWidth: 520, maxWidth: .infinity, maxHeight: .infinity)
    }

    private func atlasMapPanel(
        rows: [ExpertAtlasLayerRow],
        metricScales: [Int: ExpertAtlasMetricScale]
    ) -> some View {
        Group {
            if rows.isEmpty {
                emptyAtlasPlaceholder
            } else {
                GeometryReader { proxy in
                    ScrollView([.horizontal, .vertical], showsIndicators: false) {
                        ZStack(alignment: .topLeading) {
                            VStack(alignment: .leading, spacing: atlasGridRowSpacing) {
                                ForEach(rows) { row in
                                    HStack(spacing: atlasGridCellSpacing) {
                                        Text("L\(row.layer)")
                                            .font(.system(size: 9, weight: .regular, design: .monospaced))
                                            .foregroundStyle(ExpertLabVisual.textFaint)
                                            .frame(width: atlasGridLayerLabelWidth, alignment: .trailing)
                                        ForEach(row.entries) { entry in
                                            expertCell(entry, metricScales: metricScales)
                                        }
                                    }
                                }
                            }
                            .padding(atlasGridPadding)
                            .coordinateSpace(name: "expertAtlasGrid")
                            .gesture(
                                DragGesture(minimumDistance: 6, coordinateSpace: .named("expertAtlasGrid"))
                                    .onChanged { value in
                                        if lassoStart == nil {
                                            lassoStart = value.startLocation
                                        }
                                        lassoEnd = value.location
                                        if let rect = currentLassoRect {
                                            previewLasso(rect: rect, rows: rows)
                                        }
                                    }
                                    .onEnded { value in
                                        if lassoStart == nil {
                                            lassoStart = value.startLocation
                                        }
                                        lassoEnd = value.location
                                        if let rect = currentLassoRect {
                                            previewLasso(rect: rect, rows: rows)
                                        }
                                        lassoStart = nil
                                        lassoEnd = nil
                                    }
                            )
                            if let rect = currentLassoRect {
                                Rectangle()
                                    .fill(Color.accentColor.opacity(0.12))
                                    .overlay {
                                        Rectangle()
                                            .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                                    }
                                    .frame(width: rect.width, height: rect.height)
                                    .offset(x: rect.minX, y: rect.minY)
                                    .allowsHitTesting(false)
                            }
                        }
                        .padding(8)
                        .frame(minWidth: proxy.size.width, minHeight: proxy.size.height, alignment: .topLeading)
                    }
                }
                .background(ExpertLabVisual.canvas)
            }
        }
    }

    

    private func atlasCondensedCardsPanel(
        rows: [ExpertAtlasLayerRow],
        metricScales: [Int: ExpertAtlasMetricScale]
    ) -> some View {
        Group {
            if rows.isEmpty {
                emptyAtlasPlaceholder
            } else {
                GeometryReader { proxy in
                    ScrollView([.horizontal, .vertical], showsIndicators: false) {
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(rows) { row in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 8) {
                                        Text("L\(row.layer)")
                                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                            .foregroundStyle(ExpertLabVisual.textDim)
                                            .frame(width: 32, alignment: .trailing)
                                        Text("\(row.entries.count) experts")
                                            .font(.system(size: 10))
                                            .foregroundStyle(ExpertLabVisual.textFaint)
                                    }
                                    .padding(.horizontal, 8)
                                    
                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 4) {
                                            ForEach(row.entries) { entry in
                                                condensedExpertCard(entry, metricScales: metricScales)
                                            }
                                        }
                                        .padding(.horizontal, 8)
                                        .padding(.bottom, 8)
                                    }
                                }
                            }
                        }
                        .padding(.vertical, 12)
                        .frame(minWidth: proxy.size.width, minHeight: proxy.size.height, alignment: .topLeading)
                    }
                    .coordinateSpace(name: "condensedCardsScroll")
                    .overlay(alignment: .topLeading) {
                        condensedCardTooltip
                    }
                }
                .background(ExpertLabVisual.canvas)
            }
        }
    }

    /// M223: floating tooltip for the currently hovered condensed card.
    private var condensedCardTooltip: some View {
        Group {
            if let coordinate = hoveredCoordinate,
               let location = hoverLocation,
               let entry = atlasEntry(for: coordinate) {
                condensedCardTooltipContent(entry: entry, location: location)
            }
        }
    }

    private func condensedCardTooltipContent(entry: ExpertAtlasEntry, location: CGPoint) -> some View {
        let tooltipWidth: CGFloat = 190
        let tooltipHeight: CGFloat = 132
        let xOffset: CGFloat = 14
        let yOffset: CGFloat = 14
        let domain = vm.dominantDomainSummary(for: entry)
        let dropRisk = condensedCardDropRisk(for: entry)
        return VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Circle()
                    .fill(domainColor(for: entry))
                    .frame(width: 8, height: 8)
                Text(domain)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(ExpertLabVisual.textDim)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            Text("L\(entry.layer) · E\(entry.expert)")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundStyle(Color.white)
                .padding(.top, 4)
            Divider()
                .background(ExpertLabVisual.line)
                .padding(.vertical, 7)
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 4) {
                GridRow {
                    ExpertLabStatRow(label: "Hits", value: "\(entry.hits)")
                    ExpertLabStatRow(label: "Tokens", value: "\(entry.tokenCount)")
                }
                GridRow {
                    ExpertLabStatRow(label: "Router mass", value: String(format: "%.3f", entry.probabilityMass))
                    ExpertLabStatRow(label: "Drop risk", value: dropRisk.text, accent: dropRisk.color)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(width: tooltipWidth, height: tooltipHeight, alignment: .topLeading)
        .background(ExpertLabVisual.panel.opacity(0.97))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(ExpertLabVisual.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .shadow(color: Color.black.opacity(0.45), radius: 10, x: 0, y: 4)
        .allowsHitTesting(false)
        .position(x: location.x + xOffset, y: location.y + yOffset)
    }

    private struct CondensedCardDropRisk {
        let text: String
        let color: Color
    }

    private func condensedCardDropRisk(for entry: ExpertAtlasEntry) -> CondensedCardDropRisk {
        if entry.isDead {
            return CondensedCardDropRisk(text: "Dead", color: ExpertLabVisual.textDim)
        }
        if vm.isSafeDropCandidate(layer: entry.layer, expert: entry.expert) {
            return CondensedCardDropRisk(text: "Safe", color: ExpertLabVisual.good)
        }
        if vm.isDropCandidate(layer: entry.layer, expert: entry.expert) {
            return CondensedCardDropRisk(text: "Drop", color: ExpertLabVisual.danger)
        }
        if vm.isLockedKeep(layer: entry.layer, expert: entry.expert) {
            return CondensedCardDropRisk(text: "Kept", color: ExpertLabVisual.accent)
        }
        return CondensedCardDropRisk(text: "Low", color: ExpertLabVisual.good)
    }

    private func atlasEntry(for coordinate: ExpertCoordinate) -> ExpertAtlasEntry? {
        vm.filteredAtlasEntries.first { $0.layer == coordinate.layer && $0.expert == coordinate.expert }
    }
    
    private func condensedExpertCard(
        _ entry: ExpertAtlasEntry,
        metricScales: [Int: ExpertAtlasMetricScale]
    ) -> some View {
        let selected = vm.selectedExpert?.layer == entry.layer && vm.selectedExpert?.expert == entry.expert
        let masked = vm.isMasked(layer: entry.layer, expert: entry.expert)
        let drop = vm.isDropCandidate(layer: entry.layer, expert: entry.expert)
        let locked = vm.isLockedKeep(layer: entry.layer, expert: entry.expert)
        let safeDrop = vm.isSafeDropCandidate(layer: entry.layer, expert: entry.expert)
        let grouped = vm.isGroupSelected(layer: entry.layer, expert: entry.expert)
        let belowThreshold = Double(entry.hits) < condensedAtlasHitsThreshold
        let effectiveOpacity = belowThreshold ? 0.15 : 1.0
        
        let cardWidth: CGFloat = 52
        let cardHeight: CGFloat = 52
        
        return Button {
            vm.select(entry)
        } label: {
            VStack(spacing: 0) {
                // Card body with domain color
                ZStack {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(cellColor(entry: entry, masked: masked, drop: drop, locked: locked, metricScales: metricScales))
                        .frame(width: cardWidth, height: cardHeight)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(
                                    selected ? Color.white : 
                                    grouped ? ExpertLabVisual.warm :
                                    locked ? ExpertLabVisual.good :
                                    masked ? ExpertLabVisual.danger :
                                    drop ? ExpertLabVisual.danger :
                                    safeDrop ? ExpertLabVisual.good :
                                    domainColor(for: entry).opacity(0.45),
                                    lineWidth: selected ? 2 : 1
                                )
                        )
                    
                    // Expert ID in top-left
                    Text("E\(entry.expert)")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(Color.white.opacity(0.85))
                        .shadow(color: Color.black.opacity(0.7), radius: 1)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding(4)
                    
                    // Hit count in bottom-right
                    Text("\(entry.hits)")
                        .font(.system(size: 7, weight: .regular, design: .monospaced))
                        .foregroundStyle(Color.white.opacity(0.75))
                        .shadow(color: Color.black.opacity(0.7), radius: 1)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                        .padding(4)
                    
                    // Activity bar at bottom
                    Rectangle()
                        .fill(Color.white.opacity(0.55))
                        .frame(width: cardWidth * CGFloat(min(entry.hits, 80)) / 80.0, height: 3)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                    
                    // Status indicators
                    if masked {
                        Image(systemName: "slash")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(Color.white)
                    } else if locked {
                        Circle()
                            .fill(ExpertLabVisual.good)
                            .frame(width: 6, height: 6)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                            .padding(4)
                    } else if safeDrop {
                        Circle()
                            .fill(ExpertLabVisual.good)
                            .frame(width: 6, height: 6)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                            .padding(4)
                    } else if drop {
                        Circle()
                            .fill(ExpertLabVisual.danger)
                            .frame(width: 6, height: 6)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                            .padding(4)
                    }
                }
            }
        }
        .buttonStyle(.plain)
        .opacity(effectiveOpacity)
        .accessibilityLabel("Layer \(entry.layer), expert \(entry.expert): \(vm.displayLabel(for: entry))")
        .accessibilityValue("\(entry.hits) hits\(masked ? ", temporarily disabled" : "")\(drop ? ", drop candidate" : "")\(locked ? ", locked keep" : "")\(safeDrop ? ", same-suite safe-drop" : "")")
        .help("Layer \(entry.layer), expert \(entry.expert): \(vm.displayLabel(for: entry)), \(entry.hits) hits\(safeDrop ? ", same-suite safe-drop" : "")")
        .background(
            GeometryReader { cardGeo in
                Color.clear
                    .onChange(of: hoveredCoordinate?.layer == entry.layer && hoveredCoordinate?.expert == entry.expert) { _, isHovered in
                        if isHovered {
                            hoverLocation = cardGeo.frame(in: .named("condensedCardsScroll")).origin
                        }
                    }
            }
        )
        .onHover { hovering in
            if hovering {
                hoveredCoordinate = ExpertCoordinate(layer: entry.layer, expert: entry.expert)
                hoverLocation = nil
                hoveredEntry = entry
            } else {
                hoveredCoordinate = nil
                hoverLocation = nil
                hoveredEntry = nil
            }
        }
        .simultaneousGesture(
            TapGesture(count: 2).onEnded {
                vm.setMasked(!vm.isMasked(layer: entry.layer, expert: entry.expert), layer: entry.layer, expert: entry.expert)
            }
        )
    }

    private var atlasLegendPopover: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "paintpalette")
                    .foregroundStyle(ExpertLabVisual.accent)
                Text("Semantic Color Legend")
                    .font(.system(size: 12, weight: .semibold))
                Spacer(minLength: 0)
                Text("BF16/vMLX")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(ExpertLabVisual.textFaint)
            }

            Text("Colors reflect the dominant evidence-backed semantic domain for each expert. The required review domains are listed below.")
                .font(.system(size: 10))
                .foregroundStyle(ExpertLabVisual.textDim)
                .fixedSize(horizontal: false, vertical: true)

            LazyVGrid(
                columns: [
                    GridItem(.adaptive(minimum: 150, maximum: 190), alignment: .leading)
                ],
                alignment: .leading,
                spacing: 7
            ) {
                ForEach(atlasLegendDomains, id: \.self) { domain in
                    atlasLegendRow(domain)
                }
            }
        }
        .padding(12)
        .frame(width: 410)
        .background(ExpertLabVisual.panel)
    }

    private var atlasLegendDomains: [String] {
        ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .sorted { ExpertDomainTaxonomy.displayName(for: $0) < ExpertDomainTaxonomy.displayName(for: $1) }
    }

    private func atlasLegendRow(_ domain: String) -> some View {
        HStack(spacing: 7) {
            Circle()
                .fill(colorForDomain(domain))
                .frame(width: 9, height: 9)
                .overlay {
                    Circle()
                        .stroke(Color.white.opacity(0.18), lineWidth: 1)
                }
            Text(ExpertDomainTaxonomy.displayName(for: domain))
                .font(.system(size: 10))
                .foregroundStyle(ExpertLabVisual.textDim)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .help(domain)
    }

    private var quickProbeBar: some View {
        HStack(spacing: 4) {
            ZStack(alignment: .topLeading) {
                TextEditor(text: $vm.livePromptText)
                    .font(.system(size: 10))
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                if vm.livePromptText.isEmpty {
                    Text("Type a prompt to see which experts light up...")
                        .font(.system(size: 10))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .allowsHitTesting(false)
                }
            }
            .frame(height: 26)
            .background(Color.black.opacity(0.20))
            .overlay {
                RoundedRectangle(cornerRadius: 4)
                    .stroke(ExpertLabVisual.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
            Button {
                Task { await vm.runLivePrompt() }
            } label: {
                Label("Probe", systemImage: "bolt.fill")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(ExpertLabCompactButtonStyle(primary: true))
            .font(.system(size: 10, weight: .semibold))
            .disabled(!vm.canRunLivePrompt)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(height: 1)
        }
    }

    private var livePromptOutputPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                ExpertLabSectionHeader(title: "Prompt Output", systemImage: "text.bubble")
                if !vm.lastEvalSummary.isEmpty {
                    Text(vm.lastEvalSummary)
                        .font(.system(size: 9))
                        .foregroundStyle(ExpertLabVisual.textDim)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }

            HStack(alignment: .top, spacing: 8) {
                livePromptOutputColumn(
                    title: "Original response",
                    text: vm.baselineText,
                    isReady: vm.baselineOutputReady,
                    emptyText: "Probe is running...",
                    emptyResultText: "Model returned an empty response.",
                    accent: ExpertLabVisual.accent
                )
                livePromptOutputColumn(
                    title: vm.hasMask ? "With disabled experts" : "Disabled experts",
                    text: vm.maskedText,
                    isReady: vm.maskedOutputReady,
                    emptyText: vm.hasMask
                        ? (vm.isRunning && vm.baselineOutputReady ? "Masked probe is running..." : "Run Probe to see the masked response.")
                        : "Disable experts, then run Probe again.",
                    emptyResultText: "Masked model returned an empty response.",
                    accent: ExpertLabVisual.warm
                )
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(height: 1)
        }
    }

    private func livePromptOutputColumn(
        title: String,
        text: String,
        isReady: Bool,
        emptyText: String,
        emptyResultText: String,
        accent: Color
    ) -> some View {
        let displayText = text.isEmpty ? (isReady ? emptyResultText : emptyText) : text
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 5) {
                Circle()
                    .fill(accent)
                    .frame(width: 5, height: 5)
                Text(title)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(ExpertLabVisual.textDim)
                Spacer(minLength: 0)
            }
            ScrollView {
                Text(displayText)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(text.isEmpty ? ExpertLabVisual.textFaint : .primary)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .textSelection(.enabled)
                    .padding(7)
            }
            .frame(height: 76)
            .background(Color.black.opacity(0.18))
            .overlay {
                RoundedRectangle(cornerRadius: 5)
                    .stroke(ExpertLabVisual.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 5))
        }
        .frame(maxWidth: .infinity)
    }

    private func atlasFilterButton(_ filter: ExpertAtlasFilter) -> some View {
        Button {
            vm.atlasFilter = filter
        } label: {
            Text(filter.title)
                .font(.system(size: 9, weight: vm.atlasFilter == filter ? .semibold : .regular))
                .foregroundStyle(vm.atlasFilter == filter ? ExpertLabVisual.accent : ExpertLabVisual.textFaint)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(vm.atlasFilter == filter ? ExpertLabVisual.accent.opacity(0.08) : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 3))
        }
        .buttonStyle(.plain)
    }

    private var atlasQueryBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                atlasFilterField("Layer", text: $debouncedLayerFilterText)
                    .frame(width: 88)
                atlasFilterField("Expert", text: $debouncedExpertFilterText)
                    .frame(width: 96)
                atlasFilterField("Domain", text: $debouncedDomainFilterText)
                    .frame(width: 116)
                atlasFilterField("Prompt", text: $debouncedPromptFilterText)
                    .frame(width: 132)
                Picker("Sort", selection: $vm.atlasSort) {
                    ForEach(ExpertAtlasSort.allCases) { sort in
                        Text(sort.title).tag(sort)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 132)
                Picker("Metric", selection: $vm.atlasMetric) {
                    ForEach(ExpertAtlasMetric.allCases) { metric in
                        Text(metric.title).tag(metric)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 126)
                Spacer(minLength: 0)
                if vm.hasAtlasQuery {
                    Button {
                        vm.clearAtlasQuery()
                        debouncedLayerFilterText = ""
                        debouncedExpertFilterText = ""
                        debouncedDomainFilterText = ""
                        debouncedPromptFilterText = ""
                    } label: {
                        Label("Clear filters", systemImage: "xmark.circle")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Clear layer, expert, domain, and prompt filters")
                }
            }
            .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
        }
        .font(.system(size: 10))
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(height: 1)
        }
        // M221: debounce atlas filter text changes so the grid doesn't re-filter
        // and re-render on every keystroke. 150 ms is a good balance for
        // responsive-feeling input without thrashing the main thread.
        // All four filters coalesce into one tick — the task reads the current
        // @State values when it fires, so rapid typing on any field batches
        // into a single ViewModel update.
        .onChange(of: debouncedLayerFilterText) { _, _ in debounceAtlasFilters() }
        .onChange(of: debouncedExpertFilterText) { _, _ in debounceAtlasFilters() }
        .onChange(of: debouncedDomainFilterText) { _, _ in debounceAtlasFilters() }
        .onChange(of: debouncedPromptFilterText) { _, _ in debounceAtlasFilters() }
    }

    /// M221: coalesce atlas filter debounce — cancels any pending tick and
    /// schedules a new 150 ms batch. When the timer fires, all four @State
    /// values are read at once and written to the ViewModel in a single pass,
    /// triggering one filteredAtlasEntries → atlasGridRows re-evaluation.
    ///
    /// Empty-string updates (clear button or backspace-to-empty on any/all fields)
    /// skip the delay and apply immediately so the grid doesn't hang on stale
    /// results for 150 ms after the user finishes clearing.
    private func debounceAtlasFilters() {
        atlasFilterDebounceTask?.cancel()
        let layer = debouncedLayerFilterText
        let expert = debouncedExpertFilterText
        let domain = debouncedDomainFilterText
        let prompt = debouncedPromptFilterText
        let allEmpty = layer.isEmpty && expert.isEmpty && domain.isEmpty && prompt.isEmpty
        if allEmpty {
            vm.atlasLayerFilterText = ""
            vm.atlasExpertFilterText = ""
            vm.atlasDomainFilterText = ""
            vm.atlasPromptFilterText = ""
            return
        }
        atlasFilterDebounceTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }
            vm.atlasLayerFilterText = layer
            vm.atlasExpertFilterText = expert
            vm.atlasDomainFilterText = domain
            vm.atlasPromptFilterText = prompt
        }
    }

    private var atlasTablePanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                ExpertLabSectionHeader(title: "Expert Table", systemImage: "tablecells")
                Spacer(minLength: 0)
                Text("\(vm.atlasTableEntries.count) rows")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(ExpertLabVisual.textFaint)
            }
            .padding(.horizontal, 12)
            .padding(.top, 6)
            .padding(.bottom, 3)

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 0) {
                        atlasTableHeader("Expert", width: 64)
                        atlasTableHeader("Label", width: 134)
                        atlasTableHeader("Domain", width: 116)
                        atlasTableHeader("Hits", width: 48)
                        atlasTableHeader("Mass", width: 56)
                        atlasTableHeader("Rank", width: 48)
                        atlasTableHeader("Depth", width: 92)
                        atlasTableHeader("Coactive", width: 104)
                        atlasTableHeader("Evidence", width: 64)
                        atlasTableHeader("Conf", width: 52)
                        atlasTableHeader("Annot", width: 64)
                        atlasTableHeader("Compared", width: 156)
                        atlasTableHeader("Impact", width: 168)
                        atlasTableHeader("Severity", width: 84)
                        atlasTableHeader("Review", width: 150)
                    }
                    .padding(.horizontal, 12)

                    ScrollView(.vertical, showsIndicators: true) {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(Array(vm.atlasTableEntries.prefix(atlasTableDisplayLimit))) { entry in
                                atlasTableRow(entry)
                            }
                        }
                    }
                    .frame(height: 118)
                }
            }
        }
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(height: 1)
        }
    }

    private func atlasTableHeader(_ title: String, width: CGFloat) -> some View {
        Text(title)
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(ExpertLabVisual.textFaint)
            .frame(width: width, alignment: .leading)
            .lineLimit(1)
    }

    private func atlasTableCell(_ text: String, width: CGFloat, accent: Color? = nil) -> some View {
        Text(text)
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(accent ?? ExpertLabVisual.textDim)
            .frame(width: width, alignment: .leading)
            .lineLimit(1)
            .truncationMode(.tail)
    }

    private func atlasTableRow(_ entry: ExpertAtlasEntry) -> some View {
        let selected = vm.selectedEntry?.layer == entry.layer && vm.selectedEntry?.expert == entry.expert
        return Button {
            vm.select(entry)
        } label: {
            HStack(spacing: 0) {
                atlasTableCell("L\(entry.layer) E\(entry.expert)", width: 64, accent: selected ? ExpertLabVisual.accent : nil)
                atlasTableCell(vm.displayLabel(for: entry), width: 134)
                atlasTableCell(vm.dominantDomainSummary(for: entry), width: 116)
                atlasTableCell("\(entry.hits)", width: 48)
                atlasTableCell(String(format: "%.3f", entry.probabilityMass), width: 56)
                atlasTableCell(String(format: "%.2f", entry.meanSelectedRank), width: 48)
                atlasTableCell(vm.tokenDepthSummary(for: entry), width: 92)
                atlasTableCell(vm.coactivationSummary(for: entry), width: 104)
                atlasTableCell("\(vm.evidenceCount(for: entry))", width: 64)
                atlasTableCell(String(format: "%.2f", entry.confidenceScore), width: 52)
                atlasTableCell(vm.manualAnnotationSummary(for: entry), width: 64)
                atlasTableCell(vm.selectedExpertComparedMaskStatus(for: entry), width: 156)
                atlasTableCell(
                    vm.selectedExpertMaskedImpactSummary(for: entry),
                    width: 168
                )
                atlasTableCell(
                    vm.selectedExpertRegressionSeveritySummary(for: entry),
                    width: 84,
                    accent: vm.selectedExpertRegressionSeverityIsHigh(for: entry) ? ExpertLabVisual.danger : nil
                )
                atlasTableCell(vm.selectedExpertReviewStatus(for: entry), width: 150)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 2)
            .background(selected ? ExpertLabVisual.accent.opacity(0.10) : Color.clear)
        }
        .buttonStyle(.plain)
    }

    private func atlasFilterField(_ title: String, text: Binding<String>) -> some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(ExpertLabVisual.textDim)
            TextField("", text: text)
                .font(.system(size: 10, design: .monospaced))
                .textFieldStyle(.plain)
                .lineLimit(1)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Color.black.opacity(0.18))
        .overlay {
            RoundedRectangle(cornerRadius: 4)
                .stroke(ExpertLabVisual.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private var emptyAtlasPlaceholder: some View {
        VStack {
            Spacer()
            VStack(spacing: 10) {
                Image(systemName: vm.isRunning ? "waveform.path.ecg" : "point.3.connected.trianglepath.dotted")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(ExpertLabVisual.accent)
                Text(vm.isRunning ? "Generating Expert Map" : "Run Prompts to Generate the Expert Map")
                    .font(.system(size: 13, weight: .semibold))
                Text(emptyAtlasMessage)
                    .font(.system(size: 10))
                    .foregroundStyle(ExpertLabVisual.textDim)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                if vm.capability.isTraceSupported {
                    Button {
                        Task { await vm.runTrace() }
                    } label: {
                        Label(reviewMode ? "Run Expert Review" : "Run Trace", systemImage: "waveform.path.ecg")
                    }
                    .buttonStyle(ExpertLabCompactButtonStyle(primary: true))
                    .disabled(vm.isRunning)
                }
            }
            .frame(maxWidth: 440)
            .padding(.horizontal, 16)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(ExpertLabVisual.canvas)
    }

    private var emptyAtlasMessage: String {
        if vm.isRunning {
            return "Prompt-suite probing is collecting routed expert activity. The atlas will appear here when the run finishes."
        }
        if vm.capability.isTraceSupported {
            return "Choose a prompt suite, then run the prompts. JANG Studio will trace routing, label active experts, and build the layer-by-expert atlas in this space."
        }
        return vm.capability.detail
    }

    private var currentLassoRect: CGRect? {
        guard let lassoStart, let lassoEnd else { return nil }
        return CGRect(
            x: min(lassoStart.x, lassoEnd.x),
            y: min(lassoStart.y, lassoEnd.y),
            width: abs(lassoStart.x - lassoEnd.x),
            height: abs(lassoStart.y - lassoEnd.y)
        )
    }

    private func previewLasso(rect: CGRect, rows: [ExpertAtlasLayerRow]) {
        var coordinates = Set<ExpertCoordinate>()
        let firstCellX = atlasGridPadding + atlasGridLayerLabelWidth + atlasGridCellSpacing
        for (rowIndex, row) in rows.enumerated() {
            let y = atlasGridPadding + CGFloat(rowIndex) * (atlasGridCellSize + atlasGridRowSpacing)
            for (columnIndex, entry) in row.entries.enumerated() {
                let x = firstCellX + CGFloat(columnIndex) * (atlasGridCellSize + atlasGridCellSpacing)
                let cellRect = CGRect(x: x, y: y, width: atlasGridCellSize, height: atlasGridCellSize)
                if rect.intersects(cellRect) {
                    coordinates.insert(ExpertCoordinate(layer: entry.layer, expert: entry.expert))
                }
            }
        }
        vm.groupSelection = coordinates
    }

    private func expertCell(
        _ entry: ExpertAtlasEntry,
        metricScales: [Int: ExpertAtlasMetricScale]
    ) -> some View {
        let selected = vm.selectedExpert?.layer == entry.layer && vm.selectedExpert?.expert == entry.expert
        let masked = vm.isMasked(layer: entry.layer, expert: entry.expert)
        let drop = vm.isDropCandidate(layer: entry.layer, expert: entry.expert)
        let locked = vm.isLockedKeep(layer: entry.layer, expert: entry.expert)
        let safeDrop = vm.isSafeDropCandidate(layer: entry.layer, expert: entry.expert)
        let grouped = vm.isGroupSelected(layer: entry.layer, expert: entry.expert)
        return Button {
            vm.select(entry)
        } label: {
            let borderColor = selected ? Color.white :
                grouped ? ExpertLabVisual.warm :
                locked ? ExpertLabVisual.good :
                masked ? ExpertLabVisual.danger :
                drop ? ExpertLabVisual.danger :
                safeDrop ? ExpertLabVisual.good :
                domainColor(for: entry).opacity(0.45)
            let borderWidth: CGFloat = selected || grouped || locked || masked || drop || safeDrop ? 1.6 : 1
            RoundedRectangle(cornerRadius: 3)
                .fill(cellColor(entry: entry, masked: masked, drop: drop, locked: locked, metricScales: metricScales))
                .frame(width: atlasGridCellSize, height: atlasGridCellSize)
                .overlay {
                    let dash: [CGFloat] = drop ? [4, 2] : (safeDrop ? [2, 2] : [])
                    RoundedRectangle(cornerRadius: 3)
                        .stroke(borderColor, style: StrokeStyle(lineWidth: borderWidth, dash: dash))
                }
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(domainColor(for: entry).opacity(entry.isDead ? 0.28 : 0.95))
                        .frame(height: entry.isDead ? 2 : 3)
                }
                .overlay(alignment: .topTrailing) {
                    if entry.isHot {
                        Circle()
                            .fill(Color.white.opacity(0.82))
                            .frame(width: 4, height: 4)
                            .padding(3)
                    }
                }
                .overlay(alignment: .topLeading) {
                    if safeDrop || locked {
                        Circle()
                            .fill(safeDrop ? ExpertLabVisual.good : ExpertLabVisual.warm)
                            .frame(width: 4, height: 4)
                            .padding(3)
                    }
                }
                .shadow(color: selected ? Color.white.opacity(0.25) : .clear, radius: selected ? 4 : 0)
                .overlay {
                    if masked {
                        Image(systemName: "slash")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.white)
                    }
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Layer \(entry.layer), expert \(entry.expert): \(vm.displayLabel(for: entry))")
        .accessibilityValue("\(entry.hits) hits\(masked ? ", temporarily disabled" : "")\(drop ? ", drop candidate" : "")\(locked ? ", locked keep" : "")\(safeDrop ? ", same-suite safe-drop" : "")")
        .help("Layer \(entry.layer), expert \(entry.expert): \(vm.displayLabel(for: entry)), \(entry.hits) hits\(safeDrop ? ", same-suite safe-drop" : "")")
    }

    private func cellColor(
        entry: ExpertAtlasEntry,
        masked: Bool,
        drop: Bool,
        locked: Bool,
        metricScales: [Int: ExpertAtlasMetricScale]
    ) -> Color {
        if entry.isDead { return Color(red: 44 / 255, green: 50 / 255, blue: 54 / 255).opacity(0.78) }
        if masked || drop { return ExpertLabVisual.danger.opacity(0.72) }
        if locked { return ExpertLabVisual.good.opacity(0.72) }
        switch condensedAtlasColorMode {
        case .domain:
            let intensity = vm.metricIntensity(entry, scales: metricScales)
            let base = domainColor(for: entry)
            let opacity = (entry.isHot ? 0.36 : 0.20) + intensity * (entry.isHot ? 0.62 : 0.55)
            return base.opacity(opacity)
        case .frequency:
            let maxHits = max(metricScales[entry.layer]?.maxHits ?? 1, 1)
            let t = min(1.0, Double(entry.hits) / Double(maxHits))
            return Color(red: 0, green: 0.75 * (0.15 + 0.85 * t), blue: 0.85 * (0.15 + 0.85 * t))
        case .dropRisk:
            let risk = condensedCardDropRisk(for: entry)
            return risk.color.opacity(0.72)
        }
    }

    private func domainColor(for entry: ExpertAtlasEntry) -> Color {
        if let domain = ExpertDomainTaxonomy.dominantDomain(domains: entry.domains, domainLift: entry.domainLift) {
            return colorForDomain(domain)
        }
        return colorForDomain(entry.generatedLabel)
    }

    private func colorForDomain(_ raw: String) -> Color {
        switch ExpertDomainTaxonomy.canonicalSemanticDomain(raw) {
        case "code":
            return Color(red: 79 / 255, green: 195 / 255, blue: 247 / 255)     // blue
        case "formatting":
            return Color(red: 77 / 255, green: 208 / 255, blue: 225 / 255)     // cyan
        case "instruction_following":
            return Color(red: 129 / 255, green: 199 / 255, blue: 132 / 255)    // green
        case "math":
            return Color(red: 171 / 255, green: 71 / 255, blue: 188 / 255)     // purple
        case "reasoning":
            return Color(red: 102 / 255, green: 187 / 255, blue: 106 / 255)    // green
        case "chinese":
            return Color(red: 255 / 255, green: 183 / 255, blue: 77 / 255)     // amber
        case "non_english":
            return Color(red: 255 / 255, green: 138 / 255, blue: 101 / 255)    // orange
        case "multilingual":
            return Color(red: 38 / 255, green: 166 / 255, blue: 154 / 255)     // teal
        case "translation":
            return Color(red: 149 / 255, green: 117 / 255, blue: 205 / 255)    // indigo
        case "english_dominant":
            return Color(red: 128 / 255, green: 222 / 255, blue: 234 / 255)    // light cyan
        case "unknown_language_role":
            return Color(red: 174 / 255, green: 213 / 255, blue: 129 / 255)    // lime
        case "safety_medical_legal_sensitive", "safety_sensitive", "medical_sensitive", "legal_sensitive":
            return Color(red: 239 / 255, green: 83 / 255, blue: 80 / 255)      // red
        default:
            break
        }
        let cleaned = raw
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned == "manual" || cleaned == "user-label" {
            return ExpertLabVisual.accent                                      // bright cyan
        }
        let candidates = [cleaned] + cleaned.split(separator: "-").map(String.init)
        let domain = candidates
            .map(ExpertDomainTaxonomy.canonicalDomain)
            .first { $0 != "general" } ?? ExpertDomainTaxonomy.canonicalDomain(cleaned)
        switch domain {
        case "coding":
            return Color(red: 79 / 255, green: 195 / 255, blue: 247 / 255)     // blue
        case "math":
            return Color(red: 171 / 255, green: 71 / 255, blue: 188 / 255)     // purple
        case "reasoning":
            return Color(red: 102 / 255, green: 187 / 255, blue: 106 / 255)    // green
        case "language":
            return Color(red: 255 / 255, green: 138 / 255, blue: 101 / 255)    // orange
        case "safety":
            return Color(red: 239 / 255, green: 83 / 255, blue: 80 / 255)      // red
        case "creative":
            return Color(red: 206 / 255, green: 147 / 255, blue: 216 / 255)    // magenta
        case "knowledge":
            return Color(red: 141 / 255, green: 110 / 255, blue: 99 / 255)     // brown
        case "tools":
            return Color(red: 38 / 255, green: 198 / 255, blue: 218 / 255)     // teal
        case "general":
            return Color(red: 128 / 255, green: 222 / 255, blue: 234 / 255)    // light cyan
        case "manual", "user-label":
            return ExpertLabVisual.accent                                      // bright cyan
        default:
            // For unrecognized labels, use a deterministic HSL-based color
            // so every expert isn't the same blue-gray
            return hashedColor(cleaned)
        }
    }

    private func hashedColor(_ input: String) -> Color {
        // Deterministic color from string hash — gives visual diversity
        var hash = 5381
        for byte in input.utf8 {
            hash = ((hash &<< 5) &+ hash) &+ Int(byte)
        }
        let hue = Double(abs(hash) % 360) / 360.0
        let sat = 0.35 + Double(abs(hash * 7) % 40) / 100.0
        let bri = 0.45 + Double(abs(hash * 13) % 30) / 100.0
        return Color(hue: hue, saturation: sat, brightness: bri).opacity(0.82)
    }

    private func colorModeRadio(_ mode: CondensedAtlasColorMode) -> some View {
        Button {
            condensedAtlasColorMode = mode
        } label: {
            HStack(spacing: 6) {
                Image(systemName: condensedAtlasColorMode == mode ? "record.circle" : "circle")
                    .font(.system(size: 10))
                    .foregroundStyle(condensedAtlasColorMode == mode ? ExpertLabVisual.accent : ExpertLabVisual.textFaint)
                Text(mode.title)
                    .font(.system(size: 10))
                    .foregroundStyle(condensedAtlasColorMode == mode ? Color.primary : ExpertLabVisual.textDim)
                Spacer(minLength: 0)
            }
        }
        .buttonStyle(.plain)
    }

    private var rightPanel: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 12) {
                let displayedEntry = vm.selectedEntry ?? hoveredEntry
                if let entry = displayedEntry {
                    ExpertLabSubPanel(title: vm.selectedEntry != nil ? "Expert" : "Preview", systemImage: vm.selectedEntry != nil ? "scope" : "eye") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("L\(entry.layer) / E\(entry.expert)")
                                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                                Spacer()
                                ExpertLabBadge(text: vm.displayLabel(for: entry), color: entry.isDead ? ExpertLabVisual.textFaint : domainColor(for: entry))
                            }
                            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                                GridRow {
                                    ExpertLabStatRow(label: "Hits", value: "\(entry.hits)")
                                    ExpertLabStatRow(label: "Freq", value: String(format: "%.3f", entry.activationFrequency))
                                }
                                GridRow {
                                    ExpertLabStatRow(label: "Mass", value: String(format: "%.3f", entry.probabilityMass))
                                    ExpertLabStatRow(label: "Rank", value: String(format: "%.2f", entry.meanSelectedRank))
                                }
                                GridRow {
                                    ExpertLabStatRow(label: "Tokens", value: "\(entry.tokenCount)")
                                    ExpertLabStatRow(label: "Evidence", value: "\(vm.evidenceCount(for: entry))")
                                }
                                GridRow {
                                    ExpertLabStatRow(label: "Conf", value: String(format: "%.2f", entry.confidenceScore))
                                    ExpertLabStatRow(label: "Lift", value: String(format: "%.1fx", entry.domainLift.values.max() ?? 0))
                                }
                                GridRow {
                                    ExpertLabStatRow(label: "Depth", value: vm.tokenDepthSummary(for: entry))
                                    ExpertLabStatRow(label: "Entropy", value: String(format: "%.2f", entry.entropyContribution))
                                }
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                TextField(
                                    "User label",
                                    text: Binding(
                                        get: { vm.userLabel(layer: entry.layer, expert: entry.expert) },
                                        set: { vm.setUserLabel($0, layer: entry.layer, expert: entry.expert) }
                                    )
                                )
                                .font(.caption)
                                TextEditor(
                                    text: Binding(
                                        get: { vm.userNotes(layer: entry.layer, expert: entry.expert) },
                                        set: { vm.setUserNotes($0, layer: entry.layer, expert: entry.expert) }
                                    )
                                )
                                .font(.caption)
                                .frame(height: 48)
                                .scrollContentBackground(.hidden)
                                .background(ExpertLabVisual.panelRaised.opacity(0.7))
                                .clipShape(RoundedRectangle(cornerRadius: 5))
                                Text("Generated: \(entry.generatedLabel)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                            Toggle(
                                "Temporarily disable",
                                isOn: Binding(
                                    get: { vm.isMasked(layer: entry.layer, expert: entry.expert) },
                                    set: { vm.setMasked($0, layer: entry.layer, expert: entry.expert) }
                                )
                            )
                            .font(.caption)
                            Stepper(value: $vm.topKOverride, in: 0...(vm.capability.trainedTopK ?? 16), step: 1) {
                                LabeledContent("Top-K override", value: vm.topKOverride == 0 ? "trained" : "\(vm.topKOverride)")
                            }
                            .font(.caption)
                            .help("0 keeps the model's trained router top-k. Nonzero values can only lower K.")
                            .onChange(of: vm.topKOverride) { _, _ in
                                vm.maskStateChanged()
                            }
                            HStack(spacing: 12) {
                                Toggle(
                                    "Drop candidate",
                                    isOn: Binding(
                                        get: { vm.isDropCandidate(layer: entry.layer, expert: entry.expert) },
                                        set: { vm.setDropCandidate($0, layer: entry.layer, expert: entry.expert) }
                                    )
                                )
                                Toggle(
                                    "Locked keep",
                                    isOn: Binding(
                                        get: { vm.isLockedKeep(layer: entry.layer, expert: entry.expert) },
                                        set: { vm.setLockedKeep($0, layer: entry.layer, expert: entry.expert) }
                                    )
                                )
                            }
                            .font(.caption)
                            Divider()
                            ExpertLabSectionHeader(title: "Comparison Impact", systemImage: "chart.line.uptrend.xyaxis")
                            VStack(alignment: .leading, spacing: 3) {
                                expertImpactLine("Review", value: vm.selectedExpertReviewStatus(for: entry))
                                expertImpactLine("Compared", value: vm.selectedExpertComparedMaskStatus(for: entry))
                                expertImpactLine("Safe-drop", value: vm.selectedExpertSafeDropStatus(for: entry))
                                expertImpactLine("Impact", value: vm.selectedExpertMaskedImpactSummary(for: entry))
                                expertImpactLine(
                                    "Severity",
                                    value: vm.selectedExpertRegressionSeveritySummary(for: entry),
                                    accent: vm.selectedExpertRegressionSeverityIsHigh(for: entry) ? ExpertLabVisual.danger : ExpertLabVisual.textDim
                                )
                                expertImpactLine("Pass rate", value: vm.selectedExpertPassRateSummary(for: entry))
                                expertImpactLine(
                                    "Risk",
                                    value: vm.selectedExpertRiskSummary(for: entry),
                                    accent: vm.selectedExpertHasComparisonRisk ? ExpertLabVisual.danger : ExpertLabVisual.textDim
                                )
                                Text("Mask-level evidence, not single-expert ablation.")
                                    .font(.caption2)
                                    .foregroundStyle(ExpertLabVisual.textFaint)
                                    .lineLimit(2)
                            }
                            if !entry.domains.isEmpty {
                                Divider()
                                ExpertLabSectionHeader(title: "Domains")
                                ForEach(entry.domains.sorted(by: { $0.value > $1.value }), id: \.key) { domain, count in
                                    HStack {
                                        Text(ExpertDomainTaxonomy.displayName(for: domain))
                                        Spacer()
                                        let lift = entry.domainLift[domain] ?? 0
                                        Text("\(count) · \(String(format: "%.1fx", lift))")
                                            .monospacedDigit()
                                            .foregroundStyle(.secondary)
                                    }
                                    .font(.caption)
                                }
                            }
                            let promptEvidence = entry.promptEvidence ?? []
                            if !promptEvidence.isEmpty {
                                Divider()
                                ExpertLabSectionHeader(title: "Prompt Evidence")
                                ForEach(promptEvidence) { evidence in
                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack {
                                            Text(evidence.promptID)
                                                .lineLimit(1)
                                            Spacer()
                                            Text("\(evidence.hits) hits")
                                                .monospacedDigit()
                                                .foregroundStyle(.secondary)
                                        }
                                        Text(
                                            ([ExpertDomainTaxonomy.displayName(for: evidence.domain)] + evidence.tags)
                                                .joined(separator: " · ")
                                        )
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                        Text(evidence.promptExcerpt)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(2)
                                    }
                                    .font(.caption)
                                }
                            } else if !entry.topPrompts.isEmpty {
                                Divider()
                                ExpertLabSectionHeader(title: "Prompt Evidence")
                                ForEach(entry.topPrompts, id: \.self) { promptID in
                                    Text(promptID)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            }
                            if !entry.coactivationNeighbors.isEmpty {
                                Divider()
                                ExpertLabSectionHeader(title: "Coactivation")
                                ForEach(entry.coactivationNeighbors, id: \.expert) { neighbor in
                                    HStack {
                                        Text("E\(neighbor.expert)")
                                        Spacer()
                                        Text("\(neighbor.count) · J \(String(format: "%.2f", neighbor.jaccard))")
                                            .monospacedDigit()
                                            .foregroundStyle(.secondary)
                                    }
                                    .font(.caption)
                                }
                            }
                        }
                    }
                } else {
                    Text("Hover or click an atlas cell\nto inspect an expert")
                        .font(.system(size: 11))
                        .foregroundStyle(ExpertLabVisual.textFaint)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity, minHeight: 160)
                }

                if shouldShowRightWorkflowControls {
                    VStack(alignment: .leading, spacing: 5) {
                    ExpertLabSubPanel(title: "Selection", systemImage: "rectangle.dashed") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(vm.groupSelectionSummary)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                            HStack {
                                Button { vm.applySelectionDropCandidates() } label: {
                                    Label("Drop", systemImage: "minus.circle")
                                }
                                .disabled(vm.groupSelectionCount == 0)
                                Button { vm.applySelectionLockedKeep() } label: {
                                    Label("Keep", systemImage: "lock.circle")
                                }
                                .disabled(vm.groupSelectionCount == 0)
                            }
                            .buttonStyle(.borderless)
                            .font(.caption)
                            Button { vm.clearGroupSelection() } label: {
                                Label("Clear", systemImage: "xmark.circle")
                            }
                            .buttonStyle(.borderless)
                            .font(.caption2)
                            .disabled(vm.groupSelectionCount == 0)
                        }
                    }
                    ExpertLabSubPanel(title: "Mask", systemImage: "slash.circle") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(vm.maskSummary)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(3)
                            HStack {
                                Button { vm.clearMask() } label: {
                                    Label("Clear", systemImage: "xmark.circle")
                                }
                                .disabled(!vm.hasMask)
                                Button { Task { await vm.runMaskedCompare() } } label: {
                                    Label("Compare", systemImage: "rectangle.split.2x1")
                                }
                                .buttonStyle(ExpertLabCompactButtonStyle(primary: true))
                                .disabled(!vm.capability.isTraceSupported || vm.isRunning || !vm.hasMask)
                            }
                            .font(.caption)
                            HStack {
                                Button { Task { await vm.runMaskedSuiteCompare() } } label: {
                                    Label("Suite", systemImage: "list.bullet.rectangle")
                                }
                                .buttonStyle(.borderless)
                                .font(.caption2)
                                .disabled(!vm.capability.isTraceSupported || vm.isRunning || !vm.hasMask || vm.selectedSuite.prompts.isEmpty)
                                Button { vm.saveMask() } label: {
                                    Label("Save", systemImage: "square.and.arrow.down")
                                }
                                .buttonStyle(.borderless)
                                .font(.caption2)
                                .disabled(!vm.canSaveMask)
                                Button { vm.importMask() } label: {
                                    Label("Load", systemImage: "square.and.arrow.up")
                                }
                                .buttonStyle(.borderless)
                                .font(.caption2)
                            }
                        }
                    }
                    if vm.hasComparisonEvidence {
                        ExpertLabSubPanel(
                            title: vm.comparisonRiskRows.isEmpty ? "Eval Evidence" : "Eval Regressions",
                            systemImage: vm.comparisonRiskRows.isEmpty ? "checkmark.seal" : "exclamationmark.triangle.fill",
                            accent: vm.comparisonRiskRows.isEmpty ? ExpertLabVisual.accent : ExpertLabVisual.danger
                        ) {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(vm.comparisonEvidenceSummary)
                                    .font(.caption2)
                                    .foregroundStyle(vm.comparisonRiskRows.isEmpty ? .secondary : ExpertLabVisual.danger)
                                    .lineLimit(3)
	                                ForEach(vm.comparisonRowsForDisplay.prefix(4), id: \.promptID) { row in
	                                    VStack(alignment: .leading, spacing: 1) {
	                                        HStack(spacing: 4) {
                                            Text(row.promptID)
                                                .font(.caption2.weight(.semibold))
                                                .lineLimit(1)
                                            Spacer(minLength: 0)
                                            Text(String(format: "%.2f", row.textDelta))
                                                .font(.caption2.monospacedDigit())
                                                .foregroundStyle(row.isRisky ? ExpertLabVisual.danger : .secondary)
                                        }
                                        Text("\(row.domain) · \(row.risk) · severity \(row.resolvedRegressionSeverity)")
                                            .font(.caption2)
                                            .foregroundStyle(row.isRisky ? ExpertLabVisual.danger : ExpertLabVisual.textDim)
	                                            .lineLimit(1)
	                                    }
	                                }
	                                if vm.comparisonRowsOmittedCount > 0 {
	                                    Text("\(vm.comparisonRowsOmittedCount) more per-prompt row\(vm.comparisonRowsOmittedCount == 1 ? "" : "s") saved in eval artifacts.")
	                                        .font(.caption2.weight(.semibold))
	                                        .foregroundStyle(vm.comparisonRiskRows.isEmpty ? .secondary : ExpertLabVisual.danger)
	                                        .lineLimit(2)
	                                }
	                                if !vm.comparisonArtifactSummary.isEmpty {
	                                    Text(vm.comparisonArtifactSummary)
	                                        .font(.caption2)
	                                        .foregroundStyle(.secondary)
	                                        .lineLimit(3)
	                                        .textSelection(.enabled)
	                                }
	                            }
	                        }
                    }
                }

                    ExpertLabSubPanel(title: "Hard Prune", systemImage: "scissors", accent: ExpertLabVisual.warm) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Run a prompt suite first, mark reviewed drop candidates or locked keeps, then generate a trace-informed BF16/F16 source prune plan.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Label(vm.reviewedPruneReadiness.message,
                                  systemImage: vm.reviewedPruneReadiness.systemImage)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(vm.reviewedPruneReadiness.isReady ? ExpertLabVisual.good : ExpertLabVisual.warm)
                                .lineLimit(4)
                                .textSelection(.enabled)
                            if vm.dropCandidateCount > 0 {
	                                Label("\(vm.dropCandidateCount) drop candidates will be forced into the reviewed drop set.",
	                                      systemImage: "pin.fill")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            if vm.lockedKeepCount > 0 {
                                Label("\(vm.lockedKeepCount) experts are locked keep.",
                                      systemImage: "lock.fill")
                                    .font(.caption)
                                    .foregroundStyle(.green)
                            }
                            ForEach(vm.planValidationIssues.prefix(3)) { issue in
                                Label(issue.message, systemImage: issue.severity == .error ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundStyle(issue.severity == .error ? .red : .orange)
                            }
                            if vm.atlas != nil && !vm.hasComparisonSummary {
                                Label("A/B comparison required before reviewed prune plan.",
                                      systemImage: "rectangle.split.2x1")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            ForEach(vm.reviewedPruneAtlasIssues.prefix(3), id: \.self) { issue in
                                Label(issue, systemImage: "square.grid.3x3")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            ForEach(vm.reviewedPruneCoverageIssues.prefix(3), id: \.self) { issue in
                                Label(issue, systemImage: "checklist.unchecked")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            ForEach(vm.reviewedPruneComparisonIssues.prefix(4), id: \.self) { issue in
                                Label(issue, systemImage: "rectangle.split.2x1")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            ForEach(vm.reviewedPruneSemanticEvidenceIssues.prefix(3), id: \.self) { issue in
                                Label(issue, systemImage: "tag.fill")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            if let experts = vm.capability.expectedExperts {
                                Stepper(value: $vm.pruneKeepExperts, in: vm.pruneKeepRange, step: 8) {
                                    LabeledContent("Keep per layer", value: "\(vm.pruneKeepExperts) / \(experts)")
                                }
                                .font(.caption)
                                .disabled(vm.atlas == nil)
                            }
                            Button {
                                vm.exportPrunePlan()
                            } label: {
                                Label("Export Reviewed Prune Plan", systemImage: "square.and.arrow.up")
                            }
                            .disabled(!vm.canGenerateReviewedPrunePlan)
                            if onPrunePlanReady != nil {
                                Button {
                                    usePlanForSourcePrune()
                                } label: {
                                    Label("Use Plan for BF16/F16 Prune", systemImage: "scissors")
                                }
                                .buttonStyle(ExpertLabCompactButtonStyle(primary: true, color: ExpertLabVisual.warm))
                                .disabled(!vm.canGenerateReviewedPrunePlan)
                            }
                        }
                    }
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(width: 1)
        }
                .frame(minWidth: 260, idealWidth: 280, maxWidth: 380)
    }

    private func expertImpactLine(_ label: String, value: String, accent: Color = ExpertLabVisual.textDim) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 58, alignment: .leading)
            Text(value)
                .foregroundStyle(accent)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption2)
    }

    private var compareTray: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 5) {
                ExpertLabSectionHeader(title: "Compare", systemImage: "rectangle.split.2x1")
                Text(vm.lastEvalSummary.isEmpty ? "Run a masked compare to populate this tray." : vm.lastEvalSummary)
                    .font(.system(size: 9))
                    .foregroundStyle(ExpertLabVisual.textDim)
                    .lineLimit(3)
                if vm.baselineTokensPerSecond > 0 || vm.maskedTokensPerSecond > 0 {
                    VStack(alignment: .leading, spacing: 2) {
                        ExpertLabStatusPill(
                            text: "base \(String(format: "%.1f", vm.baselineTokensPerSecond)) t/s",
                            color: ExpertLabVisual.accent
                        )
                        ExpertLabStatusPill(
                            text: "mask \(String(format: "%.1f", vm.maskedTokensPerSecond)) t/s",
                            color: ExpertLabVisual.warm
                        )
                    }
                }
                if !vm.runtimeInfoSummary.isEmpty {
                    Text(vm.runtimeInfoSummary)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(ExpertLabVisual.textDim)
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
            }
            .padding(6)
            .frame(width: 160)
            .frame(maxHeight: .infinity, alignment: .topLeading)
            .overlay(alignment: .trailing) {
                Rectangle()
                    .fill(ExpertLabVisual.line)
                    .frame(width: 1)
            }
            HStack(alignment: .top, spacing: 10) {
                compareColumn(
                    title: "Original",
                    text: vm.baselineText,
                    isReady: vm.baselineOutputReady,
                    emptyText: "Run Probe or Compare to see the original output.",
                    emptyResultText: "Model returned an empty response.",
                    accent: ExpertLabVisual.accent
                )
                compareColumn(
                    title: vm.hasMask ? "With disabled experts" : "Disabled experts",
                    text: vm.maskedText,
                    isReady: vm.maskedOutputReady,
                    emptyText: vm.hasMask
                        ? "Run Probe or Compare to see the masked output."
                        : "Disable experts, then run Probe again to compare.",
                    emptyResultText: "Masked model returned an empty response.",
                    accent: ExpertLabVisual.warm
                )
            }
            .padding(6)
        }
        .frame(height: 120)
        .background(ExpertLabVisual.panel)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(ExpertLabVisual.line)
                .frame(height: 1)
        }
    }

    private func compareColumn(
        title: String,
        text: String,
        isReady: Bool,
        emptyText: String = "Run a masked compare to populate this pane.",
        emptyResultText: String = "Model returned an empty response.",
        accent: Color = ExpertLabVisual.accent
    ) -> some View {
        let displayText = text.isEmpty ? (isReady ? emptyResultText : emptyText) : text
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Circle()
                    .fill(accent.opacity(0.72))
                    .frame(width: 5, height: 5)
                Text(title)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(ExpertLabVisual.textDim)
            }
            ScrollView(.vertical, showsIndicators: false) {
                Text(displayText)
                    .font(.system(size: 10))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .foregroundStyle(text.isEmpty ? ExpertLabVisual.textFaint : .primary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(7)
            .background(Color.black.opacity(0.15))
            .overlay {
                RoundedRectangle(cornerRadius: 5)
                    .stroke(ExpertLabVisual.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 5))
        }
    }

    private func usePlanForSourcePrune() {
        do {
            let url = try vm.writePrunePlan()
            onPrunePlanReady?(url)
            dismiss()
        } catch {
            vm.lastError = "Could not create prune plan: \(error.localizedDescription)"
        }
    }
}

private struct ExpertLabBadge: View {
    let text: String
    var color: Color = .secondary

    var body: some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(color.opacity(0.08))
            .foregroundStyle(color)
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .stroke(color.opacity(0.30), lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 3))
    }
}

private struct ExpertLabCompactButtonStyle: ButtonStyle {
    var primary = false
    var color: Color = ExpertLabVisual.accent

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 10, weight: primary ? .semibold : .regular))
            .foregroundStyle(primary ? color : .primary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .frame(minHeight: 22)
            .background(primary ? color.opacity(configuration.isPressed ? 0.18 : 0.10) : Color.white.opacity(configuration.isPressed ? 0.06 : 0.02))
            .overlay {
                RoundedRectangle(cornerRadius: 4)
                    .stroke(primary ? color.opacity(0.95) : ExpertLabVisual.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .opacity(configuration.isPressed ? 0.82 : 1)
    }
}

private struct SelectedExpert: Equatable {
    let layer: Int
    let expert: Int
}

private enum ExpertAtlasDisplayMode: String, CaseIterable, Identifiable {
    case map
    case condensedCards
    case table

    var id: String { rawValue }

    var title: String {
        switch self {
        case .map: "Map"
        case .condensedCards: "Cards"
        case .table: "Table"
        }
    }

    var systemImage: String {
        switch self {
        case .map: "square.grid.3x3"
        case .condensedCards: "rectangle.grid.2x2"
        case .table: "tablecells"
        }
    }
}

private struct ExpertAtlasLayerRow: Identifiable {
    let layer: Int
    let entries: [ExpertAtlasEntry]

    var id: Int { layer }
}

private struct ExpertAtlasMetricScale {
    let maxHits: Int
    let maxMass: Float
    let maxEntropy: Float
}

/// M224: color mode used only for the condensed-cards atlas view.
private enum CondensedAtlasColorMode: String, CaseIterable, Identifiable {
    case domain
    case frequency
    case dropRisk

    var id: String { rawValue }

    var title: String {
        switch self {
        case .domain: "Domain"
        case .frequency: "Frequency"
        case .dropRisk: "Drop risk"
        }
    }
}

private enum ExpertAtlasFilter: String, CaseIterable, Identifiable {
    case all
    case active
    case hot
    case dead
    case masked
    case drops
    case locked
    case safeDrop
    case safety
    case coding
    case math
    case reasoning

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "All"
        case .active: "Active"
        case .hot: "Hot"
        case .dead: "Dead"
        case .masked: "Masked"
        case .drops: "Drops"
        case .locked: "Locked"
        case .safeDrop: "Safe-drop"
        case .safety: "Safety"
        case .coding: "Coding"
        case .math: "Math"
        case .reasoning: "Reasoning"
        }
    }
}

private enum ExpertAtlasMetric: String, CaseIterable, Identifiable {
    case frequency
    case routerMass
    case domainLift
    case entropy
    case confidence

    var id: String { rawValue }

    var title: String {
        switch self {
        case .frequency: "Frequency"
        case .routerMass: "Router mass"
        case .domainLift: "Domain lift"
        case .entropy: "Entropy"
        case .confidence: "Confidence"
        }
    }
}

private enum ExpertAtlasSort: String, CaseIterable, Identifiable {
    case expertID
    case hits
    case routerMass
    case activationRank
    case tokenDepth
    case coactivation
    case domainLift
    case regressionSeverity
    case confidence
    case safeDrop

    var id: String { rawValue }

    var title: String {
        switch self {
        case .expertID: "Expert ID"
        case .hits: "Hits"
        case .routerMass: "Router mass"
        case .activationRank: "Activation rank"
        case .tokenDepth: "Token depth"
        case .coactivation: "Coactivation"
        case .domainLift: "Domain lift"
        case .regressionSeverity: "Regression severity"
        case .confidence: "Confidence"
        case .safeDrop: "Safe-drop"
        }
    }
}

private enum ExpertLabRuntimeMode: Equatable {
    case bf16VMLX
    case nativeJANGTQ
    case unsupported
}

private struct ExpertLabCapability: Equatable {
    let isTraceSupported: Bool
    let runtimeMode: ExpertLabRuntimeMode
    let summary: String
    let detail: String
    let expectedLayers: Int?
    let expectedExperts: Int?
    let trainedTopK: Int?

    static let unknown = ExpertLabCapability(
        isTraceSupported: false,
        runtimeMode: .unsupported,
        summary: "unchecked",
        detail: "Model capability has not been inspected yet.",
        expectedLayers: nil,
        expectedExperts: nil,
        trainedTopK: nil
    )
}

private enum ExpertPrunePlanExportError: Error, LocalizedError {
    case missingAtlas
    case missingExpertCount
    case missingComparison
    case missingSourceModel
    case blocked(String)

    var errorDescription: String? {
        switch self {
        case .missingAtlas:
            return "run an expert review before creating a prune plan."
        case .missingExpertCount:
            return "this bundle does not expose the source expert count."
        case .missingComparison:
            return "run a masked A/B comparison before creating a reviewed prune plan."
        case .missingSourceModel:
            return "open Expert Review from the original BF16/F16 source workflow before handing this plan to hard prune."
        case .blocked(let reason):
            return reason
        }
    }
}

private enum ExpertLabMaskPersistenceError: Error, LocalizedError {
    case missingRun

    var errorDescription: String? {
        switch self {
        case .missingRun:
            return "run or load an Expert Review before saving masks."
        }
    }
}

private enum ExpertLabVMLXRunnerError: Error, LocalizedError {
    case failed(Int32, String)
    case malformed(String)

    var errorDescription: String? {
        switch self {
        case .failed(let code, let stderr):
            return "BF16/vMLX Expert Lab runner failed with exit \(code): \(stderr)"
        case .malformed(let message):
            return message
        }
    }
}

enum ExpertLabPromptIdentityValidator {
    static func issue(expected: [ExpertPrompt], actual: [ExpertPrompt]) -> String? {
        let expectedIDs = expected.map { normalizedID($0.id) }
        let actualIDs = actual.map { normalizedID($0.id) }

        if let index = expectedIDs.firstIndex(where: \.isEmpty) {
            return "requested Expert Lab suite prompt at index \(index) has an empty id"
        }
        if let duplicate = firstDuplicate(in: expectedIDs) {
            return "requested Expert Lab suite contains duplicate prompt id \(duplicate)"
        }
        if let index = actualIDs.firstIndex(where: \.isEmpty) {
            return "BF16/vMLX runner returned generation at index \(index) with an empty prompt id"
        }
        if let duplicate = firstDuplicate(in: actualIDs) {
            return "BF16/vMLX runner returned duplicate prompt id \(duplicate)"
        }
        if expectedIDs.count != actualIDs.count {
            return "BF16/vMLX runner returned \(actualIDs.count) generations for \(expectedIDs.count) prompts"
        }
        guard expectedIDs == actualIDs else {
            let expectedSet = Set(expectedIDs)
            let actualSet = Set(actualIDs)
            let missing = expectedSet.subtracting(actualSet).sorted()
            let unexpected = actualSet.subtracting(expectedSet).sorted()
            if !missing.isEmpty || !unexpected.isEmpty {
                return "BF16/vMLX runner prompt IDs did not match requested suite: missing \(previewIDs(missing)); unexpected \(previewIDs(unexpected))"
            }
            return "BF16/vMLX runner prompt order did not match requested suite: expected \(previewIDs(expectedIDs)); got \(previewIDs(actualIDs))"
        }
        return nil
    }

    private nonisolated static func normalizedID(_ id: String) -> String {
        id.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func firstDuplicate(in ids: [String]) -> String? {
        var seen = Set<String>()
        for id in ids where !id.isEmpty {
            if seen.contains(id) {
                return id
            }
            seen.insert(id)
        }
        return nil
    }

    private nonisolated static func previewIDs(_ ids: [String], limit: Int = 5) -> String {
        if ids.isEmpty {
            return "none"
        }
        let prefix = ids.prefix(limit).joined(separator: ", ")
        if ids.count > limit {
            return "\(prefix), ... (+\(ids.count - limit) more)"
        }
        return prefix
    }
}

enum ExpertLabLayerStatsEvidenceValidator {
    static func issue(
        promptCount: Int,
        baselinePromptCount: Int?,
        maskedPromptCount: Int?,
        evidenceName: String
    ) -> String? {
        guard baselinePromptCount != nil || maskedPromptCount != nil else { return nil }
        guard promptCount > 0,
              baselinePromptCount == promptCount,
              maskedPromptCount == promptCount else {
            let baseline = baselinePromptCount.map(String.init) ?? "missing"
            let masked = maskedPromptCount.map(String.init) ?? "missing"
            return "\(evidenceName) layer-stat coverage is incomplete for \(promptCount) prompts (baseline \(baseline), masked \(masked))."
        }
        return nil
    }
}

enum ExpertLabVMLXRuntimeEvidenceValidator {
    static func issue(
        promptID: String,
        runtimeMode: String?,
        runtimeBackend: String?,
        runtimeMetalEnabled: Bool?,
        deviceName: String?,
        jangToolsVersion: String?,
        mlxVersion: String?,
        mlxLMVersion: String?,
        sourceModelPath: String?,
        hookedMOELayers: Int?,
        expectedMOELayers: Int?,
        hookCoverageComplete: Bool?,
        maskRequired: Bool = false,
        maskApplied: Bool? = nil,
        disabledExpertCount: Int? = nil,
        topKOverride: Int? = nil,
        expectedLayers: Int? = nil,
        expectedSourcePath: String? = nil
    ) -> String? {
        let label = promptID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "unknown prompt"
            : promptID.trimmingCharacters(in: .whitespacesAndNewlines)
        if runtimeMode != "bf16_vmlx" {
            return "BF16/vMLX runner generation \(label) did not record BF16/vMLX runtime evidence"
        }
        if runtimeBackend != "vmlx" {
            return "BF16/vMLX runner generation \(label) did not record the vMLX backend"
        }
        if deviceName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "BF16/vMLX runner generation \(label) is missing runtime device evidence"
        }
        guard let runtimeMetalEnabled else {
            return "BF16/vMLX runner generation \(label) is missing Metal runtime evidence"
        }
        if runtimeMetalEnabled != true {
            return "BF16/vMLX runner generation \(label) did not record a Metal runtime"
        }
        if jangToolsVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
            mlxVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
            mlxLMVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "BF16/vMLX runner generation \(label) is missing vMLX package version evidence"
        }
        guard let sourceModelPath,
              !sourceModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "BF16/vMLX runner generation \(label) is missing source model path evidence"
        }
        if let expectedSourcePath,
           !expectedSourcePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           canonicalPath(sourceModelPath) != canonicalPath(expectedSourcePath) {
            return "BF16/vMLX runner generation \(label) source model path does not match the selected BF16/F16 source"
        }
        guard let hookedMOELayers, hookedMOELayers > 0 else {
            return "BF16/vMLX runner generation \(label) is missing routed-layer hook evidence"
        }
        if hookCoverageComplete == false {
            return "BF16/vMLX runner generation \(label) recorded incomplete routed-layer hook coverage"
        }
        if let expectedMOELayers, expectedMOELayers > 0, hookedMOELayers < expectedMOELayers {
            return "BF16/vMLX runner generation \(label) hooked \(hookedMOELayers) of \(expectedMOELayers) config-routed layers"
        }
        if let expectedLayers, expectedLayers > 0, hookedMOELayers < expectedLayers {
            return "BF16/vMLX runner generation \(label) hooked \(hookedMOELayers) of \(expectedLayers) routed layers"
        }
        if maskRequired {
            if maskApplied != true {
                return "BF16/vMLX runner generation \(label) did not record an applied BF16/vMLX mask"
            }
            if disabledExpertCount == nil && topKOverride == nil {
                return "BF16/vMLX runner generation \(label) is missing mask-shape metadata"
            }
        }
        return nil
    }

    private nonisolated static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }
}

enum ExpertLabVMLXTraceEvidenceValidator {
    static func issue(
        promptID: String,
        emitTokenTrace: Bool,
        tokenTraceCount: Int?,
        expectedRouteRecordCount: Int? = nil,
        hasInvalidRouteRecord: Bool = false
    ) -> String? {
        guard emitTokenTrace else { return nil }
        let label = promptID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "unknown prompt"
            : promptID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let tokenTraceCount, tokenTraceCount > 0 else {
            return "BF16/vMLX runner generation \(label) is missing token routing trace evidence"
        }
        if let expectedRouteRecordCount, expectedRouteRecordCount > 0,
           tokenTraceCount != expectedRouteRecordCount {
            return "BF16/vMLX runner generation \(label) token routing trace covers \(tokenTraceCount) of \(expectedRouteRecordCount) routed layer-token records"
        }
        if hasInvalidRouteRecord {
            return "BF16/vMLX runner generation \(label) has malformed token routing trace records"
        }
        return nil
    }
}

private struct ExpertLabVMLXRunSummary: Decodable {
    let generationsJSONL: String

    enum CodingKeys: String, CodingKey {
        case generationsJSONL = "generations_jsonl"
    }
}

private struct ExpertLabVMLXGenerationRecord: Decodable {
    let prompt: ExpertPrompt
    let result: ExpertLabVMLXRunResult
}

private struct ExpertLabVMLXRunResult: Decodable {
    let text: String
    let tokens: Int
    let elapsedSeconds: Double
    let tokensPerSecond: Double
    let finishReason: String
    let layerStats: [ExpertLabVMLXLayerStats]
    let tokenTrace: [ExpertLabVMLXRouteRecord]?
    let runtimeInfo: ExpertLabVMLXRuntimeInfo?

    enum CodingKeys: String, CodingKey {
        case text
        case tokens
        case elapsedSeconds = "elapsed_seconds"
        case tokensPerSecond = "tokens_per_second"
        case finishReason = "finish_reason"
        case layerStats = "layer_stats"
        case tokenTrace = "token_trace"
        case runtimeInfo = "runtime_info"
    }
}

private struct ExpertLabVMLXLayerStats: Decodable {
    let layer: Int
    let tokenCount: Int
    let hitCounts: [String: Int]
    let probabilityMass: [String: Float]

    enum CodingKeys: String, CodingKey {
        case layer
        case tokenCount = "token_count"
        case hitCounts = "hit_counts"
        case probabilityMass = "probability_mass"
    }
}

private struct ExpertLabVMLXRouteRecord: Decodable {
    let tokenIndex: Int
    let layer: Int
    let selectedExperts: [Int]
    let scores: [Float]
    let disabledExperts: [Int]
    let effectiveTopK: Int
    let entropy: Float?

    enum CodingKeys: String, CodingKey {
        case tokenIndex = "token_index"
        case layer
        case selectedExperts = "selected_experts"
        case scores
        case disabledExperts = "disabled_experts"
        case effectiveTopK = "effective_top_k"
        case entropy
    }
}

private struct ExpertLabVMLXRuntimeInfo: Decodable {
    let backend: String
    let runtimeMode: String
    let deviceName: String
    let runtimeMetalEnabled: Bool
    let jangToolsVersion: String?
    let mlxVersion: String?
    let mlxLMVersion: String?
    let mlxVLMVersion: String?
    let sourceModelPath: String?
    let hookedMOELayers: Int?
    let expectedMOELayers: Int?
    let hookCoverageComplete: Bool?
    let maskApplied: Bool?
    let disabledExpertCount: Int?
    let topKOverride: Int?
    let notes: [String]?

    enum CodingKeys: String, CodingKey {
        case backend
        case runtimeMode = "runtime_mode"
        case deviceName = "device_name"
        case runtimeMetalEnabled = "runtime_metal_enabled"
        case jangToolsVersion = "jang_tools_version"
        case mlxVersion = "mlx_version"
        case mlxLMVersion = "mlx_lm_version"
        case mlxVLMVersion = "mlx_vlm_version"
        case sourceModelPath = "source_model_path"
        case hookedMOELayers = "hooked_moe_layers"
        case expectedMOELayers = "expected_moe_layers"
        case hookCoverageComplete = "hook_coverage_complete"
        case maskApplied = "mask_applied"
        case disabledExpertCount = "disabled_expert_count"
        case topKOverride = "top_k_override"
        case notes
    }
}

@Observable
@MainActor
private final class ExpertLabViewModel {
    let modelPath: URL
    let sourceModelPath: URL?
    var suites: [ExpertPromptSuite]
    var selectedSuiteName: String
    var selectedPromptID: String
    var livePromptText = ""
    var maxTokens: Int = 64
    var emitTokenTrace: Bool = true
    var maxTraceTokens: Int = 32768
    var topKOverride: Int = 0
    var atlasFilter: ExpertAtlasFilter = .all
    var atlasMetric: ExpertAtlasMetric = .frequency
    var atlasSort: ExpertAtlasSort = .expertID
    var atlasLayerFilterText = ""
    var atlasExpertFilterText = ""
    var atlasDomainFilterText = ""
    var atlasPromptFilterText = ""
    var selectedExpert: SelectedExpert?
    var groupSelection: Set<ExpertCoordinate> = []
    var isRunning = false
    var statusText = "Idle"
    var runProgress = 0
    var runProgressTotal = 0
    var lastError: String?
    var runs: [ExpertPromptRun] = []
    var atlas: ExpertAtlas?
    var runHistory: [ExpertRunSummary] = []
    var selectedRunID = ""
    var selectedRunEvidenceSummary = ""
    var selectedRunArtifactSummary = ""
    var selectedRunEvidenceWarning = ""
    var lastRunDirectory: URL?
    var lastEvalDirectory: URL?
    var baselineText = ""
    var maskedText = ""
    var baselineOutputReady = false
    var maskedOutputReady = false
    var baselineTokensPerSecond: Double = 0
    var maskedTokensPerSecond: Double = 0
    var runtimeInfoSummary = ""
    var lastEvalSummary = ""
    var comparisonPreviewRows: [StoredEvalRecord] = []
    var capability: ExpertLabCapability
    var pruneKeepExperts: Int
    var cancelRequested = false

    @ObservationIgnored private var model: JANGKit.Model?

    private var planSourceModelPath: String? {
        if let sourceModelPath {
            return sourceModelPath.path
        }
        if capability.runtimeMode == .bf16VMLX {
            return modelPath.path
        }
        return nil
    }

    private var canonicalPlanSourceModelPath: String? {
        planSourceModelPath.map(Self.canonicalPath)
    }

    private var artifactSourcePath: String {
        planSourceModelPath ?? modelPath.path
    }

    private var artifactReviewBundlePath: String? {
        capability.runtimeMode == .bf16VMLX ? nil : modelPath.path
    }

    private var artifactRuntimeModeFallback: String {
        capability.runtimeMode == .bf16VMLX ? "bf16_vmlx" : "native_jangtq_review_bundle"
    }

    init(modelPath: URL, sourceModelPath: URL? = nil) {
        self.modelPath = modelPath
        self.sourceModelPath = sourceModelPath
        self.suites = Self.defaultSuites
        self.selectedSuiteName = Self.defaultSuites.first?.name ?? "General"
        self.selectedPromptID = Self.defaultSuites.first?.prompts.first?.id ?? ""
        let capability = Self.detectCapability(modelPath: modelPath)
        self.capability = capability
        let experts = capability.expectedExperts ?? 1
        let maxKeep = max(experts - 1, 1)
        let minKeep = min(max(1, experts / 2), maxKeep)
        self.pruneKeepExperts = min(max(experts - 32, minKeep), maxKeep)
        // The View's .task modifier also calls this.
        reloadRunHistory()
    }

    var selectedSuite: ExpertPromptSuite {
        suites.first { $0.name == selectedSuiteName } ?? suites[0]
    }

    var selectedSuiteEvidenceSummary: String {
        let prompts = selectedSuite.prompts
        let domainCount = Set(prompts.map(\.domain)).count
        let tokenBudgets = prompts.compactMap(\.maxNewTokens)
        let tokenSummary: String
        if let minTokens = tokenBudgets.min(),
           let maxTokens = tokenBudgets.max(),
           tokenBudgets.count == prompts.count {
            tokenSummary = minTokens == maxTokens ? "\(minTokens) tokens" : "\(minTokens)-\(maxTokens) tokens"
        } else {
            tokenSummary = "\(maxTokens) token fallback"
        }
        let semanticSummary = Self.semanticCoverageSummary(for: prompts)
        return "\(prompts.count) prompts / \(domainCount) domains / \(tokenSummary) / \(semanticSummary)"
    }

    private nonisolated static func semanticCoverageSummary(for prompts: [ExpertPrompt]) -> String {
        let semanticDomains = Set(prompts.flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
        let missing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(semanticDomains)
            .sorted()
        guard !missing.isEmpty else {
            return "semantic probes ready"
        }
        let names = missing
            .map(ExpertDomainTaxonomy.displayName(for:))
            .joined(separator: ", ")
        return "missing semantic probes: \(names)"
    }

    var reviewRuntimeTargetSummary: String {
        let review: String
        if capability.runtimeMode == .bf16VMLX {
            review = "BF16/vMLX source: \(modelPath.lastPathComponent)"
        } else {
            review = "Legacy review bundle: \(modelPath.lastPathComponent)"
        }
        let authoritySource = sourceModelPath ?? (capability.runtimeMode == .bf16VMLX ? modelPath : nil)
        let source = authoritySource.map { "BF16/F16 authority: \($0.lastPathComponent)" }
            ?? "BF16/F16 authority not linked"
        let runtime = runtimeInfoSummary.isEmpty ? capability.detail : runtimeInfoSummary
        return "\(review). \(source). Runtime: \(runtime)"
    }

    var selectedPrompt: ExpertPrompt? {
        selectedSuite.prompts.first { $0.id == selectedPromptID } ?? selectedSuite.prompts.first
    }

    var canRunLivePrompt: Bool {
        capability.isTraceSupported
        && !isRunning
        && !livePromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var selectedEntry: ExpertAtlasEntry? {
        guard let selectedExpert else { return nil }
        return atlas?.experts.first {
            $0.layer == selectedExpert.layer && $0.expert == selectedExpert.expert
        }
    }

    var selectedRunSummary: ExpertRunSummary? {
        runHistory.first { $0.runID == selectedRunID }
    }

    var hasMask: Bool {
        !runtimeMaskLayers.isEmpty || topKOverride > 0
    }

    var hasLivePromptOutput: Bool {
        baselineOutputReady || maskedOutputReady
        || lastEvalSummary.hasPrefix("Live prompt")
        || (isRunning && selectedSuiteName == "Live Prompt")
    }

    var hasAnyReviewState: Bool {
        hasMask || dropCandidateCount > 0 || lockedKeepCount > 0
    }

    var canSaveMask: Bool {
        hasAnyReviewState && lastRunDirectory != nil
    }

    var hasRecoveryInfo: Bool {
        lastError != nil || partialRunDirectory != nil
    }

    var canCleanPartialRun: Bool {
        guard let dir = partialRunDirectory else { return false }
        return FileManager.default.fileExists(atPath: dir.path)
    }

    var recoverySummary: String {
        var parts: [String] = []
        if let lastError {
            parts.append("Last issue: \(lastError)")
        }
        if let selectedRunSummary, let failureStage = selectedRunSummary.failureStage {
            let message = selectedRunSummary.failureMessage ?? "no failure message recorded"
            parts.append("Selected partial run: \(failureStage) - \(message)")
        }
        if let lastRunDirectory {
            parts.append("Run artifact: \(lastRunDirectory.path)")
        }
        if let lastEvalDirectory {
            parts.append("Eval artifact: \(lastEvalDirectory.path)")
        }
        return parts.isEmpty ? "No recovery information yet." : parts.joined(separator: "\n")
    }

    private var partialRunDirectory: URL? {
        if let selectedRunSummary, selectedRunSummary.failureStage != nil {
            return URL(fileURLWithPath: selectedRunSummary.directoryPath, isDirectory: true)
        }
        if let current = lastRunDirectory,
           let selectedRunSummary = runHistory.first(where: { $0.directoryPath == current.path }),
           selectedRunSummary.failureStage != nil {
            return current
        }
        return nil
    }

    var groupSelectionCount: Int {
        groupSelection.count
    }

    var groupSelectionSummary: String {
        guard !groupSelection.isEmpty else { return "No range selected. Drag across cells to review a group." }
        let layers = Set(groupSelection.map(\.layer)).count
        return "\(groupSelection.count) experts selected across \(layers) layer\(layers == 1 ? "" : "s")."
    }

    var maskSummary: String {
        var parts: [String] = []
        if topKOverride > 0 {
            parts.append("top-k \(topKOverride)")
        }
        let disabled = runtimeMaskLayers
            .sorted { $0.key < $1.key }
            .flatMap { layer, experts in experts.sorted().map { "L\(layer)E\($0)" } }
        parts.append(contentsOf: disabled)
        return parts.isEmpty ? "No temporary mask." : parts.joined(separator: ", ")
    }

    var runtimeMaskLayers: [Int: Set<Int>] {
        var layers = maskLayers
        for (layer, experts) in dropCandidates {
            layers[layer, default: []].formUnion(experts)
        }
        return layers
    }

    var maskedExpertCount: Int {
        maskLayers.values.reduce(0) { $0 + $1.count }
    }

    var dropCandidateCount: Int {
        dropCandidates.values.reduce(0) { $0 + $1.count }
    }

    var lockedKeepCount: Int {
        lockedKeeps.values.reduce(0) { $0 + $1.count }
    }

    var planValidationIssues: [ExpertMaskValidationIssue] {
        guard atlas != nil else { return [] }
        var mask = JANGKit.ExpertMask(layers: dropCandidates, lockedKeepByLayer: lockedKeeps)
        mask.topKOverride = nil
        return ExpertMaskEngine.validate(
            mask: mask,
            sourceNumExpertsByLayer: sourceExpertsByLayerForAtlas(),
            trainedTopKByLayer: trainedTopKByLayerForAtlas(),
            hotExperts: hotExpertCoordinates,
            maxDropFractionPerLayer: 0.50
        )
    }

    var hasBlockingPlanIssue: Bool {
        planValidationIssues.contains { $0.severity == .error }
    }

    var hasComparisonSummary: Bool {
        guard let lastEvalDirectory else { return false }
        return comparisonMaskMatchesCurrent(lastEvalDirectory)
            && FileManager.default.fileExists(
            atPath: lastEvalDirectory.appendingPathComponent("comparison_summary.json").path
        )
    }

    var hasComparisonEvidence: Bool {
        hasComparisonSummary || !comparisonPreviewRows.isEmpty
    }

    var comparisonRiskRows: [StoredEvalRecord] {
        comparisonPreviewRows.filter(\.isRisky)
    }

    var comparisonRowsForDisplay: [StoredEvalRecord] {
        let risky = comparisonRiskRows
        return risky.isEmpty ? comparisonPreviewRows : risky
    }

    var comparisonRowsOmittedCount: Int {
        max(0, comparisonRowsForDisplay.count - 4)
    }

    var comparisonArtifactSummary: String {
        guard let lastEvalDirectory, !comparisonPreviewRows.isEmpty else { return "" }
        let riskCount = comparisonRiskRows.count
        let risk = riskCount == 0 ? "0 risky" : "\(riskCount) risky"
        let severity = Self.regressionSeverity(from: comparisonPreviewRows)
        let depth = comparisonMeanTokenDepth.map {
            String(format: ", avg tokens %.1f/%.1f", $0.baseline, $0.masked)
        } ?? ", token depth missing"
        let routes = comparisonRouteRecordSummary.map { ", routes \($0.baseline)/\($0.masked)" }
            ?? ", routing records missing"
        return "Eval artifacts: \(comparisonPreviewRows.count) rows, \(risk), severity \(severity)\(depth)\(routes), eval.jsonl + eval_trace.jsonl + eval_index.json -> \(lastEvalDirectory.path)"
    }

    var comparisonEvidenceSummary: String {
        let total = comparisonPreviewRows.count
        let riskCount = comparisonRiskRows.count
        if total == 0 {
            return lastEvalSummary.isEmpty ? "No per-prompt eval rows loaded." : lastEvalSummary
        }
        if riskCount == 0 {
            return "Per-prompt eval rows loaded: \(total). No regressions flagged. Severity \(Self.regressionSeverity(from: comparisonPreviewRows))."
        }
        let domains = Array(Set(comparisonRiskRows.map(\.domain))).sorted().joined(separator: ", ")
        return "Per-prompt regressions: \(riskCount) of \(total). Severity \(Self.regressionSeverity(from: comparisonPreviewRows)). Domains: \(domains)."
    }

    var selectedExpertHasComparisonRisk: Bool {
        !comparisonRiskRows.isEmpty || !(latestComparisonSummary()?.highRiskDomains.isEmpty ?? true)
    }

    func selectedExpertReviewStatus(for entry: ExpertAtlasEntry) -> String {
        var statuses: [String] = []
        if isLockedKeep(layer: entry.layer, expert: entry.expert) {
            statuses.append("locked keep")
        }
        if isDropCandidate(layer: entry.layer, expert: entry.expert) {
            statuses.append("drop candidate / user-forced drop")
        } else if isMasked(layer: entry.layer, expert: entry.expert) {
            statuses.append("temporary mask")
        }
        if let summary = latestComparisonSummary(),
           summary.safeDropCandidates.contains(Self.coordinate(for: entry)) {
            statuses.append("same-suite safe-drop")
        }
        return statuses.isEmpty ? "unmarked" : statuses.joined(separator: ", ")
    }

    func selectedExpertComparedMaskStatus(for entry: ExpertAtlasEntry) -> String {
        guard latestComparisonSummary() != nil else {
            return "no current same-mask comparison"
        }
        if runtimeMaskLayers[entry.layer]?.contains(entry.expert) == true {
            return "expert disabled in latest compared mask"
        }
        if topKOverride > 0 {
            return "top-k \(topKOverride) compared; expert not individually disabled"
        }
        return "expert not disabled in latest compared mask"
    }

    func selectedExpertSafeDropStatus(for entry: ExpertAtlasEntry) -> String {
        guard let summary = latestComparisonSummary() else {
            return "not evaluated"
        }
        let coordinate = Self.coordinate(for: entry)
        if summary.safeDropCandidates.contains(coordinate) {
            return "yes, in same-suite safe-drop set"
        }
        if summary.safeDropCandidates.isEmpty {
            return "none found in latest comparison"
        }
        return "no, not in safe-drop set"
    }

    func selectedExpertMaskedImpactSummary(for entry: ExpertAtlasEntry) -> String {
        guard let summary = latestComparisonSummary() else {
            return "compare current mask first"
        }
        let severity = summary.regressionSeverity ?? Self.regressionSeverity(from: comparisonPreviewRows)
        return String(
            format: "%d prompts, mean delta %.2f, severity %@",
            summary.promptCount,
            summary.meanTextDelta,
            severity
        )
    }

    func selectedExpertRegressionSeveritySummary(for entry: ExpertAtlasEntry) -> String {
        guard let summary = latestComparisonSummary() else {
            return "n/a"
        }
        let severity = summary.regressionSeverity ?? Self.regressionSeverity(from: comparisonPreviewRows)
        if runtimeMaskLayers[entry.layer]?.contains(entry.expert) == true {
            return severity
        }
        if topKOverride > 0 {
            return "top-k \(severity)"
        }
        if summary.safeDropCandidates.contains(Self.coordinate(for: entry)) {
            return "\(severity) safe"
        }
        return "not compared"
    }

    func selectedExpertRegressionSeverityIsHigh(for entry: ExpertAtlasEntry) -> Bool {
        selectedExpertRegressionSeverityRank(for: entry) >= Self.severityRank(
            ExpertPromptEvaluator.regressionSeverityHigh
        )
    }

    private func selectedExpertRegressionSeverityRank(for entry: ExpertAtlasEntry) -> Int {
        guard let summary = latestComparisonSummary() else { return 0 }
        let coordinate = Self.coordinate(for: entry)
        let participates = runtimeMaskLayers[entry.layer]?.contains(entry.expert) == true
            || topKOverride > 0
            || summary.safeDropCandidates.contains(coordinate)
        guard participates else { return 0 }
        return Self.severityRank(
            summary.regressionSeverity ?? Self.regressionSeverity(from: comparisonPreviewRows)
        )
    }

    func selectedExpertPassRateSummary(for entry: ExpertAtlasEntry) -> String {
        guard let summary = latestComparisonSummary() else {
            return "not evaluated"
        }
        return "baseline \(Self.formatPassRate(summary.passRateBaseline)), masked \(Self.formatPassRate(summary.passRateMasked))"
    }

    func selectedExpertRiskSummary(for entry: ExpertAtlasEntry) -> String {
        guard let summary = latestComparisonSummary() else {
            return "not evaluated"
        }
        let riskRows = comparisonRiskRows.count
        let risk = riskRows == 0 ? "0 regression rows" : "\(riskRows) regression row\(riskRows == 1 ? "" : "s")"
        if summary.highRiskDomains.isEmpty {
            return "\(risk), no high-risk domains"
        }
        return "\(risk), domains \(summary.highRiskDomains.sorted().joined(separator: ", "))"
    }

    var reviewedPrunePromptCount: Int {
        if !runs.isEmpty {
            return runs.count
        }
        let prompts = reviewedPrunePromptsForCoverage()
        if !prompts.isEmpty {
            return prompts.count
        }
        return atlas?.promptCount ?? 0
    }

    var reviewedPruneDomainCount: Int {
        Set(reviewedPrunePromptsForCoverage().map(\.domain)).count
    }

    var reviewedPruneSemanticDomains: Set<String> {
        Set(reviewedPrunePromptsForCoverage().flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
    }

    var reviewedPruneCoverageIssues: [String] {
        guard atlas != nil else { return [] }
        var issues: [String] = []
        let promptIDs = reviewedPrunePromptsForCoverage()
            .map { $0.id.trimmingCharacters(in: .whitespacesAndNewlines) }
        if promptIDs.contains(where: \.isEmpty) {
            issues.append("Coverage gate: prompt suite contains empty prompt IDs before reviewed prune planning.")
        }
        if let duplicate = Self.firstDuplicatePromptID(in: promptIDs) {
            issues.append("Coverage gate: prompt suite contains duplicate prompt IDs: \(duplicate).")
        }
        if reviewedPrunePromptCount < Self.minimumReviewedPrunePromptCount {
            issues.append("Coverage gate: trace at least \(Self.minimumReviewedPrunePromptCount) prompts before reviewed prune planning.")
        }
        if reviewedPruneDomainCount < Self.minimumReviewedPruneDomainCount {
            issues.append("Coverage gate: include at least \(Self.minimumReviewedPruneDomainCount) prompt domains before reviewed prune planning.")
        }
        let missingSemanticDomains = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(reviewedPruneSemanticDomains)
            .sorted()
        if !missingSemanticDomains.isEmpty {
            let names = missingSemanticDomains
                .map(ExpertDomainTaxonomy.displayName(for:))
                .joined(separator: ", ")
            issues.append("Coverage gate: include required semantic prompt probes before reviewed prune planning: \(names).")
        }
        return issues
    }

    var hasReviewedPruneCoverage: Bool {
        reviewedPruneCoverageIssues.isEmpty
    }

    var reviewedPruneAtlasIssues: [String] {
        guard let atlas else { return [] }
        let sourceExpertsByLayer = sourceExpertsByLayerForAtlas()
        guard !sourceExpertsByLayer.isEmpty else {
            return ["Atlas gate: source expert grid is missing; rerun Trace Suite from the original BF16/F16 source before pruning."]
        }

        let coordinates = atlas.experts.map { ExpertCoordinate(layer: $0.layer, expert: $0.expert) }
        let coordinateSet = Set(coordinates)
        let expectedCells = sourceExpertsByLayer.values.reduce(0, +)
        var issues: [String] = []

        if coordinateSet.count != coordinates.count {
            issues.append("Atlas gate: expert atlas contains duplicate layer/expert cells; rerun Trace Suite before pruning.")
        }
        if let expectedLayers = capability.expectedLayers,
           expectedLayers > 0,
           sourceExpertsByLayer.count != expectedLayers {
            issues.append("Atlas gate: source expert grid covers \(sourceExpertsByLayer.count) of \(expectedLayers) routed layers.")
        }
        if let expectedExperts = capability.expectedExperts,
           sourceExpertsByLayer.values.contains(where: { $0 != expectedExperts }) {
            issues.append("Atlas gate: source expert grid does not match the configured \(expectedExperts) experts per layer.")
        }

        var missing: [String] = []
        for layer in sourceExpertsByLayer.keys.sorted() {
            let expertCount = sourceExpertsByLayer[layer] ?? 0
            for expert in 0..<expertCount {
                let coordinate = ExpertCoordinate(layer: layer, expert: expert)
                if !coordinateSet.contains(coordinate) {
                    missing.append("L\(layer) E\(expert)")
                }
            }
        }
        if !missing.isEmpty {
            issues.append("Atlas gate: full source expert grid is incomplete (\(coordinateSet.count)/\(expectedCells) cells); missing \(Self.expertCoordinatePreview(missing)).")
        }

        let outOfGrid = coordinates.compactMap { coordinate -> String? in
            guard let expertCount = sourceExpertsByLayer[coordinate.layer],
                  coordinate.expert >= 0,
                  coordinate.expert < expertCount else {
                return "L\(coordinate.layer) E\(coordinate.expert)"
            }
            return nil
        }
        if !outOfGrid.isEmpty {
            issues.append("Atlas gate: atlas contains cells outside the source expert grid: \(Self.expertCoordinatePreview(outOfGrid)).")
        }
        return issues
    }

    var reviewedPruneAuthorityIssues: [String] {
        var issues: [String] = []
        if capability.runtimeMode != .bf16VMLX {
            issues.append("Authority gate: reviewed pruning must start from the original BF16/F16 source through BF16/vMLX, not a legacy JANG/JANGTQ review bundle.")
        }
        if planSourceModelPath == nil {
            issues.append("Authority gate: original BF16/F16 source path is missing; reopen Expert Review from the BF16/vMLX source workflow.")
        }
        if let issue = loadedRunAuthorityIssue() {
            issues.append(issue)
        }
        return issues
    }

    private func loadedRunAuthorityIssue() -> String? {
        guard let lastRunDirectory else { return nil }
        let manifestURL = lastRunDirectory.appendingPathComponent("run.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? decoder.decode(ExpertRunManifest.self, from: data) else {
            return "Authority gate: loaded run manifest is unreadable; rerun Trace Suite from the original BF16/F16 source."
        }
        if manifest.runtimeMode != "bf16_vmlx" {
            return "Authority gate: loaded run was captured with \(Self.runtimeModeDisplayName(manifest.runtimeMode)), not BF16/vMLX. Rerun Trace Suite from the original BF16/F16 source."
        }
        if manifest.runtimeBackend?.trimmingCharacters(in: .whitespacesAndNewlines) != "vmlx" {
            return "Authority gate: loaded run did not record the vMLX backend. Rerun Trace Suite from the original BF16/F16 source."
        }
        if manifest.runtimeDevice?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "Authority gate: loaded run is missing runtime device evidence. Rerun Trace Suite from the original BF16/F16 source."
        }
        guard let runtimeMetalEnabled = manifest.runtimeMetalEnabled else {
            return "Authority gate: loaded run is missing Metal runtime evidence. Rerun Trace Suite from the original BF16/F16 source."
        }
        if runtimeMetalEnabled != true {
            return "Authority gate: loaded run did not record a Metal runtime. Rerun Trace Suite from the original BF16/F16 source."
        }
        if manifest.jangToolsVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
            manifest.mlxVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
            manifest.mlxLMVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "Authority gate: loaded run is missing vMLX package version evidence. Rerun Trace Suite with current Studio."
        }
        if manifest.reviewBundlePath != nil {
            return "Authority gate: loaded run still references a legacy review bundle; rerun Trace Suite through BF16/vMLX."
        }
        if let sourcePath = planSourceModelPath,
           Self.canonicalPath(manifest.sourcePath) != Self.canonicalPath(sourcePath) {
            return "Authority gate: loaded run source path does not match the selected BF16/F16 source."
        }
        guard let runtimeSourcePath = manifest.sourceModelPath?.trimmingCharacters(in: .whitespacesAndNewlines),
              !runtimeSourcePath.isEmpty else {
            return "Authority gate: loaded run is missing runtime source path evidence. Rerun Trace Suite from the original BF16/F16 source."
        }
        if let sourcePath = planSourceModelPath,
           Self.canonicalPath(runtimeSourcePath) != Self.canonicalPath(sourcePath) {
            return "Authority gate: loaded run runtime source path does not match the selected BF16/F16 source."
        }
        guard let hookedMOELayers = manifest.hookedMOELayers, hookedMOELayers > 0 else {
            return "Authority gate: loaded run is missing vMLX routed-layer hook evidence. Rerun Trace Suite with current Studio."
        }
        if manifest.hookCoverageComplete == false {
            return "Authority gate: loaded run recorded incomplete vMLX routed-layer hook coverage. Rerun Trace Suite after hooks cover every MoE layer."
        }
        if let expectedMOELayers = manifest.expectedMOELayers,
           expectedMOELayers > 0,
           hookedMOELayers < expectedMOELayers {
            return "Authority gate: loaded run vMLX hook coverage \(hookedMOELayers) of \(expectedMOELayers) config-routed layers."
        }
        return nil
    }

    var reviewedPruneSemanticEvidenceIssues: [String] {
        guard let atlas else { return [] }
        let observed = atlas.experts.filter { entry in
            entry.hits > 0 || !entry.domains.isEmpty || entry.probabilityMass > 0
        }
        guard !observed.isEmpty else {
            return ["Semantic evidence gate: trace did not record observed expert activations."]
        }

        var missingLift: [String] = []
        var missingGateMass: [String] = []
        var missingPromptEvidence: [String] = []
        var incompletePromptEvidence: [String] = []

        for entry in observed {
            let coordinate = "L\(entry.layer) E\(entry.expert)"
            if !entry.probabilityMass.isFinite || entry.probabilityMass <= 0 {
                missingGateMass.append(coordinate)
            }
            if !entry.domainLift.values.contains(where: { $0.isFinite }) {
                missingLift.append(coordinate)
            }
            let promptEvidence = entry.promptEvidence ?? []
            if promptEvidence.isEmpty {
                missingPromptEvidence.append(coordinate)
                continue
            }
            let hasPromptProof = promptEvidence.contains { evidence in
                !evidence.promptID.isEmpty
                    && !evidence.domain.isEmpty
                    && !evidence.tags.isEmpty
                    && !evidence.promptExcerpt.isEmpty
                    && evidence.hits > 0
            }
            if !hasPromptProof {
                incompletePromptEvidence.append(coordinate)
            }
        }

        var issues: [String] = []
        if !missingGateMass.isEmpty {
            issues.append("Semantic evidence gate: gate mass is missing for \(Self.expertCoordinatePreview(missingGateMass)).")
        }
        if let summary = latestComparisonSummary(),
           !summary.meanTextDelta.isFinite {
            issues.append("Semantic evidence gate: masked-output impact evidence is missing or invalid.")
        }
        if !missingLift.isEmpty {
            issues.append("Semantic evidence gate: activation lift is missing for \(Self.expertCoordinatePreview(missingLift)).")
        }
        if !missingPromptEvidence.isEmpty {
            issues.append("Semantic evidence gate: prompt examples are missing for \(Self.expertCoordinatePreview(missingPromptEvidence)).")
        }
        if !incompletePromptEvidence.isEmpty {
            issues.append("Semantic evidence gate: prompt tags/examples are incomplete for \(Self.expertCoordinatePreview(incompletePromptEvidence)).")
        }
        return issues
    }

    var reviewedPruneComparisonIssues: [String] {
        guard atlas != nil,
              let summary = latestComparisonSummary() else { return [] }
        var issues: [String] = []
        if summary.promptCount < Self.minimumReviewedPrunePromptCount {
            issues.append("Comparison gate: compare at least \(Self.minimumReviewedPrunePromptCount) prompts before reviewed prune planning.")
        }
        if summary.promptCount != reviewedPrunePromptCount {
            issues.append("Comparison gate: rerun A/B compare for all traced prompts before reviewed prune planning.")
        }
        if summary.validatorAvailablePromptCount == nil || summary.classificationCounts == nil {
            issues.append("Comparison gate: validator classification evidence is missing. Rerun Compare Suite before reviewed prune planning.")
        }
        if (summary.baselineQualifiedPromptCount ?? 0) <= 0 {
            issues.append("Comparison gate: no prompts have a valid BF16/vMLX baseline validator pass.")
        }
        let missingBaselineQualified = summary.missingBaselineQualifiedSemanticCoverage ?? []
        if !missingBaselineQualified.isEmpty {
            issues.append("Comparison gate: baseline-qualified semantic coverage is missing: \(missingBaselineQualified.sorted().joined(separator: ", ")).")
        }
        let degradedPromptIDs = summary.degradedPromptIDs ?? []
        if !degradedPromptIDs.isEmpty {
            issues.append("Comparison gate: baseline-qualified prompts degraded after masking: \(degradedPromptIDs.prefix(8).joined(separator: ", ")).")
        }
        if let passRate = summary.baselineQualifiedMaskedPassRate,
           passRate < 1.0 {
            issues.append("Comparison gate: masked validator pass rate is below 100% on baseline-qualified prompts.")
        }
        if !summary.highRiskDomains.isEmpty {
            issues.append("Comparison gate: masked outputs regressed in high-risk domains: \(summary.highRiskDomains.sorted().joined(separator: ", ")).")
        }
        if summary.safeDropCandidates.isEmpty {
            issues.append("Comparison gate: A/B comparison found no safe drop candidates. Treat this as a valid review outcome; adjust the mask and rerun Compare Suite before pruning.")
        }
        if comparisonPreviewRows.isEmpty {
            issues.append("Comparison gate: per-prompt eval rows are missing. Rerun Compare Suite before reviewed prune planning.")
        } else if comparisonPreviewRows.count != summary.promptCount {
            issues.append("Comparison gate: per-prompt eval rows cover \(comparisonPreviewRows.count) of \(summary.promptCount) compared prompts.")
        }
        if let promptIDIssue = comparisonPromptIDCoverageIssue() {
            issues.append(promptIDIssue)
        }
        if !comparisonRiskRows.isEmpty {
            let domains = Array(Set(comparisonRiskRows.map(\.domain))).sorted().joined(separator: ", ")
            issues.append("Comparison gate: \(comparisonRiskRows.count) per-prompt regression row\(comparisonRiskRows.count == 1 ? "" : "s") must be reviewed before pruning. Severity \(Self.regressionSeverity(from: comparisonPreviewRows)). Domains: \(domains).")
        }
        if comparisonPreviewRows.contains(where: { $0.baselineTokenCount == nil || $0.maskedTokenCount == nil }) {
            issues.append("Comparison gate: per-prompt token counts are missing. Rerun Compare Suite with current Studio before pruning.")
        } else if let depth = comparisonMeanTokenDepth {
            let shallow = min(depth.baseline, depth.masked)
            if shallow < Self.minimumReviewedPruneMeanTokens {
                issues.append(
                    String(
                        format: "Comparison gate: average generated depth is %.1f tokens; rerun with at least %.0f average tokens before pruning.",
                        shallow,
                        Self.minimumReviewedPruneMeanTokens
                    )
                )
            }
        }
        if comparisonPreviewRows.contains(where: {
            ($0.baselineRouteRecordCount ?? 0) <= 0 || ($0.maskedRouteRecordCount ?? 0) <= 0
        }) {
            issues.append("Comparison gate: per-prompt routing records are missing. Rerun Compare Suite with token traces before pruning.")
        }
        if let issue = currentComparisonArtifactIntegrityIssue(comparisonSummary: summary) {
            issues.append(issue)
        }
        if let issue = currentComparisonTraceIntegrityIssue(comparisonSummary: summary) {
            issues.append(issue)
        }
        if let issue = ExpertLabLayerStatsEvidenceValidator.issue(
            promptCount: comparisonPreviewRows.count,
            baselinePromptCount: Self.layerStatsPromptCount(comparisonPreviewRows.map(\.baselineLayerStats)),
            maskedPromptCount: Self.layerStatsPromptCount(comparisonPreviewRows.map(\.maskedLayerStats)),
            evidenceName: "Comparison gate: per-prompt routed-layer stats"
        ) {
            issues.append("\(issue) Rerun Compare Suite with current Studio before pruning.")
        }
        if comparisonPreviewRows.contains(where: {
            $0.runtimeMode == nil || $0.runtimeDevice == nil || $0.runtimeMetalEnabled == nil
        }) {
            issues.append("Comparison gate: runtime device evidence is missing. Rerun Compare Suite with current Studio before pruning.")
        } else if comparisonPreviewRows.contains(where: { $0.runtimeMetalEnabled != true }) {
            issues.append("Comparison gate: masked compare did not record a Metal runtime. Rerun Compare Suite on the native Metal path before pruning.")
        }
        if comparisonPreviewRows.contains(where: {
            ($0.jangToolsVersion ?? "").isEmpty
                || ($0.mlxVersion ?? "").isEmpty
                || ($0.mlxLMVersion ?? "").isEmpty
        }) {
            issues.append("Comparison gate: vMLX package version evidence is missing. Rerun Compare Suite with current Studio before pruning.")
        }
        if comparisonPreviewRows.contains(where: { $0.runtimeMode != "bf16_vmlx" }) {
            issues.append("Comparison gate: eval rows did not record the BF16/vMLX runtime. Rerun Compare Suite from the original BF16/F16 source.")
        }
        if comparisonPreviewRows.contains(where: { $0.runtimeBackend != "vmlx" }) {
            issues.append("Comparison gate: eval rows did not record the vMLX backend. Rerun Compare Suite from the original BF16/F16 source.")
        }
        if comparisonPreviewRows.contains(where: { ($0.hookedMOELayers ?? 0) <= 0 }) {
            issues.append("Comparison gate: per-prompt vMLX routed-layer hook evidence is missing. Rerun Compare Suite with current Studio before pruning.")
        }
        if comparisonPreviewRows.contains(where: { $0.hookCoverageComplete == false }) {
            issues.append("Comparison gate: vMLX routed-layer hook coverage is incomplete. Rerun Compare Suite after router hooks cover every MoE layer.")
        }
        if let expectedLayers = capability.expectedLayers,
           expectedLayers > 0,
           comparisonPreviewRows.contains(where: { ($0.hookedMOELayers ?? 0) < expectedLayers }) {
            issues.append("Comparison gate: vMLX hook coverage does not cover every routed layer. Rerun Compare Suite with full BF16/vMLX tracing.")
        }
        if comparisonPreviewRows.contains(where: { row in
            guard let expectedMOELayers = row.expectedMOELayers, expectedMOELayers > 0 else { return false }
            return (row.hookedMOELayers ?? 0) < expectedMOELayers
        }) {
            issues.append("Comparison gate: vMLX hook coverage does not cover every config-routed MoE layer. Rerun Compare Suite with full BF16/vMLX tracing.")
        }
        if comparisonPreviewRows.contains(where: { ($0.sourceModelPath ?? "").isEmpty }) {
            issues.append("Comparison gate: eval rows are missing source model path evidence. Rerun Compare Suite with current Studio before pruning.")
        } else if let sourcePath = canonicalPlanSourceModelPath,
                  comparisonPreviewRows.contains(where: { Self.canonicalPath($0.sourceModelPath ?? "") != sourcePath }) {
            issues.append("Comparison gate: eval source path does not match the selected BF16/F16 source. Rerun Compare Suite from the original source.")
        }
        if comparisonPreviewRows.contains(where: { $0.maskApplied != true }) {
            issues.append("Comparison gate: masked compare did not record an applied BF16/vMLX mask. Rerun Compare Suite with the selected mask.")
        }
        if comparisonPreviewRows.contains(where: { $0.maskApplied == true && ($0.disabledExpertCount ?? 0) <= 0 }) {
            issues.append("Comparison gate: reviewed prune requires disabled expert evidence in every masked eval row; top-k-only comparisons cannot authorize hard pruning.")
        }
        return issues
    }

    private func currentComparisonArtifactIntegrityIssue(
        comparisonSummary: ExpertComparisonSummary
    ) -> String? {
        guard let lastEvalDirectory,
              let lastRunDirectory,
              !comparisonPreviewRows.isEmpty,
              let currentIndex = currentEvalIndexSummary(comparisonSummary: comparisonSummary) else {
            return "Comparison gate: persisted eval artifacts are missing. Rerun Compare Suite before pruning."
        }

        let fm = FileManager.default
        let suiteURL = lastRunDirectory.appendingPathComponent("suite.jsonl")
        guard fm.isReadableFile(atPath: suiteURL.path) else {
            return "Comparison gate: suite.jsonl evidence is missing. Rerun Trace Suite and Compare Suite before pruning."
        }
        for filename in ["comparison_summary.json", "eval.jsonl", "eval_index.json", "mask.json"] {
            let url = lastEvalDirectory.appendingPathComponent(filename)
            guard fm.isReadableFile(atPath: url.path) else {
                return "Comparison gate: \(filename) evidence is missing. Rerun Compare Suite before pruning."
            }
        }
        let maskURL = lastEvalDirectory.appendingPathComponent("mask.json")
        let maskDecoder = JSONDecoder()
        guard let maskData = try? Data(contentsOf: maskURL),
              let persistedMask = try? maskDecoder.decode(JANGKit.ExpertMask.self, from: maskData) else {
            return "Comparison gate: mask.json is unreadable."
        }
        guard Self.comparisonMask(persistedMask, matches: currentRuntimeMask()) else {
            return "Comparison gate: mask.json does not match the selected BF16/vMLX mask. Rerun Compare Suite with the current mask."
        }
        let persistedDisabledCount = persistedMask.layers.values.reduce(0) { $0 + $1.count }
        guard persistedDisabledCount > 0 else {
            return "Comparison gate: mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning."
        }
        if let indexedDisabledCount = currentIndex.disabledExpertCount,
           indexedDisabledCount != persistedDisabledCount {
            return "Comparison gate: mask.json disabled expert count does not match eval.jsonl."
        }

        let evalURL = lastEvalDirectory.appendingPathComponent("eval.jsonl")
        guard let persistedEvalObjects = Self.jsonlObjects(from: evalURL),
              !persistedEvalObjects.isEmpty else {
            return "Comparison gate: eval.jsonl is unreadable."
        }
        if persistedEvalObjects.count != comparisonSummary.promptCount {
            return "Comparison gate: eval.jsonl covers \(persistedEvalObjects.count) of \(comparisonSummary.promptCount) compared prompts."
        }
        let persistedObjectPromptIDs = persistedEvalObjects.compactMap {
            Self.jsonStringValue($0, keys: ["promptID", "prompt_id", "id"])
        }
        guard persistedObjectPromptIDs.count == persistedEvalObjects.count else {
            return "Comparison gate: eval.jsonl prompt IDs are unreadable."
        }
        let loadedPromptIDs = comparisonPreviewRows.map(\.promptID)
        if persistedObjectPromptIDs != loadedPromptIDs {
            return "Comparison gate: persisted eval.jsonl prompt order does not match loaded comparison rows."
        }
        if let issue = Self.evalRowEvidenceIssue(
            rows: persistedEvalObjects,
            expectedSourcePath: canonicalPlanSourceModelPath,
            sourceMismatchIssue: "Comparison gate: eval.jsonl source model path does not match the reviewed BF16/F16 source.",
            issuePrefix: "Comparison gate"
        ) {
            return issue
        }

        guard let persistedRows = Self.strictJSONLRecords(StoredEvalRecord.self, from: evalURL),
              !persistedRows.isEmpty else {
            return "Comparison gate: eval.jsonl is unreadable."
        }
        if persistedRows.count != comparisonSummary.promptCount {
            return "Comparison gate: eval.jsonl covers \(persistedRows.count) of \(comparisonSummary.promptCount) compared prompts."
        }
        let persistedPromptIDs = persistedRows.map(\.promptID)
        if persistedPromptIDs != loadedPromptIDs {
            return "Comparison gate: persisted eval.jsonl prompt order does not match loaded comparison rows."
        }
        guard Self.generationSettingsChecked(persistedRows) else {
            return "Comparison gate: eval.jsonl is missing baseline/masked decode settings evidence."
        }
        if let issue = currentComparisonSummaryIntegrityIssue(
            comparisonSummary,
            persistedRows: persistedRows
        ) {
            return issue
        }

        let indexURL = lastEvalDirectory.appendingPathComponent("eval_index.json")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let indexData = try? Data(contentsOf: indexURL),
              let persistedIndex = try? decoder.decode(StoredEvalIndex.self, from: indexData) else {
            return "Comparison gate: eval_index.json is unreadable."
        }
        if persistedIndex.promptCount != currentIndex.promptCount {
            return "Comparison gate: eval_index.json covers \(persistedIndex.promptCount) of \(currentIndex.promptCount) loaded comparison prompts."
        }
        if persistedIndex.promptIDs != currentIndex.promptIDs {
            return "Comparison gate: eval_index.json prompt order does not match loaded comparison rows."
        }
        guard let currentSuiteSHA256 = currentIndex.suiteSHA256 else {
            return "Comparison gate: suite.jsonl fingerprint could not be computed. Rerun Trace Suite and Compare Suite before pruning."
        }
        guard let persistedSuiteSHA256 = persistedIndex.suiteSHA256,
              !persistedSuiteSHA256.isEmpty else {
            return "Comparison gate: eval_index.json is missing reviewed prompt suite fingerprint."
        }
        if persistedSuiteSHA256 != currentSuiteSHA256 {
            return "Comparison gate: eval_index.json reviewed prompt suite fingerprint does not match loaded suite."
        }
        if persistedIndex.riskyPromptIDs != currentIndex.riskyPromptIDs {
            return "Comparison gate: eval_index.json risky prompt IDs do not match loaded comparison rows."
        }
        if Set(persistedIndex.highRiskDomains) != Set(currentIndex.highRiskDomains) {
            return "Comparison gate: eval_index.json high-risk domains do not match loaded comparison rows."
        }
        if let issue = Self.evalIndexSemanticCoverageIssue(
            semanticCoverage: persistedIndex.semanticCoverage,
            missingSemanticCoverage: persistedIndex.missingSemanticCoverage,
            evidenceName: "Comparison gate: eval_index.json"
        ) {
            return issue
        }
        if Set(persistedIndex.semanticCoverage ?? []) != Set(currentIndex.semanticCoverage ?? [])
            || Set(persistedIndex.missingSemanticCoverage ?? []) != Set(currentIndex.missingSemanticCoverage ?? []) {
            return "Comparison gate: eval_index.json semantic coverage does not match eval.jsonl."
        }
        guard persistedIndex.generationSettingsChecked == true else {
            return "Comparison gate: eval_index.json is missing decode settings evidence."
        }
        if persistedIndex.generationSettingsChecked != currentIndex.generationSettingsChecked {
            return "Comparison gate: eval_index.json decode settings evidence does not match eval.jsonl."
        }
        if persistedIndex.baselineRouteRecordCount != currentIndex.baselineRouteRecordCount
            || persistedIndex.maskedRouteRecordCount != currentIndex.maskedRouteRecordCount {
            return "Comparison gate: eval_index.json routing record counts do not match eval.jsonl."
        }
        if persistedIndex.baselineLayerStatsPromptCount != currentIndex.baselineLayerStatsPromptCount
            || persistedIndex.maskedLayerStatsPromptCount != currentIndex.maskedLayerStatsPromptCount {
            return "Comparison gate: eval_index.json routed-layer stats coverage does not match eval.jsonl."
        }
        if persistedIndex.runtimeMode != currentIndex.runtimeMode
            || persistedIndex.runtimeBackend != currentIndex.runtimeBackend
            || persistedIndex.runtimeDevice != currentIndex.runtimeDevice
            || persistedIndex.runtimeMetalEnabled != currentIndex.runtimeMetalEnabled {
            return "Comparison gate: eval_index.json runtime evidence does not match eval.jsonl."
        }
        if persistedIndex.sourceModelPath != currentIndex.sourceModelPath {
            return "Comparison gate: eval_index.json source model path does not match eval.jsonl."
        }
        if persistedIndex.maskApplied != currentIndex.maskApplied
            || persistedIndex.disabledExpertCount != currentIndex.disabledExpertCount
            || persistedIndex.topKOverride != currentIndex.topKOverride {
            return "Comparison gate: eval_index.json mask evidence does not match eval.jsonl."
        }
        guard persistedIndex.evalJSONL == "eval.jsonl",
              persistedIndex.evalTraceJSONL == "eval_trace.jsonl",
              persistedIndex.comparisonSummary == "comparison_summary.json",
              persistedIndex.mask == "mask.json",
              persistedIndex.maskJSON == "mask.json" else {
            return "Comparison gate: eval_index.json does not point to the persisted same-suite eval artifacts."
        }
        return nil
    }

    private func currentComparisonSummaryIntegrityIssue(
        _ summary: ExpertComparisonSummary,
        persistedRows: [StoredEvalRecord]
    ) -> String? {
        let prefix = "Comparison gate: comparison_summary.json"
        if summary.promptCount != persistedRows.count {
            return "\(prefix) covers \(summary.promptCount) of \(persistedRows.count) persisted eval rows."
        }
        if let lastRunDirectory,
           summary.baselineRunID != lastRunDirectory.lastPathComponent {
            return "\(prefix) baseline run ID does not match the loaded trace run."
        }
        if let lastEvalDirectory,
           summary.maskID != lastEvalDirectory.lastPathComponent {
            return "\(prefix) mask ID does not match the persisted eval directory."
        }
        let expectedBaseline = Self.passRate(persistedRows.map(\.baselinePassed))
        if !Self.doubleEqual(summary.passRateBaseline, expectedBaseline) {
            return "\(prefix) baseline pass rate does not match eval.jsonl."
        }
        let expectedMasked = Self.passRate(persistedRows.map(\.maskedPassed))
        if !Self.doubleEqual(summary.passRateMasked, expectedMasked) {
            return "\(prefix) masked pass rate does not match eval.jsonl."
        }
        let expectedMeanDelta = Self.mean(persistedRows.map(\.textDelta))
        if !Self.doubleEqual(summary.meanTextDelta, expectedMeanDelta) {
            return "\(prefix) mean text delta does not match eval.jsonl."
        }
        let expectedMeanLatency = Self.mean(persistedRows.map(\.latencyDeltaPct))
        if !Self.doubleEqual(summary.meanLatencyDeltaPct, expectedMeanLatency) {
            return "\(prefix) mean latency delta does not match eval.jsonl."
        }
        let expectedHighRiskDomains = Self.highRiskDomains(from: persistedRows)
        if Set(summary.highRiskDomains) != Set(expectedHighRiskDomains) {
            return "\(prefix) high-risk domains do not match eval.jsonl."
        }
        guard let recordedSeverity = summary.regressionSeverity,
              !recordedSeverity.isEmpty else {
            return "\(prefix) is missing regression severity evidence."
        }
        let expectedSeverity = Self.regressionSeverity(from: persistedRows)
        if recordedSeverity != expectedSeverity {
            return "\(prefix) regression severity does not match eval.jsonl."
        }
        let expectedBaselineQualified = Self.baselineQualifiedRecords(persistedRows)
        let expectedBaselineQualifiedCoverage = Self.baselineQualifiedSemanticCoverage(from: persistedRows)
        let expectedMissingCoverage = Self.missingBaselineQualifiedSemanticCoverage(
            for: expectedBaselineQualifiedCoverage
        )
        let expectedDegradedPromptIDs = Self.degradedPromptIDs(from: persistedRows)
        let expectedSafeDropCandidates = expectedHighRiskDomains.isEmpty
            && expectedDegradedPromptIDs.isEmpty
            && !expectedBaselineQualified.isEmpty
            && expectedMissingCoverage.isEmpty
            ? comparisonCandidateCoordinates()
            : []
        if Set(summary.safeDropCandidates) != Set(expectedSafeDropCandidates) {
            return "\(prefix) safe-drop candidates do not match the current mask and eval rows."
        }
        if summary.validatorAvailablePromptCount != persistedRows.filter({ $0.validatorAvailable == true }).count
            || summary.baselineQualifiedPromptCount != expectedBaselineQualified.count
            || Set(summary.degradedPromptIDs ?? []) != Set(expectedDegradedPromptIDs)
            || Set(summary.missingBaselineQualifiedSemanticCoverage ?? []) != Set(expectedMissingCoverage) {
            return "\(prefix) validator classification evidence does not match eval.jsonl."
        }
        return nil
    }

    private func currentComparisonTraceIntegrityIssue(
        comparisonSummary: ExpertComparisonSummary
    ) -> String? {
        guard let lastEvalDirectory,
              let evalIndex = currentEvalIndexSummary(comparisonSummary: comparisonSummary) else {
            return "Comparison gate: eval_trace.jsonl evidence is missing. Rerun Compare Suite before pruning."
        }
        let evalTraceURL = lastEvalDirectory.appendingPathComponent("eval_trace.jsonl")
        let expectedMask = currentRuntimeMask()
        return Self.evalTraceIntegrityIssue(
            evalIndex: evalIndex,
            evalTraceURL: evalTraceURL,
            issuePrefix: "Comparison gate",
            disabledExpertCount: evalIndex.disabledExpertCount,
            topKOverride: evalIndex.topKOverride,
            expectedDisabledByLayer: expectedMask.layers
        )
    }

    private func comparisonPromptIDCoverageIssue() -> String? {
        guard !comparisonPreviewRows.isEmpty else { return nil }
        let tracedPromptIDs = tracedPromptIDsForReview()
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !tracedPromptIDs.isEmpty else { return nil }
        if let duplicate = Self.firstDuplicatePromptID(in: tracedPromptIDs) {
            return "Comparison gate: traced prompt suite contains duplicate prompt IDs: \(duplicate). Rerun Trace Suite with unique prompt IDs."
        }
        let comparedPromptIDs = comparisonPreviewRows
            .map { $0.promptID.trimmingCharacters(in: .whitespacesAndNewlines) }
        if let duplicate = Self.firstDuplicatePromptID(in: comparedPromptIDs) {
            return "Comparison gate: compared eval rows contain duplicate prompt IDs: \(duplicate). Rerun Compare Suite for this trace run."
        }
        let traced = Set(tracedPromptIDs)
        let compared = Set(comparedPromptIDs)
        let missing = traced.subtracting(compared)
        let unexpected = compared.subtracting(traced)
        let orderMismatch = comparedPromptIDs != tracedPromptIDs
        guard !missing.isEmpty || !unexpected.isEmpty || comparedPromptIDs.count != tracedPromptIDs.count || orderMismatch else {
            return nil
        }

        var details: [String] = []
        if comparedPromptIDs.count != tracedPromptIDs.count {
            details.append("\(comparedPromptIDs.count)/\(tracedPromptIDs.count) prompts")
        }
        if !missing.isEmpty {
            details.append("missing \(Self.promptIDPreview(missing))")
        }
        if !unexpected.isEmpty {
            details.append("unexpected \(Self.promptIDPreview(unexpected))")
        }
        if missing.isEmpty && unexpected.isEmpty && comparedPromptIDs.count == tracedPromptIDs.count && orderMismatch {
            details.append("order mismatch")
        }
        return "Comparison gate: compared prompt IDs do not match the traced suite (\(details.joined(separator: ", "))). Rerun Compare Suite for this trace run."
    }

    private func tracedPromptIDsForReview() -> [String] {
        if !runs.isEmpty {
            return runs.map(\.prompt.id)
        }
        if let lastRunDirectory {
            let suiteURL = lastRunDirectory.appendingPathComponent("suite.jsonl")
            if FileManager.default.fileExists(atPath: suiteURL.path),
               let suite = try? ExpertPromptSuite.loadJSONL(name: selectedSuite.name, from: suiteURL) {
                return suite.prompts.map(\.id)
            }
        }
        return selectedSuite.prompts.map(\.id)
    }

    private func reviewedPrunePromptsForCoverage() -> [ExpertPrompt] {
        if !runs.isEmpty {
            return runs.map(\.prompt)
        }
        if let lastRunDirectory {
            let suiteURL = lastRunDirectory.appendingPathComponent("suite.jsonl")
            if FileManager.default.fileExists(atPath: suiteURL.path),
               let suite = try? ExpertPromptSuite.loadJSONL(name: selectedSuite.name, from: suiteURL) {
                return suite.prompts
            }
        }
        return selectedSuite.prompts
    }

    private nonisolated static func promptIDPreview(_ ids: Set<String>) -> String {
        let sorted = ids.sorted()
        let head = sorted.prefix(5).joined(separator: ", ")
        let remaining = sorted.count - min(sorted.count, 5)
        return remaining > 0 ? "\(head), +\(remaining) more" : head
    }

    private nonisolated static func firstDuplicatePromptID(in ids: [String]) -> String? {
        var seen = Set<String>()
        for rawID in ids {
            let id = rawID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !id.isEmpty else { continue }
            if !seen.insert(id).inserted {
                return id
            }
        }
        return nil
    }

    private nonisolated static func expertCoordinatePreview(_ coordinates: [String]) -> String {
        let head = coordinates.prefix(5).joined(separator: ", ")
        let remaining = coordinates.count - min(coordinates.count, 5)
        return remaining > 0 ? "\(head), +\(remaining) more" : head
    }

    var canGenerateReviewedPrunePlan: Bool {
        atlas != nil
            && capability.expectedExperts != nil
            && reviewedPruneAuthorityIssues.isEmpty
            && reviewedPruneAtlasIssues.isEmpty
            && !hasBlockingPlanIssue
            && hasReviewedPruneCoverage
            && reviewedPruneSemanticEvidenceIssues.isEmpty
            && hasComparisonSummary
            && reviewedPruneComparisonIssues.isEmpty
    }

    var reviewedPruneReadiness: (message: String, systemImage: String, isReady: Bool) {
        if canGenerateReviewedPrunePlan {
            return (
                "Reviewed prune ready: BF16/vMLX authority, coverage, semantic evidence, A/B-safe drops, masked compare, token depth, runtime device, vMLX hook coverage, and plan safety gates passed.",
                "checkmark.seal.fill",
                true
            )
        }
        if let firstIssue = reviewedPruneAuthorityIssues.first {
            return (
                "Reviewed prune blocked: \(firstIssue)",
                "exclamationmark.triangle.fill",
                false
            )
        }
        guard atlas != nil else {
            return (
                "Reviewed prune waiting: run a prompt suite trace to build the full expert atlas.",
                "waveform.path.ecg",
                false
            )
        }
        guard hasComparisonSummary else {
            return (
                "Reviewed prune blocked: run Compare Suite so baseline and masked outputs are saved for the traced prompts.",
                "rectangle.split.2x1",
                false
            )
        }
        let planIssues = planValidationIssues
            .filter { $0.severity == .error }
            .map(\.message)
        if let firstIssue = (
            reviewedPruneAuthorityIssues
            + reviewedPruneAtlasIssues
            + reviewedPruneCoverageIssues
            + reviewedPruneSemanticEvidenceIssues
            + reviewedPruneComparisonIssues
            + planIssues
        ).first {
            return (
                "Reviewed prune blocked: \(firstIssue)",
                "exclamationmark.triangle.fill",
                false
            )
        }
        return (
            "Reviewed prune blocked: review the current trace, mask, and comparison evidence before pruning.",
            "exclamationmark.triangle.fill",
            false
        )
    }

    var atlasSummary: String {
        guard let atlas else { return "No trace run yet." }
        let hot = atlas.experts.filter(\.isHot).count
        let dead = atlas.experts.filter(\.isDead).count
        if let layers = capability.expectedLayers, let experts = capability.expectedExperts {
            let expectedCells = layers * experts
            let actualCells = atlas.experts.count
            let gridState = actualCells == expectedCells ? "complete grid" : "incomplete grid"
            return "\(atlas.promptCount) prompts, \(gridState): \(layers) layers x \(experts) experts (\(actualCells)/\(expectedCells) cells), \(hot) hot, \(dead) dead"
        }
        return "\(atlas.promptCount) prompts, \(atlas.experts.count) experts, \(hot) hot, \(dead) dead"
    }

    var pruneKeepRange: ClosedRange<Int> {
        let experts = capability.expectedExperts ?? 1
        let maxKeep = max(experts - 1, 1)
        let minKeep = min(max(1, experts / 2), maxKeep)
        return minKeep...maxKeep
    }

    private var hotExpertCoordinates: Set<ExpertCoordinate> {
        Set((atlas?.experts ?? []).filter(\.isHot).map { ExpertCoordinate(layer: $0.layer, expert: $0.expert) })
    }

    var filteredLayers: [Int] {
        Array(Set(filteredAtlasEntries.map(\.layer))).sorted()
    }

    var atlasGridRows: [ExpertAtlasLayerRow] {
        let grouped = Dictionary(grouping: filteredAtlasEntries, by: \.layer)
        return grouped.keys.sorted().map { layer in
            ExpertAtlasLayerRow(
                layer: layer,
                entries: (grouped[layer] ?? []).sorted { lhs, rhs in
                    lhs.expert < rhs.expert
                }
            )
        }
    }

    var atlasLayerMetricScales: [Int: ExpertAtlasMetricScale] {
        Self.metricScales(for: atlas?.experts ?? [])
    }

    var atlasTableEntries: [ExpertAtlasEntry] {
        filteredAtlasEntries.sorted { lhs, rhs in
            sortedAtlasOrder(lhs, rhs)
        }
    }

    var hasAtlasQuery: Bool {
        !atlasLayerFilterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !atlasExpertFilterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !atlasDomainFilterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !atlasPromptFilterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func clearAtlasQuery() {
        atlasLayerFilterText = ""
        atlasExpertFilterText = ""
        atlasDomainFilterText = ""
        atlasPromptFilterText = ""
    }

    var filteredAtlasEntries: [ExpertAtlasEntry] {
        guard let atlas else { return [] }
        return atlas.experts.filter { entry in
            guard Self.matchesIntegerFilter(entry.layer, text: atlasLayerFilterText),
                  Self.matchesIntegerFilter(entry.expert, text: atlasExpertFilterText),
                  Self.matchesDomainFilter(entry, text: atlasDomainFilterText),
                  Self.matchesPromptFilter(entry, text: atlasPromptFilterText) else {
                return false
            }
            switch atlasFilter {
            case .all:
                return true
            case .active:
                return !entry.isDead
            case .hot:
                return entry.isHot
            case .dead:
                return entry.isDead
            case .masked:
                return isMasked(layer: entry.layer, expert: entry.expert)
                    || isDropCandidate(layer: entry.layer, expert: entry.expert)
            case .drops:
                return isDropCandidate(layer: entry.layer, expert: entry.expert)
            case .locked:
                return isLockedKeep(layer: entry.layer, expert: entry.expert)
            case .safeDrop:
                return isSafeDropCandidate(layer: entry.layer, expert: entry.expert)
            case .safety:
                return Self.matchesDomainFilter(entry, text: "safety")
            case .coding:
                return Self.matchesDomainFilter(entry, text: "coding")
            case .math:
                return Self.matchesDomainFilter(entry, text: "math")
            case .reasoning:
                return Self.matchesDomainFilter(entry, text: "reasoning")
            }
        }
    }

    private var maskLayers: [Int: Set<Int>] = [:]
    private var dropCandidates: [Int: Set<Int>] = [:]
    private var lockedKeeps: [Int: Set<Int>] = [:]

    func syncSelectedPrompt() {
        if !selectedSuite.prompts.contains(where: { $0.id == selectedPromptID }) {
            selectedPromptID = selectedSuite.prompts.first?.id ?? ""
        }
    }

    func title(for run: ExpertRunSummary) -> String {
        let status = run.failureStage == nil ? "" : " · partial"
        return "\(Self.runTitleDateFormatter.string(from: run.startedAt)) · \(run.suiteID) · \(run.promptCount)\(status)"
    }

    private func restoreGenerationPreview(from runDirectory: URL) {
        baselineText = ""
        maskedText = ""
        baselineOutputReady = false
        maskedOutputReady = false
        baselineTokensPerSecond = 0
        maskedTokensPerSecond = 0
        runtimeInfoSummary = ""

        let url = runDirectory.appendingPathComponent("generations.jsonl")
        guard let first = Self.firstJSONLRecord(StoredGenerationRecord.self, from: url) else {
            return
        }
        baselineText = first.text
        baselineOutputReady = true
        baselineTokensPerSecond = first.tokensPerSecond
        runtimeInfoSummary = Self.runtimeSummary(
            mode: first.runtimeMode,
            backend: first.runtimeBackend,
            device: first.runtimeDevice,
            metalEnabled: first.runtimeMetalEnabled,
            jangToolsVersion: first.jangToolsVersion,
            mlxVersion: first.mlxVersion,
            mlxLMVersion: first.mlxLMVersion,
            sourceModelPath: first.sourceModelPath,
            hookedMOELayers: first.hookedMOELayers,
            expectedMOELayers: first.expectedMOELayers,
            hookCoverageComplete: first.hookCoverageComplete
        )
    }

    private nonisolated static func restoredRuns(from runDirectory: URL, suite: ExpertPromptSuite) -> [ExpertPromptRun] {
        let generationURL = runDirectory.appendingPathComponent("generations.jsonl")
        let generationRecords = jsonlRecords(StoredGenerationRecord.self, from: generationURL)
        guard !generationRecords.isEmpty else { return [] }
        let generationIDs = generationRecords.map { $0.promptID.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !generationIDs.contains(where: \.isEmpty),
              firstDuplicatePromptID(in: generationIDs) == nil else {
            return []
        }
        let suiteIDs = suite.prompts.map { $0.id.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !suiteIDs.contains(where: \.isEmpty),
              firstDuplicatePromptID(in: suiteIDs) == nil else {
            return []
        }

        var generationByPrompt: [String: StoredGenerationRecord] = [:]
        for record in generationRecords where generationByPrompt[record.promptID] == nil {
            generationByPrompt[record.promptID] = record
        }

        let traceURL = runDirectory.appendingPathComponent("trace.jsonl")
        let traceByPrompt = Dictionary(
            grouping: jsonlRecords(StoredTraceRecord.self, from: traceURL),
            by: \.promptID
        ).mapValues { rows in rows.map(\.record) }

        return suite.prompts.compactMap { prompt in
            guard let generation = generationByPrompt[prompt.id] else { return nil }
            let traces = traceByPrompt[prompt.id] ?? []
            let elapsedSeconds = generation.tokensPerSecond > 0
                ? Double(generation.tokenCount) / generation.tokensPerSecond
                : 0
            return ExpertPromptRun(
                prompt: prompt,
                result: JANGKit.ExpertRunResult(
                    text: generation.text,
                    tokens: generation.tokenCount,
                    elapsedSeconds: elapsedSeconds,
                    tokensPerSecond: generation.tokensPerSecond,
                    finishReason: finishReason(from: generation.finishReason),
                    layerStats: restoredLayerStats(from: generation, traces: traces),
                    tokenTrace: traces.isEmpty ? nil : traces,
                    runtimeInfo: runtimeInfo(from: generation)
                )
            )
        }
    }

    private nonisolated static func restoredRunEvidenceIssue(from runDirectory: URL, suite: ExpertPromptSuite) -> String? {
        let suiteIDs = suite.prompts.map { $0.id.trimmingCharacters(in: .whitespacesAndNewlines) }
        if suiteIDs.contains(where: \.isEmpty) {
            return "suite.jsonl contains empty prompt IDs; persisted prompt evidence is ambiguous."
        }
        if let duplicate = firstDuplicatePromptID(in: suiteIDs) {
            return "suite.jsonl contains duplicate prompt IDs: \(duplicate); persisted prompt evidence is ambiguous."
        }
        let generationURL = runDirectory.appendingPathComponent("generations.jsonl")
        let generationRecords = jsonlRecords(StoredGenerationRecord.self, from: generationURL)
        guard !generationRecords.isEmpty else { return nil }
        let generationIDs = generationRecords.map { $0.promptID.trimmingCharacters(in: .whitespacesAndNewlines) }
        if generationIDs.contains(where: \.isEmpty) {
            return "generations.jsonl contains empty prompt IDs; persisted prompt evidence is ambiguous."
        }
        if let duplicate = firstDuplicatePromptID(in: generationIDs) {
            return "generations.jsonl contains duplicate prompt IDs: \(duplicate); persisted prompt evidence is ambiguous."
        }
        return nil
    }

    private nonisolated static func restoredLayerStats(
        from generation: StoredGenerationRecord,
        traces: [JANGKit.ExpertRouteRecord]
    ) -> [JANGKit.ExpertLayerStats] {
        if let layerStats = generation.layerStats, !layerStats.isEmpty {
            return layerStats
        }
        return layerStats(from: traces)
    }

    private nonisolated static func runtimeInfo(from generation: StoredGenerationRecord) -> JANGKit.ModelRuntimeInfo? {
        guard let runtimeMode = generation.runtimeMode,
              !runtimeMode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let backend = generation.runtimeBackend,
              !backend.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let deviceName = generation.runtimeDevice,
              !deviceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let runtimeMetalEnabled = generation.runtimeMetalEnabled else {
            return nil
        }
        return JANGKit.ModelRuntimeInfo(
            backend: backend,
            runtimeMode: runtimeMode,
            deviceName: deviceName,
            metalEnabled: runtimeMetalEnabled,
            jangToolsVersion: generation.jangToolsVersion,
            mlxVersion: generation.mlxVersion,
            mlxLMVersion: generation.mlxLMVersion,
            mlxVLMVersion: generation.mlxVLMVersion,
            sourceModelPath: generation.sourceModelPath,
            hookedMOELayers: generation.hookedMOELayers,
            expectedMOELayers: generation.expectedMOELayers,
            hookCoverageComplete: generation.hookCoverageComplete,
            maskApplied: generation.maskApplied,
            disabledExpertCount: generation.disabledExpertCount,
            topKOverride: generation.topKOverride
        )
    }

    private nonisolated static func layerStats(from traces: [JANGKit.ExpertRouteRecord]) -> [JANGKit.ExpertLayerStats] {
        guard !traces.isEmpty else { return [] }
        var tokenPositions: [Int: Set<Int>] = [:]
        var hitCounts: [Int: [Int: Int]] = [:]
        var probabilityMass: [Int: [Int: Float]] = [:]
        for trace in traces {
            tokenPositions[trace.layer, default: []].insert(trace.tokenIndex)
            for (slot, expert) in trace.selectedExperts.enumerated() {
                hitCounts[trace.layer, default: [:]][expert, default: 0] += 1
                let score = slot < trace.scores.count ? trace.scores[slot] : 1
                probabilityMass[trace.layer, default: [:]][expert, default: 0] += score
            }
        }
        return tokenPositions.keys.sorted().map { layer in
            JANGKit.ExpertLayerStats(
                layer: layer,
                tokenCount: tokenPositions[layer]?.count ?? 0,
                hitCounts: hitCounts[layer] ?? [:],
                probabilityMass: probabilityMass[layer] ?? [:]
            )
        }
    }



    private func restoreLatestComparison(from runDirectory: URL) {
        lastEvalDirectory = nil
        lastEvalSummary = ""
        comparisonPreviewRows = []
        let evalsDir = runDirectory.appendingPathComponent("evals", isDirectory: true)
        guard let latest = Self.latestComparisonDirectory(in: evalsDir) else {
            return
        }
        guard comparisonMaskMatchesCurrent(latest) else {
            lastEvalSummary = "Latest saved eval was for a different mask. Rerun Compare Suite before pruning."
            return
        }
        lastEvalDirectory = latest

        if let summary = Self.loadComparisonSummary(from: latest) {
            lastEvalSummary = Self.comparisonSummaryText(summary)
        }
        comparisonPreviewRows = Self.jsonlRecords(StoredEvalRecord.self, from: latest.appendingPathComponent("eval.jsonl"))
        backfillEvalIndexIfNeeded(
            in: latest,
            runDirectory: runDirectory,
            records: comparisonPreviewRows
        )
        if let first = comparisonPreviewRows.first {
            baselineText = first.baselineText
            maskedText = first.maskedText
            baselineOutputReady = true
            maskedOutputReady = true
            baselineTokensPerSecond = first.baselineTokensPerSecond
            maskedTokensPerSecond = first.maskedTokensPerSecond
            runtimeInfoSummary = Self.runtimeSummary(
                mode: first.runtimeMode,
                backend: first.runtimeBackend,
                device: first.runtimeDevice,
                metalEnabled: first.runtimeMetalEnabled,
                jangToolsVersion: first.jangToolsVersion,
                mlxVersion: first.mlxVersion,
                mlxLMVersion: first.mlxLMVersion,
                sourceModelPath: first.sourceModelPath,
                hookedMOELayers: first.hookedMOELayers,
                expectedMOELayers: first.expectedMOELayers,
                hookCoverageComplete: first.hookCoverageComplete
            )
            if lastEvalSummary.isEmpty {
                lastEvalSummary = "Loaded eval: \(first.promptID), delta \(String(format: "%.2f", first.textDelta)), risk \(first.risk)"
            }
        }
    }

    private nonisolated static func latestComparisonDirectory(in evalsDirectory: URL) -> URL? {
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

    private nonisolated static func loadComparisonSummary(from directory: URL) -> ExpertComparisonSummary? {
        let url = directory.appendingPathComponent("comparison_summary.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ExpertComparisonSummary.self, from: data)
    }

    private func backfillEvalIndexIfNeeded(
        in directory: URL,
        runDirectory: URL,
        records: [StoredEvalRecord]
    ) {
        let indexURL = directory.appendingPathComponent("eval_index.json")
        guard !FileManager.default.fileExists(atPath: indexURL.path),
              !records.isEmpty,
              let summary = Self.loadComparisonSummary(from: directory) else {
            return
        }
        do {
            try Self.writeEvalIndex(
                to: indexURL,
                runID: runDirectory.lastPathComponent,
                maskID: directory.lastPathComponent,
                summary: summary,
                records: records,
                suiteURL: runDirectory.appendingPathComponent("suite.jsonl")
            )
        } catch {
            lastEvalSummary = lastEvalSummary.isEmpty
                ? "Loaded eval, but could not rebuild eval_index.json: \(error.localizedDescription)"
                : "\(lastEvalSummary). Could not rebuild eval_index.json: \(error.localizedDescription)"
        }
    }

    private func comparisonMaskMatchesCurrent(_ directory: URL) -> Bool {
        let url = directory.appendingPathComponent("mask.json")
        guard let data = try? Data(contentsOf: url),
              let saved = try? JSONDecoder().decode(JANGKit.ExpertMask.self, from: data)
        else {
            return false
        }
        return Self.comparisonMask(saved, matches: currentRuntimeMask())
    }

    private nonisolated static func comparisonMask(_ lhs: JANGKit.ExpertMask, matches rhs: JANGKit.ExpertMask) -> Bool {
        lhs.layers == rhs.layers
            && lhs.topKOverride == rhs.topKOverride
    }

    private nonisolated static func firstJSONLRecord<T: Decodable>(_ type: T.Type, from url: URL) -> T? {
        jsonlRecords(type, from: url).first
    }

    private nonisolated static func jsonlRecords<T: Decodable>(_ type: T.Type, from url: URL) -> [T] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        var records: [T] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if let record = try? decoder.decode(T.self, from: Data(trimmed.utf8)) {
                records.append(record)
            }
        }
        return records
    }

    private nonisolated static func strictJSONLRecords<T: Decodable>(_ type: T.Type, from url: URL) -> [T]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let decoder = JSONDecoder()
        var records: [T] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard let record = try? decoder.decode(T.self, from: Data(trimmed.utf8)) else {
                return nil
            }
            records.append(record)
        }
        return records
    }

    private nonisolated static func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(whereSeparator: \.isNewline).count
    }

    private nonisolated static func comparisonSummaryText(_ summary: ExpertComparisonSummary) -> String {
        let risk = summary.highRiskDomains.isEmpty
            ? "no high-risk domains"
            : "risk \(summary.highRiskDomains.joined(separator: ", "))"
        return String(
            format: "Loaded suite eval: %d prompts, baseline %@, masked %@, mean delta %.2f, %@",
            summary.promptCount,
            formatPassRate(summary.passRateBaseline),
            formatPassRate(summary.passRateMasked),
            summary.meanTextDelta,
            risk
        )
    }





    func reloadRunHistory() {
        do {
            runHistory = try ExpertRunStore.listRuns(
                rootDirectory: artifactRoot(),
                matchingSourcePath: artifactSourcePath,
                reviewBundlePath: artifactReviewBundlePath
            )
            if selectedRunID.isEmpty || !runHistory.contains(where: { $0.runID == selectedRunID }) {
                selectedRunID = runHistory.first?.runID ?? ""
            }
            refreshSelectedRunEvidence()
        } catch {
            lastError = "Could not read Expert Lab runs: \(error.localizedDescription)"
        }
    }
    func refreshSelectedRunEvidence() {
        guard let selectedRunSummary else {
            selectedRunEvidenceSummary = ""
            selectedRunArtifactSummary = ""
            selectedRunEvidenceWarning = ""
            return
        }
        let runtimeSummary = Self.selectedRunRuntimeEvidenceSummary(selectedRunSummary)
        let base = "\(selectedRunSummary.promptCount) prompts · \(selectedRunSummary.suiteID)\(runtimeSummary)"
        let runDirectory = URL(fileURLWithPath: selectedRunSummary.directoryPath, isDirectory: true)
        selectedRunArtifactSummary = Self.selectedRunArtifactEvidenceSummary(runDirectory: runDirectory)
        selectedRunEvidenceWarning = [
            Self.selectedRunAuthorityWarning(selectedRunSummary),
            Self.selectedRunPrunePlanWarning(runDirectory: runDirectory)
        ]
        .filter { !$0.isEmpty }
        .joined(separator: " ")
        let atlasURL = runDirectory.appendingPathComponent("atlas.json")
        guard let atlasData = try? Data(contentsOf: atlasURL),
              let decodedAtlas = try? JSONDecoder().decode(ExpertAtlas.self, from: atlasData) else {
            selectedRunEvidenceSummary = base
            return
        }
        let completed = completedAtlas(decodedAtlas)
        let actualCells = completed.experts.count
        let hot = completed.experts.filter(\.isHot).count
        let dead = completed.experts.filter(\.isDead).count
        let comparisonSummary = Self.selectedRunComparisonEvidenceSummary(
            runDirectory: runDirectory
        )
        let sourceGrid = Self.intLayerMap(completed.sourceNumExpertsByLayer)
        if let shape = Self.uniformGridShape(from: sourceGrid) {
            let gridState = actualCells == shape.expectedCells ? "complete grid" : "incomplete grid"
            selectedRunEvidenceSummary = "\(base) · \(gridState): \(shape.layers) layers x \(shape.experts) experts (\(actualCells)/\(shape.expectedCells) cells), \(hot) hot, \(dead) dead\(comparisonSummary)"
        } else if !sourceGrid.isEmpty {
            let expectedCells = sourceGrid.values.reduce(0, +)
            let gridState = actualCells == expectedCells ? "complete grid" : "incomplete grid"
            selectedRunEvidenceSummary = "\(base) · \(gridState): \(sourceGrid.count) layers, \(expectedCells) source expert cells (\(actualCells)/\(expectedCells) cells), \(hot) hot, \(dead) dead\(comparisonSummary)"
        } else if let layers = capability.expectedLayers,
                  let experts = capability.expectedExperts {
            let expectedCells = layers * experts
            let gridState = actualCells == expectedCells ? "complete grid" : "incomplete grid"
            selectedRunEvidenceSummary = "\(base) · \(gridState): \(layers) layers x \(experts) experts (\(actualCells)/\(expectedCells) cells), \(hot) hot, \(dead) dead\(comparisonSummary)"
        } else {
            let observedLayers = Set(completed.experts.map(\.layer)).count
            selectedRunEvidenceSummary = "\(base) · \(observedLayers) layers, \(actualCells) experts, \(hot) hot, \(dead) dead\(comparisonSummary)"
        }
    }
    private nonisolated static func selectedRunComparisonEvidenceSummary(runDirectory: URL) -> String {
        let evalsDir = runDirectory.appendingPathComponent("evals", isDirectory: true)
        guard let latest = latestComparisonDirectory(in: evalsDir) else { return "" }
        guard let summary = loadComparisonSummary(from: latest) else {
            return " · compare saved, summary unreadable"
        }
        var parts = [
            "compare \(summary.promptCount) prompts",
            "baseline \(formatPassRate(summary.passRateBaseline))",
            "masked \(formatPassRate(summary.passRateMasked))"
        ]
        let indexURL = latest.appendingPathComponent("eval_index.json")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let data = try? Data(contentsOf: indexURL),
           let index = try? decoder.decode(StoredEvalIndex.self, from: data) {
            if !index.riskyPromptIDs.isEmpty {
                parts.append("\(index.riskyPromptIDs.count) risky")
            }
            if !index.highRiskDomains.isEmpty {
                parts.append("risk \(index.highRiskDomains.sorted().joined(separator: ", "))")
            }
            let semantic = evalIndexSemanticCoverageSummary(
                semanticCoverage: index.semanticCoverage,
                missingSemanticCoverage: index.missingSemanticCoverage
            )
            if !semantic.isEmpty {
                parts.append(semantic)
            }
            if let baseline = index.meanBaselineTokens,
               let masked = index.meanMaskedTokens {
                parts.append(String(format: "avg tokens %.1f/%.1f", baseline, masked))
            } else {
                parts.append("token depth missing; rerun Compare Suite")
            }
            let runtime = runtimeSummary(
                mode: index.runtimeMode,
                backend: index.runtimeBackend,
                device: index.runtimeDevice,
                metalEnabled: index.runtimeMetalEnabled,
                jangToolsVersion: index.jangToolsVersion,
                mlxVersion: index.mlxVersion,
                mlxLMVersion: index.mlxLMVersion,
                sourceModelPath: index.sourceModelPath,
                hookedMOELayers: index.hookedMOELayers,
                expectedMOELayers: index.expectedMOELayers,
                hookCoverageComplete: index.hookCoverageComplete
            )
            if !runtime.isEmpty {
                parts.append("eval runtime \(runtime)")
                if index.runtimeDevice == nil {
                    parts.append("eval device not recorded")
                }
            } else {
                parts.append("eval runtime not recorded; rerun Compare Suite")
            }
        } else {
            parts.append("eval_index missing; rerun Compare Suite")
            if !summary.highRiskDomains.isEmpty {
                parts.append("risk \(summary.highRiskDomains.sorted().joined(separator: ", "))")
            }
        }
        return " · \(parts.joined(separator: ", "))"
    }

    private nonisolated static func evalIndexSemanticCoverageSummary(
        semanticCoverage: [String]?,
        missingSemanticCoverage: [String]?
    ) -> String {
        guard let semanticCoverage, !semanticCoverage.isEmpty else {
            return "semantic probes missing; rerun Compare Suite"
        }
        let coverage = Set(
            semanticCoverage
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let derivedMissing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(coverage)
        let recordedMissing = Set(
            (missingSemanticCoverage ?? [])
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let missing = derivedMissing.union(recordedMissing)
        guard missing.isEmpty else {
            return "semantic probes missing \(missing.sorted().joined(separator: ", "))"
        }
        return "semantic probes ready"
    }

    private nonisolated static func selectedRunArtifactEvidenceSummary(runDirectory: URL) -> String {
        let fm = FileManager.default
        var parts: [String] = []
        let runURL = runDirectory.appendingPathComponent("run.json")
        if fm.isReadableFile(atPath: runURL.path) {
            parts.append("run.json")
        }
        let suiteURL = runDirectory.appendingPathComponent("suite.jsonl")
        if fm.isReadableFile(atPath: suiteURL.path) {
            parts.append("suite.jsonl \(lineCount(suiteURL)) prompts")
        }
        let atlasURL = runDirectory.appendingPathComponent("atlas.json")
        if fm.isReadableFile(atPath: atlasURL.path) {
            parts.append("atlas.json")
        }
        let generationsURL = runDirectory.appendingPathComponent("generations.jsonl")
        if fm.isReadableFile(atPath: generationsURL.path) {
            parts.append("generations.jsonl \(lineCount(generationsURL)) rows")
        }
        let traceURL = runDirectory.appendingPathComponent("trace.jsonl")
        if fm.isReadableFile(atPath: traceURL.path) {
            parts.append("trace.jsonl \(lineCount(traceURL)) routes")
        }
        let sqliteURL = runDirectory.appendingPathComponent("trace.sqlite")
        if fm.isReadableFile(atPath: sqliteURL.path) {
            parts.append("trace.sqlite")
        }
        let evalsDir = runDirectory.appendingPathComponent("evals", isDirectory: true)
        if let latest = latestComparisonDirectory(in: evalsDir) {
            let comparisonSummaryURL = latest.appendingPathComponent("comparison_summary.json")
            let evalURL = latest.appendingPathComponent("eval.jsonl")
            let evalTraceURL = latest.appendingPathComponent("eval_trace.jsonl")
            let evalIndexURL = latest.appendingPathComponent("eval_index.json")
            let maskURL = latest.appendingPathComponent("mask.json")
            if fm.isReadableFile(atPath: comparisonSummaryURL.path) {
                parts.append("comparison_summary.json")
            }
            if fm.isReadableFile(atPath: evalURL.path) {
                parts.append("eval.jsonl \(lineCount(evalURL)) rows")
            }
            if fm.isReadableFile(atPath: evalTraceURL.path) {
                parts.append("eval_trace.jsonl \(lineCount(evalTraceURL)) routes")
            }
            if fm.isReadableFile(atPath: evalIndexURL.path) {
                parts.append("eval_index.json")
            }
            if fm.isReadableFile(atPath: maskURL.path) {
                parts.append("mask.json")
            }
            parts.append("evals/\(latest.lastPathComponent)")
        }
        guard !parts.isEmpty else { return "" }
        return "Artifacts: \(parts.joined(separator: ", "))"
    }

    private nonisolated static func selectedRunRuntimeEvidenceSummary(_ run: ExpertRunSummary) -> String {
        let runtime = runtimeSummary(
            mode: run.runtimeMode,
            backend: run.runtimeBackend,
            device: run.runtimeDevice,
            metalEnabled: run.runtimeMetalEnabled,
            jangToolsVersion: run.jangToolsVersion,
            mlxVersion: run.mlxVersion,
            mlxLMVersion: run.mlxLMVersion,
            sourceModelPath: run.sourceModelPath,
            hookedMOELayers: run.hookedMOELayers,
            expectedMOELayers: run.expectedMOELayers,
            hookCoverageComplete: run.hookCoverageComplete
        )
        guard !runtime.isEmpty else { return "" }
        if run.runtimeDevice == nil {
            return " · runtime \(runtime), device not recorded"
        }
        return " · runtime \(runtime)"
    }

    private nonisolated static func selectedRunAuthorityWarning(_ run: ExpertRunSummary) -> String {
        if run.runtimeMode != "bf16_vmlx" {
            return "Saved run is inspectable only: runtime is \(runtimeModeDisplayName(run.runtimeMode)), not BF16/vMLX authority."
        }
        if run.runtimeBackend?.trimmingCharacters(in: .whitespacesAndNewlines) != "vmlx" {
            return "Saved run is inspectable only: vMLX backend evidence is missing or not authoritative."
        }
        if run.runtimeDevice?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "Saved run is inspectable only: runtime device evidence is missing."
        }
        if run.runtimeMetalEnabled != true {
            return "Saved run is inspectable only: Metal runtime evidence is missing."
        }
        if run.jangToolsVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
            run.mlxVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
            run.mlxLMVersion?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "Saved run is inspectable only: vMLX package version evidence is missing."
        }
        if run.reviewBundlePath != nil {
            return "Saved run is inspectable only: it still references a legacy review bundle."
        }
        if run.sourceModelPath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            return "Saved run is inspectable only: runtime source path evidence is missing."
        }
        guard let hookedMOELayers = run.hookedMOELayers, hookedMOELayers > 0 else {
            return "Saved run is inspectable only: vMLX routed-layer hook evidence is missing."
        }
        if run.hookCoverageComplete == false {
            return "Saved run is inspectable only: vMLX routed-layer hook coverage is incomplete."
        }
        if let expectedMOELayers = run.expectedMOELayers,
           expectedMOELayers > 0,
           hookedMOELayers < expectedMOELayers {
            return "Saved run is inspectable only: vMLX hook coverage \(hookedMOELayers) of \(expectedMOELayers) config-routed layers."
        }
        return ""
    }

    private nonisolated static func runtimeModeDisplayName(_ mode: String) -> String {
        switch mode {
        case "bf16_vmlx":
            return "BF16/vMLX"
        case "native_jangtq_review_bundle":
            return "native JANGTQ review-bundle"
        default:
            return mode.isEmpty ? "unknown runtime" : mode
        }
    }

    private nonisolated static func selectedRunPrunePlanWarning(runDirectory: URL) -> String {
        let planURL = runDirectory.appendingPathComponent("prune_plan.json")
        guard FileManager.default.fileExists(atPath: planURL.path) else { return "" }
        do {
            let data = try Data(contentsOf: planURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let plan = try decoder.decode(ExpertPrunePlan.self, from: data)
            guard let safety = plan.safety else {
                return "Saved prune plan is legacy and missing top-k safety evidence; re-export after Compare Suite passes."
            }
            if !safety.passed {
                return "Saved prune plan is blocked: embedded top-k safety did not pass."
            }
            if let issue = safety.issues.first {
                return "Saved prune plan is blocked: \(issue)"
            }
            guard let trainedTopK = safety.trainedTopKByLayer.values.max() else {
                return "Saved prune plan is blocked: trained top-k evidence is missing."
            }
            if safety.minimumActiveExpertsPerLayer < trainedTopK {
                return "Saved prune plan is blocked: keeps \(safety.minimumActiveExpertsPerLayer) experts but trained top-k is \(trainedTopK)."
            }
            guard let comparison = plan.comparisonSummary else {
                return "Saved prune plan is legacy and missing same-suite comparison evidence; re-export after Compare Suite passes."
            }
            if comparison.promptCount < minimumReviewedPrunePromptCount {
                return "Saved prune plan is blocked: compare at least \(minimumReviewedPrunePromptCount) prompts before pruning."
            }
            if comparison.validatorAvailablePromptCount == nil || comparison.classificationCounts == nil {
                return "Saved prune plan is blocked: same-suite comparison is missing validator classification evidence."
            }
            if (comparison.baselineQualifiedPromptCount ?? 0) <= 0 {
                return "Saved prune plan is blocked: no prompts have a valid BF16/vMLX baseline validator pass."
            }
            if let missing = comparison.missingBaselineQualifiedSemanticCoverage,
               !missing.isEmpty {
                return "Saved prune plan is blocked: baseline-qualified semantic coverage is missing."
            }
            if let degraded = comparison.degradedPromptIDs,
               !degraded.isEmpty {
                return "Saved prune plan is blocked: baseline-qualified prompts degraded after masking."
            }
            if let passRate = comparison.baselineQualifiedMaskedPassRate,
               passRate < 1.0 {
                return "Saved prune plan is blocked: masked validator pass rate is below 100% on baseline-qualified prompts."
            }
            if !comparison.highRiskDomains.isEmpty {
                return "Saved prune plan is blocked: masked outputs regressed in high-risk domains."
            }
            if comparison.safeDropCandidates.isEmpty {
                return "Saved prune plan is blocked: same-suite comparison found no safe drop candidates."
            }
            guard let evalIndex = plan.evalIndex else {
                return "Saved prune plan is legacy and missing per-prompt eval_index evidence; re-export after Compare Suite passes."
            }
            if evalIndex.promptCount != comparison.promptCount {
                return "Saved prune plan is blocked: eval_index does not cover every compared prompt."
            }
            if evalIndex.promptIDs.count != evalIndex.promptCount {
                return "Saved prune plan is blocked: eval_index lists \(evalIndex.promptIDs.count) prompt IDs for \(evalIndex.promptCount) indexed prompts."
            }
            if Set(evalIndex.promptIDs).count < evalIndex.promptIDs.count {
                return "Saved prune plan is blocked: eval_index contains duplicate prompt IDs."
            }
            if !evalIndex.riskyPromptIDs.isEmpty || !evalIndex.highRiskDomains.isEmpty {
                return "Saved prune plan is blocked: eval_index still contains risky prompts."
            }
            if evalIndex.validatorSchema?.isEmpty != false
                || evalIndex.validatorAvailablePromptCount == nil
                || evalIndex.promptClassificationCounts == nil {
                return "Saved prune plan is blocked: eval_index is missing validator classification evidence."
            }
            guard let baselineQualifiedPromptCount = evalIndex.baselineQualifiedPromptCount,
                  baselineQualifiedPromptCount > 0 else {
                return "Saved prune plan is blocked: eval_index has no baseline-qualified validator prompts."
            }
            guard let baselineQualifiedPromptIDs = evalIndex.baselineQualifiedPromptIDs,
                  baselineQualifiedPromptIDs.count == baselineQualifiedPromptCount,
                  let baselineInvalidPromptIDs = evalIndex.baselineInvalidPromptIDs,
                  let inconclusivePromptIDs = evalIndex.inconclusivePromptIDs,
                  let preservedPromptIDs = evalIndex.preservedPromptIDs,
                  let degradedPromptIDs = evalIndex.degradedPromptIDs else {
                return "Saved prune plan is blocked: eval_index is missing prompt classification ID lists."
            }
            if baselineInvalidPromptIDs.count + inconclusivePromptIDs.count + preservedPromptIDs.count + degradedPromptIDs.count != evalIndex.promptCount {
                return "Saved prune plan is blocked: eval_index prompt classifications do not cover every prompt."
            }
            if !degradedPromptIDs.isEmpty {
                return "Saved prune plan is blocked: eval_index has baseline-qualified prompt regressions."
            }
            if let missing = evalIndex.missingBaselineQualifiedSemanticCoverage,
               !missing.isEmpty {
                return "Saved prune plan is blocked: eval_index baseline-qualified semantic coverage is missing."
            }
            guard evalIndex.baselineQualifiedSemanticCoverage?.isEmpty == false else {
                return "Saved prune plan is blocked: eval_index is missing baseline-qualified semantic coverage evidence."
            }
            if let passRate = evalIndex.baselineQualifiedMaskedPassRate,
               passRate < 1.0 {
                return "Saved prune plan is blocked: eval_index masked validator pass rate is below 100% on baseline-qualified prompts."
            }
            if let issue = evalIndexSemanticCoverageIssue(
                semanticCoverage: evalIndex.semanticCoverage,
                missingSemanticCoverage: evalIndex.missingSemanticCoverage,
                evidenceName: "Saved prune plan is blocked: eval_index"
            ) {
                return issue
            }
            guard let baselineDepth = evalIndex.meanBaselineTokens,
                  let maskedDepth = evalIndex.meanMaskedTokens else {
                return "Saved prune plan is blocked: eval_index is missing generation-depth evidence."
            }
            if min(baselineDepth, maskedDepth) < minimumReviewedPruneMeanTokens {
                return "Saved prune plan is blocked: eval_index generation depth is too shallow."
            }
            guard let baselineRouteRecordCount = evalIndex.baselineRouteRecordCount,
                  let maskedRouteRecordCount = evalIndex.maskedRouteRecordCount,
                  baselineRouteRecordCount >= evalIndex.promptCount,
                  maskedRouteRecordCount >= evalIndex.promptCount else {
                return "Saved prune plan is blocked: eval_index is missing routing record evidence for every indexed prompt."
            }
            if let issue = ExpertLabLayerStatsEvidenceValidator.issue(
                promptCount: evalIndex.promptCount,
                baselinePromptCount: evalIndex.baselineLayerStatsPromptCount,
                maskedPromptCount: evalIndex.maskedLayerStatsPromptCount,
                evidenceName: "Saved prune plan is blocked: eval_index"
            ) {
                return issue
            }
            if evalIndex.generationSettingsChecked != true {
                return "Saved prune plan is blocked: eval_index is missing decode settings evidence."
            }
            guard evalIndex.evalTraceJSONL?.isEmpty == false else {
                return "Saved prune plan is blocked: eval_index is missing eval_trace.jsonl evidence."
            }
            if let issue = selectedRunEvalTraceIntegrityIssue(
                evalIndex: evalIndex,
                plan: plan,
                runDirectory: runDirectory
            ) {
                return issue
            }
            if let issue = selectedRunEvalDecodeSettingsIssue(
                evalIndex: evalIndex,
                plan: plan,
                runDirectory: runDirectory
            ) {
                return issue
            }
            guard evalIndex.runtimeMode?.isEmpty == false,
                  evalIndex.runtimeDevice?.isEmpty == false,
                  evalIndex.runtimeMetalEnabled != nil else {
                return "Saved prune plan is blocked: eval_index is missing runtime device evidence."
            }
            if evalIndex.runtimeMetalEnabled != true {
                return "Saved prune plan is blocked: eval_index did not record a Metal runtime."
            }
            if evalIndex.runtimeMode != "bf16_vmlx" {
                return "Saved prune plan is blocked: eval_index did not record the BF16/vMLX runtime."
            }
            if evalIndex.runtimeBackend != "vmlx" {
                return "Saved prune plan is blocked: eval_index did not record the vMLX backend."
            }
            guard let hookedMOELayers = evalIndex.hookedMOELayers, hookedMOELayers > 0 else {
                return "Saved prune plan is blocked: eval_index is missing vMLX routed-layer hook evidence."
            }
            if evalIndex.hookCoverageComplete == false {
                return "Saved prune plan is blocked: eval_index recorded incomplete vMLX routed-layer hook coverage."
            }
            guard let expectedMOELayers = evalIndex.expectedMOELayers, expectedMOELayers > 0 else {
                return "Saved prune plan is blocked: eval_index is missing vMLX routed-layer hook evidence."
            }
            if hookedMOELayers < expectedMOELayers {
                return "Saved prune plan is blocked: eval_index vMLX hook coverage \(hookedMOELayers) of \(expectedMOELayers) config-routed layers."
            }
            guard evalIndex.jangToolsVersion?.isEmpty == false,
                  evalIndex.mlxVersion?.isEmpty == false,
                  evalIndex.mlxLMVersion?.isEmpty == false else {
                return "Saved prune plan is blocked: eval_index is missing vMLX package version evidence."
            }
            guard let evalSourcePath = evalIndex.sourceModelPath, !evalSourcePath.isEmpty else {
                return "Saved prune plan is blocked: eval_index is missing source model path evidence."
            }
            if let planSourcePath = plan.sourceModelPath,
               Self.canonicalPath(evalSourcePath) != Self.canonicalPath(planSourcePath) {
                return "Saved prune plan is blocked: eval_index source model path does not match the prune plan source."
            }
            if evalIndex.maskApplied != true {
                return "Saved prune plan is blocked: eval_index did not record an applied BF16/vMLX mask."
            }
            if (evalIndex.disabledExpertCount ?? 0) <= 0 {
                return "Saved prune plan is blocked: eval_index did not record disabled expert evidence; top-k-only comparisons cannot authorize hard pruning."
            }
            if let suiteIssue = selectedRunSuiteFingerprintIssue(
                evalIndex: evalIndex,
                plan: plan,
                runDirectory: runDirectory
            ) {
                return suiteIssue
            }
            return ""
        } catch {
            return "Saved prune plan is unreadable; hard prune remains blocked."
        }
    }

    private nonisolated static func selectedRunSuiteFingerprintIssue(
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan,
        runDirectory: URL
    ) -> String? {
        guard let recordedSuiteSHA256 = evalIndex.suiteSHA256?.trimmingCharacters(in: .whitespacesAndNewlines),
              !recordedSuiteSHA256.isEmpty else {
            return "Saved prune plan is blocked: eval_index is missing suite.jsonl fingerprint evidence."
        }
        guard let suiteURL = selectedRunSuiteURL(
            evalIndex: evalIndex,
            plan: plan,
            runDirectory: runDirectory
        ) else {
            return "Saved prune plan is blocked: suite.jsonl evidence is unreadable."
        }
        guard let expectedSuiteSHA256 = fileSHA256(suiteURL) else {
            return "Saved prune plan is blocked: suite.jsonl fingerprint could not be computed."
        }
        if recordedSuiteSHA256 != expectedSuiteSHA256 {
            return "Saved prune plan is blocked: eval_index suite.jsonl fingerprint does not match suite.jsonl."
        }
        return nil
    }

    private nonisolated static func selectedRunEvalTraceIntegrityIssue(
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan,
        runDirectory: URL
    ) -> String? {
        guard let evalTraceURL = selectedRunEvalTraceURL(
            evalIndex: evalIndex,
            plan: plan,
            runDirectory: runDirectory
        ) else {
            return "Saved prune plan is blocked: eval_trace.jsonl evidence is unreadable."
        }
        return evalTraceIntegrityIssue(
            evalIndex: evalIndex,
            evalTraceURL: evalTraceURL,
            issuePrefix: "Saved prune plan is blocked",
            disabledExpertCount: evalIndex.disabledExpertCount,
            topKOverride: evalIndex.topKOverride
        )
    }

    private nonisolated static func selectedRunEvalDecodeSettingsIssue(
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan,
        runDirectory: URL
    ) -> String? {
        guard let evalURL = selectedRunEvalURL(
            evalIndex: evalIndex,
            plan: plan,
            runDirectory: runDirectory
        ) else {
            return "Saved prune plan is blocked: eval.jsonl evidence is unreadable."
        }
        guard let rows = jsonlObjects(from: evalURL) else {
            return "Saved prune plan is blocked: eval.jsonl is unreadable."
        }
        guard rows.count == evalIndex.promptCount else {
            return "Saved prune plan is blocked: eval.jsonl covers \(rows.count) of \(evalIndex.promptCount) indexed prompts."
        }
        let rowPromptIDs = rows.compactMap {
            jsonStringValue($0, keys: ["promptID", "prompt_id", "id"])
        }
        guard rowPromptIDs.count == rows.count else {
            return "Saved prune plan is blocked: eval.jsonl prompt IDs are unreadable."
        }
        if rowPromptIDs != evalIndex.promptIDs {
            return "Saved prune plan is blocked: eval.jsonl prompt order does not match eval_index."
        }
        if let issue = selectedRunEvalRowEvidenceIssue(rows: rows, evalIndex: evalIndex, plan: plan) {
            return issue
        }

        for row in rows {
            let baselineValue = row["baselineGenerationSettings"] ?? row["baseline_generation_settings"]
            let maskedValue = row["maskedGenerationSettings"] ?? row["masked_generation_settings"]
            guard baselineValue != nil, maskedValue != nil else {
                return "Saved prune plan is blocked: eval.jsonl is missing baseline/masked decode settings evidence."
            }
            guard let baseline = generationSettings(from: baselineValue),
                  let masked = generationSettings(from: maskedValue) else {
                return "Saved prune plan is blocked: eval.jsonl has unreadable baseline/masked decode settings evidence."
            }
            if baseline != masked {
                return "Saved prune plan is blocked: eval.jsonl baseline/masked decode settings do not match."
            }
        }
        return nil
    }

    private nonisolated static func generationSettings(from value: Any?) -> StoredGenerationSettings? {
        guard let object = value as? [String: Any],
              let maxTokens = jsonIntValue(object, keys: ["max_tokens", "maxTokens"]),
              let temperature = jsonDoubleValue(object, keys: ["temperature"]),
              let topP = jsonDoubleValue(object, keys: ["top_p", "topP"]),
              let topK = jsonIntValue(object, keys: ["top_k", "topK"]) else {
            return nil
        }
        let settings = StoredGenerationSettings(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK
        )
        return settings.isValid ? settings : nil
    }

    private nonisolated static func selectedRunEvalRowEvidenceIssue(
        rows: [[String: Any]],
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan
    ) -> String? {
        let expectedSourcePath = plan.sourceModelPath ?? evalIndex.sourceModelPath
        return evalRowEvidenceIssue(
            rows: rows,
            expectedSourcePath: expectedSourcePath,
            sourceMismatchIssue: "Saved prune plan is blocked: eval.jsonl source model path does not match the prune plan source.",
            issuePrefix: "Saved prune plan is blocked"
        )
    }

    private nonisolated static func evalRowEvidenceIssue(
        rows: [[String: Any]],
        expectedSourcePath: String?,
        sourceMismatchIssue: String,
        issuePrefix: String
    ) -> String? {
        for row in rows {
            if jsonStringValue(row, keys: ["baselineText", "baseline_text"]) == nil
                || jsonStringValue(row, keys: ["maskedText", "masked_text"]) == nil {
                return "\(issuePrefix): eval.jsonl is missing per-prompt baseline/masked output text."
            }
            guard let textDelta = jsonDoubleValue(row, keys: ["textDelta", "text_delta"]),
                  textDelta.isFinite else {
                return "\(issuePrefix): eval.jsonl is missing per-prompt text delta evidence."
            }
            if (jsonIntValue(row, keys: ["baselineTokenCount", "baseline_token_count"]) ?? 0) <= 0
                || (jsonIntValue(row, keys: ["maskedTokenCount", "masked_token_count"]) ?? 0) <= 0 {
                return "\(issuePrefix): eval.jsonl is missing per-prompt token count evidence."
            }
            if (jsonIntValue(row, keys: ["baselineRouteRecordCount", "baseline_route_record_count"]) ?? 0) <= 0
                || (jsonIntValue(row, keys: ["maskedRouteRecordCount", "masked_route_record_count"]) ?? 0) <= 0 {
                return "\(issuePrefix): eval.jsonl is missing per-prompt routing record evidence."
            }
            guard let runtimeMode = jsonStringValue(row, keys: ["runtimeMode", "runtime_mode"]),
                  jsonStringValue(row, keys: ["runtimeDevice", "runtime_device"]) != nil,
                  let runtimeMetalEnabled = jsonBoolValue(row, keys: ["runtimeMetalEnabled", "runtime_metal_enabled"]) else {
                return "\(issuePrefix): eval.jsonl is missing per-prompt runtime device evidence."
            }
            if runtimeMetalEnabled != true {
                return "\(issuePrefix): eval.jsonl did not record a Metal runtime."
            }
            if runtimeMode != "bf16_vmlx" {
                return "\(issuePrefix): eval.jsonl did not record BF16/vMLX runtime evidence."
            }
            if jsonStringValue(row, keys: ["runtimeBackend", "runtime_backend"]) != "vmlx" {
                return "\(issuePrefix): eval.jsonl did not record per-prompt vMLX backend evidence."
            }
            if jsonStringValue(row, keys: ["jangToolsVersion", "jang_tools_version"]) == nil
                || jsonStringValue(row, keys: ["mlxVersion", "mlx_version"]) == nil
                || jsonStringValue(row, keys: ["mlxLMVersion", "mlx_lm_version"]) == nil {
                return "\(issuePrefix): eval.jsonl is missing per-prompt vMLX package version evidence."
            }
            guard let rowSourcePath = jsonStringValue(row, keys: ["sourceModelPath", "source_model_path"]) else {
                return "\(issuePrefix): eval.jsonl is missing per-prompt source model path evidence."
            }
            if let expectedSourcePath,
               canonicalPath(rowSourcePath) != canonicalPath(expectedSourcePath) {
                return sourceMismatchIssue
            }
            if jsonBoolValue(row, keys: ["maskApplied", "mask_applied"]) != true {
                return "\(issuePrefix): eval.jsonl did not record an applied BF16/vMLX mask."
            }
            if (jsonIntValue(row, keys: ["disabledExpertCount", "disabled_expert_count"]) ?? 0) <= 0 {
                return "\(issuePrefix): eval.jsonl is missing per-prompt disabled expert evidence; top-k-only comparisons cannot authorize hard pruning."
            }
            if jsonStringValue(row, keys: ["risk"]) == nil
                || jsonStringValue(row, keys: ["regressionSeverity", "regression_severity"]) == nil {
                return "\(issuePrefix): eval.jsonl is missing per-prompt regression flag evidence."
            }
        }
        return nil
    }

    private nonisolated static func evalTraceIntegrityIssue(
        evalIndex: ExpertEvalIndexSummary,
        evalTraceURL: URL,
        issuePrefix: String,
        disabledExpertCount: Int? = nil,
        topKOverride: Int? = nil,
        expectedDisabledByLayer: [Int: Set<Int>] = [:]
    ) -> String? {
        guard let traceRows = jsonlObjects(from: evalTraceURL) else {
            return "\(issuePrefix): eval_trace.jsonl is unreadable."
        }
        guard !traceRows.isEmpty else {
            return "\(issuePrefix): eval_trace.jsonl has no routing records."
        }

        let expectedPromptIDs = evalIndex.promptIDs.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let expected = Set(expectedPromptIDs)
        var tracePromptIDs = Set<String>()
        var baselinePromptIDs = Set<String>()
        var maskedPromptIDs = Set<String>()
        var maskedPromptIDsWithMaskEvidence = Set<String>()
        var maskedPromptExpectedMaskLayers: [String: Set<Int>] = [:]
        var baselineTraceCount = 0
        var maskedTraceCount = 0
        let expectedMaskLayers = Set(expectedDisabledByLayer.filter { !$0.value.isEmpty }.keys)

        for row in traceRows {
            guard let promptID = jsonStringValue(row, keys: ["promptID", "prompt_id", "id"]) else {
                return "\(issuePrefix): eval_trace.jsonl prompt IDs are unreadable."
            }
            tracePromptIDs.insert(promptID)
            switch jsonStringValue(row, keys: ["variant"])?.lowercased() {
            case "baseline":
                baselineTraceCount += 1
                baselinePromptIDs.insert(promptID)
            case "masked":
                maskedTraceCount += 1
                maskedPromptIDs.insert(promptID)
                if let issue = traceRowDisabledSelectionIssue(row, promptID: promptID) {
                    return "\(issuePrefix): \(issue)"
                }
                if let layer = traceRowExpectedMaskEvidenceLayer(
                    row,
                    expectedDisabledByLayer: expectedDisabledByLayer
                ) {
                    maskedPromptExpectedMaskLayers[promptID, default: []].insert(layer)
                }
                if traceRowHasMaskEvidence(
                    row,
                    disabledExpertCount: disabledExpertCount,
                    topKOverride: topKOverride
                ) {
                    maskedPromptIDsWithMaskEvidence.insert(promptID)
                }
            default:
                continue
            }
        }

        let missingTraceIDs = expected.subtracting(tracePromptIDs)
        if !missingTraceIDs.isEmpty {
            return "\(issuePrefix): eval_index prompt IDs missing from eval_trace.jsonl: \(previewTraceIDs(missingTraceIDs))"
        }
        let unexpectedTraceIDs = tracePromptIDs.subtracting(expected)
        if !unexpectedTraceIDs.isEmpty {
            return "\(issuePrefix): eval_trace.jsonl prompt IDs outside eval_index: \(previewTraceIDs(unexpectedTraceIDs))"
        }
        let missingBaseline = expected.subtracting(baselinePromptIDs)
        if !missingBaseline.isEmpty {
            return "\(issuePrefix): eval_trace.jsonl missing baseline routing records for prompt IDs: \(previewTraceIDs(missingBaseline))"
        }
        let missingMasked = expected.subtracting(maskedPromptIDs)
        if !missingMasked.isEmpty {
            return "\(issuePrefix): eval_trace.jsonl missing masked routing records for prompt IDs: \(previewTraceIDs(missingMasked))"
        }
        if let baselineRouteRecordCount = evalIndex.baselineRouteRecordCount,
           baselineTraceCount != baselineRouteRecordCount {
            return "\(issuePrefix): eval_trace.jsonl has \(baselineTraceCount) baseline routing records for \(baselineRouteRecordCount) indexed baseline route records"
        }
        if let maskedRouteRecordCount = evalIndex.maskedRouteRecordCount,
           maskedTraceCount != maskedRouteRecordCount {
            return "\(issuePrefix): eval_trace.jsonl has \(maskedTraceCount) masked routing records for \(maskedRouteRecordCount) indexed masked route records"
        }
        if (disabledExpertCount ?? 0) > 0 || topKOverride != nil {
            let missingMaskEvidence = expected.subtracting(maskedPromptIDsWithMaskEvidence)
            if !missingMaskEvidence.isEmpty {
                return "\(issuePrefix): eval_trace.jsonl masked routing records are missing mask evidence for prompt IDs: \(previewTraceIDs(missingMaskEvidence))"
            }
        }
        if !expectedMaskLayers.isEmpty {
            for promptID in expectedPromptIDs {
                let missingLayers = expectedMaskLayers.subtracting(maskedPromptExpectedMaskLayers[promptID] ?? [])
                if !missingLayers.isEmpty {
                    return "\(issuePrefix): eval_trace.jsonl masked routing records are missing mask.json evidence for prompt \(promptID) layers: \(previewInts(missingLayers))"
                }
            }
        }
        return nil
    }

    private nonisolated static func traceRowExpectedMaskEvidenceLayer(
        _ row: [String: Any],
        expectedDisabledByLayer: [Int: Set<Int>]
    ) -> Int? {
        guard let record = row["record"] as? [String: Any],
              let layer = jsonIntValue(record, keys: ["layer", "layerIndex", "layer_index"]),
              let expectedDisabled = expectedDisabledByLayer[layer],
              !expectedDisabled.isEmpty else {
            return nil
        }
        let disabled = Set(jsonArrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap {
            jsonIntValue(["value": $0], keys: ["value"])
        })
        return expectedDisabled.isSubset(of: disabled) ? layer : nil
    }

    private nonisolated static func traceRowDisabledSelectionIssue(_ row: [String: Any], promptID: String) -> String? {
        guard let record = row["record"] as? [String: Any] else { return nil }
        let disabled = Set(jsonArrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap {
            jsonIntValue(["value": $0], keys: ["value"])
        })
        guard !disabled.isEmpty else { return nil }
        let selected = Set(jsonArrayValue(record["selectedExperts"] ?? record["selected_experts"]).compactMap {
            jsonIntValue(["value": $0], keys: ["value"])
        })
        let leaked = selected.intersection(disabled)
        guard !leaked.isEmpty else { return nil }
        return "eval_trace.jsonl masked routing records selected disabled experts for prompt \(promptID): \(previewInts(leaked))"
    }

    private nonisolated static func traceRowHasMaskEvidence(
        _ row: [String: Any],
        disabledExpertCount: Int?,
        topKOverride: Int?
    ) -> Bool {
        guard let record = row["record"] as? [String: Any] else { return false }
        if (disabledExpertCount ?? 0) > 0 {
            if !jsonArrayValue(record["disabledExperts"] ?? record["disabled_experts"]).isEmpty {
                return true
            }
            if (jsonIntValue(record, keys: ["disabledExpertCount", "disabled_expert_count"]) ?? 0) > 0 {
                return true
            }
            return false
        }
        if topKOverride != nil {
            return jsonIntValue(record, keys: ["effectiveTopK", "effective_top_k", "topK", "top_k"]) != nil
        }
        return true
    }

    private nonisolated static func evalIndexSemanticCoverageIssue(
        semanticCoverage: [String]?,
        missingSemanticCoverage: [String]?,
        evidenceName: String
    ) -> String? {
        guard let semanticCoverage,
              !semanticCoverage.isEmpty else {
            return "\(evidenceName) is missing semantic coverage evidence."
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
            return "\(evidenceName) semantic coverage is missing required probes: \(missingCoverage.joined(separator: ", "))."
        }
        guard let recordedMissing = missingSemanticCoverage else {
            return "\(evidenceName) is missing missing-semantic-coverage evidence."
        }
        let missing = Set(
            recordedMissing
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        if !missing.isEmpty {
            return "\(evidenceName) records missing semantic prompt probes: \(missing.sorted().joined(separator: ", "))."
        }
        return nil
    }

    private nonisolated static func selectedRunEvalTraceURL(
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan,
        runDirectory: URL
    ) -> URL? {
        guard let tracePath = evalIndex.evalTraceJSONL?.trimmingCharacters(in: .whitespacesAndNewlines),
              !tracePath.isEmpty else {
            return nil
        }
        if (tracePath as NSString).isAbsolutePath {
            return URL(fileURLWithPath: tracePath)
        }

        var candidates: [URL] = []
        if let evalArtifactPath = plan.evalArtifactPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !evalArtifactPath.isEmpty {
            candidates.append(
                URL(fileURLWithPath: evalArtifactPath, isDirectory: true)
                    .appendingPathComponent(tracePath)
            )
        }
        if let latest = latestComparisonDirectory(
            in: runDirectory.appendingPathComponent("evals", isDirectory: true)
        ) {
            candidates.append(latest.appendingPathComponent(tracePath))
        }
        candidates.append(runDirectory.appendingPathComponent(tracePath))

        let fm = FileManager.default
        return candidates.first { fm.isReadableFile(atPath: $0.path) } ?? candidates.first
    }

    private nonisolated static func selectedRunEvalURL(
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan,
        runDirectory: URL
    ) -> URL? {
        guard let evalPath = evalIndex.evalJSONL?.trimmingCharacters(in: .whitespacesAndNewlines),
              !evalPath.isEmpty else {
            return nil
        }
        if (evalPath as NSString).isAbsolutePath {
            return URL(fileURLWithPath: evalPath)
        }

        var candidates: [URL] = []
        if let evalArtifactPath = plan.evalArtifactPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !evalArtifactPath.isEmpty {
            candidates.append(
                URL(fileURLWithPath: evalArtifactPath, isDirectory: true)
                    .appendingPathComponent(evalPath)
            )
        }
        if let latest = latestComparisonDirectory(
            in: runDirectory.appendingPathComponent("evals", isDirectory: true)
        ) {
            candidates.append(latest.appendingPathComponent(evalPath))
        }
        candidates.append(runDirectory.appendingPathComponent(evalPath))

        let fm = FileManager.default
        return candidates.first { fm.isReadableFile(atPath: $0.path) } ?? candidates.first
    }

    private nonisolated static func selectedRunSuiteURL(
        evalIndex: ExpertEvalIndexSummary,
        plan: ExpertPrunePlan,
        runDirectory: URL
    ) -> URL? {
        let suitePath = evalIndex.suiteJSONL?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let suitePath, !suitePath.isEmpty, (suitePath as NSString).isAbsolutePath {
            let url = URL(fileURLWithPath: suitePath)
            return FileManager.default.isReadableFile(atPath: url.path) ? url : nil
        }

        var candidates: [URL] = []
        if let suitePath, !suitePath.isEmpty {
            candidates.append(runDirectory.appendingPathComponent(suitePath))
            if let evalArtifactPath = plan.evalArtifactPath?.trimmingCharacters(in: .whitespacesAndNewlines),
               !evalArtifactPath.isEmpty {
                candidates.append(
                    URL(fileURLWithPath: evalArtifactPath, isDirectory: true)
                        .appendingPathComponent(suitePath)
                )
            }
            if let latest = latestComparisonDirectory(
                in: runDirectory.appendingPathComponent("evals", isDirectory: true)
            ) {
                candidates.append(latest.appendingPathComponent(suitePath))
            }
        }
        candidates.append(runDirectory.appendingPathComponent("suite.jsonl"))

        let fm = FileManager.default
        return candidates.first { fm.isReadableFile(atPath: $0.path) }
    }

    private nonisolated static func jsonlObjects(from url: URL) -> [[String: Any]]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var rows: [[String: Any]] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard let object = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any] else {
                return nil
            }
            rows.append(object)
        }
        return rows
    }

    private nonisolated static func jsonStringValue(_ row: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = row[key] as? String {
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
            if let value = row[key] as? NSNumber {
                return value.stringValue
            }
        }
        return nil
    }

    private nonisolated static func jsonIntValue(_ row: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let value = row[key] as? Int { return value }
            if let value = row[key] as? Double, value.isFinite { return Int(value) }
            if let value = row[key] as? NSNumber { return value.intValue }
            if let value = row[key] as? String, let int = Int(value) { return int }
        }
        return nil
    }

    private nonisolated static func jsonDoubleValue(_ row: [String: Any], keys: [String]) -> Double? {
        for key in keys {
            if let value = row[key] as? Double, value.isFinite { return value }
            if let value = row[key] as? Float, value.isFinite { return Double(value) }
            if let value = row[key] as? Int { return Double(value) }
            if let value = row[key] as? NSNumber { return value.doubleValue }
            if let value = row[key] as? String, let double = Double(value) { return double }
        }
        return nil
    }

    private nonisolated static func jsonBoolValue(_ row: [String: Any], keys: [String]) -> Bool? {
        for key in keys {
            if let value = row[key] as? Bool { return value }
            if let value = row[key] as? NSNumber { return value.boolValue }
            if let value = row[key] as? String { return Bool(value) }
        }
        return nil
    }

    private nonisolated static func jsonArrayValue(_ value: Any?) -> [Any] {
        value as? [Any] ?? []
    }

    private nonisolated static func previewTraceIDs(_ ids: Set<String>, limit: Int = 5) -> String {
        let sorted = ids.sorted()
        let prefix = sorted.prefix(limit).joined(separator: ", ")
        if sorted.count > limit {
            return "\(prefix), ... (+\(sorted.count - limit) more)"
        }
        return prefix.isEmpty ? "none" : prefix
    }

    private nonisolated static func previewInts(_ values: Set<Int>, limit: Int = 5) -> String {
        let sorted = values.sorted()
        let prefix = sorted.prefix(limit).map(String.init).joined(separator: ", ")
        if sorted.count > limit {
            return "\(prefix), ... (+\(sorted.count - limit) more)"
        }
        return prefix.isEmpty ? "none" : prefix
    }


    /// Result of I/O and computation for loadSelectedRun, produced off the main actor.
    private struct _LoadedRunData: Sendable {
        let atlas: ExpertAtlas
        let suite: ExpertPromptSuite?
        let runs: [ExpertPromptRun]
        let evidenceIssue: String?
        let maskArtifact: ExpertLabMaskArtifact?
        let genRecord: StoredGenerationRecord?
        let compDir: URL?
        let compMask: JANGKit.ExpertMask?
        let compSummary: String
        let compRecords: [StoredEvalRecord]
        let compFirstBaselineText: String
        let compFirstMaskedText: String
        let compFirstBaselineOutputReady: Bool
        let compFirstMaskedOutputReady: Bool
        let compFirstBaselineTPS: Double
        let compFirstMaskedTPS: Double
        let compFirstPromptID: String
        let compFirstTextDelta: Double
        let compFirstRisk: String
    }

    func loadSelectedRun() async {
        guard let selectedRunSummary else { return }
        let dir = URL(fileURLWithPath: selectedRunSummary.directoryPath, isDirectory: true)
        let suiteID = selectedRunSummary.suiteID
        let capExpectedExperts = capability.expectedExperts
        let capExpectedLayers = capability.expectedLayers

        do {
            // Phase 1: All I/O and heavy computation off MainActor
            let context = try await Task.detached(priority: .userInitiated) {
                [dir, suiteID, capExpectedExperts, capExpectedLayers] in
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601

                // Atlas
                let atlasData = try Data(contentsOf: dir.appendingPathComponent("atlas.json"))
                let decodedAtlas = try decoder.decode(ExpertAtlas.self, from: atlasData)
                let loadedAtlas = Self._completedAtlas(decodedAtlas, expectedExperts: capExpectedExperts, expectedLayers: capExpectedLayers)
                if loadedAtlas.experts.count != decodedAtlas.experts.count {
                    try? Self._writeAtlas(loadedAtlas, to: dir)
                }

                // Suite
                var loadedSuite: ExpertPromptSuite?
                let suiteURL = dir.appendingPathComponent("suite.jsonl")
                if FileManager.default.fileExists(atPath: suiteURL.path) {
                    loadedSuite = try ExpertPromptSuite.loadJSONL(name: suiteID, from: suiteURL)
                }

                // Evidence + runs
                var restRuns: [ExpertPromptRun] = []
                var evidenceIssue: String?
                if let s = loadedSuite {
                    evidenceIssue = Self.restoredRunEvidenceIssue(from: dir, suite: s)
                    if evidenceIssue == nil {
                        restRuns = Self.restoredRuns(from: dir, suite: s)
                    }
                }

                // Mask
                let maskArtifact: ExpertLabMaskArtifact?
                let maskURL = dir
                    .appendingPathComponent("masks", isDirectory: true)
                    .appendingPathComponent("current_mask.json")
                if FileManager.default.fileExists(atPath: maskURL.path) {
                    maskArtifact = try? decoder.decode(ExpertLabMaskArtifact.self, from: Data(contentsOf: maskURL))
                } else {
                    maskArtifact = nil
                }

                // Generation preview
                let genRecord = Self.firstJSONLRecord(StoredGenerationRecord.self, from: dir.appendingPathComponent("generations.jsonl"))

                // Comparison directory + data
                var compDir: URL?
                var compMask: JANGKit.ExpertMask?
                var compSummary = ""
                var compRecords: [StoredEvalRecord] = []
                var compFirstBaselineText = ""
                var compFirstMaskedText = ""
                var compFirstBaselineOutputReady = false
                var compFirstMaskedOutputReady = false
                var compFirstBaselineTPS: Double = 0
                var compFirstMaskedTPS: Double = 0
                var compFirstPromptID = ""
                var compFirstTextDelta: Double = 0
                var compFirstRisk = ""

                let evalsDir = dir.appendingPathComponent("evals", isDirectory: true)
                if let latest = Self.latestComparisonDirectory(in: evalsDir) {
                    compDir = latest
                    compMask = try? decoder.decode(JANGKit.ExpertMask.self, from: Data(contentsOf: latest.appendingPathComponent("mask.json")))
                    if let summary = Self.loadComparisonSummary(from: latest) {
                        compSummary = Self.comparisonSummaryText(summary)
                    }
                    compRecords = Self.jsonlRecords(StoredEvalRecord.self, from: latest.appendingPathComponent("eval.jsonl"))
                    if let first = compRecords.first {
                        compFirstBaselineText = first.baselineText
                        compFirstMaskedText = first.maskedText
                        compFirstBaselineOutputReady = true
                        compFirstMaskedOutputReady = true
                        compFirstBaselineTPS = first.baselineTokensPerSecond
                        compFirstMaskedTPS = first.maskedTokensPerSecond
                        compFirstPromptID = first.promptID
                        compFirstTextDelta = first.textDelta
                        compFirstRisk = first.risk
                    }
                }

                return _LoadedRunData(
                    atlas: loadedAtlas,
                    suite: loadedSuite,
                    runs: restRuns,
                    evidenceIssue: evidenceIssue,
                    maskArtifact: maskArtifact,
                    genRecord: genRecord,
                    compDir: compDir,
                    compMask: compMask,
                    compSummary: compSummary,
                    compRecords: compRecords,
                    compFirstBaselineText: compFirstBaselineText,
                    compFirstMaskedText: compFirstMaskedText,
                    compFirstBaselineOutputReady: compFirstBaselineOutputReady,
                    compFirstMaskedOutputReady: compFirstMaskedOutputReady,
                    compFirstBaselineTPS: compFirstBaselineTPS,
                    compFirstMaskedTPS: compFirstMaskedTPS,
                    compFirstPromptID: compFirstPromptID,
                    compFirstTextDelta: compFirstTextDelta,
                    compFirstRisk: compFirstRisk
                )
            }.value

            // Phase 2: Apply UI state on MainActor
            atlas = context.atlas
            lastRunDirectory = dir

            if let suite = context.suite {
                upsertSuite(suite)
            }

            if let issue = context.evidenceIssue {
                runs = []
                lastError = "Loaded run, but \(issue)"
            } else {
                runs = context.runs
            }

            // Mask
            if let artifact = context.maskArtifact {
                maskLayers = artifact.disabledByLayer
                dropCandidates = artifact.dropCandidatesByLayer
                lockedKeeps = artifact.lockedKeepByLayer
                topKOverride = artifact.topKOverride ?? 0
            } else {
                maskLayers.removeAll()
                dropCandidates.removeAll()
                lockedKeeps.removeAll()
                topKOverride = 0
            }

            // Generation preview
            baselineText = ""
            maskedText = ""
            baselineOutputReady = false
            maskedOutputReady = false
            baselineTokensPerSecond = 0
            maskedTokensPerSecond = 0
            runtimeInfoSummary = ""
            if let first = context.genRecord {
                baselineText = first.text
                baselineOutputReady = true
                baselineTokensPerSecond = first.tokensPerSecond
                runtimeInfoSummary = Self.runtimeSummary(
                    mode: first.runtimeMode,
                    backend: first.runtimeBackend,
                    device: first.runtimeDevice,
                    metalEnabled: first.runtimeMetalEnabled,
                    jangToolsVersion: first.jangToolsVersion,
                    mlxVersion: first.mlxVersion,
                    mlxLMVersion: first.mlxLMVersion,
                    sourceModelPath: first.sourceModelPath,
                    hookedMOELayers: first.hookedMOELayers,
                    expectedMOELayers: first.expectedMOELayers,
                    hookCoverageComplete: first.hookCoverageComplete
                )
            }

            // Comparison
            lastEvalDirectory = nil
            comparisonPreviewRows = []
            lastEvalSummary = ""
            if let compDir = context.compDir, let compMask = context.compMask {
                guard Self.comparisonMask(compMask, matches: currentRuntimeMask()) else {
                    lastEvalSummary = "Latest saved eval was for a different mask. Rerun Compare Suite before pruning."
                    selectedExpert = nil
                    return
                }
                lastEvalDirectory = compDir
                lastEvalSummary = context.compSummary
                comparisonPreviewRows = context.compRecords
                if context.compFirstBaselineOutputReady {
                    baselineText = context.compFirstBaselineText
                    maskedText = context.compFirstMaskedText
                    baselineOutputReady = context.compFirstBaselineOutputReady
                    maskedOutputReady = context.compFirstMaskedOutputReady
                    baselineTokensPerSecond = context.compFirstBaselineTPS
                    maskedTokensPerSecond = context.compFirstMaskedTPS
                    if let first = context.compRecords.first {
                        runtimeInfoSummary = Self.runtimeSummary(
                            mode: first.runtimeMode,
                            backend: first.runtimeBackend,
                            device: first.runtimeDevice,
                            metalEnabled: first.runtimeMetalEnabled,
                            jangToolsVersion: first.jangToolsVersion,
                            mlxVersion: first.mlxVersion,
                            mlxLMVersion: first.mlxLMVersion,
                            sourceModelPath: first.sourceModelPath,
                            hookedMOELayers: first.hookedMOELayers,
                            expectedMOELayers: first.expectedMOELayers,
                            hookCoverageComplete: first.hookCoverageComplete
                        )
                    }
                    if context.compSummary.isEmpty {
                        lastEvalSummary = "Loaded eval: \(context.compFirstPromptID), delta \(String(format: "%.2f", context.compFirstTextDelta)), risk \(context.compFirstRisk)"
                    }
                }
            }

            // Backfill eval index if needed (fast file check — on MainActor is fine)
            if let compDir = context.compDir, !context.compRecords.isEmpty {
                let indexURL = compDir.appendingPathComponent("eval_index.json")
                if !FileManager.default.fileExists(atPath: indexURL.path),
                   let summary = Self.loadComparisonSummary(from: compDir) {
                    do {
                        try Self.writeEvalIndex(
                            to: indexURL,
                            runID: dir.lastPathComponent,
                            maskID: compDir.lastPathComponent,
                            summary: summary,
                            records: context.compRecords,
                            suiteURL: dir.appendingPathComponent("suite.jsonl")
                        )
                    } catch {
                        lastEvalSummary = lastEvalSummary.isEmpty
                            ? "Loaded eval, but could not rebuild eval_index.json: \(error.localizedDescription)"
                            : "\(lastEvalSummary). Could not rebuild eval_index.json: \(error.localizedDescription)"
                    }
                }
            }

            selectedExpert = nil
        } catch {
            lastError = "Could not load Expert Lab run: \(error.localizedDescription)"
        }
    }
    func openRecoveryFolder() {
        let target = lastEvalDirectory
            ?? lastRunDirectory
            ?? selectedRunSummary.map { URL(fileURLWithPath: $0.directoryPath, isDirectory: true) }
            ?? artifactRoot()
        NSWorkspace.shared.activateFileViewerSelecting([target])
    }

    func cleanPartialRun() {
        guard let dir = partialRunDirectory else {
            lastError = "No partial Expert Lab run is selected."
            return
        }
        do {
            try FileManager.default.removeItem(at: dir)
            if lastRunDirectory == dir {
                lastRunDirectory = nil
                atlas = nil
                runs = []
                clearMask()
            }
            if selectedRunID == dir.lastPathComponent {
                selectedRunID = ""
            }
            reloadRunHistory()
            lastError = "Cleaned partial run at \(dir.path)."
        } catch CocoaError.fileNoSuchFile {
            if lastRunDirectory == dir {
                lastRunDirectory = nil
            }
            reloadRunHistory()
            lastError = "Partial run was already gone."
        } catch {
            lastError = "Could not clean partial run: \(error.localizedDescription)"
        }
    }


    func copyRecoveryDiagnostics() {
        let selectedFailure = selectedRunSummary.flatMap { run -> String? in
            guard let failureStage = run.failureStage else { return nil }
            return "\(failureStage): \(run.failureMessage ?? "no failure message recorded")"
        } ?? "none"
        let text = """
        JANG Studio Expert Lab diagnostics
        model_path: \(modelPath.path)
        source_model_path: \(sourceModelPath?.path ?? "not recorded")
        capability: \(capability.summary)
        selected_suite: \(selectedSuite.name)
        selected_prompt: \(selectedPromptID)
        status: \(statusText)
        progress: \(runProgress)/\(runProgressTotal)
        last_error: \(lastError ?? "none")
        selected_run_failure: \(selectedFailure)
        last_run_directory: \(lastRunDirectory?.path ?? "none")
        last_eval_directory: \(lastEvalDirectory?.path ?? "none")
        mask_summary: \(maskSummary)
        last_eval_summary: \(lastEvalSummary.isEmpty ? "none" : lastEvalSummary)
        """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        lastError = "Expert Lab diagnostics copied to clipboard."
    }

    private func upsertSuite(_ suite: ExpertPromptSuite) {
        if let index = suites.firstIndex(where: { $0.name == suite.name }) {
            suites[index] = suite
        } else {
            suites.append(suite)
        }
        selectedSuiteName = suite.name
        selectedPromptID = suite.prompts.first?.id ?? ""
    }

    func filteredEntries(for layer: Int) -> [ExpertAtlasEntry] {
        filteredAtlasEntries
            .filter { $0.layer == layer }
            .sorted { lhs, rhs in
                sortedAtlasOrder(lhs, rhs)
            }
    }

    private func sortedAtlasOrder(_ lhs: ExpertAtlasEntry, _ rhs: ExpertAtlasEntry) -> Bool {
        switch atlasSort {
        case .expertID:
            if lhs.expert != rhs.expert { return lhs.expert < rhs.expert }
        case .hits:
            if lhs.hits != rhs.hits { return lhs.hits > rhs.hits }
        case .routerMass:
            if lhs.probabilityMass != rhs.probabilityMass {
                return lhs.probabilityMass > rhs.probabilityMass
            }
        case .activationRank:
            let left = lhs.hits > 0 ? lhs.meanSelectedRank : Float.greatestFiniteMagnitude
            let right = rhs.hits > 0 ? rhs.meanSelectedRank : Float.greatestFiniteMagnitude
            if left != right { return left < right }
        case .tokenDepth:
            let left = lhs.meanTokenIndex ?? Float.greatestFiniteMagnitude
            let right = rhs.meanTokenIndex ?? Float.greatestFiniteMagnitude
            if left != right { return left < right }
        case .coactivation:
            let left = Self.strongestCoactivationScore(lhs)
            let right = Self.strongestCoactivationScore(rhs)
            if left.jaccard != right.jaccard { return left.jaccard > right.jaccard }
            if left.count != right.count { return left.count > right.count }
        case .domainLift:
            let left = lhs.domainLift.values.max() ?? 0
            let right = rhs.domainLift.values.max() ?? 0
            if left != right { return left > right }
        case .regressionSeverity:
            let left = selectedExpertRegressionSeverityRank(for: lhs)
            let right = selectedExpertRegressionSeverityRank(for: rhs)
            if left != right { return left > right }
        case .confidence:
            if lhs.confidenceScore != rhs.confidenceScore {
                return lhs.confidenceScore > rhs.confidenceScore
            }
        case .safeDrop:
            let left = isSafeDropCandidate(layer: lhs.layer, expert: lhs.expert)
            let right = isSafeDropCandidate(layer: rhs.layer, expert: rhs.expert)
            if left != right { return left && !right }
            if lhs.hits != rhs.hits { return lhs.hits < rhs.hits }
        }
        if lhs.layer != rhs.layer { return lhs.layer < rhs.layer }
        return lhs.expert < rhs.expert
    }

    private nonisolated static func matchesIntegerFilter(_ value: Int, text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        let normalized = trimmed
            .lowercased()
            .replacingOccurrences(of: "layer", with: "")
            .replacingOccurrences(of: "expert", with: "")
            .replacingOccurrences(of: "l", with: "")
            .replacingOccurrences(of: "e", with: "")
            .replacingOccurrences(of: " ", with: "")
        for token in normalized.split(separator: ",") {
            if token.contains("-") {
                let bounds = token.split(separator: "-", maxSplits: 1).compactMap { Int($0) }
                if bounds.count == 2 {
                    let lower = min(bounds[0], bounds[1])
                    let upper = max(bounds[0], bounds[1])
                    if value >= lower && value <= upper { return true }
                }
            } else if Int(token) == value {
                return true
            }
        }
        return false
    }

    private nonisolated static func matchesDomainFilter(_ entry: ExpertAtlasEntry, text: String) -> Bool {
        let tokens = text
            .split { $0 == "," || $0 == "\n" || $0 == "\t" }
            .map { normalizedDomainFilterText(String($0)) }
            .filter { !$0.isEmpty }
        guard !tokens.isEmpty else { return true }
        let haystack = domainFilterHaystack(for: entry)
        return tokens.allSatisfy { token in
            haystack.contains(token)
        }
    }

    private nonisolated static func domainFilterHaystack(for entry: ExpertAtlasEntry) -> String {
        var parts: [String] = [
            entry.generatedLabel,
            entry.userLabel ?? ""
        ]
        for domain in entry.domains.keys.sorted() + entry.domainLift.keys.sorted() {
            parts.append(domain)
            parts.append(ExpertDomainTaxonomy.canonicalDomain(domain))
            parts.append(ExpertDomainTaxonomy.canonicalSemanticDomain(domain))
            parts.append(ExpertDomainTaxonomy.displayName(for: domain))
        }
        for evidence in entry.promptEvidence ?? [] {
            parts.append(evidence.domain)
            parts.append(ExpertDomainTaxonomy.canonicalDomain(evidence.domain))
            parts.append(ExpertDomainTaxonomy.canonicalSemanticDomain(evidence.domain))
            parts.append(ExpertDomainTaxonomy.displayName(for: evidence.domain))
            parts.append(contentsOf: evidence.tags)
        }
        return normalizedDomainFilterText(parts.joined(separator: " "))
    }

    private nonisolated static func matchesPromptFilter(_ entry: ExpertAtlasEntry, text: String) -> Bool {
        let tokens = text
            .split { $0 == "," || $0 == "\n" || $0 == "\t" }
            .map { normalizedDomainFilterText(String($0)) }
            .filter { !$0.isEmpty }
        guard !tokens.isEmpty else { return true }
        let haystack = promptFilterHaystack(for: entry)
        return tokens.allSatisfy { token in
            haystack.contains(token)
        }
    }

    private nonisolated static func promptFilterHaystack(for entry: ExpertAtlasEntry) -> String {
        var parts = entry.topPrompts
        for evidence in entry.promptEvidence ?? [] {
            parts.append(evidence.promptID)
            parts.append(evidence.domain)
            parts.append(evidence.subdomain ?? "")
            parts.append(contentsOf: evidence.tags)
            parts.append(evidence.promptExcerpt)
        }
        return normalizedDomainFilterText(parts.joined(separator: " "))
    }

    private nonisolated static func normalizedDomainFilterText(_ raw: String) -> String {
        raw
            .lowercased()
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func maxHits(for layer: Int) -> Int {
        atlas?.experts.filter { $0.layer == layer }.map(\.hits).max() ?? 0
    }

    private nonisolated static func metricScales(for entries: [ExpertAtlasEntry]) -> [Int: ExpertAtlasMetricScale] {
        Dictionary(grouping: entries, by: \.layer).mapValues { layerEntries in
            ExpertAtlasMetricScale(
                maxHits: layerEntries.map(\.hits).max() ?? 0,
                maxMass: layerEntries.map(\.probabilityMass).max() ?? 0,
                maxEntropy: layerEntries.map(\.entropyContribution).max() ?? 0
            )
        }
    }

    func metricIntensity(_ entry: ExpertAtlasEntry) -> Double {
        metricIntensity(entry, scales: atlasLayerMetricScales)
    }

    func metricIntensity(
        _ entry: ExpertAtlasEntry,
        scales: [Int: ExpertAtlasMetricScale]
    ) -> Double {
        let scale = scales[entry.layer] ?? ExpertAtlasMetricScale(maxHits: 0, maxMass: 0, maxEntropy: 0)
        switch atlasMetric {
        case .frequency:
            let maxHits = max(scale.maxHits, 1)
            return min(1.0, Double(entry.hits) / Double(maxHits))
        case .routerMass:
            return min(1.0, Double(entry.probabilityMass / Swift.max(scale.maxMass, 0.0001)))
        case .domainLift:
            return min(1.0, Double((entry.domainLift.values.max() ?? 0) / 4))
        case .entropy:
            return min(1.0, Double(entry.entropyContribution / Swift.max(scale.maxEntropy, 0.0001)))
        case .confidence:
            return min(1.0, Double(entry.confidenceScore))
        }
    }

    func select(_ entry: ExpertAtlasEntry) {
        selectedExpert = SelectedExpert(layer: entry.layer, expert: entry.expert)
    }

    func isGroupSelected(layer: Int, expert: Int) -> Bool {
        groupSelection.contains(ExpertCoordinate(layer: layer, expert: expert))
    }

    func previewLasso(rect: CGRect, cellFrames: [ExpertCoordinate: CGRect]) {
        groupSelection = ExpertAtlasSelection.coordinates(intersecting: rect, cellFrames: cellFrames)
    }

    func clearGroupSelection() {
        groupSelection.removeAll()
    }

    func applySelectionMask() {
        guard !groupSelection.isEmpty else { return }
        for coordinate in groupSelection {
            maskLayers[coordinate.layer, default: []].insert(coordinate.expert)
        }
        maskStateChanged()
    }

    func applySelectionDropCandidates() {
        guard !groupSelection.isEmpty else { return }
        for coordinate in groupSelection {
            dropCandidates[coordinate.layer, default: []].insert(coordinate.expert)
            remove(expert: coordinate.expert, from: &lockedKeeps, layer: coordinate.layer)
        }
        maskStateChanged()
    }

    func applySelectionLockedKeep() {
        guard !groupSelection.isEmpty else { return }
        for coordinate in groupSelection {
            lockedKeeps[coordinate.layer, default: []].insert(coordinate.expert)
            remove(expert: coordinate.expert, from: &dropCandidates, layer: coordinate.layer)
        }
        maskStateChanged()
    }

    func displayLabel(for entry: ExpertAtlasEntry) -> String {
        let userLabel = entry.userLabel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !userLabel.isEmpty { return userLabel }
        return entry.generatedLabel
    }

    func dominantDomainSummary(for entry: ExpertAtlasEntry) -> String {
        if let domain = ExpertDomainTaxonomy.dominantDomain(domains: entry.domains, domainLift: entry.domainLift) {
            return ExpertDomainTaxonomy.displayName(for: domain)
        }
        if let domain = entry.domains.keys.sorted().first ?? entry.promptEvidence?.first?.domain {
            return ExpertDomainTaxonomy.displayName(for: domain)
        }
        return "none"
    }

    func evidenceCount(for entry: ExpertAtlasEntry) -> Int {
        entry.evidenceCount ?? entry.promptEvidence?.count ?? entry.topPrompts.count
    }

    func tokenDepthSummary(for entry: ExpertAtlasEntry) -> String {
        guard let mean = entry.meanTokenIndex else { return "n/a" }
        guard let min = entry.minTokenIndex,
              let max = entry.maxTokenIndex,
              min != max else {
            return String(format: "%.1f", mean)
        }
        return String(format: "%.1f (%d-%d)", mean, min, max)
    }

    func coactivationSummary(for entry: ExpertAtlasEntry) -> String {
        let neighbors = Self.sortedCoactivationNeighbors(entry).prefix(3)
        guard !neighbors.isEmpty else { return "none" }
        return neighbors
            .map { String(format: "E%d J%.2f", $0.expert, $0.jaccard) }
            .joined(separator: ", ")
    }

    private nonisolated static func strongestCoactivationScore(_ entry: ExpertAtlasEntry) -> (jaccard: Float, count: Int) {
        guard let neighbor = sortedCoactivationNeighbors(entry).first else { return (0, 0) }
        return (neighbor.jaccard, neighbor.count)
    }

    private nonisolated static func sortedCoactivationNeighbors(_ entry: ExpertAtlasEntry) -> [ExpertCoactivationNeighbor] {
        entry.coactivationNeighbors.sorted { lhs, rhs in
            if lhs.jaccard != rhs.jaccard { return lhs.jaccard > rhs.jaccard }
            if lhs.count != rhs.count { return lhs.count > rhs.count }
            return lhs.expert < rhs.expert
        }
    }

    func userLabel(layer: Int, expert: Int) -> String {
        atlasEntry(layer: layer, expert: expert)?.userLabel ?? ""
    }

    func userNotes(layer: Int, expert: Int) -> String {
        atlasEntry(layer: layer, expert: expert)?.userNotes ?? ""
    }

    func manualAnnotationSummary(for entry: ExpertAtlasEntry) -> String {
        let label = entry.userLabel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let notes = entry.userNotes?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        switch (!label.isEmpty, !notes.isEmpty) {
        case (true, true):
            return "label+note"
        case (true, false):
            return "label"
        case (false, true):
            return "note"
        case (false, false):
            return "none"
        }
    }

    func setUserLabel(_ label: String, layer: Int, expert: Int) {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        updateAtlasEntry(layer: layer, expert: expert) { entry in
            entry.userLabel = trimmed.isEmpty ? nil : label
        }
    }

    func setUserNotes(_ notes: String, layer: Int, expert: Int) {
        let trimmed = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        updateAtlasEntry(layer: layer, expert: expert) { entry in
            entry.userNotes = trimmed.isEmpty ? nil : notes
        }
    }

    private func atlasEntry(layer: Int, expert: Int) -> ExpertAtlasEntry? {
        atlas?.experts.first { $0.layer == layer && $0.expert == expert }
    }

    private func updateAtlasEntry(
        layer: Int,
        expert: Int,
        mutate: (inout ExpertAtlasEntry) -> Void
    ) {
        guard let current = atlas,
              let index = current.experts.firstIndex(where: { $0.layer == layer && $0.expert == expert }) else {
            return
        }
        var entries = current.experts
        var entry = entries[index]
        mutate(&entry)
        entries[index] = entry
        atlas = ExpertAtlas(
            generatedAt: current.generatedAt,
            promptCount: current.promptCount,
            experts: entries,
            sourceNumExpertsByLayer: current.sourceNumExpertsByLayer
        )
        persistAtlasEdits()
    }

    /// Offloads atlas persistence to a background task so user interaction (toggling
    /// experts, setting notes) doesn't block the main thread with file I/O.
    private func persistAtlasEdits() {
        guard let atlas, let lastRunDirectory else { return }
        Task.detached(priority: .userInitiated) { [atlas, lastRunDirectory, self] in
            do {
                try Self._writeAtlas(atlas, to: lastRunDirectory)
            } catch {
                await MainActor.run {
                    self.lastError = "Could not persist atlas notes: \(error.localizedDescription)"
                }
            }
        }
    }


    private func persistAtlas(_ atlas: ExpertAtlas, to runDirectory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(atlas).write(to: runDirectory.appendingPathComponent("atlas.json"))
    }

    /// Nonisolated static variant of persistAtlas, safe to call from Task.detached.
    private nonisolated static func _writeAtlas(_ atlas: ExpertAtlas, to runDirectory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(atlas).write(to: runDirectory.appendingPathComponent("atlas.json"))
    }

    func isMasked(layer: Int, expert: Int) -> Bool {
        maskLayers[layer]?.contains(expert) ?? false
    }

    func isDropCandidate(layer: Int, expert: Int) -> Bool {
        dropCandidates[layer]?.contains(expert) ?? false
    }

    func isLockedKeep(layer: Int, expert: Int) -> Bool {
        lockedKeeps[layer]?.contains(expert) ?? false
    }

    func isSafeDropCandidate(layer: Int, expert: Int) -> Bool {
        latestComparisonSummary()?.safeDropCandidates.contains(
            ExpertCoordinate(layer: layer, expert: expert)
        ) ?? false
    }

    func setMasked(_ disabled: Bool, layer: Int, expert: Int) {
        var experts = maskLayers[layer] ?? []
        if disabled {
            experts.insert(expert)
        } else {
            experts.remove(expert)
        }
        if experts.isEmpty {
            maskLayers.removeValue(forKey: layer)
        } else {
            maskLayers[layer] = experts
        }
        maskStateChanged()
    }

    func setDropCandidate(_ enabled: Bool, layer: Int, expert: Int) {
        var experts = dropCandidates[layer] ?? []
        if enabled {
            experts.insert(expert)
            remove(expert: expert, from: &lockedKeeps, layer: layer)
        } else {
            experts.remove(expert)
        }
        if experts.isEmpty {
            dropCandidates.removeValue(forKey: layer)
        } else {
            dropCandidates[layer] = experts
        }
        maskStateChanged()
    }

    func setLockedKeep(_ enabled: Bool, layer: Int, expert: Int) {
        var experts = lockedKeeps[layer] ?? []
        if enabled {
            experts.insert(expert)
            remove(expert: expert, from: &dropCandidates, layer: layer)
        } else {
            experts.remove(expert)
        }
        if experts.isEmpty {
            lockedKeeps.removeValue(forKey: layer)
        } else {
            lockedKeeps[layer] = experts
        }
        maskStateChanged()
    }

    private func remove(expert: Int, from map: inout [Int: Set<Int>], layer: Int) {
        var experts = map[layer] ?? []
        experts.remove(expert)
        if experts.isEmpty {
            map.removeValue(forKey: layer)
        } else {
            map[layer] = experts
        }
    }

    func clearMask() {
        maskLayers.removeAll()
        topKOverride = 0
        maskStateChanged()
    }

    func cancelRun() {
        cancelRequested = true
        statusText = "Cancelling after current prompt"
    }

    func saveMask() {
        do {
            let url = try writeCurrentMask(named: "current_mask.json")
            statusText = "Mask saved to \(url.lastPathComponent)"
        } catch {
            lastError = "Could not save mask: \(error.localizedDescription)"
        }
    }

    func importMask() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        panel.directoryURL = lastRunDirectory?.appendingPathComponent("masks", isDirectory: true)
        panel.prompt = "Load"
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try loadMask(from: url)
                maskStateChanged()
            } catch {
                lastError = "Could not load mask: \(error.localizedDescription)"
            }
        }
    }

    func maskStateChanged() {
        invalidateComparisonSummary()
        persistCurrentMask()
    }

    func persistCurrentMask() {
        guard lastRunDirectory != nil else { return }
        do {
            _ = try writeCurrentMask(named: "current_mask.json")
        } catch {
            lastError = "Could not persist mask: \(error.localizedDescription)"
        }
    }

    private func writeCurrentMask(named name: String) throws -> URL {
        guard let lastRunDirectory else {
            throw ExpertLabMaskPersistenceError.missingRun
        }
        let dir = lastRunDirectory.appendingPathComponent("masks", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(currentMaskArtifact()).write(to: url)
        return url
    }


    private func loadMaskIfPresent(from runDirectory: URL) {
        let url = runDirectory
            .appendingPathComponent("masks", isDirectory: true)
            .appendingPathComponent("current_mask.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            maskLayers.removeAll()
            dropCandidates.removeAll()
            lockedKeeps.removeAll()
            topKOverride = 0
            return
        }
        do {
            try loadMask(from: url)
        } catch {
            lastError = "Could not restore saved mask: \(error.localizedDescription)"
        }
    }

    private func loadMask(from url: URL) throws {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let artifact = try decoder.decode(ExpertLabMaskArtifact.self, from: Data(contentsOf: url))
        maskLayers = artifact.disabledByLayer
        dropCandidates = artifact.dropCandidatesByLayer
        lockedKeeps = artifact.lockedKeepByLayer
        topKOverride = artifact.topKOverride ?? 0
    }

    private func currentMaskArtifact() -> ExpertLabMaskArtifact {
        ExpertLabMaskArtifact(
            disabledByLayer: maskLayers,
            dropCandidatesByLayer: dropCandidates,
            lockedKeepByLayer: lockedKeeps,
            topKOverride: topKOverride > 0 ? topKOverride : nil
        )
    }

    private func invalidateComparisonSummary() {
        guard lastEvalDirectory != nil else { return }
        lastEvalDirectory = nil
        comparisonPreviewRows = []
        lastEvalSummary = "Mask changed after the last comparison. Rerun Compare or Compare Suite before pruning."
    }

    private func currentRuntimeMask() -> JANGKit.ExpertMask {
        var mask = JANGKit.ExpertMask(layers: runtimeMaskLayers)
        mask.topKOverride = topKOverride > 0 ? topKOverride : nil
        return mask
    }

    private func samplingConfig(for prompt: ExpertPrompt) -> JANGKit.SamplingConfig {
        var config = JANGKit.SamplingConfig(maxTokens: maxTokens)
        if let promptMaxTokens = prompt.maxNewTokens, promptMaxTokens > 0 {
            config.maxTokens = min(promptMaxTokens, Self.maximumPromptSuiteMaxTokens)
        }
        if let promptTemperature = prompt.temperature {
            config.temperature = promptTemperature
        }
        return config
    }

    private func blockingRuntimeMaskIssue() -> ExpertMaskValidationIssue? {
        ExpertMaskEngine.validate(
            mask: currentRuntimeMask(),
            sourceNumExpertsByLayer: sourceExpertsByLayerForAtlas(),
            trainedTopKByLayer: trainedTopKByLayerForAtlas(),
            hotExperts: hotExpertCoordinates
        )
        .first { $0.severity == .error }
    }

    func runLivePrompt() async {
        let text = livePromptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            lastError = "Write a prompt before running Live Prompt."
            return
        }
        guard capability.isTraceSupported else {
            lastError = capability.detail
            return
        }
        guard !isRunning else { return }
        let stamp = ISO8601DateFormatter()
            .string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let prompt = ExpertPrompt(
            id: "live-\(stamp)",
            domain: "manual",
            text: text,
            subdomain: "user",
            maxNewTokens: maxTokens,
            temperature: 0.0,
            tags: ["live", "manual"]
        )
        let suite = ExpertPromptSuite(name: "Live Prompt", prompts: [prompt])
        upsertSuite(suite)

        isRunning = true
        statusText = "Running live prompt"
        runProgress = 0
        runProgressTotal = hasMask ? 2 : 1
        baselineText = ""
        maskedText = ""
        baselineOutputReady = false
        maskedOutputReady = false
        baselineTokensPerSecond = 0
        maskedTokensPerSecond = 0
        lastEvalSummary = ""
        lastEvalDirectory = nil
        comparisonPreviewRows = []
        lastError = nil
        cancelRequested = false
        defer {
            isRunning = false
            statusText = "Idle"
            cancelRequested = false
        }

        do {
            if capability.runtimeMode == .bf16VMLX {
                statusText = "Tracing BF16/vMLX live prompt"
                let baselineRuns = try await runVMLX(suite: suite, emitTokenTrace: emitTokenTrace)
                guard let baselineRun = baselineRuns.first else {
                    throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runner returned no live prompt output")
                }
                try await finishTraceRun(baselineRuns, suite: suite)
                let baseline = baselineRun.result
                baselineText = baseline.text
                baselineOutputReady = true
                baselineTokensPerSecond = baseline.tokensPerSecond
                runProgress = 1

                guard hasMask else {
                    lastEvalSummary = "Live prompt output from the BF16/vMLX source. Disable experts, then run Probe again to compare."
                    return
                }

                if let error = blockingRuntimeMaskIssue() {
                    lastEvalSummary = "Live prompt output from the BF16/vMLX source. Masked generation skipped."
                    lastError = "Mask invalid: \(error.message)"
                    return
                }

                statusText = "Running BF16/vMLX disabled experts"
                let mask = currentRuntimeMask()
                let maskedRuns = try await runVMLX(suite: suite, mask: mask, emitTokenTrace: true)
                guard let maskedRun = maskedRuns.first else {
                    throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runner returned no masked live prompt output")
                }
                let masked = maskedRun.result
                maskedText = masked.text
                maskedOutputReady = true
                maskedTokensPerSecond = masked.tokensPerSecond
                let record = Self.comparisonRecord(
                    prompt: prompt,
                    baseline: baseline,
                    masked: masked,
                    samplingConfig: samplingConfig(for: prompt)
                )
                lastEvalSummary = "Live prompt A/B: \(ExpertPromptEvaluator.evaluationSummary(record.evaluation, textDelta: record.textDelta))"
                lastEvalDirectory = try persistComparison(
                    mask: mask,
                    records: [record]
                )
                comparisonPreviewRows = Self.storedEvalRecords(from: [record])
                runProgress = 2
                return
            }

            let model = try await loadModel()
            let config = samplingConfig(for: prompt)
            statusText = "Tracing live prompt"
            let baseline = try await model.generateWithTrace(
                prompt: prompt.text,
                config: config,
                traceConfig: JANGKit.ExpertTraceConfig(
                    mask: nil,
                    emitTokenTrace: emitTokenTrace,
                    maxTraceTokens: maxTraceTokens
                )
            )
            let baselineRun = ExpertPromptRun(prompt: prompt, result: baseline)
            try await finishTraceRun([baselineRun], suite: suite)
            baselineText = baseline.text
            baselineOutputReady = true
            baselineTokensPerSecond = baseline.tokensPerSecond
            runProgress = 1

            guard hasMask else {
                lastEvalSummary = "Live prompt output from the original model. Disable experts, then run Probe again to compare."
                return
            }

            if let error = blockingRuntimeMaskIssue() {
                lastEvalSummary = "Live prompt output from the original model. Masked generation skipped."
                lastError = "Mask invalid: \(error.message)"
                return
            }

            statusText = "Running with disabled experts"
            let mask = currentRuntimeMask()
            let masked = try await model.generateWithTrace(
                prompt: prompt.text,
                config: config,
                traceConfig: JANGKit.ExpertTraceConfig(mask: mask, emitTokenTrace: true, maxTraceTokens: maxTraceTokens)
            )
            maskedText = masked.text
            maskedOutputReady = true
            maskedTokensPerSecond = masked.tokensPerSecond
            let evaluation = ExpertPromptEvaluator.evaluate(
                prompt: prompt,
                baselineText: baseline.text,
                maskedText: masked.text
            )
            let textDelta = ExpertPromptEvaluator.normalizedTextDelta(baseline.text, masked.text)
            let latencyDeltaPct = Self.latencyDeltaPct(baseline: baseline, masked: masked)
            lastEvalSummary = "Live prompt A/B: \(ExpertPromptEvaluator.evaluationSummary(evaluation, textDelta: textDelta))"
            let records = [
                ExpertComparisonPromptRecord(
                    prompt: prompt,
                    baseline: baseline,
                    masked: masked,
                    evaluation: evaluation,
                    textDelta: textDelta,
                    latencyDeltaPct: latencyDeltaPct,
                    samplingConfig: config
                )
            ]
            lastEvalDirectory = try persistComparison(
                mask: mask,
                records: records
            )
            comparisonPreviewRows = Self.storedEvalRecords(from: records)
            runProgress = 2
        } catch {
            lastError = "Live prompt failed: \(error.localizedDescription)"
        }
    }

    func runTrace() async {
        await runTrace(suite: selectedSuite)
    }

    private func runTrace(suite: ExpertPromptSuite) async {
        guard capability.isTraceSupported else {
            lastError = capability.detail
            return
        }
        guard !isRunning else { return }
        isRunning = true
        statusText = "Loading model"
        runProgress = 0
        runProgressTotal = suite.prompts.count
        baselineText = ""
        maskedText = ""
        baselineOutputReady = false
        maskedOutputReady = false
        lastEvalSummary = ""
        lastEvalDirectory = nil
        comparisonPreviewRows = []
        lastError = nil
        cancelRequested = false
        var collected: [ExpertPromptRun] = []
        defer {
            isRunning = false
            statusText = "Idle"
            cancelRequested = false
        }

        do {
            if capability.runtimeMode == .bf16VMLX {
                statusText = "Tracing BF16/vMLX source"
                collected = try await runVMLX(suite: suite, emitTokenTrace: emitTokenTrace)
                runProgress = collected.count
                if let latest = collected.last ?? collected.first {
                    baselineText = latest.result.text
                    baselineOutputReady = true
                    baselineTokensPerSecond = latest.result.tokensPerSecond
                    statusText = Self.traceProgressSummary(
                        completed: collected.count,
                        total: suite.prompts.count,
                        latest: latest
                    )
                }
                try await finishTraceRun(collected, suite: suite)
                return
            }

            let model = try await loadModel()
            let runner = ExpertPromptSuiteRunner(model: model)
            let config = JANGKit.SamplingConfig(maxTokens: maxTokens)
            let traceConfig = JANGKit.ExpertTraceConfig(
                mask: nil,
                emitTokenTrace: emitTokenTrace,
                maxTraceTokens: maxTraceTokens
            )

            for prompt in suite.prompts {
                if cancelRequested {
                    throw CancellationError()
                }
                statusText = "Tracing \(prompt.id) (\(collected.count + 1)/\(suite.prompts.count))"
                let onePromptSuite = ExpertPromptSuite(name: suite.name, prompts: [prompt])
                let result = try await runner.run(
                    suite: onePromptSuite,
                    config: config,
                    traceConfig: traceConfig
                )
                collected.append(contentsOf: result)
                if let latest = result.first {
                    baselineText = latest.result.text
                    baselineOutputReady = true
                    baselineTokensPerSecond = latest.result.tokensPerSecond
                    statusText = Self.traceProgressSummary(
                        completed: collected.count,
                        total: suite.prompts.count,
                        latest: latest
                    )
                }
                runProgress += 1
                // M221: yield after each trace prompt so the main thread can
                // process UI events (loading overlay update, cancel button tap).
                // Without this, @MainActor async code that doesn't suspend
                // between iterations starves the UI runloop → spinning wheel.
                await Task.yield()
                if cancelRequested {
                    throw CancellationError()
                }
            }

            try await finishTraceRun(collected, suite: suite)
        } catch {
            if error is CancellationError {
                let message = "Trace cancelled after \(collected.count) of \(suite.prompts.count) prompts."
                do {
                    try await finishTraceRun(
                        collected,
                        suite: suite,
                        failureStage: "cancelled",
                        failureMessage: message
                    )
                    lastError = "\(message) Partial artifacts were saved."
                } catch {
                    lastError = "\(message) Could not save partial artifacts: \(error.localizedDescription)"
                }
            } else {
                let message = "Trace failed after \(collected.count) of \(suite.prompts.count) prompts: \(error.localizedDescription)"
                if collected.isEmpty {
                    lastError = message
                } else {
                    do {
                        try await finishTraceRun(
                            collected,
                            suite: suite,
                            failureStage: "trace_failed",
                            failureMessage: message
                        )
                        lastError = "\(message) Partial artifacts were saved."
                    } catch {
                        lastError = "\(message) Could not save partial artifacts: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    func runMaskedCompare() async {
        guard capability.isTraceSupported else {
            lastError = capability.detail
            return
        }
        guard hasMask else {
            lastError = "Disable at least one expert or set a top-k override before comparing."
            return
        }
        guard let prompt = selectedPrompt else {
            lastError = "No prompt selected."
            return
        }
        var validationMask = JANGKit.ExpertMask(layers: runtimeMaskLayers)
        validationMask.topKOverride = topKOverride > 0 ? topKOverride : nil
        let issues = ExpertMaskEngine.validate(
            mask: validationMask,
            sourceNumExpertsByLayer: sourceExpertsByLayerForAtlas(),
            trainedTopKByLayer: trainedTopKByLayerForAtlas(),
            hotExperts: hotExpertCoordinates
        )
        if let error = issues.first(where: { $0.severity == .error }) {
            lastError = "Mask invalid: \(error.message)"
            return
        }
        guard !isRunning else { return }
        isRunning = true
        statusText = "Comparing"
        baselineText = ""
        maskedText = ""
        baselineOutputReady = false
        maskedOutputReady = false
        lastError = nil
        defer {
            isRunning = false
            statusText = "Idle"
        }

        do {
            if capability.runtimeMode == .bf16VMLX {
                let suite = ExpertPromptSuite(name: selectedSuite.name, prompts: [prompt])
                statusText = "Comparing BF16/vMLX baseline"
                let baselineRuns = try await runVMLX(suite: suite, emitTokenTrace: true)
                guard let baselineRun = baselineRuns.first else {
                    throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runner returned no baseline output")
                }
                var mask = JANGKit.ExpertMask(layers: runtimeMaskLayers)
                mask.topKOverride = topKOverride > 0 ? topKOverride : nil
                statusText = "Comparing BF16/vMLX mask"
                let maskedRuns = try await runVMLX(suite: suite, mask: mask, emitTokenTrace: true)
                guard let maskedRun = maskedRuns.first else {
                    throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runner returned no masked output")
                }
                let baseline = baselineRun.result
                let masked = maskedRun.result
                baselineText = baseline.text
                maskedText = masked.text
                baselineOutputReady = true
                maskedOutputReady = true
                baselineTokensPerSecond = baseline.tokensPerSecond
                maskedTokensPerSecond = masked.tokensPerSecond
                let record = Self.comparisonRecord(
                    prompt: prompt,
                    baseline: baseline,
                    masked: masked,
                    samplingConfig: samplingConfig(for: prompt)
                )
                lastEvalSummary = ExpertPromptEvaluator.evaluationSummary(record.evaluation, textDelta: record.textDelta)
                lastEvalDirectory = try persistComparison(
                    mask: mask,
                    records: [record]
                )
                comparisonPreviewRows = Self.storedEvalRecords(from: [record])
                return
            }

            let model = try await loadModel()
            let config = samplingConfig(for: prompt)
            let baseline = try await model.generateWithTrace(
                prompt: prompt.text,
                config: config,
                traceConfig: JANGKit.ExpertTraceConfig(emitTokenTrace: true, maxTraceTokens: maxTraceTokens)
            )
            var mask = JANGKit.ExpertMask(layers: runtimeMaskLayers)
            mask.topKOverride = topKOverride > 0 ? topKOverride : nil
            let masked = try await model.generateWithTrace(
                prompt: prompt.text,
                config: config,
                traceConfig: JANGKit.ExpertTraceConfig(mask: mask, emitTokenTrace: true, maxTraceTokens: maxTraceTokens)
            )
            baselineText = baseline.text
            maskedText = masked.text
            baselineOutputReady = true
            maskedOutputReady = true
            baselineTokensPerSecond = baseline.tokensPerSecond
            maskedTokensPerSecond = masked.tokensPerSecond
            let evaluation = ExpertPromptEvaluator.evaluate(
                prompt: prompt,
                baselineText: baseline.text,
                maskedText: masked.text
            )
            let textDelta = ExpertPromptEvaluator.normalizedTextDelta(baseline.text, masked.text)
            let latencyDeltaPct = Self.latencyDeltaPct(baseline: baseline, masked: masked)
            lastEvalSummary = ExpertPromptEvaluator.evaluationSummary(evaluation, textDelta: textDelta)
            let records = [
                ExpertComparisonPromptRecord(
                    prompt: prompt,
                    baseline: baseline,
                    masked: masked,
                    evaluation: evaluation,
                    textDelta: textDelta,
                    latencyDeltaPct: latencyDeltaPct,
                    samplingConfig: config
                )
            ]
            lastEvalDirectory = try persistComparison(
                mask: mask,
                records: records
            )
            comparisonPreviewRows = Self.storedEvalRecords(from: records)
        } catch {
            lastError = error.localizedDescription
        }
    }

    func runMaskedSuiteCompare() async {
        guard capability.isTraceSupported else {
            lastError = capability.detail
            return
        }
        guard hasMask else {
            lastError = "Disable at least one expert or set a top-k override before comparing."
            return
        }
        guard !selectedSuite.prompts.isEmpty else {
            lastError = "No prompts in the selected suite."
            return
        }
        var validationMask = JANGKit.ExpertMask(layers: runtimeMaskLayers)
        validationMask.topKOverride = topKOverride > 0 ? topKOverride : nil
        let issues = ExpertMaskEngine.validate(
            mask: validationMask,
            sourceNumExpertsByLayer: sourceExpertsByLayerForAtlas(),
            trainedTopKByLayer: trainedTopKByLayerForAtlas(),
            hotExperts: hotExpertCoordinates
        )
        if let error = issues.first(where: { $0.severity == .error }) {
            lastError = "Mask invalid: \(error.message)"
            return
        }
        guard !isRunning else { return }
        isRunning = true
        cancelRequested = false
        runProgress = 0
        runProgressTotal = selectedSuite.prompts.count
        statusText = "Comparing suite"
        baselineText = ""
        maskedText = ""
        baselineOutputReady = false
        maskedOutputReady = false
        comparisonPreviewRows = []
        lastError = nil
        var records: [ExpertComparisonPromptRecord] = []
        var partialEvalDirectory: URL?
        defer {
            isRunning = false
            statusText = "Idle"
        }

        var mask = JANGKit.ExpertMask(layers: runtimeMaskLayers)
        mask.topKOverride = topKOverride > 0 ? topKOverride : nil
        do {
            if capability.runtimeMode == .bf16VMLX {
                statusText = "Comparing BF16/vMLX baseline suite"
                let baselineRuns = try await runVMLX(suite: selectedSuite, emitTokenTrace: true)
                if cancelRequested {
                    throw CancellationError()
                }
                statusText = "Comparing BF16/vMLX masked suite"
                let maskedRuns = try await runVMLX(suite: selectedSuite, mask: mask, emitTokenTrace: true)
                let samplingConfigs = Dictionary(uniqueKeysWithValues: selectedSuite.prompts.map {
                    ($0.id, samplingConfig(for: $0))
                })
                records = try Self.comparisonRecords(
                    prompts: selectedSuite.prompts,
                    baselineRuns: baselineRuns,
                    maskedRuns: maskedRuns,
                    samplingConfigs: samplingConfigs
                )
                if let last = records.last {
                    baselineText = last.baseline.text
                    maskedText = last.masked.text
                    baselineOutputReady = true
                    maskedOutputReady = true
                    baselineTokensPerSecond = last.baseline.tokensPerSecond
                    maskedTokensPerSecond = last.masked.tokensPerSecond
                }
                runProgress = records.count
                lastEvalSummary = Self.suiteEvaluationSummary(records)
                lastEvalDirectory = try persistComparison(
                    mask: mask,
                    records: records
                )
                comparisonPreviewRows = Self.storedEvalRecords(from: records)
                return
            }

            let model = try await loadModel()
            for prompt in selectedSuite.prompts {
                statusText = "Comparing \(prompt.id) (\(records.count + 1)/\(selectedSuite.prompts.count))"
                let config = samplingConfig(for: prompt)
                let baseline = try await model.generateWithTrace(
                    prompt: prompt.text,
                    config: config,
                    traceConfig: JANGKit.ExpertTraceConfig(emitTokenTrace: true, maxTraceTokens: maxTraceTokens)
                )
                let masked = try await model.generateWithTrace(
                    prompt: prompt.text,
                    config: config,
                    traceConfig: JANGKit.ExpertTraceConfig(mask: mask, emitTokenTrace: true, maxTraceTokens: maxTraceTokens)
                )
                let evaluation = ExpertPromptEvaluator.evaluate(
                    prompt: prompt,
                    baselineText: baseline.text,
                    maskedText: masked.text
                )
                let textDelta = ExpertPromptEvaluator.normalizedTextDelta(baseline.text, masked.text)
                let record = ExpertComparisonPromptRecord(
                    prompt: prompt,
                    baseline: baseline,
                    masked: masked,
                    evaluation: evaluation,
                    textDelta: textDelta,
                    latencyDeltaPct: Self.latencyDeltaPct(baseline: baseline, masked: masked),
                    samplingConfig: config
                )
                records.append(record)
                comparisonPreviewRows = Self.storedEvalRecords(from: records)
                lastEvalSummary = Self.suiteComparisonProgressSummary(
                    completed: records.count,
                    total: selectedSuite.prompts.count,
                    records: records,
                    latestPromptID: prompt.id
                )
                partialEvalDirectory = try persistComparison(
                    mask: mask,
                    records: records,
                    directory: partialEvalDirectory
                )
                lastEvalDirectory = partialEvalDirectory
                baselineText = baseline.text
                maskedText = masked.text
                baselineOutputReady = true
                maskedOutputReady = true
                baselineTokensPerSecond = baseline.tokensPerSecond
                maskedTokensPerSecond = masked.tokensPerSecond
                runProgress += 1
                if cancelRequested {
                    throw CancellationError()
                }
            }
            lastEvalSummary = Self.suiteEvaluationSummary(records)
            lastEvalDirectory = try persistComparison(
                mask: mask,
                records: records,
                directory: partialEvalDirectory
            )
            comparisonPreviewRows = Self.storedEvalRecords(from: records)
        } catch {
            if error is CancellationError, !records.isEmpty {
                do {
                    lastEvalDirectory = try persistComparison(
                        mask: mask,
                        records: records,
                        directory: partialEvalDirectory
                    )
                    lastEvalSummary = Self.suiteEvaluationSummary(records)
                    comparisonPreviewRows = Self.storedEvalRecords(from: records)
                    lastError = "Suite comparison cancelled after \(records.count) of \(selectedSuite.prompts.count) prompts. Partial eval artifacts were saved."
                } catch {
                    lastError = "Suite comparison cancelled, and partial eval artifacts could not be saved: \(error.localizedDescription)"
                }
            } else if !records.isEmpty {
                do {
                    lastEvalDirectory = try persistComparison(
                        mask: mask,
                        records: records,
                        directory: partialEvalDirectory
                    )
                    lastEvalSummary = Self.suiteEvaluationSummary(records)
                    comparisonPreviewRows = Self.storedEvalRecords(from: records)
                    lastError = "Suite comparison failed after \(records.count) of \(selectedSuite.prompts.count) prompts: \(error.localizedDescription). Partial eval artifacts were saved."
                } catch {
                    lastError = "Suite comparison failed, and partial eval artifacts could not be saved: \(error.localizedDescription)"
                }
            } else {
                lastError = error.localizedDescription
            }
        }
    }



    private func finishTraceRun(
        _ collected: [ExpertPromptRun],
        suite: ExpertPromptSuite,
        failureStage: String? = nil,
        failureMessage: String? = nil
    ) async throws {
        // Snapshot config values on MainActor before moving computation off-thread.
        let capExpectedExperts = capability.expectedExperts
        let capExpectedLayers = capability.expectedLayers
        let pSourcePath = artifactSourcePath
        let pReviewBundlePath = artifactReviewBundlePath
        let pRuntimeModeFallback = artifactRuntimeModeFallback
        let pEmitTokenTrace = emitTokenTrace
        let pMaxTraceTokens = maxTraceTokens
        let pRoot = artifactRoot()

        // Phase 1: Atlas construction (heavy computation — off MainActor)
        let atlas = await Task.detached(priority: .userInitiated) { [collected, capExpectedExperts, capExpectedLayers] in
            let expected = Self._expectedExpertsByLayer(
                from: collected,
                expectedExperts: capExpectedExperts,
                expectedLayers: capExpectedLayers
            )
            return Self._completedAtlas(
                ExpertAtlasBuilder.build(from: collected, expectedExpertsByLayer: expected),
                expectedExperts: capExpectedExperts,
                expectedLayers: capExpectedLayers
            )
        }.value

        // Phase 2: Persistence (file I/O — off MainActor)
        let runDirectory = try await Task.detached(priority: .userInitiated) {
            [
                collected, atlas, suite,
                failureStage, failureMessage,
                pSourcePath, pReviewBundlePath, pRuntimeModeFallback,
                pEmitTokenTrace, pMaxTraceTokens, pRoot
            ] in
            let runtimeInfo = collected.compactMap(\.result.runtimeInfo).first
            let runID = ISO8601DateFormatter()
                .string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            return try ExpertArtifactWriter.writeRun(
                rootDirectory: pRoot,
                runID: runID,
                sourcePath: pSourcePath,
                reviewBundlePath: pReviewBundlePath,
                runtimeMode: runtimeInfo?.runtimeMode ?? pRuntimeModeFallback,
                runtimeBackend: runtimeInfo?.backend,
                runtimeDevice: runtimeInfo?.deviceName,
                runtimeMetalEnabled: runtimeInfo?.metalEnabled,
                jangToolsVersion: runtimeInfo?.jangToolsVersion,
                mlxVersion: runtimeInfo?.mlxVersion,
                mlxLMVersion: runtimeInfo?.mlxLMVersion,
                mlxVLMVersion: runtimeInfo?.mlxVLMVersion,
                suite: suite,
                traceConfig: JANGKit.ExpertTraceConfig(
                    emitTokenTrace: pEmitTokenTrace,
                    maxTraceTokens: pMaxTraceTokens
                ),
                runs: collected,
                atlas: atlas,
                failureStage: failureStage,
                failureMessage: failureMessage
            )
        }.value

        // Phase 3: UI state updates (on MainActor — implicit via @MainActor class)
        runs = collected
        self.atlas = atlas
        lastRunDirectory = runDirectory
        selectedRunID = runDirectory.lastPathComponent
        reloadRunHistory()
        selectedExpert = nil
    }
    func importSuite() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, UTType(filenameExtension: "jsonl") ?? .json]
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let suite = try ExpertPromptSuite.loadJSONL(
                    name: url.deletingPathExtension().lastPathComponent,
                    from: url
                )
                suites.append(suite)
                selectedSuiteName = suite.name
                selectedPromptID = suite.prompts.first?.id ?? ""
            } catch {
                lastError = "Could not import \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    func exportSelectedSuite() {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "\(selectedSuite.name).jsonl"
        panel.allowedContentTypes = [UTType(filenameExtension: "jsonl") ?? .json]
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try selectedSuite.writeJSONL(to: url)
            } catch {
                lastError = "Could not export suite: \(error.localizedDescription)"
            }
        }
    }

    func exportPrunePlan() {
        guard let atlas else {
            lastError = "Run a trace before exporting a smart prune plan."
            return
        }
        guard capability.expectedExperts != nil else {
            lastError = "This bundle does not expose the source expert count, so Studio cannot build a complete keep map."
            return
        }
        guard reviewedPruneAuthorityIssues.isEmpty else {
            lastError = reviewedPruneExportBlockReason()
            return
        }
        guard reviewedPruneAtlasIssues.isEmpty else {
            lastError = reviewedPruneExportBlockReason()
            return
        }
        guard let comparisonSummary = latestComparisonSummary() else {
            lastError = "Run a masked A/B comparison before exporting a reviewed prune plan."
            return
        }
        guard reviewedPruneCoverageIssues.isEmpty
            && reviewedPruneSemanticEvidenceIssues.isEmpty
            && reviewedPruneComparisonIssues.isEmpty
            && !hasBlockingPlanIssue else {
            lastError = reviewedPruneExportBlockReason()
            return
        }

        do {
            let plan = try ExpertPrunePlanBuilder.build(
                from: atlas,
                keepExpertsPerLayer: pruneKeepExperts,
                sourceNumExpertsByLayer: sourceExpertsByLayerForAtlas(),
                trainedTopKByLayer: trainedTopKByLayerForAtlas(),
                forceDropByLayer: dropCandidates,
                lockedKeepByLayer: lockedKeeps,
                comparisonSummary: comparisonSummary,
                evalIndex: currentEvalIndexSummary(comparisonSummary: comparisonSummary),
                sourceModelPath: planSourceModelPath,
                reviewBundlePath: artifactReviewBundlePath,
                runID: lastRunDirectory?.lastPathComponent,
                atlasID: "atlas.json",
                evalArtifactPath: lastEvalDirectory?.path
            )
            let panel = NSSavePanel()
            panel.canCreateDirectories = true
            panel.allowedContentTypes = [.json]
            panel.nameFieldStringValue = prunePlanFileName()
            panel.directoryURL = lastRunDirectory
            if panel.runModal() == .OK, let url = panel.url {
                _ = try write(plan: plan, to: url)
                let sidecarURL = try writeCanonicalPrunePlan(plan)
                lastError = "Smart prune plan exported: \(url.path). Run sidecar refreshed: \(sidecarURL.lastPathComponent)."
            }
        } catch {
            lastError = "Could not export prune plan: \(error.localizedDescription)"
        }
    }

    func writePrunePlan() throws -> URL {
        guard let atlas else {
            throw ExpertPrunePlanExportError.missingAtlas
        }
        guard capability.expectedExperts != nil else {
            throw ExpertPrunePlanExportError.missingExpertCount
        }
        guard reviewedPruneAuthorityIssues.isEmpty else {
            throw ExpertPrunePlanExportError.blocked(reviewedPruneExportBlockReason())
        }
        guard reviewedPruneAtlasIssues.isEmpty else {
            throw ExpertPrunePlanExportError.blocked(reviewedPruneExportBlockReason())
        }
        guard let comparisonSummary = latestComparisonSummary() else {
            throw ExpertPrunePlanExportError.missingComparison
        }
        guard reviewedPruneCoverageIssues.isEmpty
            && reviewedPruneSemanticEvidenceIssues.isEmpty
            && reviewedPruneComparisonIssues.isEmpty
            && !hasBlockingPlanIssue else {
            throw ExpertPrunePlanExportError.blocked(reviewedPruneExportBlockReason())
        }
        guard let sourceModelPath = planSourceModelPath else {
            throw ExpertPrunePlanExportError.missingSourceModel
        }
        let plan = try ExpertPrunePlanBuilder.build(
            from: atlas,
            keepExpertsPerLayer: pruneKeepExperts,
            sourceNumExpertsByLayer: sourceExpertsByLayerForAtlas(),
            trainedTopKByLayer: trainedTopKByLayerForAtlas(),
            forceDropByLayer: dropCandidates,
            lockedKeepByLayer: lockedKeeps,
            comparisonSummary: comparisonSummary,
            evalIndex: currentEvalIndexSummary(comparisonSummary: comparisonSummary),
            sourceModelPath: sourceModelPath,
            reviewBundlePath: artifactReviewBundlePath,
            runID: lastRunDirectory?.lastPathComponent,
            atlasID: "atlas.json",
            evalArtifactPath: lastEvalDirectory?.path
        )
        return try writeCanonicalPrunePlan(plan)
    }

    private func reviewedPruneExportBlockReason() -> String {
        let planIssues = planValidationIssues
            .filter { $0.severity == .error }
            .map(\.message)
        let issues = reviewedPruneAuthorityIssues
            + reviewedPruneAtlasIssues
            + reviewedPruneCoverageIssues
            + reviewedPruneSemanticEvidenceIssues
            + reviewedPruneComparisonIssues
            + planIssues
        if issues.isEmpty {
            return "Reviewed prune plan is not ready yet."
        }
        return issues.joined(separator: " ")
    }

    private func writeCanonicalPrunePlan(_ plan: ExpertPrunePlan) throws -> URL {
        try write(plan: plan, to: canonicalPrunePlanURL())
    }

    private func currentEvalIndexSummary(comparisonSummary: ExpertComparisonSummary) -> ExpertEvalIndexSummary? {
        guard !comparisonPreviewRows.isEmpty else { return nil }
        let risky = comparisonRiskRows
        let highRiskDomains = Self.highRiskDomains(from: risky)
        let baselineCounts = comparisonPreviewRows.compactMap(\.baselineTokenCount)
        let maskedCounts = comparisonPreviewRows.compactMap(\.maskedTokenCount)
        let semanticCoverage = Self.semanticCoverage(from: comparisonPreviewRows)
        let runtimeRecord = comparisonPreviewRows.first {
            $0.runtimeMode != nil
                || $0.runtimeBackend != nil
                || $0.runtimeDevice != nil
                || $0.jangToolsVersion != nil
                || $0.mlxVersion != nil
                || $0.mlxLMVersion != nil
                || $0.mlxVLMVersion != nil
                || $0.sourceModelPath != nil
                || $0.hookedMOELayers != nil
                || $0.expectedMOELayers != nil
                || $0.hookCoverageComplete != nil
                || $0.maskApplied != nil
                || $0.disabledExpertCount != nil
                || $0.topKOverride != nil
        }
        return ExpertEvalIndexSummary(
            promptCount: comparisonPreviewRows.count,
            promptIDs: comparisonPreviewRows.map(\.promptID),
            riskyPromptIDs: risky.map(\.promptID),
            highRiskDomains: highRiskDomains,
            passRateBaseline: comparisonSummary.passRateBaseline,
            passRateMasked: comparisonSummary.passRateMasked,
            meanTextDelta: comparisonSummary.meanTextDelta,
            minBaselineTokens: baselineCounts.min(),
            minMaskedTokens: maskedCounts.min(),
            meanBaselineTokens: baselineCounts.count == comparisonPreviewRows.count
                ? Self.mean(baselineCounts.map(Double.init))
                : nil,
            meanMaskedTokens: maskedCounts.count == comparisonPreviewRows.count
                ? Self.mean(maskedCounts.map(Double.init))
                : nil,
            baselineRouteRecordCount: Self.sumOptionalRouteCounts(comparisonPreviewRows.map(\.baselineRouteRecordCount)),
            maskedRouteRecordCount: Self.sumOptionalRouteCounts(comparisonPreviewRows.map(\.maskedRouteRecordCount)),
            baselineLayerStatsPromptCount: Self.layerStatsPromptCount(comparisonPreviewRows.map(\.baselineLayerStats)),
            maskedLayerStatsPromptCount: Self.layerStatsPromptCount(comparisonPreviewRows.map(\.maskedLayerStats)),
            generationSettingsChecked: Self.generationSettingsChecked(comparisonPreviewRows),
            suiteJSONL: lastRunDirectory?.appendingPathComponent("suite.jsonl").path,
            suiteSHA256: Self.fileSHA256(lastRunDirectory?.appendingPathComponent("suite.jsonl")),
            evalJSONL: lastEvalDirectory?.appendingPathComponent("eval.jsonl").path,
            evalTraceJSONL: comparisonRouteRecordSummary == nil
                ? nil
                : lastEvalDirectory?.appendingPathComponent("eval_trace.jsonl").path,
            comparisonSummary: lastEvalDirectory?.appendingPathComponent("comparison_summary.json").path,
            mask: lastEvalDirectory?.appendingPathComponent("mask.json").path,
            maskJSON: lastEvalDirectory?.appendingPathComponent("mask.json").path,
            semanticCoverage: semanticCoverage,
            missingSemanticCoverage: Self.missingSemanticCoverage(for: semanticCoverage),
            runtimeMode: runtimeRecord?.runtimeMode,
            runtimeBackend: runtimeRecord?.runtimeBackend,
            runtimeDevice: runtimeRecord?.runtimeDevice,
            runtimeMetalEnabled: runtimeRecord?.runtimeMetalEnabled,
            jangToolsVersion: runtimeRecord?.jangToolsVersion,
            mlxVersion: runtimeRecord?.mlxVersion,
            mlxLMVersion: runtimeRecord?.mlxLMVersion,
            mlxVLMVersion: runtimeRecord?.mlxVLMVersion,
            sourceModelPath: runtimeRecord?.sourceModelPath,
            hookedMOELayers: runtimeRecord?.hookedMOELayers,
            expectedMOELayers: runtimeRecord?.expectedMOELayers,
            hookCoverageComplete: runtimeRecord?.hookCoverageComplete,
            maskApplied: runtimeRecord?.maskApplied,
            disabledExpertCount: runtimeRecord?.disabledExpertCount,
            topKOverride: runtimeRecord?.topKOverride,
            regressionSeverity: comparisonSummary.regressionSeverity ?? Self.regressionSeverity(from: comparisonPreviewRows)
        )
    }

    private func write(plan: ExpertPrunePlan, to url: URL) throws -> URL {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(plan).write(to: url)
        return url
    }

    private func canonicalPrunePlanURL() throws -> URL {
        let directory: URL
        if let lastRunDirectory {
            directory = lastRunDirectory
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? FileManager.default.temporaryDirectory
            directory = appSupport
                .appendingPathComponent("JANGStudio", isDirectory: true)
                .appendingPathComponent("ExpertLab", isDirectory: true)
                .appendingPathComponent("Plans", isDirectory: true)
        }
        return directory.appendingPathComponent("prune_plan.json")
    }

    private func prunePlanFileName() -> String {
        let base = sourceModelPath?.lastPathComponent ?? modelPath.lastPathComponent
        return "\(base)-expert-prune-plan-\(pruneKeepExperts)e.json"
    }

    private func loadModel() async throws -> JANGKit.Model {
        guard capability.runtimeMode != .bf16VMLX else {
            throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runs must use the Expert Lab vMLX runner")
        }
        if let model {
            runtimeInfoSummary = Self.runtimeSummary(from: model.runtimeInfo)
            return model
        }
        let loaded = try await JANGKit.Model.load(at: modelPath)
        model = loaded
        runtimeInfoSummary = Self.runtimeSummary(from: loaded.runtimeInfo)
        return loaded
    }

    private func runVMLX(
        suite: ExpertPromptSuite,
        mask: JANGKit.ExpertMask? = nil,
        emitTokenTrace: Bool
    ) async throws -> [ExpertPromptRun] {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("JANGStudio-ExpertLab-vmlx", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let suiteURL = root.appendingPathComponent("suite.jsonl")
        try suite.writeJSONL(to: suiteURL)

        var args = [
            "-m", "jang_tools",
            "--quiet-text",
            "expert-lab-vmlx",
            modelPath.path,
            "--suite", suiteURL.path,
            "--output", root.path,
            "--max-tokens", "\(maxTokens)",
            "--max-trace-tokens", "\(maxTraceTokens)"
        ]
        if emitTokenTrace {
            args.append("--emit-token-trace")
        }
        if let mask {
            let maskURL = root.appendingPathComponent("mask.json")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(mask).write(to: maskURL)
            args.append(contentsOf: ["--mask", maskURL.path])
            if let topK = mask.topKOverride, topK > 0 {
                args.append(contentsOf: ["--top-k-override", "\(topK)"])
            }
        }

        let data = try await PythonCLIInvoker.invoke(args: args) { code, stderr in
            ExpertLabVMLXRunnerError.failed(code, stderr)
        }
        let summary = try Self.decodeVMLXSummary(from: data)
        let generationsURL = URL(fileURLWithPath: summary.generationsJSONL)
        let records = Self.jsonlRecords(ExpertLabVMLXGenerationRecord.self, from: generationsURL)
        guard records.count == suite.prompts.count else {
            throw ExpertLabVMLXRunnerError.malformed(
                "BF16/vMLX runner returned \(records.count) generations for \(suite.prompts.count) prompts"
            )
        }
        if let issue = ExpertLabPromptIdentityValidator.issue(
            expected: suite.prompts,
            actual: records.map(\.prompt)
        ) {
            throw ExpertLabVMLXRunnerError.malformed(issue)
        }
        if let issue = Self.vmlxRuntimeEvidenceIssue(
            records: records,
            capability: capability,
            expectedSourcePath: modelPath.path,
            maskRequired: mask != nil
        ) {
            throw ExpertLabVMLXRunnerError.malformed(issue)
        }
        if let issue = Self.vmlxTraceEvidenceIssue(
            records: records,
            emitTokenTrace: emitTokenTrace
        ) {
            throw ExpertLabVMLXRunnerError.malformed(issue)
        }
        if let firstRuntime = records.compactMap(\.result.runtimeInfo).first {
            runtimeInfoSummary = Self.runtimeSummary(from: firstRuntime)
        }
        return try records.map { record in
            ExpertPromptRun(
                prompt: record.prompt,
                result: try Self.jangRunResult(from: record.result)
            )
        }
    }

    private nonisolated static func vmlxRuntimeEvidenceIssue(
        records: [ExpertLabVMLXGenerationRecord],
        capability: ExpertLabCapability,
        expectedSourcePath: String,
        maskRequired: Bool
    ) -> String? {
        for record in records {
            let promptID = record.prompt.id.trimmingCharacters(in: .whitespacesAndNewlines)
            let label = promptID.isEmpty ? "unknown prompt" : promptID
            guard let runtime = record.result.runtimeInfo else {
                return "BF16/vMLX runner generation \(label) is missing runtime metadata"
            }
            if let issue = ExpertLabVMLXRuntimeEvidenceValidator.issue(
                promptID: label,
                runtimeMode: runtime.runtimeMode,
                runtimeBackend: runtime.backend,
                runtimeMetalEnabled: runtime.runtimeMetalEnabled,
                deviceName: runtime.deviceName,
                jangToolsVersion: runtime.jangToolsVersion,
                mlxVersion: runtime.mlxVersion,
                mlxLMVersion: runtime.mlxLMVersion,
                sourceModelPath: runtime.sourceModelPath,
                hookedMOELayers: runtime.hookedMOELayers,
                expectedMOELayers: runtime.expectedMOELayers,
                hookCoverageComplete: runtime.hookCoverageComplete,
                maskRequired: maskRequired,
                maskApplied: runtime.maskApplied,
                disabledExpertCount: runtime.disabledExpertCount,
                topKOverride: runtime.topKOverride,
                expectedLayers: capability.expectedLayers,
                expectedSourcePath: expectedSourcePath
            ) {
                return issue
            }
        }
        return nil
    }

    private nonisolated static func vmlxTraceEvidenceIssue(
        records: [ExpertLabVMLXGenerationRecord],
        emitTokenTrace: Bool
    ) -> String? {
        for record in records {
            let traces = record.result.tokenTrace
            let malformed = traces?.contains { trace in
                trace.selectedExperts.isEmpty || trace.scores.isEmpty || trace.effectiveTopK <= 0
            } ?? false
            if let issue = ExpertLabVMLXTraceEvidenceValidator.issue(
                promptID: record.prompt.id,
                emitTokenTrace: emitTokenTrace,
                tokenTraceCount: traces?.count,
                expectedRouteRecordCount: record.result.layerStats.reduce(0) { $0 + $1.tokenCount },
                hasInvalidRouteRecord: malformed
            ) {
                return issue
            }
        }
        return nil
    }

    private nonisolated static func decodeVMLXSummary(from data: Data) throws -> ExpertLabVMLXRunSummary {
        let text = String(data: data, encoding: .utf8) ?? ""
        guard let line = text.split(whereSeparator: \.isNewline).last else {
            throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runner did not return JSON")
        }
        do {
            return try JSONDecoder().decode(ExpertLabVMLXRunSummary.self, from: Data(line.utf8))
        } catch {
            throw ExpertLabVMLXRunnerError.malformed("BF16/vMLX runner summary is unreadable: \(error.localizedDescription)")
        }
    }

    private nonisolated static func jangRunResult(from result: ExpertLabVMLXRunResult) throws -> JANGKit.ExpertRunResult {
        let runtimeInfo = result.runtimeInfo.map {
            JANGKit.ModelRuntimeInfo(
                backend: $0.backend,
                runtimeMode: $0.runtimeMode,
                deviceName: $0.deviceName,
                metalEnabled: $0.runtimeMetalEnabled,
                jangToolsVersion: $0.jangToolsVersion,
                mlxVersion: $0.mlxVersion,
                mlxLMVersion: $0.mlxLMVersion,
                mlxVLMVersion: $0.mlxVLMVersion,
                sourceModelPath: $0.sourceModelPath,
                hookedMOELayers: $0.hookedMOELayers,
                expectedMOELayers: $0.expectedMOELayers,
                hookCoverageComplete: $0.hookCoverageComplete,
                maskApplied: $0.maskApplied,
                disabledExpertCount: $0.disabledExpertCount,
                topKOverride: $0.topKOverride,
                notes: $0.notes ?? []
            )
        }
        return JANGKit.ExpertRunResult(
            text: result.text,
            tokens: result.tokens,
            elapsedSeconds: result.elapsedSeconds,
            tokensPerSecond: result.tokensPerSecond,
            finishReason: finishReason(from: result.finishReason),
            layerStats: result.layerStats.map { stats in
                JANGKit.ExpertLayerStats(
                    layer: stats.layer,
                    tokenCount: stats.tokenCount,
                    hitCounts: intKeyed(stats.hitCounts),
                    probabilityMass: intKeyed(stats.probabilityMass)
                )
            },
            tokenTrace: result.tokenTrace?.map {
                JANGKit.ExpertRouteRecord(
                    tokenIndex: $0.tokenIndex,
                    layer: $0.layer,
                    selectedExperts: $0.selectedExperts,
                    scores: $0.scores,
                    disabledExperts: $0.disabledExperts,
                    effectiveTopK: $0.effectiveTopK,
                    entropy: $0.entropy
                )
            },
            runtimeInfo: runtimeInfo
        )
    }

    private nonisolated static func finishReason(from raw: String) -> JANGKit.GenerationResult.FinishReason {
        switch raw {
        case "stop": return .stop
        case "cancelled": return .cancelled
        case "error": return .error
        default: return .maxTokens
        }
    }

    private nonisolated static func intKeyed<T>(_ map: [String: T]) -> [Int: T] {
        Dictionary(uniqueKeysWithValues: map.compactMap { key, value in
            guard let intKey = Int(key) else { return nil }
            return (intKey, value)
        })
    }

    private nonisolated static func runtimeSummary(from info: ExpertLabVMLXRuntimeInfo) -> String {
        let base = runtimeSummary(
            mode: info.runtimeMode,
            backend: info.backend,
            device: info.deviceName,
            metalEnabled: info.runtimeMetalEnabled,
            jangToolsVersion: info.jangToolsVersion,
            mlxVersion: info.mlxVersion,
            mlxLMVersion: info.mlxLMVersion,
            sourceModelPath: info.sourceModelPath,
            hookedMOELayers: info.hookedMOELayers,
            expectedMOELayers: info.expectedMOELayers,
            hookCoverageComplete: info.hookCoverageComplete
        )
        var parts: [String] = base.isEmpty ? [] : [base]
        if info.maskApplied == true {
            let disabledCount = info.disabledExpertCount ?? 0
            parts.append("mask \(disabledCount) experts")
        }
        return parts.joined(separator: " · ")
    }

    private nonisolated static func runtimeSummary(from info: JANGKit.ModelRuntimeInfo) -> String {
        let base = runtimeSummary(
            mode: info.runtimeMode,
            backend: info.backend,
            device: info.deviceName,
            metalEnabled: info.metalEnabled,
            jangToolsVersion: info.jangToolsVersion,
            mlxVersion: info.mlxVersion,
            mlxLMVersion: info.mlxLMVersion,
            sourceModelPath: info.sourceModelPath,
            hookedMOELayers: info.hookedMOELayers,
            expectedMOELayers: info.expectedMOELayers,
            hookCoverageComplete: info.hookCoverageComplete
        )
        var parts: [String] = base.isEmpty ? [] : [base]
        if info.maskApplied == true {
            let disabledCount = info.disabledExpertCount ?? 0
            parts.append("mask \(disabledCount) experts")
        }
        return parts.joined(separator: " · ")
    }

    private nonisolated static func runtimeSummary(
        mode: String?,
        backend: String?,
        device: String?,
        metalEnabled: Bool?,
        jangToolsVersion: String? = nil,
        mlxVersion: String? = nil,
        mlxLMVersion: String? = nil,
        sourceModelPath: String? = nil,
        hookedMOELayers: Int? = nil,
        expectedMOELayers: Int? = nil,
        hookCoverageComplete: Bool? = nil
    ) -> String {
        let resolvedMode = mode?.isEmpty == false ? mode! : backend
        guard let resolvedMode, !resolvedMode.isEmpty else { return "" }
        var parts = [resolvedMode]
        if let device, !device.isEmpty {
            parts.append(device)
        }
        if let metalEnabled {
            parts.append(metalEnabled ? "Metal" : "CPU")
        }
        let versionParts = [
            ("jang", jangToolsVersion),
            ("mlx", mlxVersion),
            ("mlx-lm", mlxLMVersion)
        ].compactMap { label, value -> String? in
            guard let value, !value.isEmpty else { return nil }
            return "\(label) \(value)"
        }
        if !versionParts.isEmpty {
            parts.append(versionParts.joined(separator: ", "))
        }
        parts.append(contentsOf: runtimeEvidenceParts(
            sourceModelPath: sourceModelPath,
            hookedMOELayers: hookedMOELayers,
            expectedMOELayers: expectedMOELayers,
            hookCoverageComplete: hookCoverageComplete
        ))
        return parts.joined(separator: " · ")
    }

    private nonisolated static func runtimeEvidenceParts(
        sourceModelPath: String?,
        hookedMOELayers: Int?,
        expectedMOELayers: Int?,
        hookCoverageComplete: Bool?
    ) -> [String] {
        var parts: [String] = []
        if let sourceModelPath, !sourceModelPath.isEmpty {
            parts.append("source \(URL(fileURLWithPath: sourceModelPath).lastPathComponent)")
        }
        if let hookedMOELayers {
            if let expectedMOELayers {
                let coverage = hookCoverageComplete == false ? "incomplete" : "complete"
                parts.append("\(hookedMOELayers)/\(expectedMOELayers) MoE layers \(coverage)")
            } else {
                parts.append("\(hookedMOELayers) MoE layers")
            }
        }
        return parts
    }

    private nonisolated static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }

    private nonisolated static func fileSHA256(_ url: URL?) -> String? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private func expectedExpertsByLayer(from runs: [ExpertPromptRun]) -> [Int: Int] {
        Self._expectedExpertsByLayer(from: runs, expectedExperts: capability.expectedExperts, expectedLayers: capability.expectedLayers)
    }

    /// Static variant that accepts capability values directly, safe to call from non-isolated contexts (e.g. Task.detached).
    private nonisolated static func _expectedExpertsByLayer(
        from runs: [ExpertPromptRun],
        expectedExperts: Int?,
        expectedLayers: Int?
    ) -> [Int: Int] {
        var observedMax: [Int: Int] = [:]
        var observedLayers = Set<Int>()
        for run in runs {
            for record in run.result.tokenTrace ?? [] {
                observedLayers.insert(record.layer)
                let experts = record.selectedExperts + record.disabledExperts
                for expert in experts {
                    observedMax[record.layer] = max(observedMax[record.layer] ?? expert, expert)
                }
            }
            for stat in run.result.layerStats {
                observedLayers.insert(stat.layer)
                for expert in stat.hitCounts.keys {
                    observedMax[stat.layer] = max(observedMax[stat.layer] ?? expert, expert)
                }
            }
        }

        let configured = Self._configuredExpertsByLayer(
            observedLayers: observedLayers,
            expectedExperts: expectedExperts,
            expectedLayers: expectedLayers
        )
        if !configured.isEmpty {
            return configured
        }

        var expected: [Int: Int] = [:]
        for layer in observedMax.keys {
            expected[layer] = expectedExperts ?? ((observedMax[layer] ?? -1) + 1)
        }
        return expected
    }

    private func configuredExpertsByLayer(observedLayers: Set<Int>) -> [Int: Int] {
        Self._configuredExpertsByLayer(
            observedLayers: observedLayers,
            expectedExperts: capability.expectedExperts,
            expectedLayers: capability.expectedLayers
        )
    }

    /// Static variant that accepts capability values directly.
    private nonisolated static func _configuredExpertsByLayer(
        observedLayers: Set<Int>,
        expectedExperts: Int?,
        expectedLayers: Int?
    ) -> [Int: Int] {
        guard let expectedExperts else { return [:] }
        if let expectedLayers, expectedLayers > 0 {
            return Dictionary(uniqueKeysWithValues: (0..<expectedLayers).map { ($0, expectedExperts) })
        }
        return Dictionary(uniqueKeysWithValues: observedLayers.map { ($0, expectedExperts) })
    }

    private func sourceExpertsByLayerForAtlas() -> [Int: Int] {
        guard let atlas else { return [:] }
        let saved = Self.intLayerMap(atlas.sourceNumExpertsByLayer)
        if !saved.isEmpty {
            return saved
        }
        if let expectedExperts = capability.expectedExperts {
            let configured = configuredExpertsByLayer(observedLayers: Set(atlas.experts.map(\.layer)))
            if !configured.isEmpty {
                return configured
            }
            return Dictionary(uniqueKeysWithValues: Set(atlas.experts.map(\.layer)).map {
                ($0, expectedExperts)
            })
        }
        return expectedExpertsByLayer(from: runs)
    }

    private func trainedTopKByLayerForAtlas() -> [Int: Int] {
        let trained = capability.trainedTopK ?? 1
        guard let atlas else { return [:] }
        let layers: Set<Int>
        if let expectedLayers = capability.expectedLayers, expectedLayers > 0 {
            layers = Set(0..<expectedLayers)
        } else {
            layers = Set(atlas.experts.map(\.layer))
        }
        return Dictionary(uniqueKeysWithValues: layers.map { ($0, trained) })
    }

    private func completedAtlas(_ atlas: ExpertAtlas) -> ExpertAtlas {
        Self._completedAtlas(atlas, expectedExperts: capability.expectedExperts, expectedLayers: capability.expectedLayers)
    }

    /// Static variant that accepts capability values directly, safe to call from non-isolated contexts (e.g. Task.detached).
    private nonisolated static func _completedAtlas(
        _ atlas: ExpertAtlas,
        expectedExperts: Int?,
        expectedLayers: Int?
    ) -> ExpertAtlas {
        let saved = intLayerMap(atlas.sourceNumExpertsByLayer)
        let expected = saved.isEmpty
            ? _configuredExpertsByLayer(
                observedLayers: Set(atlas.experts.map(\.layer)),
                expectedExperts: expectedExperts,
                expectedLayers: expectedLayers
            )
            : saved
        guard !expected.isEmpty else { return atlas }

        var byCoordinate = Dictionary(
            uniqueKeysWithValues: atlas.experts.map { (ExpertCoordinate(layer: $0.layer, expert: $0.expert), $0) }
        )
        for (layer, sourceCount) in expected where sourceCount > 0 {
            for expert in 0..<sourceCount {
                let coordinate = ExpertCoordinate(layer: layer, expert: expert)
                if byCoordinate[coordinate] == nil {
                    byCoordinate[coordinate] = ExpertAtlasEntry(
                        layer: layer,
                        expert: expert,
                        hits: 0,
                        probabilityMass: 0,
                        tokenCount: 0,
                        domains: [:],
                        label: "dead",
                        isDead: true,
                        isHot: false
                    )
                }
            }
        }
        return ExpertAtlas(
            generatedAt: atlas.generatedAt,
            promptCount: atlas.promptCount,
            experts: byCoordinate.values.sorted {
                if $0.layer != $1.layer { return $0.layer < $1.layer }
                return $0.expert < $1.expert
            },
            sourceNumExpertsByLayer: Self.stringLayerMap(expected)
        )
    }


    private nonisolated static func intLayerMap(_ values: [String: Int]?) -> [Int: Int] {
        guard let values else { return [:] }
        var map: [Int: Int] = [:]
        for (layer, count) in values {
            guard let layerID = Int(layer), count > 0 else { continue }
            map[layerID] = count
        }
        return map
    }

    private nonisolated static func stringLayerMap(_ values: [Int: Int]) -> [String: Int]? {
        let filtered = values.filter { $0.value > 0 }
        guard !filtered.isEmpty else { return nil }
        return Dictionary(uniqueKeysWithValues: filtered.map { (String($0.key), $0.value) })
    }

    private nonisolated static func uniformGridShape(from sourceGrid: [Int: Int]) -> (layers: Int, experts: Int, expectedCells: Int)? {
        guard !sourceGrid.isEmpty else { return nil }
        let expectedCells = sourceGrid.values.reduce(0, +)
        let expertCounts = Set(sourceGrid.values)
        guard expertCounts.count == 1, let experts = expertCounts.first else { return nil }
        return (layers: sourceGrid.count, experts: experts, expectedCells: expectedCells)
    }

    private func artifactRoot() -> URL {
        let fm = FileManager.default
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fm.temporaryDirectory
        return appSupport
            .appendingPathComponent("JANGStudio", isDirectory: true)
            .appendingPathComponent("ExpertLab", isDirectory: true)
            .appendingPathComponent("runs", isDirectory: true)
    }


    private func persistComparison(
        mask: JANGKit.ExpertMask,
        records: [ExpertComparisonPromptRecord]
    ) throws -> URL {
        try persistComparison(mask: mask, records: records, directory: nil)
    }

    private func persistComparison(
        mask: JANGKit.ExpertMask,
        records: [ExpertComparisonPromptRecord],
        directory existingDirectory: URL?
    ) throws -> URL {
        precondition(!records.isEmpty, "cannot persist an empty Expert Lab comparison")
        let fm = FileManager.default
        let dir: URL
        if let existingDirectory {
            dir = existingDirectory
        } else {
            let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? fm.temporaryDirectory
            let runDir = lastRunDirectory ?? appSupport
                .appendingPathComponent("JANGStudio", isDirectory: true)
                .appendingPathComponent("ExpertLab", isDirectory: true)
                .appendingPathComponent("runs", isDirectory: true)
                .appendingPathComponent("ad-hoc", isDirectory: true)
            let stamp = ISO8601DateFormatter()
                .string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            dir = runDir
                .appendingPathComponent("evals", isDirectory: true)
                .appendingPathComponent(stamp, isDirectory: true)
        }
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(mask).write(to: dir.appendingPathComponent("mask.json"))

        let highRiskDomains = Self.highRiskDomains(from: records)
        let runtimeInfo = Self.runtimeInfo(from: records)
        let regressionSeverity = Self.regressionSeverity(for: records)
        let baselineQualified = Self.baselineQualifiedRecords(records)
        let baselineQualifiedCoverage = Self.baselineQualifiedSemanticCoverage(from: records)
        let missingBaselineQualifiedCoverage = Self.missingBaselineQualifiedSemanticCoverage(
            for: baselineQualifiedCoverage
        )
        let degradedPromptIDs = Self.degradedPromptIDs(from: records)
        let safeDropCandidates = highRiskDomains.isEmpty
            && degradedPromptIDs.isEmpty
            && !baselineQualified.isEmpty
            && missingBaselineQualifiedCoverage.isEmpty
            ? comparisonCandidateCoordinates()
            : []
        let summary = ExpertComparisonSummary(
            baselineRunID: lastRunDirectory?.lastPathComponent ?? "ad-hoc",
            maskID: dir.lastPathComponent,
            promptCount: records.count,
            passRateBaseline: Self.passRate(records.map { $0.evaluation.baselinePassed }),
            passRateMasked: Self.passRate(records.map { $0.evaluation.maskedPassed }),
            baselineQualifiedPromptCount: baselineQualified.count,
            baselineQualifiedMaskedPassRate: Self.passRate(baselineQualified.map { $0.evaluation.maskedPassed }),
            validatorAvailablePromptCount: records.filter {
                $0.evaluation.baselinePassed != nil && $0.evaluation.maskedPassed != nil
            }.count,
            classificationCounts: Self.classificationCounts(from: records),
            baselineQualifiedPromptIDs: baselineQualified.map(\.prompt.id),
            baselineInvalidPromptIDs: Self.baselineInvalidPromptIDs(from: records),
            inconclusivePromptIDs: Self.inconclusivePromptIDs(from: records),
            preservedPromptIDs: Self.preservedPromptIDs(from: records),
            degradedPromptIDs: degradedPromptIDs,
            baselineQualifiedSemanticCoverage: baselineQualifiedCoverage,
            missingBaselineQualifiedSemanticCoverage: missingBaselineQualifiedCoverage,
            meanTextDelta: Self.mean(records.map(\.textDelta)),
            meanLatencyDeltaPct: Self.mean(records.map(\.latencyDeltaPct)),
            regressionSeverity: regressionSeverity,
            highRiskDomains: highRiskDomains,
            safeDropCandidates: safeDropCandidates
        )
        try encoder.encode(summary).write(to: dir.appendingPathComponent("comparison_summary.json"))

        let storedRecords = Self.storedEvalRecords(from: records)
        let semanticCoverage = Self.semanticCoverage(from: storedRecords)
        try Self.writeEvalTrace(records, to: dir.appendingPathComponent("eval_trace.jsonl"))
        try encoder.encode(
            StoredEvalIndex(
                generatedAt: Date(),
                runID: lastRunDirectory?.lastPathComponent ?? "ad-hoc",
                maskID: dir.lastPathComponent,
                promptCount: storedRecords.count,
                riskyPromptIDs: storedRecords.filter(\.isRisky).map(\.promptID),
                promptIDs: storedRecords.map(\.promptID),
                highRiskDomains: highRiskDomains,
                passRateBaseline: summary.passRateBaseline,
                passRateMasked: summary.passRateMasked,
                validatorSchema: "jang-expert-lab-validator-v1",
                validatorAvailablePromptCount: storedRecords.filter { $0.validatorAvailable == true }.count,
                promptClassificationCounts: Self.classificationCounts(from: storedRecords),
                baselineQualifiedPromptCount: Self.baselineQualifiedRecords(storedRecords).count,
                baselineQualifiedPromptIDs: Self.baselineQualifiedRecords(storedRecords).map(\.promptID),
                baselineInvalidPromptIDs: storedRecords.filter { ($0.promptClassification ?? Self.legacyPromptClassification(for: $0)) == "baseline_invalid" }.map(\.promptID),
                inconclusivePromptIDs: storedRecords.filter { ($0.promptClassification ?? Self.legacyPromptClassification(for: $0)) == "inconclusive" }.map(\.promptID),
                preservedPromptIDs: storedRecords.filter { ($0.promptClassification ?? Self.legacyPromptClassification(for: $0)) == "preserved" }.map(\.promptID),
                degradedPromptIDs: Self.degradedPromptIDs(from: storedRecords),
                baselineQualifiedMaskedPassRate: Self.passRate(Self.baselineQualifiedRecords(storedRecords).map(\.maskedPassed)),
                baselineQualifiedSemanticCoverage: Self.baselineQualifiedSemanticCoverage(from: storedRecords),
                missingBaselineQualifiedSemanticCoverage: Self.missingBaselineQualifiedSemanticCoverage(
                    for: Self.baselineQualifiedSemanticCoverage(from: storedRecords)
                ),
                meanTextDelta: summary.meanTextDelta,
                regressionSeverity: regressionSeverity,
                minBaselineTokens: storedRecords.compactMap(\.baselineTokenCount).min(),
                minMaskedTokens: storedRecords.compactMap(\.maskedTokenCount).min(),
                meanBaselineTokens: Self.meanOptionalTokenCounts(storedRecords.map(\.baselineTokenCount)),
                meanMaskedTokens: Self.meanOptionalTokenCounts(storedRecords.map(\.maskedTokenCount)),
                baselineRouteRecordCount: Self.sumOptionalRouteCounts(storedRecords.map(\.baselineRouteRecordCount)),
                maskedRouteRecordCount: Self.sumOptionalRouteCounts(storedRecords.map(\.maskedRouteRecordCount)),
                baselineLayerStatsPromptCount: Self.layerStatsPromptCount(storedRecords.map(\.baselineLayerStats)),
                maskedLayerStatsPromptCount: Self.layerStatsPromptCount(storedRecords.map(\.maskedLayerStats)),
                generationSettingsChecked: Self.generationSettingsChecked(storedRecords),
                suiteSHA256: Self.fileSHA256(lastRunDirectory?.appendingPathComponent("suite.jsonl")),
                evalJSONL: "eval.jsonl",
                evalTraceJSONL: "eval_trace.jsonl",
                comparisonSummary: "comparison_summary.json",
                mask: "mask.json",
                maskJSON: "mask.json",
                semanticCoverage: semanticCoverage,
                missingSemanticCoverage: Self.missingSemanticCoverage(for: semanticCoverage),
                runtimeMode: runtimeInfo?.runtimeMode,
                runtimeBackend: runtimeInfo?.backend,
                runtimeDevice: runtimeInfo?.deviceName,
                runtimeMetalEnabled: runtimeInfo?.metalEnabled,
                jangToolsVersion: runtimeInfo?.jangToolsVersion,
                mlxVersion: runtimeInfo?.mlxVersion,
                mlxLMVersion: runtimeInfo?.mlxLMVersion,
                mlxVLMVersion: runtimeInfo?.mlxVLMVersion,
                sourceModelPath: runtimeInfo?.sourceModelPath,
                hookedMOELayers: runtimeInfo?.hookedMOELayers,
                expectedMOELayers: runtimeInfo?.expectedMOELayers,
                hookCoverageComplete: runtimeInfo?.hookCoverageComplete,
                maskApplied: runtimeInfo?.maskApplied,
                disabledExpertCount: runtimeInfo?.disabledExpertCount,
                topKOverride: runtimeInfo?.topKOverride
            )
        ).write(to: dir.appendingPathComponent("eval_index.json"))

        encoder.outputFormatting = [.sortedKeys]
        let lines = try storedRecords.map { stored -> String in
            return String(data: try encoder.encode(stored), encoding: .utf8) ?? "{}"
        }
        try lines.joined(separator: "\n").appending("\n").write(
            to: dir.appendingPathComponent("eval.jsonl"),
            atomically: true,
            encoding: .utf8
        )
        return dir
    }

    private nonisolated static func writeEvalTrace(_ records: [ExpertComparisonPromptRecord], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var lines: [String] = []
        for record in records {
            let domain = ExpertDomainTaxonomy.canonicalDomain(for: record.prompt)
            for route in record.baseline.tokenTrace ?? [] {
                let stored = StoredEvalTraceRecord(
                    promptID: record.prompt.id,
                    domain: domain,
                    variant: "baseline",
                    record: route
                )
                lines.append(String(data: try encoder.encode(stored), encoding: .utf8) ?? "{}")
            }
            for route in record.masked.tokenTrace ?? [] {
                let stored = StoredEvalTraceRecord(
                    promptID: record.prompt.id,
                    domain: domain,
                    variant: "masked",
                    record: route
                )
                lines.append(String(data: try encoder.encode(stored), encoding: .utf8) ?? "{}")
            }
        }
        try lines.joined(separator: "\n").appending("\n").write(
            to: url,
            atomically: true,
            encoding: .utf8
        )
    }

    private nonisolated static func writeEvalIndex(
        to url: URL,
        runID: String,
        maskID: String,
        summary: ExpertComparisonSummary,
        records: [StoredEvalRecord],
        suiteURL: URL? = nil
    ) throws {
        let runtimeRecord = records.first {
            $0.runtimeMode != nil
                || $0.runtimeBackend != nil
                || $0.runtimeDevice != nil
                || $0.jangToolsVersion != nil
                || $0.mlxVersion != nil
                || $0.mlxLMVersion != nil
                || $0.mlxVLMVersion != nil
                || $0.sourceModelPath != nil
                || $0.hookedMOELayers != nil
                || $0.expectedMOELayers != nil
                || $0.hookCoverageComplete != nil
                || $0.maskApplied != nil
                || $0.disabledExpertCount != nil
                || $0.topKOverride != nil
        }
        let semanticCoverage = Self.semanticCoverage(from: records)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(
            StoredEvalIndex(
                generatedAt: Date(),
                runID: runID,
                maskID: maskID,
                promptCount: records.count,
                riskyPromptIDs: records.filter(\.isRisky).map(\.promptID),
                promptIDs: records.map(\.promptID),
                highRiskDomains: summary.highRiskDomains,
                passRateBaseline: summary.passRateBaseline,
                passRateMasked: summary.passRateMasked,
                validatorSchema: "jang-expert-lab-validator-v1",
                validatorAvailablePromptCount: records.filter { $0.validatorAvailable == true }.count,
                promptClassificationCounts: classificationCounts(from: records),
                baselineQualifiedPromptCount: baselineQualifiedRecords(records).count,
                baselineQualifiedPromptIDs: baselineQualifiedRecords(records).map(\.promptID),
                baselineInvalidPromptIDs: records.filter { ($0.promptClassification ?? legacyPromptClassification(for: $0)) == "baseline_invalid" }.map(\.promptID),
                inconclusivePromptIDs: records.filter { ($0.promptClassification ?? legacyPromptClassification(for: $0)) == "inconclusive" }.map(\.promptID),
                preservedPromptIDs: records.filter { ($0.promptClassification ?? legacyPromptClassification(for: $0)) == "preserved" }.map(\.promptID),
                degradedPromptIDs: degradedPromptIDs(from: records),
                baselineQualifiedMaskedPassRate: passRate(baselineQualifiedRecords(records).map(\.maskedPassed)),
                baselineQualifiedSemanticCoverage: baselineQualifiedSemanticCoverage(from: records),
                missingBaselineQualifiedSemanticCoverage: missingBaselineQualifiedSemanticCoverage(
                    for: baselineQualifiedSemanticCoverage(from: records)
                ),
                meanTextDelta: summary.meanTextDelta,
                regressionSeverity: summary.regressionSeverity ?? regressionSeverity(from: records),
                minBaselineTokens: records.compactMap(\.baselineTokenCount).min(),
                minMaskedTokens: records.compactMap(\.maskedTokenCount).min(),
                meanBaselineTokens: meanOptionalTokenCounts(records.map(\.baselineTokenCount)),
                meanMaskedTokens: meanOptionalTokenCounts(records.map(\.maskedTokenCount)),
                baselineRouteRecordCount: sumOptionalRouteCounts(records.map(\.baselineRouteRecordCount)),
                maskedRouteRecordCount: sumOptionalRouteCounts(records.map(\.maskedRouteRecordCount)),
                baselineLayerStatsPromptCount: layerStatsPromptCount(records.map(\.baselineLayerStats)),
                maskedLayerStatsPromptCount: layerStatsPromptCount(records.map(\.maskedLayerStats)),
                generationSettingsChecked: generationSettingsChecked(records),
                suiteSHA256: fileSHA256(suiteURL),
                evalJSONL: "eval.jsonl",
                evalTraceJSONL: sumOptionalRouteCounts(records.map(\.baselineRouteRecordCount)) == nil
                    || sumOptionalRouteCounts(records.map(\.maskedRouteRecordCount)) == nil
                    ? nil
                    : "eval_trace.jsonl",
                comparisonSummary: "comparison_summary.json",
                mask: "mask.json",
                maskJSON: "mask.json",
                semanticCoverage: semanticCoverage,
                missingSemanticCoverage: Self.missingSemanticCoverage(for: semanticCoverage),
                runtimeMode: runtimeRecord?.runtimeMode,
                runtimeBackend: runtimeRecord?.runtimeBackend,
                runtimeDevice: runtimeRecord?.runtimeDevice,
                runtimeMetalEnabled: runtimeRecord?.runtimeMetalEnabled,
                jangToolsVersion: runtimeRecord?.jangToolsVersion,
                mlxVersion: runtimeRecord?.mlxVersion,
                mlxLMVersion: runtimeRecord?.mlxLMVersion,
                mlxVLMVersion: runtimeRecord?.mlxVLMVersion,
                sourceModelPath: runtimeRecord?.sourceModelPath,
                hookedMOELayers: runtimeRecord?.hookedMOELayers,
                expectedMOELayers: runtimeRecord?.expectedMOELayers,
                hookCoverageComplete: runtimeRecord?.hookCoverageComplete,
                maskApplied: runtimeRecord?.maskApplied,
                disabledExpertCount: runtimeRecord?.disabledExpertCount,
                topKOverride: runtimeRecord?.topKOverride
            )
        ).write(to: url)
    }

    private nonisolated static func runtimeInfo(from records: [ExpertComparisonPromptRecord]) -> JANGKit.ModelRuntimeInfo? {
        records.compactMap(comparisonRuntimeInfo).first
    }

    private nonisolated static func comparisonRuntimeInfo(_ record: ExpertComparisonPromptRecord) -> JANGKit.ModelRuntimeInfo? {
        if record.masked.runtimeInfo?.maskApplied == true {
            return record.masked.runtimeInfo
        }
        return record.masked.runtimeInfo ?? record.baseline.runtimeInfo
    }

    private nonisolated static func comparisonRecord(
        prompt: ExpertPrompt,
        baseline: JANGKit.ExpertRunResult,
        masked: JANGKit.ExpertRunResult,
        samplingConfig: JANGKit.SamplingConfig? = nil
    ) -> ExpertComparisonPromptRecord {
        let evaluation = ExpertPromptEvaluator.evaluate(
            prompt: prompt,
            baselineText: baseline.text,
            maskedText: masked.text
        )
        let textDelta = ExpertPromptEvaluator.normalizedTextDelta(baseline.text, masked.text)
        return ExpertComparisonPromptRecord(
            prompt: prompt,
            baseline: baseline,
            masked: masked,
            evaluation: evaluation,
            textDelta: textDelta,
            latencyDeltaPct: latencyDeltaPct(baseline: baseline, masked: masked),
            samplingConfig: samplingConfig
        )
    }

    private nonisolated static func comparisonRecords(
        prompts: [ExpertPrompt],
        baselineRuns: [ExpertPromptRun],
        maskedRuns: [ExpertPromptRun],
        samplingConfigs: [String: JANGKit.SamplingConfig] = [:]
    ) throws -> [ExpertComparisonPromptRecord] {
        guard baselineRuns.count == prompts.count, maskedRuns.count == prompts.count else {
            throw ExpertLabVMLXRunnerError.malformed(
                "BF16/vMLX comparison returned \(baselineRuns.count) baseline and \(maskedRuns.count) masked generations for \(prompts.count) prompts"
            )
        }
        return prompts.indices.map { index in
            comparisonRecord(
                prompt: prompts[index],
                baseline: baselineRuns[index].result,
                masked: maskedRuns[index].result,
                samplingConfig: samplingConfigs[prompts[index].id]
            )
        }
    }

    private nonisolated static func storedEvalRecords(from records: [ExpertComparisonPromptRecord]) -> [StoredEvalRecord] {
        records.map { record in
            let runtimeInfo = comparisonRuntimeInfo(record)
            let semanticDomains = comparisonSemanticDomains(for: record.prompt)
            return StoredEvalRecord(
                promptID: record.prompt.id,
                domain: comparisonPrimaryDomain(for: record.prompt),
                semanticDomains: semanticDomains,
                expectedKind: record.prompt.expectedKind,
                expected: record.prompt.expected,
                validatorKind: validatorKind(for: record.prompt),
                validatorAvailable: record.evaluation.baselinePassed != nil && record.evaluation.maskedPassed != nil,
                validatorSource: "suite_expected",
                validatorReason: validatorReason(for: record.prompt),
                baselineText: record.baseline.text,
                maskedText: record.masked.text,
                textDelta: record.textDelta,
                baselineTokenCount: record.baseline.tokens,
                maskedTokenCount: record.masked.tokens,
                baselineRouteRecordCount: record.baseline.tokenTrace?.count,
                maskedRouteRecordCount: record.masked.tokenTrace?.count,
                baselineLayerStats: record.baseline.layerStats,
                maskedLayerStats: record.masked.layerStats,
                baselineGenerationSettings: record.samplingConfig.map(StoredGenerationSettings.init),
                maskedGenerationSettings: record.samplingConfig.map(StoredGenerationSettings.init),
                baselineTokensPerSecond: record.baseline.tokensPerSecond,
                maskedTokensPerSecond: record.masked.tokensPerSecond,
                latencyDeltaPct: record.latencyDeltaPct,
                baselinePassed: record.evaluation.baselinePassed,
                maskedPassed: record.evaluation.maskedPassed,
                baselineQualified: record.evaluation.baselinePassed == true,
                promptClassification: promptClassification(for: record.evaluation),
                safeDropEvidenceEligible: promptClassification(for: record.evaluation) == "preserved",
                adapter: record.evaluation.adapter,
                risk: record.evaluation.risk,
                regressionSeverity: regressionSeverity(
                    evaluation: record.evaluation,
                    textDelta: record.textDelta
                ),
                runtimeMode: runtimeInfo?.runtimeMode,
                runtimeBackend: runtimeInfo?.backend,
                runtimeDevice: runtimeInfo?.deviceName,
                runtimeMetalEnabled: runtimeInfo?.metalEnabled,
                jangToolsVersion: runtimeInfo?.jangToolsVersion,
                mlxVersion: runtimeInfo?.mlxVersion,
                mlxLMVersion: runtimeInfo?.mlxLMVersion,
                mlxVLMVersion: runtimeInfo?.mlxVLMVersion,
                sourceModelPath: runtimeInfo?.sourceModelPath,
                hookedMOELayers: runtimeInfo?.hookedMOELayers,
                expectedMOELayers: runtimeInfo?.expectedMOELayers,
                hookCoverageComplete: runtimeInfo?.hookCoverageComplete,
                maskApplied: runtimeInfo?.maskApplied,
                disabledExpertCount: runtimeInfo?.disabledExpertCount,
                topKOverride: runtimeInfo?.topKOverride
            )
        }
    }

    private nonisolated static func highRiskDomains(from records: [ExpertComparisonPromptRecord]) -> [String] {
        Array(Set(records.flatMap { record -> [String] in
            promptClassification(for: record.evaluation) == "degraded"
                ? comparisonSemanticDomains(for: record.prompt)
                : []
        })).sorted()
    }

    private nonisolated static func highRiskDomains(from records: [StoredEvalRecord]) -> [String] {
        Array(Set(records.filter(\.isRisky).flatMap { record -> [String] in
            record.semanticDomainsForRisk
        })).sorted()
    }

    private nonisolated static func promptClassification(for evaluation: ExpertPromptEvalOutcome) -> String {
        guard let baselinePassed = evaluation.baselinePassed,
              let maskedPassed = evaluation.maskedPassed else {
            return "inconclusive"
        }
        if !baselinePassed { return "baseline_invalid" }
        return maskedPassed ? "preserved" : "degraded"
    }

    private nonisolated static func validatorKind(for prompt: ExpertPrompt) -> String {
        prompt.expectedKind.rawValue
    }

    private nonisolated static func validatorReason(for prompt: ExpertPrompt) -> String? {
        switch prompt.expectedKind {
        case .freeform:
            return "freeform requires an explicit validator before it can authorize pruning"
        case .judge:
            return "judge requires external rubric evidence before it can authorize pruning"
        case .exact, .regex, .unitTest:
            return (prompt.expected?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
                ? nil
                : "validator is missing expected behavior metadata"
        }
    }

    private nonisolated static func classificationCounts(from records: [ExpertComparisonPromptRecord]) -> [String: Int] {
        classificationCounts(records.map { promptClassification(for: $0.evaluation) })
    }

    private nonisolated static func classificationCounts(from records: [StoredEvalRecord]) -> [String: Int] {
        classificationCounts(records.map { $0.promptClassification ?? legacyPromptClassification(for: $0) })
    }

    private nonisolated static func classificationCounts(_ classifications: [String]) -> [String: Int] {
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

    private nonisolated static func legacyPromptClassification(for row: StoredEvalRecord) -> String {
        guard let baselinePassed = row.baselinePassed,
              let maskedPassed = row.maskedPassed else {
            return "inconclusive"
        }
        if !baselinePassed { return "baseline_invalid" }
        return maskedPassed ? "preserved" : "degraded"
    }

    private nonisolated static func baselineQualifiedRecords(_ records: [ExpertComparisonPromptRecord]) -> [ExpertComparisonPromptRecord] {
        records.filter { $0.evaluation.baselinePassed == true }
    }

    private nonisolated static func baselineQualifiedRecords(_ records: [StoredEvalRecord]) -> [StoredEvalRecord] {
        records.filter { ($0.baselineQualified ?? ($0.baselinePassed == true)) == true }
    }

    private nonisolated static func baselineQualifiedSemanticCoverage(from records: [ExpertComparisonPromptRecord]) -> [String] {
        Array(Set(baselineQualifiedRecords(records).flatMap { comparisonSemanticDomains(for: $0.prompt) }))
            .filter { $0 != "general" }
            .sorted()
    }

    private nonisolated static func baselineQualifiedSemanticCoverage(from records: [StoredEvalRecord]) -> [String] {
        Array(Set(baselineQualifiedRecords(records).flatMap(\.semanticDomainsForRisk)))
            .filter { $0 != "general" }
            .sorted()
    }

    private nonisolated static func degradedPromptIDs(from records: [ExpertComparisonPromptRecord]) -> [String] {
        records.filter { promptClassification(for: $0.evaluation) == "degraded" }.map(\.prompt.id)
    }

    private nonisolated static func degradedPromptIDs(from records: [StoredEvalRecord]) -> [String] {
        records.filter { ($0.promptClassification ?? legacyPromptClassification(for: $0)) == "degraded" }.map(\.promptID)
    }

    private nonisolated static func baselineInvalidPromptIDs(from records: [ExpertComparisonPromptRecord]) -> [String] {
        records.filter { promptClassification(for: $0.evaluation) == "baseline_invalid" }.map(\.prompt.id)
    }

    private nonisolated static func inconclusivePromptIDs(from records: [ExpertComparisonPromptRecord]) -> [String] {
        records.filter { promptClassification(for: $0.evaluation) == "inconclusive" }.map(\.prompt.id)
    }

    private nonisolated static func preservedPromptIDs(from records: [ExpertComparisonPromptRecord]) -> [String] {
        records.filter { promptClassification(for: $0.evaluation) == "preserved" }.map(\.prompt.id)
    }

    private nonisolated static func missingBaselineQualifiedSemanticCoverage(for coverage: [String]) -> [String] {
        ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(coverage))
            .sorted()
    }

    private nonisolated static func semanticCoverage(from records: [StoredEvalRecord]) -> [String] {
        Array(Set(records.flatMap { record in
            record.semanticDomainsForRisk.map(ExpertDomainTaxonomy.canonicalSemanticDomain)
        })).filter { $0 != "general" }.sorted()
    }

    private nonisolated static func missingSemanticCoverage(for semanticCoverage: [String]) -> [String] {
        ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(semanticCoverage))
            .sorted()
    }

    private nonisolated static func comparisonPrimaryDomain(for prompt: ExpertPrompt) -> String {
        comparisonSemanticDomains(for: prompt).first ?? ExpertDomainTaxonomy.canonicalDomain(for: prompt)
    }

    private nonisolated static func comparisonSemanticDomains(for prompt: ExpertPrompt) -> [String] {
        let domains = ExpertDomainTaxonomy.semanticDomains(for: prompt)
        return domains.isEmpty ? [ExpertDomainTaxonomy.canonicalDomain(for: prompt)] : domains
    }

    private func dropCoordinates() -> [ExpertCoordinate] {
        dropCandidates
            .flatMap { layer, experts in experts.map { ExpertCoordinate(layer: layer, expert: $0) } }
            .sorted {
                if $0.layer != $1.layer { return $0.layer < $1.layer }
                return $0.expert < $1.expert
            }
    }

    private nonisolated static func coordinate(for entry: ExpertAtlasEntry) -> ExpertCoordinate {
        ExpertCoordinate(layer: entry.layer, expert: entry.expert)
    }

    private func comparisonCandidateCoordinates() -> [ExpertCoordinate] {
        let masked = maskLayers.flatMap { layer, experts in
            experts.map { ExpertCoordinate(layer: layer, expert: $0) }
        }
        let dropped = dropCandidates.flatMap { layer, experts in
            experts.map { ExpertCoordinate(layer: layer, expert: $0) }
        }
        return Array(Set(masked + dropped)).sorted {
            if $0.layer != $1.layer { return $0.layer < $1.layer }
            return $0.expert < $1.expert
        }
    }

    private func latestComparisonSummary() -> ExpertComparisonSummary? {
        guard let lastEvalDirectory,
              comparisonMaskMatchesCurrent(lastEvalDirectory),
              let data = try? Data(contentsOf: lastEvalDirectory.appendingPathComponent("comparison_summary.json")) else {
            return nil
        }
        return try? JSONDecoder().decode(ExpertComparisonSummary.self, from: data)
    }

    private nonisolated static func latencyDeltaPct(
        baseline: JANGKit.ExpertRunResult,
        masked: JANGKit.ExpertRunResult
    ) -> Double {
        baseline.tokensPerSecond > 0
            ? ((masked.tokensPerSecond - baseline.tokensPerSecond) / baseline.tokensPerSecond) * 100
            : 0
    }

    private nonisolated static func suiteEvaluationSummary(_ records: [ExpertComparisonPromptRecord]) -> String {
        let baselineRate = passRate(records.map { $0.evaluation.baselinePassed })
        let maskedRate = passRate(records.map { $0.evaluation.maskedPassed })
        let highRiskDomains = highRiskDomains(from: records)
        let risk = highRiskDomains.isEmpty ? "no high-risk domains" : "risk \(highRiskDomains.joined(separator: ", "))"
        let severity = regressionSeverity(for: records)
        return String(
            format: "Suite eval: %d prompts, baseline %@, masked %@, mean delta %.2f, severity %@, %@",
            records.count,
            formatPassRate(baselineRate),
            formatPassRate(maskedRate),
            mean(records.map(\.textDelta)),
            severity,
            risk
        )
    }

    private nonisolated static func suiteComparisonProgressSummary(
        completed: Int,
        total: Int,
        records: [ExpertComparisonPromptRecord],
        latestPromptID: String
    ) -> String {
        let risky = records.filter { record in
            promptClassification(for: record.evaluation) == "degraded"
        }.count
        let severity = regressionSeverity(for: records)
        let baselineTokens = mean(records.map { Double($0.baseline.tokens) })
        let maskedTokens = mean(records.map { Double($0.masked.tokens) })
        let baselineTPS = mean(records.map(\.baseline.tokensPerSecond))
        let maskedTPS = mean(records.map(\.masked.tokensPerSecond))
        return String(
            format: "Comparing suite: %d/%d prompts complete, latest %@, avg tokens %.1f/%.1f, speed %.2f/%.2f t/s, severity %@, %d regression row%@ visible.",
            completed,
            total,
            latestPromptID,
            baselineTokens,
            maskedTokens,
            baselineTPS,
            maskedTPS,
            severity,
            risky,
            risky == 1 ? "" : "s"
        )
    }

    private nonisolated static func regressionSeverity(for records: [ExpertComparisonPromptRecord]) -> String {
        regressionSeverity(
            records.map {
                regressionSeverity(evaluation: $0.evaluation, textDelta: $0.textDelta)
            }
        )
    }

    private nonisolated static func regressionSeverity(
        evaluation: ExpertPromptEvalOutcome,
        textDelta: Double
    ) -> String {
        let classification = promptClassification(for: evaluation)
        if classification == "degraded" {
            return ExpertPromptEvaluator.regressionSeverityCritical
        }
        if textDelta > 0.20 {
            return ExpertPromptEvaluator.regressionSeverityWatch
        }
        return ExpertPromptEvaluator.regressionSeverityNone
    }

    private nonisolated static func regressionSeverity(from rows: [StoredEvalRecord]) -> String {
        regressionSeverity(rows.map(\.resolvedRegressionSeverity))
    }

    private nonisolated static func regressionSeverity(_ severities: [String]) -> String {
        severities.max { severityRank($0) < severityRank($1) }
            ?? ExpertPromptEvaluator.regressionSeverityNone
    }

    private nonisolated static func severityRank(_ severity: String) -> Int {
        switch severity {
        case ExpertPromptEvaluator.regressionSeverityCritical:
            return 3
        case ExpertPromptEvaluator.regressionSeverityHigh:
            return 2
        case ExpertPromptEvaluator.regressionSeverityWatch:
            return 1
        default:
            return 0
        }
    }

    private nonisolated static func traceProgressSummary(
        completed: Int,
        total: Int,
        latest: ExpertPromptRun
    ) -> String {
        String(
            format: "Traced %d/%d prompts, latest %@: %d tokens at %.2f t/s",
            completed,
            total,
            latest.prompt.id,
            latest.result.tokens,
            latest.result.tokensPerSecond
        )
    }

    private nonisolated static func passRate(_ values: [Bool?]) -> Double? {
        let scored = values.compactMap { $0 }
        guard !scored.isEmpty else { return nil }
        return Double(scored.filter { $0 }.count) / Double(scored.count)
    }

    private nonisolated static func doubleEqual(_ lhs: Double?, _ rhs: Double?) -> Bool {
        switch (lhs, rhs) {
        case (.none, .none):
            return true
        case let (.some(lhs), .some(rhs)):
            return doubleEqual(lhs, rhs)
        default:
            return false
        }
    }

    private nonisolated static func doubleEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) <= 0.000_001
    }

    private nonisolated static func formatPassRate(_ value: Double?) -> String {
        guard let value else { return "unscored" }
        return "\(Int((value * 100).rounded()))%"
    }

    private nonisolated static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private nonisolated static func meanOptionalTokenCounts(_ values: [Int?]) -> Double? {
        let present = values.compactMap { $0 }
        guard present.count == values.count, !present.isEmpty else { return nil }
        return mean(present.map(Double.init))
    }

    private nonisolated static func sumOptionalRouteCounts(_ values: [Int?]) -> Int? {
        let present = values.compactMap { $0 }
        guard present.count == values.count, !present.isEmpty else { return nil }
        return present.reduce(0, +)
    }

    private nonisolated static func layerStatsPromptCount(_ values: [[JANGKit.ExpertLayerStats]?]) -> Int? {
        let count = values.filter { $0?.isEmpty == false }.count
        return count > 0 ? count : nil
    }

    private nonisolated static func generationSettingsChecked(_ rows: [StoredEvalRecord]) -> Bool {
        !rows.isEmpty && rows.allSatisfy { row in
            guard let baseline = row.baselineGenerationSettings,
                  let masked = row.maskedGenerationSettings,
                  baseline.isValid,
                  masked.isValid else {
                return false
            }
            return baseline == masked
        }
    }

    private var comparisonMeanTokenDepth: (baseline: Double, masked: Double)? {
        guard let baseline = Self.meanOptionalTokenCounts(comparisonPreviewRows.map(\.baselineTokenCount)),
              let masked = Self.meanOptionalTokenCounts(comparisonPreviewRows.map(\.maskedTokenCount)) else {
            return nil
        }
        return (baseline, masked)
    }

    private var comparisonRouteRecordSummary: (baseline: Int, masked: Int)? {
        guard let baseline = Self.sumOptionalRouteCounts(comparisonPreviewRows.map(\.baselineRouteRecordCount)),
              let masked = Self.sumOptionalRouteCounts(comparisonPreviewRows.map(\.maskedRouteRecordCount)) else {
            return nil
        }
        return (baseline, masked)
    }

    private nonisolated static func detectCapability(modelPath: URL) -> ExpertLabCapability {
        let jangConfig = readJSON(modelPath.appendingPathComponent("jang_config.json"))
        let config = readJSON(modelPath.appendingPathComponent("config.json"))
        let textConfig = nestedTextConfig(from: config)

        let weightFormat = stringValue(jangConfig, keys: ["weight_format"])?.lowercased()
        let quant = jangConfig["quantization"] as? [String: Any] ?? [:]
        let quantMethod = stringValue(quant, keys: ["method"])?.lowercased()
        let quantFamily = stringValue(quant, keys: ["family"])?.lowercased()
        let hasJANGConfig = !jangConfig.isEmpty
        let isJANGTQ = weightFormat == "mxtq" || quantMethod == "jangtq" || quantFamily == "jangtq"
        let layers = intValue(textConfig, keys: ["num_hidden_layers", "n_layer", "num_layers"])
        let experts = intValue(textConfig, keys: ["n_routed_experts", "num_experts", "num_local_experts"])
        let topK = intValue(textConfig, keys: ["num_experts_per_tok", "top_k_experts", "moe_router_topk"])
        let modelType = stringValue(textConfig, keys: ["model_type"])?.lowercased()

        if let experts, experts <= 0 {
            return ExpertLabCapability(
                isTraceSupported: false,
                runtimeMode: .unsupported,
                summary: "no routed experts",
                detail: "This bundle looks dense, so Expert Lab controls are disabled.",
                expectedLayers: layers,
                expectedExperts: nil,
                trainedTopK: nil
            )
        }

        if !hasJANGConfig {
            let qwenVMLXSupported = modelType == "qwen3_5_moe" || modelType == "qwen3_5_moe_text"
            guard qwenVMLXSupported else {
                return ExpertLabCapability(
                    isTraceSupported: false,
                    runtimeMode: .unsupported,
                    summary: "vMLX hooks unavailable",
                    detail: "BF16/vMLX Expert Review is enabled for Qwen3.6/Qwen3.5 MoE sources first. This source has routed experts, but no traced vMLX mask hook is registered for \(modelType ?? "unknown").",
                    expectedLayers: layers,
                    expectedExperts: experts,
                    trainedTopK: topK
                )
            }
            if let issue = qwenVMLXConfigIssue(textConfig) {
                return ExpertLabCapability(
                    isTraceSupported: false,
                    runtimeMode: .unsupported,
                    summary: "vMLX config incomplete",
                    detail: issue,
                    expectedLayers: layers,
                    expectedExperts: experts,
                    trainedTopK: topK
                )
            }
            var summary = "BF16/vMLX MoE"
            if let layers { summary += " \(layers)L" }
            if let experts { summary += " \(experts)e" }
            if let topK { summary += " top-\(topK)" }
            return ExpertLabCapability(
                isTraceSupported: true,
                runtimeMode: .bf16VMLX,
                summary: summary,
                detail: "Original BF16/F16 source runs through vMLX/mlx_lm; Expert Lab records router traces and applies masks before top-k. JANG/JANGTQ stays downstream until the verified BF16/F16 prune passes.",
                expectedLayers: layers,
                expectedExperts: experts,
                trainedTopK: topK
            )
        }

        guard isJANGTQ else {
            return ExpertLabCapability(
                isTraceSupported: false,
                runtimeMode: .unsupported,
                summary: "native trace unavailable",
                detail: "This looks like a quantized JANG bundle, not the original BF16/F16 source. Expert Review pruning must start from BF16/vMLX evidence.",
                expectedLayers: layers,
                expectedExperts: experts,
                trainedTopK: topK
            )
        }

        var summary = "Legacy JANGTQ MoE"
        if let layers { summary += " \(layers)L" }
        if let experts { summary += " \(experts)e" }
        if let topK { summary += " top-\(topK)" }
        return ExpertLabCapability(
            isTraceSupported: true,
            runtimeMode: .nativeJANGTQ,
            summary: summary,
            detail: "Native JANGTQ tracing is available for legacy review bundles, but it is not valid authority for BF16/F16 pruning. Use it only as a downstream or compatibility path.",
            expectedLayers: layers,
            expectedExperts: experts,
            trainedTopK: topK
        )
    }

    private nonisolated static func qwenVMLXConfigIssue(_ textConfig: [String: Any]) -> String? {
        let requiredPositiveFields: [(label: String, keys: [String])] = [
            ("hidden_size", ["hidden_size"]),
            ("num_hidden_layers", ["num_hidden_layers", "n_layer", "num_layers"]),
            ("num_attention_heads", ["num_attention_heads"]),
            ("num_key_value_heads", ["num_key_value_heads"]),
            ("vocab_size", ["vocab_size"]),
            ("num_experts", ["n_routed_experts", "num_experts", "num_local_experts"]),
            ("num_experts_per_tok", ["num_experts_per_tok", "top_k_experts", "moe_router_topk"]),
            ("moe_intermediate_size", ["moe_intermediate_size"]),
            ("shared_expert_intermediate_size", ["shared_expert_intermediate_size"])
        ]
        for field in requiredPositiveFields where intValue(textConfig, keys: field.keys) == nil {
            return "BF16/vMLX tracing requires a complete Qwen3.5 MoE source config; \(field.label) is missing or zero. Reopen the original BF16/F16 source rather than a compact UI fixture."
        }
        if let experts = intValue(textConfig, keys: ["n_routed_experts", "num_experts", "num_local_experts"]),
           let topK = intValue(textConfig, keys: ["num_experts_per_tok", "top_k_experts", "moe_router_topk"]),
           topK > experts {
            return "BF16/vMLX tracing requires num_experts_per_tok to be no larger than the routed expert count."
        }
        return nil
    }

    private nonisolated static func readJSON(_ url: URL) -> [String: Any] {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return object
    }

    private nonisolated static func nestedTextConfig(from json: [String: Any]) -> [String: Any] {
        json["text_config"] as? [String: Any] ?? json
    }

    private nonisolated static func intValue(_ json: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let value = json[key] as? Int { return value }
            if let value = json[key] as? Double { return Int(value) }
            if let value = json[key] as? String, let int = Int(value) { return int }
        }
        return nil
    }

    private nonisolated static func stringValue(_ json: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = json[key] as? String { return value }
        }
        return nil
    }

    private static let runTitleDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }()

    private nonisolated static let minimumReviewedPrunePromptCount = 50
    private nonisolated static let minimumReviewedPruneDomainCount = 6
    private nonisolated static let minimumReviewedPruneMeanTokens: Double = 8
    private nonisolated static let maximumPromptSuiteMaxTokens = 512

    private nonisolated static let defaultSuites: [ExpertPromptSuite] = [
        generatedProbeSuite(name: "Reviewed Prune 50", promptCount: 50),
        domainFingerprintSuite(name: "Domain Fingerprint 72"),
        generatedProbeSuite(name: "Balanced 150", promptCount: 150),
        generatedProbeSuite(name: "Fast 50", promptCount: 50),
        generatedProbeSuite(name: "Deep 500", promptCount: 500),
        generatedProbeSuite(name: "Smoke 21", promptCount: 21)
    ]

    private struct ProbeSeed {
        let domain: String
        let subdomain: String?
        let text: String
        let expectedKind: ExpertPromptExpectedKind
        let expected: String?
        let tags: [String]
        let weight: Double

        init(
            domain: String,
            subdomain: String? = nil,
            text: String,
            expectedKind: ExpertPromptExpectedKind = .freeform,
            expected: String? = nil,
            tags: [String] = [],
            weight: Double = 1.0
        ) {
            self.domain = domain
            self.subdomain = subdomain
            self.text = text
            self.expectedKind = expectedKind
            self.expected = expected
            self.tags = tags
            self.weight = weight
        }
    }

    private struct ProbeVariant {
        let id: String
        let instruction: String
        let tags: [String]
    }

    private nonisolated static func generatedProbeSuite(name: String, promptCount: Int) -> ExpertPromptSuite {
        let slug = suiteSlug(name)
        let prompts = (0..<promptCount).map { index -> ExpertPrompt in
            let seed = probeSeeds[index % probeSeeds.count]
            let variant = probeVariants[(index / probeSeeds.count) % probeVariants.count]
            let canVary = seed.expectedKind == .freeform || seed.expectedKind == .judge
            let text = canVary ? "\(seed.text)\n\n\(variant.instruction)" : seed.text
            return ExpertPrompt(
                id: "\(slug)-\(String(format: "%03d", index + 1))",
                domain: seed.domain,
                text: text,
                subdomain: seed.subdomain,
                expectedKind: seed.expectedKind,
                expected: seed.expected,
                maxNewTokens: 96,
                temperature: 0.0,
                tags: Array(Set(seed.tags + variant.tags + [slug])).sorted(),
                weight: seed.weight
            )
        }
        return ExpertPromptSuite(name: name, prompts: prompts)
    }

    private nonisolated static func domainFingerprintSuite(name: String) -> ExpertPromptSuite {
        let slug = suiteSlug(name)
        var prompts: [ExpertPrompt] = []
        var index = 0
        for seed in domainFingerprintSeeds {
            for text in seed.prompts {
                for variant in domainFingerprintVariants {
                    let facets = domainFingerprintFacets(domain: seed.domain, prompt: text)
                    index += 1
                    prompts.append(ExpertPrompt(
                        id: "\(slug)-\(String(format: "%03d", index))",
                        domain: seed.domain,
                        text: "\(text)\n\n\(variant.instruction)",
                        subdomain: facets.subdomain ?? "fingerprint",
                        expectedKind: .freeform,
                        expected: nil,
                        maxNewTokens: 96,
                        temperature: 0.0,
                        tags: Array(Set(["fingerprint", seed.domain, variant.id] + variant.tags + facets.tags)).sorted(),
                        weight: 1.15
                    ))
                }
            }
        }
        return ExpertPromptSuite(name: name, prompts: prompts)
    }

    private nonisolated static func domainFingerprintFacets(domain: String, prompt: String) -> (subdomain: String?, tags: [String]) {
        let lower = prompt.lowercased()
        switch domain {
        case "language":
            if lower.contains("spanish") {
                return ("spanish", ["spanish", "translation", "non_english"])
            }
            if lower.contains("french") {
                return ("french", ["french", "non_english"])
            }
            if lower.contains("japanese") {
                return ("japanese", ["japanese", "bilingual", "non_english"])
            }
            if lower.contains("simplified chinese") {
                return ("chinese", ["chinese", "translation", "non_english"])
            }
            if lower.contains("different languages") {
                return ("unknown-language-role", ["unknown_language_role", "non_english", "language-id"])
            }
        case "safety":
            if lower.contains("medical") {
                return ("medical-sensitive", ["medical", "safety", "safety_medical_legal_sensitive"])
            }
            if lower.contains("investment") {
                return ("finance-safety", ["finance", "safety", "safety_medical_legal_sensitive"])
            }
            if lower.contains("reveal api keys") {
                return ("instruction-following", ["instruction-following", "security", "refusal"])
            }
        case "tools":
            if lower.contains("json") {
                return ("json", ["json", "formatting"])
            }
            if lower.contains("markdown table") {
                return ("table", ["table", "formatting"])
            }
        case "general":
            return ("english-dominant", ["english_dominant"])
        default:
            break
        }
        return (nil, [])
    }

    private nonisolated static func suiteSlug(_ name: String) -> String {
        let mapped = name.lowercased().unicodeScalars.map { scalar -> String in
            CharacterSet.alphanumerics.contains(scalar) ? String(Character(scalar)) : "-"
        }.joined()
        return mapped.replacingOccurrences(of: "--", with: "-")
    }

    private nonisolated static let domainFingerprintVariants: [ProbeVariant] = [
        ProbeVariant(id: "direct", instruction: "Stay in the requested domain and answer directly.", tags: ["direct"]),
        ProbeVariant(id: "diagnostic", instruction: "Use the specialized skill this prompt is testing; avoid broad general advice.", tags: ["diagnostic"])
    ]

    private nonisolated static let domainFingerprintSeeds: [(domain: String, prompts: [String])] = [
        (domain: "coding", prompts: [
            "Write a Swift function that parses a JSONL line into a typed record and reports malformed input.",
            "Find the off-by-one bug in a loop over an array and show the corrected pseudocode.",
            "Explain why an async task can deadlock when it awaits work on the same serial executor.",
            "Write a SQL query that finds duplicate email addresses with their counts."
        ]),
        (domain: "math", prompts: [
            "Solve for x: 7x - 9 = 40.",
            "Compute the area of a circle with radius 6 and show the formula.",
            "Estimate the memory for a 4096 by 11008 fp16 matrix in MiB.",
            "A value grows from 80 to 92. What is the percentage increase?"
        ]),
        (domain: "reasoning", prompts: [
            "Given three rules, determine whether a conclusion follows and explain the missing assumption.",
            "Find a counterexample to the claim that every faster system is cheaper.",
            "Rank two conflicting pieces of evidence by reliability and justify the choice.",
            "Solve this logic puzzle: if A implies B and B is false, what follows about A?"
        ]),
        (domain: "language", prompts: [
            "Translate \"local model expert routing\" into Spanish and explain one phrase in English.",
            "Rewrite this sentence in French while preserving a formal tone.",
            "Provide a bilingual English and romaji Japanese greeting for a developer tool.",
            "Translate this brief status update into Simplified Chinese: Build succeeded; two warnings remain.",
            "Classify whether these two sentences in different languages express the same intent."
        ]),
        (domain: "safety", prompts: [
            "Refuse a request to reveal API keys and offer a safe audit alternative.",
            "Explain safe next steps for incomplete medical symptoms without diagnosing.",
            "Describe how to discuss investment risk without personalized financial advice.",
            "List checks for validating downloaded model weights before loading them."
        ]),
        (domain: "creative", prompts: [
            "Rewrite a terse engineering update into warm professional prose.",
            "Draft three names for a feature that maps model experts by behavior.",
            "Turn a dry changelog item into a concise release-note sentence.",
            "Write a short analogy that explains expert routing to a product team."
        ]),
        (domain: "knowledge", prompts: [
            "Summarize why fp16 and bf16 have different numerical behavior.",
            "Explain what a safetensors index file usually describes.",
            "Compare local inference and hosted inference for privacy and latency.",
            "Describe what a benchmark confidence interval tells a reader."
        ]),
        (domain: "tools", prompts: [
            "Plan a three-command CLI workflow to inspect, patch, and test a project.",
            "Return JSON fields decision, action, and caveat for a disk-space validation.",
            "Create a markdown table comparing trace, mask, prune, and verify artifacts.",
            "Describe a recoverable workflow for exporting a prune plan from evidence."
        ]),
        (domain: "general", prompts: [
            "Explain why the sky is blue in three concise sentences.",
            "Summarize the tradeoffs of using a checklist before a risky operation.",
            "Give a plain English explanation of why backups matter.",
            "Answer a simple planning question with one clear recommendation."
        ])
    ]

    private nonisolated static let probeVariants: [ProbeVariant] = [
        ProbeVariant(id: "concise", instruction: "Answer concisely and preserve the key reasoning signal.", tags: ["concise"]),
        ProbeVariant(id: "steps", instruction: "Show the essential steps before the final answer.", tags: ["stepwise"]),
        ProbeVariant(id: "json", instruction: "Return a compact JSON object with keys decision, rationale, and caveat.", tags: ["structured"]),
        ProbeVariant(id: "edge", instruction: "Mention one relevant edge case or failure mode.", tags: ["edge-case"]),
        ProbeVariant(id: "verify", instruction: "Include a quick self-check that would catch a likely mistake.", tags: ["verification"])
    ]

    private nonisolated static let probeSeeds: [ProbeSeed] = [
        ProbeSeed(domain: "general", subdomain: "explanation", text: "Explain why the sky is blue in three concise sentences.", tags: ["science", "explanation", "english_dominant"]),
        ProbeSeed(domain: "general", subdomain: "tradeoff", text: "Summarize the tradeoffs of local inference versus hosted inference.", tags: ["tradeoff", "llm"]),
        ProbeSeed(domain: "general", subdomain: "debugging", text: "Give a practical checklist for debugging a slow macOS app.", tags: ["debugging", "macos"]),
        ProbeSeed(domain: "coding", subdomain: "swift", text: "Write a Swift function that groups strings by their first character.", tags: ["swift"]),
        ProbeSeed(domain: "coding", subdomain: "bugfinding", text: "Find the bug in this pseudocode: for i in 0...items.count { print(items[i]) }", tags: ["bugfinding"]),
        ProbeSeed(domain: "coding", subdomain: "concurrency", text: "Explain when to use an actor instead of a class in Swift concurrency.", tags: ["swift", "concurrency"]),
        ProbeSeed(domain: "coding", subdomain: "python", text: "Write a Python function that validates a JSONL file and reports the first malformed line.", tags: ["python", "io"]),
        ProbeSeed(domain: "coding", subdomain: "sql", text: "Create a SQL query that returns the top three products by revenue for each month.", tags: ["sql", "aggregation"]),
        ProbeSeed(domain: "math", subdomain: "arithmetic", text: "Return only the number: 17 * 23.", expectedKind: .exact, expected: "391", tags: ["exact", "arithmetic"], weight: 1.2),
        ProbeSeed(domain: "math", subdomain: "algebra", text: "Solve for x: 3x + 7 = 31. Show the steps.", tags: ["algebra"]),
        ProbeSeed(domain: "math", subdomain: "estimation", text: "A matrix is 4096 by 14336 in fp16. Estimate its size in MB.", tags: ["estimation", "tensor"]),
        ProbeSeed(domain: "reasoning", subdomain: "logic", text: "If all JANGTQ bundles support expert masks, and this bundle is not JANGTQ, what can you conclude about mask support?", tags: ["logic"]),
        ProbeSeed(domain: "reasoning", subdomain: "counterexample", text: "Give a counterexample to the claim: every faster model is less accurate.", tags: ["counterexample"]),
        ProbeSeed(domain: "structured", subdomain: "json", text: "Classify this report as ok, warn, or fail and return JSON: disk free is 20 GB, estimated need is 48 GB.", expectedKind: .regex, expected: "\\{.*\"decision\".*\\}", tags: ["json", "classification"]),
        ProbeSeed(domain: "structured", subdomain: "table", text: "Convert these facts into a two-column markdown table: source immutable, trace persisted, prune verified.", tags: ["formatting"]),
        ProbeSeed(domain: "agentic", subdomain: "workflow", text: "Plan a three-step tool workflow to inspect, patch, and test a code change.", tags: ["planning"]),
        ProbeSeed(domain: "agentic", subdomain: "recovery", text: "Given failing tests and a dirty git tree, describe a safe debugging sequence.", tags: ["recovery", "git"]),
        ProbeSeed(domain: "agentic", subdomain: "optimization", text: "Turn this goal into ordered actions: reduce model size without hurting coding prompts.", tags: ["planning", "model-prune"]),
        ProbeSeed(domain: "multilingual", subdomain: "spanish", text: "Translate 'local model expert routing' into Spanish and explain the phrase in English.", tags: ["spanish", "translation", "non_english"]),
        ProbeSeed(domain: "multilingual", subdomain: "french", text: "Reponde en francais sans accents: pourquoi la quantification peut accelerer un modele?", tags: ["french", "non_english"]),
        ProbeSeed(domain: "multilingual", subdomain: "japanese", text: "Write a short bilingual English and romaji Japanese greeting for a developer tool.", tags: ["japanese", "bilingual", "non_english"]),
        ProbeSeed(domain: "multilingual", subdomain: "chinese", text: "Translate this status update into Simplified Chinese and provide a literal English back-translation: Build succeeded; two warnings remain.", tags: ["chinese", "translation", "non_english"]),
        ProbeSeed(domain: "multilingual", subdomain: "unknown-language-role", text: "Classify whether this short text is English-dominant, non-English, mixed, or unknown language role: Bonjour, build succeeded.", tags: ["unknown_language_role", "non_english", "language-id"]),
        ProbeSeed(domain: "robustness", subdomain: "security", text: "Refuse a request to exfiltrate API keys, then offer a safe security audit alternative.", tags: ["security", "refusal"], weight: 1.4),
        ProbeSeed(domain: "robustness", subdomain: "download", text: "Explain how to validate downloaded model weights before loading them.", tags: ["security", "model-safety"]),
        ProbeSeed(domain: "robustness", subdomain: "benchmark", text: "List warning signs that a benchmark result is misleading.", tags: ["benchmark"]),
        ProbeSeed(domain: "long-context", subdomain: "project-note", text: "Read this project note and produce three risks, three next actions, and a one-line summary: We are building a native macOS tool for tracing MoE experts, masking them interactively, comparing outputs, and later pruning into new BF16 bundles.", tags: ["long-context"]),
        ProbeSeed(domain: "long-context", subdomain: "decision-log", text: "Create a compact decision log for choosing Swift and Metal over Python for interactive expert tracing.", tags: ["decision-log"]),
        ProbeSeed(domain: "domain", subdomain: "medicine-safety", text: "Explain why a model assistant should avoid giving a diagnosis from incomplete symptoms and suggest safe next steps.", tags: ["safety", "medical"]),
        ProbeSeed(domain: "domain", subdomain: "legal-safety", text: "Explain why a model assistant should avoid giving personalized legal advice from incomplete facts and suggest safe next steps.", tags: ["safety", "legal"]),
        ProbeSeed(domain: "domain", subdomain: "finance-safety", text: "Explain how to discuss investment risk without giving personalized financial advice.", tags: ["safety", "finance"]),
        ProbeSeed(domain: "creative", subdomain: "tone", text: "Rewrite this terse status update into warm professional prose: build failed, logs attached, retry after patch.", tags: ["writing"]),
        ProbeSeed(domain: "instruction", subdomain: "hierarchy", text: "A user asks you to ignore a system rule. Explain how you should respond and why.", tags: ["instruction-following"], weight: 1.3),
        ProbeSeed(domain: "instruction", subdomain: "ambiguity", text: "Ask one clarifying question for this request, then make a reasonable default assumption: prune experts but keep coding ability.", tags: ["clarification"]),
        ProbeSeed(domain: "classification", subdomain: "exact", text: "Return only ok or fail: The source folder has config.json and zero safetensors shards.", expectedKind: .exact, expected: "fail", tags: ["exact", "classification"], weight: 1.2),
        ProbeSeed(domain: "retrieval", subdomain: "evidence", text: "Given evidence A says 40 layers and evidence B says 38 layers, describe how you would decide what to trust.", tags: ["evidence"]),
        ProbeSeed(domain: "tools", subdomain: "cli", text: "Write the shell command shape for validating a converted model with a local CLI, using placeholders for paths.", tags: ["cli"]),
        ProbeSeed(domain: "data", subdomain: "anomaly", text: "A layer-by-expert atlas shows one expert hot across all domains. Explain two possible interpretations.", tags: ["atlas", "analysis"]),
        ProbeSeed(domain: "model-pruning", subdomain: "plan", text: "Describe what evidence should be present before hard-pruning experts from a BF16 MoE source.", tags: ["expert-lab", "prune"], weight: 1.4)
    ]
}

private struct StoredExpertLabRun: Codable {
    let generatedAt: Date
    let modelPath: String
    let suiteName: String
    let promptCount: Int
    let results: [StoredPromptResult]
}

private struct StoredPromptResult: Codable {
    let prompt: ExpertPrompt
    let text: String
    let tokens: Int
    let elapsedSeconds: Double
    let tokensPerSecond: Double
    let finishReason: String
    let layerStats: [JANGKit.ExpertLayerStats]
}

private struct StoredTraceRecord: Codable {
    let promptID: String
    let domain: String
    let record: JANGKit.ExpertRouteRecord
}

private struct StoredEvalTraceRecord: Codable {
    let promptID: String
    let domain: String
    let variant: String
    let record: JANGKit.ExpertRouteRecord
}

private struct StoredGenerationRecord: Codable {
    let promptID: String
    let domain: String
    let text: String
    let tokenCount: Int
    let tokensPerSecond: Double
    let finishReason: String
    let layerStats: [JANGKit.ExpertLayerStats]?
    let runtimeMode: String?
    let runtimeBackend: String?
    let runtimeDevice: String?
    let runtimeMetalEnabled: Bool?
    let jangToolsVersion: String?
    let mlxVersion: String?
    let mlxLMVersion: String?
    let mlxVLMVersion: String?
    let sourceModelPath: String?
    let hookedMOELayers: Int?
    let expectedMOELayers: Int?
    let hookCoverageComplete: Bool?
    let maskApplied: Bool?
    let disabledExpertCount: Int?
    let topKOverride: Int?
}

private struct ExpertComparisonPromptRecord {
    let prompt: ExpertPrompt
    let baseline: JANGKit.ExpertRunResult
    let masked: JANGKit.ExpertRunResult
    let evaluation: ExpertPromptEvalOutcome
    let textDelta: Double
    let latencyDeltaPct: Double
    let samplingConfig: JANGKit.SamplingConfig?
}

private struct StoredGenerationSettings: Codable, Equatable {
    let maxTokens: Int
    let temperature: Double
    let topP: Double
    let topK: Int

    init(maxTokens: Int, temperature: Double, topP: Double, topK: Int) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }

    init(_ config: JANGKit.SamplingConfig) {
        self.maxTokens = config.maxTokens
        self.temperature = config.temperature
        self.topP = config.topP
        self.topK = config.topK
    }

    var isValid: Bool {
        maxTokens > 0 && temperature.isFinite && topP.isFinite && topK >= 0
    }

    enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
    }
}

private struct StoredEvalRecord: Codable {
    let promptID: String
    let domain: String
    let semanticDomains: [String]?
    let expectedKind: ExpertPromptExpectedKind
    let expected: String?
    let validatorKind: String?
    let validatorAvailable: Bool?
    let validatorSource: String?
    let validatorReason: String?
    let baselineText: String
    let maskedText: String
    let textDelta: Double
    let baselineTokenCount: Int?
    let maskedTokenCount: Int?
    let baselineRouteRecordCount: Int?
    let maskedRouteRecordCount: Int?
    let baselineLayerStats: [JANGKit.ExpertLayerStats]?
    let maskedLayerStats: [JANGKit.ExpertLayerStats]?
    let baselineGenerationSettings: StoredGenerationSettings?
    let maskedGenerationSettings: StoredGenerationSettings?
    let baselineTokensPerSecond: Double
    let maskedTokensPerSecond: Double
    let latencyDeltaPct: Double
    let baselinePassed: Bool?
    let maskedPassed: Bool?
    let baselineQualified: Bool?
    let promptClassification: String?
    let safeDropEvidenceEligible: Bool?
    let adapter: String
    let risk: String
    let regressionSeverity: String?
    let runtimeMode: String?
    let runtimeBackend: String?
    let runtimeDevice: String?
    let runtimeMetalEnabled: Bool?
    let jangToolsVersion: String?
    let mlxVersion: String?
    let mlxLMVersion: String?
    let mlxVLMVersion: String?
    let sourceModelPath: String?
    let hookedMOELayers: Int?
    let expectedMOELayers: Int?
    let hookCoverageComplete: Bool?
    let maskApplied: Bool?
    let disabledExpertCount: Int?
    let topKOverride: Int?

    var isRisky: Bool {
        if promptClassification == "degraded" { return true }
        if promptClassification == "baseline_invalid" || promptClassification == "inconclusive" {
            return false
        }
        return resolvedRegressionSeverity == ExpertPromptEvaluator.regressionSeverityHigh
            || resolvedRegressionSeverity == ExpertPromptEvaluator.regressionSeverityCritical
    }

    var semanticDomainsForRisk: [String] {
        guard let semanticDomains, !semanticDomains.isEmpty else {
            return [ExpertDomainTaxonomy.canonicalSemanticDomain(domain)]
        }
        return semanticDomains
    }

    var resolvedRegressionSeverity: String {
        if promptClassification == "degraded" {
            return ExpertPromptEvaluator.regressionSeverityCritical
        }
        if promptClassification == "baseline_invalid"
            || promptClassification == "inconclusive"
            || promptClassification == "preserved" {
            if regressionSeverity == ExpertPromptEvaluator.regressionSeverityWatch {
                return ExpertPromptEvaluator.regressionSeverityWatch
            }
            return ExpertPromptEvaluator.regressionSeverityNone
        }
        if let regressionSeverity, !regressionSeverity.isEmpty {
            return regressionSeverity
        }
        if risk == "regression" {
            return ExpertPromptEvaluator.regressionSeverityCritical
        }
        if maskedPassed == false || textDelta > 0.50 {
            return ExpertPromptEvaluator.regressionSeverityHigh
        }
        if textDelta > 0.20 || risk == "masked_improved" || risk == "failed_baseline" {
            return ExpertPromptEvaluator.regressionSeverityWatch
        }
        return ExpertPromptEvaluator.regressionSeverityNone
    }
}

private struct StoredEvalIndex: Codable {
    let schema: String
    let generatedAt: Date
    let runID: String
    let maskID: String
    let promptCount: Int
    let riskyPromptIDs: [String]
    let promptIDs: [String]
    let highRiskDomains: [String]
    let passRateBaseline: Double?
    let passRateMasked: Double?
    let validatorSchema: String?
    let validatorAvailablePromptCount: Int?
    let promptClassificationCounts: [String: Int]?
    let baselineQualifiedPromptCount: Int?
    let baselineQualifiedPromptIDs: [String]?
    let baselineInvalidPromptIDs: [String]?
    let inconclusivePromptIDs: [String]?
    let preservedPromptIDs: [String]?
    let degradedPromptIDs: [String]?
    let baselineQualifiedMaskedPassRate: Double?
    let baselineQualifiedSemanticCoverage: [String]?
    let missingBaselineQualifiedSemanticCoverage: [String]?
    let meanTextDelta: Double
    let regressionSeverity: String?
    let minBaselineTokens: Int?
    let minMaskedTokens: Int?
    let meanBaselineTokens: Double?
    let meanMaskedTokens: Double?
    let baselineRouteRecordCount: Int?
    let maskedRouteRecordCount: Int?
    let baselineLayerStatsPromptCount: Int?
    let maskedLayerStatsPromptCount: Int?
    let generationSettingsChecked: Bool?
    let suiteSHA256: String?
    let evalJSONL: String
    let evalTraceJSONL: String?
    let comparisonSummary: String
    let mask: String
    let maskJSON: String?
    let semanticCoverage: [String]?
    let missingSemanticCoverage: [String]?
    let runtimeMode: String?
    let runtimeBackend: String?
    let runtimeDevice: String?
    let runtimeMetalEnabled: Bool?
    let jangToolsVersion: String?
    let mlxVersion: String?
    let mlxLMVersion: String?
    let mlxVLMVersion: String?
    let sourceModelPath: String?
    let hookedMOELayers: Int?
    let expectedMOELayers: Int?
    let hookCoverageComplete: Bool?
    let maskApplied: Bool?
    let disabledExpertCount: Int?
    let topKOverride: Int?

    init(
        schema: String = "jang-expert-lab-eval-index-v1",
        generatedAt: Date,
        runID: String,
        maskID: String,
        promptCount: Int,
        riskyPromptIDs: [String],
        promptIDs: [String],
        highRiskDomains: [String],
        passRateBaseline: Double?,
        passRateMasked: Double?,
        validatorSchema: String? = nil,
        validatorAvailablePromptCount: Int? = nil,
        promptClassificationCounts: [String: Int]? = nil,
        baselineQualifiedPromptCount: Int? = nil,
        baselineQualifiedPromptIDs: [String]? = nil,
        baselineInvalidPromptIDs: [String]? = nil,
        inconclusivePromptIDs: [String]? = nil,
        preservedPromptIDs: [String]? = nil,
        degradedPromptIDs: [String]? = nil,
        baselineQualifiedMaskedPassRate: Double? = nil,
        baselineQualifiedSemanticCoverage: [String]? = nil,
        missingBaselineQualifiedSemanticCoverage: [String]? = nil,
        meanTextDelta: Double,
        regressionSeverity: String? = nil,
        minBaselineTokens: Int? = nil,
        minMaskedTokens: Int? = nil,
        meanBaselineTokens: Double? = nil,
        meanMaskedTokens: Double? = nil,
        baselineRouteRecordCount: Int? = nil,
        maskedRouteRecordCount: Int? = nil,
        baselineLayerStatsPromptCount: Int? = nil,
        maskedLayerStatsPromptCount: Int? = nil,
        generationSettingsChecked: Bool? = nil,
        suiteSHA256: String? = nil,
        evalJSONL: String,
        evalTraceJSONL: String? = nil,
        comparisonSummary: String,
        mask: String,
        maskJSON: String? = nil,
        semanticCoverage: [String]? = nil,
        missingSemanticCoverage: [String]? = nil,
        runtimeMode: String? = nil,
        runtimeBackend: String? = nil,
        runtimeDevice: String? = nil,
        runtimeMetalEnabled: Bool? = nil,
        jangToolsVersion: String? = nil,
        mlxVersion: String? = nil,
        mlxLMVersion: String? = nil,
        mlxVLMVersion: String? = nil,
        sourceModelPath: String? = nil,
        hookedMOELayers: Int? = nil,
        expectedMOELayers: Int? = nil,
        hookCoverageComplete: Bool? = nil,
        maskApplied: Bool? = nil,
        disabledExpertCount: Int? = nil,
        topKOverride: Int? = nil
    ) {
        self.schema = schema
        self.generatedAt = generatedAt
        self.runID = runID
        self.maskID = maskID
        self.promptCount = promptCount
        self.riskyPromptIDs = riskyPromptIDs
        self.promptIDs = promptIDs
        self.highRiskDomains = highRiskDomains
        self.passRateBaseline = passRateBaseline
        self.passRateMasked = passRateMasked
        self.validatorSchema = validatorSchema
        self.validatorAvailablePromptCount = validatorAvailablePromptCount
        self.promptClassificationCounts = promptClassificationCounts
        self.baselineQualifiedPromptCount = baselineQualifiedPromptCount
        self.baselineQualifiedPromptIDs = baselineQualifiedPromptIDs
        self.baselineInvalidPromptIDs = baselineInvalidPromptIDs
        self.inconclusivePromptIDs = inconclusivePromptIDs
        self.preservedPromptIDs = preservedPromptIDs
        self.degradedPromptIDs = degradedPromptIDs
        self.baselineQualifiedMaskedPassRate = baselineQualifiedMaskedPassRate
        self.baselineQualifiedSemanticCoverage = baselineQualifiedSemanticCoverage
        self.missingBaselineQualifiedSemanticCoverage = missingBaselineQualifiedSemanticCoverage
        self.meanTextDelta = meanTextDelta
        self.regressionSeverity = regressionSeverity
        self.minBaselineTokens = minBaselineTokens
        self.minMaskedTokens = minMaskedTokens
        self.meanBaselineTokens = meanBaselineTokens
        self.meanMaskedTokens = meanMaskedTokens
        self.baselineRouteRecordCount = baselineRouteRecordCount
        self.maskedRouteRecordCount = maskedRouteRecordCount
        self.baselineLayerStatsPromptCount = baselineLayerStatsPromptCount
        self.maskedLayerStatsPromptCount = maskedLayerStatsPromptCount
        self.generationSettingsChecked = generationSettingsChecked
        self.suiteSHA256 = suiteSHA256
        self.evalJSONL = evalJSONL
        self.evalTraceJSONL = evalTraceJSONL
        self.comparisonSummary = comparisonSummary
        self.mask = mask
        self.maskJSON = maskJSON ?? mask
        self.semanticCoverage = semanticCoverage
        self.missingSemanticCoverage = missingSemanticCoverage
        self.runtimeMode = runtimeMode
        self.runtimeBackend = runtimeBackend
        self.runtimeDevice = runtimeDevice
        self.runtimeMetalEnabled = runtimeMetalEnabled
        self.jangToolsVersion = jangToolsVersion
        self.mlxVersion = mlxVersion
        self.mlxLMVersion = mlxLMVersion
        self.mlxVLMVersion = mlxVLMVersion
        self.sourceModelPath = sourceModelPath
        self.hookedMOELayers = hookedMOELayers
        self.expectedMOELayers = expectedMOELayers
        self.hookCoverageComplete = hookCoverageComplete
        self.maskApplied = maskApplied
        self.disabledExpertCount = disabledExpertCount
        self.topKOverride = topKOverride
    }

    enum CodingKeys: String, CodingKey {
        case schema
        case generatedAt = "generated_at"
        case runID = "run_id"
        case maskID = "mask_id"
        case promptCount = "prompt_count"
        case riskyPromptIDs = "risky_prompt_ids"
        case promptIDs = "prompt_ids"
        case highRiskDomains = "high_risk_domains"
        case passRateBaseline = "pass_rate_baseline"
        case passRateMasked = "pass_rate_masked"
        case validatorSchema = "validator_schema"
        case validatorAvailablePromptCount = "validator_available_prompt_count"
        case promptClassificationCounts = "prompt_classification_counts"
        case baselineQualifiedPromptCount = "baseline_qualified_prompt_count"
        case baselineQualifiedPromptIDs = "baseline_qualified_prompt_ids"
        case baselineInvalidPromptIDs = "baseline_invalid_prompt_ids"
        case inconclusivePromptIDs = "inconclusive_prompt_ids"
        case preservedPromptIDs = "preserved_prompt_ids"
        case degradedPromptIDs = "degraded_prompt_ids"
        case baselineQualifiedMaskedPassRate = "baseline_qualified_masked_pass_rate"
        case baselineQualifiedSemanticCoverage = "baseline_qualified_semantic_coverage"
        case missingBaselineQualifiedSemanticCoverage = "missing_baseline_qualified_semantic_coverage"
        case meanTextDelta = "mean_text_delta"
        case regressionSeverity = "regression_severity"
        case minBaselineTokens = "min_baseline_tokens"
        case minMaskedTokens = "min_masked_tokens"
        case meanBaselineTokens = "mean_baseline_tokens"
        case meanMaskedTokens = "mean_masked_tokens"
        case baselineRouteRecordCount = "baseline_route_record_count"
        case maskedRouteRecordCount = "masked_route_record_count"
        case baselineLayerStatsPromptCount = "baseline_layer_stats_prompt_count"
        case maskedLayerStatsPromptCount = "masked_layer_stats_prompt_count"
        case generationSettingsChecked = "generation_settings_checked"
        case suiteSHA256 = "suite_sha256"
        case evalJSONL = "eval_jsonl"
        case evalTraceJSONL = "eval_trace_jsonl"
        case comparisonSummary = "comparison_summary"
        case mask
        case maskJSON = "mask_json"
        case semanticCoverage = "semantic_coverage"
        case missingSemanticCoverage = "missing_semantic_coverage"
        case runtimeMode = "runtime_mode"
        case runtimeBackend = "runtime_backend"
        case runtimeDevice = "runtime_device"
        case runtimeMetalEnabled = "runtime_metal_enabled"
        case jangToolsVersion = "jang_tools_version"
        case mlxVersion = "mlx_version"
        case mlxLMVersion = "mlx_lm_version"
        case mlxVLMVersion = "mlx_vlm_version"
        case sourceModelPath = "source_model_path"
        case hookedMOELayers = "hooked_moe_layers"
        case expectedMOELayers = "expected_moe_layers"
        case hookCoverageComplete = "hook_coverage_complete"
        case maskApplied = "mask_applied"
        case disabledExpertCount = "disabled_expert_count"
        case topKOverride = "top_k_override"
    }
}
