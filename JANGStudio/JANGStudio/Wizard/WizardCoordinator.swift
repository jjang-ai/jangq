// JANGStudio/JANGStudio/Wizard/WizardCoordinator.swift
import AppKit
import SwiftUI

enum ExpertLabVisual {
    static let canvas = Color(red: 7 / 255, green: 9 / 255, blue: 14 / 255)
    static let panel = Color(red: 14 / 255, green: 19 / 255, blue: 22 / 255)
    static let panelRaised = Color(red: 19 / 255, green: 26 / 255, blue: 30 / 255)
    static let line = Color.white.opacity(0.05)
    static let accent = Color(red: 51 / 255, green: 224 / 255, blue: 232 / 255)
    static let warm = Color(red: 235 / 255, green: 184 / 255, blue: 94 / 255)
    static let good = Color(red: 112 / 255, green: 219 / 255, blue: 160 / 255)
    static let danger = Color(red: 255 / 255, green: 89 / 255, blue: 100 / 255)
    static let textDim = Color(red: 106 / 255, green: 120 / 255, blue: 125 / 255)
    static let textFaint = Color(red: 58 / 255, green: 69 / 255, blue: 72 / 255)
}

struct ExpertLabConsoleCard<Content: View>: View {
    var accent: Color = ExpertLabVisual.accent
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
            .background(ExpertLabVisual.panel.opacity(0.96))
            .overlay(alignment: .top) {
                Rectangle()
                    .fill(accent.opacity(0.55))
                    .frame(height: 1)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 5)
                    .stroke(ExpertLabVisual.line, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 5))
    }
}

struct ExpertLabKicker: View {
    let text: String
    var color: Color = ExpertLabVisual.accent

    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundStyle(color)
            .tracking(1.6)
    }
}

/// A small dark section header used inside console cards and right rails.
struct ExpertLabSectionHeader: View {
    let title: String
    var systemImage: String? = nil
    var trailing: AnyView? = nil

    var body: some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(ExpertLabVisual.accent.opacity(0.82))
            }
            Text(title)
                .font(.system(size: 9, weight: .semibold, design: .default))
                .foregroundStyle(ExpertLabVisual.textDim)
                .textCase(.uppercase)
                .tracking(0.6)
            Spacer()
            if let trailing { trailing }
        }
    }
}

/// A compact monospaced stat row for grids and right rails.
struct ExpertLabStatRow: View {
    let label: String
    let value: String
    var accent: Color = ExpertLabVisual.accent

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit().foregroundStyle(accent)
        }
        .font(.system(size: 10))
    }
}

/// A rounded control-group wrapper for the right rail's collapsible sections.
struct ExpertLabSubPanel<Content: View>: View {
    let title: String
    var systemImage: String? = nil
    var accent: Color = ExpertLabVisual.accent
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ExpertLabSectionHeader(title: title, systemImage: systemImage)
            content
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ExpertLabVisual.panelRaised.opacity(0.55))
        .overlay {
            RoundedRectangle(cornerRadius: 5)
                .stroke(ExpertLabVisual.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 5))
    }
}

/// A status pill — small, bordered, colored.
struct ExpertLabStatusPill: View {
    let text: String
    var color: Color = ExpertLabVisual.accent
    var icon: String? = nil
    var body: some View {
        HStack(spacing: 3) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 8, weight: .bold))
            }
            Text(text)
        }
        .font(.system(size: 9, weight: .semibold, design: .monospaced))
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(color.opacity(0.14))
        .foregroundStyle(color)
        .overlay {
            Capsule()
                .stroke(color.opacity(0.28), lineWidth: 1)
        }
        .clipShape(Capsule())
    }
}

/// A threshold bar: horizontal bar capped at `max`, showing current `value`.
struct ExpertLabThresholdBar: View {
    let label: String
    let value: Double
    let maxValue: Double
    var color: Color = ExpertLabVisual.accent

    init(label: String, value: Double, max: Double, color: Color = ExpertLabVisual.accent) {
        self.label = label
        self.value = value
        self.maxValue = max
        self.color = color
    }

    private var pct: Double {
        guard maxValue > 0 else { return 0 }
        return min(1.0, Swift.max(0, value / maxValue))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.2f / %.2f", value, maxValue))
                    .font(.caption2)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(ExpertLabVisual.panelRaised)
                        .frame(height: 4)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color.opacity(0.72))
                        .frame(width: Swift.max(4, proxy.size.width * pct), height: 4)
                }
            }
            .frame(height: 4)
        }
    }
}

/// A two-column row for tables like Prune Plan
/// and Verified Pruned Source.
struct ExpertLabLabeledRow: View {
    let key: String
    var value: String

    init(_ key: String, _ value: String) {
        self.key = key
        self.value = value
    }
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(key)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(value)
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.primary)
        }
    }
}

struct ExpertLabWorkflowStrip: View {
    let steps: [String]
    var activeIndex: Int = 0
    var completedIndices: Set<Int> = []

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                let isCompleted = completedIndices.contains(index)
                let isActive = index == activeIndex
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 3) {
                        if isCompleted && !isActive {
                            Image(systemName: "checkmark")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundStyle(ExpertLabVisual.good)
                        }
                        Text(String(format: "%02d", index + 1))
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundStyle(isActive ? ExpertLabVisual.accent : isCompleted ? ExpertLabVisual.good : .secondary)
                    }
                    Text(displayStep(step))
                        .font(.system(size: 11, weight: isActive ? .semibold : .regular))
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .foregroundStyle(isActive ? .primary : .secondary)
                }
                .padding(.vertical, 7)
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity, minHeight: 48, alignment: .topLeading)
                .background(isActive ? ExpertLabVisual.accent.opacity(0.10) : Color.clear)
                if index < steps.count - 1 {
                    Rectangle()
                        .fill(isCompleted && index + 1 <= activeIndex ? ExpertLabVisual.good.opacity(0.22) : ExpertLabVisual.line)
                        .frame(width: 1)
                }
            }
        }
        .overlay {
            RoundedRectangle(cornerRadius: 5)
                .stroke(ExpertLabVisual.line, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 5))
    }

    private func displayStep(_ step: String) -> String {
        switch step {
        case "BF16/vMLX Review":
            return "BF16/vMLX\nReview"
        case "Build Review Bundle":
            return "Legacy\nBundle"
        case "Mask/Compare":
            return "Mask/\nCompare"
        case "Prune Plan":
            return "Prune\nPlan"
        case "BF16/F16 Prune":
            return "BF16/F16\nPrune"
        default:
            return step.replacingOccurrences(of: "/", with: "/\n")
        }
    }
}

enum WizardStep: Int, CaseIterable, Identifiable {
    case source = 1, expertReview, pruneReview, profile, run, verify
    var id: Int { rawValue }

    /// Name only — never includes step numbers.
    /// Numbered labels: `WizardCoordinator.displayTitle(for:)`.
    var title: String {
        switch self {
        case .source:       "Source Model"
        case .expertReview: "Expert Review"
        case .pruneReview:  "Prune Review"
        case .profile:      "Conversion Profile"
        case .run:          "Build / Convert"
        case .verify:       "Verify"
        }
    }
}

@Observable
final class WizardCoordinator {
    var plan = ConversionPlan()
    var active: WizardStep = .source

    /// Steps shown in the sidebar for the current plan mode + detection.
    func visibleSteps() -> [WizardStep] {
        switch plan.workflowMode {
        case .convert:
            return [.source, .profile, .run, .verify]
        case .expertLab:
            // Expert Lab mode is only valid for MoE; fall back to Convert list if mis-set.
            guard plan.detected?.isMoE == true else {
                return [.source, .profile, .run, .verify]
            }
            return [.source, .expertReview, .pruneReview, .profile, .run, .verify]
        }
    }

    func displayTitle(for step: WizardStep) -> String {
        let index = (visibleSteps().firstIndex(of: step) ?? 0) + 1
        return "\(index) · \(step.title)"
    }

    /// Invariant: `active` is always an element of `visibleSteps()`.
    func ensureActiveIsVisible() {
        guard !visibleSteps().contains(active) else { return }
        if canActivate(.profile) {
            active = .profile
        } else {
            active = .source
        }
    }

    /// Preferred mode mutation — clears in-progress Expert Lab session when
    /// switching to Convert (keeps adopted pruned-source rails).
    func setWorkflowMode(_ mode: WizardMode) {
        plan.workflowMode = mode
        if mode == .convert {
            if plan.expertReviewIntent == .smartPrequantPrune
                || plan.expertReviewPlanURL != nil
                || plan.expertReviewSourceURL != nil {
                plan.expertReviewIntent = .none
                plan.expertReviewPlanURL = nil
                plan.expertReviewSourceURL = nil
                // Keep expertReviewPrunedSourceURL / original / report if already adopted.
            }
        }
        ensureActiveIsVisible()
    }

    /// Enter Expert Lab review from Source / Expert Review CTAs.
    func enterExpertLabReview() {
        guard plan.detected?.isMoE == true else {
            setWorkflowMode(.convert)
            return
        }
        plan.workflowMode = .expertLab
        plan.expertReviewIntent = .smartPrequantPrune
        plan.expertReviewSourceURL = plan.sourceURL
        plan.expertReviewPlanURL = nil
        ensureActiveIsVisible()
        if canActivate(.expertReview) {
            active = .expertReview
        }
    }

    func canActivate(_ step: WizardStep) -> Bool {
        // Mode-invisible steps are never activatable (defense in depth).
        guard visibleSteps().contains(step) else { return false }

        switch step {
        case .source:       return true
        case .expertReview: return plan.isStep1Complete && plan.detected?.isMoE == true
        case .pruneReview:  return plan.expertReviewPlanURL != nil && plan.expertReviewSourceURL != nil
        case .profile:
            if plan.expertReviewIntent == .smartPrequantPrune,
               plan.expertReviewPrunedSourceURL == nil {
                return false
            }
            return plan.isStep2Complete
        case .run:
            if plan.expertReviewPrunedSourceURL != nil,
               PreflightRunner.reviewedPruneVerifiedCheck(plan: plan)?.status != .pass {
                return false
            }
            return plan.isStep3Complete
        case .verify:       return plan.isStep4Complete
        }
    }
}

struct WizardView: View {
    @State private var coord = WizardCoordinator()
    @Environment(AppSettings.self) private var settings
    @State private var defaultsApplied = false

    var body: some View {
        NavigationSplitView {
            List(coord.visibleSteps(), selection: Binding(
                get: { coord.active },
                // M176 (iter 109): gate sidebar navigation on canActivate.
                // PR3: also ignore steps not in the current mode's visible list.
                set: { newValue in
                    guard let step = newValue else { return }
                    guard coord.visibleSteps().contains(step),
                          coord.canActivate(step) else { return }
                    coord.active = step
                }
            )) { step in
                HStack {
                    Image(systemName: stepIcon(step))
                    Text(coord.displayTitle(for: step))
                }
                .foregroundStyle(coord.canActivate(step) ? .primary : .secondary)
                .tag(step)
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 220, ideal: 240)
        } detail: {
            Group {
            switch coord.active {
            case .source:       SourceStep(coord: coord)
            case .expertReview: ExpertReviewStep(coord: coord)
            case .pruneReview:  PruneReviewStep(coord: coord)
            case .profile:      ProfileStep(coord: coord)
            case .run:          RunStep(coord: coord)
            case .verify:       VerifyStep(coord: coord)
            }
            }
            .background(ExpertLabVisual.canvas)
            .scrollContentBackground(.hidden)
        }
        .tint(ExpertLabVisual.accent)
        .background(ExpertLabVisual.canvas)
        .onChange(of: coord.plan.workflowMode) { _, _ in
            coord.ensureActiveIsVisible()
        }
        .onChange(of: coord.plan.detected?.isMoE) { _, _ in
            coord.ensureActiveIsVisible()
        }
        .task {
            // Apply settings-configured defaults once on first wizard entry.
            // After reset() (VerifyStep → Convert another), we re-apply there.
            guard !defaultsApplied else { return }
            coord.plan.applyDefaults(from: settings)
            defaultsApplied = true
        }
    }

    private func stepIcon(_ s: WizardStep) -> String {
        if !coord.canActivate(s) { return "lock" }
        if s == coord.active { return "circle.fill" }
        switch s {
        case .source:       return coord.plan.isStep1Complete ? "checkmark.circle.fill" : "circle"
        case .expertReview: return coord.plan.expertReviewPlanURL != nil ? "checkmark.circle.fill" : "point.3.connected.trianglepath.dotted"
        case .pruneReview:  return coord.plan.expertReviewPlanURL != nil ? "doc.badge.gearshape" : "circle"
        case .profile:      return coord.plan.isStep3Complete ? "checkmark.circle.fill" : "circle"
        case .run:          return coord.plan.isStep4Complete ? "checkmark.circle.fill" : "circle"
        case .verify:       return "flag.checkered"
        }
    }
}

struct ExpertReviewStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(AppSettings.self) private var settings
    @State private var setupMessage: String?
    @State private var showsAdvancedReviewSettings = false

    private var sourceURL: URL? { coord.plan.expertReviewSourceURL ?? coord.plan.sourceURL }
    private var reviewBundleURL: URL? { coord.plan.expertReviewBundleURL ?? coord.plan.outputURL }

    var body: some View {
        if coord.plan.expertReviewIntent == .smartPrequantPrune, let sourceURL {
            ExpertLabSheet(
                modelPath: sourceURL,
                modelType: coord.plan.detected?.modelType ?? "unknown",
                profile: "BF16/vMLX",
                sizeGb: Double(coord.plan.detected?.totalBytes ?? 0) / 1_000_000_000.0,
                sourceModelPath: sourceURL,
                reviewMode: true,
                showsCloseButton: false,
                presentation: .embeddedInWizard,
                onPrunePlanReady: { planURL in
                    coord.plan.expertReviewPlanURL = planURL
                    coord.active = .pruneReview
                }
            )
        } else {
            Form {
                Section("Expert Review Setup") {
                    ExpertLabConsoleCard {
                        VStack(alignment: .leading, spacing: 12) {
                            ExpertLabKicker(text: "BF16/vMLX source runtime")
                            Label("Analyze Experts Before Pruning", systemImage: "point.3.connected.trianglepath.dotted")
                                .font(.title3.weight(.semibold))
                            Text("Open the original BF16/F16 source in Expert Lab, run prompt-suite routing traces through vMLX, inspect the expert atlas, mask and compare experts, then generate a reviewed BF16/F16 prune plan.")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            ExpertLabWorkflowStrip(
                                steps: ["Load BF16 Source", "Trace", "Atlas", "Mask/Compare", "Prune Plan", "BF16/F16 Prune", "Verify", "Convert"],
                                activeIndex: 0
                            )
                            Label("JANG and JANGTQ stay downstream until the reviewed BF16/F16 prune has passed verification.",
                                  systemImage: "lock.shield")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Runtime Mode") {
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow {
                            Text("BF16/vMLX Source").fontWeight(.medium)
                            Text("Default")
                                .foregroundStyle(.green)
                            Text("Original BF16/F16 weights run through vMLX; router hooks trace selections and apply masks before top-k.")
                                .foregroundStyle(.secondary)
                        }
                        GridRow {
                            Text("Native JANGTQ Review Bundle").fontWeight(.medium)
                            Text("Downstream only")
                                .foregroundStyle(.secondary)
                            Text("Useful for compatibility checks after BF16/F16 prune verification, but not authoritative for pruning.")
                                .foregroundStyle(.secondary)
                        }
                        GridRow {
                            Text("Router-Only Static Analysis").fontWeight(.medium)
                            Text("Blocked for pruning")
                                .foregroundStyle(.orange)
                            Text("Ranks static router weights only; no prompt traces or output comparison, so it cannot authorize reviewed prune.")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .font(.caption)
                    LabeledContent("Source", value: sourceURL?.path ?? "No source selected")
                    LabeledContent("Review runtime", value: "BF16/vMLX")
                    Label("Choose the final JANG/JANGTQ profile only after BF16/F16 hard-prune verification.",
                          systemImage: "arrow.down.forward.and.arrow.up.backward")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Label("Original source directories are immutable; review output and trace artifacts are written separately.", systemImage: "lock.shield")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Legacy Review Settings") {
                    Toggle("Show legacy JANGTQ settings", isOn: $showsAdvancedReviewSettings)
                    if showsAdvancedReviewSettings {
                        Picker("Legacy profile", selection: $coord.plan.expertReviewBundleProfile) {
                            Text("JANGTQ2").tag("JANGTQ2")
                            Text("JANGTQ3").tag("JANGTQ3")
                            Text("JANGTQ4").tag("JANGTQ4")
                        }
                        .pickerStyle(.segmented)
                        .onChange(of: coord.plan.expertReviewBundleProfile) { _, _ in
                            if coord.plan.expertReviewBundleURL == coord.plan.outputURL {
                                coord.plan.outputURL = nil
                            }
                            coord.plan.expertReviewBundleURL = nil
                        }
                        HStack {
                            Text(reviewBundleURL?.path ?? "Default path will be generated")
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button("Choose...", action: pickReviewOutput)
                            Button("Use Default") {
                                coord.plan.expertReviewBundleURL = nil
                                if coord.plan.expertReviewIntent == .smartPrequantPrune {
                                    coord.plan.outputURL = nil
                                }
                            }
                        }
                        Text("These settings are kept for legacy review-bundle compatibility. Reviewed pruning now uses the BF16/F16 source through vMLX.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Legacy Review Bundle Output") {
                    LabeledContent("Output path", value: reviewOutputDescription)
                    LabeledContent("Estimated disk", value: estimatedReviewDiskDescription)
                    LabeledContent("Free disk", value: freeDiskDescription)
                    LabeledContent("Estimated time", value: estimatedReviewTimeDescription)
                    Text("Legacy review bundles are compatibility artifacts. BF16/vMLX Expert Review writes trace and comparison artifacts separately from the immutable source.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if reviewOutputExists {
                        Label("A review output folder already exists. You can resume/retry the build or clean it before starting over.",
                              systemImage: "arrow.clockwise.circle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                        HStack {
                            Button {
                                prepareReviewBundleForBuild()
                                coord.active = .run
                            } label: {
                                Label("Resume / Retry Build", systemImage: "arrow.clockwise")
                            }
                            Button(role: .destructive) {
                                cleanReviewOutput()
                            } label: {
                                Label("Clean Review Output", systemImage: "trash")
                            }
                        }
                    }
                    if let setupMessage {
                        Text(setupMessage)
                            .font(.caption)
                            .foregroundStyle(setupMessage.contains("failed") ? .red : .secondary)
                            .textSelection(.enabled)
                    }
                }

                Section("Prompt Suite") {
                    Text("Default probing uses the Reviewed Prune 50 suite across instruction following, coding, math, agentic reasoning, multilingual, robustness, long-context, structured output, and domain tasks. Balanced 150, Deep 500, Smoke 21, and custom JSONL imports are available inside Expert Lab.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section {
                    Button {
                        coord.enterExpertLabReview()
                    } label: {
                        Label("Open BF16 Expert Review", systemImage: "point.3.connected.trianglepath.dotted")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(sourceURL == nil)

                    Button {
                        coord.plan.outputURL = nil
                        coord.plan.applyDefaults(from: settings)
                        coord.setWorkflowMode(.convert)
                        if coord.canActivate(.profile) {
                            coord.active = .profile
                        }
                    } label: {
                        Label("Direct Convert Without Pruning", systemImage: "arrow.right.circle")
                    }
                }
            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
            .background(ExpertLabVisual.canvas)
            .padding()
        }
    }

    private var estimatedReviewDiskBytes: Int64 {
        guard let total = coord.plan.detected?.totalBytes else { return 0 }
        return max(Int64(Double(total) * 0.45), 1_000_000_000)
    }

    private var estimatedReviewDiskDescription: String {
        guard estimatedReviewDiskBytes > 0 else { return "unknown until source detection completes" }
        return "about \(formatGB(estimatedReviewDiskBytes))"
    }

    private var estimatedReviewTimeDescription: String {
        guard let detected = coord.plan.detected else { return "unknown" }
        let sourceGB = Double(detected.totalBytes) / 1_000_000_000.0
        if sourceGB >= 150 { return "tens of minutes on a large MoE source" }
        if sourceGB >= 40 { return "several minutes on this source" }
        return "usually a few minutes or less"
    }

    private var reviewOutputDescription: String {
        if let output = reviewBundleURL {
            return output.path
        }
        if let sourceURL {
            return suggestedReviewOutputURL(for: sourceURL).path + " (suggested)"
        }
        return "Choose a source first"
    }

    private var freeDiskDescription: String {
        guard let url = reviewBundleURL ?? sourceURL?.deletingLastPathComponent() else { return "unknown" }
        guard let free = freeBytes(onVolumeFor: url) else { return "unknown" }
        let need = estimatedReviewDiskBytes
        if need > 0, free < need {
            return "\(formatGB(free)) free; below estimated \(formatGB(need))"
        }
        if need > 0 {
            return "\(formatGB(free)) free; enough for estimated \(formatGB(need))"
        }
        return "\(formatGB(free)) free"
    }

    private var reviewOutputExists: Bool {
        guard let output = reviewBundleURL else { return false }
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: output.path, isDirectory: &isDir)
            && isDir.boolValue
    }

    private func prepareReviewBundleForBuild(resetOutput: Bool = false) {
        coord.plan.expertReviewIntent = .smartPrequantPrune
        coord.plan.expertReviewSourceURL = coord.plan.sourceURL
        coord.plan.expertReviewPlanURL = nil
        coord.plan.family = .jangtq
        if !coord.plan.expertReviewBundleProfile.hasPrefix("JANGTQ") {
            coord.plan.expertReviewBundleProfile = "JANGTQ3"
        }
        coord.plan.profile = coord.plan.expertReviewBundleProfile
        coord.plan.hadamard = false
        if resetOutput {
            coord.plan.expertReviewBundleURL = nil
            coord.plan.outputURL = nil
        }
        if coord.plan.expertReviewBundleURL == nil, let sourceURL {
            coord.plan.expertReviewBundleURL = suggestedReviewOutputURL(for: sourceURL)
        }
        coord.plan.outputURL = coord.plan.expertReviewBundleURL
        coord.plan.run = .idle
    }

    private func cleanReviewOutput() {
        guard let output = reviewBundleURL else { return }
        guard output != sourceURL else {
            setupMessage = "Clean failed: review output path matches the source path."
            return
        }
        do {
            try FileManager.default.removeItem(at: output)
            if coord.plan.outputURL == output {
                coord.plan.outputURL = nil
            }
            coord.plan.expertReviewBundleURL = nil
            coord.plan.run = .idle
            coord.plan.expertReviewPlanURL = nil
            setupMessage = "Cleaned review output at \(output.path)."
        } catch CocoaError.fileNoSuchFile {
            if coord.plan.outputURL == output {
                coord.plan.outputURL = nil
            }
            coord.plan.expertReviewBundleURL = nil
            coord.plan.run = .idle
            setupMessage = "Review output was already gone."
        } catch {
            setupMessage = "Clean failed: \(error.localizedDescription)"
        }
    }

    private func suggestedReviewOutputURL(for source: URL) -> URL {
        let profile = coord.plan.expertReviewBundleProfile.hasPrefix("JANGTQ")
            ? coord.plan.expertReviewBundleProfile
            : "JANGTQ3"
        return source
            .deletingLastPathComponent()
            .appendingPathComponent("\(source.lastPathComponent)-\(profile)-review")
    }

    private func pickReviewOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK {
            coord.plan.expertReviewBundleURL = panel.url
            if coord.plan.expertReviewIntent == .smartPrequantPrune {
                coord.plan.outputURL = panel.url
            }
        }
    }

    private func freeBytes(onVolumeFor url: URL) -> Int64? {
        let probe = url.hasDirectoryPath ? url : url.deletingLastPathComponent()
        if let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let free = values.volumeAvailableCapacityForImportantUsage {
            return Int64(free)
        }
        if let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityKey]),
           let free = values.volumeAvailableCapacity {
            return Int64(free)
        }
        return nil
    }

    private func formatGB(_ bytes: Int64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_000_000_000.0)
    }
}

struct PruneReviewStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(AppSettings.self) private var settings
    @State private var detectingPrunedSource = false
    @State private var errorText: String?

    var body: some View {
        Group {
            if let source = coord.plan.expertReviewSourceURL,
               let detected = coord.plan.detected,
               let planURL = coord.plan.expertReviewPlanURL {
                PrequantPruneSheet(
                    sourceURL: source,
                    detected: detected,
                    initialPrunePlanURL: planURL,
                    showsCloseButton: false
                ) { prunedURL in
                    adoptPrunedSource(prunedURL)
                }
            } else {
                ContentUnavailableView {
                    Label("No reviewed prune plan", systemImage: "doc.badge.gearshape")
                } description: {
                    Text("Run Expert Review, inspect the atlas, and generate a prune plan before hard-pruning a BF16/F16 source.")
                } actions: {
                    Button("Back to Expert Review") { coord.active = .expertReview }
                }
            }
        }
        .overlay(alignment: .bottomLeading) {
            if detectingPrunedSource {
                Label("Inspecting pruned source...", systemImage: "magnifyingglass")
                    .padding(8)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding()
            } else if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .padding(8)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding()
            }
        }
    }

    private func adoptPrunedSource(_ prunedURL: URL) {
        coord.plan.adoptReviewedPrunedSource(prunedURL)
        coord.plan.applyDefaults(from: settings)
        coord.plan.outputURL = nil
        coord.ensureActiveIsVisible()
        detectingPrunedSource = true
        errorText = nil
        Task {
            do {
                let detected = try await SourceDetector.inspect(url: prunedURL)
                await MainActor.run {
                    coord.plan.detected = detected
                    detectingPrunedSource = false
                    coord.ensureActiveIsVisible()
                    if coord.canActivate(.profile) {
                        coord.active = .profile
                    }
                }
            } catch {
                await MainActor.run {
                    detectingPrunedSource = false
                    errorText = "Pruned source was written, but inspection failed: \(error.localizedDescription)"
                    coord.ensureActiveIsVisible()
                    if coord.canActivate(.profile) {
                        coord.active = .profile
                    }
                }
            }
        }
    }
}
