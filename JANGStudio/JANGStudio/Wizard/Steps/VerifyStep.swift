// JANGStudio/JANGStudio/Wizard/Steps/VerifyStep.swift
import SwiftUI

struct VerifyStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(CapabilitiesService.self) private var capsSvc
    @Environment(AppSettings.self) private var settings
    @State private var checks: [VerifyCheck] = []
    @State private var busy = true
    @State private var showingInference = false
    @State private var showingExpertLab = false
    @State private var showingExamples = false
    @State private var showingModelCard = false
    @State private var showingPublish = false
    @State private var revealFiredOnce = false
    @State private var reviewAutoOpened = false

    var body: some View {
        Form {
            Section("Output verification") {
                if busy { ProgressView() }
                ForEach(checks) { c in
                    HStack {
                        Image(systemName: icon(c.status)).foregroundStyle(color(c.status))
                        Text(c.title)
                        if !c.required { Text("(warn)").font(.caption).foregroundStyle(.secondary) }
                        if let h = c.hint { Text(h).font(.caption).foregroundStyle(.secondary) }
                    }
                }
            }
            if !busy, isFinalQuantFromReviewedPrune {
                Section("Post-Quant Expert Lab") {
                    Label(postQuantExpertLabSummary,
                          systemImage: postQuantExpertLabPassed ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                        .font(.headline)
                        .foregroundStyle(postQuantExpertLabPassed ? .green : .orange)
                    if let finalComparison = expertLabFinalComparisonCheck,
                       finalComparison.status != .pass {
                        Text(finalComparison.hint ?? finalComparison.title)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    } else if let promptCheck = postQuantPromptCheck {
                        Text(promptCheck.hint ?? promptCheck.title)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    if let report = expertLabFinalReportCheck?.hint {
                        LabeledContent("Final report", value: report)
                            .font(.caption)
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: report)])
                        } label: {
                            Label("Reveal Expert Lab Final Report", systemImage: "doc.text.magnifyingglass")
                        }
                    }
                }
            }
            if !busy, finishable() {
                if coord.plan.expertReviewIntent == .smartPrequantPrune {
                    Section("Expert Review Before Pruning") {
                        Text("Expert Review uses the original BF16/F16 source through vMLX as the pruning authority. Run prompt suites, map expert activity, mask and compare outputs, then send the reviewed plan back to BF16/F16 source pruning.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let planURL = coord.plan.expertReviewPlanURL {
                            Label("Review plan ready for BF16/F16 pruning.", systemImage: "checkmark.seal.fill")
                                .foregroundStyle(.green)
                            Text(planURL.path)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .truncationMode(.middle)
                            Button {
                                coord.active = .pruneReview
                            } label: {
                                Label("Open Reviewed BF16/F16 Prune", systemImage: "scissors")
                            }
                            .buttonStyle(.borderedProminent)
                        } else {
                            Button {
                                coord.active = .expertReview
                            } label: {
                                Label("Run Prompt Review", systemImage: "waveform.path.ecg")
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                }

                Section {
                    if let url = coord.plan.outputURL {
                        LabeledContent("Ready at", value: url.path)
                    }

                    // Adoption actions row
                    HStack {
                        Button {
                            showingInference = true
                        } label: {
                            Label("Test Inference", systemImage: "bubble.left.and.bubble.right")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            if coord.plan.expertReviewIntent == .smartPrequantPrune {
                                coord.active = .expertReview
                            } else {
                                showingExpertLab = true
                            }
                        } label: {
                            Label(coord.plan.expertReviewIntent == .smartPrequantPrune ? "Expert Review" : "Expert Lab",
                                  systemImage: "point.3.connected.trianglepath.dotted")
                        }
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            showingExamples = true
                        } label: {
                            Label("View Usage Examples", systemImage: "doc.text")
                        }
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            showingModelCard = true
                        } label: {
                            Label("Generate Model Card", systemImage: "doc.richtext")
                        }
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            showingPublish = true
                        } label: {
                            Label("Publish to HF", systemImage: "arrow.up.doc.on.clipboard")
                        }
                        .disabled(coord.plan.outputURL == nil)
                    }

                    HStack {
                        Button {
                            revealOutput()
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                        }
                        Button {
                            copyPath()
                        } label: {
                            Label("Copy Path", systemImage: "doc.on.doc")
                        }
                    }

                    HStack {
                        Button("Convert another") { reset() }
                        Spacer()
                        Button("Finish") { finishApp() }
                    }
                }
            } else if !busy {
                Section {
                    Label("Output incomplete — cannot finish.", systemImage: "xmark.octagon.fill")
                        .foregroundStyle(.red)
                    Button("Retry conversion") { coord.active = .run }
                }
            }
        }
        .formStyle(.grouped).padding()
        // `.task` ties the refresh to the VIEW lifecycle — SwiftUI auto-cancels
        // it on dismount, so navigating away mid-verify doesn't leave an
        // orphan Python subprocess running. Prior to iter 19 (M42) this was
        // `.onAppear { Task { await refresh() } }` which spawned a detached
        // Task unbound to the view lifecycle.
        .task { await refresh() }
        .sheet(isPresented: $showingInference) {
            if let url = coord.plan.outputURL {
                TestInferenceSheet(
                    modelPath: url,
                    isVL: coord.plan.detected?.isVL ?? false,
                    isVideoVL: coord.plan.detected?.isVideoVL ?? false,
                    modelType: coord.plan.detected?.modelType ?? "unknown",
                    profile: coord.plan.profile,
                    sizeGb: Double(coord.plan.detected?.totalBytes ?? 0) / 1_000_000_000.0
                )
            }
        }
        .sheet(isPresented: standaloneExpertLabSheetBinding) {
            if coord.plan.expertReviewIntent != .smartPrequantPrune,
               let url = coord.plan.outputURL {
                ExpertLabSheet(
                    modelPath: url,
                    modelType: coord.plan.detected?.modelType ?? "unknown",
                    profile: coord.plan.profile,
                    sizeGb: Double(coord.plan.detected?.totalBytes ?? 0) / 1_000_000_000.0,
                    sourceModelPath: coord.plan.expertReviewSourceURL,
                    reviewMode: false
                )
            }
        }
        .sheet(isPresented: $showingExamples) {
            if let url = coord.plan.outputURL {
                UsageExamplesSheet(modelPath: url)
            }
        }
        .sheet(isPresented: $showingModelCard) {
            if let url = coord.plan.outputURL {
                GenerateModelCardSheet(modelPath: url)
            }
        }
        .sheet(isPresented: $showingPublish) {
            if let url = coord.plan.outputURL {
                PublishToHuggingFaceSheet(modelPath: url)
            }
        }
    }

    private var standaloneExpertLabSheetBinding: Binding<Bool> {
        Binding(
            get: {
                showingExpertLab && coord.plan.expertReviewIntent != .smartPrequantPrune
            },
            set: { newValue in
                showingExpertLab = newValue
            }
        )
    }

    private var isFinalQuantFromReviewedPrune: Bool {
        coord.plan.expertReviewPrunedSourceURL != nil
    }

    private var postQuantPromptCheck: VerifyCheck? {
        checks.first { $0.id == .expertLabNativeSmoke }
    }

    private var expertLabFinalReportCheck: VerifyCheck? {
        checks.first { $0.id == .expertLabFinalReport }
    }

    private var expertLabFinalComparisonCheck: VerifyCheck? {
        checks.first { $0.id == .expertLabFinalComparison }
    }

    private var postQuantExpertLabPassed: Bool {
        postQuantPromptCheck?.status == .pass && expertLabFinalComparisonCheck?.status == .pass
    }

    private var postQuantExpertLabSummary: String {
        guard let promptCheck = postQuantPromptCheck else {
            return "Post-quant same-suite prompt verification has not run yet."
        }
        if promptCheck.status == .fail {
            return "Post-quant same-suite prompt verification failed."
        }
        if promptCheck.status == .warn {
            return "Post-quant same-suite prompt verification needs review."
        }
        guard let finalComparison = expertLabFinalComparisonCheck else {
            return "Post-quant final same-suite comparison has not run yet."
        }
        switch finalComparison.status {
        case .pass:
            return "Post-quant same-suite prompt and pruned BF16 comparison passed."
        case .warn:
            return "Post-quant final same-suite comparison needs review."
        case .fail:
            return "Post-quant final same-suite comparison failed."
        }
    }

    private func refresh() async {
        busy = true
        let c = await PostConvertVerifier().run(plan: coord.plan, capabilities: capsSvc.capabilities)
        await MainActor.run {
            checks = c
            busy = false
            // M62: auto-reveal the output in Finder once, on first successful
            // finishable render. Respects Settings → General → Behavior →
            // "Reveal in Finder on finish". Fires once per VerifyStep lifetime
            // so re-opening the tab doesn't keep popping Finder.
            if settings.revealInFinderOnFinish, !revealFiredOnce, finishable() {
                revealFiredOnce = true
                revealOutput()
            }
            if coord.plan.expertReviewIntent == .smartPrequantPrune,
               coord.plan.expertReviewPlanURL == nil,
               !reviewAutoOpened,
               finishable() {
                reviewAutoOpened = true
                coord.active = .expertReview
            }
        }
    }

    private func finishable() -> Bool { !checks.contains { $0.required && $0.status == .fail } }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
    private func revealOutput() {
        if let url = coord.plan.outputURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }
    private func copyPath() {
        if let p = coord.plan.outputURL?.path {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(p, forType: .string)
        }
    }
    private func reset() {
        // "Convert another" — clear state + re-apply user defaults so the
        // fresh plan still reflects Settings → General → Defaults. Before M62
        // this dropped back to hardcoded JANG_4K/jang/mse regardless of
        // user settings.
        let newPlan = ConversionPlan()
        newPlan.applyDefaults(from: settings)
        // Fresh plan defaults to workflowMode = .convert.
        coord.plan = newPlan
        coord.active = .source
        coord.ensureActiveIsVisible()
        revealFiredOnce = false
        reviewAutoOpened = false
    }
    private func finishApp() { NSApp.windows.first?.close() }
}
