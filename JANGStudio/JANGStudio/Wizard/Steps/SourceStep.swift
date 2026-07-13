// JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
import SwiftUI

enum SourceStepExpertPruneSupport {
    private static let qwenRawMoEModelTypes: Set<String> = [
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    ]

    static func supportsRawQwenPrequantPrune(_ detected: ArchitectureSummary) -> Bool {
        return detected.isMoE
            && detected.architectureModelTypes.contains { qwenRawMoEModelTypes.contains($0) }
            && [.bf16, .fp16].contains(detected.dtype)
    }
}

struct SourceStep: View {
    @Bindable var coord: WizardCoordinator
    // M143 (iter 65): access AppSettings so applyRecommendation can tell
    // whether the current profile is still at the user's configured
    // default (safe to overwrite) vs. a value the user manually picked in
    // ProfileStep (must preserve). Pre-iter-65 the check was hardcoded
    // `== "JANG_4K"` which misfired for any user whose
    // settings.defaultProfile wasn't JANG_4K.
    @Environment(AppSettings.self) private var settings
    @State private var isDetecting = false
    @State private var isRecommending = false
    @State private var errorText: String?
    @State private var recommendation: Recommendation?
    @State private var showingPrequantPrune = false
    @State private var autoOpenedAttachedPlan = false
    /// M135 (iter 57): stale-task handle tracking. User picks folder A →
    /// detection starts (Task A, ~5s) → user changes mind, picks folder B →
    /// detection starts (Task B, ~1s). Without this handle, Task A continues
    /// running after Task B finishes and eventually stomps
    /// `coord.plan.detected` / `self.recommendation` with folder A's result.
    /// The user sees A's metadata while sourceURL points at B — the
    /// conversion later uses wrong metadata → misdetected architecture →
    /// wrong quantization profile applied. Subprocess kill propagates
    /// through SourceDetector's iter-34 M105 withTaskCancellationHandler
    /// wrap and RecommendationService's iter-33 M101 wrap.
    @State private var detectionTask: Task<Void, Never>?

    var body: some View {
        Form {
            // MARK: - Folder picker
            Section {
                HStack {
                    if let url = coord.plan.sourceURL {
                        Text(url.lastPathComponent).font(.headline)
                        Text(url.deletingLastPathComponent().path)
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("No folder selected").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Choose Folder…", action: pickFolder)
                }
                // M206 (iter 140): always-visible cold-start guidance when
                // no folder is picked. Pre-M206 a first-time user saw only
                // "No folder selected" + "Choose Folder…" — the explanation
                // of WHAT to pick lived exclusively in the header InfoHint
                // (hover-only). A stranger who doesn't discover the (i)
                // hover icon gets zero instruction. Now visible without
                // interaction: what the folder should contain, a concrete
                // example path, and a link to huggingface.co for users
                // who don't have a model locally.
                if coord.plan.sourceURL == nil {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("A HuggingFace model directory contains `config.json` and one or more `.safetensors` files.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Example: `~/Downloads/Qwen3-0.6B-Base/`")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let hfURL = URL(string: "https://huggingface.co/models?sort=downloads") {
                            Link("Don't have one yet? Browse models on HuggingFace →",
                                 destination: hfURL)
                                .font(.caption)
                        }
                    }
                    .padding(.top, 4)
                }
            } header: {
                HStack(spacing: 4) {
                    Text("Source model folder")
                    InfoHint("Pick a HuggingFace model directory — one containing `config.json` and `.safetensors` shards. JANG Studio auto-detects the architecture and recommends a conversion plan.")
                }
            }

            // MARK: - Detected
            if let detected = coord.plan.detected {
                Section("Detected") {
                    LabeledContent("Model type", value: detected.modelType)
                    if let textModelType = detected.textModelType,
                       textModelType != detected.modelType {
                        LabeledContent("Text model type", value: textModelType)
                    }
                    LabeledContent(
                        "Parameters",
                        value: detected.parameterSummary
                    )
                    LabeledContent("Source dtype", value: detected.dtype.rawValue.uppercased())
                    LabeledContent(
                        "Disk",
                        value: "\(detected.totalBytes / 1_000_000_000) GB (\(detected.shardCount) shards)"
                    )
                    if detected.isVideoVL {
                        Label("Vision + Video", systemImage: "film")
                            .foregroundStyle(.purple)
                    } else if detected.isVL {
                        Label("Vision (image)", systemImage: "eye")
                            .foregroundStyle(.blue)
                    }
                    // No safetensors → hard-fail with a specific hint.
                    if detected.shardCount == 0 {
                        Label(
                            "No .safetensors files found in this folder. You likely picked a parent folder, a docs folder, or a download that didn't complete. Pick the actual model directory.",
                            systemImage: "xmark.octagon.fill"
                        )
                        .foregroundStyle(.red)
                        .font(.callout)
                    }
                }

                Section {
                    if detected.isMoE {
                        // PR3: required segmented control — Convert is default primary path.
                        Picker("Workflow", selection: Binding(
                            get: { coord.plan.workflowMode },
                            set: { newMode in
                                if newMode == .expertLab {
                                    // Flip mode without navigating; user picks CTA.
                                    coord.plan.workflowMode = .expertLab
                                    coord.ensureActiveIsVisible()
                                } else {
                                    coord.setWorkflowMode(.convert)
                                }
                            }
                        )) {
                            Text("Convert").tag(WizardMode.convert)
                            Text("Expert Lab").tag(WizardMode.expertLab)
                        }
                        .pickerStyle(.segmented)

                        if coord.plan.workflowMode == .convert {
                            Text("Profile → quantize → verify. Skip expert prune.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Button {
                                goDirectConvert()
                            } label: {
                                Label("Continue to Profile", systemImage: "arrow.right.circle")
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.large)

                            Button {
                                coord.enterExpertLabReview()
                            } label: {
                                Label("Analyze Experts Before Pruning", systemImage: "point.3.connected.trianglepath.dotted")
                            }
                            .disabled(!canPrepareExpertLabBundle(detected))
                        } else {
                            ExpertLabConsoleCard {
                                VStack(alignment: .leading, spacing: 12) {
                                    ExpertLabKicker(text: "Expert Lab path")
                                    Label("Analyze experts before pruning", systemImage: "point.3.connected.trianglepath.dotted")
                                        .font(.title3.weight(.semibold))
                                    Text("Trace prompt-suite routing, inspect the expert atlas, mask and compare behavior, then generate a reviewed BF16/F16 prune plan before final conversion.")
                                        .font(.callout)
                                        .foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                    ExpertLabWorkflowStrip(
                                        steps: ["Trace", "Atlas", "Mask/Compare", "Prune Plan", "BF16/F16 Prune", "Verify", "Convert"],
                                        activeIndex: 0
                                    )
                                    HStack(spacing: 8) {
                                        Label(
                                            detected.routedExpertTotal.map { "\($0) routed slots" } ?? "\(detected.numExperts) experts",
                                            systemImage: "square.grid.3x3"
                                        )
                                        Label("source stays immutable", systemImage: "lock.shield")
                                        Label("prune before quantize", systemImage: "scissors")
                                    }
                                    .font(.caption)
                                    .foregroundStyle(.secondary)

                                    Button {
                                        coord.enterExpertLabReview()
                                    } label: {
                                        Label("Analyze Experts Before Pruning", systemImage: "point.3.connected.trianglepath.dotted")
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .controlSize(.large)
                                    .disabled(!canPrepareExpertLabBundle(detected))

                                    Text("Runs the original BF16/F16 source through vMLX in Expert Lab, lets you disable experts and compare outputs, then returns a smart keep/drop plan for BF16/F16 source pruning.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }

                            Button {
                                goDirectConvert()
                            } label: {
                                Label("Direct Convert Without Pruning", systemImage: "arrow.right.circle")
                            }
                        }

                        if let attachedReviewPlanURL {
                            Label("Smart expert review plan ready. The next prune will use prompt-suite trace evidence and any experts you disabled during review.",
                                  systemImage: "checkmark.seal.fill")
                                .font(.caption)
                                .foregroundStyle(.green)
                            Text(attachedReviewPlanURL.path)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .truncationMode(.middle)
                        }

                        // Router-only prune stays under Convert (non-reviewed escape hatch).
                        if coord.plan.workflowMode == .convert || attachedReviewPlanURL != nil {
                            VStack(alignment: .leading, spacing: 8) {
                                Button {
                                    if attachedReviewPlanURL != nil {
                                        coord.plan.workflowMode = .expertLab
                                        coord.ensureActiveIsVisible()
                                        if coord.canActivate(.pruneReview) {
                                            coord.active = .pruneReview
                                        }
                                    } else {
                                        showingPrequantPrune = true
                                    }
                                } label: {
                                    Label(attachedReviewPlanURL == nil ? "Router-Only Prune..." : "Open Reviewed Prune Plan",
                                          systemImage: "scissors")
                                }
                                .disabled(!canPrequantPrune(detected))
                                .fixedSize(horizontal: false, vertical: true)

                                if canPrequantPrune(detected), attachedReviewPlanURL == nil {
                                    Label("Fallback uses router-row strength only; it does not probe prompts. It cannot unlock final quantization.",
                                          systemImage: "exclamationmark.triangle.fill")
                                        .font(.caption)
                                        .foregroundStyle(ExpertLabVisual.warm)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }

                        if !canPrequantPrune(detected) {
                            Label("Pre-quant source pruning is currently wired for qwen3_5_moe/qwen3_5_moe_text raw MoE sources. You can still inspect Expert Lab traces, but hard pruning should wait for a supported source-prune adapter.",
                                  systemImage: "info.circle")
                                .font(.caption)
                        }
                    } else {
                        Label("No routed experts were detected, so Expert Lab is not available. Continue with Convert: Profile → Run → Verify.",
                              systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text(detected.isMoE ? "Workflow" : "Expert Lab & pruning")
                }
            } else if isDetecting {
                Section("Detected") {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Inspecting source…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // MARK: - Recommendation (beginner-friendly)
            if let rec = recommendation {
                Section {
                    Text(rec.beginnerSummary)
                        .font(.callout)
                        .padding(.vertical, 4)

                    Divider()

                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow {
                            Text("Family").foregroundStyle(.secondary)
                            Text(rec.recommended.family.uppercased())
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.family)
                        }
                        GridRow {
                            Text("Profile").foregroundStyle(.secondary)
                            Text(rec.recommended.profile)
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.profile)
                        }
                        GridRow {
                            Text("Method").foregroundStyle(.secondary)
                            Text(rec.recommended.method.uppercased())
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.method)
                        }
                        GridRow {
                            Text("Hadamard").foregroundStyle(.secondary)
                            Text(rec.recommended.hadamard ? "On" : "Off")
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.hadamard)
                        }
                        if let forceDtype = rec.recommended.forceDtype {
                            GridRow {
                                Text("Force dtype").foregroundStyle(.secondary)
                                Text(forceDtype)
                                    .fontWeight(.medium)
                                InfoHint(rec.whyEachChoice.forceDtype)
                            }
                        }
                    }

                    ForEach(rec.warnings, id: \.self) { warning in
                        Label(warning, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                            .padding(.top, 2)
                    }

                    if !rec.recommended.alternatives.isEmpty {
                        DisclosureGroup("Other options") {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(rec.recommended.alternatives) { alt in
                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack(spacing: 4) {
                                            if let fam = alt.family {
                                                Text(fam.uppercased())
                                                    .font(.caption2)
                                                    .padding(.horizontal, 5)
                                                    .padding(.vertical, 2)
                                                    .background(Color.secondary.opacity(0.2))
                                                    .cornerRadius(4)
                                            }
                                            Text(alt.profile).fontWeight(.medium)
                                        }
                                        Text(alt.useWhen)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                } header: {
                    HStack(spacing: 4) {
                        Text("Recommended for this model")
                        InfoHint("Smart defaults auto-filled based on what JANG Studio detected. Most beginners can skip Steps 2 and 3 and hit Start in Step 4.")
                    }
                }
            } else if isRecommending {
                Section {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Building recommendation…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    HStack(spacing: 4) {
                        Text("Recommended for this model")
                        InfoHint("Smart defaults auto-filled based on what JANG Studio detected. Most beginners can skip Steps 2 and 3 and hit Start in Step 4.")
                    }
                }
            }

            // MARK: - Detection status
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }

            // MARK: - Continue
            if coord.plan.isStep1Complete {
                Button("Continue →") {
                    // PR3: honor workflow mode. Convert → Profile; Expert Lab → review.
                    if coord.plan.workflowMode == .expertLab,
                       coord.plan.detected?.isMoE == true {
                        coord.enterExpertLabReview()
                    } else {
                        goDirectConvert()
                    }
                }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(ExpertLabVisual.canvas)
        .padding()
        .sheet(isPresented: $showingPrequantPrune) {
            if let url = coord.plan.sourceURL, let detected = coord.plan.detected {
                PrequantPruneSheet(
                    sourceURL: url,
                    detected: detected,
                    initialPrunePlanURL: attachedReviewPlanURL
                ) { prunedURL in
                    if attachedReviewPlanURL != nil {
                        adoptReviewedPrunedSource(url: prunedURL)
                    } else {
                        adoptSource(url: prunedURL)
                    }
                }
            }
        }
        .task(id: coord.plan.expertReviewPlanURL) {
            if attachedReviewPlanURL != nil, !autoOpenedAttachedPlan {
                autoOpenedAttachedPlan = true
                coord.active = .pruneReview
            }
        }
        .onDisappear {
            // M171 (iter 94): cancel the detection Task when SourceStep
            // unmounts. SwiftUI fires .onDisappear on sidebar-jump, window
            // close, and app quit (cmd-Q). Pre-M171 the Python inspect-
            // source subprocess would keep running for a few seconds after
            // the user moved away. iter-57 M135's cancel-on-new-pickFolder
            // handles the concurrent-pick case within a live SourceStep
            // instance; iter-84 M161's URL-match guard handles orphan
            // state-corruption; this hook closes the last gap (subprocess
            // teardown on view destruction). Cancel propagates through
            // iter-34 M105's SourceDetector + iter-76 M153's PythonCLIInvoker
            // withTaskCancellationHandler → SIGTERM subprocess.
            detectionTask?.cancel()
        }
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            detectionTask?.cancel()
            adoptSource(url: url)
        }
    }

    private func adoptSource(url: URL) {
        coord.plan.sourceURL = url
        coord.plan.detected = nil
        coord.plan.outputURL = nil
        coord.plan.run = .idle
        coord.plan.workflowMode = .convert
        coord.plan.expertReviewIntent = .none
        coord.plan.expertReviewSourceURL = nil
        coord.plan.expertReviewPlanURL = nil
        coord.plan.expertReviewOriginalSourceURL = nil
        coord.plan.expertReviewPrunedSourceURL = nil
        coord.plan.expertReviewPrunePlanURL = nil
        coord.plan.expertReviewPruneReportURL = nil
        autoOpenedAttachedPlan = false
        recommendation = nil
        errorText = nil
        coord.ensureActiveIsVisible()
        // M135 (iter 57): cancel any previous detection task before
        // starting a new one. Without this, a slow previous detection
        // can stomp the new URL's state after it finishes.
        detectionTask?.cancel()
        detectionTask = Task { await detectAndRecommend(url: url) }
    }

    private func adoptReviewedPrunedSource(url: URL) {
        coord.plan.adoptReviewedPrunedSource(url)
        coord.ensureActiveIsVisible()
        autoOpenedAttachedPlan = false
        recommendation = nil
        errorText = nil
        detectionTask?.cancel()
        detectionTask = Task { await detectAndRecommend(url: url) }
    }

    private func goDirectConvert() {
        coord.setWorkflowMode(.convert)
        if coord.canActivate(.profile) {
            coord.active = .profile
        }
    }

    private func canPrequantPrune(_ detected: ArchitectureSummary) -> Bool {
        SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(detected)
    }

    private func canPrepareExpertLabBundle(_ detected: ArchitectureSummary) -> Bool {
        SourceStepExpertPruneSupport.supportsRawQwenPrequantPrune(detected)
    }

    private var attachedReviewPlanURL: URL? {
        guard coord.plan.expertReviewIntent == .smartPrequantPrune,
              coord.plan.expertReviewSourceURL == coord.plan.sourceURL else {
            return nil
        }
        return coord.plan.expertReviewPlanURL
    }

    private func startExpertReview() {
        autoOpenedAttachedPlan = false
        coord.enterExpertLabReview()
    }

    private func prepareExpertLabBundle() {
        coord.plan.family = .jangtq
        if !coord.plan.profile.hasPrefix("JANGTQ") {
            coord.plan.profile = "JANGTQ3"
        }
        coord.plan.hadamard = false
        coord.plan.outputURL = nil
        coord.plan.run = .idle
        coord.active = .profile
    }

    private func detectAndRecommend(url: URL) async {
        // Step A: fast inspect-source call
        isDetecting = true
        do {
            let detected = try await SourceDetector.inspect(url: url)
            // M135 (iter 57): guard against stale-task overwrite when the task
            // was explicitly cancelled (user picked a different folder and
            // `pickFolder` called `detectionTask?.cancel()`).
            guard !Task.isCancelled else { return }
            await MainActor.run {
                // M161 (iter 84): second-line guard for ORPHANED tasks —
                // ones that survived view destruction. `detectionTask` is
                // @State private to SourceStep, so if the user sidebar-jumps
                // away from Source and back (creating a fresh SourceStep
                // instance), the NEW view's `detectionTask` is nil and
                // can't cancel the OLD view's task. The old task then
                // keeps running, Task.isCancelled stays false, and it
                // overwrites `coord.plan.detected` with the OLD folder's
                // result — silently corrupting the conversion plan when
                // the user has since picked a new folder. The URL-match
                // check is the authoritative "is this still relevant?"
                // signal: if sourceURL has moved on, this detection's
                // result is stale regardless of cancel state.
                guard coord.plan.sourceURL == url else { return }
                coord.plan.detected = detected
                // PR3: dense models force Convert; MoE keeps current mode
                // (already .convert after adoptSource).
                if !detected.isMoE {
                    coord.plan.workflowMode = .convert
                }
                coord.ensureActiveIsVisible()
            }
        } catch {
            await MainActor.run {
                // Also guard the error-path — a cancelled task's subprocess
                // kill shouldn't surface as a user-facing "Detection failed".
                guard !Task.isCancelled else { return }
                // M161 (iter 84): orphaned-task guard on the error path too.
                // Without it, an old folder's detection failure would flash
                // as an errorText banner against a NEW folder the user just
                // picked — user sees "Detection failed: …" while looking at
                // a model that is actually fine.
                guard coord.plan.sourceURL == url else { return }
                errorText = "Detection failed: \(error.localizedDescription)"
                isDetecting = false
            }
            return
        }
        guard !Task.isCancelled else { return }
        await MainActor.run {
            // M161 (iter 84): the isDetecting indicator belongs to the URL
            // whose task is finishing. If sourceURL has moved on, the NEW
            // url has its own isDetecting=true in flight; don't stomp it.
            guard coord.plan.sourceURL == url else { return }
            isDetecting = false
        }

        // Step B: recommendation call (also fast, reads same config.json)
        isRecommending = true
        defer { Task { @MainActor in isRecommending = false } }
        do {
            let rec = try await RecommendationService.fetch(modelURL: url)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                // M161 (iter 84): same orphaned-task guard for recommendation.
                // A stale recommendation write would set self.recommendation
                // to the OLD folder's suggestions AND call applyRecommendation
                // which mutates plan.profile/family/method — a direct user-
                // visible data-corruption vector.
                guard coord.plan.sourceURL == url else { return }
                self.recommendation = rec
                self.applyRecommendation(rec)
            }
        } catch {
            await MainActor.run {
                guard !Task.isCancelled else { return }
                // M161 (iter 84): symmetric guard on recommendation error.
                guard coord.plan.sourceURL == url else { return }
                // Recommendation failure is soft — user can still convert manually.
                errorText = "Recommendation failed (conversion still works): \(error.localizedDescription)"
            }
        }
    }

    /// Fill in ConversionPlan defaults based on the recommendation. Only touches
    /// fields the user hasn't manually changed (heuristic: fields still at their
    /// initial defaults get replaced; anything else is preserved).
    private func applyRecommendation(_ rec: Recommendation) {
        let plan = coord.plan

        // M144 (iter 66): family was previously overwritten
        // unconditionally every time the user picked a new source. That
        // created an INCONSISTENT-STATE bug: user picks source A, goes to
        // ProfileStep, manually switches to JANGTQ2 (family=.jangtq), then
        // re-picks source A again (or a similar source). Recommendation
        // comes back with family=jang → unconditional overwrite sets
        // family=.jang. Profile preserved as "JANGTQ2" (iter-65 M143 fix
        // does the right thing). Result: family=.jang but profile=JANGTQ2
        // — an invalid pair. Pre-M144, the user ended up stuck until they
        // manually re-synced ProfileStep.
        //
        // Fix: couple family + profile. If profile was preserved (user
        // manually set it), family stays preserved too. If profile was
        // overwritten, derive family from the new profile name so the
        // pair is always consistent.
        let seedDefault = settings.defaultProfile.isEmpty ? "JANG_4K" : settings.defaultProfile
        if plan.profile == seedDefault {
            // User hasn't touched profile → both profile AND family get
            // overwritten together, derived from the new profile so they
            // can't disagree.
            plan.profile = rec.recommended.profile
            plan.family = plan.profile.hasPrefix("JANGTQ") ? .jangtq : .jang
        }
        // else: user manually picked a profile in ProfileStep. Preserve
        // BOTH profile and family to keep them in sync.

        // M145 (iter 67): extend iter-66 M144's "user hasn't touched"
        // preservation to method, hadamard, and forceDtype. Pre-iter-67
        // these three fields were ALL unconditionally overwritten every
        // time the user re-picked a source — silently wiping any manual
        // ProfileStep choices the user made (e.g., toggling Hadamard off
        // for a 2-bit profile, switching to RTN method for speed, forcing
        // bfloat16). Same UX pathology as the pre-iter-65 profile bug.
        //
        // "User hasn't touched" signal = field still matches what
        // applyDefaults seeded it with (from settings.default*).

        // Method: preserve if user manually picked something different
        // from their Settings default.
        let recMethod: QuantMethod = switch rec.recommended.method {
        case "rtn": .rtn
        case "mse-all": .mseAll
        default: .mse
        }
        let seedMethod: QuantMethod = switch settings.defaultMethod.lowercased() {
        case "rtn": .rtn
        case "mse-all", "mseall", "mse_all": .mseAll
        case "mse": .mse
        default: .mse   // applyDefaults leaves method untouched on unknown; seed with init default
        }
        if plan.method == seedMethod {
            plan.method = recMethod
        }

        // Hadamard: preserve if user manually toggled from their Settings
        // default. Pre-iter-67 a user who turned hadamard off for a 2-bit
        // convert got it silently re-enabled on source re-pick.
        if plan.hadamard == settings.defaultHadamardEnabled {
            plan.hadamard = rec.recommended.hadamard
        }

        // Force dtype — only set if recommendation has one AND the user
        // hasn't already chosen an override. Pre-iter-67 this overwrote
        // any user-set forceDtype when rec supplied one (which it does
        // for 512+ expert MoE models — so a user who manually forced fp16
        // for speed on a smaller variant got bfloat16 slammed back on
        // re-pick).
        if plan.overrides.forceDtype == nil,
           let forceDtypeStr = rec.recommended.forceDtype {
            let forceDtype: SourceDtype? = switch forceDtypeStr {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: nil
            }
            if let ft = forceDtype {
                plan.overrides.forceDtype = ft
            }
        }

        // Block size
        if plan.overrides.forceBlockSize == nil {
            plan.overrides.forceBlockSize = rec.recommended.blockSize
        }
    }
}

/// M155 (iter 78): typed error for the SourceDetector's subprocess CLI
/// call — matches iter-51 M129's typed-error pattern on the peer adoption
/// services. Pre-M155 this path threw `NSError(domain: "SourceDetector")`,
/// the only remaining NSError-based surface after iter-51.
enum SourceDetectorError: Error, LocalizedError {
    case cliError(code: Int32, message: String)

    var errorDescription: String? {
        switch self {
        case .cliError(_, let message):
            return message
        }
    }
}

enum SourceDetector {
    struct SourceInfo: Decodable {
        let model_type: String
        let text_model_type: String?
        let is_moe: Bool
        let num_experts: Int
        let num_hidden_layers: Int?
        let num_experts_per_tok: Int?
        let dtype: String
        let total_bytes: Int64
        let shard_count: Int
        let is_vl: Bool
        let is_video_vl: Bool
        let has_generation_config: Bool
        let jangtq_compatible: Bool
    }

    static func inspect(url: URL) async throws -> ArchitectureSummary {
        // M155 (iter 78): migrated to shared PythonCLIInvoker.
        // Previously a standalone copy of the iter-33 M101 cross-layer
        // cancel pattern (iter-34 M105 wired it; iter-43 M120 added stderr
        // surfacing). Now delegates to the canonical helper that
        // iter-76 M153 extracted — with matching typed-error surface
        // (iter-51 M129 retired the last NSError uses in the peer
        // services; SourceDetector was a 6th copy the M153 sweep missed).
        let data = try await PythonCLIInvoker.invoke(
            args: ["-m", "jang_tools", "inspect-source", "--json", url.path]
        ) { code, stderr in
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            let desc = trimmed.isEmpty
                ? "inspect-source exited \(code)"
                : "inspect-source exited \(code): \(trimmed)"
            return SourceDetectorError.cliError(code: code, message: desc)
        }

        let info = try JSONDecoder().decode(SourceInfo.self, from: data)
        let dtype: SourceDtype = switch info.dtype {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: .unknown
        }
        return .init(modelType: info.model_type, isMoE: info.is_moe, numExperts: info.num_experts,
                     isVL: info.is_vl, isVideoVL: info.is_video_vl,
                     hasGenerationConfig: info.has_generation_config,
                     dtype: dtype, totalBytes: info.total_bytes, shardCount: info.shard_count,
                     textModelType: info.text_model_type,
                     numHiddenLayers: positive(info.num_hidden_layers),
                     numExpertsPerTok: positive(info.num_experts_per_tok))
    }

    private static func positive(_ value: Int?) -> Int? {
        guard let value, value > 0 else { return nil }
        return value
    }
}
