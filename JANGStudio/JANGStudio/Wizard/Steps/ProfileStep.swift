// JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift
import SwiftUI

struct ProfileStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(ProfilesService.self) private var profilesSvc
    @Environment(CapabilitiesService.self) private var capsSvc
    // M210 (iter 142): inject AppSettings so the auto-generated output
    // path honors Settings → General → Default output parent path AND
    // Settings → General → Output naming template. Pre-M210 both
    // fields were Settings-UI lies: the UI persisted the values but
    // ProfileStep's auto-path code hardcoded `<basename>-<profile>` at
    // `src.deletingLastPathComponent()`. Flipping either setting had
    // zero effect on what dir got created.
    @Environment(AppSettings.self) private var settings
    @State private var preflight: [PreflightCheck] = []
    @State private var isChecking = false
    @State private var showOverrides = false

    private var jangProfileNames: [String] {
        profilesSvc.profiles.jang.map { $0.name }
    }
    private var jangtqProfileNames: [String] {
        profilesSvc.profiles.jangtq.map { $0.name }
    }
    private var isJANGTQAllowed: Bool {
        coord.plan.isJANGTQAllowed(for: capsSvc.capabilities.jangtqWhitelist)
    }
    private var isExpertLabBundleTarget: Bool {
        coord.plan.detected?.isMoE == true && coord.plan.family == .jangtq
    }
    private var isSmartReviewFlow: Bool {
        coord.plan.expertReviewIntent == .smartPrequantPrune
    }
    private var isFinalConversionFromReviewedPrune: Bool {
        coord.plan.expertReviewPrunedSourceURL != nil
    }

    var body: some View {
        Form {
            if isFinalConversionFromReviewedPrune {
                Section("Reviewed BF16/F16 Source") {
                    ExpertLabConsoleCard(accent: ExpertLabVisual.good) {
                        VStack(alignment: .leading, spacing: 8) {
                            ExpertLabKicker(text: "Final conversion profile", color: ExpertLabVisual.good)
                            Label("Choose the final JANG or JANGTQ profile", systemImage: "slider.horizontal.3")
                                .font(.headline)
                            Text("Expert Lab already traced, compared, generated a reviewed prune plan, and hard-pruned a new BF16/F16 source. Any legacy review bundle is no longer used as pruning authority.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            if let pruned = coord.plan.expertReviewPrunedSourceURL {
                                Text(pruned.path)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            if let evidence = reviewedPruneEvidence {
                                Label(
                                    evidence.sameSuiteVerificationReady
                                        ? "Same-suite Expert Lab evidence ready"
                                        : "Same-suite Expert Lab evidence incomplete",
                                    systemImage: evidence.sameSuiteVerificationReady ? "checkmark.seal.fill" : "exclamationmark.triangle.fill"
                                )
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(evidence.sameSuiteVerificationReady ? ExpertLabVisual.good : ExpertLabVisual.warm)
                                HStack(spacing: 10) {
                                    Label("\(evidence.promptCount) prompts", systemImage: "text.bubble")
                                    Label("\(evidence.layerCount) layers", systemImage: "square.stack.3d.down.right")
                                    Label("\(evidence.keepExpertsPerLayer) kept/layer", systemImage: "switch.2")
                                }
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                Label(
                                    evidence.prunedSuiteVerificationReady == true
                                        ? "Pruned BF16/F16 vMLX generation ready"
                                        : "Pruned BF16/F16 vMLX generation missing",
                                    systemImage: evidence.prunedSuiteVerificationReady == true ? "checkmark.seal.fill" : "exclamationmark.triangle.fill"
                                )
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(evidence.prunedSuiteVerificationReady == true ? ExpertLabVisual.good : ExpertLabVisual.warm)
                                Text("Pruned artifacts: \(evidence.prunedSuiteArtifactDescription)")
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .textSelection(.enabled)
                                Text("Reviewed suite SHA-256: \(evidence.reviewedSuiteFingerprintDescription)")
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .textSelection(.enabled)
                                if let prunedIssue = evidence.prunedSuiteVerificationIssue, !prunedIssue.isEmpty {
                                    Text(prunedIssue)
                                        .font(.caption2)
                                        .foregroundStyle(ExpertLabVisual.warm)
                                        .textSelection(.enabled)
                                }
                                if let compare = ReviewedPruneComparisonEvidence.load(from: evidence) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Label(
                                            "Masked compare: \(compare.promptCount) prompts, baseline \(compare.baselinePassRateDescription), masked \(compare.maskedPassRateDescription)",
                                            systemImage: "rectangle.split.2x1"
                                        )
                                        Text(
                                            String(
                                                format: "Eval deltas: mean text %.4f, risky prompts %d, avg tokens %.1f / %.1f",
                                                compare.meanTextDelta,
                                                compare.riskyPromptCount,
                                                compare.meanBaselineTokens ?? 0,
                                                compare.meanMaskedTokens ?? 0
                                            )
                                        )
                                        Text("Risk domains: \(compare.riskDomainDescription)")
                                        Text("Artifacts: \(compare.artifactDescription)")
                                            .foregroundStyle(.tertiary)
                                    }
                                    .font(.caption2)
                                    .foregroundStyle(compare.riskyPromptCount == 0 && compare.highRiskDomains.isEmpty ? .secondary : ExpertLabVisual.warm)
                                    .textSelection(.enabled)
                                } else {
                                    Label("Masked compare sidecars are not readable from the pruned source.",
                                          systemImage: "exclamationmark.triangle.fill")
                                        .font(.caption2)
                                        .foregroundStyle(ExpertLabVisual.warm)
                                }
                                if let issue = evidence.sameSuiteVerificationIssue, !issue.isEmpty {
                                    Text(issue)
                                        .font(.caption2)
                                        .foregroundStyle(ExpertLabVisual.warm)
                                        .textSelection(.enabled)
                                }
                            } else {
                                Label("Same-suite Expert Lab evidence summary is missing from the pruned source.",
                                      systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(ExpertLabVisual.warm)
                            }
                        }
                    }
                }
            }
            Section("Family") {
                Picker("", selection: $coord.plan.family) {
                    Text("JANG").tag(Family.jang)
                    Text("JANGTQ").tag(Family.jangtq).disabled(!isJANGTQAllowed)
                }.pickerStyle(.segmented)
                GuidanceCard(
                    title: QuantizationGuidance.familyTitle(coord.plan.family),
                    systemImage: coord.plan.family == .jang ? "slider.horizontal.3" : "point.3.connected.trianglepath.dotted",
                    bodyText: QuantizationGuidance.familyBody(
                        coord.plan.family,
                        isMoE: coord.plan.detected?.isMoE == true
                    )
                )
                if !isJANGTQAllowed {
                    Label("JANGTQ supports \(capsSvc.capabilities.jangtqWhitelist.joined(separator: ", ")) only.",
                          systemImage: "info.circle").font(.caption)
                }
            }
            if coord.plan.detected?.isMoE == true {
                Section("Expert Lab & pruning") {
                    if isExpertLabBundleTarget {
                        Label(isSmartReviewFlow
                              ? "This opens the original BF16/F16 source in Expert Lab for vMLX prompt-suite expert probing."
                              : "This will build a JANGTQ MoE bundle for Expert Lab tracing, temporary expert masks, and A/B prompt comparisons.",
                              systemImage: "point.3.connected.trianglepath.dotted")
                        Text(isSmartReviewFlow
                             ? "Expert Review runs the prompt suite through BF16/vMLX, lets you mask and compare experts, and returns a keep/drop plan for BF16/F16 source pruning. Final JANG/JANGTQ conversion stays downstream of the verified pruned source."
                             : "Legacy JANGTQ tracing is downstream compatibility evidence only. Reviewed hard pruning starts by opening the original BF16/F16 source in BF16/vMLX Expert Review before any final conversion.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Label("Standard JANG conversion will not enable native interactive expert masks. For reviewed hard pruning, go back to Source and run BF16/vMLX Expert Review on the original BF16/F16 source first.",
                              systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                    }
                }
            }
            Section("Profile") {
                Picker("", selection: $coord.plan.profile) {
                    ForEach(coord.plan.family == .jang ? jangProfileNames : jangtqProfileNames, id: \.self) { p in
                        Text(p).tag(p)
                    }
                }.pickerStyle(.menu)
                QuantProfileGuideView(profileName: coord.plan.profile, profiles: profilesSvc.profiles)
                QuantProfileCatalog(family: coord.plan.family, profiles: profilesSvc.profiles)
            }
            Section("Output folder") {
                HStack {
                    Text(coord.plan.outputURL?.path ?? "—").foregroundStyle(.secondary)
                    Spacer()
                    Button("Choose…", action: pickOutput)
                }
            }
            Section("Options") {
                Picker("Method", selection: $coord.plan.method) {
                    Text("MSE").tag(QuantMethod.mse)
                    Text("RTN").tag(QuantMethod.rtn)
                    Text("MSE (all)").tag(QuantMethod.mseAll)
                }
                .pickerStyle(.segmented)
                .disabled(coord.plan.family == .jangtq)
                GuidanceCard(
                    title: "Quant method",
                    systemImage: "function",
                    bodyText: QuantizationGuidance.methodBody(
                        coord.plan.method,
                        family: coord.plan.family
                    )
                )
                Toggle("Hadamard rotation", isOn: $coord.plan.hadamard)
                    .disabled(coord.plan.family == .jangtq)
                GuidanceCard(
                    title: "Hadamard rotation",
                    systemImage: "arrow.triangle.2.circlepath",
                    bodyText: QuantizationGuidance.hadamardBody(
                        isEnabled: coord.plan.hadamard,
                        profile: coord.plan.profile,
                        family: coord.plan.family
                    )
                )
            }
            // PR2: Architecture step was orphaned from WizardStep; Advanced
            // overrides (force dtype / block size) live here for the JANG
            // family. JANGTQ converters ignore these flags today.
            if coord.plan.family == .jang {
                Section {
                    DisclosureGroup("Advanced overrides", isExpanded: $showOverrides) {
                        Picker("Force dtype", selection: Binding(
                            get: { coord.plan.overrides.forceDtype ?? .unknown },
                            set: { coord.plan.overrides.forceDtype = ($0 == .unknown) ? nil : $0 }
                        )) {
                            Text("Auto").tag(SourceDtype.unknown)
                            ForEach(capsSvc.capabilities.supportedSourceDtypes, id: \.name) { d in
                                Text(d.alias.uppercased()).tag(dtypeFromAlias(d.alias))
                            }
                        }
                        GuidanceCard(
                            title: "Force dtype",
                            systemImage: "number.square",
                            bodyText: QuantizationGuidance.forceDtypeBody(coord.plan.overrides.forceDtype)
                        )
                        Picker("Block size", selection: Binding(
                            get: { coord.plan.overrides.forceBlockSize ?? 0 },
                            set: { coord.plan.overrides.forceBlockSize = ($0 == 0) ? nil : $0 }
                        )) {
                            Text("Auto").tag(0)
                            ForEach(capsSvc.capabilities.blockSizes, id: \.self) { bs in
                                Text("\(bs)").tag(bs)
                            }
                        }
                        GuidanceCard(
                            title: "Block size",
                            systemImage: "square.grid.3x3.square",
                            bodyText: QuantizationGuidance.blockSizeBody(coord.plan.overrides.forceBlockSize)
                        )
                    }
                }
            }
            Section("Pre-flight") {
                if isChecking {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Checking pre-flight conditions…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(preflight) { check in
                    HStack {
                        Image(systemName: icon(check.status))
                            .foregroundStyle(color(check.status))
                        Text(check.title)
                        if let hint = check.hint {
                            Text(hint).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Button(primaryActionTitle) {
                coord.plan.run = .idle
                coord.active = .run
            }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!allMandatoryPass())
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(ExpertLabVisual.canvas)
        .padding()
        .onChange(of: coord.plan.profile) { oldProfile, newProfile in
            // M146 (iter 68): when the user switches profiles, the
            // auto-filled output folder name (`<src>-<oldProfile>`) becomes
            // stale. Result pre-M146: user converts as JANG_2L but the
            // folder is named `-JANG_4K` — wrong label on every diagnostic,
            // HF publish, and `ls` going forward. If the current outputURL
            // matches the auto-pattern for the OLD profile (i.e., we
            // generated it, not the user), regenerate for the NEW profile.
            // If outputURL was user-picked via pickOutput(), it won't match
            // the auto-pattern and we leave it alone.
            //
            // M210 (iter 142): both the "is this auto-generated?" check
            // AND the regeneration now route through autoOutputURL()
            // so Settings → defaultOutputParentPath + outputNamingTemplate
            // are honored consistently.
            if let src = coord.plan.sourceURL,
               let cur = coord.plan.outputURL {
                let autoOld = autoOutputURL(for: src, profile: oldProfile)
                if cur == autoOld {
                    coord.plan.outputURL = autoOutputURL(for: src, profile: newProfile)
                }
            }
            refresh()
        }
        .onChange(of: coord.plan.family) { _, _ in
            syncProfileForCurrentFamily()
            refresh()
        }
        .onChange(of: coord.plan.outputURL) { _, _ in refresh() }
        .onChange(of: coord.plan.hadamard) { _, _ in refresh() }
        .onChange(of: coord.plan.overrides.forceDtype) { _, _ in refresh() }
        .onChange(of: coord.plan.overrides.forceBlockSize) { _, _ in refresh() }
        .onAppear {
            if coord.plan.outputURL == nil, let src = coord.plan.sourceURL {
                coord.plan.outputURL = autoOutputURL(for: src, profile: coord.plan.profile)
            }
            refresh()
        }
    }

    /// M210 (iter 142): compute the auto-generated output URL honoring
    /// both `settings.defaultOutputParentPath` (non-empty overrides the
    /// source's parent dir) and `settings.outputNamingTemplate` (token
    /// substitution for basename/profile/family/date/time/user).
    ///
    /// Parent resolution:
    ///   1. If `settings.defaultOutputParentPath` is non-empty and
    ///      points at a valid directory, use it.
    ///   2. Otherwise fall back to `src.deletingLastPathComponent()`
    ///      (the source's parent) — matches pre-M210 hardcoded behavior.
    ///
    /// Basename resolution:
    ///   `settings.renderOutputName(basename:profile:family:)` applies
    ///   the template. Default template `{basename}-{profile}`
    ///   reproduces pre-M210 naming for users who haven't touched the
    ///   setting. Power users who set e.g. `{basename}_q{profile}` or
    ///   `{date}-{basename}-{profile}` now see the template actually
    ///   take effect on the auto-generated folder name.
    private func autoOutputURL(for source: URL, profile: String) -> URL {
        let parent: URL = {
            let configured = settings.defaultOutputParentPath
            if !configured.isEmpty {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: configured, isDirectory: &isDir),
                   isDir.boolValue {
                    return URL(fileURLWithPath: configured)
                }
            }
            return source.deletingLastPathComponent()
        }()
        let name = settings.renderOutputName(
            basename: source.lastPathComponent,
            profile: profile,
            family: coord.plan.family.rawValue
        )
        return parent.appendingPathComponent(name)
    }

    private func refresh() {
        Task { @MainActor in
            isChecking = true
            preflight = PreflightRunner().run(
                plan: coord.plan,
                capabilities: capsSvc.capabilities,
                profiles: profilesSvc.profiles
            )
            isChecking = false
        }
    }

    private func syncProfileForCurrentFamily() {
        let validNames = coord.plan.family == .jang ? jangProfileNames : jangtqProfileNames
        guard !validNames.contains(coord.plan.profile) else { return }

        if coord.plan.family == .jangtq, validNames.contains("JANGTQ3") {
            coord.plan.profile = "JANGTQ3"
        } else if coord.plan.family == .jang, validNames.contains("JANG_4K") {
            coord.plan.profile = "JANG_4K"
        } else if let first = validNames.first {
            coord.plan.profile = first
        }
    }

    private var reviewedPruneEvidence: ReviewedPruneEvidenceSummary? {
        guard let pruned = coord.plan.expertReviewPrunedSourceURL else { return nil }
        let url = pruned.appendingPathComponent("expert_lab_review_summary.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ReviewedPruneEvidenceSummary.self, from: data)
    }

    private func allMandatoryPass() -> Bool {
        guard !preflight.isEmpty else { return false }
        if isFinalConversionFromReviewedPrune {
            guard preflight.contains(where: {
                $0.id == .reviewedPruneVerified && $0.status == .pass
            }) else {
                return false
            }
        }
        return !preflight.contains { $0.status == .fail }
    }

    private var primaryActionTitle: String {
        if isFinalConversionFromReviewedPrune {
            return coord.plan.family == .jangtq ? "Quantize with JANGTQ" : "Quantize with JANG"
        }
        if isSmartReviewFlow { return "Open BF16/vMLX Expert Review" }
        if isExpertLabBundleTarget { return "Build Expert Lab Bundle" }
        return "Start Conversion"
    }

    private func pickOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.allowsMultipleSelection = false; panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK { coord.plan.outputURL = panel.url }
    }

    private func dtypeFromAlias(_ alias: String) -> SourceDtype {
        switch alias {
        case "bf16": return .bf16
        case "fp16": return .fp16
        case "fp8": return .fp8
        case "fp8-e5m2": return .fp8
        default: return .unknown
        }
    }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
}

private struct ReviewedPruneEvidenceSummary: Decodable {
    let promptCount: Int
    let layerCount: Int
    let keepExpertsPerLayer: Int
    let sameSuiteVerificationReady: Bool
    let sameSuiteVerificationIssue: String?
    let suiteJSONL: String?
    let suiteSHA256: String?
    let comparisonSummary: String?
    let evalJSONL: String?
    let evalTraceJSONL: String?
    let evalIndex: String?
    let mask: String?
    let maskJSON: String?
    let prunedSuiteVerificationReady: Bool?
    let prunedSuiteVerificationIssue: String?
    let prunedSuiteSummary: String?
    let prunedSuiteGenerations: String?

    var prunedSuiteArtifactDescription: String {
        let names = [
            prunedSuiteSummary,
            prunedSuiteGenerations
        ]
            .compactMap { $0 }
            .map { URL(fileURLWithPath: $0).lastPathComponent }
            .joined(separator: ", ")
        return names.isEmpty ? "missing" : names
    }

    var reviewedSuiteFingerprintDescription: String {
        guard let suiteSHA256,
              !suiteSHA256.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "missing"
        }
        return suiteSHA256
    }

    private enum CodingKeys: String, CodingKey {
        case promptCount = "prompt_count"
        case layerCount = "layer_count"
        case keepExpertsPerLayer = "keep_experts_per_layer"
        case sameSuiteVerificationReady = "same_suite_verification_ready"
        case sameSuiteVerificationIssue = "same_suite_verification_issue"
        case suiteJSONL = "suite_jsonl"
        case suiteSHA256 = "suite_sha256"
        case comparisonSummary = "comparison_summary"
        case evalJSONL = "eval_jsonl"
        case evalTraceJSONL = "eval_trace_jsonl"
        case evalIndex = "eval_index"
        case mask
        case maskJSON = "mask_json"
        case prunedSuiteVerificationReady = "pruned_suite_verification_ready"
        case prunedSuiteVerificationIssue = "pruned_suite_verification_issue"
        case prunedSuiteSummary = "pruned_suite_summary"
        case prunedSuiteGenerations = "pruned_suite_generations"
    }
}

private struct ReviewedPruneComparisonEvidence {
    let promptCount: Int
    let passRateBaseline: Double?
    let passRateMasked: Double?
    let meanTextDelta: Double
    let highRiskDomains: [String]
    let riskyPromptCount: Int
    let meanBaselineTokens: Double?
    let meanMaskedTokens: Double?
    let artifactNames: [String]

    var baselinePassRateDescription: String { Self.passRateDescription(passRateBaseline) }
    var maskedPassRateDescription: String { Self.passRateDescription(passRateMasked) }
    var riskDomainDescription: String {
        highRiskDomains.isEmpty ? "none" : highRiskDomains.sorted().joined(separator: ", ")
    }
    var artifactDescription: String {
        artifactNames.isEmpty ? "missing" : artifactNames.joined(separator: ", ")
    }

    static func load(from evidence: ReviewedPruneEvidenceSummary) -> ReviewedPruneComparisonEvidence? {
        guard let comparisonPath = evidence.comparisonSummary,
              let evalIndexPath = evidence.evalIndex,
              let comparison = readObject(at: comparisonPath),
              let evalIndex = readObject(at: evalIndexPath) else {
            return nil
        }
        let artifacts = [
            evidence.suiteJSONL,
            evidence.comparisonSummary,
            evidence.evalJSONL,
            evidence.evalTraceJSONL,
            evidence.evalIndex,
            evidence.maskJSON ?? evidence.mask
        ]
            .compactMap { $0 }
            .map { URL(fileURLWithPath: $0).lastPathComponent }
        return ReviewedPruneComparisonEvidence(
            promptCount: intValue(comparison["promptCount"] ?? comparison["prompt_count"])
                ?? intValue(evalIndex["prompt_count"] ?? evalIndex["promptCount"])
                ?? evidence.promptCount,
            passRateBaseline: doubleValue(comparison["passRateBaseline"] ?? comparison["pass_rate_baseline"]),
            passRateMasked: doubleValue(comparison["passRateMasked"] ?? comparison["pass_rate_masked"]),
            meanTextDelta: doubleValue(comparison["meanTextDelta"] ?? comparison["mean_text_delta"]) ?? 0,
            highRiskDomains: stringArrayValue(comparison["highRiskDomains"] ?? comparison["high_risk_domains"]) ?? [],
            riskyPromptCount: (stringArrayValue(evalIndex["risky_prompt_ids"] ?? evalIndex["riskyPromptIDs"]) ?? []).count,
            meanBaselineTokens: doubleValue(evalIndex["mean_baseline_tokens"] ?? evalIndex["meanBaselineTokens"]),
            meanMaskedTokens: doubleValue(evalIndex["mean_masked_tokens"] ?? evalIndex["meanMaskedTokens"]),
            artifactNames: artifacts
        )
    }

    private static func passRateDescription(_ value: Double?) -> String {
        guard let value else { return "unscored" }
        return String(format: "%.0f%%", value * 100)
    }

    private static func readObject(at path: String) -> [String: Any]? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return object
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
}
