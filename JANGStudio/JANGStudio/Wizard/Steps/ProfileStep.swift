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

    private var jangProfileNames: [String] {
        profilesSvc.profiles.jang.map { $0.name }
    }
    private var jangtqProfileNames: [String] {
        profilesSvc.profiles.jangtqProfileNames(for: coord.plan.detected?.modelType)
    }
    private var isJANGTQAllowed: Bool {
        coord.plan.isJANGTQAllowed(for: capsSvc.capabilities.jangtqWhitelist)
    }

    var body: some View {
        Form {
            Section("Family") {
                Picker("", selection: $coord.plan.family) {
                    Text("JANG").tag(Family.jang)
                    Text("JANGTQ").tag(Family.jangtq).disabled(!isJANGTQAllowed)
                }.pickerStyle(.segmented)
                if !isJANGTQAllowed {
                    Label("JANGTQ supports \(capsSvc.capabilities.jangtqWhitelist.joined(separator: ", ")) only.",
                          systemImage: "info.circle").font(.caption)
                } else if coord.plan.family == .jangtq,
                          let mt = coord.plan.detected?.modelType,
                          let info = profilesSvc.profiles.jangtqFamilies[mt],
                          !info.note.isEmpty {
                    Label(info.note, systemImage: "info.circle").font(.caption)
                }
            }
            Section("Profile") {
                Picker("", selection: $coord.plan.profile) {
                    ForEach(coord.plan.family == .jang ? jangProfileNames : jangtqProfileNames, id: \.self) { p in
                        Text(p).tag(p)
                    }
                }.pickerStyle(.menu)
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
                }.pickerStyle(.segmented)
                Toggle("Hadamard rotation", isOn: $coord.plan.hadamard)
            }
            Section("Pre-flight") {
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
            Button("Start Conversion") { coord.active = .run }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!allMandatoryPass())
        }
        .formStyle(.grouped)
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
            ensureSupportedProfile()
            refresh()
        }
        .onChange(of: coord.plan.outputURL) { _, _ in refresh() }
        .onChange(of: coord.plan.hadamard) { _, _ in refresh() }
        .onAppear {
            if coord.plan.outputURL == nil, let src = coord.plan.sourceURL {
                coord.plan.outputURL = autoOutputURL(for: src, profile: coord.plan.profile)
            }
            ensureSupportedProfile()
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
        // M141 (iter 63): pass profiles so the disk-space check can do a
        // profile-aware size estimate instead of short-circuiting to .pass.
        preflight = PreflightRunner().run(
            plan: coord.plan,
            capabilities: capsSvc.capabilities,
            profiles: profilesSvc.profiles
        )
    }

    private func ensureSupportedProfile() {
        let available = coord.plan.family == .jang ? jangProfileNames : jangtqProfileNames
        guard !available.isEmpty, !available.contains(coord.plan.profile) else { return }
        let mt = coord.plan.detected?.modelType
        let preferred = profilesSvc.profiles.jangtqFamilies[mt ?? ""]?.defaultProfile
        coord.plan.profile = preferred.flatMap { available.contains($0) ? $0 : nil } ?? available[0]
    }
    private func allMandatoryPass() -> Bool { !preflight.contains { $0.status == .fail } }

    private func pickOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.allowsMultipleSelection = false; panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK { coord.plan.outputURL = panel.url }
    }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
}
