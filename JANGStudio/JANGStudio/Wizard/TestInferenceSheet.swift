import SwiftUI
import UniformTypeIdentifiers

struct TestInferenceSheet: View {
    let modelPath: URL
    let isVL: Bool
    let isVideoVL: Bool
    let modelType: String
    let profile: String
    let sizeGb: Double

    @Environment(\.dismiss) private var dismiss
    @State private var vm: TestInferenceViewModel
    @State private var showingSettings = false
    @State private var exportErrorMessage: String? = nil   // M107: surface save failures

    init(modelPath: URL,
         isVL: Bool = false,
         isVideoVL: Bool = false,
         modelType: String = "unknown",
         profile: String = "JANG_4K",
         sizeGb: Double = 0) {
        self.modelPath = modelPath
        self.isVL = isVL
        self.isVideoVL = isVideoVL
        self.modelType = modelType
        self.profile = profile
        self.sizeGb = sizeGb
        self._vm = State(initialValue: TestInferenceViewModel(modelPath: modelPath))
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messagesView
            Divider()
            footer
        }
        .frame(minWidth: 680, minHeight: 540)
        .onDisappear {
            // M162 (iter 85): closing the sheet mid-generate was leaving
            // the Python inference subprocess running to completion — the
            // user saw the sheet disappear (and assumed the work stopped)
            // while the GPU + memory stayed pinned for the remaining
            // 5-60 seconds of the generate call. Not as severe as the
            // publish-sheet variant (no data goes anywhere), but wastes
            // compute + can block a subsequent Test Inference from loading
            // the same model if memory is tight. Cancelling on disappear
            // funnels through vm.cancel() → InferenceRunner.cancel() →
            // SIGTERM + 3 s SIGKILL.
            Task { await vm.cancel() }
        }
        .popover(isPresented: $showingSettings, arrowEdge: .bottom) {
            settingsPopover
        }
        // M107 (iter 35): surface Export Transcript save failures instead of
        // silently swallowing them.
        .alert("Export failed",
               isPresented: Binding(get: { exportErrorMessage != nil },
                                    set: { if !$0 { exportErrorMessage = nil } })) {
            Button("OK") { exportErrorMessage = nil }
        } message: {
            Text(exportErrorMessage ?? "")
        }
    }

    // MARK: - Header
    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Test Inference")
                    .font(.headline)
                HStack(spacing: 8) {
                    Badge(text: modelType)
                    Badge(text: profile)
                    Badge(text: "\(String(format: "%.2f", sizeGb)) GB")
                    if isVideoVL {
                        Badge(text: "video-VL", color: .purple)
                    } else if isVL {
                        Badge(text: "VL", color: .blue)
                    }
                }
                Text(modelPath.path)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button(action: { showingSettings.toggle() }) {
                Image(systemName: "gearshape")
            }
            .buttonStyle(.borderless)
            .help("Inference settings")
            Button("Clear") { vm.clear() }
                .buttonStyle(.borderless)
                .disabled(vm.messages.isEmpty)
            Button("Close") { dismiss() }
        }
        .padding(12)
        .background(Color(.windowBackgroundColor))
    }

    // MARK: - Messages
    private var messagesView: some View {
        ScrollViewReader { scroller in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if vm.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(vm.messages) { msg in
                            ChatBubble(msg: msg).id(msg.id)
                        }
                    }
                    if vm.isGenerating {
                        HStack {
                            ProgressView().controlSize(.small)
                            // M225 (iter 150): use the ViewModel's
                            // workingStatusLabel() which distinguishes
                            // load vs. generate + cites elapsed wall-
                            // clock. Pre-M225 the label was a constant
                            // "Generating..." which misled users into
                            // thinking the model was slow at generation
                            // when loading was the real cost.
                            Text(vm.workingStatusLabel()).foregroundStyle(.secondary)
                                .font(.caption)
                            Spacer()
                        }.padding(.horizontal, 8)
                    }
                    if let err = vm.lastError {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.red)
                            .padding(8)
                            .background(Color.red.opacity(0.08))
                            .cornerRadius(6)
                    }
                }
                .padding(16)
            }
            .onChange(of: vm.messages.count) { _, _ in
                if let last = vm.messages.last {
                    withAnimation { scroller.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)
            Text("No messages yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text(isVL ? "Type a prompt or drop an image to get started."
                     : "Type a prompt to see how your converted model responds.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)

            // M223 (iter 149): suggested-prompt buttons + reasoning-model
            // hint. Pre-M223 a stranger arriving at an empty Test Inference
            // sheet had no example of what a good test prompt looks like —
            // they'd type "Hello" and see a generic response, learning
            // nothing about model capabilities. The 3 suggested prompts
            // demonstrate (a) factual recall, (b) reasoning, (c) creativity
            // in one click each. For reasoning models (Qwen3.6 / GLM-5.1 /
            // MiniMax M2.7), an additional hint surfaces the documented
            // failure mode where 150-token smoke tests are eaten by
            // <think>...</think> blocks → user sees no answer + assumes
            // model is broken. The hint points at the existing "Skip
            // thinking" Settings toggle which fixes this.
            VStack(spacing: 6) {
                Text("Try a sample prompt:")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    samplePromptButton("What is the capital of France?")
                    samplePromptButton("Explain why the sky is blue.")
                    samplePromptButton("Write a short poem about Apple Silicon.")
                }
            }
            .padding(.top, 4)

            if Self.isReasoningModelType(modelType) && !vm.skipThinking {
                Text("Reasoning model detected. If your first answer looks empty or cut off, open Settings (⚙) and turn on \"Skip thinking\" — this model wraps prompts in `<think>…</think>` by default which eats short test budgets.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .padding(.top, 6)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
    }

    /// M223 (iter 149): one-click sample prompt. Tapping prefills the
    /// prompt field; user can edit before sending or just hit Send.
    /// Stays as a Button (not Send-on-tap) because a stranger may
    /// want to read the suggested prompt before committing — and may
    /// realize they want to tweak the wording.
    private func samplePromptButton(_ prompt: String) -> some View {
        Button(prompt) {
            vm.promptText = prompt
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .font(.caption2)
        .lineLimit(1)
    }

    /// M223 (iter 149): identify reasoning models that need the
    /// Skip-thinking toggle to be useful for short smoke tests. The
    /// list mirrors the comment block in TestInferenceViewModel's
    /// `--no-thinking` flag handling (M121 iter 45) — Qwen3.6,
    /// GLM-5.1, MiniMax M2.7. Future reasoning models added to JANG
    /// should be added here too. Match is substring-based on
    /// model_type to handle dotted variants ("qwen3_5_moe" matches
    /// "qwen3_5", etc.).
    private static func isReasoningModelType(_ modelType: String) -> Bool {
        let mt = modelType.lowercased()
        return mt.contains("qwen3_5") || mt.contains("qwen3_6") ||
               mt.contains("glm") || mt.contains("minimax")
    }

    // MARK: - Footer
    private var footer: some View {
        VStack(spacing: 8) {
            if isVL || isVideoVL {
                HStack {
                    if isVL {
                        dropTargetView(
                            label: vm.pendingImagePath?.lastPathComponent ?? "Drop image here",
                            systemImage: "photo",
                            types: [.image]
                        ) { url in
                            vm.pendingImagePath = url
                        }
                    }
                    if isVideoVL {
                        dropTargetView(
                            label: vm.pendingVideoPath?.lastPathComponent ?? "Drop video here",
                            systemImage: "film",
                            types: [.movie, .video]
                        ) { url in
                            vm.pendingVideoPath = url
                        }
                    }
                }
                .padding(.horizontal, 12)
            }

            HStack(spacing: 8) {
                TextField("Prompt", text: $vm.promptText, axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await vm.send() } }
                    .disabled(vm.isGenerating)
                if vm.isGenerating {
                    Button("Cancel", role: .destructive) {
                        Task { await vm.cancel() }
                    }
                } else {
                    Button("Send") {
                        Task { await vm.send() }
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(vm.promptText.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(.horizontal, 12)

            HStack(spacing: 12) {
                if vm.lastTokensPerSec > 0 {
                    StatView(label: "tok/s", value: String(format: "%.1f", vm.lastTokensPerSec))
                }
                if vm.lastPeakRssMb > 0 {
                    StatView(label: "peak RAM", value: String(format: "%.0f MB", vm.lastPeakRssMb))
                }
                Spacer()
                Button("Export...") { exportTranscript() }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .disabled(vm.messages.isEmpty)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
        }
        .padding(.top, 8)
        .background(Color(.windowBackgroundColor))
    }

    // MARK: - Settings popover
    private var settingsPopover: some View {
        Form {
            TextField("System prompt", text: $vm.systemPrompt, axis: .vertical)
                .lineLimit(2...4)
            Slider(value: $vm.temperature, in: 0.0...2.0, step: 0.05) {
                Text("Temperature")
            } minimumValueLabel: {
                Text("0.0").font(.caption2)
            } maximumValueLabel: {
                Text("2.0").font(.caption2)
            }
            LabeledContent("Temperature", value: String(format: "%.2f", vm.temperature))
            Stepper(value: $vm.maxTokens, in: 8...4096, step: 32) {
                LabeledContent("Max tokens", value: "\(vm.maxTokens)")
            }
            // M121 (iter 45): reasoning-model smoke-test toggle. When on,
            // passes --no-thinking to jang_tools inference so the chat
            // template skips the <think>…</think> wrapper. Essential for
            // short-answer smoke tests against GLM-5.1 / Qwen3.6 / MiniMax
            // M2.7 — without it, 150-token smoke tests are consumed by the
            // thinking block and the user never sees an answer. Non-reasoning
            // templates silently ignore the flag.
            Toggle("Skip thinking (reasoning models)", isOn: $vm.skipThinking)
                .help("GLM-5.1 / Qwen3.6 / MiniMax M2.7 wrap the prompt in <think>…</think> by default, which eats 100+ tokens before answering. Turn this on for quick factual questions.")
        }
        .padding(16)
        .frame(width: 340)
    }

    // MARK: - Helpers

    private func dropTargetView(label: String,
                                systemImage: String,
                                types: [UTType],
                                onDrop onDropCallback: @escaping @Sendable (URL) -> Void) -> some View {
        HStack(spacing: 4) {
            Image(systemName: systemImage)
            Text(label).font(.caption).lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(6)
        .onDrop(of: types, isTargeted: nil) { providers in
            for prov in providers {
                _ = prov.loadObject(ofClass: URL.self) { url, _ in
                    if let url {
                        Task { @MainActor in onDropCallback(url) }
                    }
                }
            }
            return true
        }
    }

    private func exportTranscript() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "chat-transcript.json"
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url {
            // M107 (iter 35): previously silent-swallowed the error via try?.
            // Disk-full / permission-denied / read-only-volume failures left
            // the user thinking they saved a file that doesn't exist. Now we
            // surface the error as an alert. Successful saves stay quiet —
            // Finder open is the user's feedback that the file landed.
            do {
                try vm.exportTranscript(to: url)
            } catch {
                exportErrorMessage = "Couldn't save transcript to \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Subviews

private struct Badge: View {
    let text: String
    var color: Color = .secondary
    var body: some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .cornerRadius(4)
    }
}

private struct StatView: View {
    let label: String
    let value: String
    var body: some View {
        HStack(spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(.tertiary)
            Text(value).font(.caption).monospacedDigit()
        }
    }
}

private struct ChatBubble: View {
    let msg: ChatMessage
    var body: some View {
        HStack {
            if msg.role == .user { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(msg.role.rawValue.uppercased())
                        .font(.caption2)
                        .foregroundStyle(roleColor.opacity(0.8))
                    if let tps = msg.tokensPerSec {
                        Text("\(String(format: "%.1f", tps)) tok/s")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                Text(msg.text)
                    .textSelection(.enabled)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(roleColor.opacity(0.12))
                    .cornerRadius(8)
            }
            if msg.role == .assistant { Spacer(minLength: 40) }
        }
    }
    private var roleColor: Color {
        switch msg.role {
        case .user: return .accentColor
        case .assistant: return .green
        case .system: return .secondary
        }
    }
}
