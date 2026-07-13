/*
 * JANGTQ Tokenizer — MiniMax-aware wrapper over JANGTokenizer.
 * Created by Jinho Jang (eric@jangq.ai)
 *
 * The base `JANGTokenizer` already handles GPT-2-style byte-level BPE +
 * tokenizer.json + special tokens correctly. What it does NOT handle is
 * MiniMax M2.7's specific chat template (which uses `]~!b[`, `]~b]`,
 * `[e~[`, `<think>`, etc.) and multi-EOS stop sets.
 *
 * This wrapper:
 *   - Loads JANGTokenizer to get vocab, merges, BPE encoding/decoding
 *   - Looks up MiniMax's special token IDs by name
 *   - Provides `applyChatTemplate(messages:)` matching the model's
 *     `chat_template.jinja` byte-for-byte for typical user-only conversations
 *   - Provides a `stopTokenIds` set with all valid stop tokens (the model
 *     emits `[e~[` for end-of-turn, but a multi-eos model could have more)
 *
 * Chat template (from MiniMax-M2.7-JANGTQ_2L/chat_template.jinja):
 *
 *   ]~!b[]~b]system
 *   {system_msg}[e~[
 *   ]~b]user
 *   {user_msg}[e~[
 *   ]~b]ai
 *   <think>
 *
 * The default system message is
 *   "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
 *
 * Token IDs (from MiniMax-M2.7-JANGTQ_2L tokenizer.json):
 *   ]!p~[      = 200000
 *   ]~b]       = 200019  ("turn marker")
 *   [e~[       = 200020  (end-of-turn / EOS)
 *   ]~!b[      = 200034  ("bos / start of conversation")
 *   <think>    = 200050
 *   </think>   = 200051
 */

import Foundation

public struct JANGTQChatMessage {
    public let role: String     // "system", "user", "assistant"
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public final class JANGTQTokenizer {
    public enum ChatTemplateStyle: Sendable, Equatable {
        case minimax
        case qwenIM
        case roleNewline
    }

    public let inner: JANGTokenizer

    /// All token IDs that should terminate generation. For MiniMax M2.7
    /// this is just `[e~[` (200020), but multi-eos models like GLM-5.1
    /// have multiple stop tokens.
    public let stopTokenIds: Set<Int>

    /// MiniMax-specific named special tokens. nil if not present in vocab.
    public let bosBegin: Int?       // ]~!b[ — start-of-conversation marker
    public let turnMarker: Int?     // ]~b] — turn marker
    public let endOfTurn: Int?      // [e~[ — eos / end-of-turn
    public let thinkStart: Int?     // <think>
    public let thinkEnd: Int?       // </think>
    public let chatTemplateStyle: ChatTemplateStyle
    public let startsWithThinking: Bool

    /// Default system message from the MiniMax chat template.
    public let defaultSystemPrompt: String

    public init(modelDir: URL) throws {
        let tokPath = modelDir.appendingPathComponent("tokenizer.json")
        let tk = try JANGTokenizer(tokenizerPath: tokPath)
        let tokCfgPath = modelDir.appendingPathComponent("tokenizer_config.json")
        let tokCfg = (try? Data(contentsOf: tokCfgPath))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]

        // Look up MiniMax special tokens by name. The base JANGTokenizer
        // segments added tokens before BPE so `encode("[e~[")` returns a
        // singleton list with the special token's ID if present.
        func tokenId(_ name: String) -> Int? {
            let ids = tk.encode(name)
            return ids.count == 1 ? ids[0] : nil
        }
        let bosBeginID   = tokenId("]~!b[")
        let turnMarkerID = tokenId("]~b]")
        let endOfTurnID  = tokenId("[e~[")
        let thinkStartID = tokenId("<think>")
        let thinkEndID   = tokenId("</think>")

        // Build stop token set.
        // Try generation_config.json first — it has the canonical eos list.
        var stops: Set<Int> = []
        let genCfgPath = modelDir.appendingPathComponent("generation_config.json")
        if let data = try? Data(contentsOf: genCfgPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let eos = json["eos_token_id"] as? Int {
                stops.insert(eos)
            } else if let eosList = json["eos_token_id"] as? [Int] {
                for e in eosList { stops.insert(e) }
            }
        }
        if let eot = endOfTurnID { stops.insert(eot) }
        if stops.isEmpty {
            stops.insert(tk.eosTokenId)
        }

        self.inner = tk
        self.bosBegin = bosBeginID
        self.turnMarker = turnMarkerID
        self.endOfTurn = endOfTurnID
        self.thinkStart = thinkStartID
        self.thinkEnd = thinkEndID
        self.stopTokenIds = stops
        let chatTemplate = tokCfg["chat_template"] as? String ?? ""
        if chatTemplate.contains("]~!b[") || chatTemplate.contains("]~b]") || chatTemplate.isEmpty && bosBeginID != nil && turnMarkerID != nil {
            self.chatTemplateStyle = .minimax
        } else if chatTemplate.contains("<|im_start|>") {
            self.chatTemplateStyle = .qwenIM
        } else {
            self.chatTemplateStyle = .roleNewline
        }
        self.startsWithThinking = chatTemplate.contains("<think>")
        self.defaultSystemPrompt =
            "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
    }

    public var vocabSize: Int { inner.vocabSize }

    /// Encode a chat conversation per MiniMax's chat_template.jinja layout.
    /// `messages` should NOT include a system message — pass it via `system`
    /// so the default can fill in. To suppress the system message entirely
    /// pass `system: ""`.
    public func applyChatTemplate(
        messages: [JANGTQChatMessage],
        system: String? = nil,
        addGenerationPrompt: Bool = true,
        startWithThink: Bool? = nil
    ) -> [Int] {
        switch chatTemplateStyle {
        case .minimax:
            return applyMiniMaxChatTemplate(
                messages: messages,
                system: system,
                addGenerationPrompt: addGenerationPrompt,
                startWithThink: startWithThink ?? true
            )
        case .qwenIM:
            return applyQwenIMChatTemplate(
                messages: messages,
                system: system,
                addGenerationPrompt: addGenerationPrompt,
                startWithThink: startWithThink ?? startsWithThinking
            )
        case .roleNewline:
            return applyRoleNewlineChatTemplate(
                messages: messages,
                system: system,
                addGenerationPrompt: addGenerationPrompt,
                startWithThink: startWithThink ?? startsWithThinking
            )
        }
    }

    private func applyMiniMaxChatTemplate(
        messages: [JANGTQChatMessage],
        system: String?,
        addGenerationPrompt: Bool,
        startWithThink: Bool
    ) -> [Int] {
        var tokens: [Int] = []

        // Opening: ]~!b[ ]~b]system
        if let bos = bosBegin { tokens.append(bos) }
        if let turn = turnMarker {
            tokens.append(turn)
            tokens.append(contentsOf: inner.encode("system\n"))
        } else {
            tokens.append(contentsOf: inner.encode("system\n"))
        }

        // System content
        let sys = system ?? defaultSystemPrompt
        if !sys.isEmpty {
            tokens.append(contentsOf: inner.encode(sys))
        }
        if let eot = endOfTurn { tokens.append(eot) }
        tokens.append(contentsOf: inner.encode("\n"))

        // Each message turn
        for msg in messages {
            if let turn = turnMarker {
                tokens.append(turn)
                tokens.append(contentsOf: inner.encode("\(msg.role)\n"))
            } else {
                tokens.append(contentsOf: inner.encode("\(msg.role)\n"))
            }
            tokens.append(contentsOf: inner.encode(msg.content))
            if let eot = endOfTurn { tokens.append(eot) }
            tokens.append(contentsOf: inner.encode("\n"))
        }

        // Generation prompt: ]~b]ai
        if addGenerationPrompt {
            if let turn = turnMarker {
                tokens.append(turn)
                tokens.append(contentsOf: inner.encode("ai\n"))
            } else {
                tokens.append(contentsOf: inner.encode("ai\n"))
            }
            // The MiniMax model expects to start in thinking mode by default.
            // The Jinja template appends `<think>\n` after `]~b]ai\n`.
            if startWithThink, let think = thinkStart {
                tokens.append(think)
                tokens.append(contentsOf: inner.encode("\n"))
            }
        }

        return tokens
    }

    private func applyQwenIMChatTemplate(
        messages: [JANGTQChatMessage],
        system: String?,
        addGenerationPrompt: Bool,
        startWithThink: Bool
    ) -> [Int] {
        var tokens: [Int] = []

        if let system, !system.isEmpty {
            if let imStart = inner.imStartId { tokens.append(imStart) }
            tokens.append(contentsOf: inner.encode("system\n\(system)"))
            if let imEnd = inner.imEndId { tokens.append(imEnd) }
            tokens.append(contentsOf: inner.encode("\n"))
        }

        for msg in messages {
            if let imStart = inner.imStartId { tokens.append(imStart) }
            tokens.append(contentsOf: inner.encode("\(msg.role)\n\(msg.content)"))
            if let imEnd = inner.imEndId { tokens.append(imEnd) }
            tokens.append(contentsOf: inner.encode("\n"))
        }

        if addGenerationPrompt {
            if let imStart = inner.imStartId { tokens.append(imStart) }
            tokens.append(contentsOf: inner.encode("assistant\n"))
            if startWithThink, let think = thinkStart {
                tokens.append(think)
                tokens.append(contentsOf: inner.encode("\n"))
            }
        }

        return tokens
    }

    private func applyRoleNewlineChatTemplate(
        messages: [JANGTQChatMessage],
        system: String?,
        addGenerationPrompt: Bool,
        startWithThink: Bool
    ) -> [Int] {
        var tokens: [Int] = []

        if let system, !system.isEmpty {
            tokens.append(contentsOf: inner.encode("system\n\(system)"))
            if let eot = endOfTurn { tokens.append(eot) }
            tokens.append(contentsOf: inner.encode("\n"))
        }

        for msg in messages {
            tokens.append(contentsOf: inner.encode("\(msg.role)\n\(msg.content)"))
            if let eot = endOfTurn { tokens.append(eot) }
            tokens.append(contentsOf: inner.encode("\n"))
        }

        if addGenerationPrompt {
            tokens.append(contentsOf: inner.encode("assistant\n"))
            if startWithThink, let think = thinkStart {
                tokens.append(think)
                tokens.append(contentsOf: inner.encode("\n"))
            }
        }

        return tokens
    }

    public func encode(_ text: String) -> [Int] { inner.encode(text) }
    public func decode(_ ids: [Int]) -> String { inner.decode(ids) }
    public func decodePreservingSpecialTokens(_ ids: [Int]) -> String {
        inner.decodePreservingSpecialTokens(ids)
    }
    public func decodeToken(_ id: Int) -> String { inner.decodeToken(id) }

    public func tokenHasVisibleText(_ id: Int) -> Bool {
        inner.decodeToken(id).unicodeScalars.contains { scalar in
            !CharacterSet.whitespacesAndNewlines.contains(scalar)
                && !CharacterSet.controlCharacters.contains(scalar)
        }
    }

    public var suppressedGenerationTokenIds: Set<Int> {
        var suppressed = inner.specialTokenIdSet
        stopTokenIds.forEach { suppressed.remove($0) }
        if let thinkEnd { suppressed.remove(thinkEnd) }
        return suppressed
    }

    public var leadingInvisibleGenerationTokenIds: Set<Int> {
        var invisible: Set<Int> = []
        for id in 0..<vocabSize where !tokenHasVisibleText(id) {
            invisible.insert(id)
        }
        stopTokenIds.forEach { invisible.remove($0) }
        if let thinkEnd { invisible.remove(thinkEnd) }
        return invisible
    }

    /// Strip `<think>...</think>` content from a decoded answer string.
    /// Used after generation to extract just the user-visible answer.
    public func stripThinking(_ text: String) -> String {
        var out = text
        while let openRange = out.range(of: "<think>"),
              let closeRange = out.range(of: "</think>", range: openRange.upperBound..<out.endIndex) {
            out.replaceSubrange(openRange.lowerBound..<closeRange.upperBound, with: "")
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
