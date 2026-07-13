//
// JANGTQ tokenizer + generator integration tests.
// Builds a tiny vocab/merges/tokenizer.json + chat_template + generation_config
// inside a temp dir and verifies the wrapper picks up MiniMax-specific tokens.
//

import Foundation
import XCTest
@testable import JANG

final class JANGTQTokenizerTests: XCTestCase {

    private func buildTinyTokenizerDir() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_tok_test_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // Minimal tokenizer.json with a vocab, no merges, and added MiniMax tokens.
        // Real BPE is exercised by the existing JANGTokenizer tests; here we just
        // verify that the wrapper finds the right special token IDs.
        let tokenizerJson: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": [
                    "h": 1, "i": 2, "Ġ": 3, "system": 4, "user": 5, "ai": 6,
                    "\n": 7,
                ],
                "merges": [],
            ],
            "added_tokens": [
                ["id": 200000, "content": "]!p~[", "special": true],
                ["id": 200019, "content": "]~b]",  "special": true],
                ["id": 200020, "content": "[e~[",  "special": true],
                ["id": 200034, "content": "]~!b[", "special": true],
                ["id": 200050, "content": "<think>",  "special": true],
                ["id": 200051, "content": "</think>", "special": true],
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerJson)
            .write(to: dir.appendingPathComponent("tokenizer.json"))

        // tokenizer_config.json with eos_token = [e~[
        let tokConfig: [String: Any] = [
            "tokenizer_class": "GPT2Tokenizer",
            "eos_token": "[e~[",
            "bos_token": "]~!b[",
        ]
        try JSONSerialization.data(withJSONObject: tokConfig)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))

        // generation_config.json with eos_token_id = 200020
        let genConfig: [String: Any] = [
            "eos_token_id": 200020,
            "bos_token_id": 200019,
        ]
        try JSONSerialization.data(withJSONObject: genConfig)
            .write(to: dir.appendingPathComponent("generation_config.json"))

        return dir
    }

    private func buildRoleNewlineTokenizerDir() throws -> URL {
        let dir = try buildTinyTokenizerDir()
        let tokConfig: [String: Any] = [
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "[e~[",
            "pad_token": "[e~[",
            "chat_template": "{% for message in messages %}{{ message['role'] }}\n{{ message['content'] }}[e~[\n{% endfor %}assistant\n",
        ]
        try JSONSerialization.data(withJSONObject: tokConfig)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        return dir
    }

    func testWrapperFindsMiniMaxSpecialTokens() throws {
        let dir = try buildTinyTokenizerDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try JANGTQTokenizer(modelDir: dir)
        XCTAssertEqual(tok.bosBegin,   200034, "]~!b[ should map to 200034")
        XCTAssertEqual(tok.turnMarker, 200019, "]~b] should map to 200019")
        XCTAssertEqual(tok.endOfTurn,  200020, "[e~[ should map to 200020")
        XCTAssertEqual(tok.thinkStart, 200050, "<think> should map to 200050")
        XCTAssertEqual(tok.thinkEnd,   200051, "</think> should map to 200051")
        XCTAssertTrue(tok.stopTokenIds.contains(200020),
            "stopTokenIds should include the EOS")
    }

    func testChatTemplateProducesExpectedTokenStructure() throws {
        let dir = try buildTinyTokenizerDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try JANGTQTokenizer(modelDir: dir)
        let messages = [JANGTQChatMessage(role: "user", content: "hi")]
        let ids = tok.applyChatTemplate(messages: messages, system: "")

        // Must START with bosBegin then turnMarker
        XCTAssertEqual(ids.first, tok.bosBegin)
        XCTAssertEqual(ids[1], tok.turnMarker)

        // Must contain at least one [e~[ (after system message and after user message)
        let eotCount = ids.filter { $0 == tok.endOfTurn }.count
        XCTAssertGreaterThanOrEqual(eotCount, 2)

        // Must END with <think> + newline (start of thinking mode)
        XCTAssertTrue(ids.contains(tok.thinkStart!),
            "chat template should end with <think>")
    }

    func testRoleNewlineTemplateDoesNotInjectMiniMaxControlTokens() throws {
        let dir = try buildRoleNewlineTokenizerDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try JANGTQTokenizer(modelDir: dir)
        let ids = tok.applyChatTemplate(
            messages: [JANGTQChatMessage(role: "user", content: "hi")],
            system: nil
        )

        XCTAssertEqual(tok.chatTemplateStyle, .roleNewline)
        XCTAssertFalse(ids.contains(tok.bosBegin!))
        XCTAssertFalse(ids.contains(tok.turnMarker!))
        XCTAssertFalse(ids.contains(tok.thinkStart!))
        XCTAssertTrue(ids.contains(tok.endOfTurn!))
    }

    func testDecodePreservingSpecialTokensKeepsThinkMarkers() throws {
        let dir = try buildTinyTokenizerDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try JANGTQTokenizer(modelDir: dir)
        let raw = tok.decodePreservingSpecialTokens([tok.thinkStart!, 1, 2, tok.thinkEnd!])

        XCTAssertEqual(raw, "<think>hi</think>")
        XCTAssertEqual(tok.decode([tok.thinkStart!, 1, 2, tok.thinkEnd!]), "hi")
    }

    func testLeadingInvisibleTokensSkipWhitespaceButAllowStop() throws {
        let dir = try buildTinyTokenizerDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try JANGTQTokenizer(modelDir: dir)

        XCTAssertTrue(tok.leadingInvisibleGenerationTokenIds.contains(7), "newline should be skipped before visible output")
        XCTAssertFalse(tok.leadingInvisibleGenerationTokenIds.contains(1), "visible text tokens should remain available")
        XCTAssertFalse(tok.leadingInvisibleGenerationTokenIds.contains(tok.endOfTurn!), "EOS must remain available to stop cleanly")
    }

    func testStripThinkingRemovesThinkBlocks() throws {
        let dir = try buildTinyTokenizerDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try JANGTQTokenizer(modelDir: dir)
        let raw = "<think>let me reason about this</think>The answer is Tokyo."
        let stripped = tok.stripThinking(raw)
        XCTAssertEqual(stripped, "The answer is Tokyo.")
    }

    func testStopTokenIdsFromMultiEosConfig() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_multi_eos_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        // tokenizer.json without any [e~[ token
        let tokenizerJson: [String: Any] = [
            "model": ["type": "BPE", "vocab": ["a": 1, "b": 2], "merges": []],
            "added_tokens": [],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerJson)
            .write(to: dir.appendingPathComponent("tokenizer.json"))
        let tokCfg: [String: Any] = ["eos_token": "a"]  // dummy
        try JSONSerialization.data(withJSONObject: tokCfg)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))

        // generation_config.json with multi-eos list (GLM-5.1 style)
        let genCfg: [String: Any] = ["eos_token_id": [154820, 154827, 154829]]
        try JSONSerialization.data(withJSONObject: genCfg)
            .write(to: dir.appendingPathComponent("generation_config.json"))

        let tok = try JANGTQTokenizer(modelDir: dir)
        XCTAssertEqual(tok.stopTokenIds, [154820, 154827, 154829])
    }
}
