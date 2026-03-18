/*
 * MXQ Tokenizer — BPE Tokenizer for HuggingFace Models
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Reads tokenizer.json (HuggingFace format) and implements BPE encoding/decoding.
 * Supports:
 * - Standard BPE with merges
 * - Byte fallback for unknown characters
 * - Special tokens (im_start, im_end, etc.)
 * - Chat template formatting
 *
 * This is a critical component — wrong tokenization = wrong model input = garbage output.
 */

import Foundation

public final class JANGTokenizer {
    // Core tokenizer data
    private let vocab: [String: Int]       // token string → token ID
    private let reverseVocab: [Int: String] // token ID → token string
    private let merges: [(String, String)]  // BPE merge pairs, in priority order
    private let mergeRanks: [String: Int]   // "tokenA tokenB" → merge priority

    // Special tokens
    public let eosTokenId: Int
    public let bosTokenId: Int?
    public let padTokenId: Int?
    private let specialTokens: [String: Int]  // special string → ID
    private let specialTokenIds: Set<Int>

    // Chat template tokens (Qwen-style)
    public let imStartId: Int?
    public let imEndId: Int?

    // Byte-level BPE mapping
    private let byteEncoder: [UInt8: String]   // byte → unicode char
    private let byteDecoder: [String: UInt8]   // unicode char → byte

    public init(tokenizerPath: URL) throws {
        let configPath = tokenizerPath.deletingLastPathComponent()
            .appendingPathComponent("tokenizer_config.json")

        // Load tokenizer config
        let configData = try Data(contentsOf: configPath)
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw JANGError.tokenizerError("Invalid tokenizer_config.json")
        }

        // Load tokenizer.json
        let tokData = try Data(contentsOf: tokenizerPath)
        guard let tokJson = try JSONSerialization.jsonObject(with: tokData) as? [String: Any] else {
            throw JANGError.tokenizerError("Invalid tokenizer.json")
        }

        guard let model = tokJson["model"] as? [String: Any] else {
            throw JANGError.tokenizerError("Missing 'model' in tokenizer.json")
        }

        // Load vocab
        guard let vocabDict = model["vocab"] as? [String: Int] else {
            throw JANGError.tokenizerError("Missing vocab in tokenizer model")
        }
        self.vocab = vocabDict

        var reverse: [Int: String] = [:]
        for (token, id) in vocabDict {
            reverse[id] = token
        }

        // Load added tokens
        let addedTokens = tokJson["added_tokens"] as? [[String: Any]] ?? []
        var specialToks: [String: Int] = [:]
        var specialIds: Set<Int> = []

        for tok in addedTokens {
            if let content = tok["content"] as? String,
               let id = tok["id"] as? Int {
                reverse[id] = content
                specialToks[content] = id
                if tok["special"] as? Bool == true {
                    specialIds.insert(id)
                }
            }
        }
        self.reverseVocab = reverse
        self.specialTokens = specialToks
        self.specialTokenIds = specialIds

        // Load merges
        guard let mergesList = model["merges"] as? [String] else {
            throw JANGError.tokenizerError("Missing merges in tokenizer model")
        }

        var parsedMerges: [(String, String)] = []
        var ranks: [String: Int] = [:]
        for (i, merge) in mergesList.enumerated() {
            let parts = merge.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                let pair = (String(parts[0]), String(parts[1]))
                parsedMerges.append(pair)
                ranks[merge] = i
            }
        }
        self.merges = parsedMerges
        self.mergeRanks = ranks

        // EOS/BOS
        let eosStr = (config["eos_token"] as? String)
            ?? (config["eos_token"] as? [String: Any])?["content"] as? String
            ?? "<|endoftext|>"
        self.eosTokenId = specialToks[eosStr] ?? vocabDict[eosStr] ?? 151643
        self.bosTokenId = nil  // Qwen doesn't use BOS
        self.padTokenId = self.eosTokenId

        // Chat template tokens
        self.imStartId = specialToks["<|im_start|>"]
        self.imEndId = specialToks["<|im_end|>"]

        // Build byte-level BPE encoding table
        // Maps bytes 0-255 to Unicode characters that are safe in the vocab
        self.byteEncoder = Self.buildByteEncoder()
        var byteDec: [String: UInt8] = [:]
        for (byte, char) in self.byteEncoder {
            byteDec[char] = byte
        }
        self.byteDecoder = byteDec
    }

    // MARK: - Encoding

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        var tokens: [Int] = []

        // Check for special tokens first
        let remaining = tokenizeWithSpecials(text)

        for segment in remaining {
            if segment.isSpecial {
                if let id = specialTokens[segment.text] ?? vocab[segment.text] {
                    tokens.append(id)
                }
            } else {
                // BPE encode the text segment
                let bpeTokens = bpeEncode(segment.text)
                tokens.append(contentsOf: bpeTokens)
            }
        }

        return tokens
    }

    /// Encode a chat message with the Qwen chat template.
    /// Default system prompt matches Qwen's chat template: "You are a helpful assistant."
    public func encodeChatPrompt(system: String? = nil, user: String) -> [Int] {
        var tokens: [Int] = []

        // System message (Qwen defaults to "You are a helpful assistant.")
        let systemMsg = system ?? "You are a helpful assistant."
        if let imStart = imStartId { tokens.append(imStart) }
        tokens.append(contentsOf: encode("system\n\(systemMsg)"))
        if let imEnd = imEndId { tokens.append(imEnd) }
        tokens.append(contentsOf: encode("\n"))

        // User message
        if let imStart = imStartId { tokens.append(imStart) }
        tokens.append(contentsOf: encode("user\n\(user)"))
        if let imEnd = imEndId { tokens.append(imEnd) }
        tokens.append(contentsOf: encode("\n"))

        // Assistant start
        if let imStart = imStartId { tokens.append(imStart) }
        tokens.append(contentsOf: encode("assistant\n"))

        return tokens
    }

    // MARK: - Decoding

    /// Decode token IDs to text.
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []

        for id in ids {
            if specialTokenIds.contains(id) {
                // Don't append special tokens to output text
                continue
            }

            guard let token = reverseVocab[id] else { continue }

            // Convert BPE token back to bytes
            for char in token {
                let charStr = String(char)
                if let byte = byteDecoder[charStr] {
                    bytes.append(byte)
                }
            }
        }

        return String(bytes: bytes, encoding: .utf8)
            ?? bytes.map({ String(UnicodeScalar($0)) }).joined()
    }

    /// Decode a single token ID to its string representation.
    public func decodeToken(_ id: Int) -> String {
        guard let token = reverseVocab[id] else { return "" }

        if specialTokenIds.contains(id) {
            return ""  // Don't print special tokens
        }

        var bytes: [UInt8] = []
        for char in token {
            if let byte = byteDecoder[String(char)] {
                bytes.append(byte)
            }
        }

        return String(bytes: bytes, encoding: .utf8)
            ?? bytes.map({ String(UnicodeScalar($0)) }).joined()
    }

    public var vocabSize: Int { return vocab.count + specialTokens.count }

    // MARK: - BPE Implementation

    private func bpeEncode(_ text: String) -> [Int] {
        // Convert text to byte-level BPE tokens
        let utf8Bytes = Array(text.utf8)

        // Map bytes to BPE characters
        let chars = utf8Bytes.map { byte -> String in
            if let encoded = byteEncoder[byte] {
                return encoded
            }
            return String(UnicodeScalar(byte))
        }

        // Apply BPE merges
        var tokens = chars

        while tokens.count > 1 {
            // Find the highest-priority merge pair
            var bestRank = Int.max
            var bestIdx = -1

            for i in 0..<(tokens.count - 1) {
                let pair = "\(tokens[i]) \(tokens[i + 1])"
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if bestIdx == -1 { break }  // No more merges possible

            // Apply the merge
            let merged = tokens[bestIdx] + tokens[bestIdx + 1]
            tokens.remove(at: bestIdx + 1)
            tokens[bestIdx] = merged
        }

        // Convert to IDs
        return tokens.compactMap { vocab[$0] }
    }

    private struct TextSegment {
        let text: String
        let isSpecial: Bool
    }

    private func tokenizeWithSpecials(_ text: String) -> [TextSegment] {
        // Split text on special token boundaries
        var segments: [TextSegment] = []
        var remaining = text

        // Sort special tokens by length (longest first) for greedy matching
        let sortedSpecials = specialTokens.keys.sorted { $0.count > $1.count }

        while !remaining.isEmpty {
            var foundSpecial = false

            for special in sortedSpecials {
                if remaining.hasPrefix(special) {
                    segments.append(TextSegment(text: special, isSpecial: true))
                    remaining = String(remaining.dropFirst(special.count))
                    foundSpecial = true
                    break
                }
            }

            if !foundSpecial {
                // Find the next special token position
                var nextSpecialIdx = remaining.endIndex
                for special in sortedSpecials {
                    if let range = remaining.range(of: special) {
                        if range.lowerBound < nextSpecialIdx {
                            nextSpecialIdx = range.lowerBound
                        }
                    }
                }

                let normalText = String(remaining[remaining.startIndex..<nextSpecialIdx])
                if !normalText.isEmpty {
                    segments.append(TextSegment(text: normalText, isSpecial: false))
                }
                remaining = String(remaining[nextSpecialIdx...])
            }
        }

        return segments
    }

    // MARK: - Byte Encoder

    /// Build the byte-to-unicode mapping used by GPT-2 style BPE.
    /// Maps bytes 0-255 to Unicode characters, keeping printable ASCII as-is
    /// and mapping everything else to Unicode code points starting at 256.
    private static func buildByteEncoder() -> [UInt8: String] {
        var encoder: [UInt8: String] = [:]

        // Printable ASCII ranges that map to themselves
        let ranges: [ClosedRange<UInt8>] = [
            33...126,  // ASCII printable (except space)
            161...172, // Latin-1 supplement
            174...255, // Latin-1 supplement continued
        ]

        for range in ranges {
            for byte in range {
                encoder[byte] = String(UnicodeScalar(byte))
            }
        }

        // Map remaining bytes (0-32, 127-160, 173) to Unicode 256+
        var unicodeOffset: UInt32 = 256
        for byte: UInt8 in 0...255 {
            if encoder[byte] == nil {
                if let scalar = UnicodeScalar(unicodeOffset) {
                    encoder[byte] = String(scalar)
                }
                unicodeOffset += 1
            }
        }

        return encoder
    }
}
