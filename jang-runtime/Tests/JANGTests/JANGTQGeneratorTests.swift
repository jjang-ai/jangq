//
// JANGTQ generator end-to-end test — full pipeline: synthetic model
// + synthetic tokenizer + generator → decoded text.
//
// Verifies that the full chain (load → tokenize → forward 62 layers
// (well, 1 here) → sample → decode → strip thinking) runs without error
// and produces a deterministic non-empty result.
//

import Foundation
import XCTest
@testable import JANG
import JANGCoreMetal
import Metal

final class JANGTQGeneratorTests: XCTestCase {

    func testSamplerSkipsSuppressedControlTokens() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let buffer = device.makeBuffer(length: 5 * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            throw XCTSkip("Metal unavailable")
        }
        let logits = buffer.contents().bindMemory(to: Float.self, capacity: 5)
        logits[0] = 0.0
        logits[1] = 0.1
        logits[2] = 100.0
        logits[3] = 2.0
        logits[4] = 1.0

        let sampler = JANGTQSampler()
        XCTAssertEqual(sampler.argmax(logits: buffer, vocabSize: 5), 2)
        XCTAssertEqual(
            sampler.argmax(logits: buffer, vocabSize: 5, suppressedTokenIds: [2]),
            3
        )
    }

    private func writeShard(url: URL, tensors: [(String, String, [Int], Data)]) throws {
        var headerDict: [String: Any] = [:]
        var offset = 0
        for (name, dtype, shape, data) in tensors {
            headerDict[name] = [
                "dtype": dtype, "shape": shape,
                "data_offsets": [offset, offset + data.count],
            ] as [String: Any]
            offset += data.count
        }
        let headerData = try JSONSerialization.data(withJSONObject: headerDict)
        let paddedHeaderSize = (headerData.count + 7) & ~7
        let padBytes = paddedHeaderSize - headerData.count
        var fileData = Data()
        var sizeLE: UInt64 = UInt64(paddedHeaderSize)
        fileData.append(Data(bytes: &sizeLE, count: 8))
        fileData.append(headerData)
        fileData.append(Data(repeating: 0x20, count: padBytes))
        for (_, _, _, data) in tensors { fileData.append(data) }
        try fileData.write(to: url)
    }

    private func floatsToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 4)
        for v in values { var x = v; bytes.append(Data(bytes: &x, count: 4)) }
        return bytes
    }
    private func halvesToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 2)
        for v in values { var h = Float16(v); bytes.append(Data(bytes: &h, count: 2)) }
        return bytes
    }

    private func affine8(base: String, outF: Int, inF: Int, gs: Int = 16,
                          fillQ: UInt8 = 0, fillScale: Float = 0.0) -> [(String, String, [Int], Data)] {
        let packedIn = inF / 4
        let nGroups = inF / gs
        var w = Data(count: outF * packedIn * 4)
        w.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            let word: UInt32 = UInt32(fillQ) | UInt32(fillQ) << 8 | UInt32(fillQ) << 16 | UInt32(fillQ) << 24
            for i in 0..<(outF * packedIn) { p[i] = word }
        }
        let scales = halvesToData([Float](repeating: fillScale, count: outF * nGroups))
        let biases = halvesToData([Float](repeating: 0.0, count: outF * nGroups))
        return [
            (base + ".weight", "U32", [outF, packedIn], w),
            (base + ".scales", "F16", [outF, nGroups],  scales),
            (base + ".biases", "F16", [outF, nGroups],  biases),
        ]
    }

    private func tqTriplet(base: String, outF: Int, packedIn: Int) -> [(String, String, [Int], Data)] {
        var p = Data(count: outF * packedIn * 4)
        p.withUnsafeMutableBytes { raw in
            let q = raw.bindMemory(to: UInt32.self)
            for i in 0..<(outF * packedIn) { q[i] = 0 }
        }
        var bv: Int32 = 2
        return [
            (base + ".tq_packed", "U32", [outF, packedIn], p),
            (base + ".tq_norms",  "F16", [outF],           halvesToData([Float](repeating: 1.0, count: outF))),
            (base + ".tq_bits",   "I32", [],               Data(bytes: &bv, count: 4)),
        ]
    }

    private func makeNorm(_ name: String, dim: Int) -> (String, String, [Int], Data) {
        (name, "F16", [dim], halvesToData([Float](repeating: 1.0, count: dim)))
    }

    private func buildModel() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_gen_e2e_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // Tiny dims that satisfy 8-bit GEMV constraints (in_features % 4 == 0,
        // in_features % group_size == 0)
        let hidden = 64, inter = 32
        let nHeads = 4, nKVHeads = 2, headDim = 16
        let nExperts = 4, K = 2
        let vocab = 256
        let gs = 16

        // config.json
        let cfg: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": hidden, "intermediate_size": 128,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 1,
            "num_attention_heads": nHeads, "num_key_value_heads": nKVHeads,
            "head_dim": headDim,
            "vocab_size": vocab,
            "num_local_experts": nExperts, "num_experts_per_tok": K,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        ]
        try JSONSerialization.data(withJSONObject: cfg)
            .write(to: dir.appendingPathComponent("config.json"))

        // jang_config.json
        let jangCfg: [String: Any] = [
            "version": 2, "weight_format": "mxtq", "profile": "JANGTQ_2L",
            "source_model": ["name": "Test", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": gs, "bits_default": 2],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        // tokenizer.json with MiniMax special tokens at the right IDs
        // (pick small IDs since vocab=256)
        let tokenizerJson: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": [
                    "h": 1, "i": 2, "Ġ": 3, "system": 4, "user": 5, "ai": 6,
                    "\n": 7, "?": 8, "T": 9, "a": 10, "k": 11, "y": 12, "o": 13,
                ],
                "merges": [],
            ],
            "added_tokens": [
                ["id": 200, "content": "]~b]",   "special": true],
                ["id": 201, "content": "[e~[",   "special": true],
                ["id": 202, "content": "]~!b[",  "special": true],
                ["id": 203, "content": "<think>",  "special": true],
                ["id": 204, "content": "</think>", "special": true],
            ],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerJson)
            .write(to: dir.appendingPathComponent("tokenizer.json"))
        let tokCfg: [String: Any] = ["eos_token": "[e~[", "tokenizer_class": "GPT2Tokenizer"]
        try JSONSerialization.data(withJSONObject: tokCfg)
            .write(to: dir.appendingPathComponent("tokenizer_config.json"))
        let genCfg: [String: Any] = ["eos_token_id": 201]
        try JSONSerialization.data(withJSONObject: genCfg)
            .write(to: dir.appendingPathComponent("generation_config.json"))

        // Build all tensors
        var tensors: [(String, String, [Int], Data)] = []
        tensors.append(contentsOf: affine8(
            base: "model.embed_tokens", outF: vocab, inF: hidden,
            fillQ: 1, fillScale: 0.01
        ))
        tensors.append(contentsOf: affine8(
            base: "lm_head", outF: vocab, inF: hidden,
            fillQ: 1, fillScale: 0.01
        ))
        tensors.append(makeNorm("model.norm.weight", dim: hidden))
        tensors.append(makeNorm("model.layers.0.input_layernorm.weight", dim: hidden))
        tensors.append(makeNorm("model.layers.0.post_attention_layernorm.weight", dim: hidden))
        tensors.append(makeNorm("model.layers.0.self_attn.q_norm.weight", dim: nHeads * headDim))
        tensors.append(makeNorm("model.layers.0.self_attn.k_norm.weight", dim: nKVHeads * headDim))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.q_proj", outF: nHeads * headDim, inF: hidden
        ))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.k_proj", outF: nKVHeads * headDim, inF: hidden
        ))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.v_proj", outF: nKVHeads * headDim, inF: hidden
        ))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.o_proj", outF: hidden, inF: nHeads * headDim
        ))
        let pcGate = hidden / 16, pcDown = inter / 16
        for e in 0..<nExperts {
            tensors.append(contentsOf: tqTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w1",
                outF: inter, packedIn: pcGate
            ))
            tensors.append(contentsOf: tqTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w2",
                outF: hidden, packedIn: pcDown
            ))
            tensors.append(contentsOf: tqTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w3",
                outF: inter, packedIn: pcGate
            ))
        }
        tensors.append((
            "model.layers.0.block_sparse_moe.gate.weight", "F16", [nExperts, hidden],
            halvesToData([Float](repeating: 0.0, count: nExperts * hidden))
        ))
        tensors.append((
            "model.layers.0.block_sparse_moe.e_score_correction_bias", "F16", [nExperts],
            halvesToData([Float](repeating: 0.0, count: nExperts))
        ))
        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: tensors
        )

        // Sidecar
        let sidecar: [(String, String, [Int], Data)] = [
            ("signs.\(hidden).42", "F32", [hidden], floatsToData([Float](repeating: 1.0, count: hidden))),
            ("signs.\(inter).42",  "F32", [inter],  floatsToData([Float](repeating: 1.0, count: inter))),
            ("codebook.\(hidden).2", "F32", [4], floatsToData([0, 0, 0, 0])),
            ("codebook.\(inter).2",  "F32", [4], floatsToData([0, 0, 0, 0])),
        ]
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecar
        )

        return dir
    }

    func testFullEndToEndGeneration() throws {
        let dir = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let model = try JANGTQModel(bundle: bundle, context: ctx, maxSeqLen: 32)
        let tok = try JANGTQTokenizer(modelDir: dir)
        let gen = JANGTQGenerator(model: model, tokenizer: tok)

        // Sanity-check: tokenizer found the synthetic special tokens
        XCTAssertEqual(tok.endOfTurn, 201)
        XCTAssertEqual(tok.thinkStart, 203)
        XCTAssertTrue(tok.stopTokenIds.contains(201))

        // Run generate — small max_tokens because all-zero MoE means
        // the model generates the same token forever (lm_head is all
        // identical so argmax returns token 0).
        let result = try gen.generate(
            userMessage: "hi",
            system: "",
            maxTokens: 5,
            verbose: false
        )

        // We expect maxTokens to be hit (no stop token because all logits
        // are equal and argmax picks 0, which is not a stop).
        XCTAssertEqual(result.outputTokens, 5)
        XCTAssertEqual(result.stopReason, .maxTokens)
        XCTAssertGreaterThan(result.promptTokens, 0)
        XCTAssertGreaterThan(result.elapsedSec, 0)
        XCTAssertGreaterThan(result.tokensPerSec, 0)
    }
}
