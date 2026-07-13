//
// JANGTQ loader tests — synthetic model dir, no fixtures needed.
//
// Builds a tiny MoE model on disk with 2 experts × 1 layer and verifies:
//
//   - JANGModelConfig parses weight_format='mxtq' correctly
//   - The loader walks safetensors, groups per-expert tensors via the
//     MiniMax (w1/w2/w3) regex, and stacks them into 3D switch_mlp tensors
//   - The runtime sidecar is loaded and indexed by (in_features, seed/bits)
//   - The resulting MTLBuffer contents match the source tensor data
//
// All file I/O is in a temp directory that's torn down per test.
//

import Foundation
import XCTest
@testable import JANG
import Metal

final class JANGTQLoaderTests: XCTestCase {

    // MARK: - Tiny model fixture builder

    /// Write a 2-expert, 1-layer MiniMax-style JANGTQ model to a temp dir.
    /// Each expert has gate (w1), down (w2), up (w3) tensors with deterministic
    /// content so we can verify the loader stacks them in the right order.
    private func buildSyntheticModel() throws -> URL {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_test_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)

        // Minimal config.json — only needs fields the loader reads.
        let configJSON: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1024,
            "num_local_experts": 2,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
        ]
        let configData = try JSONSerialization.data(
            withJSONObject: configJSON, options: [.prettyPrinted]
        )
        try configData.write(to: tmp.appendingPathComponent("config.json"))

        // jang_config.json with mxtq format
        let jangCfg: [String: Any] = [
            "version": 2,
            "weight_format": "mxtq",
            "profile": "JANGTQ_2L",
            "source_model": ["name": "MiniMax-Test", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": 64, "bits_default": 2, "method": "affine+mxtq"],
        ]
        let jangData = try JSONSerialization.data(withJSONObject: jangCfg, options: [.prettyPrinted])
        try jangData.write(to: tmp.appendingPathComponent("jang_config.json"))

        // Build per-expert tensors. Shapes:
        //   gate (w1): out=32, in=64 → packed_in = 64/16 = 4
        //   down (w2): out=64, in=32 → packed_in = 32/16 = 2
        //   up   (w3): out=32, in=64 → packed_in = 4
        // Norms: half precision, shape (out,)
        try writeShard(
            url: tmp.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                makeTQTriplet(base: "model.layers.0.block_sparse_moe.experts.0.w1",
                              outFeatures: 32, packedIn: 4, expertSeed: 0),
                makeTQTriplet(base: "model.layers.0.block_sparse_moe.experts.0.w2",
                              outFeatures: 64, packedIn: 2, expertSeed: 1),
                makeTQTriplet(base: "model.layers.0.block_sparse_moe.experts.0.w3",
                              outFeatures: 32, packedIn: 4, expertSeed: 2),
                makeTQTriplet(base: "model.layers.0.block_sparse_moe.experts.1.w1",
                              outFeatures: 32, packedIn: 4, expertSeed: 100),
                makeTQTriplet(base: "model.layers.0.block_sparse_moe.experts.1.w2",
                              outFeatures: 64, packedIn: 2, expertSeed: 101),
                makeTQTriplet(base: "model.layers.0.block_sparse_moe.experts.1.w3",
                              outFeatures: 32, packedIn: 4, expertSeed: 102),
            ].flatMap { $0 }
        )

        // Sidecar: signs + codebook for in_features=64 (gate/up) and 32 (down).
        let sidecarTensors: [(name: String, dtype: String, shape: [Int], data: Data)] = [
            ("signs.64.42",      "F32", [64], makeSignsBytes(n: 64, seed: 0)),
            ("signs.32.42",      "F32", [32], makeSignsBytes(n: 32, seed: 1)),
            ("codebook.64.2",    "F32", [4],  floatsToData([-0.05, -0.015, 0.015, 0.05])),
            ("codebook.32.2",    "F32", [4],  floatsToData([-0.07, -0.02, 0.02, 0.07])),
        ]
        try writeShard(
            url: tmp.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecarTensors
        )

        return tmp
    }

    /// Make the (.tq_packed, .tq_norms, .tq_bits) triplet for one weight matrix.
    private func makeTQTriplet(
        base: String, outFeatures: Int, packedIn: Int, expertSeed: Int
    ) -> [(name: String, dtype: String, shape: [Int], data: Data)] {
        // Deterministic packed content: pack[r, c] = expertSeed * 1000 + r*8 + c
        var packedBytes = Data(capacity: outFeatures * packedIn * 4)
        for r in 0..<outFeatures {
            for c in 0..<packedIn {
                let value = UInt32(expertSeed * 1000 + r * 8 + c)
                var v = value
                packedBytes.append(Data(bytes: &v, count: 4))
            }
        }
        // Norms: half precision (2 bytes each)
        var normsBytes = Data(capacity: outFeatures * 2)
        for r in 0..<outFeatures {
            var h = Float16(Float(expertSeed) * 0.01 + Float(r) * 0.001)
            normsBytes.append(Data(bytes: &h, count: 2))
        }
        // Bits: int32 scalar
        var bitsValue: Int32 = 2
        let bitsBytes = Data(bytes: &bitsValue, count: 4)

        return [
            (base + ".tq_packed", "U32", [outFeatures, packedIn], packedBytes),
            (base + ".tq_norms",  "F16", [outFeatures],           normsBytes),
            (base + ".tq_bits",   "I32", [],                      bitsBytes),
        ]
    }

    private func makeSignsBytes(n: Int, seed: Int) -> Data {
        var bytes = Data(capacity: n * 4)
        for i in 0..<n {
            // Deterministic ±1 signs from a tiny LCG (just for the test)
            var x = UInt32((seed * 31 + i) & 0xFFFF)
            x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
            var v: Float = (x & 1) == 0 ? -1.0 : 1.0
            bytes.append(Data(bytes: &v, count: 4))
        }
        return bytes
    }

    private func floatsToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 4)
        for v in values {
            var x = v
            bytes.append(Data(bytes: &x, count: 4))
        }
        return bytes
    }

    private func halvesToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * MemoryLayout<Float16>.stride)
        for v in values {
            var x = Float16(v)
            bytes.append(Data(bytes: &x, count: MemoryLayout<Float16>.stride))
        }
        return bytes
    }

    /// Write a minimal safetensors file with the given (name, dtype, shape, data) tuples.
    /// Aligns to 8-byte boundaries for the data section.
    private func writeShard(
        url: URL,
        tensors: [(name: String, dtype: String, shape: [Int], data: Data)]
    ) throws {
        // 1. Compute offsets (data section is naturally aligned within the file
        //    since safetensors only requires the full file to be 8-byte aligned
        //    after the header).
        var headerDict: [String: Any] = [:]
        var offset = 0
        for (name, dtype, shape, data) in tensors {
            headerDict[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + data.count],
            ] as [String: Any]
            offset += data.count
        }
        let headerData = try JSONSerialization.data(withJSONObject: headerDict, options: [])
        // Pad header to 8-byte alignment
        let headerSize = headerData.count
        let paddedHeaderSize = (headerSize + 7) & ~7
        let padBytes = paddedHeaderSize - headerSize

        // 2. Assemble file: [8-byte header_size][header (padded)][tensor data...]
        var fileData = Data()
        var headerSizeLE: UInt64 = UInt64(paddedHeaderSize)
        fileData.append(Data(bytes: &headerSizeLE, count: 8))
        fileData.append(headerData)
        fileData.append(Data(repeating: 0x20, count: padBytes))  // space-pad
        for (_, _, _, data) in tensors {
            fileData.append(data)
        }

        try fileData.write(to: url)
    }

    // MARK: - Tests

    func testConfigParsesMxtqFormat() throws {
        let dir = try buildSyntheticModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try JANGModelConfig.load(from: dir)
        XCTAssertEqual(cfg.quant.format, .mxtq)
        XCTAssertEqual(cfg.quant.mxtqSeed, 42)
        XCTAssertEqual(cfg.quant.bits(for: "routed_expert"), 2)
        XCTAssertEqual(cfg.quant.bits(for: "attention"), 8)
        XCTAssertEqual(cfg.model.numLocalExperts, 2)
        XCTAssertEqual(cfg.model.numExpertsPerTok, 2)
        XCTAssertTrue(cfg.model.isMoE)
    }

    func testLoaderStacksExperts() throws {
        let dir = try buildSyntheticModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let loader = JANGTQLoader(device: device)
        let bundle = try loader.load(from: dir)

        // Should have stacked 3 groups: gate_proj, up_proj, down_proj
        // (each with n_experts=2)
        XCTAssertEqual(bundle.nStackedGroups, 3)

        let gateBase = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj"
        let downBase = "model.layers.0.block_sparse_moe.switch_mlp.down_proj"
        let upBase   = "model.layers.0.block_sparse_moe.switch_mlp.up_proj"
        guard let gate = bundle.weights[gateBase],
              let down = bundle.weights[downBase],
              let up   = bundle.weights[upBase] else {
            XCTFail("missing stacked weight; got: \(bundle.weights.keys.sorted())")
            return
        }

        XCTAssertEqual(gate.packedShape, [2, 32, 4])
        XCTAssertEqual(gate.normsShape,  [2, 32])
        XCTAssertEqual(down.packedShape, [2, 64, 2])
        XCTAssertEqual(down.normsShape,  [2, 64])
        XCTAssertEqual(up.packedShape,   [2, 32, 4])
        XCTAssertEqual(up.normsShape,    [2, 32])

        XCTAssertEqual(gate.bits, 2)
        XCTAssertEqual(gate.inFeatures, 64)
        XCTAssertEqual(gate.outFeatures, 32)
        XCTAssertEqual(gate.nExperts, 2)

        // Verify the packed bytes are stacked in expert order: expert 0 first,
        // then expert 1. Each expert's packed[r, c] = expertSeed * 1000 + r*8 + c.
        let packedPtr = gate.packed.contents().bindMemory(to: UInt32.self, capacity: 2 * 32 * 4)
        // expert 0, row 0, col 0 should be 0*1000 + 0 = 0
        XCTAssertEqual(packedPtr[0], 0)
        // expert 0, row 1, col 0 should be 0*1000 + 8 + 0 = 8
        XCTAssertEqual(packedPtr[4], 8)
        // expert 1 starts at index (32 * 4) = 128
        // expert 1, row 0, col 0 should be 100 * 1000 + 0 = 100000
        XCTAssertEqual(packedPtr[128], 100000)
        // expert 1, row 1, col 0 should be 100 * 1000 + 8 = 100008
        XCTAssertEqual(packedPtr[132], 100008)
    }

    func testLoaderReadsRuntimeSidecar() throws {
        let dir = try buildSyntheticModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let loader = JANGTQLoader(device: device)
        let bundle = try loader.load(from: dir)

        // signs.64.42 should be present
        let s64 = try bundle.sidecar.signs(inFeatures: 64, seed: 42)
        XCTAssertEqual(s64.length, 64 * 4)
        let s64Ptr = s64.contents().bindMemory(to: Float.self, capacity: 64)
        for i in 0..<64 {
            XCTAssertTrue(s64Ptr[i] == 1.0 || s64Ptr[i] == -1.0,
                "sign[\(i)] = \(s64Ptr[i]) should be ±1")
        }

        // codebook.32.2 should have 4 entries
        let cb32 = try bundle.sidecar.codebook(inFeatures: 32, bits: 2)
        XCTAssertEqual(cb32.length, 4 * 4)
        let cb32Ptr = cb32.contents().bindMemory(to: Float.self, capacity: 4)
        XCTAssertEqual(cb32Ptr[0], -0.07, accuracy: 1e-6)
        XCTAssertEqual(cb32Ptr[3],  0.07, accuracy: 1e-6)

        // codebook.64.2 should differ from cb32 (sqrt(2) factor in real models)
        let cb64 = try bundle.sidecar.codebook(inFeatures: 64, bits: 2)
        let cb64Ptr = cb64.contents().bindMemory(to: Float.self, capacity: 4)
        XCTAssertNotEqual(cb64Ptr[0], cb32Ptr[0])
    }

    func testQwenUnsanitizedNormWeightsAreShiftedOnLoad() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_qwen_norm_shift_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let configJSON: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5_MoeForCausalLM"],
            "hidden_size": 4,
            "intermediate_size": 8,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 2,
        ]
        try JSONSerialization.data(withJSONObject: configJSON, options: [.prettyPrinted])
            .write(to: dir.appendingPathComponent("config.json"))

        let jangCfg: [String: Any] = [
            "version": 2,
            "weight_format": "mxtq",
            "source_model": ["name": "QwenNormShift", "architecture": "qwen3_5_moe"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": 4, "bits_default": 2],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg, options: [.prettyPrinted])
            .write(to: dir.appendingPathComponent("jang_config.json"))

        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                (
                    "language_model.model.layers.0.linear_attn.conv1d.weight",
                    "F16",
                    [4, 1, 2],
                    halvesToData([Float](repeating: 0.0, count: 8))
                ),
                (
                    "language_model.model.layers.0.input_layernorm.weight",
                    "F16",
                    [2],
                    halvesToData([0.25, -0.5])
                ),
                (
                    "language_model.model.layers.0.post_attention_layernorm.weight",
                    "F16",
                    [2],
                    halvesToData([1.5, -1.0])
                ),
                (
                    "language_model.model.layers.0.self_attn.q_norm.weight",
                    "F16",
                    [2],
                    halvesToData([0.0, 2.0])
                ),
                (
                    "language_model.model.layers.0.self_attn.k_norm.weight",
                    "F16",
                    [2],
                    halvesToData([-0.25, 0.75])
                ),
                (
                    "language_model.model.norm.weight",
                    "F16",
                    [2],
                    halvesToData([0.5, -0.125])
                ),
                (
                    "language_model.model.layers.0.linear_attn.norm.weight",
                    "F16",
                    [2],
                    halvesToData([0.875, 1.125])
                ),
            ]
        )
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: []
        )

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let bundle = try JANGTQLoader(device: device).load(from: dir)

        func readHalfTensor(_ name: String, count: Int) throws -> [Float] {
            guard let buffer = bundle.halfTensors[name] else {
                XCTFail("missing half tensor \(name)")
                return []
            }
            let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        }

        func assertHalfTensor(_ name: String, equals expected: [Float]) throws {
            let actual = try readHalfTensor(name, count: expected.count)
            XCTAssertEqual(actual.count, expected.count)
            for i in expected.indices {
                XCTAssertEqual(actual[i], expected[i], accuracy: 1e-3, "\(name)[\(i)]")
            }
        }

        try assertHalfTensor(
            "language_model.model.layers.0.input_layernorm.weight",
            equals: [1.25, 0.5]
        )
        try assertHalfTensor(
            "language_model.model.layers.0.post_attention_layernorm.weight",
            equals: [2.5, 0.0]
        )
        try assertHalfTensor(
            "language_model.model.layers.0.self_attn.q_norm.weight",
            equals: [1.0, 3.0]
        )
        try assertHalfTensor(
            "language_model.model.layers.0.self_attn.k_norm.weight",
            equals: [0.75, 1.75]
        )
        try assertHalfTensor(
            "language_model.model.norm.weight",
            equals: [1.5, 0.875]
        )
        try assertHalfTensor(
            "language_model.model.layers.0.linear_attn.norm.weight",
            equals: [0.875, 1.125]
        )
    }

    func testMissingSidecarThrows() throws {
        let dir = try buildSyntheticModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        // Delete the sidecar
        try FileManager.default.removeItem(at: dir.appendingPathComponent("jangtq_runtime.safetensors"))

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let loader = JANGTQLoader(device: device)
        XCTAssertThrowsError(try loader.load(from: dir)) { error in
            // Should be a clear "sidecar missing, run export script" error
            let msg = String(describing: error).lowercased()
            XCTAssertTrue(msg.contains("sidecar") || msg.contains("jangtq_runtime"),
                "expected sidecar-missing error, got: \(msg)")
        }
    }
}
