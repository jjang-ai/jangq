//
// End-to-end JANGTQ decoder engine test — runs the full MoE block
// (router + JANGTQMoEBlock + combine) on a synthetic model and verifies
// the math against an analytical reference.
//

import Foundation
import XCTest
@testable import JANG
import JANGCoreMetal
import Metal

final class JANGTQDecoderEngineTests: XCTestCase {

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
        for v in values {
            var x = v
            bytes.append(Data(bytes: &x, count: 4))
        }
        return bytes
    }

    private func halvesToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 2)
        for v in values {
            var h = Float16(v)
            bytes.append(Data(bytes: &h, count: 2))
        }
        return bytes
    }

    private func tqTriplet(
        base: String, outFeatures: Int, packedIn: Int
    ) -> [(String, String, [Int], Data)] {
        var packedBytes = Data(count: outFeatures * packedIn * 4)
        packedBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            for i in 0..<(outFeatures * packedIn) { p[i] = 0 }  // every weight = codebook[0]
        }
        let normsBytes = halvesToData([Float](repeating: 1.0, count: outFeatures))
        var bitsValue: Int32 = 2
        let bitsBytes = Data(bytes: &bitsValue, count: 4)
        return [
            (base + ".tq_packed", "U32", [outFeatures, packedIn], packedBytes),
            (base + ".tq_norms",  "F16", [outFeatures],           normsBytes),
            (base + ".tq_bits",   "I32", [],                      bitsBytes),
        ]
    }

    private func buildModel(
        moePrefix: String = "block_sparse_moe",
        qwenProjectionNames: Bool = false
    ) throws -> (URL, hidden: Int, inter: Int, K: Int, nExperts: Int) {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_engine_test_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let hidden = 64
        let inter = 32
        let nExperts = 4
        let K = 2

        let cfg: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": 128,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "num_local_experts": nExperts,
            "num_experts_per_tok": K,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
        ]
        try JSONSerialization.data(withJSONObject: cfg)
            .write(to: dir.appendingPathComponent("config.json"))
        let jangCfg: [String: Any] = [
            "version": 2, "weight_format": "mxtq", "profile": "JANGTQ_2L",
            "source_model": ["name": "Test", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2],
            "quantization": ["group_size": 64, "bits_default": 2, "method": "affine+mxtq"],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        // MoE expert weights — all packed=0, all norms=1
        var tensors: [(String, String, [Int], Data)] = []
        let pcGate = hidden / 16  // 4
        let pcDown = inter / 16   // 2
        let layerPrefix = "model.layers.0.\(moePrefix)"
        let gateProjection = qwenProjectionNames ? "gate_proj" : "w1"
        let downProjection = qwenProjectionNames ? "down_proj" : "w2"
        let upProjection = qwenProjectionNames ? "up_proj" : "w3"
        for e in 0..<nExperts {
            tensors.append(contentsOf: tqTriplet(
                base: "\(layerPrefix).experts.\(e).\(gateProjection)",
                outFeatures: inter, packedIn: pcGate
            ))
            tensors.append(contentsOf: tqTriplet(
                base: "\(layerPrefix).experts.\(e).\(downProjection)",
                outFeatures: hidden, packedIn: pcDown
            ))
            tensors.append(contentsOf: tqTriplet(
                base: "\(layerPrefix).experts.\(e).\(upProjection)",
                outFeatures: inter, packedIn: pcGate
            ))
        }

        // Router gate (n_experts × hidden) — set so that experts 0 and 1 always win.
        // Easy way: set gate weights so expert 0 has the largest sum.
        var gateW = [Float](repeating: 0.001, count: nExperts * hidden)
        // Boost experts 0 and 1
        for h in 0..<hidden { gateW[0 * hidden + h] = 1.0 }
        for h in 0..<hidden { gateW[1 * hidden + h] = 0.5 }
        tensors.append((
            "\(layerPrefix).gate.weight",
            "F16", [nExperts, hidden], halvesToData(gateW)
        ))
        if !qwenProjectionNames {
            tensors.append((
                "\(layerPrefix).e_score_correction_bias",
                "F16", [nExperts], halvesToData([Float](repeating: 0, count: nExperts))
            ))
        }

        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: tensors
        )

        // Sidecar
        let sidecar: [(String, String, [Int], Data)] = [
            ("signs.\(hidden).42", "F32", [hidden], floatsToData([Float](repeating: 1.0, count: hidden))),
            ("signs.\(inter).42",  "F32", [inter],  floatsToData([Float](repeating: 1.0, count: inter))),
            ("codebook.\(hidden).2", "F32", [4], floatsToData([1.0, 0, 0, 0])),
            ("codebook.\(inter).2",  "F32", [4], floatsToData([1.0, 0, 0, 0])),
        ]
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecar
        )

        return (dir, hidden: hidden, inter: inter, K: K, nExperts: nExperts)
    }

    // MARK: - Tests

    func testRouterCPUSigmoidBiasSelectsExpectedExperts() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }

        // Gates: highest at index 5, then 2, then 7. With K=3, top-3 should be {5, 2, 7}.
        let gates: [Float] = [0.1, 0.2, 0.5, -0.1, 0.0, 0.9, 0.3, 0.7]
        let bias  = [Float](repeating: 0, count: 8)
        let result = try jangtqRouterCPU(
            gates: gates, eScoreBias: bias, k: 3,
            variant: .sigmoidBias, device: device
        )

        XCTAssertEqual(Set(result.indices), Set([5, 2, 7]))
        let s: Float = result.scores.reduce(0, +)
        XCTAssertEqual(s, 1.0, accuracy: 1e-5)
    }

    /// Qwen3-Next / Qwen3.5 / Qwen3.6 router: softmax → top-k, optional renorm.
    func testRouterCPUSoftmaxTopKSelectsExpectedExperts() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        // Same logits — softmax is monotonic so the top-k set must still be {5, 2, 7}.
        let gates: [Float] = [0.1, 0.2, 0.5, -0.1, 0.0, 0.9, 0.3, 0.7]
        let result = try jangtqRouterCPU(
            gates: gates, eScoreBias: nil, k: 3,
            variant: .softmaxTopK(renormalize: true), device: device
        )
        XCTAssertEqual(Set(result.indices), Set([5, 2, 7]))
        let s: Float = result.scores.reduce(0, +)
        XCTAssertEqual(s, 1.0, accuracy: 1e-5, "renormalized scores must sum to 1")

        // Without renormalize, scores should be the raw softmax values (< 1 total).
        let raw = try jangtqRouterCPU(
            gates: gates, eScoreBias: nil, k: 3,
            variant: .softmaxTopK(renormalize: false), device: device
        )
        let rawSum: Float = raw.scores.reduce(0, +)
        XCTAssertLessThan(rawSum, 1.0, "raw softmax top-3 sum must be < 1")
        XCTAssertGreaterThan(rawSum, 0.5, "top-3 of 8 should capture most of the mass")
    }

    func testRouterCPUMaskExcludesDisabledExpertsAndRenormalizes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let gates: [Float] = [0.1, 0.2, 0.5, -0.1, 0.0, 0.9, 0.3, 0.7]

        let result = try jangtqRouterCPU(
            gates: gates, eScoreBias: nil, k: 3,
            variant: .softmaxTopK(renormalize: false), device: device,
            disabledExperts: [5, 7]
        )

        XCTAssertFalse(result.indices.contains(5))
        XCTAssertFalse(result.indices.contains(7))
        XCTAssertEqual(Set(result.indices), Set([2, 6, 1]))
        XCTAssertEqual(result.scores.reduce(0, +), 1.0, accuracy: 1e-5,
            "masked router scores must renormalize even when the base router does not")
    }

    func testRouterCPUMaskRejectsTooFewActiveExperts() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        XCTAssertThrowsError(try jangtqRouterCPU(
            gates: [0.1, 0.2, 0.3], eScoreBias: nil, k: 2,
            variant: .softmaxTopK(renormalize: true), device: device,
            disabledExperts: [0, 1]
        ))
    }

    func testFullMoEForwardEndToEnd() throws {
        let (dir, hidden, inter, K, _) = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }

        // Load
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let kernels = try JANGTQKernels(context: ctx)
        let affine8 = try JANGTQAffine8Matmul(context: ctx)
        let opsK = try JANGTQDecodeOps(context: ctx)
        let cache = try JANGTQKVCache(
            device: device, nLayers: 1, kvHeads: 2, headDim: 16, maxSeqLen: 16
        )

        let engine = JANGTQDecoderEngine(
            bundle: bundle, context: ctx, kernels: kernels, affine8: affine8,
            ops: opsK, cache: cache
        )

        // Build an all-ones half input (`hidden,`)
        let x = device.makeBuffer(length: hidden * MemoryLayout<Float16>.stride,
                                   options: .storageModeShared)!
        let xPtr = x.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { xPtr[i] = 1.0 }

        let combined = try engine.runMoE(layer: 0, normedX: x, hidden: hidden, k: K)
        XCTAssertEqual(combined.length, hidden * MemoryLayout<Float16>.stride)

        // Analytical reference (same as JANGTQMoEBlockTests):
        //   sum(x_rot) = sqrt(hidden)
        //   gate = up = sum(x_rot)
        //   x_act = SiLU(s) * s
        //   sum(x_act_rot) = sqrt(inter) * x_act (Hadamard of constant)
        //   y[k, r] = sqrt(inter) * x_act for every (k, r)
        //   combined[r] = sum_k scores[k] * y[k, r] = sqrt(inter) * x_act * sum(scores)
        //                = sqrt(inter) * x_act * 1.0  (scores normalize to 1)
        let s: Float = sqrt(Float(hidden))
        let xAct: Float = (s / (1.0 + Float(exp(-Double(s))))) * s
        let expected: Float = sqrt(Float(inter)) * xAct

        let outPtr = combined.contents().bindMemory(to: Float16.self, capacity: hidden)
        var maxDiff: Float = 0
        for i in 0..<hidden {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - expected))
        }
        XCTAssertLessThan(maxDiff, 1.0,
            "decoder engine MoE forward max diff = \(maxDiff), expected ≈ \(expected)")
    }

    func testRunMoEAcceptsTextQwenMLPPrefix() throws {
        let (dir, hidden, _, K, _) = try buildModel(moePrefix: "mlp", qwenProjectionNames: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let engine = JANGTQDecoderEngine(
            bundle: bundle,
            context: ctx,
            kernels: try JANGTQKernels(context: ctx),
            affine8: try JANGTQAffine8Matmul(context: ctx),
            ops: try JANGTQDecodeOps(context: ctx),
            cache: try JANGTQKVCache(device: device, nLayers: 1, kvHeads: 2, headDim: 16, maxSeqLen: 16)
        )
        let x = device.makeBuffer(length: hidden * MemoryLayout<Float16>.stride,
                                  options: .storageModeShared)!
        let xPtr = x.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { xPtr[i] = 1.0 }

        let combined = try engine.runMoE(layer: 0, normedX: x, hidden: hidden, k: K)
        XCTAssertEqual(combined.length, hidden * MemoryLayout<Float16>.stride)
    }

    func testRunMoEDefaultsMissingNormTopKProbToRawSoftmaxScores() throws {
        let (dir, hidden, _, K, nExperts) = try buildModel(moePrefix: "mlp", qwenProjectionNames: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let gateKey = "model.layers.0.mlp.gate.weight"
        let gate = try XCTUnwrap(bundle.halfTensors[gateKey])
        let gatePtr = gate.contents().bindMemory(to: Float16.self, capacity: nExperts * hidden)
        for expert in 0..<nExperts {
            for h in 0..<hidden {
                gatePtr[expert * hidden + h] = Float16(Float(expert + 1) * 0.001)
            }
        }

        let ctx = try MetalContext()
        let engine = JANGTQDecoderEngine(
            bundle: bundle,
            context: ctx,
            kernels: try JANGTQKernels(context: ctx),
            affine8: try JANGTQAffine8Matmul(context: ctx),
            ops: try JANGTQDecodeOps(context: ctx),
            cache: try JANGTQKVCache(device: device, nLayers: 1, kvHeads: 2, headDim: 16, maxSeqLen: 16)
        )
        let x = device.makeBuffer(length: hidden * MemoryLayout<Float16>.stride,
                                  options: .storageModeShared)!
        let xPtr = x.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { xPtr[i] = 1.0 }

        let collector = JANGExpertTraceCollector()
        _ = try engine.runMoE(
            layer: 0,
            normedX: x,
            hidden: hidden,
            k: K,
            traceConfig: JANGExpertTraceConfig(emitTokenTrace: true, maxTraceTokens: 8),
            traceCollector: collector,
            tokenIndex: 0
        )

        let record = try XCTUnwrap(collector.snapshot().first)
        XCTAssertEqual(Set(record.selectedExperts), Set([2, 3]))
        XCTAssertLessThan(record.scores.reduce(0, +), 1.0)
        XCTAssertGreaterThan(record.scores.reduce(0, +), 0.45)
    }

    func testRunMoERecordsTraceAndAppliesMask() throws {
        let (dir, hidden, _, K, _) = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let engine = JANGTQDecoderEngine(
            bundle: bundle,
            context: ctx,
            kernels: try JANGTQKernels(context: ctx),
            affine8: try JANGTQAffine8Matmul(context: ctx),
            ops: try JANGTQDecodeOps(context: ctx),
            cache: try JANGTQKVCache(device: device, nLayers: 1, kvHeads: 2, headDim: 16, maxSeqLen: 16)
        )
        let x = device.makeBuffer(length: hidden * MemoryLayout<Float16>.stride,
                                  options: .storageModeShared)!
        let xPtr = x.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { xPtr[i] = 1.0 }

        let collector = JANGExpertTraceCollector()
        let traceConfig = JANGExpertTraceConfig(
            mask: JANGExpertMask(layers: [0: [0]], topKOverride: 1),
            emitTokenTrace: true,
            maxTraceTokens: 8
        )
        _ = try engine.runMoE(
            layer: 0,
            normedX: x,
            hidden: hidden,
            k: K,
            traceConfig: traceConfig,
            traceCollector: collector,
            tokenIndex: 12
        )

        let records = collector.snapshot()
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records[0].tokenIndex, 12)
        XCTAssertEqual(records[0].layer, 0)
        XCTAssertEqual(records[0].effectiveTopK, 1)
        XCTAssertEqual(records[0].disabledExperts, [0])
        XCTAssertFalse(records[0].selectedExperts.contains(0))
    }

    func testRMSNormCPUMatchesAnalyticalForOnes() throws {
        let dim = 64
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let xBuf = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let xPtr = xBuf.contents().bindMemory(to: Float16.self, capacity: dim)
        for i in 0..<dim { xPtr[i] = 1.0 }
        let gBuf = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let gPtr = gBuf.contents().bindMemory(to: Float16.self, capacity: dim)
        for i in 0..<dim { gPtr[i] = 1.0 }

        // For all-ones input: rms = 1, output = input * 1 * gamma = ones
        let outBuf = jangtqRMSNormCPU(inputBuf: xBuf, gammaBuf: gBuf, dim: dim, eps: 1e-5)
        let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: dim)
        for i in 0..<dim { XCTAssertEqual(Float(outPtr[i]), 1.0, accuracy: 1e-3) }
    }

    func testEmbeddingDequantizeRow() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }

        // Build a tiny embed_tokens: vocab=4, hidden=16, group_size=16
        // q_int = i (0..15) per group, scale = 1, bias = 0 → row[i] = i
        let outF = 4
        let inF = 16
        let gs = 16
        let valsPerU32 = 4  // 8-bit
        let packedIn = inF / valsPerU32

        var weightBytes = Data(count: outF * packedIn * 4)
        weightBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            for r in 0..<outF {
                // word k holds vals 4k..4k+3, each with q_int = (4k+0)..(4k+3)
                for k in 0..<packedIn {
                    let v0 = UInt32(4 * k + 0)
                    let v1 = UInt32(4 * k + 1) << 8
                    let v2 = UInt32(4 * k + 2) << 16
                    let v3 = UInt32(4 * k + 3) << 24
                    p[r * packedIn + k] = v0 | v1 | v2 | v3
                }
            }
        }
        var scalesBytes = Data(count: outF * 1 * 2)  // 1 group
        scalesBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for r in 0..<outF { p[r] = 1.0 }
        }
        var biasesBytes = Data(count: outF * 1 * 2)
        biasesBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for r in 0..<outF { p[r] = 0.0 }
        }

        let qBuf = device.makeBuffer(bytes: (weightBytes as NSData).bytes,
                                      length: weightBytes.count, options: .storageModeShared)!
        let sBuf = device.makeBuffer(bytes: (scalesBytes as NSData).bytes,
                                      length: scalesBytes.count, options: .storageModeShared)!
        let bBuf = device.makeBuffer(bytes: (biasesBytes as NSData).bytes,
                                      length: biasesBytes.count, options: .storageModeShared)!

        let embed = JANGTQAffineWeight(
            basePath: "embed", bits: 8, groupSize: gs,
            inFeatures: inF, outFeatures: outF,
            qweight: qBuf, scales: sBuf, biases: bBuf
        )

        let row = try jangtqDequantizeEmbedRow(embed: embed, tokenId: 2, device: device)
        let p = row.contents().bindMemory(to: Float16.self, capacity: inF)
        for i in 0..<inF {
            XCTAssertEqual(Float(p[i]), Float(i), accuracy: 1e-3,
                "embed row[i=\(i)] = \(Float(p[i])), expected \(i)")
        }
    }
}
