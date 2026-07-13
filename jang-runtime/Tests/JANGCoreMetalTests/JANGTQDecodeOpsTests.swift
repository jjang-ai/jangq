//
// Tests for JANGTQ decode-time GPU helper kernels.
// Each kernel is verified against an in-Swift reference implementation
// to ensure GPU and CPU paths agree to half-precision tolerance.
//

import XCTest
import Metal
@testable import JANGCoreMetal

final class JANGTQDecodeOpsTests: XCTestCase {

    private func makeHalf(_ values: [Float], _ device: MTLDevice) -> MTLBuffer {
        let buf = device.makeBuffer(length: values.count * 2, options: .storageModeShared)!
        let p = buf.contents().bindMemory(to: Float16.self, capacity: values.count)
        for i in 0..<values.count { p[i] = Float16(values[i]) }
        return buf
    }

    private func readHalf(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(p[$0]) }
    }

    private func referenceRMSNorm(_ x: [Float], _ gamma: [Float], _ eps: Float) -> [Float] {
        let dim = x.count
        var sumSq: Float = 0
        for v in x { sumSq += v * v }
        let rrms = 1.0 / Float((sumSq / Float(dim) + eps).squareRoot())
        return zip(x, gamma).map { $0 * rrms * $1 }
    }

    func testRMSNormGPUMatchesReference() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQRMSNormKernel(context: ctx)

        let dim = 256
        var x = [Float](repeating: 0, count: dim)
        for i in 0..<dim { x[i] = Float(i % 13) * 0.1 - 0.5 }
        var gamma = [Float](repeating: 0, count: dim)
        for i in 0..<dim { gamma[i] = 1.0 + Float(i % 7) * 0.01 }
        let eps: Float = 1e-5

        let xBuf = makeHalf(x, ctx.device)
        let gBuf = makeHalf(gamma, ctx.device)
        let outBuf = try kernel.run(x: xBuf, gamma: gBuf, dim: dim, eps: eps)
        let actual = readHalf(outBuf, dim)

        // Reference uses half-rounded inputs so they match what the kernel actually saw
        let xH = x.map { Float(Float16($0)) }
        let gH = gamma.map { Float(Float16($0)) }
        let expected = referenceRMSNorm(xH, gH, eps)

        var maxDiff: Float = 0
        for i in 0..<dim { maxDiff = max(maxDiff, abs(actual[i] - expected[i])) }
        XCTAssertLessThan(maxDiff, 5e-3, "RMSNorm GPU max diff = \(maxDiff)")
    }

    func testRoPEGPUMatchesReference() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQRoPEKernel(context: ctx)

        let nHeads = 4
        let headDim = 16
        let pos = 7
        let base: Float = 10000.0

        var x = [Float](repeating: 0, count: nHeads * headDim)
        for i in 0..<x.count { x[i] = Float(i % 11) * 0.05 - 0.25 }
        let xBuf = makeHalf(x, ctx.device)

        try kernel.run(qk: xBuf, nHeads: nHeads, headDim: headDim, position: pos, base: base)
        let actual = readHalf(xBuf, x.count)

        // CPU reference
        let half = headDim / 2
        var ref = x.map { Float(Float16($0)) }
        for h in 0..<nHeads {
            let rowOff = h * headDim
            for i in 0..<half {
                let freq = Foundation.pow(base, -2.0 * Float(i) / Float(headDim))
                let angle = Float(pos) * freq
                let c = Foundation.cos(angle)
                let s = Foundation.sin(angle)
                let r = ref[rowOff + i]
                let im = ref[rowOff + i + half]
                ref[rowOff + i]        = r * c - im * s
                ref[rowOff + i + half] = r * s + im * c
            }
        }

        var maxDiff: Float = 0
        for i in 0..<x.count { maxDiff = max(maxDiff, abs(actual[i] - ref[i])) }
        XCTAssertLessThan(maxDiff, 5e-3, "RoPE GPU max diff = \(maxDiff)")
    }

    func testPartialRoPERotatesOnlyRotaryDimensions() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQRoPEKernel(context: ctx)

        let nHeads = 2
        let headDim = 16
        let rotaryDim = 4
        let pos = 5
        let base: Float = 10000.0

        var x = [Float](repeating: 0, count: nHeads * headDim)
        for i in 0..<x.count { x[i] = Float(i + 1) * 0.03125 }
        let xBuf = makeHalf(x, ctx.device)

        try kernel.run(
            qk: xBuf,
            nHeads: nHeads,
            headDim: headDim,
            rotaryDim: rotaryDim,
            position: pos,
            base: base
        )
        let actual = readHalf(xBuf, x.count)

        let half = rotaryDim / 2
        var ref = x.map { Float(Float16($0)) }
        for h in 0..<nHeads {
            let rowOff = h * headDim
            for i in 0..<half {
                let freq = Foundation.pow(base, -2.0 * Float(i) / Float(rotaryDim))
                let angle = Float(pos) * freq
                let c = Foundation.cos(angle)
                let s = Foundation.sin(angle)
                let realIdx = rowOff + i
                let imagIdx = rowOff + i + half
                let r = ref[realIdx]
                let im = ref[imagIdx]
                ref[realIdx] = r * c - im * s
                ref[imagIdx] = r * s + im * c
            }
        }

        var maxDiff: Float = 0
        for i in 0..<x.count { maxDiff = max(maxDiff, abs(actual[i] - ref[i])) }
        XCTAssertLessThan(maxDiff, 5e-3, "partial RoPE GPU max diff = \(maxDiff)")

        for h in 0..<nHeads {
            let rowOff = h * headDim
            for i in rotaryDim..<headDim {
                XCTAssertEqual(
                    actual[rowOff + i],
                    Float(Float16(x[rowOff + i])),
                    accuracy: 1e-4,
                    "partial RoPE should not alter head \(h) dim \(i)"
                )
            }
        }
    }

    func testResidualAdd() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQResidualKernel(context: ctx)
        let dim = 128
        let a = (0..<dim).map { Float($0) * 0.01 }
        let b = (0..<dim).map { Float(dim - $0) * 0.01 }

        let aBuf = makeHalf(a, ctx.device)
        let bBuf = makeHalf(b, ctx.device)
        let outBuf = try kernel.run(a: aBuf, b: bBuf, dim: dim)
        let actual = readHalf(outBuf, dim)

        for i in 0..<dim {
            XCTAssertEqual(actual[i], Float(Float16(a[i])) + Float(Float16(b[i])), accuracy: 1e-3)
        }
    }

    func testCastF32ToF16() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQF32toF16Kernel(context: ctx)
        let count = 64
        var floats = [Float](repeating: 0, count: count)
        for i in 0..<count { floats[i] = Float(i) * 0.123 }

        let srcBuf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
        let p = srcBuf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { p[i] = floats[i] }

        let outBuf = try kernel.run(src: srcBuf, count: count)
        let actual = readHalf(outBuf, count)
        for i in 0..<count {
            XCTAssertEqual(actual[i], Float(Float16(floats[i])), accuracy: 1e-3)
        }
    }

    func testQwenGatedQSplitMatchesPerHeadLayout() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQGatedQSplitKernel(context: ctx)

        let nHeads = 3
        let headDim = 4
        var packed = [Float]()
        for h in 0..<nHeads {
            for d in 0..<headDim {
                packed.append(Float(100 * h + d))
            }
            for d in 0..<headDim {
                packed.append(Float(1000 + 100 * h + d))
            }
        }

        let qFull = makeHalf(packed, ctx.device)
        let q = ctx.device.makeBuffer(length: nHeads * headDim * 2, options: .storageModeShared)!
        let gate = ctx.device.makeBuffer(length: nHeads * headDim * 2, options: .storageModeShared)!

        try kernel.runInto(qFull: qFull, q: q, gate: gate, nHeads: nHeads, headDim: headDim)

        let qActual = readHalf(q, nHeads * headDim)
        let gateActual = readHalf(gate, nHeads * headDim)
        var qExpected: [Float] = []
        var gateExpected: [Float] = []
        for h in 0..<nHeads {
            for d in 0..<headDim { qExpected.append(Float(100 * h + d)) }
            for d in 0..<headDim { gateExpected.append(Float(1000 + 100 * h + d)) }
        }
        XCTAssertEqual(qActual, qExpected)
        XCTAssertEqual(gateActual, gateExpected)
    }

    func testAttentionGateAppliesSigmoidElementwise() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQAttentionGateKernel(context: ctx)

        let attn = [-2.0, -0.5, 0.25, 1.5, 3.0].map(Float.init)
        let gate = [-4.0, -1.0, 0.0, 1.0, 4.0].map(Float.init)
        let attnBuf = makeHalf(attn, ctx.device)
        let gateBuf = makeHalf(gate, ctx.device)

        try kernel.runInto(attnOut: attnBuf, gate: gateBuf, count: attn.count)

        let actual = readHalf(attnBuf, attn.count)
        for i in 0..<attn.count {
            let a = Float(Float16(attn[i]))
            let g = Float(Float16(gate[i]))
            let expected = a * (1.0 / (1.0 + Foundation.exp(-g)))
            XCTAssertEqual(actual[i], expected, accuracy: 2e-3)
        }
    }

    func testSDPAGPUMatchesReference() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQSDPAKernel(context: ctx)

        // Tiny shape: 4 query heads, 2 KV heads (GQA group_size=2), head_dim=16, cur_len=5
        let nHeads = 4, nKVHeads = 2, headDim = 16, curLen = 5, maxSeq = 8

        var q = [Float](repeating: 0, count: nHeads * headDim)
        for i in 0..<q.count { q[i] = Float(i % 7) * 0.1 - 0.3 }
        var kCache = [Float](repeating: 0, count: maxSeq * nKVHeads * headDim)
        var vCache = [Float](repeating: 0, count: maxSeq * nKVHeads * headDim)
        for i in 0..<(curLen * nKVHeads * headDim) {
            kCache[i] = Float((i * 13) % 11) * 0.05 - 0.2
            vCache[i] = Float((i * 17) % 13) * 0.04 - 0.15
        }

        let qBuf = makeHalf(q, ctx.device)
        let kBuf = makeHalf(kCache, ctx.device)
        let vBuf = makeHalf(vCache, ctx.device)
        let outBuf = try kernel.run(
            q: qBuf, kCache: kBuf, vCache: vBuf,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            curLen: curLen, maxSeq: maxSeq
        )
        let actual = readHalf(outBuf, nHeads * headDim)

        // Reference SDPA
        let qH = q.map { Float(Float16($0)) }
        let kH = kCache.map { Float(Float16($0)) }
        let vH = vCache.map { Float(Float16($0)) }
        let scale: Float = 1.0 / Float(headDim).squareRoot()
        let groupSize = nHeads / nKVHeads
        var ref = [Float](repeating: 0, count: nHeads * headDim)
        for h in 0..<nHeads {
            let kvHead = h / groupSize
            var logits = [Float](repeating: 0, count: curLen)
            for t in 0..<curLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += qH[h * headDim + d]
                         * kH[t * nKVHeads * headDim + kvHead * headDim + d]
                }
                logits[t] = dot * scale
            }
            var mx: Float = -Float.infinity
            for v in logits where v > mx { mx = v }
            var sumE: Float = 0
            for t in 0..<curLen { logits[t] = Foundation.exp(logits[t] - mx); sumE += logits[t] }
            for t in 0..<curLen { logits[t] /= sumE }
            for d in 0..<headDim {
                var acc: Float = 0
                for t in 0..<curLen {
                    acc += logits[t] * vH[t * nKVHeads * headDim + kvHead * headDim + d]
                }
                ref[h * headDim + d] = acc
            }
        }

        var maxDiff: Float = 0
        for i in 0..<actual.count { maxDiff = max(maxDiff, abs(actual[i] - ref[i])) }
        XCTAssertLessThan(maxDiff, 5e-3, "SDPA GPU max diff = \(maxDiff)")
    }
}
