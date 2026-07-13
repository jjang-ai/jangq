/*
 * JANGTQ Qwen3.6 linear attention block.
 *
 * Decode-only native Swift implementation of the Qwen3.5/3.6 GatedDeltaNet
 * path used by hybrid `linear_attention` layers.
 */

import Foundation
import Dispatch
import Metal
import JANGCoreMetal

private struct JANGUnsafeFloatBuffer: @unchecked Sendable {
    let base: UnsafePointer<Float>

    func load(_ index: Int) -> Float {
        base[index]
    }
}

private struct JANGUnsafeMutableFloatBuffer: @unchecked Sendable {
    let base: UnsafeMutablePointer<Float>

    func load(_ index: Int) -> Float {
        base[index]
    }

    func store(_ index: Int, _ value: Float) {
        base[index] = value
    }
}

private struct JANGUnsafeHalfBuffer: @unchecked Sendable {
    let base: UnsafePointer<Float16>

    func load(_ index: Int) -> Float16 {
        base[index]
    }
}

public final class JANGTQLinearAttentionBlock {
    public let layerIndex: Int
    public let hidden: Int
    public let numKeyHeads: Int
    public let numValueHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let keyDim: Int
    public let valueDim: Int
    public let convDim: Int
    public let convKernelSize: Int
    public let normEps: Float

    public let inputLayernorm: MTLBuffer
    public let inProjQKV: JANGTQAffineWeight
    public let inProjZ: JANGTQAffineWeight
    public let inProjA: JANGTQAffineWeight
    public let inProjB: JANGTQAffineWeight
    public let outProj: JANGTQAffineWeight
    public let convWeight: MTLBuffer
    public let dtBias: MTLBuffer
    public let aLog: MTLBuffer
    public let normWeight: MTLBuffer

    public let affine8: JANGTQAffine8Matmul
    public let ops: JANGTQDecodeOps

    private var convState: [Float]
    private var deltaState: [Float]
    private let usesParallelCPU: Bool

    public init(
        layerIndex: Int,
        config: ModelConfig,
        bundle: JANGTQModelBundle,
        layerPrefix: String,
        inputLayernormPath: String,
        affine8: JANGTQAffine8Matmul,
        ops: JANGTQDecodeOps
    ) throws {
        self.layerIndex = layerIndex
        self.affine8 = affine8
        self.ops = ops
        self.hidden = config.hiddenSize
        self.numKeyHeads = config.linearNumKeyHeads ?? config.numAttentionHeads
        self.numValueHeads = config.linearNumValueHeads ?? config.numAttentionHeads
        self.keyHeadDim = config.linearKeyHeadDim ?? config.headDim
        self.valueHeadDim = config.linearValueHeadDim ?? config.headDim
        self.convKernelSize = config.linearConvKernelDim ?? 4
        self.keyDim = self.numKeyHeads * self.keyHeadDim
        self.valueDim = self.numValueHeads * self.valueHeadDim
        self.convDim = (2 * self.keyDim) + self.valueDim
        self.normEps = config.normEps

        guard numKeyHeads > 0, numValueHeads > 0, keyHeadDim > 0, valueHeadDim > 0 else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) has invalid head shape " +
                "Hk=\(numKeyHeads) Hv=\(numValueHeads) Dk=\(keyHeadDim) Dv=\(valueHeadDim)"
            )
        }
        guard numValueHeads % numKeyHeads == 0 else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) requires value heads divisible by key heads; " +
                "got Hv=\(numValueHeads), Hk=\(numKeyHeads)"
            )
        }
        guard convKernelSize > 0 else {
            throw JANGError.invalidFormat("linear_attn layer \(layerIndex) has empty conv kernel")
        }

        guard let inputNorm = bundle.halfTensors[inputLayernormPath] else {
            throw JANGError.tensorNotFound(inputLayernormPath)
        }
        self.inputLayernorm = inputNorm

        func affine(_ suffix: String) throws -> JANGTQAffineWeight {
            let key = "\(layerPrefix).\(suffix)"
            guard let weight = bundle.affineWeights[key] else {
                throw JANGError.tensorNotFound(key)
            }
            return weight
        }
        self.inProjQKV = try affine("in_proj_qkv")
        self.inProjZ = try affine("in_proj_z")
        self.inProjA = try affine("in_proj_a")
        self.inProjB = try affine("in_proj_b")
        self.outProj = try affine("out_proj")

        func half(_ suffix: String) throws -> MTLBuffer {
            let key = "\(layerPrefix).\(suffix)"
            guard let tensor = bundle.halfTensors[key] else {
                throw JANGError.tensorNotFound(key)
            }
            return tensor
        }
        self.convWeight = try half("conv1d.weight")
        self.dtBias = try half("dt_bias")
        self.aLog = try half("A_log")
        self.normWeight = try half("norm.weight")

        guard inProjQKV.inFeatures == hidden, inProjQKV.outFeatures == convDim else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) in_proj_qkv shape " +
                "\(inProjQKV.inFeatures)->\(inProjQKV.outFeatures), expected \(hidden)->\(convDim)"
            )
        }
        guard inProjZ.inFeatures == hidden, inProjZ.outFeatures == valueDim else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) in_proj_z shape " +
                "\(inProjZ.inFeatures)->\(inProjZ.outFeatures), expected \(hidden)->\(valueDim)"
            )
        }
        guard inProjA.inFeatures == hidden, inProjA.outFeatures == numValueHeads,
              inProjB.inFeatures == hidden, inProjB.outFeatures == numValueHeads
        else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) in_proj_a/b must be \(hidden)->\(numValueHeads)"
            )
        }
        guard outProj.inFeatures == valueDim, outProj.outFeatures == hidden else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) out_proj shape " +
                "\(outProj.inFeatures)->\(outProj.outFeatures), expected \(valueDim)->\(hidden)"
            )
        }
        guard convWeight.length >= convDim * convKernelSize * MemoryLayout<Float16>.stride else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) conv1d.weight too small for " +
                "convDim=\(convDim), kernel=\(convKernelSize)"
            )
        }
        guard dtBias.length >= numValueHeads * MemoryLayout<Float16>.stride,
              aLog.length >= numValueHeads * MemoryLayout<Float16>.stride,
              normWeight.length >= valueHeadDim * MemoryLayout<Float16>.stride
        else {
            throw JANGError.invalidFormat(
                "linear_attn layer \(layerIndex) has invalid dt_bias/A_log/norm.weight lengths"
            )
        }

        self.convState = [Float](repeating: 0, count: max(0, convKernelSize - 1) * convDim)
        self.deltaState = [Float](repeating: 0, count: numValueHeads * valueHeadDim * keyHeadDim)
        self.usesParallelCPU = ProcessInfo.processInfo.activeProcessorCount > 1
    }

    public func resetState() {
        for i in convState.indices {
            convState[i] = 0
        }
        for i in deltaState.indices {
            deltaState[i] = 0
        }
    }

    public func forward(x: MTLBuffer, position: Int) throws -> MTLBuffer {
        let normed = try ops.rmsnorm.run(
            x: x, gamma: inputLayernorm, dim: hidden, eps: normEps
        )

        let mixedQKV = try runAffine(inProjQKV, normed)
        let z = try runAffine(inProjZ, normed)
        let b = try runAffine(inProjB, normed)
        let a = try runAffine(inProjA, normed)

        var convOut = [Float](repeating: 0, count: convDim)
        runDepthwiseConv(mixedQKV: mixedQKV, convOut: &convOut)
        updateConvState(mixedQKV: mixedQKV)

        var q = [Float](repeating: 0, count: keyDim)
        var k = [Float](repeating: 0, count: keyDim)
        var v = [Float](repeating: 0, count: valueDim)
        for i in 0..<keyDim {
            q[i] = convOut[i]
            k[i] = convOut[keyDim + i]
        }
        for i in 0..<valueDim {
            v[i] = convOut[(2 * keyDim) + i]
        }
        normalizeQK(q: &q, k: &k)

        var deltaOut = [Float](repeating: 0, count: valueDim)
        runGatedDelta(q: q, k: k, v: v, a: a, b: b, out: &deltaOut)
        applyGatedNorm(hiddenStates: &deltaOut, gate: z)

        let valueHalf = try makeHalfBuffer(deltaOut)
        let outF32 = try affine8.run(
            qweightBuf: outProj.qweight,
            scalesBuf: outProj.scales,
            biasesBuf: outProj.biases,
            xBuf: valueHalf,
            inFeatures: outProj.inFeatures,
            outFeatures: outProj.outFeatures,
            groupSize: outProj.groupSize
        )
        return try makeHalfBuffer(outF32, count: hidden)
    }

    private func runAffine(_ weight: JANGTQAffineWeight, _ x: MTLBuffer) throws -> [Float] {
        let out = try affine8.run(
            qweightBuf: weight.qweight,
            scalesBuf: weight.scales,
            biasesBuf: weight.biases,
            xBuf: x,
            inFeatures: weight.inFeatures,
            outFeatures: weight.outFeatures,
            groupSize: weight.groupSize
        )
        let ptr = out.contents().bindMemory(to: Float.self, capacity: weight.outFeatures)
        return Array(UnsafeBufferPointer(start: ptr, count: weight.outFeatures))
    }

    private func runDepthwiseConv(mixedQKV: [Float], convOut: inout [Float]) {
        let w = convWeight.contents().bindMemory(
            to: Float16.self,
            capacity: convDim * convKernelSize
        )
        let stateRows = convKernelSize - 1
        let convDim = self.convDim
        let convKernelSize = self.convKernelSize
        let wBuf = JANGUnsafeHalfBuffer(base: w)
        mixedQKV.withUnsafeBufferPointer { mixedPtr in
            convState.withUnsafeBufferPointer { statePtr in
                convOut.withUnsafeMutableBufferPointer { outPtr in
                    guard let mixedBase = mixedPtr.baseAddress,
                          let outBase = outPtr.baseAddress
                    else { return }
                    let mixedBuf = JANGUnsafeFloatBuffer(base: mixedBase)
                    let stateBuf = JANGUnsafeFloatBuffer(base: statePtr.baseAddress ?? mixedBase)
                    let outBuf = JANGUnsafeMutableFloatBuffer(base: outBase)
                    Self.forEach(convDim, parallel: usesParallelCPU) { c in
                        var acc: Float = 0
                        if stateRows > 0 {
                            for t in 0..<stateRows {
                                acc += stateBuf.load(t * convDim + c) * Float(wBuf.load(c * convKernelSize + t))
                            }
                        }
                        acc += mixedBuf.load(c) * Float(wBuf.load(c * convKernelSize + stateRows))
                        outBuf.store(c, Self.silu(acc))
                    }
                }
            }
        }
    }

    private func updateConvState(mixedQKV: [Float]) {
        let stateRows = convKernelSize - 1
        guard stateRows > 0 else { return }
        if stateRows > 1 {
            for row in 0..<(stateRows - 1) {
                let dst = row * convDim
                let src = (row + 1) * convDim
                for c in 0..<convDim {
                    convState[dst + c] = convState[src + c]
                }
            }
        }
        let last = (stateRows - 1) * convDim
        mixedQKV.withUnsafeBufferPointer { mixedPtr in
            convState.withUnsafeMutableBufferPointer { statePtr in
                guard let mixedBase = mixedPtr.baseAddress,
                      let stateBase = statePtr.baseAddress
                else { return }
                let mixedBuf = JANGUnsafeFloatBuffer(base: mixedBase)
                let stateBuf = JANGUnsafeMutableFloatBuffer(base: stateBase)
                Self.forEach(convDim, parallel: usesParallelCPU) { c in
                    stateBuf.store(last + c, mixedBuf.load(c))
                }
            }
        }
    }

    private func normalizeQK(q: inout [Float], k: inout [Float]) {
        let invScale = 1.0 / Float(keyHeadDim).squareRoot()
        let keyHeadDim = self.keyHeadDim
        q.withUnsafeMutableBufferPointer { qPtr in
            k.withUnsafeMutableBufferPointer { kPtr in
                guard let qBase = qPtr.baseAddress,
                      let kBase = kPtr.baseAddress
                else { return }
                let qBuf = JANGUnsafeMutableFloatBuffer(base: qBase)
                let kBuf = JANGUnsafeMutableFloatBuffer(base: kBase)
                Self.forEach(numKeyHeads, parallel: usesParallelCPU) { head in
                    let base = head * keyHeadDim
                    Self.rmsNormNoWeight(values: qBuf, base: base, count: keyHeadDim, eps: 1e-6, scale: invScale * invScale)
                    Self.rmsNormNoWeight(values: kBuf, base: base, count: keyHeadDim, eps: 1e-6, scale: invScale)
                }
            }
        }
    }

    private static func rmsNormNoWeight(
        values: JANGUnsafeMutableFloatBuffer,
        base: Int,
        count: Int,
        eps: Float,
        scale: Float
    ) {
        var sumSq: Float = 0
        for i in 0..<count {
            let value = values.load(base + i)
            sumSq += value * value
        }
        let rrms = 1.0 / Float((sumSq / Float(count) + eps).squareRoot())
        for i in 0..<count {
            values.store(base + i, values.load(base + i) * rrms * scale)
        }
    }

    private func runGatedDelta(
        q: [Float],
        k: [Float],
        v: [Float],
        a: [Float],
        b: [Float],
        out: inout [Float]
    ) {
        let aLogPtr = aLog.contents().bindMemory(to: Float16.self, capacity: numValueHeads)
        let dtPtr = dtBias.contents().bindMemory(to: Float16.self, capacity: numValueHeads)
        let repeatFactor = numValueHeads / numKeyHeads
        let keyHeadDim = self.keyHeadDim
        let valueHeadDim = self.valueHeadDim
        let numValueHeads = self.numValueHeads
        let aLogBuf = JANGUnsafeHalfBuffer(base: aLogPtr)
        let dtBuf = JANGUnsafeHalfBuffer(base: dtPtr)

        q.withUnsafeBufferPointer { qPtr in
            k.withUnsafeBufferPointer { kPtr in
                v.withUnsafeBufferPointer { vPtr in
                    a.withUnsafeBufferPointer { aPtr in
                        b.withUnsafeBufferPointer { bPtr in
                            deltaState.withUnsafeMutableBufferPointer { statePtr in
                                out.withUnsafeMutableBufferPointer { outPtr in
                                    guard let qBasePtr = qPtr.baseAddress,
                                          let kBasePtr = kPtr.baseAddress,
                                          let vBasePtr = vPtr.baseAddress,
                                          let aBasePtr = aPtr.baseAddress,
                                          let bBasePtr = bPtr.baseAddress,
                                          let stateBasePtr = statePtr.baseAddress,
                                          let outBasePtr = outPtr.baseAddress
                                    else { return }
                                    let qBuf = JANGUnsafeFloatBuffer(base: qBasePtr)
                                    let kBuf = JANGUnsafeFloatBuffer(base: kBasePtr)
                                    let vBuf = JANGUnsafeFloatBuffer(base: vBasePtr)
                                    let aBuf = JANGUnsafeFloatBuffer(base: aBasePtr)
                                    let bBuf = JANGUnsafeFloatBuffer(base: bBasePtr)
                                    let stateBuf = JANGUnsafeMutableFloatBuffer(base: stateBasePtr)
                                    let outBuf = JANGUnsafeMutableFloatBuffer(base: outBasePtr)
                                    Self.forEach(numValueHeads, parallel: usesParallelCPU) { hv in
                                        let hk = hv / repeatFactor
                                        let decay = Foundation.exp(
                                            -Foundation.exp(Float(aLogBuf.load(hv))) *
                                            Self.softplus(aBuf.load(hv) + Float(dtBuf.load(hv)))
                                        )
                                        let beta = Self.sigmoid(bBuf.load(hv))
                                        let qBase = hk * keyHeadDim
                                        let kBase = hk * keyHeadDim
                                        let vBase = hv * valueHeadDim
                                        let stateBase = hv * valueHeadDim * keyHeadDim

                                        for dv in 0..<valueHeadDim {
                                            let stateRow = stateBase + dv * keyHeadDim
                                            var kvMem: Float = 0
                                            for dk in 0..<keyHeadDim {
                                                let decayed = stateBuf.load(stateRow + dk) * decay
                                                stateBuf.store(stateRow + dk, decayed)
                                                kvMem += decayed * kBuf.load(kBase + dk)
                                            }

                                            let delta = (vBuf.load(vBase + dv) - kvMem) * beta
                                            var y: Float = 0
                                            for dk in 0..<keyHeadDim {
                                                let updated = stateBuf.load(stateRow + dk) + kBuf.load(kBase + dk) * delta
                                                stateBuf.store(stateRow + dk, updated)
                                                y += updated * qBuf.load(qBase + dk)
                                            }
                                            outBuf.store(vBase + dv, y)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func applyGatedNorm(hiddenStates: inout [Float], gate: [Float]) {
        let gamma = normWeight.contents().bindMemory(to: Float16.self, capacity: valueHeadDim)
        let valueHeadDim = self.valueHeadDim
        let normEps = self.normEps
        let gammaBuf = JANGUnsafeHalfBuffer(base: gamma)
        gate.withUnsafeBufferPointer { gatePtr in
            hiddenStates.withUnsafeMutableBufferPointer { hiddenPtr in
                guard let gateBase = gatePtr.baseAddress,
                      let hiddenBase = hiddenPtr.baseAddress
                else { return }
                let gateBuf = JANGUnsafeFloatBuffer(base: gateBase)
                let hiddenBuf = JANGUnsafeMutableFloatBuffer(base: hiddenBase)
                Self.forEach(numValueHeads, parallel: usesParallelCPU) { hv in
                    let base = hv * valueHeadDim
                    var sumSq: Float = 0
                    for d in 0..<valueHeadDim {
                        let value = hiddenBuf.load(base + d)
                        sumSq += value * value
                    }
                    let rrms = 1.0 / Float((sumSq / Float(valueHeadDim) + normEps).squareRoot())
                    for d in 0..<valueHeadDim {
                        let normalized = hiddenBuf.load(base + d) * rrms * Float(gammaBuf.load(d))
                        hiddenBuf.store(base + d, Self.silu(gateBuf.load(base + d)) * normalized)
                    }
                }
            }
        }
    }

    private func makeHalfBuffer(_ values: [Float]) throws -> MTLBuffer {
        let count = values.count
        guard let buffer = affine8.context.device.makeBuffer(
            length: count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw JANGError.bufferAllocationFailed(count * MemoryLayout<Float16>.stride)
        }
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float16(values[i])
        }
        return buffer
    }

    private func makeHalfBuffer(_ f32Buffer: MTLBuffer, count: Int) throws -> MTLBuffer {
        guard let buffer = affine8.context.device.makeBuffer(
            length: count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw JANGError.bufferAllocationFailed(count * MemoryLayout<Float16>.stride)
        }
        let src = f32Buffer.contents().bindMemory(to: Float.self, capacity: count)
        let dst = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            dst[i] = Float16(src[i])
        }
        return buffer
    }

    private static func forEach(
        _ count: Int,
        parallel: Bool,
        _ body: @escaping @Sendable (Int) -> Void
    ) {
        guard parallel, count >= 16 else {
            for i in 0..<count {
                body(i)
            }
            return
        }
        DispatchQueue.concurrentPerform(iterations: count, execute: body)
    }

    private static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + Foundation.exp(-x))
    }

    private static func silu(_ x: Float) -> Float {
        x / (1.0 + Foundation.exp(-x))
    }

    private static func softplus(_ x: Float) -> Float {
        if x > 20 { return x }
        if x < -20 { return Foundation.exp(x) }
        return Foundation.log1p(Foundation.exp(x))
    }
}
