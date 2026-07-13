/*
 * JANGTQ MoE block — Swift wrapper that runs one decoder layer's MoE MLP.
 * Created by Jinho Jang (eric@jangq.ai)
 *
 * One instance per layer, holding references to that layer's three stacked
 * weight tensors (gate_proj, up_proj, down_proj) plus the kernel pipelines
 * and the matching codebook/signs from the runtime sidecar.
 *
 * The forward pass mirrors `_fused_switchglu_call` in the Python loader,
 * fast-path branch:
 *
 *   1. Hadamard rotate x (hidden) using `signs_in` from sidecar
 *   2. fused_gate_up_swiglu — single dispatch, computes SiLU(g) * u
 *      for K experts at once → x_act of shape (K, intermediate)
 *      DSV4 passes swigluLimit=10 for silu(min(g, 10)) * clip(u, +/-10).
 *   3. Hadamard rotate x_act using `signs_dn`
 *   4. gather_tq_matmul — down projection per (k) → y of shape (K, hidden)
 *   5. (Combine across K with router scores happens at the caller)
 *
 * The block does NOT include the router math (gate Linear + sigmoid +
 * argpart + normalize). That's left to the caller because the gate weight
 * is plain f16/bf16, not codebook-quantized, and lives on a different
 * code path.
 */

import Foundation
import Metal
import JANGCoreMetal

public final class JANGTQMoEBlock {
    public let layerIndex: Int
    public let inFeatures: Int      // hidden
    public let outFeatures: Int     // moe_intermediate
    public let nExperts: Int
    public let bits: Int
    public let swigluLimit: Float

    private let kernels: JANGTQKernels

    public let gateProj: JANGTQWeight
    public let upProj: JANGTQWeight
    public let downProj: JANGTQWeight

    /// Hadamard signs for the input (`hidden` dim). Shared with all layers
    /// of the same hidden size.
    public let signsIn: MTLBuffer
    /// Hadamard signs for x_act (`intermediate` dim).
    public let signsDn: MTLBuffer

    /// Lloyd-Max codebook for the gate/up matmul (keyed on `inFeatures`, `bits`).
    public let codebookGate: MTLBuffer
    /// Lloyd-Max codebook for the down matmul (keyed on `outFeatures`, `bits`).
    public let codebookDown: MTLBuffer

    // Pre-allocated scratch buffers (reused across decode steps)
    public let xRotIn: MTLBuffer        // (1, hidden) fp32
    public let xActOut: MTLBuffer       // (K, intermediate) fp32 — fused gate_up output
    public let xActRot: MTLBuffer       // (K, intermediate) fp32 — Hadamard rotated
    public let yOut: MTLBuffer          // (K, hidden) fp32 — gather output

    /// Optional shared-expert path (Qwen3-Next / Qwen3.5 / Qwen3.6 and some
    /// GLM variants). When present, every token additionally flows through
    /// `shared_expert` scaled by `sigmoid(shared_expert_gate(x))`. Stored as
    /// plain affine 8-bit weights in `bundle.affineWeights`.
    public let sharedGateProj: JANGTQAffineWeight?
    public let sharedUpProj:   JANGTQAffineWeight?
    public let sharedDownProj: JANGTQAffineWeight?
    public let sharedGateScalar: JANGTQAffineWeight?  // shared_expert_gate, shape (1, hidden)
    public let sharedGateScalarHalf: MTLBuffer?       // shared_expert_gate.weight, shape (1, hidden)

    public var hasSharedExpert: Bool { sharedGateProj != nil }

    /// Build a block from a loaded `JANGTQModelBundle` and the kernel bundle.
    ///
    /// Two initialization styles:
    ///   - Legacy: `layerIndex + moePrefix` → builds the MiniMax/GLM path prefix
    ///     `model.layers.{L}.<moePrefix>.switch_mlp.{gate,up,down}_proj` (no shared).
    ///   - Explicit: pass `layerPrefix` directly (e.g., for Qwen3.6:
    ///     `language_model.model.layers.\(L).mlp`). Caller is responsible for
    ///     the trailing `.switch_mlp` composition; shared-expert weights are
    ///     looked up at the same `layerPrefix.shared_expert.*` if present.
    public convenience init(
        layerIndex: Int,
        moePrefix: String,                // "block_sparse_moe" or "mlp"
        bundle: JANGTQModelBundle,
        kernels: JANGTQKernels,
        seed: Int? = nil,
        topK: Int = 8,
        swigluLimit: Float = 0
    ) throws {
        try self.init(
            layerIndex: layerIndex,
            layerPrefix: "model.layers.\(layerIndex).\(moePrefix)",
            bundle: bundle, kernels: kernels, seed: seed, topK: topK,
            swigluLimit: swigluLimit
        )
    }

    /// Build a block with an explicit full layer+MoE prefix.
    ///
    /// `layerPrefix` is the path up to (but not including) `switch_mlp` or
    /// `shared_expert`. Examples:
    ///   - MiniMax M2.7: `"model.layers.5.block_sparse_moe"`
    ///   - GLM 5.1:      `"model.layers.30.mlp"`
    ///   - Qwen3.6:      `"language_model.model.layers.3.mlp"`
    public init(
        layerIndex: Int,
        layerPrefix: String,
        bundle: JANGTQModelBundle,
        kernels: JANGTQKernels,
        seed: Int? = nil,
        topK: Int = 8,
        swigluLimit: Float = 0
    ) throws {
        self.layerIndex = layerIndex
        self.kernels = kernels
        self.swigluLimit = swigluLimit

        let basePrefix = "\(layerPrefix).switch_mlp"
        guard
            let g = bundle.weights["\(basePrefix).gate_proj"],
            let u = bundle.weights["\(basePrefix).up_proj"],
            let d = bundle.weights["\(basePrefix).down_proj"]
        else {
            throw JANGError.tensorNotFound(
                "JANGTQMoEBlock: layer \(layerIndex) missing one of " +
                "\(basePrefix).{gate_proj,up_proj,down_proj}"
            )
        }
        self.gateProj = g
        self.upProj = u
        self.downProj = d

        // Sanity-check shapes match: gate.in == up.in == down.out (= hidden),
        // gate.out == up.out == down.in (= intermediate)
        guard g.inFeatures == u.inFeatures, g.inFeatures == d.outFeatures else {
            throw JANGError.invalidFormat(
                "layer \(layerIndex): in/out features mismatch " +
                "gate.in=\(g.inFeatures) up.in=\(u.inFeatures) down.out=\(d.outFeatures)"
            )
        }
        guard g.outFeatures == u.outFeatures, g.outFeatures == d.inFeatures else {
            throw JANGError.invalidFormat(
                "layer \(layerIndex): intermediate features mismatch " +
                "gate.out=\(g.outFeatures) up.out=\(u.outFeatures) down.in=\(d.inFeatures)"
            )
        }
        guard g.nExperts == u.nExperts, g.nExperts == d.nExperts else {
            throw JANGError.invalidFormat(
                "layer \(layerIndex): n_experts mismatch " +
                "gate=\(g.nExperts) up=\(u.nExperts) down=\(d.nExperts)"
            )
        }
        guard g.bits == u.bits, g.bits == d.bits else {
            throw JANGError.invalidFormat(
                "layer \(layerIndex): bits mismatch gate=\(g.bits) up=\(u.bits) down=\(d.bits)"
            )
        }

        self.inFeatures = g.inFeatures
        self.outFeatures = g.outFeatures
        self.nExperts = g.nExperts
        self.bits = g.bits

        let sd = seed ?? bundle.config.quant.mxtqSeed
        self.signsIn = try bundle.sidecar.signs(inFeatures: inFeatures, seed: sd)
        self.signsDn = try bundle.sidecar.signs(inFeatures: outFeatures, seed: sd)
        self.codebookGate = try bundle.sidecar.codebook(inFeatures: inFeatures, bits: bits)
        self.codebookDown = try bundle.sidecar.codebook(inFeatures: outFeatures, bits: bits)

        // Shared expert (optional). All four weights must co-exist to activate.
        let shGatePath = "\(layerPrefix).shared_expert.gate_proj"
        let shUpPath   = "\(layerPrefix).shared_expert.up_proj"
        let shDownPath = "\(layerPrefix).shared_expert.down_proj"
        let shGateScalarPath = "\(layerPrefix).shared_expert_gate"
        self.sharedGateProj   = bundle.affineWeights[shGatePath]
        self.sharedUpProj     = bundle.affineWeights[shUpPath]
        self.sharedDownProj   = bundle.affineWeights[shDownPath]
        self.sharedGateScalar = bundle.affineWeights[shGateScalarPath]
        self.sharedGateScalarHalf = bundle.halfTensors["\(shGateScalarPath).weight"]

        // Pre-allocate scratch buffers
        let dev = kernels.context.device
        let f32 = MemoryLayout<Float>.stride
        func mk(_ bytes: Int) throws -> MTLBuffer {
            guard let b = dev.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw JANGError.bufferAllocationFailed(bytes) }
            return b
        }
        self.xRotIn  = try mk(inFeatures * f32)
        self.xActOut = try mk(topK * outFeatures * f32)
        self.xActRot = try mk(topK * outFeatures * f32)
        self.yOut    = try mk(topK * inFeatures * f32)
    }

    /// Run gate+up+SwiGLU+down for K selected experts on a single token.
    /// Returns a fresh fp32 buffer of shape `(K, inFeatures)`.
    /// (Convenience wrapper — for the batched-cb fast path use `encode(into:...)`.)
    public func runMLP(
        xHalfBuf: MTLBuffer,
        selectedExpertsBuf: MTLBuffer,
        K: Int
    ) throws -> MTLBuffer {
        // 1. Hadamard rotate x (hidden) → fp32
        let xRot = try kernels.hadamard.run(
            xBuf: xHalfBuf, signsBuf: signsIn,
            batch: 1, dim: inFeatures
        )
        // 2. Fused gate+up+SwiGLU → (K, intermediate) fp32
        let xAct = try kernels.fusedGateUp.run(
            xRotBuf: xRot,
            packedGateBuf: gateProj.packed, normsGateBuf: gateProj.norms,
            packedUpBuf: upProj.packed,     normsUpBuf: upProj.norms,
            codebookBuf: codebookGate,
            rhsIndicesBuf: selectedExpertsBuf,
            K: K, inFeatures: inFeatures, outFeatures: outFeatures, bits: bits,
            swigluLimit: swigluLimit
        )
        // 3. Hadamard rotate x_act
        let xActHalf = try copyFloatToHalf(xAct, count: K * outFeatures)
        let xActRot = try kernels.hadamard.run(
            xBuf: xActHalf, signsBuf: signsDn,
            batch: K, dim: outFeatures
        )
        // 4. Gather TQ matmul (down_proj) → (K, hidden) fp32
        let y = try kernels.gather.run(
            xRotBuf: xActRot,
            packedBuf: downProj.packed, normsBuf: downProj.norms,
            codebookBuf: codebookDown,
            rhsIndicesBuf: selectedExpertsBuf,
            K: K, inFeatures: outFeatures, outFeatures: inFeatures, bits: bits
        )
        return y
    }

    /// Encode the full MoE MLP path (rotate → fused → rotate → gather)
    /// into a caller-managed encoder using the pre-allocated scratch buffers.
    /// Result lands in `self.yOut`; caller must not commit until they consume it.
    ///
    /// `xHalfStaging` must be a half MTLBuffer of length `K * outFeatures * 2`
    /// that the caller will use to receive the fp32→fp16 cast of x_act between
    /// the two kernels. Caller pre-allocates this once and reuses.
    public func encode(
        into enc: MTLComputeCommandEncoder,
        xHalfBuf: MTLBuffer,
        selectedExpertsBuf: MTLBuffer,
        K: Int,
        ops: JANGTQDecodeOps,
        xActHalfStaging: MTLBuffer
    ) {
        // 1. Hadamard rotate x → xRotIn (fp32)
        kernels.hadamard.encode(
            into: enc, xBuf: xHalfBuf, signsBuf: signsIn,
            outBuf: xRotIn, batch: 1, dim: inFeatures
        )
        // 2. Fused gate+up+SwiGLU → xActOut (fp32)
        kernels.fusedGateUp.encode(
            into: enc,
            xRotBuf: xRotIn,
            packedGateBuf: gateProj.packed, normsGateBuf: gateProj.norms,
            packedUpBuf: upProj.packed,     normsUpBuf: upProj.norms,
            codebookBuf: codebookGate, rhsIndicesBuf: selectedExpertsBuf,
            outBuf: xActOut,
            K: K, inFeatures: inFeatures, outFeatures: outFeatures, bits: bits,
            swigluLimit: swigluLimit
        )
        // 3a. Cast xActOut (fp32) → xActHalfStaging (fp16) so hadamard can read it
        ops.castF32ToF16.encode(into: enc, src: xActOut, dst: xActHalfStaging,
                                 count: K * outFeatures)
        // 3b. Hadamard rotate x_act → xActRot (fp32)
        kernels.hadamard.encode(
            into: enc, xBuf: xActHalfStaging, signsBuf: signsDn,
            outBuf: xActRot, batch: K, dim: outFeatures
        )
        // 4. Gather TQ matmul (down_proj) → yOut (fp32)
        kernels.gather.encode(
            into: enc,
            xRotBuf: xActRot,
            packedBuf: downProj.packed, normsBuf: downProj.norms,
            codebookBuf: codebookDown, rhsIndicesBuf: selectedExpertsBuf,
            outBuf: yOut,
            K: K, inFeatures: outFeatures, outFeatures: inFeatures, bits: bits
        )
    }

    /// Tiny CPU-side copy that demotes an fp32 buffer to fp16. Used because
    /// our Hadamard kernel takes half input but fused_gate_up_swiglu produces
    /// fp32. For decode T=1 this is a few KB of data and the cost is in the
    /// noise — proper Metal-side cast can replace it later.
    private func copyFloatToHalf(_ floatBuf: MTLBuffer, count: Int) throws -> MTLBuffer {
        let device = kernels.context.device
        let outBytes = count * MemoryLayout<Float16>.stride
        guard let halfBuf = device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGError.bufferAllocationFailed(outBytes)
        }
        let src = floatBuf.contents().bindMemory(to: Float.self, capacity: count)
        let dst = halfBuf.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            dst[i] = Float16(src[i])
        }
        return halfBuf
    }
}
