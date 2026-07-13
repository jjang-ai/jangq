/*
 * JANG CLI — Mixed-Precision Inference Engine for Apple Silicon
 * Created by Eric Jang (eric@vmlx.net)
 */

import ArgumentParser
import Foundation
import Metal
import JANG
import JANGCoreMetal

@main
struct JANGCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "jang",
        abstract: "JANG — Mixed-Precision Quantization for MLX on Apple Silicon",
        discussion: """
        Created by Eric Jang (eric@vmlx.net)

        JANG loads and runs mixed-precision quantized models on Apple Silicon
        GPUs with custom Metal kernels for maximum performance.
        """,
        version: "0.1.0",
        subcommands: [Info.self, Run.self, Debug.self, JangTQ.self]
    )
}

// MARK: - Info Command

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show model information"
    )

    @Argument(help: "Path to JANG model directory")
    var modelPath: String

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)
        let config = try JANGModelConfig.load(from: url)

        print("""

          JANG Model Info
          ──────────────────────────────────
          Source: \(config.quant.sourceModelName)
          Format: JANG v\(config.quant.formatVersion)
          Bits: \(config.quant.actualBits) avg (\(config.quant.targetBits) target)
          Block size: \(config.quant.blockSize)
          Architecture: \(config.model.modelType ?? "unknown")
          Layers: \(config.model.numHiddenLayers)
          Hidden: \(config.model.hiddenSize)
          Vocab: \(config.model.vocabSize)
          Heads: \(config.model.numAttentionHeads) Q, \(config.model.kvHeads) KV
          Head dim: \(config.model.headDim)
          RoPE theta: \(config.model.ropeBase)
          Weights: \(config.quant.totalWeightBytes / 1_000_000) MB

          Created by Eric Jang (eric@vmlx.net)
        """)
    }
}

// MARK: - Run Command

struct Run: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run inference on a JANG model"
    )

    @Argument(help: "Path to JANG model directory")
    var modelPath: String

    @Option(name: .shortAndLong, help: "Prompt text")
    var prompt: String = "Hello"

    @Option(name: .long, help: "System prompt")
    var system: String?

    @Option(name: .long, help: "Temperature (0 = greedy)")
    var temperature: Float = 0.7

    @Option(name: .long, help: "Top-k sampling (0 = disabled)")
    var topK: Int = 40

    @Option(name: .long, help: "Top-p nucleus sampling")
    var topP: Float = 0.9

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 256

    @Flag(name: .long, help: "Interactive chat mode")
    var interactive: Bool = false

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)

        print("""

          ╔══════════════════════════════════════════════════════╗
          ║  JANG Runtime v0.1.0                                  ║
          ║  Mixed-Precision Inference for Apple Silicon          ║
          ║  Created by Eric Jang (eric@vmlx.net)                ║
          ╚══════════════════════════════════════════════════════╝
        """)

        // 1. Initialize Metal
        print("  Initializing Metal...")
        let metalDevice = try JANGMetalDevice()
        print("  GPU: \(metalDevice.deviceInfo)")

        // 2. Load model
        print("  Loading model...")
        let model = try loadModel(url: url, device: metalDevice.device)

        // 3. Load tokenizer
        print("  Loading tokenizer...")
        let tokenizerPath = url.appendingPathComponent("tokenizer.json")
        let tokenizer = try JANGTokenizer(tokenizerPath: tokenizerPath)

        // 4. Initialize inference engine
        print("  Initializing inference engine...")
        let engine = try JANGInferenceEngine(
            model: model,
            metalDevice: metalDevice,
            maxSeqLen: 2048
        )

        // 5. Tokenize prompt
        let tokens = tokenizer.encodeChatPrompt(system: system, user: prompt)
        print("  Prompt tokens: \(tokens.count)")
        print("  Token IDs: \(tokens)")
        print()

        // 6. Generate
        var params = SamplingParams()
        params.temperature = temperature
        params.topK = topK
        params.topP = topP
        params.maxTokens = maxTokens

        let sampler = JANGSampler()

        print("  Generating...")
        print("  ─────────────────────────────────")

        // Prefill: process prompt tokens (all but last)
        let prefillStart = CFAbsoluteTimeGetCurrent()
        for tokenId in tokens.dropLast() {
            _ = try engine.forward(tokenId: tokenId)
        }

        // Process last prompt token and get logits for first generated token
        // engine.debugLayers = true  // uncomment to dump per-layer norms
        var generatedTokens: [Int] = []
        var lastLogits = try engine.forward(tokenId: tokens.last ?? 0)
        engine.debugLayers = false
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        let prefillTps = Double(tokens.count) / prefillTime
        print("  Prefill: \(tokens.count) tokens in \(String(format: "%.2f", prefillTime))s (\(String(format: "%.1f", prefillTps)) tok/s)")

        // Dump top logits for debugging
        let logitsPtr = lastLogits.contents().bindMemory(
            to: Float16.self, capacity: model.config.model.vocabSize)
        var topVal: Float = -Float.infinity
        var topIdx = 0
        for i in 0..<model.config.model.vocabSize {
            let v = Float(logitsPtr[i])
            if v > topVal { topVal = v; topIdx = i }
        }
        print("  Top logit: token \(topIdx) = \(topVal)")
        let decodedTop = tokenizer.decodeToken(topIdx)
        print("  Top token: '\(decodedTop)'")
        // Also dump first 8 logits for comparison with reference
        let first8 = (0..<8).map { Float(logitsPtr[$0]) }
        print("  Logits[:8]: \(first8.map { String(format: "%.4f", $0) })")
        print()

        let decodeStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<maxTokens {
            let nextToken = sampler.sample(
                logits: lastLogits,
                vocabSize: model.config.model.vocabSize,
                params: params
            )

            // Check for EOS
            if nextToken == tokenizer.eosTokenId { break }
            if let imEnd = tokenizer.imEndId, nextToken == imEnd { break }

            generatedTokens.append(nextToken)

            // Print token (streaming)
            let text = tokenizer.decodeToken(nextToken)
            print(text, terminator: "")
            fflush(stdout)

            // Forward pass for next token
            lastLogits = try engine.forward(tokenId: nextToken)
        }
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart

        print()
        print("  ─────────────────────────────────")
        let decodeTps = generatedTokens.count > 0 ? Double(generatedTokens.count) / decodeTime : 0
        print("  Generated \(generatedTokens.count) tokens")
        print("  Decode: \(String(format: "%.2f", decodeTime))s (\(String(format: "%.1f", decodeTps)) tok/s)")
        print("  Total: \(String(format: "%.2f", prefillTime + decodeTime))s")
        print()
    }
}

// MARK: - Debug Command

struct Debug: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Debug: verify GPU kernel outputs vs expected values"
    )

    @Argument(help: "Path to JANG model directory")
    var modelPath: String

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)

        print("\n  JANG Debug Mode")
        print("  ──────────────────────────────────")

        let metalDevice = try JANGMetalDevice()
        print("  GPU: \(metalDevice.deviceInfo)")

        let model = try loadModel(url: url, device: metalDevice.device)
        let engine = try JANGInferenceEngine(model: model, metalDevice: metalDevice, maxSeqLen: 128)

        print("\n  Testing embedding dequant (token 0)...")
        try engine.debugEmbedding(tokenId: 0)

        print("\n  Expected (from CPU): [-0.0070, 0.0420, 0.0070, 0.0000, -0.0280, 0.0000, 0.0000, -0.0210]")

        print("\n  Testing one forward layer...")
        engine.reset()
        try engine.debugForwardOneLayer(tokenId: 0)

        print("\n  Debug complete.\n")
    }
}


// MARK: - JANGTQ Command (codebook + Hadamard runtime)

struct JangTQ: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tq",
        abstract: "Run a JANGTQ (codebook + Hadamard) MoE model end-to-end."
    )

    @Argument(help: "Path to JANGTQ model directory")
    var modelPath: String

    @Option(name: .shortAndLong, help: "User prompt")
    var prompt: String = "What is the capital of France?"

    @Option(name: .long, help: "System prompt (defaults to MiniMax default)")
    var system: String?

    @Option(name: .long, help: "Max tokens to generate")
    var maxTokens: Int = 200

    @Option(name: .long, help: "Max sequence length for KV cache")
    var maxSeqLen: Int = 2048

    @Option(name: .long, help: "MoE prefix in the safetensors keys")
    var moePrefix: String = "block_sparse_moe"

    @Flag(name: .long, help: "Stream tokens as they decode")
    var stream: Bool = false

    @Flag(name: .long, help: "Print prompt + token counts + tok/s")
    var verbose: Bool = false

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw JANGError.inferenceError("No Metal device available")
        }

        print("Loading JANGTQ model from \(url.path)…", flush: true)
        let t0 = Date()

        let bundle = try JANGTQLoader(device: device).load(from: url)
        let ctx = try MetalContext()
        let model = try JANGTQModel(
            bundle: bundle, context: ctx, maxSeqLen: maxSeqLen, moePrefix: moePrefix
        )
        let tok = try JANGTQTokenizer(modelDir: url)
        let gen = JANGTQGenerator(model: model, tokenizer: tok)

        let loadElapsed = Date().timeIntervalSince(t0)
        print(String(format: "Loaded in %.1fs.", loadElapsed))

        if verbose {
            let cfg = bundle.config
            print("""
              ──────────────────────────────────
              Source: \(cfg.quant.sourceModelName)
              Architecture: \(cfg.model.modelType ?? "?")
              Layers: \(cfg.model.numHiddenLayers)
              Hidden: \(cfg.model.hiddenSize)
              Experts: \(cfg.model.numLocalExperts ?? 0) / top-\(cfg.model.numExpertsPerTok ?? 0)
              Vocab: \(cfg.model.vocabSize)
              Stop tokens: \(tok.stopTokenIds.sorted())
              ──────────────────────────────────
            """)
        }

        print("\n> \(prompt)\n")

        let result = try gen.generate(
            userMessage: prompt, system: system,
            maxTokens: maxTokens, verbose: stream
        )

        if !stream {
            print(result.text)
        }

        if verbose, result.text.isEmpty {
            print("  Raw decode: \(String(reflecting: result.rawText))")
            print("  Token IDs: \(result.tokenIds)")
        }

        print()
        print(String(format: "  %d prompt + %d output tokens in %.2fs (%.1f tok/s, stop: %@)",
                     result.promptTokens, result.outputTokens,
                     result.elapsedSec, result.tokensPerSec, result.stopReason.rawValue))
    }
}

// Foundation's print does not flush by default — wrap it for the loading trace.
fileprivate func print(_ s: String, flush: Bool) {
    Swift.print(s)
    if flush { fflush(stdout) }
}
