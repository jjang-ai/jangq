/*
 * MLXQ CLI — Mixed-Precision Inference Engine for Apple Silicon
 * Created by Eric Jang (eric@vmlx.net)
 */

import ArgumentParser
import Foundation
import MXQ

@main
struct MXQCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlxq",
        abstract: "MLXQ — Mixed-Precision Quantization for MLX on Apple Silicon",
        discussion: """
        Created by Eric Jang (eric@vmlx.net)

        MXQ loads and runs mixed-precision quantized models on Apple Silicon
        GPUs with custom Metal kernels for maximum performance.
        """,
        version: "0.1.0",
        subcommands: [Info.self, Run.self, Debug.self]
    )
}

// MARK: - Info Command

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show model information"
    )

    @Argument(help: "Path to MXQ model directory")
    var modelPath: String

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)
        let config = try MXQModelConfig.load(from: url)

        print("""

          MLXQ Model Info
          ──────────────────────────────────
          Source: \(config.quant.sourceModelName)
          Format: MXQ v\(config.quant.formatVersion)
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
        abstract: "Run inference on an MXQ model"
    )

    @Argument(help: "Path to MXQ model directory")
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
          ║  MLXQ Runtime v0.1.0                                  ║
          ║  Mixed-Precision Inference for Apple Silicon          ║
          ║  Created by Eric Jang (eric@vmlx.net)                ║
          ╚══════════════════════════════════════════════════════╝
        """)

        // 1. Initialize Metal
        print("  Initializing Metal...")
        let metalDevice = try MXQMetalDevice()
        print("  GPU: \(metalDevice.deviceInfo)")

        // 2. Load model
        print("  Loading model...")
        let model = try loadModel(url: url, device: metalDevice.device)

        // 3. Load tokenizer
        print("  Loading tokenizer...")
        let tokenizerPath = url.appendingPathComponent("tokenizer.json")
        let tokenizer = try MXQTokenizer(tokenizerPath: tokenizerPath)

        // 4. Initialize inference engine
        print("  Initializing inference engine...")
        let engine = try MXQInferenceEngine(
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

        let sampler = MXQSampler()

        print("  Generating...")
        print("  ─────────────────────────────────")

        // Prefill: process prompt tokens (all but last)
        for tokenId in tokens.dropLast() {
            _ = try engine.forward(tokenId: tokenId)
        }

        // Process last prompt token and get logits for first generated token
        engine.debugLayers = true  // dump per-layer norms for last prefill token
        var generatedTokens: [Int] = []
        var lastLogits = try engine.forward(tokenId: tokens.last ?? 0)
        engine.debugLayers = false

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

        print()
        print("  ─────────────────────────────────")
        print("  Generated \(generatedTokens.count) tokens")
        print()
    }
}

// MARK: - Debug Command

struct Debug: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Debug: verify GPU kernel outputs vs expected values"
    )

    @Argument(help: "Path to MXQ model directory")
    var modelPath: String

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)

        print("\n  MLXQ Debug Mode")
        print("  ──────────────────────────────────")

        let metalDevice = try MXQMetalDevice()
        print("  GPU: \(metalDevice.deviceInfo)")

        let model = try loadModel(url: url, device: metalDevice.device)
        let engine = try MXQInferenceEngine(model: model, metalDevice: metalDevice, maxSeqLen: 128)

        print("\n  Testing embedding dequant (token 0)...")
        try engine.debugEmbedding(tokenId: 0)

        print("\n  Expected (from CPU): [-0.0070, 0.0420, 0.0070, 0.0000, -0.0280, 0.0000, 0.0000, -0.0210]")

        print("\n  Testing one forward layer...")
        engine.reset()
        try engine.debugForwardOneLayer(tokenId: 0)

        print("\n  Debug complete.\n")
    }
}
