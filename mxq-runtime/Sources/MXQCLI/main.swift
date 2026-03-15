/*
 * MXQ CLI — Mixed-Precision Inference Engine for Apple Silicon
 * Created by Eric Jang (eric@vmlx.net)
 */

import ArgumentParser
import Foundation
import MXQ

@main
struct MXQCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mxq",
        abstract: "MXQ — Mixed-Precision Inference Engine for Apple Silicon",
        discussion: """
        Created by Eric Jang (eric@vmlx.net)

        MXQ loads and runs mixed-precision quantized models on Apple Silicon
        GPUs with custom Metal kernels for maximum performance.
        """,
        version: "0.1.0",
        subcommands: [Info.self, Run.self]
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

          MXQ Model Info
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
          ║  MXQ Runtime v0.1.0                                  ║
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

        // Prefill: process prompt tokens
        for tokenId in tokens {
            _ = try engine.forward(tokenId: tokenId)
        }

        // Decode: generate tokens one at a time
        var generatedTokens: [Int] = []
        var lastLogits = try engine.forward(tokenId: tokens.last ?? 0)

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
