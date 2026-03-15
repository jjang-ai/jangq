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
        subcommands: [Info.self]
    )
}

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show model information"
    )

    @Argument(help: "Path to MXQ model directory")
    var modelPath: String

    func run() throws {
        let url = URL(fileURLWithPath: modelPath)

        // Load config only (don't load weights)
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
          Weights: \(config.quant.totalWeightBytes / 1_000_000) MB

          Created by Eric Jang (eric@vmlx.net)
        """)
    }
}
