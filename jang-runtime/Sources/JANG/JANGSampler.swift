/*
 * MXQ Sampler — Token Sampling from Logits
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Converts raw logits into a next token ID using:
 * - Temperature scaling
 * - Top-k filtering
 * - Top-p (nucleus) filtering
 * - Min-p filtering
 * - Repetition penalty
 */

import Foundation
import Metal

public struct SamplingParams: Sendable {
    public var temperature: Float = 0.7
    public var topK: Int = 40
    public var topP: Float = 0.9
    public var minP: Float = 0.0
    public var repetitionPenalty: Float = 1.1
    public var maxTokens: Int = 256
    public var seed: UInt64? = nil

    public init() {}

    public static var greedy: SamplingParams {
        var p = SamplingParams()
        p.temperature = 0.0
        p.topK = 1
        return p
    }

    public static var creative: SamplingParams {
        var p = SamplingParams()
        p.temperature = 0.9
        p.topK = 50
        p.topP = 0.95
        return p
    }
}

public final class JANGSampler {
    private var rng: RandomNumberGenerator
    private var recentTokens: [Int] = []
    private let maxRecentTokens = 64

    public init(seed: UInt64? = nil) {
        if let seed = seed {
            self.rng = SeedableRNG(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator() as RandomNumberGenerator
        }
    }

    /// Sample a token from logits buffer.
    public func sample(logits: MTLBuffer, vocabSize: Int,
                       params: SamplingParams) -> Int {
        // Read logits from GPU buffer
        let logitsPtr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
        var logitsArray = [Float](repeating: 0, count: vocabSize)
        for i in 0..<vocabSize {
            logitsArray[i] = Float(logitsPtr[i])
        }

        // Apply repetition penalty
        if params.repetitionPenalty != 1.0 {
            for tokenId in recentTokens {
                if tokenId < vocabSize {
                    if logitsArray[tokenId] > 0 {
                        logitsArray[tokenId] /= params.repetitionPenalty
                    } else {
                        logitsArray[tokenId] *= params.repetitionPenalty
                    }
                }
            }
        }

        // Greedy (temperature = 0)
        if params.temperature <= 0.0 || params.topK == 1 {
            let tokenId = argmax(logitsArray)
            recordToken(tokenId)
            return tokenId
        }

        // Temperature scaling
        if params.temperature != 1.0 {
            let invTemp = 1.0 / params.temperature
            for i in 0..<vocabSize {
                logitsArray[i] *= invTemp
            }
        }

        // Convert to probabilities (softmax)
        let maxLogit = logitsArray.max() ?? 0
        var probs = logitsArray.map { exp($0 - maxLogit) }
        let sum = probs.reduce(0, +)
        if sum > 0 {
            for i in 0..<vocabSize {
                probs[i] /= sum
            }
        }

        // Build sorted indices
        var indices = Array(0..<vocabSize)
        indices.sort { probs[$0] > probs[$1] }

        // Top-k filtering
        var cutoff = vocabSize
        if params.topK > 0 && params.topK < vocabSize {
            cutoff = params.topK
        }

        // Min-p filtering
        if params.minP > 0 {
            let maxProb = probs[indices[0]]
            let threshold = maxProb * params.minP
            for i in 0..<cutoff {
                if probs[indices[i]] < threshold {
                    cutoff = i
                    break
                }
            }
        }

        // Top-p (nucleus) filtering
        if params.topP < 1.0 {
            var cumProb: Float = 0
            for i in 0..<cutoff {
                cumProb += probs[indices[i]]
                if cumProb >= params.topP {
                    cutoff = i + 1
                    break
                }
            }
        }

        cutoff = max(cutoff, 1)

        // Renormalize
        var filteredSum: Float = 0
        for i in 0..<cutoff {
            filteredSum += probs[indices[i]]
        }

        // Random sample
        var r = Float.random(in: 0..<1, using: &rng)
        r *= filteredSum

        var cumulative: Float = 0
        for i in 0..<cutoff {
            cumulative += probs[indices[i]]
            if cumulative >= r {
                let tokenId = indices[i]
                recordToken(tokenId)
                return tokenId
            }
        }

        // Fallback
        let tokenId = indices[0]
        recordToken(tokenId)
        return tokenId
    }

    public func reset() {
        recentTokens.removeAll()
    }

    private func recordToken(_ id: Int) {
        recentTokens.append(id)
        if recentTokens.count > maxRecentTokens {
            recentTokens.removeFirst()
        }
    }

    private func argmax(_ arr: [Float]) -> Int {
        var best = 0
        var bestVal = arr[0]
        for i in 1..<arr.count {
            if arr[i] > bestVal {
                bestVal = arr[i]
                best = i
            }
        }
        return best
    }
}

// Simple seedable RNG for reproducibility
private struct SeedableRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
