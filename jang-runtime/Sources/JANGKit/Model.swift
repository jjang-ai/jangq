// JANGKit.Model — unified high-level API for loading + generating with JANG models.
//
// Supports both standard JANG (MLX affine quantized, weight_format != "mxtq") and
// JANGTQ (custom TQ-quantized, weight_format == "mxtq") through the same API surface.
// The backend is detected from jang_config.json at load time.
//
// API gaps (Phase P2 work):
//   - JANG (standard) generate(): uses a one-token-at-a-time prefill loop identical
//     to the previous implementation.
//   - JANGTQ generate(): pure greedy argmax only; temperature and topP from
//     SamplingConfig are accepted but currently ignored (JANGTQSampler is argmax-only).
//   - chat() on the JANG backend uses JANGTokenizer.encodeChatPrompt() which implements
//     the Qwen im_start/im_end template only. For other families, format prompts manually.
//   - chat() on the JANGTQ backend uses JANGTQTokenizer.applyChatTemplate() which
//     implements the MiniMax M2.7 template. Other JANGTQ families may need adjustment.

import Foundation
import Metal
import JANG
import JANGCoreMetal

extension JANGKit {

    // MARK: - GenerationResult

    /// Result of a generate() call — the generated text plus timing info.
    public struct GenerationResult: Sendable, Equatable {
        public let text: String
        public let tokens: Int
        public let elapsedSeconds: Double
        public let tokensPerSecond: Double
        public let finishReason: FinishReason

        public enum FinishReason: String, Sendable, Equatable {
            case stop       // EOS / stop token produced
            case maxTokens  // hit the maxTokens cap
            case cancelled  // reserved for future streaming cancellation
            case error      // mid-generation exception (text contains partial output)
        }
    }

    // MARK: - SamplingConfig

    /// Sampling parameters for generate().
    ///
    /// Note: for JANGTQ models, `temperature` and `topP` are accepted but currently
    /// unused — `JANGTQSampler` performs argmax (greedy) decoding only.
    public struct SamplingConfig: Sendable {
        public var temperature: Double
        public var topP: Double
        public var topK: Int
        public var maxTokens: Int

        public init(temperature: Double = 0.0,
                    topP: Double = 1.0,
                    topK: Int = 0,
                    maxTokens: Int = 200) {
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.maxTokens = maxTokens
        }

        /// Convenience: converts to the lower-level SamplingParams.
        fileprivate var asSamplingParams: SamplingParams {
            var p = SamplingParams()
            p.temperature = Float(temperature)
            p.topP = Float(topP)
            p.topK = topK
            p.repetitionPenalty = 1.0
            return p
        }
    }

    public typealias ExpertMask = JANGExpertMask
    public typealias ExpertTraceConfig = JANGExpertTraceConfig
    public typealias ExpertRouteRecord = JANGExpertRouteRecord

    public struct ModelRuntimeInfo: Codable, Equatable, Sendable {
        public let backend: String
        public let runtimeMode: String
        public let deviceName: String
        public let deviceRecommendedMaxWorkingSetGB: Double?
        public let metalEnabled: Bool
        public let jangToolsVersion: String?
        public let mlxVersion: String?
        public let mlxLMVersion: String?
        public let mlxVLMVersion: String?
        public let sourceModelPath: String?
        public let hookedMOELayers: Int?
        public let expectedMOELayers: Int?
        public let hookCoverageComplete: Bool?
        public let maskApplied: Bool?
        public let disabledExpertCount: Int?
        public let topKOverride: Int?
        public let notes: [String]

        public init(
            backend: String,
            runtimeMode: String,
            deviceName: String,
            deviceRecommendedMaxWorkingSetGB: Double? = nil,
            metalEnabled: Bool,
            jangToolsVersion: String? = nil,
            mlxVersion: String? = nil,
            mlxLMVersion: String? = nil,
            mlxVLMVersion: String? = nil,
            sourceModelPath: String? = nil,
            hookedMOELayers: Int? = nil,
            expectedMOELayers: Int? = nil,
            hookCoverageComplete: Bool? = nil,
            maskApplied: Bool? = nil,
            disabledExpertCount: Int? = nil,
            topKOverride: Int? = nil,
            notes: [String] = []
        ) {
            self.backend = backend
            self.runtimeMode = runtimeMode
            self.deviceName = deviceName
            self.deviceRecommendedMaxWorkingSetGB = deviceRecommendedMaxWorkingSetGB
            self.metalEnabled = metalEnabled
            self.jangToolsVersion = jangToolsVersion
            self.mlxVersion = mlxVersion
            self.mlxLMVersion = mlxLMVersion
            self.mlxVLMVersion = mlxVLMVersion
            self.sourceModelPath = sourceModelPath
            self.hookedMOELayers = hookedMOELayers
            self.expectedMOELayers = expectedMOELayers
            self.hookCoverageComplete = hookCoverageComplete
            self.maskApplied = maskApplied
            self.disabledExpertCount = disabledExpertCount
            self.topKOverride = topKOverride
            self.notes = notes
        }
    }

    public struct ExpertLayerStats: Codable, Equatable, Sendable {
        public let layer: Int
        public let tokenCount: Int
        public let hitCounts: [Int: Int]
        public let probabilityMass: [Int: Float]

        public init(
            layer: Int,
            tokenCount: Int,
            hitCounts: [Int: Int],
            probabilityMass: [Int: Float]
        ) {
            self.layer = layer
            self.tokenCount = tokenCount
            self.hitCounts = hitCounts
            self.probabilityMass = probabilityMass
        }
    }

    public struct ExpertRunResult: Sendable {
        public let text: String
        public let tokens: Int
        public let elapsedSeconds: Double
        public let tokensPerSecond: Double
        public let finishReason: GenerationResult.FinishReason
        public let layerStats: [ExpertLayerStats]
        public let tokenTrace: [ExpertRouteRecord]?
        public let runtimeInfo: ModelRuntimeInfo?

        public init(
            text: String,
            tokens: Int,
            elapsedSeconds: Double,
            tokensPerSecond: Double,
            finishReason: GenerationResult.FinishReason,
            layerStats: [ExpertLayerStats],
            tokenTrace: [ExpertRouteRecord]?,
            runtimeInfo: ModelRuntimeInfo? = nil
        ) {
            self.text = text
            self.tokens = tokens
            self.elapsedSeconds = elapsedSeconds
            self.tokensPerSecond = tokensPerSecond
            self.finishReason = finishReason
            self.layerStats = layerStats
            self.tokenTrace = tokenTrace
            self.runtimeInfo = runtimeInfo
        }
    }

    // MARK: - ModelError

    /// Errors surfaced by the high-level JANGKit API.
    public enum ModelError: Error, LocalizedError {
        case metalDeviceUnavailable
        case modelLoadFailed(String)
        case tokenizerLoadFailed(String)
        case generationFailed(String)

        public var errorDescription: String? {
            switch self {
            case .metalDeviceUnavailable:
                return "Metal device not available — JANG requires Apple Silicon."
            case .modelLoadFailed(let m):
                return "Model load failed: \(m)"
            case .tokenizerLoadFailed(let m):
                return "Tokenizer load failed: \(m)"
            case .generationFailed(let m):
                return "Generation failed: \(m)"
            }
        }
    }

    // MARK: - Model

    /// A loaded JANG model ready for single-shot generation.
    ///
    /// Create via `Model.load(at:)`. Thread safety: this type is an actor; all
    /// generation state (KV cache) is serialized.
    ///
    /// Both standard JANG and JANGTQ models are supported. The correct backend is
    /// auto-detected from `jang_config.json` at load time — callers use the same API
    /// regardless of which family the model belongs to.
    public actor Model {

        // MARK: Discriminated backend

        private enum Backend {
            /// Standard JANG model (MLX affine quantized weights).
            case jang(mxqModel: MXQModel,
                      engine: JANGInferenceEngine,
                      tokenizer: JANGTokenizer)
            /// JANGTQ model (custom TQ-quantized MoE weights).
            case jangtq(generator: JANGTQGenerator)
        }

        private let backend: Backend

        /// The directory from which this model was loaded.
        public nonisolated let modelURL: URL

        /// Model family: "jang" for standard models, "jangtq" for JANGTQ models.
        public nonisolated let family: String

        /// Runtime/device identity for artifacts and UI diagnostics.
        public nonisolated let runtimeInfo: ModelRuntimeInfo

        private init(backend: Backend, modelURL: URL, family: String, runtimeInfo: ModelRuntimeInfo) {
            self.backend = backend
            self.modelURL = modelURL
            self.family = family
            self.runtimeInfo = runtimeInfo
        }

        // MARK: - load

        /// Load a JANG model from a directory.
        ///
        /// Auto-detects format via `jang_config.json`. Both standard JANG and JANGTQ
        /// (`weight_format == "mxtq"`) models are fully supported.
        ///
        /// - Parameter url: Path to the model directory. Must contain `jang_config.json`,
        ///   `config.json`, `tokenizer.json`, and one or more `.safetensors` shards.
        public static func load(at url: URL) async throws -> Model {
            let configURL = url.appendingPathComponent("jang_config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw ModelError.modelLoadFailed(
                    "jang_config.json not found at \(configURL.path)"
                )
            }

            let isJANGTQ = detectJANGTQ(configURL: configURL)

            if isJANGTQ {
                return try await loadJANGTQ(from: url)
            } else {
                return try await loadJANG(from: url)
            }
        }

        // MARK: Detection

        private static func detectJANGTQ(configURL: URL) -> Bool {
            guard let data = try? Data(contentsOf: configURL),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return false
            }
            // Top-level weight_format == "mxtq"
            if (obj["weight_format"] as? String)?.lowercased() == "mxtq" { return true }
            // Nested quantization.method == "jangtq"
            if let q = obj["quantization"] as? [String: Any],
               (q["method"] as? String)?.lowercased() == "jangtq" {
                return true
            }
            // Nested quantization.family == "jangtq"
            if let q = obj["quantization"] as? [String: Any],
               (q["family"] as? String)?.lowercased() == "jangtq" {
                return true
            }
            return false
        }

        // MARK: - Standard JANG loading

        private static func loadJANG(from url: URL) async throws -> Model {
            let metalDevice: JANGMetalDevice
            do {
                metalDevice = try JANGMetalDevice()
            } catch {
                throw ModelError.metalDeviceUnavailable
            }

            let mxqModel: MXQModel
            do {
                mxqModel = try loadModel(url: url, device: metalDevice.device)
            } catch {
                throw ModelError.modelLoadFailed("\(error)")
            }

            let engine: JANGInferenceEngine
            do {
                engine = try JANGInferenceEngine(model: mxqModel, metalDevice: metalDevice)
            } catch {
                throw ModelError.modelLoadFailed("inference engine init: \(error)")
            }

            let tokenizerFileURL = url.appendingPathComponent("tokenizer.json")
            let tokenizer: JANGTokenizer
            do {
                tokenizer = try JANGTokenizer(tokenizerPath: tokenizerFileURL)
            } catch {
                throw ModelError.tokenizerLoadFailed("\(error)")
            }

            return Model(
                backend: .jang(mxqModel: mxqModel, engine: engine, tokenizer: tokenizer),
                modelURL: url,
                family: "jang",
                runtimeInfo: ModelRuntimeInfo(
                    backend: "jang",
                    runtimeMode: "native_jang_metal",
                    deviceName: metalDevice.device.name,
                    deviceRecommendedMaxWorkingSetGB: Double(metalDevice.device.recommendedMaxWorkingSetSize) / (1024 * 1024 * 1024),
                    metalEnabled: true,
                    notes: ["Standard JANG tensors execute through native Metal kernels."]
                )
            )
        }

        // MARK: - JANGTQ loading

        private static func loadJANGTQ(from url: URL) async throws -> Model {
            // MetalContext owns device + queue + compiled Metal library for JANGTQ kernels.
            let context: MetalContext
            do {
                context = try MetalContext()
            } catch {
                throw ModelError.metalDeviceUnavailable
            }

            // JANGTQLoader reads .safetensors shards and runtime sidecar.
            let loader = JANGTQLoader(device: context.device)
            let bundle: JANGTQModelBundle
            do {
                bundle = try loader.load(from: url)
            } catch {
                throw ModelError.modelLoadFailed("JANGTQ bundle load: \(error)")
            }

            // JANGTQModel wires embedding + decoder layers + KV cache.
            let model: JANGTQModel
            do {
                model = try JANGTQModel(bundle: bundle, context: context)
            } catch {
                throw ModelError.modelLoadFailed("JANGTQ model init: \(error)")
            }

            // JANGTQTokenizer reads tokenizer.json + generation_config.json for stop IDs.
            let tokenizer: JANGTQTokenizer
            do {
                tokenizer = try JANGTQTokenizer(modelDir: url)
            } catch {
                throw ModelError.tokenizerLoadFailed("JANGTQ tokenizer: \(error)")
            }

            // JANGTQSampler is argmax-only (greedy). Wrap everything in a generator.
            let sampler = JANGTQSampler()
            let generator = JANGTQGenerator(model: model, tokenizer: tokenizer, sampler: sampler)

            return Model(
                backend: .jangtq(generator: generator),
                modelURL: url,
                family: "jangtq",
                runtimeInfo: ModelRuntimeInfo(
                    backend: "jangtq",
                    runtimeMode: "native_jangtq_review_bundle",
                    deviceName: context.device.name,
                    deviceRecommendedMaxWorkingSetGB: Double(context.device.recommendedMaxWorkingSetSize) / (1024 * 1024 * 1024),
                    metalEnabled: true,
                    notes: ["JANGTQ matmul/decode kernels run on Metal; router selection and tracing use CPU bookkeeping."]
                )
            )
        }

        // MARK: - generate

        /// Generate text from a raw prompt string.
        ///
        /// For JANGTQ models, the prompt is treated as a user message and wrapped via
        /// the model's chat template. If you need raw-token generation with JANGTQ,
        /// use `JANGTQGenerator` directly from the `JANG` module.
        ///
        /// - Parameters:
        ///   - prompt: The prompt string.
        ///   - config: Sampling parameters. Note: `temperature` and `topP` are used only
        ///     for standard JANG models; JANGTQ uses greedy (argmax) decoding.
        public func generate(
            prompt: String,
            config: SamplingConfig = SamplingConfig()
        ) async throws -> GenerationResult {
            switch backend {
            case .jang(let mxq, let engine, let tokenizer):
                return try generateJANG(
                    mxqModel: mxq, engine: engine, tokenizer: tokenizer,
                    prompt: prompt, config: config
                )
            case .jangtq(let generator):
                return try generateJANGTQ(generator: generator, userMessage: prompt, config: config)
            }
        }

        /// Generate text while collecting native MoE router traces.
        ///
        /// V1 supports traced execution for JANGTQ MoE models loaded by the
        /// Swift/Metal runtime. Standard JANG models keep using `generate`.
        public func generateWithTrace(
            prompt: String,
            config: SamplingConfig = SamplingConfig(),
            traceConfig: ExpertTraceConfig = ExpertTraceConfig()
        ) async throws -> ExpertRunResult {
            switch backend {
            case .jang:
                throw ModelError.generationFailed(
                    "Expert tracing is available for JANGTQ MoE models only."
                )
            case .jangtq(let generator):
                return try generateJANGTQWithTrace(
                    generator: generator,
                    userMessage: prompt,
                    config: config,
                    traceConfig: traceConfig
                )
            }
        }

        // MARK: - chat

        /// Apply a chat template and generate.
        ///
        /// For standard JANG models, uses `JANGTokenizer.encodeChatPrompt()` (Qwen
        /// im_start/im_end template). For JANGTQ models, uses
        /// `JANGTQTokenizer.applyChatTemplate()` (MiniMax M2.7 template).
        ///
        /// - Parameters:
        ///   - system: Optional system prompt. Defaults to the model's default system message.
        ///   - user: The user message content.
        ///   - config: Sampling parameters.
        public func chat(
            system: String? = nil,
            user: String,
            config: SamplingConfig = SamplingConfig()
        ) async throws -> GenerationResult {
            switch backend {
            case .jang(let mxq, let engine, let tokenizer):
                return try chatJANG(mxqModel: mxq, engine: engine, tokenizer: tokenizer,
                                    system: system, user: user, config: config)
            case .jangtq(let generator):
                return try chatJANGTQ(generator: generator, system: system,
                                      user: user, config: config)
            }
        }

        // MARK: - Standard JANG generate impl

        private func generateJANG(
            mxqModel: MXQModel,
            engine: JANGInferenceEngine,
            tokenizer: JANGTokenizer,
            prompt: String,
            config: SamplingConfig
        ) throws -> GenerationResult {
            let t0 = Date()
            let promptIds = tokenizer.encode(prompt)

            engine.reset()
            for tokenId in promptIds {
                _ = try engine.forward(tokenId: tokenId)
            }

            let sampler = JANGSampler()
            let samplingParams = config.asSamplingParams
            var generatedIds: [Int] = []
            var reason: GenerationResult.FinishReason = .maxTokens
            let eosId = tokenizer.eosTokenId

            for step in 0..<config.maxTokens {
                let inputTokenId: Int
                if step == 0 {
                    inputTokenId = promptIds.last ?? eosId
                } else {
                    inputTokenId = generatedIds.last ?? eosId
                }
                let logits = try engine.forward(tokenId: inputTokenId)
                let nextId = sampler.sample(
                    logits: logits,
                    vocabSize: mxqModel.config.model.vocabSize,
                    params: samplingParams
                )
                if nextId == eosId {
                    reason = .stop
                    break
                }
                generatedIds.append(nextId)
            }

            let text = tokenizer.decode(generatedIds)
            let elapsed = Date().timeIntervalSince(t0)
            let tps = elapsed > 0 ? Double(generatedIds.count) / elapsed : 0
            return GenerationResult(
                text: text,
                tokens: generatedIds.count,
                elapsedSeconds: elapsed,
                tokensPerSecond: tps,
                finishReason: reason
            )
        }

        // MARK: - Standard JANG chat impl

        private func chatJANG(
            mxqModel: MXQModel,
            engine: JANGInferenceEngine,
            tokenizer: JANGTokenizer,
            system: String?,
            user: String,
            config: SamplingConfig
        ) throws -> GenerationResult {
            let t0 = Date()
            let promptIds = tokenizer.encodeChatPrompt(system: system, user: user)

            engine.reset()
            for tokenId in promptIds {
                _ = try engine.forward(tokenId: tokenId)
            }

            let sampler = JANGSampler()
            let samplingParams = config.asSamplingParams
            var generatedIds: [Int] = []
            var reason: GenerationResult.FinishReason = .maxTokens
            let eosId = tokenizer.eosTokenId

            for step in 0..<config.maxTokens {
                let inputTokenId: Int
                if step == 0 {
                    inputTokenId = promptIds.last ?? eosId
                } else {
                    inputTokenId = generatedIds.last ?? eosId
                }
                let logits = try engine.forward(tokenId: inputTokenId)
                let nextId = sampler.sample(
                    logits: logits,
                    vocabSize: mxqModel.config.model.vocabSize,
                    params: samplingParams
                )
                if nextId == eosId {
                    reason = .stop
                    break
                }
                generatedIds.append(nextId)
            }

            let text = tokenizer.decode(generatedIds)
            let elapsed = Date().timeIntervalSince(t0)
            let tps = elapsed > 0 ? Double(generatedIds.count) / elapsed : 0
            return GenerationResult(
                text: text,
                tokens: generatedIds.count,
                elapsedSeconds: elapsed,
                tokensPerSecond: tps,
                finishReason: reason
            )
        }

        // MARK: - JANGTQ generate impl

        private func generateJANGTQ(
            generator: JANGTQGenerator,
            userMessage: String,
            config: SamplingConfig
        ) throws -> GenerationResult {
            // JANGTQGenerator wraps the user message in the model's chat template internally.
            let result = try generator.generate(
                userMessage: userMessage,
                maxTokens: config.maxTokens
            )
            return GenerationResult(
                text: result.text,
                tokens: result.outputTokens,
                elapsedSeconds: result.elapsedSec,
                tokensPerSecond: result.tokensPerSec,
                finishReason: mapJANGTQStopReason(result.stopReason)
            )
        }

        // MARK: - JANGTQ chat impl

        private func chatJANGTQ(
            generator: JANGTQGenerator,
            system: String?,
            user: String,
            config: SamplingConfig
        ) throws -> GenerationResult {
            let result = try generator.generate(
                messages: [JANGTQChatMessage(role: "user", content: user)],
                system: system,
                maxTokens: config.maxTokens
            )
            return GenerationResult(
                text: result.text,
                tokens: result.outputTokens,
                elapsedSeconds: result.elapsedSec,
                tokensPerSecond: result.tokensPerSec,
                finishReason: mapJANGTQStopReason(result.stopReason)
            )
        }

        private func generateJANGTQWithTrace(
            generator: JANGTQGenerator,
            userMessage: String,
            config: SamplingConfig,
            traceConfig: ExpertTraceConfig
        ) throws -> ExpertRunResult {
            let result = try generator.generateWithTrace(
                userMessage: userMessage,
                maxTokens: config.maxTokens,
                traceConfig: traceConfig
            )
            let generation = result.generation
            let trace = result.trace
            return ExpertRunResult(
                text: generation.text,
                tokens: generation.outputTokens,
                elapsedSeconds: generation.elapsedSec,
                tokensPerSecond: generation.tokensPerSec,
                finishReason: mapJANGTQStopReason(generation.stopReason),
                layerStats: Self.buildLayerStats(from: trace),
                tokenTrace: traceConfig.emitTokenTrace ? trace : nil,
                runtimeInfo: runtimeInfo
            )
        }

        // MARK: - Helpers

        private func mapJANGTQStopReason(
            _ reason: JANGTQGenerationResult.StopReason
        ) -> GenerationResult.FinishReason {
            switch reason {
            case .stopToken: return .stop
            case .maxTokens: return .maxTokens
            case .error:     return .error
            }
        }

        private static func buildLayerStats(
            from trace: [ExpertRouteRecord]
        ) -> [ExpertLayerStats] {
            var tokenSets: [Int: Set<Int>] = [:]
            var hits: [Int: [Int: Int]] = [:]
            var mass: [Int: [Int: Float]] = [:]

            for record in trace {
                tokenSets[record.layer, default: []].insert(record.tokenIndex)
                for (slot, expert) in record.selectedExperts.enumerated() {
                    hits[record.layer, default: [:]][expert, default: 0] += 1
                    let score = slot < record.scores.count ? record.scores[slot] : 1.0
                    mass[record.layer, default: [:]][expert, default: 0] += score
                }
            }

            return tokenSets.keys.sorted().map { layer in
                ExpertLayerStats(
                    layer: layer,
                    tokenCount: tokenSets[layer]?.count ?? 0,
                    hitCounts: hits[layer] ?? [:],
                    probabilityMass: mass[layer] ?? [:]
                )
            }
        }
    }
}
