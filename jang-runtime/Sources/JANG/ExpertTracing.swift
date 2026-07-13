import Foundation

/// Non-destructive expert controls for one traced run.
///
/// `layers` maps a layer index to the experts disabled in that layer.
/// `topKOverride` lowers the router's active expert count for the run; it
/// cannot raise K above the model's trained value.
public struct JANGExpertMask: Codable, Equatable, Sendable {
    public var layers: [Int: Set<Int>]
    public var lockedKeepByLayer: [Int: Set<Int>]
    public var topKOverride: Int?

    public init(
        layers: [Int: Set<Int>] = [:],
        lockedKeepByLayer: [Int: Set<Int>] = [:],
        topKOverride: Int? = nil
    ) {
        self.layers = layers
        self.lockedKeepByLayer = lockedKeepByLayer
        self.topKOverride = topKOverride
    }

    public var disabledExpertsByLayer: [Int: Set<Int>] {
        get { layers }
        set { layers = newValue }
    }

    public func disabledExperts(for layer: Int) -> Set<Int> {
        layers[layer] ?? []
    }

    public func lockedKeepExperts(for layer: Int) -> Set<Int> {
        lockedKeepByLayer[layer] ?? []
    }
}

/// Runtime trace controls for JANGTQ MoE inference.
public struct JANGExpertTraceConfig: Codable, Equatable, Sendable {
    public var mask: JANGExpertMask?
    public var emitTokenTrace: Bool
    public var maxTraceTokens: Int

    public init(
        mask: JANGExpertMask? = nil,
        emitTokenTrace: Bool = true,
        maxTraceTokens: Int = 512
    ) {
        self.mask = mask
        self.emitTokenTrace = emitTokenTrace
        self.maxTraceTokens = max(0, maxTraceTokens)
    }
}

/// One router decision for one token at one MoE layer.
public struct JANGExpertRouteRecord: Codable, Equatable, Sendable {
    public let tokenIndex: Int
    public let layer: Int
    public let selectedExperts: [Int]
    public let scores: [Float]
    public let disabledExperts: [Int]
    public let effectiveTopK: Int
    public let entropy: Float?

    public init(
        tokenIndex: Int,
        layer: Int,
        selectedExperts: [Int],
        scores: [Float],
        disabledExperts: [Int] = [],
        effectiveTopK: Int,
        entropy: Float? = nil
    ) {
        self.tokenIndex = tokenIndex
        self.layer = layer
        self.selectedExperts = selectedExperts
        self.scores = scores
        self.disabledExperts = disabledExperts
        self.effectiveTopK = effectiveTopK
        self.entropy = entropy
    }
}

/// Thread-safe collection sink used by the decode loop while tracing.
public final class JANGExpertTraceCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var records: [JANGExpertRouteRecord] = []

    public init() {}

    public func record(_ record: JANGExpertRouteRecord, config: JANGExpertTraceConfig?) {
        // Always collect bounded records internally so aggregate layer stats can
        // still be computed when callers suppress the returned per-token trace.
        let maxRecords = config?.maxTraceTokens ?? Int.max
        lock.lock()
        defer { lock.unlock() }
        guard records.count < maxRecords else { return }
        records.append(record)
    }

    public func snapshot() -> [JANGExpertRouteRecord] {
        lock.lock()
        defer { lock.unlock() }
        return records
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        records.removeAll()
    }
}

/// Result shape returned by native traced JANGTQ generation.
public struct JANGTQExpertRunResult: Sendable {
    public let generation: JANGTQGenerationResult
    public let trace: [JANGExpertRouteRecord]

    public init(generation: JANGTQGenerationResult, trace: [JANGExpertRouteRecord]) {
        self.generation = generation
        self.trace = trace
    }
}
