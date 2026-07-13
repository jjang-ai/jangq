import Foundation
import CoreGraphics
import SQLite3
import JANG
import JANGKit

public enum ExpertPromptExpectedKind: String, Codable, Equatable, Sendable, CaseIterable {
    case freeform
    case exact
    case regex
    case unitTest = "unit_test"
    case judge
}

public struct ExpertPrompt: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let domain: String
    public let subdomain: String?
    public let text: String
    public let expectedKind: ExpertPromptExpectedKind
    public let expected: String?
    public let maxNewTokens: Int?
    public let temperature: Double?
    public let tags: [String]
    public let weight: Double

    public init(
        id: String,
        domain: String,
        text: String,
        subdomain: String? = nil,
        expectedKind: ExpertPromptExpectedKind = .freeform,
        expected: String? = nil,
        maxNewTokens: Int? = nil,
        temperature: Double? = nil,
        tags: [String] = [],
        weight: Double = 1.0
    ) {
        self.id = id
        self.domain = domain
        self.subdomain = subdomain
        self.text = text
        self.expectedKind = expectedKind
        self.expected = expected
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.tags = tags
        self.weight = weight
    }

    enum CodingKeys: String, CodingKey {
        case id
        case domain
        case subdomain
        case text
        case prompt
        case expectedKind = "expected_kind"
        case expected
        case maxNewTokens = "max_new_tokens"
        case temperature
        case tags
        case weight
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        domain = try c.decode(String.self, forKey: .domain)
        subdomain = try c.decodeIfPresent(String.self, forKey: .subdomain)
        text = try c.decodeIfPresent(String.self, forKey: .prompt)
            ?? c.decode(String.self, forKey: .text)
        expectedKind = try c.decodeIfPresent(ExpertPromptExpectedKind.self, forKey: .expectedKind) ?? .freeform
        expected = try c.decodeIfPresent(String.self, forKey: .expected)
        maxNewTokens = try c.decodeIfPresent(Int.self, forKey: .maxNewTokens)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
        weight = try c.decodeIfPresent(Double.self, forKey: .weight) ?? 1.0
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(domain, forKey: .domain)
        try c.encodeIfPresent(subdomain, forKey: .subdomain)
        try c.encode(text, forKey: .prompt)
        try c.encode(expectedKind, forKey: .expectedKind)
        try c.encodeIfPresent(expected, forKey: .expected)
        try c.encodeIfPresent(maxNewTokens, forKey: .maxNewTokens)
        try c.encodeIfPresent(temperature, forKey: .temperature)
        try c.encode(tags, forKey: .tags)
        try c.encode(weight, forKey: .weight)
    }
}

public struct ExpertPromptSuite: Codable, Equatable, Sendable {
    public let name: String
    public let prompts: [ExpertPrompt]

    public init(name: String, prompts: [ExpertPrompt]) {
        self.name = name
        self.prompts = prompts
    }

    public static func loadJSONL(name: String, from url: URL) throws -> ExpertPromptSuite {
        let text = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        let prompts = try text.split(whereSeparator: \.isNewline)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { line in
                try decoder.decode(ExpertPrompt.self, from: Data(line.utf8))
            }
        return ExpertPromptSuite(name: name, prompts: prompts)
    }

    public func writeJSONL(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let lines = try prompts.map { prompt -> String in
            let data = try encoder.encode(prompt)
            return String(data: data, encoding: .utf8) ?? "{}"
        }
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
    }
}

public enum ExpertDomainTaxonomy {
    public static let domains: [String] = [
        "general",
        "code",
        "coding",
        "math",
        "formatting",
        "instruction_following",
        "reasoning",
        "language",
        "multilingual",
        "non_english",
        "chinese",
        "translation",
        "english_dominant",
        "unknown_language_role",
        "safety",
        "safety_medical_legal_sensitive",
        "safety_sensitive",
        "medical_sensitive",
        "legal_sensitive",
        "creative",
        "knowledge",
        "tools"
    ]

    public static let requiredReviewedPruneSemanticDomains: Set<String> = [
        "math",
        "code",
        "formatting",
        "instruction_following",
        "reasoning",
        "safety_medical_legal_sensitive",
        "chinese",
        "non_english",
        "multilingual",
        "translation",
        "english_dominant",
        "unknown_language_role"
    ]

    public static func semanticDomains(for prompt: ExpertPrompt) -> [String] {
        var domains: [String] = []
        func append(_ raw: String) {
            let canonical = canonicalSemanticDomain(raw)
            guard canonical != "general", !domains.contains(canonical) else { return }
            domains.append(canonical)
        }
        func appendFacets(_ raw: String) {
            append(raw)
            let slug = normalizedSemanticSlug(raw)
            if isNonEnglishSignal(slug), !domains.contains("non_english") {
                domains.append("non_english")
            }
            if isTranslationSignal(slug), !domains.contains("translation") {
                domains.append("translation")
            }
            if isSensitiveSignal(slug), !domains.contains("safety_medical_legal_sensitive") {
                domains.append("safety_medical_legal_sensitive")
            }
        }

        appendFacets(prompt.domain)
        if let subdomain = prompt.subdomain {
            appendFacets(subdomain)
        }
        for tag in prompt.tags {
            appendFacets(tag)
        }
        if domains.isEmpty {
            domains.append(canonicalDomain(prompt.domain))
        }
        return domains
    }

    public static func canonicalDomain(for prompt: ExpertPrompt) -> String {
        let domain = canonicalDomain(prompt.domain)
        if domain != "general" || prompt.domain.lowercased() == "general" {
            return domain
        }
        for tag in prompt.tags {
            let tagged = canonicalDomain(tag)
            if tagged != "general" {
                return tagged
            }
        }
        if let subdomain = prompt.subdomain {
            let sub = canonicalDomain(subdomain)
            if sub != "general" {
                return sub
            }
        }
        return domain
    }

    public static func canonicalDomain(_ raw: String) -> String {
        let slug = raw
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        switch slug {
        case "code", "coding", "swift", "python", "sql", "bugfinding", "concurrency", "syntax":
            return "coding"
        case "math", "arithmetic", "algebra", "estimation", "geometry", "statistics", "tensor":
            return "math"
        case "reasoning", "logic", "counterexample", "evidence", "verification":
            return "reasoning"
        case "multilingual", "language", "lang", "translation", "spanish", "french", "japanese", "chinese", "bilingual":
            return "language"
        case "robustness", "safety", "security", "refusal", "model-safety", "medical", "medicine-safety", "finance-safety":
            return "safety"
        case "creative", "writing", "tone":
            return "creative"
        case "knowledge", "domain", "retrieval", "data", "long-context", "science", "finance":
            return "knowledge"
        case "agentic", "tools", "tool", "cli", "structured", "json", "table", "classification", "model-pruning", "expert-lab", "prune", "workflow", "recovery", "optimization", "planning":
            return "tools"
        case "instruction", "instruction-following", "hierarchy", "clarification":
            return "reasoning"
        case "manual", "general", "explanation", "tradeoff", "debugging", "llm", "concise", "stepwise", "edge-case":
            return "general"
        default:
            return domains.contains(slug) ? slug : "general"
        }
    }

    public static func canonicalSemanticDomain(_ raw: String) -> String {
        let slug = normalizedSemanticSlug(raw)
        switch slug {
        case "code", "coding", "swift", "python", "sql", "bugfinding", "concurrency", "syntax":
            return "code"
        case "format", "formatting", "structured", "json", "table", "markdown":
            return "formatting"
        case "instruction", "instruction-following", "hierarchy", "clarification":
            return "instruction_following"
        case "multilingual", "language", "lang", "bilingual", "spanish", "french", "japanese":
            return "multilingual"
        case "non-english", "nonenglish", "romaji":
            return "non_english"
        case "chinese", "simplified-chinese", "traditional-chinese", "zh", "zh-cn", "zh-hans":
            return "chinese"
        case "translation", "translate", "back-translation":
            return "translation"
        case "english", "english-dominant", "english-dominant-role":
            return "english_dominant"
        case "unknown-language", "unknown-language-role", "language-id", "language-identification":
            return "unknown_language_role"
        case "safety-medical-legal-sensitive", "sensitive-domain", "sensitive":
            return "safety_medical_legal_sensitive"
        case "safety", "safety-sensitive", "robustness", "security", "refusal", "model-safety":
            return "safety_sensitive"
        case "medical", "medicine-safety", "medical-sensitive":
            return "medical_sensitive"
        case "legal", "law", "legal-sensitive":
            return "legal_sensitive"
        default:
            let broad = canonicalDomain(slug)
            return broad == "coding" ? "code" : broad
        }
    }

    public static func displayName(for raw: String) -> String {
        switch canonicalSemanticDomain(raw) {
        case "instruction_following":
            return "instruction following"
        case "safety_medical_legal_sensitive":
            return "safety/medical/legal sensitive"
        case "safety_sensitive":
            return "safety sensitive"
        case "medical_sensitive":
            return "medical sensitive"
        case "legal_sensitive":
            return "legal sensitive"
        default:
            return raw
        }
    }

    private static func normalizedSemanticSlug(_ raw: String) -> String {
        raw
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isNonEnglishSignal(_ slug: String) -> Bool {
        switch slug {
        case "multilingual", "bilingual", "spanish", "french", "japanese", "chinese",
             "simplified-chinese", "traditional-chinese", "zh", "zh-cn", "zh-hans",
             "non-english", "nonenglish", "romaji":
            return true
        default:
            return false
        }
    }

    private static func isTranslationSignal(_ slug: String) -> Bool {
        switch slug {
        case "translation", "translate", "back-translation":
            return true
        default:
            return false
        }
    }

    private static func isSensitiveSignal(_ slug: String) -> Bool {
        switch slug {
        case "safety", "safety-sensitive", "robustness", "security", "refusal", "model-safety",
             "medical", "medicine-safety", "medical-sensitive",
             "legal", "law", "legal-sensitive",
             "finance-safety":
            return true
        default:
            return false
        }
    }

    public static func canonicalCounts(_ counts: [String: Int]) -> [String: Int] {
        var out: [String: Int] = [:]
        for (domain, count) in counts {
            out[canonicalSemanticDomain(domain), default: 0] += count
        }
        return out
    }

    public static func canonicalLift(_ lift: [String: Float]) -> [String: Float] {
        var out: [String: Float] = [:]
        for (domain, value) in lift {
            let canonical = canonicalSemanticDomain(domain)
            out[canonical] = max(out[canonical] ?? 0, value)
        }
        return out
    }

    public static func dominantDomain(
        domains rawDomains: [String: Int],
        domainLift rawLift: [String: Float] = [:]
    ) -> String? {
        let counts = canonicalCounts(rawDomains).filter { $0.value > 0 }
        guard !counts.isEmpty else { return nil }
        let lift = canonicalLift(rawLift)
        let total = max(1, counts.values.reduce(0, +))
        let candidates = counts.map { domain, count -> (domain: String, count: Int, lift: Float, share: Float) in
            (domain, count, max(lift[domain] ?? 1, 0.001), Float(count) / Float(total))
        }
        let ranked = candidates.sorted {
            let lhsScore = signatureScore(domain: $0.domain, count: $0.count, lift: $0.lift, share: $0.share)
            let rhsScore = signatureScore(domain: $1.domain, count: $1.count, lift: $1.lift, share: $1.share)
            if lhsScore != rhsScore { return lhsScore > rhsScore }
            if $0.count != $1.count { return $0.count > $1.count }
            return $0.domain < $1.domain
        }
        return ranked.first?.domain
    }

    private static func signatureScore(domain: String, count: Int, lift: Float, share: Float) -> Float {
        let support = min(1, Float(log1p(Double(count)) / log(25.0)))
        let normalizedLift = max(0.05, min(lift, 4))
        let generalPenalty: Float = domain == "general" ? 0.82 : 1.0
        let aggregatePenalty: Float = domain == "safety_medical_legal_sensitive" ? 0.94 : 1.0
        return generalPenalty * aggregatePenalty * ((0.72 * normalizedLift) + (0.28 * share)) * support
    }
}

public struct ExpertPromptEvalOutcome: Codable, Equatable, Sendable {
    public let expectedKind: ExpertPromptExpectedKind
    public let expected: String?
    public let baselinePassed: Bool?
    public let maskedPassed: Bool?
    public let adapter: String
    public let risk: String

    public init(
        expectedKind: ExpertPromptExpectedKind,
        expected: String?,
        baselinePassed: Bool?,
        maskedPassed: Bool?,
        adapter: String,
        risk: String
    ) {
        self.expectedKind = expectedKind
        self.expected = expected
        self.baselinePassed = baselinePassed
        self.maskedPassed = maskedPassed
        self.adapter = adapter
        self.risk = risk
    }
}

public enum ExpertPromptEvaluator {
    public static let regressionSeverityNone = "none"
    public static let regressionSeverityWatch = "watch"
    public static let regressionSeverityHigh = "high"
    public static let regressionSeverityCritical = "critical"

    public static func evaluate(
        prompt: ExpertPrompt,
        baselineText: String,
        maskedText: String
    ) -> ExpertPromptEvalOutcome {
        switch prompt.expectedKind {
        case .freeform:
            return ExpertPromptEvalOutcome(
                expectedKind: prompt.expectedKind,
                expected: prompt.expected,
                baselinePassed: nil,
                maskedPassed: nil,
                adapter: "freeform_delta",
                risk: "not_scored"
            )
        case .judge:
            return ExpertPromptEvalOutcome(
                expectedKind: prompt.expectedKind,
                expected: prompt.expected,
                baselinePassed: nil,
                maskedPassed: nil,
                adapter: "external_judge_required",
                risk: "not_scored"
            )
        case .exact:
            let baselinePassed = exactMatch(baselineText, expected: prompt.expected)
            let maskedPassed = exactMatch(maskedText, expected: prompt.expected)
            return ExpertPromptEvalOutcome(
                expectedKind: prompt.expectedKind,
                expected: prompt.expected,
                baselinePassed: baselinePassed,
                maskedPassed: maskedPassed,
                adapter: "normalized_exact",
                risk: risk(baselinePassed: baselinePassed, maskedPassed: maskedPassed)
            )
        case .regex:
            let baselinePassed = regexMatch(baselineText, pattern: prompt.expected)
            let maskedPassed = regexMatch(maskedText, pattern: prompt.expected)
            return ExpertPromptEvalOutcome(
                expectedKind: prompt.expectedKind,
                expected: prompt.expected,
                baselinePassed: baselinePassed,
                maskedPassed: maskedPassed,
                adapter: "regex",
                risk: risk(baselinePassed: baselinePassed, maskedPassed: maskedPassed)
            )
        case .unitTest:
            let baselinePassed = regexMatch(baselineText, pattern: prompt.expected)
            let maskedPassed = regexMatch(maskedText, pattern: prompt.expected)
            return ExpertPromptEvalOutcome(
                expectedKind: prompt.expectedKind,
                expected: prompt.expected,
                baselinePassed: baselinePassed,
                maskedPassed: maskedPassed,
                adapter: "unit_test_expected_regex",
                risk: risk(baselinePassed: baselinePassed, maskedPassed: maskedPassed)
            )
        }
    }

    public static func normalizedTextDelta(_ lhs: String, _ rhs: String) -> Double {
        let maxLen = max(lhs.count, rhs.count, 1)
        let common = zip(lhs, rhs).filter { $0 == $1 }.count
        return Double(maxLen - common) / Double(maxLen)
    }

    public static func isHighRisk(evaluation: ExpertPromptEvalOutcome, textDelta: Double) -> Bool {
        let severity = regressionSeverity(evaluation: evaluation, textDelta: textDelta)
        return severity == regressionSeverityHigh || severity == regressionSeverityCritical
    }

    public static func regressionSeverity(evaluation: ExpertPromptEvalOutcome, textDelta: Double) -> String {
        if evaluation.risk == "regression" {
            return regressionSeverityCritical
        }
        if textDelta > 0.20 || evaluation.risk == "masked_improved" || evaluation.risk == "failed_baseline" {
            return regressionSeverityWatch
        }
        return regressionSeverityNone
    }

    public static func evaluationSummary(_ evaluation: ExpertPromptEvalOutcome, textDelta: Double) -> String {
        let delta = String(format: "%.2f", textDelta)
        guard let baselinePassed = evaluation.baselinePassed,
              let maskedPassed = evaluation.maskedPassed else {
            return "Eval \(evaluation.adapter): text delta \(delta)"
        }
        let baseline = baselinePassed ? "pass" : "fail"
        let masked = maskedPassed ? "pass" : "fail"
        return "Eval \(evaluation.expectedKind.rawValue): baseline \(baseline), masked \(masked), risk \(evaluation.risk), text delta \(delta)"
    }

    private static func exactMatch(_ text: String, expected: String?) -> Bool? {
        guard let expected, !expected.isEmpty else { return nil }
        return normalizedForExact(text) == normalizedForExact(expected)
    }

    private static func regexMatch(_ text: String, pattern: String?) -> Bool? {
        guard let pattern, !pattern.isEmpty else { return nil }
        do {
            let regex = try NSRegularExpression(pattern: pattern)
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            return regex.firstMatch(in: text, options: [], range: range) != nil
        } catch {
            return nil
        }
    }

    private static func normalizedForExact(_ text: String) -> String {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func risk(baselinePassed: Bool?, maskedPassed: Bool?) -> String {
        if baselinePassed == true && maskedPassed == false { return "regression" }
        if baselinePassed == false && maskedPassed == true { return "masked_improved" }
        if baselinePassed == true && maskedPassed == true { return "passed" }
        if baselinePassed == false && maskedPassed == false { return "failed_baseline" }
        return "not_scored"
    }
}

public struct ExpertPromptRun: Sendable {
    public let prompt: ExpertPrompt
    public let result: JANGKit.ExpertRunResult

    public init(prompt: ExpertPrompt, result: JANGKit.ExpertRunResult) {
        self.prompt = prompt
        self.result = result
    }
}

public actor ExpertPromptSuiteRunner {
    private let model: JANGKit.Model

    public init(model: JANGKit.Model) {
        self.model = model
    }

    public func run(
        suite: ExpertPromptSuite,
        config: JANGKit.SamplingConfig = JANGKit.SamplingConfig(maxTokens: 64),
        traceConfig: JANGKit.ExpertTraceConfig = JANGKit.ExpertTraceConfig()
    ) async throws -> [ExpertPromptRun] {
        var runs: [ExpertPromptRun] = []
        runs.reserveCapacity(suite.prompts.count)
        for prompt in suite.prompts {
            var promptConfig = config
            if let maxNewTokens = prompt.maxNewTokens, maxNewTokens > 0 {
                promptConfig.maxTokens = maxNewTokens
            }
            if let temperature = prompt.temperature {
                promptConfig.temperature = temperature
            }
            let result = try await model.generateWithTrace(
                prompt: prompt.text,
                config: promptConfig,
                traceConfig: traceConfig
            )
            runs.append(ExpertPromptRun(prompt: prompt, result: result))
        }
        return runs
    }
}

public struct ExpertAtlas: Codable, Equatable, Sendable {
    public let generatedAt: Date
    public let promptCount: Int
    public let sourceNumExpertsByLayer: [String: Int]?
    public let experts: [ExpertAtlasEntry]

    public init(
        generatedAt: Date = Date(),
        promptCount: Int,
        experts: [ExpertAtlasEntry],
        sourceNumExpertsByLayer: [String: Int]? = nil
    ) {
        self.generatedAt = generatedAt
        self.promptCount = promptCount
        self.experts = experts
        self.sourceNumExpertsByLayer = sourceNumExpertsByLayer
    }
}

public struct ExpertPromptEvidence: Codable, Equatable, Sendable, Identifiable {
    public var id: String { promptID }
    public let promptID: String
    public let domain: String
    public let subdomain: String?
    public let tags: [String]
    public let promptExcerpt: String
    public let hits: Int

    public init(
        promptID: String,
        domain: String,
        subdomain: String? = nil,
        tags: [String] = [],
        promptExcerpt: String,
        hits: Int
    ) {
        self.promptID = promptID
        self.domain = domain
        self.subdomain = subdomain
        self.tags = tags
        self.promptExcerpt = promptExcerpt
        self.hits = hits
    }
}

public struct ExpertAtlasEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: String { "\(layer):\(expert)" }
    public let layer: Int
    public let expert: Int
    public let hits: Int
    public let activationFrequency: Float
    public let probabilityMass: Float
    public let tokenCount: Int
    public let domains: [String: Int]
    public let domainLift: [String: Float]
    public let meanSelectedRank: Float
    public let entropyContribution: Float
    public let coactivationNeighbors: [ExpertCoactivationNeighbor]
    public let topPrompts: [String]
    public let promptEvidence: [ExpertPromptEvidence]?
    public let evidenceCount: Int?
    public let confidenceScore: Float
    public let meanTokenIndex: Float?
    public let minTokenIndex: Int?
    public let maxTokenIndex: Int?
    public let label: String
    public var generatedLabel: String
    public var userLabel: String?
    public var userNotes: String?
    public let isDead: Bool
    public let isHot: Bool

    private enum CodingKeys: String, CodingKey {
        case layer
        case expert
        case hits
        case activationFrequency
        case probabilityMass
        case tokenCount
        case domains
        case domainLift
        case meanSelectedRank
        case entropyContribution
        case coactivationNeighbors
        case topPrompts
        case promptEvidence
        case evidenceCount
        case confidenceScore
        case meanTokenIndex
        case minTokenIndex
        case maxTokenIndex
        case label
        case generatedLabel
        case userLabel
        case userNotes
        case isDead
        case isHot
    }

    public init(
        layer: Int,
        expert: Int,
        hits: Int,
        activationFrequency: Float = 0,
        probabilityMass: Float,
        tokenCount: Int,
        domains: [String: Int],
        domainLift: [String: Float] = [:],
        meanSelectedRank: Float = 0,
        entropyContribution: Float = 0,
        coactivationNeighbors: [ExpertCoactivationNeighbor] = [],
        topPrompts: [String] = [],
        promptEvidence: [ExpertPromptEvidence] = [],
        evidenceCount: Int? = nil,
        confidenceScore: Float = 0,
        meanTokenIndex: Float? = nil,
        minTokenIndex: Int? = nil,
        maxTokenIndex: Int? = nil,
        label: String,
        generatedLabel: String? = nil,
        userLabel: String? = nil,
        userNotes: String? = nil,
        isDead: Bool,
        isHot: Bool
    ) {
        self.layer = layer
        self.expert = expert
        self.hits = hits
        self.activationFrequency = activationFrequency
        self.probabilityMass = probabilityMass
        self.tokenCount = tokenCount
        self.domains = domains
        self.domainLift = domainLift
        self.meanSelectedRank = meanSelectedRank
        self.entropyContribution = entropyContribution
        self.coactivationNeighbors = coactivationNeighbors
        self.topPrompts = topPrompts
        self.promptEvidence = promptEvidence
        self.evidenceCount = evidenceCount
        self.confidenceScore = confidenceScore
        self.meanTokenIndex = meanTokenIndex
        self.minTokenIndex = minTokenIndex
        self.maxTokenIndex = maxTokenIndex
        self.label = label
        self.generatedLabel = generatedLabel ?? label
        self.userLabel = userLabel
        self.userNotes = userNotes
        self.isDead = isDead
        self.isHot = isHot
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let label = try container.decode(String.self, forKey: .label)
        self.layer = try container.decode(Int.self, forKey: .layer)
        self.expert = try container.decode(Int.self, forKey: .expert)
        self.hits = try container.decode(Int.self, forKey: .hits)
        self.activationFrequency = try container.decodeIfPresent(Float.self, forKey: .activationFrequency) ?? 0
        self.probabilityMass = try container.decode(Float.self, forKey: .probabilityMass)
        self.tokenCount = try container.decode(Int.self, forKey: .tokenCount)
        self.domains = try container.decode([String: Int].self, forKey: .domains)
        self.domainLift = try container.decodeIfPresent([String: Float].self, forKey: .domainLift) ?? [:]
        self.meanSelectedRank = try container.decodeIfPresent(Float.self, forKey: .meanSelectedRank) ?? 0
        self.entropyContribution = try container.decodeIfPresent(Float.self, forKey: .entropyContribution) ?? 0
        self.coactivationNeighbors = try container.decodeIfPresent(
            [ExpertCoactivationNeighbor].self,
            forKey: .coactivationNeighbors
        ) ?? []
        self.topPrompts = try container.decodeIfPresent([String].self, forKey: .topPrompts) ?? []
        self.promptEvidence = try container.decodeIfPresent([ExpertPromptEvidence].self, forKey: .promptEvidence)
        self.evidenceCount = try container.decodeIfPresent(Int.self, forKey: .evidenceCount)
        self.confidenceScore = try container.decodeIfPresent(Float.self, forKey: .confidenceScore) ?? 0
        self.meanTokenIndex = try container.decodeIfPresent(Float.self, forKey: .meanTokenIndex)
        self.minTokenIndex = try container.decodeIfPresent(Int.self, forKey: .minTokenIndex)
        self.maxTokenIndex = try container.decodeIfPresent(Int.self, forKey: .maxTokenIndex)
        self.label = label
        self.generatedLabel = try container.decodeIfPresent(String.self, forKey: .generatedLabel) ?? label
        self.userLabel = try container.decodeIfPresent(String.self, forKey: .userLabel)
        self.userNotes = try container.decodeIfPresent(String.self, forKey: .userNotes)
        self.isDead = try container.decode(Bool.self, forKey: .isDead)
        self.isHot = try container.decode(Bool.self, forKey: .isHot)
    }
}

public struct ExpertCoactivationNeighbor: Codable, Equatable, Sendable {
    public let expert: Int
    public let count: Int
    public let jaccard: Float
    public let pmi: Float

    public init(expert: Int, count: Int, jaccard: Float, pmi: Float) {
        self.expert = expert
        self.count = count
        self.jaccard = jaccard
        self.pmi = pmi
    }
}

public struct ExpertPrunePlan: Codable, Equatable, Sendable {
    public let version: Int
    public let schema: String
    public let generatedAt: Date
    public let method: String
    public let sourceModelPath: String?
    public let reviewBundlePath: String?
    public let runID: String?
    public let atlasID: String?
    public let evalArtifactPath: String?
    public let promptCount: Int
    public let sourceNumExperts: Int?
    public let keepExpertsPerLayer: Int
    public let comparisonSummary: ExpertComparisonSummary?
    public let evalIndex: ExpertEvalIndexSummary?
    public let safety: ExpertPrunePlanSafety?
    public let target: ExpertPrunePlanTarget
    public let layers: [String: ExpertPrunePlanLayer]

    public init(
        version: Int = 1,
        schema: String = "jang-expert-prune-plan-v1",
        generatedAt: Date = Date(),
        method: String,
        sourceModelPath: String?,
        reviewBundlePath: String? = nil,
        runID: String? = nil,
        atlasID: String? = nil,
        evalArtifactPath: String? = nil,
        promptCount: Int,
        sourceNumExperts: Int?,
        keepExpertsPerLayer: Int,
        comparisonSummary: ExpertComparisonSummary? = nil,
        evalIndex: ExpertEvalIndexSummary? = nil,
        safety: ExpertPrunePlanSafety? = nil,
        target: ExpertPrunePlanTarget? = nil,
        layers: [String: ExpertPrunePlanLayer]
    ) {
        self.version = version
        self.schema = schema
        self.generatedAt = generatedAt
        self.method = method
        self.sourceModelPath = sourceModelPath
        self.reviewBundlePath = reviewBundlePath
        self.runID = runID
        self.atlasID = atlasID
        self.evalArtifactPath = evalArtifactPath
        self.promptCount = promptCount
        self.sourceNumExperts = sourceNumExperts
        self.keepExpertsPerLayer = keepExpertsPerLayer
        self.comparisonSummary = comparisonSummary
        self.evalIndex = evalIndex
        self.safety = safety
        self.target = target ?? ExpertPrunePlanTarget(type: "keep_per_layer", keepExpertsPerLayer: keepExpertsPerLayer)
        self.layers = layers
    }

    enum CodingKeys: String, CodingKey {
        case version
        case schema
        case generatedAt
        case method
        case sourceModelPath = "source_model"
        case reviewBundlePath = "review_bundle"
        case runID = "run_id"
        case atlasID = "atlas_id"
        case evalArtifactPath = "eval_artifact"
        case promptCount
        case sourceNumExperts
        case keepExpertsPerLayer
        case comparisonSummary = "comparison_summary"
        case evalIndex = "eval_index"
        case safety
        case target
        case layers
    }
}

public struct ExpertPrunePlanSafety: Codable, Equatable, Sendable {
    public let passed: Bool
    public let minimumActiveExpertsPerLayer: Int
    public let trainedTopKByLayer: [String: Int]
    public let issues: [String]

    public init(
        passed: Bool,
        minimumActiveExpertsPerLayer: Int,
        trainedTopKByLayer: [String: Int],
        issues: [String] = []
    ) {
        self.passed = passed
        self.minimumActiveExpertsPerLayer = minimumActiveExpertsPerLayer
        self.trainedTopKByLayer = trainedTopKByLayer
        self.issues = issues
    }

    enum CodingKeys: String, CodingKey {
        case passed
        case minimumActiveExpertsPerLayer = "minimum_active_experts_per_layer"
        case trainedTopKByLayer = "trained_top_k_by_layer"
        case issues
    }
}

public struct ExpertEvalIndexSummary: Codable, Equatable, Sendable {
    public let schema: String
    public let promptCount: Int
    public let promptIDs: [String]
    public let riskyPromptIDs: [String]
    public let highRiskDomains: [String]
    public let passRateBaseline: Double?
    public let passRateMasked: Double?
    public let validatorSchema: String?
    public let validatorAvailablePromptCount: Int?
    public let promptClassificationCounts: [String: Int]?
    public let baselineQualifiedPromptCount: Int?
    public let baselineQualifiedPromptIDs: [String]?
    public let baselineInvalidPromptIDs: [String]?
    public let inconclusivePromptIDs: [String]?
    public let preservedPromptIDs: [String]?
    public let degradedPromptIDs: [String]?
    public let baselineQualifiedMaskedPassRate: Double?
    public let baselineQualifiedSemanticCoverage: [String]?
    public let missingBaselineQualifiedSemanticCoverage: [String]?
    public let meanTextDelta: Double
    public let minBaselineTokens: Int?
    public let minMaskedTokens: Int?
    public let meanBaselineTokens: Double?
    public let meanMaskedTokens: Double?
    public let baselineRouteRecordCount: Int?
    public let maskedRouteRecordCount: Int?
    public let baselineLayerStatsPromptCount: Int?
    public let maskedLayerStatsPromptCount: Int?
    public let generationSettingsChecked: Bool?
    public let suiteJSONL: String?
    public let suiteSHA256: String?
    public let evalJSONL: String?
    public let evalTraceJSONL: String?
    public let comparisonSummary: String?
    public let mask: String?
    public let maskJSON: String?
    public let semanticCoverage: [String]?
    public let missingSemanticCoverage: [String]?
    public let runtimeMode: String?
    public let runtimeBackend: String?
    public let runtimeDevice: String?
    public let runtimeMetalEnabled: Bool?
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
    public let regressionSeverity: String?

    public init(
        schema: String = "jang-expert-lab-eval-index-v1",
        promptCount: Int,
        promptIDs: [String] = [],
        riskyPromptIDs: [String] = [],
        highRiskDomains: [String] = [],
        passRateBaseline: Double? = nil,
        passRateMasked: Double? = nil,
        validatorSchema: String? = nil,
        validatorAvailablePromptCount: Int? = nil,
        promptClassificationCounts: [String: Int]? = nil,
        baselineQualifiedPromptCount: Int? = nil,
        baselineQualifiedPromptIDs: [String]? = nil,
        baselineInvalidPromptIDs: [String]? = nil,
        inconclusivePromptIDs: [String]? = nil,
        preservedPromptIDs: [String]? = nil,
        degradedPromptIDs: [String]? = nil,
        baselineQualifiedMaskedPassRate: Double? = nil,
        baselineQualifiedSemanticCoverage: [String]? = nil,
        missingBaselineQualifiedSemanticCoverage: [String]? = nil,
        meanTextDelta: Double = 0,
        minBaselineTokens: Int? = nil,
        minMaskedTokens: Int? = nil,
        meanBaselineTokens: Double? = nil,
        meanMaskedTokens: Double? = nil,
        baselineRouteRecordCount: Int? = nil,
        maskedRouteRecordCount: Int? = nil,
        baselineLayerStatsPromptCount: Int? = nil,
        maskedLayerStatsPromptCount: Int? = nil,
        generationSettingsChecked: Bool? = nil,
        suiteJSONL: String? = nil,
        suiteSHA256: String? = nil,
        evalJSONL: String? = nil,
        evalTraceJSONL: String? = nil,
        comparisonSummary: String? = nil,
        mask: String? = nil,
        maskJSON: String? = nil,
        semanticCoverage: [String]? = nil,
        missingSemanticCoverage: [String]? = nil,
        runtimeMode: String? = nil,
        runtimeBackend: String? = nil,
        runtimeDevice: String? = nil,
        runtimeMetalEnabled: Bool? = nil,
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
        regressionSeverity: String? = nil
    ) {
        self.schema = schema
        self.promptCount = promptCount
        self.promptIDs = promptIDs
        self.riskyPromptIDs = riskyPromptIDs
        self.highRiskDomains = highRiskDomains
        self.passRateBaseline = passRateBaseline
        self.passRateMasked = passRateMasked
        self.validatorSchema = validatorSchema
        self.validatorAvailablePromptCount = validatorAvailablePromptCount
        self.promptClassificationCounts = promptClassificationCounts
        self.baselineQualifiedPromptCount = baselineQualifiedPromptCount
        self.baselineQualifiedPromptIDs = baselineQualifiedPromptIDs
        self.baselineInvalidPromptIDs = baselineInvalidPromptIDs
        self.inconclusivePromptIDs = inconclusivePromptIDs
        self.preservedPromptIDs = preservedPromptIDs
        self.degradedPromptIDs = degradedPromptIDs
        self.baselineQualifiedMaskedPassRate = baselineQualifiedMaskedPassRate
        self.baselineQualifiedSemanticCoverage = baselineQualifiedSemanticCoverage
        self.missingBaselineQualifiedSemanticCoverage = missingBaselineQualifiedSemanticCoverage
        self.meanTextDelta = meanTextDelta
        self.minBaselineTokens = minBaselineTokens
        self.minMaskedTokens = minMaskedTokens
        self.meanBaselineTokens = meanBaselineTokens
        self.meanMaskedTokens = meanMaskedTokens
        self.baselineRouteRecordCount = baselineRouteRecordCount
        self.maskedRouteRecordCount = maskedRouteRecordCount
        self.baselineLayerStatsPromptCount = baselineLayerStatsPromptCount
        self.maskedLayerStatsPromptCount = maskedLayerStatsPromptCount
        self.generationSettingsChecked = generationSettingsChecked
        self.suiteJSONL = suiteJSONL
        self.suiteSHA256 = suiteSHA256
        self.evalJSONL = evalJSONL
        self.evalTraceJSONL = evalTraceJSONL
        self.comparisonSummary = comparisonSummary
        self.mask = mask
        self.maskJSON = maskJSON ?? mask
        self.semanticCoverage = semanticCoverage
        self.missingSemanticCoverage = missingSemanticCoverage
        self.runtimeMode = runtimeMode
        self.runtimeBackend = runtimeBackend
        self.runtimeDevice = runtimeDevice
        self.runtimeMetalEnabled = runtimeMetalEnabled
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
        self.regressionSeverity = regressionSeverity
    }

    enum CodingKeys: String, CodingKey {
        case schema
        case promptCount = "prompt_count"
        case promptIDs = "prompt_ids"
        case riskyPromptIDs = "risky_prompt_ids"
        case highRiskDomains = "high_risk_domains"
        case passRateBaseline = "pass_rate_baseline"
        case passRateMasked = "pass_rate_masked"
        case validatorSchema = "validator_schema"
        case validatorAvailablePromptCount = "validator_available_prompt_count"
        case promptClassificationCounts = "prompt_classification_counts"
        case baselineQualifiedPromptCount = "baseline_qualified_prompt_count"
        case baselineQualifiedPromptIDs = "baseline_qualified_prompt_ids"
        case baselineInvalidPromptIDs = "baseline_invalid_prompt_ids"
        case inconclusivePromptIDs = "inconclusive_prompt_ids"
        case preservedPromptIDs = "preserved_prompt_ids"
        case degradedPromptIDs = "degraded_prompt_ids"
        case baselineQualifiedMaskedPassRate = "baseline_qualified_masked_pass_rate"
        case baselineQualifiedSemanticCoverage = "baseline_qualified_semantic_coverage"
        case missingBaselineQualifiedSemanticCoverage = "missing_baseline_qualified_semantic_coverage"
        case meanTextDelta = "mean_text_delta"
        case minBaselineTokens = "min_baseline_tokens"
        case minMaskedTokens = "min_masked_tokens"
        case meanBaselineTokens = "mean_baseline_tokens"
        case meanMaskedTokens = "mean_masked_tokens"
        case baselineRouteRecordCount = "baseline_route_record_count"
        case maskedRouteRecordCount = "masked_route_record_count"
        case baselineLayerStatsPromptCount = "baseline_layer_stats_prompt_count"
        case maskedLayerStatsPromptCount = "masked_layer_stats_prompt_count"
        case generationSettingsChecked = "generation_settings_checked"
        case suiteJSONL = "suite_jsonl"
        case suiteSHA256 = "suite_sha256"
        case evalJSONL = "eval_jsonl"
        case evalTraceJSONL = "eval_trace_jsonl"
        case comparisonSummary = "comparison_summary"
        case mask
        case maskJSON = "mask_json"
        case semanticCoverage = "semantic_coverage"
        case missingSemanticCoverage = "missing_semantic_coverage"
        case runtimeMode = "runtime_mode"
        case runtimeBackend = "runtime_backend"
        case runtimeDevice = "runtime_device"
        case runtimeMetalEnabled = "runtime_metal_enabled"
        case jangToolsVersion = "jang_tools_version"
        case mlxVersion = "mlx_version"
        case mlxLMVersion = "mlx_lm_version"
        case mlxVLMVersion = "mlx_vlm_version"
        case sourceModelPath = "source_model_path"
        case hookedMOELayers = "hooked_moe_layers"
        case expectedMOELayers = "expected_moe_layers"
        case hookCoverageComplete = "hook_coverage_complete"
        case maskApplied = "mask_applied"
        case disabledExpertCount = "disabled_expert_count"
        case topKOverride = "top_k_override"
        case regressionSeverity = "regression_severity"
    }
}

public struct ExpertPrunePlanTarget: Codable, Equatable, Sendable {
    public let type: String
    public let keepExpertsPerLayer: Int?
    public let dropExpertsPerLayer: Int?
    public let targetSizeReductionPct: Double?

    public init(
        type: String,
        keepExpertsPerLayer: Int? = nil,
        dropExpertsPerLayer: Int? = nil,
        targetSizeReductionPct: Double? = nil
    ) {
        self.type = type
        self.keepExpertsPerLayer = keepExpertsPerLayer
        self.dropExpertsPerLayer = dropExpertsPerLayer
        self.targetSizeReductionPct = targetSizeReductionPct
    }

    enum CodingKeys: String, CodingKey {
        case type
        case keepExpertsPerLayer = "keep_experts_per_layer"
        case dropExpertsPerLayer = "drop_experts_per_layer"
        case targetSizeReductionPct = "target_size_reduction_pct"
    }
}

public struct ExpertPrunePlanLayer: Codable, Equatable, Sendable {
    public let layer: Int?
    public let numSourceExperts: Int?
    public let keep: [Int]
    public let drop: [Int]
    public let lockedKeep: [Int]
    public let userForcedDrop: [Int]
    public let evidence: [ExpertPrunePlanEvidence]

    public init(
        layer: Int? = nil,
        numSourceExperts: Int? = nil,
        keep: [Int],
        drop: [Int],
        lockedKeep: [Int] = [],
        userForcedDrop: [Int] = [],
        evidence: [ExpertPrunePlanEvidence]
    ) {
        self.layer = layer
        self.numSourceExperts = numSourceExperts
        self.keep = keep
        self.drop = drop
        self.lockedKeep = lockedKeep
        self.userForcedDrop = userForcedDrop
        self.evidence = evidence
    }

    enum CodingKeys: String, CodingKey {
        case layer
        case numSourceExperts = "num_source_experts"
        case keep
        case drop
        case lockedKeep = "locked_keep"
        case userForcedDrop = "user_forced_drop"
        case evidence
    }
}

public struct ExpertPrunePlanEvidence: Codable, Equatable, Sendable {
    public let expert: Int
    public let hits: Int
    public let probabilityMass: Float
    public let frequency: Float
    public let routerMass: Float
    public let ablationDelta: Float?
    public let maskedImpactScope: String?
    public let reviewedMaskMember: Bool
    public let evidenceCount: Int?
    public let domains: [String: Int]
    public let domainLift: [String: Float]
    public let promptEvidence: [ExpertPromptEvidence]
    public let label: String
    public let userNotes: String?
    public let reason: String
    public let userForcedDrop: Bool
    public let kept: Bool

    public init(
        expert: Int,
        hits: Int,
        probabilityMass: Float,
        frequency: Float = 0,
        routerMass: Float? = nil,
        ablationDelta: Float? = nil,
        maskedImpactScope: String? = nil,
        reviewedMaskMember: Bool = false,
        evidenceCount: Int? = nil,
        domains: [String: Int],
        domainLift: [String: Float] = [:],
        promptEvidence: [ExpertPromptEvidence] = [],
        label: String,
        userNotes: String? = nil,
        reason: String = "",
        userForcedDrop: Bool = false,
        kept: Bool
    ) {
        self.expert = expert
        self.hits = hits
        self.probabilityMass = probabilityMass
        self.frequency = frequency
        self.routerMass = routerMass ?? probabilityMass
        self.ablationDelta = ablationDelta
        self.maskedImpactScope = maskedImpactScope
        self.reviewedMaskMember = reviewedMaskMember
        self.evidenceCount = evidenceCount
        self.domains = domains
        self.domainLift = domainLift
        self.promptEvidence = promptEvidence
        self.label = label
        self.userNotes = userNotes
        self.reason = reason
        self.userForcedDrop = userForcedDrop
        self.kept = kept
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        expert = try c.decode(Int.self, forKey: .expert)
        hits = try c.decode(Int.self, forKey: .hits)
        probabilityMass = try c.decode(Float.self, forKey: .probabilityMass)
        frequency = try c.decodeIfPresent(Float.self, forKey: .frequency) ?? 0
        routerMass = try c.decodeIfPresent(Float.self, forKey: .routerMass) ?? probabilityMass
        ablationDelta = try c.decodeIfPresent(Float.self, forKey: .ablationDelta)
        maskedImpactScope = try c.decodeIfPresent(String.self, forKey: .maskedImpactScope)
        reviewedMaskMember = try c.decodeIfPresent(Bool.self, forKey: .reviewedMaskMember) ?? false
        evidenceCount = try c.decodeIfPresent(Int.self, forKey: .evidenceCount)
        domains = try c.decode([String: Int].self, forKey: .domains)
        domainLift = try c.decodeIfPresent([String: Float].self, forKey: .domainLift) ?? [:]
        promptEvidence = try c.decodeIfPresent([ExpertPromptEvidence].self, forKey: .promptEvidence) ?? []
        label = try c.decode(String.self, forKey: .label)
        userNotes = try c.decodeIfPresent(String.self, forKey: .userNotes)
        reason = try c.decodeIfPresent(String.self, forKey: .reason) ?? ""
        userForcedDrop = try c.decodeIfPresent(Bool.self, forKey: .userForcedDrop) ?? false
        kept = try c.decode(Bool.self, forKey: .kept)
    }

    enum CodingKeys: String, CodingKey {
        case expert
        case hits
        case probabilityMass
        case frequency
        case routerMass = "router_mass"
        case ablationDelta = "ablation_delta"
        case maskedImpactScope = "masked_impact_scope"
        case reviewedMaskMember = "reviewed_mask_member"
        case evidenceCount = "evidence_count"
        case domains
        case domainLift = "domain_lift"
        case promptEvidence = "prompt_evidence"
        case label
        case userNotes = "user_notes"
        case reason
        case userForcedDrop = "user_forced_drop"
        case kept
    }
}

public enum ExpertPrunePlanBuilder {
    public static func build(
        from atlas: ExpertAtlas,
        keepExpertsPerLayer: Int,
        sourceNumExpertsByLayer: [Int: Int] = [:],
        trainedTopKByLayer: [Int: Int] = [:],
        forceDropByLayer: [Int: Set<Int>] = [:],
        lockedKeepByLayer: [Int: Set<Int>] = [:],
        comparisonSummary: ExpertComparisonSummary? = nil,
        evalIndex: ExpertEvalIndexSummary? = nil,
        sourceModelPath: String? = nil,
        reviewBundlePath: String? = nil,
        runID: String? = nil,
        atlasID: String? = nil,
        evalArtifactPath: String? = nil
    ) throws -> ExpertPrunePlan {
        guard keepExpertsPerLayer > 0 else {
            throw ExpertPrunePlanError.invalidKeepCount(keepExpertsPerLayer)
        }
        if let comparisonSummary {
            if let issue = reviewedComparisonIssue(comparisonSummary, atlasPromptCount: atlas.promptCount) {
                throw ExpertPrunePlanError.reviewedComparisonFailed(issue)
            }
            guard let evalIndex else {
                throw ExpertPrunePlanError.reviewedEvalIndexFailed("missing per-prompt eval_index evidence")
            }
            if let issue = reviewedEvalIndexIssue(
                evalIndex,
                comparison: comparisonSummary,
                sourceModelPath: sourceModelPath
            ) {
                throw ExpertPrunePlanError.reviewedEvalIndexFailed(issue)
            }
        }

        let entriesByLayer = Dictionary(grouping: atlas.experts, by: \.layer)
        let safeDropCoordinates = Set(comparisonSummary?.safeDropCandidates ?? [])
        let highRiskDomains = Set(comparisonSummary?.highRiskDomains ?? [])
        var planLayers: [String: ExpertPrunePlanLayer] = [:]
        var sourceExpertCounts = Set<Int>()
        var safetyTopKByLayer: [String: Int] = [:]

        for layer in entriesByLayer.keys.sorted() {
            let entries = entriesByLayer[layer] ?? []
            let observedCount = (entries.map(\.expert).max() ?? -1) + 1
            let sourceCount = sourceNumExpertsByLayer[layer] ?? observedCount
            guard sourceCount > 0 else { continue }
            let trainedTopK = max(1, trainedTopKByLayer[layer] ?? trainedTopKByLayer.values.first ?? 1)
            guard keepExpertsPerLayer >= trainedTopK else {
                throw ExpertPrunePlanError.keepCountBelowTopK(
                    keepExpertsPerLayer: keepExpertsPerLayer,
                    trainedTopK: trainedTopK,
                    layer: layer
                )
            }
            safetyTopKByLayer[String(layer)] = trainedTopK
            guard keepExpertsPerLayer < sourceCount else {
                throw ExpertPrunePlanError.keepCountTooHigh(
                    keepExpertsPerLayer: keepExpertsPerLayer,
                    sourceExperts: sourceCount,
                    layer: layer
                )
            }
            sourceExpertCounts.insert(sourceCount)
            let forcedDrops = forceDropByLayer[layer] ?? []
            if let invalid = forcedDrops.first(where: { $0 < 0 || $0 >= sourceCount }) {
                throw ExpertPrunePlanError.forceDropOutOfRange(
                    expert: invalid,
                    sourceExperts: sourceCount,
                    layer: layer
                )
            }
            if sourceCount - forcedDrops.count < keepExpertsPerLayer {
                throw ExpertPrunePlanError.forceDropLeavesTooFewExperts(
                    keepExpertsPerLayer: keepExpertsPerLayer,
                    availableExperts: sourceCount - forcedDrops.count,
                    layer: layer
                )
            }
            let lockedKeeps = lockedKeepByLayer[layer] ?? []
            if let invalid = lockedKeeps.first(where: { $0 < 0 || $0 >= sourceCount }) {
                throw ExpertPrunePlanError.lockedKeepOutOfRange(
                    expert: invalid,
                    sourceExperts: sourceCount,
                    layer: layer
                )
            }
            if let collision = lockedKeeps.intersection(forcedDrops).first {
                throw ExpertPrunePlanError.lockedKeepForcedDropConflict(expert: collision, layer: layer)
            }
            if lockedKeeps.count > keepExpertsPerLayer {
                throw ExpertPrunePlanError.lockedKeepExceedsKeepCount(
                    lockedKeep: lockedKeeps.count,
                    keepExpertsPerLayer: keepExpertsPerLayer,
                    layer: layer
                )
            }

            let byExpert = Dictionary(uniqueKeysWithValues: entries.map { ($0.expert, $0) })
            let candidates = (0..<sourceCount).map { expert in
                byExpert[expert] ?? ExpertAtlasEntry(
                    layer: layer,
                    expert: expert,
                    hits: 0,
                    probabilityMass: 0,
                    tokenCount: 0,
                    domains: [:],
                    label: "unobserved",
                    isDead: true,
                    isHot: false
                )
            }
            let rankableCandidates = candidates.filter {
                !forcedDrops.contains($0.expert) && !lockedKeeps.contains($0.expert)
            }
            let safeDropExperts = Set(safeDropCoordinates.filter { $0.layer == layer }.map(\.expert))

            let ranked = rankableCandidates.sorted { lhs, rhs in
                let lhsHighRisk = touchesHighRiskDomain(lhs, highRiskDomains: highRiskDomains)
                let rhsHighRisk = touchesHighRiskDomain(rhs, highRiskDomains: highRiskDomains)
                if lhsHighRisk != rhsHighRisk { return lhsHighRisk && !rhsHighRisk }
                let lhsSafeDrop = safeDropExperts.contains(lhs.expert)
                let rhsSafeDrop = safeDropExperts.contains(rhs.expert)
                if lhsSafeDrop != rhsSafeDrop { return !lhsSafeDrop && rhsSafeDrop }
                if lhs.hits != rhs.hits { return lhs.hits > rhs.hits }
                if lhs.probabilityMass != rhs.probabilityMass { return lhs.probabilityMass > rhs.probabilityMass }
                if lhs.domains.count != rhs.domains.count { return lhs.domains.count > rhs.domains.count }
                if lhs.isHot != rhs.isHot { return lhs.isHot && !rhs.isHot }
                return lhs.expert < rhs.expert
            }
            let keep = (Array(lockedKeeps) + ranked.prefix(keepExpertsPerLayer - lockedKeeps.count).map(\.expert)).sorted()
            let keptSet = Set(keep)
            let drop = (0..<sourceCount).filter { !keptSet.contains($0) }
            if comparisonSummary != nil {
                let unsafeDrops = Set(drop).subtracting(safeDropExperts)
                if !unsafeDrops.isEmpty {
                    throw ExpertPrunePlanError.dropOutsideSameSuiteSafeDropCandidates(
                        experts: unsafeDrops.sorted(),
                        layer: layer
                    )
                }
            }

            let evidenceRank = candidates.sorted { lhs, rhs in
                let lhsKept = keptSet.contains(lhs.expert)
                let rhsKept = keptSet.contains(rhs.expert)
                if lhsKept != rhsKept { return lhsKept && !rhsKept }
                if lhs.hits != rhs.hits { return lhs.hits > rhs.hits }
                if lhs.probabilityMass != rhs.probabilityMass { return lhs.probabilityMass > rhs.probabilityMass }
                return lhs.expert < rhs.expert
            }
            let evidence = evidenceRank.map { entry in
                let userForced = forcedDrops.contains(entry.expert)
                let frequency = entry.activationFrequency
                let label = reviewedLabel(for: entry)
                let userNotes = reviewedNotes(for: entry)
                let coordinate = ExpertCoordinate(layer: layer, expert: entry.expert)
                let evalSafeDrop = safeDropCoordinates.contains(coordinate)
                let maskedImpactDelta = comparisonSummary.map { Float($0.meanTextDelta) }
                let maskedImpactScope = comparisonSummary == nil ? nil : "same_suite_mask_mean_text_delta"
                return ExpertPrunePlanEvidence(
                    expert: entry.expert,
                    hits: entry.hits,
                    probabilityMass: entry.probabilityMass,
                    frequency: frequency,
                    ablationDelta: maskedImpactDelta,
                    maskedImpactScope: maskedImpactScope,
                    reviewedMaskMember: evalSafeDrop || userForced,
                    evidenceCount: entry.evidenceCount,
                    domains: entry.domains,
                    domainLift: entry.domainLift,
                    promptEvidence: entry.promptEvidence ?? [],
                    label: userForced ? "\(label) · user-disabled" : label,
                    userNotes: userNotes,
                    reason: reason(
                        for: entry,
                        kept: keptSet.contains(entry.expert),
                        userForcedDrop: userForced,
                        evalSafeDrop: evalSafeDrop,
                        comparisonSummary: comparisonSummary,
                        highRiskDomains: highRiskDomains
                    ),
                    userForcedDrop: userForced,
                    kept: keptSet.contains(entry.expert)
                )
            }

            planLayers[String(layer)] = ExpertPrunePlanLayer(
                layer: layer,
                numSourceExperts: sourceCount,
                keep: keep,
                drop: drop,
                lockedKeep: lockedKeeps.sorted(),
                userForcedDrop: forcedDrops.sorted(),
                evidence: evidence
            )
        }

        let uniformSourceExperts = sourceExpertCounts.count == 1 ? sourceExpertCounts.first : nil
        let safety = ExpertPrunePlanSafety(
            passed: true,
            minimumActiveExpertsPerLayer: keepExpertsPerLayer,
            trainedTopKByLayer: safetyTopKByLayer,
            issues: []
        )
        return ExpertPrunePlan(
            method: "prompt_trace_hits_mass_domain_lift_v1",
            sourceModelPath: sourceModelPath,
            reviewBundlePath: reviewBundlePath,
            runID: runID,
            atlasID: atlasID,
            evalArtifactPath: evalArtifactPath,
            promptCount: atlas.promptCount,
            sourceNumExperts: uniformSourceExperts,
            keepExpertsPerLayer: keepExpertsPerLayer,
            comparisonSummary: comparisonSummary,
            evalIndex: evalIndex,
            safety: safety,
            layers: planLayers
        )
    }

    private static func reviewedLabel(for entry: ExpertAtlasEntry) -> String {
        let userLabel = entry.userLabel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return userLabel.isEmpty ? entry.label : userLabel
    }

    private static func reviewedNotes(for entry: ExpertAtlasEntry) -> String? {
        let notes = entry.userNotes?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return notes.isEmpty ? nil : entry.userNotes
    }

    private static func reason(
        for entry: ExpertAtlasEntry,
        kept: Bool,
        userForcedDrop: Bool,
        evalSafeDrop: Bool,
        comparisonSummary: ExpertComparisonSummary?,
        highRiskDomains: Set<String>
    ) -> String {
        let label = reviewedLabel(for: entry)
        let base: String
        if userForcedDrop {
            base = "user-forced drop; \(label); activity \(entry.hits) hits"
        } else if kept {
            if entry.isHot {
                base = "kept because it is hot and trace-active"
            } else if !entry.domains.isEmpty {
                let domains = entry.domains.keys.sorted().joined(separator: ", ")
                base = "kept for domain coverage: \(domains)"
            } else {
                base = "kept by prompt-trace activity score"
            }
        } else if entry.isDead {
            base = "drop candidate: dead across the baseline prompt suite"
        } else {
            base = "drop candidate: lower prompt activity and router mass than kept experts"
        }

        var notes: [String] = []
        if evalSafeDrop, let comparisonSummary {
            notes.append(String(
                format: "masked A/B safe: %d prompts, mean text delta %.4f, masked pass %@",
                comparisonSummary.promptCount,
                comparisonSummary.meanTextDelta,
                passRateDescription(comparisonSummary.passRateMasked)
            ))
        }
        let riskyDomains = entry.domains.keys.filter { highRiskDomains.contains($0) }.sorted()
        if !riskyDomains.isEmpty {
            notes.append("A/B high-risk domain: \(riskyDomains.joined(separator: ", "))")
        }
        return ([base] + notes).joined(separator: "; ")
    }

    private static func touchesHighRiskDomain(
        _ entry: ExpertAtlasEntry,
        highRiskDomains: Set<String>
    ) -> Bool {
        !highRiskDomains.isDisjoint(with: entry.domains.keys)
    }

    private static func passRateDescription(_ value: Double?) -> String {
        guard let value else { return "unscored" }
        return String(format: "%.0f%%", value * 100)
    }

    private static let minimumReviewedPrunePromptCount = 50
    private static let minimumReviewedPruneMeanTokens: Double = 8

    private static func reviewedComparisonIssue(
        _ comparison: ExpertComparisonSummary,
        atlasPromptCount: Int
    ) -> String? {
        if comparison.promptCount < minimumReviewedPrunePromptCount {
            return "compare at least \(minimumReviewedPrunePromptCount) prompts before exporting a reviewed prune plan"
        }
        if comparison.promptCount != atlasPromptCount {
            return "same-suite comparison covers \(comparison.promptCount) of \(atlasPromptCount) traced prompts"
        }
        guard comparison.passRateBaseline != nil,
              comparison.passRateMasked != nil else {
            return "same-suite comparison is missing baseline/masked pass-rate evidence"
        }
        if let issue = reviewedComparisonValidatorIssue(comparison) {
            return issue
        }
        if !comparison.highRiskDomains.isEmpty {
            return "masked outputs regressed in high-risk domains: \(comparison.highRiskDomains.sorted().joined(separator: ", "))"
        }
        if isBlockingRegressionSeverity(comparison.regressionSeverity) {
            return "masked comparison regression severity is high or critical"
        }
        if comparison.safeDropCandidates.isEmpty {
            return "same-suite comparison found no safe drop candidates"
        }
        return nil
    }

    private static func reviewedComparisonValidatorIssue(_ comparison: ExpertComparisonSummary) -> String? {
        guard comparison.validatorAvailablePromptCount != nil,
              comparison.classificationCounts != nil else {
            return "same-suite comparison is missing validator classification evidence"
        }
        guard let baselineQualified = comparison.baselineQualifiedPromptCount,
              baselineQualified > 0 else {
            return "same-suite comparison has no baseline-qualified validator prompts"
        }
        let missingCoverage = comparison.missingBaselineQualifiedSemanticCoverage ?? []
        if !missingCoverage.isEmpty {
            return "baseline-qualified prompts are missing semantic coverage: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        let degraded = comparison.degradedPromptIDs ?? []
        if !degraded.isEmpty {
            return "baseline-qualified prompts degraded after masking: \(degraded.prefix(8).joined(separator: ", "))"
        }
        if let passRate = comparison.baselineQualifiedMaskedPassRate,
           passRate < 1.0 {
            return "masked validator pass rate is below 100% on baseline-qualified prompts"
        }
        return nil
    }

    private static func reviewedEvalIndexIssue(
        _ index: ExpertEvalIndexSummary,
        comparison: ExpertComparisonSummary,
        sourceModelPath: String?
    ) -> String? {
        if index.promptCount != comparison.promptCount {
            return "eval_index covers \(index.promptCount) prompts but comparison covers \(comparison.promptCount)"
        }
        if index.promptIDs.count != index.promptCount {
            return "eval_index prompt IDs cover \(index.promptIDs.count) of \(index.promptCount) prompts"
        }
        if Set(index.promptIDs).count != index.promptIDs.count {
            return "eval_index prompt IDs contain duplicates"
        }
        if !index.riskyPromptIDs.isEmpty {
            return "eval_index has risky prompt regressions: \(index.riskyPromptIDs.prefix(8).joined(separator: ", "))"
        }
        if !index.highRiskDomains.isEmpty {
            return "eval_index has high-risk domain regressions: \(index.highRiskDomains.sorted().joined(separator: ", "))"
        }
        if let issue = reviewedEvalIndexValidatorIssue(index) {
            return issue
        }
        guard let indexBaseline = index.passRateBaseline,
              let indexMasked = index.passRateMasked,
              let comparisonBaseline = comparison.passRateBaseline,
              let comparisonMasked = comparison.passRateMasked else {
            return "eval_index is missing baseline/masked pass-rate evidence"
        }
        if !doubleEqual(indexBaseline, comparisonBaseline) {
            return "eval_index baseline pass rate does not match comparison summary"
        }
        if !doubleEqual(indexMasked, comparisonMasked) {
            return "eval_index masked pass rate does not match comparison summary"
        }
        if !doubleEqual(index.meanTextDelta, comparison.meanTextDelta) {
            return "eval_index mean text delta does not match comparison summary"
        }
        guard let meanBaselineTokens = index.meanBaselineTokens,
              meanBaselineTokens >= minimumReviewedPruneMeanTokens,
              let meanMaskedTokens = index.meanMaskedTokens,
              meanMaskedTokens >= minimumReviewedPruneMeanTokens else {
            return "eval_index token depth is below \(Int(minimumReviewedPruneMeanTokens)) mean generated tokens"
        }
        guard let baselineRouteCount = index.baselineRouteRecordCount,
              baselineRouteCount >= index.promptCount else {
            return "eval_index is missing baseline routing records for every prompt"
        }
        guard let maskedRouteCount = index.maskedRouteRecordCount,
              maskedRouteCount >= index.promptCount else {
            return "eval_index is missing masked routing records for every prompt"
        }
        guard let baselineLayerStatsPromptCount = index.baselineLayerStatsPromptCount,
              baselineLayerStatsPromptCount >= index.promptCount,
              let maskedLayerStatsPromptCount = index.maskedLayerStatsPromptCount,
              maskedLayerStatsPromptCount >= index.promptCount else {
            return "eval_index layer-stat coverage is incomplete for indexed prompts"
        }
        if index.generationSettingsChecked != true {
            return "eval_index does not confirm matched generation settings"
        }
        if !hasText(index.suiteJSONL) {
            return "eval_index is missing suite_jsonl"
        }
        if !hasText(index.suiteSHA256) {
            return "eval_index is missing suite_jsonl fingerprint evidence"
        }
        if !hasText(index.evalJSONL) {
            return "eval_index is missing eval_jsonl"
        }
        if !hasText(index.evalTraceJSONL) {
            return "eval_index is missing eval_trace_jsonl"
        }
        if !hasText(index.comparisonSummary) {
            return "eval_index is missing comparison_summary artifact path"
        }
        if !hasText(index.mask) && !hasText(index.maskJSON) {
            return "eval_index is missing mask artifact path"
        }
        if index.runtimeMode != "bf16_vmlx" {
            return "eval_index runtime mode must be bf16_vmlx"
        }
        if index.runtimeBackend != "vmlx" {
            return "eval_index runtime backend must be vmlx"
        }
        if !hasText(index.runtimeDevice) {
            return "eval_index is missing runtime device"
        }
        if index.runtimeMetalEnabled != true {
            return "eval_index does not confirm Metal-backed runtime execution"
        }
        if let issue = reviewedSemanticCoverageIssue(index) {
            return issue
        }
        guard let hookedMOELayers = index.hookedMOELayers,
              hookedMOELayers > 0 else {
            return "eval_index is missing vMLX routed-layer hook evidence"
        }
        if index.hookCoverageComplete == false {
            return "eval_index recorded incomplete vMLX routed-layer hook coverage"
        }
        guard let expectedMOELayers = index.expectedMOELayers,
              expectedMOELayers > 0 else {
            return "eval_index is missing expected vMLX MoE layer evidence"
        }
        if hookedMOELayers < expectedMOELayers {
            return "eval_index vMLX hook coverage \(hookedMOELayers) of \(expectedMOELayers) config-routed layers"
        }
        if !hasText(index.jangToolsVersion) || !hasText(index.mlxVersion) || !hasText(index.mlxLMVersion) {
            return "eval_index is missing vMLX package version metadata"
        }
        guard let indexSourcePath = normalizedPath(index.sourceModelPath),
              !indexSourcePath.isEmpty else {
            return "eval_index is missing BF16 source model path"
        }
        if let sourceModelPath,
           let planSourcePath = normalizedPath(sourceModelPath),
           !planSourcePath.isEmpty,
           planSourcePath != indexSourcePath {
            return "eval_index source model path does not match prune source model path"
        }
        if index.maskApplied != true {
            return "eval_index does not confirm mask application"
        }
        guard let disabledExpertCount = index.disabledExpertCount,
              disabledExpertCount > 0 else {
            return "eval_index is missing disabled expert count"
        }
        if isBlockingRegressionSeverity(index.regressionSeverity) {
            return "eval_index regression severity is high or critical"
        }
        return nil
    }

    private static func reviewedEvalIndexValidatorIssue(_ index: ExpertEvalIndexSummary) -> String? {
        guard hasText(index.validatorSchema),
              index.validatorAvailablePromptCount != nil,
              index.promptClassificationCounts != nil else {
            return "eval_index is missing validator classification evidence"
        }
        guard let baselineQualified = index.baselineQualifiedPromptCount,
              baselineQualified > 0 else {
            return "eval_index has no baseline-qualified validator prompts"
        }
        guard let baselineQualifiedPromptIDs = index.baselineQualifiedPromptIDs,
              let baselineInvalidPromptIDs = index.baselineInvalidPromptIDs,
              let inconclusivePromptIDs = index.inconclusivePromptIDs,
              let preservedPromptIDs = index.preservedPromptIDs,
              let degradedPromptIDs = index.degradedPromptIDs else {
            return "eval_index is missing prompt classification ID lists"
        }
        if baselineQualifiedPromptIDs.count != baselineQualified {
            return "eval_index baseline-qualified prompt IDs do not match the baseline-qualified count"
        }
        let classified = (baselineInvalidPromptIDs.count
            + inconclusivePromptIDs.count
            + preservedPromptIDs.count
            + degradedPromptIDs.count)
        if classified != index.promptCount {
            return "eval_index prompt classifications cover \(classified) of \(index.promptCount) prompts"
        }
        if !degradedPromptIDs.isEmpty {
            return "eval_index has baseline-qualified prompt regressions: \(degradedPromptIDs.prefix(8).joined(separator: ", "))"
        }
        let missingCoverage = index.missingBaselineQualifiedSemanticCoverage ?? []
        if !missingCoverage.isEmpty {
            return "eval_index baseline-qualified semantic coverage is missing: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        guard let coverage = index.baselineQualifiedSemanticCoverage,
              !coverage.isEmpty else {
            return "eval_index is missing baseline-qualified semantic coverage evidence"
        }
        if let passRate = index.baselineQualifiedMaskedPassRate,
           passRate < 1.0 {
            return "eval_index masked validator pass rate is below 100% on baseline-qualified prompts"
        }
        return nil
    }

    private static func reviewedSemanticCoverageIssue(_ index: ExpertEvalIndexSummary) -> String? {
        guard let semanticCoverage = index.semanticCoverage,
              !semanticCoverage.isEmpty else {
            return "eval_index is missing semantic coverage evidence"
        }
        let coverage = Set(
            semanticCoverage
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let missingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(coverage)
            .sorted()
        if !missingCoverage.isEmpty {
            return "eval_index semantic coverage is missing required probes: \(missingCoverage.joined(separator: ", "))"
        }
        guard let recordedMissing = index.missingSemanticCoverage else {
            return "eval_index is missing missing-semantic-coverage evidence"
        }
        let missing = Set(
            recordedMissing
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        if !missing.isEmpty {
            return "eval_index records missing semantic prompt probes: \(missing.sorted().joined(separator: ", "))"
        }
        return nil
    }

    private static func doubleEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) <= 0.000_001
    }

    private static func hasText(_ value: String?) -> Bool {
        !(value?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
    }

    private static func normalizedPath(_ value: String?) -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: value).standardizedFileURL.path
    }

    private static func isBlockingRegressionSeverity(_ severity: String?) -> Bool {
        severity == "high" || severity == "critical"
    }
}

public enum ExpertPrunePlanError: Error, LocalizedError, Sendable {
    case invalidKeepCount(Int)
    case keepCountBelowTopK(keepExpertsPerLayer: Int, trainedTopK: Int, layer: Int)
    case keepCountTooHigh(keepExpertsPerLayer: Int, sourceExperts: Int, layer: Int)
    case forceDropOutOfRange(expert: Int, sourceExperts: Int, layer: Int)
    case forceDropLeavesTooFewExperts(keepExpertsPerLayer: Int, availableExperts: Int, layer: Int)
    case lockedKeepOutOfRange(expert: Int, sourceExperts: Int, layer: Int)
    case lockedKeepForcedDropConflict(expert: Int, layer: Int)
    case lockedKeepExceedsKeepCount(lockedKeep: Int, keepExpertsPerLayer: Int, layer: Int)
    case dropOutsideSameSuiteSafeDropCandidates(experts: [Int], layer: Int)
    case reviewedComparisonFailed(String)
    case reviewedEvalIndexFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidKeepCount(let count):
            return "keepExpertsPerLayer must be positive, got \(count)."
        case .keepCountBelowTopK(let keep, let trainedTopK, let layer):
            return "Layer \(layer) keeps \(keep) experts but the model was trained with top-k \(trainedTopK); keep at least top-k active experts."
        case .keepCountTooHigh(let keep, let source, let layer):
            return "Layer \(layer) has \(source) experts; keep count \(keep) must leave at least one expert to drop."
        case .forceDropOutOfRange(let expert, let sourceExperts, let layer):
            return "Layer \(layer) cannot drop expert \(expert); source has experts 0...\(max(sourceExperts - 1, 0))."
        case .forceDropLeavesTooFewExperts(let keep, let available, let layer):
            return "Layer \(layer) cannot keep \(keep) experts because only \(available) remain after user-disabled drops."
        case .lockedKeepOutOfRange(let expert, let sourceExperts, let layer):
            return "Layer \(layer) cannot lock expert \(expert); source has experts 0...\(max(sourceExperts - 1, 0))."
        case .lockedKeepForcedDropConflict(let expert, let layer):
            return "Layer \(layer) marks expert \(expert) as both locked-keep and forced-drop."
        case .lockedKeepExceedsKeepCount(let lockedKeep, let keepExpertsPerLayer, let layer):
            return "Layer \(layer) locks \(lockedKeep) experts but the plan only keeps \(keepExpertsPerLayer)."
        case .dropOutsideSameSuiteSafeDropCandidates(let experts, let layer):
            let preview = experts.prefix(8).map(String.init).joined(separator: ", ")
            let suffix = experts.count > 8 ? ", +\(experts.count - 8) more" : ""
            return "Layer \(layer) would drop experts outside the same-suite safe-drop set: \(preview)\(suffix). Rerun BF16/vMLX masked comparison with every planned drop disabled before exporting."
        case .reviewedComparisonFailed(let issue):
            return "Reviewed prune comparison failed: \(issue)."
        case .reviewedEvalIndexFailed(let issue):
            return "Reviewed prune eval_index failed: \(issue)."
        }
    }
}

public enum ExpertMaskValidationSeverity: String, Codable, Equatable, Sendable {
    case warning
    case error
}

public struct ExpertMaskValidationIssue: Codable, Equatable, Sendable, Identifiable {
    public var id: String { "\(severity.rawValue):\(layer ?? -1):\(expert ?? -1):\(message)" }
    public let severity: ExpertMaskValidationSeverity
    public let layer: Int?
    public let expert: Int?
    public let message: String

    public init(
        severity: ExpertMaskValidationSeverity,
        layer: Int? = nil,
        expert: Int? = nil,
        message: String
    ) {
        self.severity = severity
        self.layer = layer
        self.expert = expert
        self.message = message
    }
}

public enum ExpertMaskEngine {
    public static func validate(
        mask: JANGKit.ExpertMask,
        sourceNumExpertsByLayer: [Int: Int],
        trainedTopKByLayer: [Int: Int],
        hotExperts: Set<ExpertCoordinate> = [],
        maxDropFractionPerLayer: Float = 0.5
    ) -> [ExpertMaskValidationIssue] {
        var issues: [ExpertMaskValidationIssue] = []
        let layers = Set(sourceNumExpertsByLayer.keys)
            .union(mask.disabledExpertsByLayer.keys)
            .union(mask.lockedKeepByLayer.keys)

        for layer in layers.sorted() {
            let sourceCount = sourceNumExpertsByLayer[layer] ?? 0
            guard sourceCount > 0 else {
                issues.append(ExpertMaskValidationIssue(
                    severity: .error,
                    layer: layer,
                    message: "Layer \(layer) has no known source expert count."
                ))
                continue
            }

            let disabled = mask.disabledExperts(for: layer)
            let locked = mask.lockedKeepExperts(for: layer)
            for expert in disabled.union(locked).sorted() where expert < 0 || expert >= sourceCount {
                issues.append(ExpertMaskValidationIssue(
                    severity: .error,
                    layer: layer,
                    expert: expert,
                    message: "Expert \(expert) is outside layer \(layer)'s range 0...\(sourceCount - 1)."
                ))
            }

            for expert in disabled.intersection(locked).sorted() {
                issues.append(ExpertMaskValidationIssue(
                    severity: .error,
                    layer: layer,
                    expert: expert,
                    message: "Expert \(expert) is both disabled and locked keep."
                ))
            }

            let trainedTopK = trainedTopKByLayer[layer] ?? trainedTopKByLayer.values.first ?? 1
            let effectiveTopK = mask.topKOverride.map { min($0, trainedTopK) } ?? trainedTopK
            let available = sourceCount - disabled.count
            if available < effectiveTopK {
                issues.append(ExpertMaskValidationIssue(
                    severity: .error,
                    layer: layer,
                    message: "Layer \(layer) leaves \(available) available experts, fewer than top-k \(effectiveTopK)."
                ))
            }

            if Float(disabled.count) / Float(sourceCount) > maxDropFractionPerLayer {
                issues.append(ExpertMaskValidationIssue(
                    severity: .warning,
                    layer: layer,
                    message: "Layer \(layer) disables \(disabled.count) of \(sourceCount) experts."
                ))
            }

            for expert in disabled.sorted() where hotExperts.contains(ExpertCoordinate(layer: layer, expert: expert)) {
                issues.append(ExpertMaskValidationIssue(
                    severity: .warning,
                    layer: layer,
                    expert: expert,
                    message: "Expert \(expert) is hot in the baseline atlas."
                ))
            }
        }

        if let topK = mask.topKOverride, topK <= 0 {
            issues.append(ExpertMaskValidationIssue(
                severity: .error,
                message: "Top-k override must be positive when set."
            ))
        }

        return issues
    }
}

public struct ExpertCoordinate: Codable, Hashable, Sendable {
    public let layer: Int
    public let expert: Int

    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }
}

public enum ExpertAtlasSelection {
    public static func coordinates(
        intersecting rect: CGRect,
        cellFrames: [ExpertCoordinate: CGRect]
    ) -> Set<ExpertCoordinate> {
        guard rect.width > 0, rect.height > 0 else { return [] }
        return Set(cellFrames.compactMap { coordinate, frame in
            rect.intersects(frame) ? coordinate : nil
        })
    }
}

public struct ExpertComparisonSummary: Codable, Equatable, Sendable {
    public let baselineRunID: String
    public let maskID: String
    public let promptCount: Int
    public let passRateBaseline: Double?
    public let passRateMasked: Double?
    public let baselineQualifiedPromptCount: Int?
    public let baselineQualifiedMaskedPassRate: Double?
    public let validatorAvailablePromptCount: Int?
    public let classificationCounts: [String: Int]?
    public let baselineQualifiedPromptIDs: [String]?
    public let baselineInvalidPromptIDs: [String]?
    public let inconclusivePromptIDs: [String]?
    public let preservedPromptIDs: [String]?
    public let degradedPromptIDs: [String]?
    public let baselineQualifiedSemanticCoverage: [String]?
    public let missingBaselineQualifiedSemanticCoverage: [String]?
    public let meanTextDelta: Double
    public let meanLatencyDeltaPct: Double
    public let regressionSeverity: String?
    public let highRiskDomains: [String]
    public let safeDropCandidates: [ExpertCoordinate]

    public init(
        baselineRunID: String,
        maskID: String,
        promptCount: Int,
        passRateBaseline: Double? = nil,
        passRateMasked: Double? = nil,
        baselineQualifiedPromptCount: Int? = nil,
        baselineQualifiedMaskedPassRate: Double? = nil,
        validatorAvailablePromptCount: Int? = nil,
        classificationCounts: [String: Int]? = nil,
        baselineQualifiedPromptIDs: [String]? = nil,
        baselineInvalidPromptIDs: [String]? = nil,
        inconclusivePromptIDs: [String]? = nil,
        preservedPromptIDs: [String]? = nil,
        degradedPromptIDs: [String]? = nil,
        baselineQualifiedSemanticCoverage: [String]? = nil,
        missingBaselineQualifiedSemanticCoverage: [String]? = nil,
        meanTextDelta: Double,
        meanLatencyDeltaPct: Double,
        regressionSeverity: String? = nil,
        highRiskDomains: [String] = [],
        safeDropCandidates: [ExpertCoordinate] = []
    ) {
        self.baselineRunID = baselineRunID
        self.maskID = maskID
        self.promptCount = promptCount
        self.passRateBaseline = passRateBaseline
        self.passRateMasked = passRateMasked
        self.baselineQualifiedPromptCount = baselineQualifiedPromptCount
        self.baselineQualifiedMaskedPassRate = baselineQualifiedMaskedPassRate
        self.validatorAvailablePromptCount = validatorAvailablePromptCount
        self.classificationCounts = classificationCounts
        self.baselineQualifiedPromptIDs = baselineQualifiedPromptIDs
        self.baselineInvalidPromptIDs = baselineInvalidPromptIDs
        self.inconclusivePromptIDs = inconclusivePromptIDs
        self.preservedPromptIDs = preservedPromptIDs
        self.degradedPromptIDs = degradedPromptIDs
        self.baselineQualifiedSemanticCoverage = baselineQualifiedSemanticCoverage
        self.missingBaselineQualifiedSemanticCoverage = missingBaselineQualifiedSemanticCoverage
        self.meanTextDelta = meanTextDelta
        self.meanLatencyDeltaPct = meanLatencyDeltaPct
        self.regressionSeverity = regressionSeverity
        self.highRiskDomains = highRiskDomains
        self.safeDropCandidates = safeDropCandidates
    }

    enum CodingKeys: String, CodingKey {
        case baselineRunID
        case maskID
        case promptCount
        case passRateBaseline
        case passRateMasked
        case baselineQualifiedPromptCount
        case baselineQualifiedMaskedPassRate
        case validatorAvailablePromptCount
        case classificationCounts
        case baselineQualifiedPromptIDs
        case baselineInvalidPromptIDs
        case inconclusivePromptIDs
        case preservedPromptIDs
        case degradedPromptIDs
        case baselineQualifiedSemanticCoverage
        case missingBaselineQualifiedSemanticCoverage
        case meanTextDelta
        case meanLatencyDeltaPct
        case regressionSeverity = "regression_severity"
        case highRiskDomains
        case safeDropCandidates
    }
}

public struct ExpertRunManifest: Codable, Equatable, Sendable {
    public let runID: String
    public let sourcePath: String
    public let reviewBundlePath: String?
    public let appVersion: String
    public let toolsVersion: String
    public let runtimeMode: String
    public let runtimeBackend: String?
    public let runtimeDevice: String?
    public let runtimeMetalEnabled: Bool?
    public let jangToolsVersion: String?
    public let mlxVersion: String?
    public let mlxLMVersion: String?
    public let mlxVLMVersion: String?
    public let sourceModelPath: String?
    public let hookedMOELayers: Int?
    public let expectedMOELayers: Int?
    public let hookCoverageComplete: Bool?
    public let suiteID: String
    public let promptCount: Int
    public let emitTokenTrace: Bool
    public let maxTraceTokens: Int
    public let startedAt: Date
    public let endedAt: Date?
    public let failureStage: String?
    public let failureMessage: String?
}

public struct ExpertRunSummary: Codable, Equatable, Sendable, Identifiable {
    public var id: String { runID }
    public let runID: String
    public let directoryPath: String
    public let sourcePath: String
    public let reviewBundlePath: String?
    public let runtimeMode: String
    public let runtimeBackend: String?
    public let runtimeDevice: String?
    public let runtimeMetalEnabled: Bool?
    public let jangToolsVersion: String?
    public let mlxVersion: String?
    public let mlxLMVersion: String?
    public let mlxVLMVersion: String?
    public let sourceModelPath: String?
    public let hookedMOELayers: Int?
    public let expectedMOELayers: Int?
    public let hookCoverageComplete: Bool?
    public let suiteID: String
    public let promptCount: Int
    public let startedAt: Date
    public let endedAt: Date?
    public let failureStage: String?
    public let failureMessage: String?

    public init(runID: String, directoryPath: String, manifest: ExpertRunManifest) {
        self.runID = runID
        self.directoryPath = directoryPath
        self.sourcePath = manifest.sourcePath
        self.reviewBundlePath = manifest.reviewBundlePath
        self.runtimeMode = manifest.runtimeMode
        self.runtimeBackend = manifest.runtimeBackend
        self.runtimeDevice = manifest.runtimeDevice
        self.runtimeMetalEnabled = manifest.runtimeMetalEnabled
        self.jangToolsVersion = manifest.jangToolsVersion
        self.mlxVersion = manifest.mlxVersion
        self.mlxLMVersion = manifest.mlxLMVersion
        self.mlxVLMVersion = manifest.mlxVLMVersion
        self.sourceModelPath = manifest.sourceModelPath
        self.hookedMOELayers = manifest.hookedMOELayers
        self.expectedMOELayers = manifest.expectedMOELayers
        self.hookCoverageComplete = manifest.hookCoverageComplete
        self.suiteID = manifest.suiteID
        self.promptCount = manifest.promptCount
        self.startedAt = manifest.startedAt
        self.endedAt = manifest.endedAt
        self.failureStage = manifest.failureStage
        self.failureMessage = manifest.failureMessage
    }
}

public struct ExpertLabMaskArtifact: Codable, Equatable, Sendable {
    public let version: Int
    public let savedAt: Date
    public let disabledByLayer: [Int: Set<Int>]
    public let dropCandidatesByLayer: [Int: Set<Int>]
    public let lockedKeepByLayer: [Int: Set<Int>]
    public let topKOverride: Int?

    public init(
        version: Int = 1,
        savedAt: Date = Date(),
        disabledByLayer: [Int: Set<Int>] = [:],
        dropCandidatesByLayer: [Int: Set<Int>] = [:],
        lockedKeepByLayer: [Int: Set<Int>] = [:],
        topKOverride: Int? = nil
    ) {
        self.version = version
        self.savedAt = savedAt
        self.disabledByLayer = disabledByLayer
        self.dropCandidatesByLayer = dropCandidatesByLayer
        self.lockedKeepByLayer = lockedKeepByLayer
        self.topKOverride = topKOverride
    }

    enum CodingKeys: String, CodingKey {
        case version
        case savedAt
        case disabledByLayer = "disabled_by_layer"
        case dropCandidatesByLayer = "drop_candidates_by_layer"
        case lockedKeepByLayer = "locked_keep_by_layer"
        case topKOverride = "top_k_override"
    }
}

public enum ExpertRunStore {
    public static func listRuns(
        rootDirectory: URL,
        matchingSourcePath sourcePath: String? = nil,
        reviewBundlePath: String? = nil
    ) throws -> [ExpertRunSummary] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: rootDirectory.path) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let urls = try fm.contentsOfDirectory(
            at: rootDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        var summaries: [ExpertRunSummary] = []
        for url in urls {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
            guard values?.isDirectory == true else { continue }
            let manifestURL = url.appendingPathComponent("run.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? decoder.decode(ExpertRunManifest.self, from: data) else {
                continue
            }
            if let sourcePath, !pathsMatch(manifest.sourcePath, sourcePath) { continue }
            if let reviewBundlePath, !pathsMatch(manifest.reviewBundlePath, reviewBundlePath) { continue }
            summaries.append(ExpertRunSummary(
                runID: manifest.runID,
                directoryPath: url.path,
                manifest: manifest
            ))
        }
        return summaries.sorted {
            if $0.startedAt != $1.startedAt { return $0.startedAt > $1.startedAt }
            return $0.runID > $1.runID
        }
    }

    private static func pathsMatch(_ lhs: String?, _ rhs: String) -> Bool {
        guard let lhs else { return false }
        return normalizedPath(lhs) == normalizedPath(rhs)
    }

    private static func normalizedPath(_ path: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
            .standardizedFileURL
            .resolvingSymlinksInPath()
            .path
    }
}

private enum SQLiteTraceError: Error, LocalizedError {
    case open(String)
    case exec(String)
    case prepare(String)
    case step(String)
    case bind(String)

    var errorDescription: String? {
        switch self {
        case .open(let message):
            return "could not open trace.sqlite: \(message)"
        case .exec(let message):
            return "could not execute trace.sqlite statement: \(message)"
        case .prepare(let message):
            return "could not prepare trace.sqlite statement: \(message)"
        case .step(let message):
            return "could not write trace.sqlite row: \(message)"
        case .bind(let message):
            return "could not bind trace.sqlite value: \(message)"
        }
    }
}

public enum ExpertArtifactWriter {
    public static func writeRun(
        rootDirectory: URL,
        runID: String = "run_\(UUID().uuidString)",
        sourcePath: String,
        reviewBundlePath: String?,
        appVersion: String = "JANGStudio",
        toolsVersion: String = "unknown",
        runtimeMode: String,
        runtimeBackend: String? = nil,
        runtimeDevice: String? = nil,
        runtimeMetalEnabled: Bool? = nil,
        jangToolsVersion: String? = nil,
        mlxVersion: String? = nil,
        mlxLMVersion: String? = nil,
        mlxVLMVersion: String? = nil,
        suite: ExpertPromptSuite,
        traceConfig: JANGKit.ExpertTraceConfig,
        runs: [ExpertPromptRun],
        atlas: ExpertAtlas,
        logs: String = "",
        failureStage: String? = nil,
        failureMessage: String? = nil
    ) throws -> URL {
        let fm = FileManager.default
        let dir = rootDirectory.appendingPathComponent(runID, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let firstRuntimeInfo = runs.compactMap(\.result.runtimeInfo).first

        let manifest = ExpertRunManifest(
            runID: runID,
            sourcePath: sourcePath,
            reviewBundlePath: reviewBundlePath,
            appVersion: appVersion,
            toolsVersion: toolsVersion,
            runtimeMode: runtimeMode,
            runtimeBackend: runtimeBackend,
            runtimeDevice: runtimeDevice,
            runtimeMetalEnabled: runtimeMetalEnabled,
            jangToolsVersion: jangToolsVersion ?? firstRuntimeInfo?.jangToolsVersion,
            mlxVersion: mlxVersion ?? firstRuntimeInfo?.mlxVersion,
            mlxLMVersion: mlxLMVersion ?? firstRuntimeInfo?.mlxLMVersion,
            mlxVLMVersion: mlxVLMVersion ?? firstRuntimeInfo?.mlxVLMVersion,
            sourceModelPath: firstRuntimeInfo?.sourceModelPath,
            hookedMOELayers: firstRuntimeInfo?.hookedMOELayers,
            expectedMOELayers: firstRuntimeInfo?.expectedMOELayers,
            hookCoverageComplete: firstRuntimeInfo?.hookCoverageComplete,
            suiteID: suite.name,
            promptCount: runs.count,
            emitTokenTrace: traceConfig.emitTokenTrace,
            maxTraceTokens: traceConfig.maxTraceTokens,
            startedAt: Date(),
            endedAt: Date(),
            failureStage: failureStage,
            failureMessage: failureMessage
        )
        try encoder.encode(manifest).write(to: dir.appendingPathComponent("run.json"))
        try suite.writeJSONL(to: dir.appendingPathComponent("suite.jsonl"))
        try encoder.encode(atlas).write(to: dir.appendingPathComponent("atlas.json"))
        try logs.write(to: dir.appendingPathComponent("logs.txt"), atomically: true, encoding: .utf8)
        try writeGenerations(runs, to: dir.appendingPathComponent("generations.jsonl"))
        try writeTrace(runs, to: dir.appendingPathComponent("trace.jsonl"))
        try writeTraceSQLite(runs, to: dir.appendingPathComponent("trace.sqlite"))
        try encoder.encode(modelFingerprint(path: sourcePath)).write(
            to: dir.appendingPathComponent("model_fingerprint.json")
        )
        return dir
    }

    private static func writeGenerations(_ runs: [ExpertPromptRun], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let lines = try runs.map { run in
            let domain = ExpertDomainTaxonomy.canonicalDomain(for: run.prompt)
            let record = StoredGenerationRecord(
                promptID: run.prompt.id,
                domain: domain,
                text: run.result.text,
                tokenCount: run.result.tokens,
                tokensPerSecond: run.result.tokensPerSecond,
                finishReason: run.result.finishReason.rawValue,
                layerStats: run.result.layerStats,
                runtimeMode: run.result.runtimeInfo?.runtimeMode,
                runtimeBackend: run.result.runtimeInfo?.backend,
                runtimeDevice: run.result.runtimeInfo?.deviceName,
                runtimeMetalEnabled: run.result.runtimeInfo?.metalEnabled,
                jangToolsVersion: run.result.runtimeInfo?.jangToolsVersion,
                mlxVersion: run.result.runtimeInfo?.mlxVersion,
                mlxLMVersion: run.result.runtimeInfo?.mlxLMVersion,
                mlxVLMVersion: run.result.runtimeInfo?.mlxVLMVersion,
                sourceModelPath: run.result.runtimeInfo?.sourceModelPath,
                hookedMOELayers: run.result.runtimeInfo?.hookedMOELayers,
                expectedMOELayers: run.result.runtimeInfo?.expectedMOELayers,
                hookCoverageComplete: run.result.runtimeInfo?.hookCoverageComplete,
                maskApplied: run.result.runtimeInfo?.maskApplied,
                disabledExpertCount: run.result.runtimeInfo?.disabledExpertCount,
                topKOverride: run.result.runtimeInfo?.topKOverride
            )
            return String(data: try encoder.encode(record), encoding: .utf8) ?? "{}"
        }
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private static func writeTrace(_ runs: [ExpertPromptRun], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var lines: [String] = []
        for run in runs {
            let domain = ExpertDomainTaxonomy.canonicalDomain(for: run.prompt)
            for record in run.result.tokenTrace ?? [] {
                let stored = StoredTraceRecord(promptID: run.prompt.id, domain: domain, record: record)
                lines.append(String(data: try encoder.encode(stored), encoding: .utf8) ?? "{}")
            }
        }
        try lines.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private static func writeTraceSQLite(_ runs: [ExpertPromptRun], to url: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            try fm.removeItem(at: url)
        }

        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, nil) == SQLITE_OK else {
            let message = sqliteMessage(db)
            if db != nil { sqlite3_close(db) }
            throw SQLiteTraceError.open(message)
        }
        defer { sqlite3_close(db) }

        try sqliteExec(db, "PRAGMA journal_mode = OFF;")
        try sqliteExec(db, "PRAGMA synchronous = OFF;")
        try sqliteExec(db, """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE prompts (
            prompt_id TEXT PRIMARY KEY,
            domain TEXT NOT NULL,
            subdomain TEXT,
            expected_kind TEXT NOT NULL,
            expected TEXT,
            prompt TEXT NOT NULL,
            max_new_tokens INTEGER,
            temperature REAL,
            weight REAL NOT NULL
        );
        CREATE TABLE route_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            token_index INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            selected_experts TEXT NOT NULL,
            disabled_experts TEXT NOT NULL,
            scores TEXT NOT NULL,
            effective_top_k INTEGER NOT NULL,
            entropy REAL
        );
        CREATE TABLE expert_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            token_index INTEGER NOT NULL,
            layer INTEGER NOT NULL,
            expert INTEGER NOT NULL,
            slot INTEGER,
            score REAL,
            is_disabled INTEGER NOT NULL
        );
        CREATE INDEX idx_route_records_layer ON route_records(layer);
        CREATE INDEX idx_route_records_prompt ON route_records(prompt_id);
        CREATE INDEX idx_expert_events_layer_expert ON expert_events(layer, expert);
        CREATE INDEX idx_expert_events_prompt ON expert_events(prompt_id);
        """)

        try sqliteExec(db, "BEGIN IMMEDIATE;")
        do {
            try insertMetadata(db, key: "schema", value: "jang-expert-trace-sqlite-v1")
            try insertMetadata(db, key: "prompt_count", value: "\(runs.count)")

            var promptStmt: OpaquePointer?
            var routeStmt: OpaquePointer?
            var eventStmt: OpaquePointer?
            try sqlitePrepare(db, """
            INSERT OR REPLACE INTO prompts (
                prompt_id, domain, subdomain, expected_kind, expected, prompt,
                max_new_tokens, temperature, weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, &promptStmt)
            try sqlitePrepare(db, """
            INSERT INTO route_records (
                prompt_id, domain, token_index, layer, selected_experts,
                disabled_experts, scores, effective_top_k, entropy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, &routeStmt)
            try sqlitePrepare(db, """
            INSERT INTO expert_events (
                prompt_id, domain, token_index, layer, expert, slot, score, is_disabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, &eventStmt)
            defer {
                sqlite3_finalize(promptStmt)
                sqlite3_finalize(routeStmt)
                sqlite3_finalize(eventStmt)
            }

            for run in runs {
                try insertPrompt(run.prompt, statement: promptStmt)
                for record in run.result.tokenTrace ?? [] {
                    try insertRouteRecord(record, prompt: run.prompt, statement: routeStmt)
                    try insertExpertEvents(record, prompt: run.prompt, statement: eventStmt)
                }
            }

            try sqliteExec(db, "COMMIT;")
        } catch {
            try? sqliteExec(db, "ROLLBACK;")
            throw error
        }
    }

    private static func insertMetadata(_ db: OpaquePointer?, key: String, value: String) throws {
        var statement: OpaquePointer?
        try sqlitePrepare(db, "INSERT INTO metadata (key, value) VALUES (?, ?);", &statement)
        defer { sqlite3_finalize(statement) }
        try sqliteBindText(statement, index: 1, value: key)
        try sqliteBindText(statement, index: 2, value: value)
        try sqliteStepDone(statement, db: db)
    }

    private static func insertPrompt(_ prompt: ExpertPrompt, statement: OpaquePointer?) throws {
        sqlite3_reset(statement)
        sqlite3_clear_bindings(statement)
        let domain = ExpertDomainTaxonomy.canonicalDomain(for: prompt)
        try sqliteBindText(statement, index: 1, value: prompt.id)
        try sqliteBindText(statement, index: 2, value: domain)
        try sqliteBindText(statement, index: 3, value: prompt.subdomain)
        try sqliteBindText(statement, index: 4, value: prompt.expectedKind.rawValue)
        try sqliteBindText(statement, index: 5, value: prompt.expected)
        try sqliteBindText(statement, index: 6, value: prompt.text)
        try sqliteBindInt(statement, index: 7, value: prompt.maxNewTokens)
        try sqliteBindDouble(statement, index: 8, value: prompt.temperature)
        try sqliteBindDouble(statement, index: 9, value: prompt.weight)
        try sqliteStepDone(statement, db: nil)
    }

    private static func insertRouteRecord(
        _ record: JANGKit.ExpertRouteRecord,
        prompt: ExpertPrompt,
        statement: OpaquePointer?
    ) throws {
        sqlite3_reset(statement)
        sqlite3_clear_bindings(statement)
        let domain = ExpertDomainTaxonomy.canonicalDomain(for: prompt)
        try sqliteBindText(statement, index: 1, value: prompt.id)
        try sqliteBindText(statement, index: 2, value: domain)
        try sqliteBindInt(statement, index: 3, value: record.tokenIndex)
        try sqliteBindInt(statement, index: 4, value: record.layer)
        try sqliteBindText(statement, index: 5, value: jsonString(record.selectedExperts))
        try sqliteBindText(statement, index: 6, value: jsonString(record.disabledExperts))
        try sqliteBindText(statement, index: 7, value: jsonString(record.scores))
        try sqliteBindInt(statement, index: 8, value: record.effectiveTopK)
        try sqliteBindDouble(statement, index: 9, value: record.entropy.map(Double.init))
        try sqliteStepDone(statement, db: nil)
    }

    private static func insertExpertEvents(
        _ record: JANGKit.ExpertRouteRecord,
        prompt: ExpertPrompt,
        statement: OpaquePointer?
    ) throws {
        let domain = ExpertDomainTaxonomy.canonicalDomain(for: prompt)
        for (slot, expert) in record.selectedExperts.enumerated() {
            sqlite3_reset(statement)
            sqlite3_clear_bindings(statement)
            try sqliteBindText(statement, index: 1, value: prompt.id)
            try sqliteBindText(statement, index: 2, value: domain)
            try sqliteBindInt(statement, index: 3, value: record.tokenIndex)
            try sqliteBindInt(statement, index: 4, value: record.layer)
            try sqliteBindInt(statement, index: 5, value: expert)
            try sqliteBindInt(statement, index: 6, value: slot)
            let score = slot < record.scores.count ? Double(record.scores[slot]) : nil
            try sqliteBindDouble(statement, index: 7, value: score)
            try sqliteBindInt(statement, index: 8, value: 0)
            try sqliteStepDone(statement, db: nil)
        }
        for expert in record.disabledExperts {
            sqlite3_reset(statement)
            sqlite3_clear_bindings(statement)
            try sqliteBindText(statement, index: 1, value: prompt.id)
            try sqliteBindText(statement, index: 2, value: domain)
            try sqliteBindInt(statement, index: 3, value: record.tokenIndex)
            try sqliteBindInt(statement, index: 4, value: record.layer)
            try sqliteBindInt(statement, index: 5, value: expert)
            try sqliteBindNull(statement, index: 6)
            try sqliteBindNull(statement, index: 7)
            try sqliteBindInt(statement, index: 8, value: 1)
            try sqliteStepDone(statement, db: nil)
        }
    }

    private static func sqlitePrepare(
        _ db: OpaquePointer?,
        _ sql: String,
        _ statement: inout OpaquePointer?
    ) throws {
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw SQLiteTraceError.prepare(sqliteMessage(db))
        }
    }

    private static func sqliteExec(_ db: OpaquePointer?, _ sql: String) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else {
            throw SQLiteTraceError.exec(sqliteMessage(db))
        }
    }

    private static func sqliteStepDone(_ statement: OpaquePointer?, db: OpaquePointer?) throws {
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw SQLiteTraceError.step(sqliteMessage(db))
        }
    }

    private static func sqliteBindText(_ statement: OpaquePointer?, index: Int32, value: String?) throws {
        guard let value else {
            try sqliteBindNull(statement, index: index)
            return
        }
        guard sqlite3_bind_text(statement, index, value, -1, sqliteTransient) == SQLITE_OK else {
            throw SQLiteTraceError.bind("text \(index)")
        }
    }

    private static func sqliteBindInt(_ statement: OpaquePointer?, index: Int32, value: Int?) throws {
        guard let value else {
            try sqliteBindNull(statement, index: index)
            return
        }
        guard sqlite3_bind_int64(statement, index, sqlite3_int64(value)) == SQLITE_OK else {
            throw SQLiteTraceError.bind("int \(index)")
        }
    }

    private static func sqliteBindDouble(_ statement: OpaquePointer?, index: Int32, value: Double?) throws {
        guard let value else {
            try sqliteBindNull(statement, index: index)
            return
        }
        guard sqlite3_bind_double(statement, index, value) == SQLITE_OK else {
            throw SQLiteTraceError.bind("double \(index)")
        }
    }

    private static func sqliteBindNull(_ statement: OpaquePointer?, index: Int32) throws {
        guard sqlite3_bind_null(statement, index) == SQLITE_OK else {
            throw SQLiteTraceError.bind("null \(index)")
        }
    }

    private static func sqliteMessage(_ db: OpaquePointer?) -> String {
        guard let message = sqlite3_errmsg(db) else { return "unknown SQLite error" }
        return String(cString: message)
    }

    private static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        return String(data: data, encoding: .utf8) ?? "null"
    }

    private static let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    private static func modelFingerprint(path: String) -> [String: String] {
        let url = URL(fileURLWithPath: path)
        let configBytes = fileSize(url.appendingPathComponent("config.json"))
        let indexBytes = fileSize(url.appendingPathComponent("model.safetensors.index.json"))
        return [
            "path": path,
            "config_bytes": configBytes.map(String.init) ?? "missing",
            "index_bytes": indexBytes.map(String.init) ?? "missing",
        ]
    }

    private static func fileSize(_ url: URL) -> Int64? {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
              let size = values.fileSize else {
            return nil
        }
        return Int64(size)
    }

    private struct StoredGenerationRecord: Codable {
        let promptID: String
        let domain: String
        let text: String
        let tokenCount: Int
        let tokensPerSecond: Double
        let finishReason: String
        let layerStats: [JANGKit.ExpertLayerStats]
        let runtimeMode: String?
        let runtimeBackend: String?
        let runtimeDevice: String?
        let runtimeMetalEnabled: Bool?
        let jangToolsVersion: String?
        let mlxVersion: String?
        let mlxLMVersion: String?
        let mlxVLMVersion: String?
        let sourceModelPath: String?
        let hookedMOELayers: Int?
        let expectedMOELayers: Int?
        let hookCoverageComplete: Bool?
        let maskApplied: Bool?
        let disabledExpertCount: Int?
        let topKOverride: Int?
    }

    private struct StoredTraceRecord: Codable {
        let promptID: String
        let domain: String
        let record: JANGKit.ExpertRouteRecord
    }
}

public enum ExpertAtlasBuilder {
    public static func build(
        from runs: [ExpertPromptRun],
        expectedExpertsByLayer: [Int: Int] = [:]
    ) -> ExpertAtlas {
        var hits: [LayerExpert: Int] = [:]
        var mass: [LayerExpert: Float] = [:]
        var rankSum: [LayerExpert: Float] = [:]
        var entropySum: [LayerExpert: Float] = [:]
        var tokenSets: [Int: Set<String>] = [:]
        var domainTokenSets: [Int: [String: Set<String>]] = [:]
        var expertTokenSets: [LayerExpert: Set<String>] = [:]
        var tokenCounts: [Int: Int] = [:]
        var domains: [LayerExpert: [String: Int]] = [:]
        var promptHits: [LayerExpert: [String: Int]] = [:]
        var promptDetails: [String: ExpertPrompt] = [:]
        var pairCounts: [LayerPair: Int] = [:]
        var tokenIndexSum: [LayerExpert: Int] = [:]
        var minTokenIndex: [LayerExpert: Int] = [:]
        var maxTokenIndex: [LayerExpert: Int] = [:]

        for run in runs {
            promptDetails[run.prompt.id] = run.prompt
            let promptDomains = ExpertDomainTaxonomy.semanticDomains(for: run.prompt)
            if let trace = run.result.tokenTrace {
                for record in trace {
                    let tokenKey = "\(run.prompt.id)#\(record.tokenIndex)"
                    tokenSets[record.layer, default: []].insert(tokenKey)
                    for promptDomain in promptDomains {
                        domainTokenSets[record.layer, default: [:]][promptDomain, default: []].insert(tokenKey)
                    }
                    let entropyShare = (record.entropy ?? 0) / Float(max(record.selectedExperts.count, 1))
                    for (slot, expert) in record.selectedExperts.enumerated() {
                        let key = LayerExpert(layer: record.layer, expert: expert)
                        hits[key, default: 0] += 1
                        let score = slot < record.scores.count ? record.scores[slot] : 1.0
                        mass[key, default: 0] += score
                        rankSum[key, default: 0] += Float(slot + 1)
                        entropySum[key, default: 0] += entropyShare
                        for promptDomain in promptDomains {
                            domains[key, default: [:]][promptDomain, default: 0] += 1
                        }
                        promptHits[key, default: [:]][run.prompt.id, default: 0] += 1
                        expertTokenSets[key, default: []].insert(tokenKey)
                        tokenIndexSum[key, default: 0] += record.tokenIndex
                        minTokenIndex[key] = min(minTokenIndex[key] ?? record.tokenIndex, record.tokenIndex)
                        maxTokenIndex[key] = max(maxTokenIndex[key] ?? record.tokenIndex, record.tokenIndex)
                    }
                    let experts = Array(Set(record.selectedExperts)).sorted()
                    if experts.count > 1 {
                        for i in 0..<(experts.count - 1) {
                            for j in (i + 1)..<experts.count {
                                pairCounts[
                                    LayerPair(layer: record.layer, a: experts[i], b: experts[j]),
                                    default: 0
                                ] += 1
                            }
                        }
                    }
                }
            } else {
                for stat in run.result.layerStats {
                    tokenCounts[stat.layer, default: 0] += stat.tokenCount
                    let layerTokenCount = max(stat.tokenCount, 1)
                    for promptDomain in promptDomains {
                        domainTokenSets[stat.layer, default: [:]][promptDomain, default: []].formUnion(
                            (0..<layerTokenCount).map { "\(run.prompt.id)#\($0)" }
                        )
                    }
                    for (expert, count) in stat.hitCounts {
                        let key = LayerExpert(layer: stat.layer, expert: expert)
                        hits[key, default: 0] += count
                        mass[key, default: 0] += stat.probabilityMass[expert] ?? 0
                        rankSum[key, default: 0] += Float(count)
                        for promptDomain in promptDomains {
                            domains[key, default: [:]][promptDomain, default: 0] += count
                        }
                        promptHits[key, default: [:]][run.prompt.id, default: 0] += count
                    }
                }
            }
        }

        var allKeys = Set(hits.keys)
        for (layer, nExperts) in expectedExpertsByLayer {
            guard nExperts > 0 else { continue }
            for expert in 0..<nExperts {
                allKeys.insert(LayerExpert(layer: layer, expert: expert))
            }
        }

        let maxHitsByLayer = Dictionary(grouping: hits.keys, by: { $0.layer })
            .mapValues { keys in keys.map { hits[$0] ?? 0 }.max() ?? 0 }

        let entries = allKeys.sorted().map { key in
            let h = hits[key] ?? 0
            let layerMax = maxHitsByLayer[key.layer] ?? h
            let domainCounts = domains[key] ?? [:]
            let layerTokenCount = tokenSets[key.layer]?.count ?? tokenCounts[key.layer] ?? 0
            let frequency = finiteFloat(layerTokenCount > 0 ? Float(h) / Float(layerTokenCount) : 0)
            let lift = finiteMap(domainLift(
                layer: key.layer,
                hits: h,
                domainCounts: domainCounts,
                layerTokenCount: layerTokenCount,
                domainTokenSets: domainTokenSets
            ))
            let coactivation = finiteNeighbors(neighbors(
                for: key,
                expertTokenSets: expertTokenSets,
                pairCounts: pairCounts,
                layerTokenCount: layerTokenCount
            ))
            let probabilityMass = finiteFloat(mass[key] ?? 0)
            let meanSelectedRank = finiteFloat(h > 0 ? (rankSum[key] ?? 0) / Float(h) : 0)
            let entropyContribution = finiteFloat(entropySum[key] ?? 0)
            let confidence = finiteFloat(confidenceScore(hits: h, layerTokenCount: layerTokenCount, domainLift: lift))
            let meanTokenIndex = tokenIndexSum[key].map { finiteFloat(Float($0) / Float(max(h, 1))) }
            let promptCounts = promptHits[key] ?? [:]
            let label = labelFor(
                hits: h,
                layerMax: layerMax,
                domains: domainCounts,
                domainLift: lift
            )
            return ExpertAtlasEntry(
                layer: key.layer,
                expert: key.expert,
                hits: h,
                activationFrequency: frequency,
                probabilityMass: probabilityMass,
                tokenCount: layerTokenCount,
                domains: domainCounts,
                domainLift: lift,
                meanSelectedRank: meanSelectedRank,
                entropyContribution: entropyContribution,
                coactivationNeighbors: coactivation,
                topPrompts: topPrompts(promptCounts),
                promptEvidence: promptEvidence(promptCounts, promptDetails: promptDetails),
                evidenceCount: promptEvidenceCount(promptCounts),
                confidenceScore: confidence,
                meanTokenIndex: meanTokenIndex,
                minTokenIndex: minTokenIndex[key],
                maxTokenIndex: maxTokenIndex[key],
                label: label,
                isDead: h == 0 || (frequency < 0.001 && probabilityMass < 0.001),
                isHot: layerMax > 0 && Float(h) >= Float(layerMax) * 0.75
            )
        }

        let sourceNumExpertsByLayer = Dictionary(
            uniqueKeysWithValues: expectedExpertsByLayer
                .filter { $0.value > 0 }
                .map { (String($0.key), $0.value) }
        )
        return ExpertAtlas(
            promptCount: runs.count,
            experts: entries,
            sourceNumExpertsByLayer: sourceNumExpertsByLayer.isEmpty ? nil : sourceNumExpertsByLayer
        )
    }

    private static func finiteFloat(_ value: Float) -> Float {
        value.isFinite ? value : 0
    }

    private static func finiteMap(_ values: [String: Float]) -> [String: Float] {
        values.mapValues(finiteFloat)
    }

    private static func finiteNeighbors(_ neighbors: [ExpertCoactivationNeighbor]) -> [ExpertCoactivationNeighbor] {
        neighbors.map {
            ExpertCoactivationNeighbor(
                expert: $0.expert,
                count: $0.count,
                jaccard: finiteFloat($0.jaccard),
                pmi: finiteFloat($0.pmi)
            )
        }
    }

    private static func labelFor(
        hits: Int,
        layerMax: Int,
        domains: [String: Int],
        domainLift: [String: Float]
    ) -> String {
        guard hits > 0 else { return "dead" }
        let dominantDomain = ExpertDomainTaxonomy.dominantDomain(domains: domains, domainLift: domainLift)
        let domain = dominantDomain ?? "general"
        if layerMax > 0 && Float(hits) >= Float(layerMax) * 0.75 {
            return domain == "general" ? "general-hot" : "\(domain)-hot"
        }
        if let dominantDomain {
            let lift = ExpertDomainTaxonomy.canonicalLift(domainLift)[dominantDomain] ?? 1
            if lift >= 1.15 {
                return "\(dominantDomain)-specialist"
            }
            return "\(dominantDomain)-leaning"
        }
        if let best = domains.max(by: { $0.value < $1.value }), best.value * 2 >= hits {
            return "\(ExpertDomainTaxonomy.canonicalDomain(best.key))-specialist"
        }
        return "mixed"
    }

    private static func domainLift(
        layer: Int,
        hits: Int,
        domainCounts: [String: Int],
        layerTokenCount: Int,
        domainTokenSets: [Int: [String: Set<String>]]
    ) -> [String: Float] {
        guard hits > 0, layerTokenCount > 0 else { return [:] }
        let allRate = Float(hits) / Float(layerTokenCount)
        guard allRate > 0 else { return [:] }
        var lift: [String: Float] = [:]
        for (domain, count) in domainCounts {
            let domainTokens = domainTokenSets[layer]?[domain]?.count ?? 0
            guard domainTokens > 0 else { continue }
            lift[domain] = (Float(count) / Float(domainTokens)) / allRate
        }
        return lift
    }

    private static func neighbors(
        for key: LayerExpert,
        expertTokenSets: [LayerExpert: Set<String>],
        pairCounts: [LayerPair: Int],
        layerTokenCount: Int
    ) -> [ExpertCoactivationNeighbor] {
        let ownTokens = expertTokenSets[key] ?? []
        guard !ownTokens.isEmpty else { return [] }
        var out: [ExpertCoactivationNeighbor] = []
        for (pair, count) in pairCounts where pair.layer == key.layer && (pair.a == key.expert || pair.b == key.expert) {
            let otherExpert = pair.a == key.expert ? pair.b : pair.a
            let otherKey = LayerExpert(layer: key.layer, expert: otherExpert)
            let otherTokens = expertTokenSets[otherKey] ?? []
            let union = ownTokens.union(otherTokens).count
            let jaccard = union > 0 ? Float(count) / Float(union) : 0
            let pmi = pointwiseMutualInformation(
                pairCount: count,
                ownCount: ownTokens.count,
                otherCount: otherTokens.count,
                total: layerTokenCount
            )
            out.append(ExpertCoactivationNeighbor(expert: otherExpert, count: count, jaccard: jaccard, pmi: pmi))
        }
        return out.sorted {
            if $0.count != $1.count { return $0.count > $1.count }
            return $0.expert < $1.expert
        }.prefix(5).map { $0 }
    }

    private static func pointwiseMutualInformation(
        pairCount: Int,
        ownCount: Int,
        otherCount: Int,
        total: Int
    ) -> Float {
        guard pairCount > 0, ownCount > 0, otherCount > 0, total > 0 else { return 0 }
        let pxy = Double(pairCount) / Double(total)
        let px = Double(ownCount) / Double(total)
        let py = Double(otherCount) / Double(total)
        return Float(log(max(pxy / max(px * py, 1e-12), 1e-12)))
    }

    private static func topPrompts(_ counts: [String: Int]) -> [String] {
        counts.sorted {
            if $0.value != $1.value { return $0.value > $1.value }
            return $0.key < $1.key
        }
        .prefix(5)
        .map(\.key)
    }

    private static func promptEvidenceCount(_ counts: [String: Int]) -> Int {
        counts.values.filter { $0 > 0 }.count
    }

    private static func promptEvidence(
        _ counts: [String: Int],
        promptDetails: [String: ExpertPrompt]
    ) -> [ExpertPromptEvidence] {
        counts.sorted {
            if $0.value != $1.value { return $0.value > $1.value }
            return $0.key < $1.key
        }
        .prefix(5)
        .map { pair in
            let promptID = pair.key
            let hits = pair.value
            let prompt = promptDetails[promptID]
            return ExpertPromptEvidence(
                promptID: promptID,
                domain: prompt?.domain ?? "unknown",
                subdomain: prompt?.subdomain,
                tags: prompt?.tags ?? [],
                promptExcerpt: promptExcerpt(prompt?.text ?? promptID),
                hits: hits
            )
        }
    }

    private static func promptExcerpt(_ text: String, limit: Int = 160) -> String {
        let trimmed = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > limit else { return trimmed }
        let end = trimmed.index(trimmed.startIndex, offsetBy: limit)
        return String(trimmed[..<end]).trimmingCharacters(in: .whitespacesAndNewlines) + "..."
    }

    private static func confidenceScore(
        hits: Int,
        layerTokenCount: Int,
        domainLift: [String: Float]
    ) -> Float {
        guard layerTokenCount > 0 else { return 0 }
        let coverage = min(1, Float(hits) / max(1, Float(layerTokenCount)))
        let lift = min(1, max(0, (domainLift.values.max() ?? 1) - 1) / 3)
        return min(1, 0.70 * coverage + 0.30 * lift)
    }

    private struct LayerExpert: Hashable, Comparable {
        let layer: Int
        let expert: Int

        static func < (lhs: LayerExpert, rhs: LayerExpert) -> Bool {
            if lhs.layer != rhs.layer { return lhs.layer < rhs.layer }
            return lhs.expert < rhs.expert
        }
    }

    private struct LayerPair: Hashable {
        let layer: Int
        let a: Int
        let b: Int

        init(layer: Int, a: Int, b: Int) {
            self.layer = layer
            self.a = min(a, b)
            self.b = max(a, b)
        }
    }
}
