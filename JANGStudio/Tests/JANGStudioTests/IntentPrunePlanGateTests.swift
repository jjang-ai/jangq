import XCTest

/// Source-inspection tests for `jang-intent-prune-plan-v1` acceptance in Studio gates.
final class IntentPrunePlanGateTests: XCTestCase {
    private static let studioRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // JANGStudioTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // JANGStudio (xcodeproj root)
            .appendingPathComponent("JANGStudio")
    }()

    private func source(at relativePath: String) throws -> String {
        let url = Self.studioRoot.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    func test_preflightRecognizesIntentPrunePlanSchema() throws {
        let src = try source(at: "Verify/PreflightRunner.swift")
        XCTAssertTrue(src.contains("jang-intent-prune-plan-v1"))
        XCTAssertTrue(src.contains("isIntentPrunePlan"))
        XCTAssertTrue(src.contains("intentPruneFingerprintIssue"))
        XCTAssertTrue(src.contains("planLayerKeepList"))
        XCTAssertTrue(src.contains("trained_top_k"))
        XCTAssertTrue(src.contains("!isIntentPrunePlan(plan), let issue = reviewedPruneSemanticEvidenceIssue"))
    }

    func test_postConvertRecognizesIntentPrunePlanSchema() throws {
        let src = try source(at: "Verify/PostConvertVerifier.swift")
        XCTAssertTrue(src.contains("jang-intent-prune-plan-v1"))
        XCTAssertTrue(src.contains("isIntentPrunePlan"))
        XCTAssertTrue(src.contains("intentPruneFingerprintIssue"))
        XCTAssertTrue(src.contains("planLayerKeepList"))
        XCTAssertTrue(src.contains("!isIntentPrunePlan(plan), let issue = reviewedPruneSemanticEvidenceIssue"))
    }

    func test_prequantPruneSheetNormalizesIntentPlans() throws {
        let src = try source(at: "Wizard/PrequantPruneSheet.swift")
        XCTAssertTrue(src.contains("jang-intent-prune-plan-v1"))
        XCTAssertTrue(src.contains("normalizeImportedPrunePlanDictionary"))
        XCTAssertTrue(src.contains("skipSemanticEvidence"))
        XCTAssertTrue(src.contains("IntentPrunePlanMeta"))
        XCTAssertTrue(src.contains("crack_pack"))
        XCTAssertTrue(src.contains("trained_top_k"))
    }
}
