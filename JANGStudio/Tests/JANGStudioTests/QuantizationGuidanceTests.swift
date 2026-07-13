// JANGStudio/Tests/JANGStudioTests/QuantizationGuidanceTests.swift
import XCTest
@testable import JANGStudio

@MainActor
final class QuantizationGuidanceTests: XCTestCase {
    func test_kQuantProfileExplainsBudgetNeutralTarget() {
        let guide = QuantizationGuidance.profileGuide(
            name: "JANG_4K",
            profiles: .frozen
        )

        XCTAssertEqual(guide.name, "JANG_4K")
        XCTAssertTrue(guide.title.contains("4-bit"))
        XCTAssertTrue(guide.body.localizedCaseInsensitiveContains("K-quant"))
        XCTAssertTrue(guide.body.localizedCaseInsensitiveContains("overall"))
        XCTAssertTrue(guide.badges.contains("K-quant"))
    }

    func test_tieredProfileExplainsCriticalImportantAndBaseBits() {
        let guide = QuantizationGuidance.profileGuide(
            name: "JANG_2L",
            profiles: .frozen
        )

        XCTAssertTrue(guide.body.contains("critical"))
        XCTAssertTrue(guide.body.contains("important"))
        XCTAssertTrue(guide.body.contains("2-bit"))
        XCTAssertTrue(guide.badges.contains("critical 8"))
        XCTAssertTrue(guide.badges.contains("important 6"))
        XCTAssertTrue(guide.badges.contains("base 2"))
    }

    func test_jangtqProfileExplainsTurboQuantAndMoe() {
        let guide = QuantizationGuidance.profileGuide(
            name: "JANGTQ3",
            profiles: .frozen
        )

        XCTAssertTrue(guide.title.contains("3-bit"))
        XCTAssertTrue(guide.title.localizedCaseInsensitiveContains("TurboQuant"))
        XCTAssertTrue(guide.body.localizedCaseInsensitiveContains("codebook"))
        XCTAssertTrue(guide.body.contains("MoE"))
        XCTAssertTrue(guide.badges.contains("JANGTQ"))
    }

    func test_jangtqMethodAndHadamardGuidanceSaysIgnored() {
        let method = QuantizationGuidance.methodBody(.mse, family: .jangtq)
        let hadamard = QuantizationGuidance.hadamardBody(
            isEnabled: true,
            profile: "JANGTQ3",
            family: .jangtq
        )

        XCTAssertTrue(method.localizedCaseInsensitiveContains("ignored"))
        XCTAssertTrue(hadamard.localizedCaseInsensitiveContains("not passed"))
    }
}
