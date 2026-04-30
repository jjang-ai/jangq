import XCTest
@testable import JANGImage

final class PixtralTests: XCTestCase {
    func testNumTokens() {
        let p = PixtralImageProcessor()
        XCTAssertEqual(p.numImageTokens(hPatch: 110, wPatch: 110),
                       (110 / 2) * (110 / 2))
        XCTAssertEqual(p.numImageTokens(hPatch: 4, wPatch: 4), 4)
    }
}
