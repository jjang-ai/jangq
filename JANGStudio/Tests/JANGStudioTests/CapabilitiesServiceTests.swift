import XCTest
@testable import JANGStudio

final class CapabilitiesServiceTests: XCTestCase {
    func test_frozen_has_known_content() {
        let f = Capabilities.frozen
        XCTAssertTrue(f.jangtqWhitelist.contains("qwen3_5_moe"))
        XCTAssertTrue(f.jangtqWhitelist.contains("qwen3_5_moe_text"))
        XCTAssertTrue(f.jangtqWhitelist.contains("minimax_m2"))
        XCTAssertEqual(f.defaultBlockSize, 64)
        XCTAssertEqual(f.defaultMethod, "mse")
        XCTAssertEqual(f.methods.count, 3)
        XCTAssertTrue(f.tokenizerClassBlocklist.contains("TokenizersBackend"))
    }

    @MainActor
    func test_service_starts_with_frozen() {
        let s = CapabilitiesService()
        XCTAssertEqual(s.capabilities, .frozen)
        XCTAssertFalse(s.isFromBundle)
    }

    func test_dtype_info_decodes() throws {
        let json = #"{"name":"bfloat16","alias":"bf16","description":"Test"}"#
        let d = try JSONDecoder().decode(Capabilities.DtypeInfo.self, from: Data(json.utf8))
        XCTAssertEqual(d.alias, "bf16")
    }

    // MARK: - M129 (iter 51): typed error parity with peer adoption services
    //
    // Pre-iter-51, invokeCLI threw a raw NSError and refresh()'s
    // `self.lastError = "\(error)"` stringified it to
    // `Error Domain=CapabilitiesService Code=1 "(null)" UserInfo={…}` —
    // ugly and leaks framework internals into the UI banner. Typed
    // CapabilitiesServiceError.cliError gives a clean errorDescription
    // matching RecommendationService / ExamplesService / ModelCardService.

    func test_capabilitiesServiceError_cliError_formats_cleanly() {
        let e = CapabilitiesServiceError.cliError(code: 1, stderr: "ModuleNotFoundError: jang_tools\n")
        XCTAssertEqual(
            e.errorDescription,
            "jang-tools capabilities exited 1: ModuleNotFoundError: jang_tools",
            "errorDescription must trim whitespace + surface the CLI name so the UI banner is self-explanatory"
        )
    }

    func test_capabilitiesServiceError_cliError_handles_empty_stderr() {
        // Python may exit non-zero with empty stderr (e.g. bundle missing).
        // errorDescription must not crash or produce garbled output.
        let e = CapabilitiesServiceError.cliError(code: 127, stderr: "")
        let desc = e.errorDescription ?? ""
        XCTAssertTrue(desc.hasPrefix("jang-tools capabilities exited 127"), "got \(desc)")
    }
}
