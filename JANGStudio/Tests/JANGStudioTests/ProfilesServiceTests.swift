import XCTest
@testable import JANGStudio

final class ProfilesServiceTests: XCTestCase {
    func test_frozen_has_15_jang_profiles() {
        XCTAssertEqual(Profiles.frozen.jang.count, 15)
    }

    func test_frozen_has_jangtq_profile_catalog_and_family_matrix() {
        let names = Set(Profiles.frozen.jangtq.map(\.name))
        XCTAssertTrue(names.isSuperset(of: ["JANGTQ1", "JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"]))
        XCTAssertEqual(
            Profiles.frozen.jangtqFamilies["deepseek_v4"]?.converter,
            "jang_tools.dsv4.convert_dsv4_jangtq"
        )
        XCTAssertTrue(Profiles.frozen.jangtqProfileNames(for: "minimax_m2").contains("JANGTQ_K"))
        XCTAssertFalse(Profiles.frozen.jangtqProfileNames(for: "zaya").contains("JANGTQ3"))
        XCTAssertFalse(Profiles.frozen.jangtqProfileNames(for: "bailing_hybrid").contains("JANGTQ3"))
    }

    func test_default_profile_is_jang_4k() {
        XCTAssertEqual(Profiles.frozen.defaultProfile, "JANG_4K")
        let defaults = Profiles.frozen.jang.filter { $0.isDefault }
        XCTAssertEqual(defaults.count, 1)
        XCTAssertEqual(defaults.first?.name, "JANG_4K")
    }

    func test_kquant_profiles_marked() {
        let kq = Profiles.frozen.jang.filter { $0.isKquant }.map { $0.name }
        XCTAssertTrue(kq.contains("JANG_3K"))
        XCTAssertTrue(kq.contains("JANG_4K"))
        XCTAssertTrue(kq.contains("JANG_5K"))
        XCTAssertTrue(kq.contains("JANG_6K"))
    }

    @MainActor
    func test_service_starts_with_frozen() {
        let s = ProfilesService()
        XCTAssertEqual(s.profiles, .frozen)
        XCTAssertFalse(s.isFromBundle)
    }

    // MARK: - M129 (iter 51): typed error parity
    //
    // Matches CapabilitiesServiceError's pattern. Pre-fix ProfilesService
    // threw raw NSError that stringified poorly into the UI banner.

    func test_profilesServiceError_cliError_formats_cleanly() {
        let e = ProfilesServiceError.cliError(code: 1, stderr: "ModuleNotFoundError: jang_tools\n")
        XCTAssertEqual(
            e.errorDescription,
            "jang-tools profiles exited 1: ModuleNotFoundError: jang_tools"
        )
    }

    func test_profilesServiceError_cliError_handles_empty_stderr() {
        let e = ProfilesServiceError.cliError(code: 127, stderr: "")
        let desc = e.errorDescription ?? ""
        XCTAssertTrue(desc.hasPrefix("jang-tools profiles exited 127"), "got \(desc)")
    }
}
