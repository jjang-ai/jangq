// JANGStudio/Tests/JANGStudioUITests/WizardFlowTests.swift
import XCTest

final class WizardFlowTests: XCTestCase {
    @MainActor
    func test_sidebarListsConvertStepsAtLaunch() {
        // PR3: cold launch is Convert mode — 4 steps, not Expert Lab 6.
        let app = XCUIApplication()
        app.launchEnvironment["JANGSTUDIO_PYTHON_OVERRIDE"] =
            Bundle(for: Self.self).path(forResource: "fake_convert", ofType: "sh")!
        app.launch()
        XCTAssertTrue(app.staticTexts["1 · Source Model"].exists)
        XCTAssertTrue(app.staticTexts["2 · Conversion Profile"].exists)
        XCTAssertTrue(app.staticTexts["3 · Build / Convert"].exists)
        XCTAssertTrue(app.staticTexts["4 · Verify"].exists)
        // Expert Lab steps must not appear until MoE + Expert Lab mode.
        XCTAssertFalse(app.staticTexts["2 · Expert Review"].exists)
        XCTAssertFalse(app.staticTexts["3 · Prune Review"].exists)
    }
}
