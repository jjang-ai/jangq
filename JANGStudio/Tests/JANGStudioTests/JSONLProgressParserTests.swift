// JANGStudio/Tests/JANGStudioTests/JSONLProgressParserTests.swift
import XCTest
@testable import JANGStudio

final class JSONLProgressParserTests: XCTestCase {
    func test_parseGoldenTrace() throws {
        let url = Bundle(for: Self.self).url(forResource: "golden_convert_events", withExtension: "jsonl")!
        let raw = try String(contentsOf: url, encoding: .utf8)
        let parser = JSONLProgressParser()
        let events = raw.split(whereSeparator: \.isNewline)
            .compactMap { parser.parse(line: String($0)) }

        XCTAssertEqual(events.first?.type, .phase)
        XCTAssertEqual(events.last?.type, .done)
        XCTAssertTrue(events.contains { if case .tick = $0.payload { return true } else { return false } })
    }

    func test_rejectsUnknownProtocolVersion() {
        let parser = JSONLProgressParser()
        let line = #"{"v":99,"type":"phase","n":1,"total":5,"name":"detect","ts":1.0}"#
        let ev = parser.parse(line: line)
        guard case .versionMismatch(let v) = ev?.payload else {
            return XCTFail("expected versionMismatch, got \(String(describing: ev))")
        }
        XCTAssertEqual(v, 99)
    }

    func test_ignoresPlainTextProgressLine() {
        let parser = JSONLProgressParser()
        let ev = parser.parse(line: "  Processing:  42%|████      | 42/100")
        XCTAssertNil(ev)
    }

    func test_ignoresArgparseChoiceHelpLine() {
        let parser = JSONLProgressParser()
        let ev = parser.parse(line: "            {inspect,validate,estimate,convert,profile}")
        XCTAssertNil(ev)
    }

    func test_tolerantOnMalformedJSONLine() {
        let parser = JSONLProgressParser()
        let ev = parser.parse(line: #"{"v":1,"type":"tick","#)
        guard case .parseError = ev?.payload else {
            return XCTFail("expected parseError, got \(String(describing: ev))")
        }
    }

    func test_doneEventWithError() throws {
        let line = #"{"v":1,"type":"done","ok":false,"error":"OOM","ts":1.0}"#
        let ev = JSONLProgressParser().parse(line: line)!
        guard case .done(let ok, let output, let error) = ev.payload else {
            return XCTFail("expected .done")
        }
        XCTAssertFalse(ok)
        XCTAssertNil(output)
        XCTAssertEqual(error, "OOM")
    }
}
