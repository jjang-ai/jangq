// JANGStudio/JANGStudio/Runner/JSONLProgressParser.swift
import Foundation

final class JSONLProgressParser {
    private static let supportedVersion = 1
    private let decoder = JSONDecoder()

    private struct Raw: Decodable {
        let v: Int?
        let type: String?
        let n: Int?
        let total: Int?
        let name: String?
        let done: Int?
        let label: String?
        let msg: String?
        let ok: Bool?
        let output: String?
        let error: String?
        let ts: TimeInterval?
    }

    func parse(line: String) -> ProgressEvent? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{") else { return nil }
        let afterBrace = trimmed.dropFirst().drop(while: { $0.isWhitespace })
        guard afterBrace.first == "\"", let data = trimmed.data(using: .utf8) else { return nil }
        let raw: Raw
        do { raw = try decoder.decode(Raw.self, from: data) }
        catch {
            return ProgressEvent(ts: Date().timeIntervalSince1970, type: .error,
                                  payload: .parseError(line))
        }
        if let v = raw.v, v != Self.supportedVersion {
            return ProgressEvent(ts: raw.ts ?? 0, type: .error, payload: .versionMismatch(v))
        }
        guard let typeStr = raw.type, let type = EventType(rawValue: typeStr) else {
            return ProgressEvent(ts: raw.ts ?? 0, type: .error, payload: .parseError(line))
        }
        let ts = raw.ts ?? Date().timeIntervalSince1970
        switch type {
        case .phase:
            guard let n = raw.n, let total = raw.total, let name = raw.name else { return nil }
            return .init(ts: ts, type: .phase, payload: .phase(n: n, total: total, name: name))
        case .tick:
            guard let done = raw.done, let total = raw.total else { return nil }
            return .init(ts: ts, type: .tick, payload: .tick(done: done, total: total, label: raw.label))
        case .info, .warn, .error:
            return .init(ts: ts, type: type, payload: .message(level: typeStr, text: raw.msg ?? ""))
        case .done:
            return .init(ts: ts, type: .done,
                         payload: .done(ok: raw.ok ?? false, output: raw.output, error: raw.error))
        }
    }
}
