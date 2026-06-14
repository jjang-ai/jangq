import Foundation

struct ChatRequest: Encodable {
    let model: String
    let messages: [Message]
    let max_tokens: Int
    let temperature: Double
}

struct Message: Codable {
    let role: String
    let content: String
}

struct ChatResponse: Decodable {
    struct Choice: Decodable {
        let message: Message
    }

    let choices: [Choice]
}

func value(after flag: String, in args: [String], default fallback: String) -> String {
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else {
        return fallback
    }
    return args[index + 1]
}

let args = CommandLine.arguments
let baseURL = value(after: "--base-url", in: args, default: "http://127.0.0.1:8000/v1")
let apiKey = value(after: "--api-key", in: args, default: "not-needed")
let model = value(after: "--model", in: args, default: "default")
let prompt = value(after: "--prompt", in: args, default: "Explain JANG in one paragraph.")

let requestBody = ChatRequest(
    model: model,
    messages: [
        Message(role: "system", content: "You are a precise technical assistant."),
        Message(role: "user", content: prompt),
    ],
    max_tokens: 256,
    temperature: 0.0
)

let endpoint = URL(string: baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/chat/completions")!
var request = URLRequest(url: endpoint)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
request.httpBody = try JSONEncoder().encode(requestBody)

let semaphore = DispatchSemaphore(value: 0)
let task = URLSession.shared.dataTask(with: request) { data, response, error in
    defer { semaphore.signal() }

    if let error {
        fputs("request failed: \(error)\n", stderr)
        return
    }

    guard let http = response as? HTTPURLResponse else {
        fputs("missing HTTP response\n", stderr)
        return
    }

    guard (200..<300).contains(http.statusCode), let data else {
        let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
        fputs("HTTP \(http.statusCode): \(body)\n", stderr)
        return
    }

    do {
        let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
        print(decoded.choices.first?.message.content ?? "")
    } catch {
        fputs("decode failed: \(error)\n", stderr)
    }
}

task.resume()
semaphore.wait()
