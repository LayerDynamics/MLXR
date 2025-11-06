// Copyright © 2025 MLXR Development
// Basic tests for MLXR client

import XCTest
@testable import MLXR

final class MLXRClientTests: XCTestCase {
    func testClientInitialization() {
        // Test Unix socket config
        let unixConfig = MLXRClientConfig.unixSocket()
        let unixClient = MLXRClient(config: unixConfig)
        XCTAssertNotNil(unixClient)

        // Test HTTP config
        let httpConfig = MLXRClientConfig.http()
        let httpClient = MLXRClient(config: httpConfig)
        XCTAssertNotNil(httpClient)
    }

    func testChatMessageEncoding() throws {
        let message = ChatMessage(
            role: "user",
            content: "Hello, world!"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(message)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ChatMessage.self, from: data)

        XCTAssertEqual(decoded.role, message.role)
        XCTAssertEqual(decoded.content, message.content)
    }

    func testChatCompletionRequestEncoding() throws {
        let request = ChatCompletionRequest(
            model: "test-model",
            messages: [
                ChatMessage(role: "system", content: "You are helpful."),
                ChatMessage(role: "user", content: "Hello!")
            ],
            temperature: 0.7,
            maxTokens: 100
        )

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(request)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let decoded = try decoder.decode(ChatCompletionRequest.self, from: data)

        XCTAssertEqual(decoded.model, request.model)
        XCTAssertEqual(decoded.messages.count, request.messages.count)
        XCTAssertEqual(decoded.temperature, request.temperature)
        XCTAssertEqual(decoded.maxTokens, request.maxTokens)
    }

    func testOllamaGenerateRequestEncoding() throws {
        let request = OllamaGenerateRequest(
            model: "test-model",
            prompt: "Test prompt",
            temperature: 0.8,
            numPredict: 50
        )

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(request)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let decoded = try decoder.decode(OllamaGenerateRequest.self, from: data)

        XCTAssertEqual(decoded.model, request.model)
        XCTAssertEqual(decoded.prompt, request.prompt)
        XCTAssertEqual(decoded.temperature, request.temperature)
        XCTAssertEqual(decoded.numPredict, request.numPredict)
    }

    func testErrorTypes() {
        let networkError = MLXRError.networkError(
            NSError(domain: "test", code: -1, userInfo: nil)
        )
        XCTAssertNotNil(networkError.errorDescription)

        let httpError = MLXRError.httpError(statusCode: 404, message: "Not found")
        XCTAssertNotNil(httpError.errorDescription)
        XCTAssertTrue(httpError.errorDescription!.contains("404"))

        let serverError = MLXRError.serverError(
            message: "Internal error",
            type: "server_error",
            code: "500"
        )
        XCTAssertNotNil(serverError.errorDescription)
        XCTAssertTrue(serverError.errorDescription!.contains("Internal error"))
    }

    func testConfigDefaults() {
        let unixConfig = MLXRClientConfig.unixSocket()

        if case .unixSocket(let path) = unixConfig.transport {
            XCTAssertTrue(path.contains("mlxrunner.sock"))
        } else {
            XCTFail("Expected Unix socket transport")
        }

        XCTAssertEqual(unixConfig.timeout, 120)

        let httpConfig = MLXRClientConfig.http()

        if case .http(let url) = httpConfig.transport {
            XCTAssertEqual(url.host, "127.0.0.1")
            XCTAssertEqual(url.port, 11434)
        } else {
            XCTFail("Expected HTTP transport")
        }
    }

    func testSSEEventParsing() {
        let sseData = """
        event: message
        data: {"test": "value"}
        id: 123

        event: done
        data: [DONE]

        """.data(using: .utf8)!

        let (events, bytesConsumed) = SSEParser.parseEvents(from: sseData)

        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0].event, "message")
        XCTAssertEqual(events[0].data, "{\"test\": \"value\"}")
        XCTAssertEqual(events[0].id, "123")

        XCTAssertEqual(events[1].event, "done")
        XCTAssertEqual(events[1].data, "[DONE]")
        XCTAssertGreaterThan(bytesConsumed, 0)
    }
}
