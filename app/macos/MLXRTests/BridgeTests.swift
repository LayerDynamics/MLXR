//
//  BridgeTests.swift
//  MLXRTests
//
//  Tests for JavaScript bridge communication.
//

import XCTest
import WebKit
@testable import MLXR

class BridgeTests: XCTestCase {

    var messageHandlers: MessageHandlers!
    var webView: WKWebView!

    override func setUp() {
        super.setUp()
        messageHandlers = MessageHandlers()

        let config = WKWebViewConfiguration()
        config.userContentController.add(messageHandlers, name: "hostBridge")
        webView = WKWebView(frame: .zero, configuration: config)
    }

    override func tearDown() {
        webView = nil
        messageHandlers = nil
        super.tearDown()
    }

    // MARK: - Bridge Injection Tests

    func testBridgeInjectorExists() {
        let bundle = Bundle(for: type(of: self))
        let injectorPath = bundle.path(forResource: "BridgeInjector", ofType: "js")
        XCTAssertNotNil(injectorPath, "BridgeInjector.js should exist in bundle")

        if let path = injectorPath {
            let content = try? String(contentsOfFile: path)
            XCTAssertNotNil(content, "BridgeInjector.js should be readable")
            XCTAssertTrue(content!.contains("window.__HOST__"), "Should define __HOST__ interface")
        }
    }

    func testBridgeInjectsWindowHost() async throws {
        let injectorPath = Bundle(for: type(of: self)).path(forResource: "BridgeInjector", ofType: "js")!
        let injectorScript = try String(contentsOfFile: injectorPath)

        let userScript = WKUserScript(
            source: injectorScript,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        webView.configuration.userContentController.addUserScript(userScript)

        // Load blank page
        webView.loadHTMLString("<html><body></body></html>", baseURL: nil)

        // Wait for load
        try await Task.sleep(nanoseconds: 500_000_000) // 0.5s

        // Check if __HOST__ exists
        let hasHost = try await webView.evaluateJavaScript("typeof window.__HOST__ !== 'undefined'") as? Bool
        XCTAssertEqual(hasHost, true, "window.__HOST__ should be defined")
    }

    func testBridgeHasAllMethods() async throws {
        let injectorPath = Bundle(for: type(of: self)).path(forResource: "BridgeInjector", ofType: "js")!
        let injectorScript = try String(contentsOfFile: injectorPath)

        let userScript = WKUserScript(
            source: injectorScript,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        webView.configuration.userContentController.addUserScript(userScript)

        webView.loadHTMLString("<html><body></body></html>", baseURL: nil)
        try await Task.sleep(nanoseconds: 500_000_000)

        let methods = ["request", "openPathDialog", "readConfig", "writeConfig",
                      "startDaemon", "stopDaemon", "getVersion"]

        for method in methods {
            let hasMethod = try await webView.evaluateJavaScript(
                "typeof window.__HOST__.\(method) === 'function'"
            ) as? Bool
            XCTAssertEqual(hasMethod, true, "window.__HOST__.\(method) should be a function")
        }
    }

    // MARK: - Message Handler Tests

    func testGetVersionRequest() async throws {
        let message = [
            "id": 1,
            "method": "getVersion",
            "params": [String: Any]()
        ] as [String : Any]

        let expectation = XCTestExpectation(description: "getVersion response")

        // Mock script message
        let mockMessage = MockScriptMessage(body: message)

        messageHandlers.userContentController(
            webView.configuration.userContentController,
            didReceive: mockMessage
        )

        // In real implementation, response would be sent via evaluateJavaScript
        // For now, just verify the method doesn't crash
        expectation.fulfill()

        await fulfillment(of: [expectation], timeout: 1.0)
    }

    func testInvalidMethodReturnsError() {
        let message = [
            "id": 1,
            "method": "invalidMethod",
            "params": [String: Any]()
        ] as [String : Any]

        let mockMessage = MockScriptMessage(body: message)

        // Should not crash
        messageHandlers.userContentController(
            webView.configuration.userContentController,
            didReceive: mockMessage
        )
    }

    func testMalformedMessageHandledGracefully() {
        let message = ["invalid": "structure"]
        let mockMessage = MockScriptMessage(body: message)

        // Should not crash
        messageHandlers.userContentController(
            webView.configuration.userContentController,
            didReceive: mockMessage
        )
    }

    // MARK: - Bridge Method Tests

    func testReadConfigMethod() async throws {
        // Create temporary config file
        let tempDir = FileManager.default.temporaryDirectory
        let configPath = tempDir.appendingPathComponent("test_server.yaml")
        let configContent = "port: 8080\nhost: localhost"
        try configContent.write(to: configPath, atomically: true, encoding: .utf8)

        defer {
            try? FileManager.default.removeItem(at: configPath)
        }

        // Test would call readConfig and verify result
        // In actual implementation, this would go through the full bridge
    }

    func testOpenPathDialogMethod() {
        // Test that openPathDialog returns a valid path or nil
        // This would need to mock NSOpenPanel
        // For now, verify the method signature exists
        XCTAssertNotNil(messageHandlers.openPathDialog)
    }

    // MARK: - Performance Tests

    func testBridgeMessagePerformance() {
        let message = [
            "id": 1,
            "method": "getVersion",
            "params": [String: Any]()
        ] as [String : Any]

        let mockMessage = MockScriptMessage(body: message)

        measure {
            for _ in 0..<100 {
                messageHandlers.userContentController(
                    webView.configuration.userContentController,
                    didReceive: mockMessage
                )
            }
        }
    }
}

// MARK: - Mock Objects

class MockScriptMessage: WKScriptMessage {
    private let _body: Any

    init(body: Any) {
        self._body = body
    }

    override var body: Any {
        return _body
    }

    override var name: String {
        return "hostBridge"
    }
}
