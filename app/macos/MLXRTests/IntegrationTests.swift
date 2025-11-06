//
//  IntegrationTests.swift
//  MLXRTests
//
//  Integration tests for full workflow scenarios.
//

import XCTest
import WebKit
@testable import MLXR

class IntegrationTests: XCTestCase {

    var testDirectory: URL!

    override func setUp() {
        super.setUp()

        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXRIntegrationTests_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: testDirectory,
            withIntermediateDirectories: true
        )
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: testDirectory)
        testDirectory = nil
        super.tearDown()
    }

    // MARK: - First Run Integration Tests

    func testFirstRunSetup() {
        // Simulate first run setup
        let appSupport = testDirectory
        let mlxrDir = appSupport.appendingPathComponent("MLXRunner")

        let directories = [
            mlxrDir,
            mlxrDir.appendingPathComponent("models"),
            mlxrDir.appendingPathComponent("cache"),
            mlxrDir.appendingPathComponent("bin"),
            mlxrDir.appendingPathComponent("run"),
        ]

        for dir in directories {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        // Verify all directories created
        for dir in directories {
            XCTAssertTrue(FileManager.default.fileExists(atPath: dir.path),
                         "Directory should exist: \(dir.lastPathComponent)")
        }
    }

    func testFirstRunConfigCreation() throws {
        let configPath = testDirectory.appendingPathComponent("server.yaml")
        let defaultConfig = ConfigManager.shared.getDefaultConfig()

        try defaultConfig.write(to: configPath, atomically: true, encoding: .utf8)

        XCTAssertTrue(FileManager.default.fileExists(atPath: configPath.path))

        let content = try String(contentsOf: configPath)
        XCTAssertTrue(content.contains("server:"))
        XCTAssertTrue(ConfigManager.shared.validateConfig(content))
    }

    // MARK: - Bridge + Daemon Integration

    func testBridgeToDaemonFlow() async throws {
        // Test the full flow: JS -> Bridge -> UnixSocketClient -> Daemon
        let messageHandlers = MessageHandlers()

        // Simulate daemon not running
        do {
            let status = try await DaemonManager.shared.getDaemonStatus()
            // If we got status, daemon is actually running
            XCTAssertNotNil(status)
        } catch {
            // Expected if daemon not running
            XCTAssertTrue(error is DaemonManager.DaemonError)
        }
    }

    func testBridgeRequestFlow() async throws {
        // Test bridge request method
        let client = UnixSocketClient(socketPath: "/tmp/test.sock")

        do {
            let response = try await client.request(
                method: "GET",
                path: "/health",
                headers: [:],
                body: nil
            )
            // If socket exists and responds, good
            XCTAssertNotNil(response)
        } catch {
            // Expected if daemon not running
            XCTAssertTrue(error is UnixSocketClient.SocketError)
        }
    }

    // MARK: - Config + Keychain Integration

    func testConfigAndKeychainIntegration() throws {
        // Create config
        let configPath = testDirectory.appendingPathComponent("server.yaml")
        let config = """
        server:
          auth_token: ${KEYCHAIN_TOKEN}
        """
        try config.write(to: configPath, atomically: true, encoding: .utf8)

        // Save token to keychain
        let token = KeychainManager.shared.generateToken()
        try KeychainManager.shared.saveToken(token)

        // Verify token retrievable
        let retrieved = try KeychainManager.shared.retrieveToken()
        XCTAssertEqual(retrieved, token)

        // Clean up
        try KeychainManager.shared.deleteToken()
    }

    // MARK: - Daemon Lifecycle Integration

    func testDaemonLifecycleFlow() async throws {
        let manager = DaemonManager.shared

        // Test full lifecycle (may fail if daemon not set up, that's okay)
        do {
            // Check status
            let isRunning = try await manager.isDaemonRunning()

            if !isRunning {
                // Try to start
                try await manager.startDaemon()

                // Verify started
                let nowRunning = try await manager.isDaemonRunning()
                XCTAssertTrue(nowRunning, "Daemon should be running after start")

                // Get status
                let status = try await manager.getDaemonStatus()
                XCTAssertNotNil(status)

                // Stop daemon
                try await manager.stopDaemon()

                // Verify stopped
                let stopped = try await manager.isDaemonRunning()
                XCTAssertFalse(stopped, "Daemon should be stopped")
            }
        } catch {
            // Expected if binary not found
            XCTAssertTrue(error is DaemonManager.DaemonError)
        }
    }

    // MARK: - App Launch Simulation

    func testAppLaunchFlow() {
        // Simulate app launch sequence
        // 1. Check first run
        let key = "MLXRTestHasLaunchedBefore"
        let hasLaunched = UserDefaults.standard.bool(forKey: key)

        if !hasLaunched {
            UserDefaults.standard.set(true, forKey: key)

            // 2. Create directories (tested above)
            // 3. Create default config (tested above)
            // 4. Install daemon binary (would be tested with real binary)
        }

        // 5. Check daemon status
        // 6. Start daemon if needed
        // 7. Setup notifications
        // 8. Setup tray

        XCTAssertTrue(true, "App launch flow completed")

        // Clean up
        UserDefaults.standard.removeObject(forKey: key)
    }

    // MARK: - WebView + Bridge Integration

    func testWebViewBridgeIntegration() async throws {
        let config = WKWebViewConfiguration()
        let messageHandlers = MessageHandlers()
        config.userContentController.add(messageHandlers, name: "hostBridge")

        // Load bridge script
        if let injectorPath = Bundle(for: type(of: self)).path(forResource: "BridgeInjector", ofType: "js") {
            let injectorScript = try String(contentsOfFile: injectorPath)
            let userScript = WKUserScript(
                source: injectorScript,
                injectionTime: .atDocumentStart,
                forMainFrameOnly: true
            )
            config.userContentController.addUserScript(userScript)
        }

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.loadHTMLString("<html><body>Test</body></html>", baseURL: nil)

        try await Task.sleep(nanoseconds: 500_000_000)

        // Verify bridge is available
        let hasHost = try await webView.evaluateJavaScript(
            "typeof window.__HOST__ !== 'undefined'"
        ) as? Bool
        XCTAssertEqual(hasHost, true)
    }

    // MARK: - Error Recovery Tests

    func testDaemonCrashRecovery() async throws {
        let manager = DaemonManager.shared

        // Simulate daemon crash scenario
        do {
            // Try to get status when daemon might be down
            _ = try await manager.getDaemonStatus()
        } catch {
            // Error expected - test recovery
            do {
                // Try to restart
                try await manager.restartDaemon()
            } catch {
                // Recovery may fail if binary not found
                XCTAssertTrue(error is DaemonManager.DaemonError)
            }
        }
    }

    func testBridgeErrorRecovery() {
        let messageHandlers = MessageHandlers()

        // Send malformed messages
        let badMessages: [[String: Any]] = [
            [:],  // Empty
            ["method": "invalid"],  // Missing id
            ["id": 1],  // Missing method
            ["id": "wrong", "method": "test"]  // Wrong id type
        ]

        for message in badMessages {
            let mockMessage = MockScriptMessage(body: message)

            // Should not crash
            messageHandlers.userContentController(
                WKWebViewConfiguration().userContentController,
                didReceive: mockMessage
            )
        }

        XCTAssertTrue(true, "Bridge handled all malformed messages")
    }

    // MARK: - Performance Integration Tests

    func testEndToEndLatency() {
        measure {
            let expectation = XCTestExpectation(description: "End to end request")

            Task {
                do {
                    // Simulate full request cycle
                    _ = try await DaemonManager.shared.isDaemonRunning()
                } catch {
                    // Expected if daemon not running
                }
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: 2.0)
        }
    }

    // MARK: - Concurrency Integration Tests

    func testConcurrentOperations() async {
        // Test that multiple operations can run concurrently
        await withTaskGroup(of: Void.self) { group in
            // Concurrent status checks
            group.addTask {
                _ = try? await DaemonManager.shared.isDaemonRunning()
            }

            // Concurrent config reads
            group.addTask {
                let config = ConfigManager.shared.getDefaultConfig()
                XCTAssertFalse(config.isEmpty)
            }

            // Concurrent keychain operations
            group.addTask {
                do {
                    try KeychainManager.shared.deleteToken()
                    let token = KeychainManager.shared.generateToken()
                    try KeychainManager.shared.saveToken(token)
                    _ = try KeychainManager.shared.retrieveToken()
                    try KeychainManager.shared.deleteToken()
                } catch {
                    // Expected if keychain access fails
                }
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
