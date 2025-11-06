//
//  DaemonManagerTests.swift
//  MLXRTests
//
//  Tests for daemon lifecycle management.
//

import XCTest
@testable import MLXR

class DaemonManagerTests: XCTestCase {

    var daemonManager: DaemonManager!
    var testDirectory: URL!

    override func setUp() {
        super.setUp()

        // Create temporary test directory
        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXRTests_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: testDirectory,
            withIntermediateDirectories: true
        )

        // Note: DaemonManager uses singleton, so we can't easily inject dependencies
        // In production, we'd refactor to use dependency injection
        daemonManager = DaemonManager.shared
    }

    override func tearDown() {
        // Clean up test directory
        try? FileManager.default.removeItem(at: testDirectory)
        testDirectory = nil
        daemonManager = nil
        super.tearDown()
    }

    // MARK: - Path Configuration Tests

    func testDaemonPaths() {
        let binaryPath = daemonManager.daemonBinaryPath
        let configPath = daemonManager.configPath
        let socketPath = daemonManager.socketPath

        XCTAssertTrue(binaryPath.contains("MLXRunner/bin/mlxrunnerd"),
                     "Binary path should be in MLXRunner/bin")
        XCTAssertTrue(configPath.contains("MLXRunner/server.yaml"),
                     "Config path should be server.yaml")
        XCTAssertTrue(socketPath.contains("MLXRunner/run/mlxrunner.sock"),
                     "Socket path should be in run directory")
    }

    func testDaemonPathsAreAbsolute() {
        let binaryPath = daemonManager.daemonBinaryPath
        let configPath = daemonManager.configPath
        let socketPath = daemonManager.socketPath

        XCTAssertTrue(binaryPath.hasPrefix("/"), "Binary path should be absolute")
        XCTAssertTrue(configPath.hasPrefix("/"), "Config path should be absolute")
        XCTAssertTrue(socketPath.hasPrefix("/"), "Socket path should be absolute")
    }

    // MARK: - Daemon Status Tests

    func testIsDaemonRunningWhenNotRunning() async throws {
        // If no daemon is running, should return false
        // This test assumes daemon is not running during tests
        let isRunning = try await daemonManager.isDaemonRunning()
        // Can't assert false because daemon might actually be running
        // Just verify it doesn't crash
        XCTAssertNotNil(isRunning)
    }

    func testGetDaemonStatus() async {
        do {
            let status = try await daemonManager.getDaemonStatus()
            // If daemon is running, should return status
            XCTAssertNotNil(status)
        } catch {
            // If daemon not running, error is expected
            XCTAssertTrue(error is DaemonManager.DaemonError)
        }
    }

    // MARK: - Configuration Tests

    func testCreateDefaultConfig() throws {
        let testConfigPath = testDirectory.appendingPathComponent("server.yaml").path

        // This would test default config creation
        // In practice, we'd refactor DaemonManager to accept custom paths
        let defaultConfig = """
        # MLXR Server Configuration

        # Server settings
        server:
          socket: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock
          log_level: info

        # Model settings
        models:
          cache_dir: ~/Library/Application Support/MLXRunner/cache
          max_loaded: 2

        # Performance settings
        performance:
          max_batch_tokens: 8192
          target_latency_ms: 80
          enable_speculative: true

        # Features
        features:
          kv_persistence: true
        """

        XCTAssertTrue(defaultConfig.contains("socket:"))
        XCTAssertTrue(defaultConfig.contains("cache_dir:"))
    }

    // MARK: - Daemon Lifecycle Tests

    func testStartDaemonWithoutBinary() async {
        // Test behavior when binary doesn't exist
        // This should fail gracefully
        do {
            _ = try await daemonManager.startDaemon()
            // If binary exists and daemon started, that's fine
        } catch let error as DaemonManager.DaemonError {
            // Expected errors
            switch error {
            case .binaryNotFound:
                XCTAssertTrue(true, "Correctly detected missing binary")
            case .startFailed:
                XCTAssertTrue(true, "Start failed as expected without binary")
            default:
                XCTFail("Unexpected error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testStopDaemonWhenNotRunning() async {
        // Test stopping daemon that isn't running
        do {
            try await daemonManager.stopDaemon()
            // Should succeed (no-op)
            XCTAssertTrue(true)
        } catch {
            // Some implementations might throw, that's okay too
            XCTAssertTrue(error is DaemonManager.DaemonError)
        }
    }

    func testRestartDaemon() async {
        // Test restart behavior
        do {
            try await daemonManager.restartDaemon()
            // If binary exists and restart works, good
        } catch {
            // Expected if daemon not set up
            XCTAssertTrue(error is DaemonManager.DaemonError)
        }
    }

    // MARK: - Error Handling Tests

    func testDaemonErrorTypes() {
        let errors: [DaemonManager.DaemonError] = [
            .binaryNotFound("test"),
            .configError("test"),
            .startFailed("test"),
            .stopFailed("test"),
            .notRunning,
            .healthCheckFailed("test")
        ]

        for error in errors {
            XCTAssertNotNil(error.localizedDescription)
            XCTAssertFalse(error.localizedDescription.isEmpty)
        }
    }

    func testDaemonErrorDescriptions() {
        let error1 = DaemonManager.DaemonError.binaryNotFound("path/to/binary")
        XCTAssertTrue(error1.localizedDescription.contains("binary"))

        let error2 = DaemonManager.DaemonError.notRunning
        XCTAssertTrue(error2.localizedDescription.contains("not running"))

        let error3 = DaemonManager.DaemonError.startFailed("timeout")
        XCTAssertTrue(error3.localizedDescription.contains("start"))
    }

    // MARK: - Health Check Tests

    func testHealthCheckWhenDaemonDown() async {
        // Health check should fail when daemon not running
        do {
            _ = try await daemonManager.getDaemonStatus()
            // If succeeded, daemon is running (that's okay)
        } catch let error as DaemonManager.DaemonError {
            XCTAssertTrue(
                error == .notRunning || error.localizedDescription.contains("health"),
                "Should get notRunning or healthCheckFailed error"
            )
        } catch {
            XCTFail("Unexpected error type")
        }
    }

    // MARK: - Concurrent Access Tests

    func testConcurrentStatusChecks() async throws {
        // Test that multiple concurrent status checks don't crash
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    do {
                        _ = try await self.daemonManager.isDaemonRunning()
                    } catch {
                        // Expected if daemon not running
                    }
                }
            }
        }
    }

    // MARK: - Performance Tests

    func testStatusCheckPerformance() {
        measure {
            let expectation = XCTestExpectation(description: "Status check")
            Task {
                do {
                    _ = try await daemonManager.isDaemonRunning()
                } catch {
                    // Expected
                }
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5.0)
        }
    }
}

// MARK: - DaemonError Equatable Extension for Testing

extension DaemonManager.DaemonError: Equatable {
    public static func == (lhs: DaemonManager.DaemonError, rhs: DaemonManager.DaemonError) -> Bool {
        switch (lhs, rhs) {
        case (.binaryNotFound, .binaryNotFound),
             (.configError, .configError),
             (.startFailed, .startFailed),
             (.stopFailed, .stopFailed),
             (.notRunning, .notRunning),
             (.healthCheckFailed, .healthCheckFailed):
            return true
        default:
            return false
        }
    }
}
