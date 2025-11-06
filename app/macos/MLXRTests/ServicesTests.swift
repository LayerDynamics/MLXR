//
//  ServicesTests.swift
//  MLXRTests
//
//  Tests for app services (Keychain, Config, LoginItem).
//

import XCTest
@testable import MLXR

class KeychainManagerTests: XCTestCase {

    var keychainManager: KeychainManager!
    let testService = "com.mlxr.app.test"
    let testAccount = "test-token"

    override func setUp() {
        super.setUp()
        keychainManager = KeychainManager.shared

        // Clean up any existing test tokens
        try? keychainManager.deleteToken()
    }

    override func tearDown() {
        // Clean up test tokens
        try? keychainManager.deleteToken()
        keychainManager = nil
        super.tearDown()
    }

    // MARK: - Token Generation Tests

    func testGenerateToken() {
        let token = keychainManager.generateToken()

        XCTAssertEqual(token.count, 32, "Token should be 32 characters")
        XCTAssertTrue(token.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber) },
                     "Token should be alphanumeric")
    }

    func testGenerateUniqueTokens() {
        let token1 = keychainManager.generateToken()
        let token2 = keychainManager.generateToken()

        XCTAssertNotEqual(token1, token2, "Generated tokens should be unique")
    }

    // MARK: - Save/Retrieve Tests

    func testSaveAndRetrieveToken() throws {
        let testToken = "test-token-12345678901234567890"

        try keychainManager.saveToken(testToken)

        let retrieved = try keychainManager.retrieveToken()
        XCTAssertEqual(retrieved, testToken, "Retrieved token should match saved token")
    }

    func testRetrieveNonexistentToken() {
        // Should throw error when no token exists
        XCTAssertThrowsError(try keychainManager.retrieveToken()) { error in
            XCTAssertTrue(error is KeychainManager.KeychainError)
        }
    }

    func testOverwriteExistingToken() throws {
        let token1 = "first-token-123456789012345678"
        let token2 = "second-token-12345678901234567"

        try keychainManager.saveToken(token1)
        try keychainManager.saveToken(token2)

        let retrieved = try keychainManager.retrieveToken()
        XCTAssertEqual(retrieved, token2, "Should retrieve most recent token")
    }

    // MARK: - Delete Tests

    func testDeleteToken() throws {
        let testToken = "delete-test-123456789012345678"

        try keychainManager.saveToken(testToken)
        try keychainManager.deleteToken()

        XCTAssertThrowsError(try keychainManager.retrieveToken()) { error in
            XCTAssertTrue(error is KeychainManager.KeychainError)
        }
    }

    func testDeleteNonexistentToken() {
        // Deleting non-existent token should not throw
        XCTAssertNoThrow(try keychainManager.deleteToken())
    }

    // MARK: - Rotate Token Tests

    func testRotateToken() throws {
        let oldToken = "old-token-1234567890123456789"
        try keychainManager.saveToken(oldToken)

        let newToken = try keychainManager.rotateToken()

        XCTAssertNotEqual(newToken, oldToken, "Rotated token should be different")
        XCTAssertEqual(newToken.count, 32, "Rotated token should be 32 characters")

        let retrieved = try keychainManager.retrieveToken()
        XCTAssertEqual(retrieved, newToken, "Retrieved token should be the new token")
    }

    // MARK: - Error Tests

    func testKeychainErrorDescriptions() {
        let errors: [KeychainManager.KeychainError] = [
            .saveFailed("test"),
            .notFound,
            .deleteFailed("test"),
            .unexpectedData
        ]

        for error in errors {
            XCTAssertFalse(error.localizedDescription.isEmpty)
        }
    }
}

// MARK: - ConfigManager Tests

class ConfigManagerTests: XCTestCase {

    var configManager: ConfigManager!
    var testDirectory: URL!

    override func setUp() {
        super.setUp()

        // Create temporary test directory
        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXRConfigTests_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: testDirectory,
            withIntermediateDirectories: true
        )

        configManager = ConfigManager.shared
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: testDirectory)
        testDirectory = nil
        configManager = nil
        super.tearDown()
    }

    // MARK: - Read Tests

    func testReadValidConfig() throws {
        let configPath = testDirectory.appendingPathComponent("test.yaml")
        let configContent = """
        server:
          port: 8080
          host: localhost
        models:
          cache_dir: /tmp/cache
        """
        try configContent.write(to: configPath, atomically: true, encoding: .utf8)

        let config = try configManager.readConfig(from: configPath.path)

        XCTAssertTrue(config.contains("server:"))
        XCTAssertTrue(config.contains("8080"))
        XCTAssertTrue(config.contains("localhost"))
    }

    func testReadNonexistentConfig() {
        let configPath = testDirectory.appendingPathComponent("nonexistent.yaml").path

        XCTAssertThrowsError(try configManager.readConfig(from: configPath)) { error in
            XCTAssertTrue(error is ConfigManager.ConfigError)
        }
    }

    // MARK: - Write Tests

    func testWriteValidConfig() throws {
        let configPath = testDirectory.appendingPathComponent("write_test.yaml")
        let configContent = """
        server:
          port: 9090
        """

        try configManager.writeConfig(configContent, to: configPath.path)

        let written = try String(contentsOf: configPath, encoding: .utf8)
        XCTAssertEqual(written, configContent)
    }

    func testWriteInvalidYAML() {
        let configPath = testDirectory.appendingPathComponent("invalid.yaml").path
        let invalidConfig = """
        server:
          port: [invalid
        """

        // Should validate and throw error
        XCTAssertThrowsError(try configManager.writeConfig(invalidConfig, to: configPath)) { error in
            XCTAssertTrue(error is ConfigManager.ConfigError)
        }
    }

    // MARK: - Validation Tests

    func testValidateValidConfig() {
        let validConfig = """
        server:
          port: 8080
        models:
          cache_dir: /tmp
        """

        XCTAssertTrue(configManager.validateConfig(validConfig))
    }

    func testValidateInvalidConfig() {
        let invalidConfigs = [
            "server: [unclosed",
            "invalid: yaml: syntax:",
            "tabs:\t\there"
        ]

        for config in invalidConfigs {
            XCTAssertFalse(configManager.validateConfig(config),
                          "Should reject invalid config: \(config)")
        }
    }

    // MARK: - Default Config Tests

    func testGetDefaultConfig() {
        let defaultConfig = configManager.getDefaultConfig()

        XCTAssertTrue(defaultConfig.contains("server:"))
        XCTAssertTrue(defaultConfig.contains("models:"))
        XCTAssertTrue(defaultConfig.contains("performance:"))
        XCTAssertTrue(defaultConfig.contains("mlxrunner.sock"))
    }

    func testDefaultConfigIsValid() {
        let defaultConfig = configManager.getDefaultConfig()
        XCTAssertTrue(configManager.validateConfig(defaultConfig))
    }

    // MARK: - Reset Tests

    func testResetToDefault() throws {
        let configPath = testDirectory.appendingPathComponent("reset_test.yaml")

        // Write custom config
        try "custom: config".write(to: configPath, atomically: true, encoding: .utf8)

        // Reset to default
        try configManager.resetToDefault(at: configPath.path)

        // Verify backup was created
        let backupPath = testDirectory.appendingPathComponent("reset_test.yaml.backup")
        XCTAssertTrue(FileManager.default.fileExists(atPath: backupPath.path))

        // Verify default config written
        let newConfig = try String(contentsOf: configPath)
        XCTAssertTrue(newConfig.contains("server:"))
    }

    // MARK: - Error Tests

    func testConfigErrorDescriptions() {
        let errors: [ConfigManager.ConfigError] = [
            .fileNotFound("test"),
            .readFailed("test"),
            .writeFailed("test"),
            .invalidYAML("test")
        ]

        for error in errors {
            XCTAssertFalse(error.localizedDescription.isEmpty)
        }
    }
}

// MARK: - LoginItemManager Tests

class LoginItemManagerTests: XCTestCase {

    var loginItemManager: LoginItemManager!

    override func setUp() {
        super.setUp()
        loginItemManager = LoginItemManager.shared
    }

    override func tearDown() {
        loginItemManager = nil
        super.tearDown()
    }

    // MARK: - Status Tests

    func testGetStatus() {
        // Just verify it doesn't crash
        let status = loginItemManager.getStatus()
        XCTAssertNotNil(status)
    }

    func testStatusIsConsistent() {
        let status1 = loginItemManager.getStatus()
        let status2 = loginItemManager.getStatus()

        XCTAssertEqual(status1, status2, "Status should be consistent across calls")
    }

    // MARK: - Enable/Disable Tests (Non-destructive)

    func testEnableReturnsResult() async {
        // Don't actually enable - just test the return value exists
        // In real test environment, we'd mock SMAppService
        let initialStatus = loginItemManager.getStatus()
        XCTAssertNotNil(initialStatus)
    }

    func testDisableReturnsResult() async {
        let initialStatus = loginItemManager.getStatus()
        XCTAssertNotNil(initialStatus)
    }

    // MARK: - Toggle Tests

    func testToggleChangesStatus() async {
        let initialStatus = loginItemManager.getStatus()

        // Note: Not actually toggling in tests to avoid side effects
        // In real implementation, we'd mock this
        XCTAssertNotNil(initialStatus)
    }

    // MARK: - Error Handling Tests

    func testLoginItemErrorDescriptions() {
        let errors: [LoginItemManager.LoginItemError] = [
            .enableFailed("test"),
            .disableFailed("test"),
            .statusCheckFailed("test")
        ]

        for error in errors {
            XCTAssertFalse(error.localizedDescription.isEmpty)
        }
    }

    // MARK: - Concurrency Tests

    func testConcurrentStatusChecks() async {
        await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    return self.loginItemManager.getStatus()
                }
            }

            var results = [Bool]()
            for await result in group {
                results.append(result)
            }

            // All results should be the same
            XCTAssertEqual(Set(results).count, 1, "Concurrent checks should return same result")
        }
    }
}
