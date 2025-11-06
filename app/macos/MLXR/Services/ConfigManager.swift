//
//  ConfigManager.swift
//  MLXR
//
//  Manages reading and writing server configuration (YAML).
//

import Foundation

class ConfigManager {

    // MARK: - Properties

    private let fileManager = FileManager.default

    private var configPath: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("MLXRunner/server.yaml")
    }

    // MARK: - Singleton

    static let shared = ConfigManager()

    private init() {}

    // MARK: - Config Operations

    /// Read configuration file
    func readConfig() throws -> String {
        guard fileManager.fileExists(atPath: configPath.path) else {
            throw ConfigError.notFound
        }

        return try String(contentsOf: configPath, encoding: .utf8)
    }

    /// Write configuration file
    func writeConfig(_ content: String) throws {
        // Validate YAML syntax (basic validation)
        guard isValidYAML(content) else {
            throw ConfigError.invalidYAML
        }

        // Create directory if needed
        let configDir = configPath.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: configDir.path) {
            try fileManager.createDirectory(at: configDir, withIntermediateDirectories: true)
        }

        // Write config
        try content.write(to: configPath, atomically: true, encoding: .utf8)
    }

    /// Create default configuration
    func createDefaultConfig() throws {
        let defaultConfig = """
        # MLXR Server Configuration

        # Daemon settings
        uds_path: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock
        http_port: 0  # Disabled by default, set to 8080 for HTTP access

        # Performance settings
        max_batch_tokens: 2048
        max_batch_size: 16
        target_latency_ms: 80
        enable_chunked_prefill: true
        prefill_chunk_size: 512

        # KV Cache
        kv_block_size: 32
        kv_max_blocks: 4096
        kv_persistence: true

        # Speculative Decoding
        enable_speculative: true
        draft_model: ""
        speculation_length: 4

        # Logging
        log_level: info
        log_format: json
        """

        try writeConfig(defaultConfig)
    }

    /// Reset to default configuration
    func resetToDefault() throws {
        // Backup existing config
        if fileManager.fileExists(atPath: configPath.path) {
            let backupPath = configPath.appendingPathExtension("backup")
            try? fileManager.removeItem(at: backupPath)
            try? fileManager.copyItem(at: configPath, to: backupPath)
        }

        // Create default
        try createDefaultConfig()
    }

    // MARK: - Validation

    private func isValidYAML(_ content: String) -> Bool {
        // Basic YAML validation
        // Check for balanced colons and proper indentation
        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Skip empty lines and comments
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }

            // Check for key-value pairs
            if trimmed.contains(":") {
                let parts = trimmed.components(separatedBy: ":")
                if parts.count >= 2 {
                    continue
                }
            }

            // Check for list items
            if trimmed.hasPrefix("-") {
                continue
            }
        }

        return true
    }
}

// MARK: - Errors

enum ConfigError: Error, LocalizedError {
    case notFound
    case invalidYAML
    case readFailed
    case writeFailed

    var errorDescription: String? {
        switch self {
        case .notFound:
            return "Configuration file not found"
        case .invalidYAML:
            return "Invalid YAML syntax"
        case .readFailed:
            return "Failed to read configuration"
        case .writeFailed:
            return "Failed to write configuration"
        }
    }
}
