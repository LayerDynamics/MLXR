//
//  DaemonManager.swift
//  MLXR
//
//  Manages daemon lifecycle (start, stop, restart, status).
//

import Foundation

class DaemonManager {

    // MARK: - Singleton

    static let shared = DaemonManager()

    // MARK: - Properties

    private let fileManager = FileManager.default
    private let launchdManager: LaunchdManager
    private let healthMonitor: HealthMonitor

    private var daemonPath: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("MLXRunner/bin/mlxrunnerd")
    }

    private var configPath: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("MLXRunner/server.yaml")
    }

    private var socketPath: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("MLXRunner/run/mlxrunner.sock")
    }

    // MARK: - Initialization

    private init() {
        self.launchdManager = LaunchdManager()
        self.healthMonitor = HealthMonitor()
    }

    // MARK: - Daemon Lifecycle

    /// Check if daemon is running
    func isDaemonRunning() async throws -> Bool {
        // Check if socket exists
        guard fileManager.fileExists(atPath: socketPath.path) else {
            return false
        }

        // Try health check
        return await healthMonitor.checkHealth()
    }

    /// Start daemon
    func startDaemon() async throws {
        print("[DaemonManager] Starting daemon...")

        // Check if already running
        if try await isDaemonRunning() {
            print("[DaemonManager] Daemon already running")
            return
        }

        // Ensure daemon binary exists
        guard fileManager.fileExists(atPath: daemonPath.path) else {
            throw DaemonError.binaryNotFound
        }

        // Ensure config exists
        if !fileManager.fileExists(atPath: configPath.path) {
            try createDefaultConfig()
        }

        // Install launchd agent if not already installed
        if !(try await launchdManager.isAgentInstalled()) {
            try await launchdManager.installAgent(daemonPath: daemonPath, configPath: configPath)
        }

        // Start agent
        try await launchdManager.startAgent()

        // Wait for daemon to be ready (up to 5 seconds)
        var attempts = 0
        while attempts < 10 {
            if try await isDaemonRunning() {
                print("[DaemonManager] Daemon started successfully")
                return
            }
            try await Task.sleep(nanoseconds: 500_000_000) // 500ms
            attempts += 1
        }

        throw DaemonError.startFailed
    }

    /// Stop daemon
    func stopDaemon() async throws {
        print("[DaemonManager] Stopping daemon...")

        // Stop launchd agent
        try await launchdManager.stopAgent()

        // Wait for daemon to stop (up to 5 seconds)
        var attempts = 0
        while attempts < 10 {
            if !(try await isDaemonRunning()) {
                print("[DaemonManager] Daemon stopped successfully")
                return
            }
            try await Task.sleep(nanoseconds: 500_000_000) // 500ms
            attempts += 1
        }

        print("[DaemonManager] Warning: Daemon may not have stopped cleanly")
    }

    /// Restart daemon
    func restartDaemon() async throws {
        print("[DaemonManager] Restarting daemon...")
        try await stopDaemon()
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        try await startDaemon()
    }

    /// Get daemon status
    func getStatus() async -> DaemonStatus {
        let isRunning = (try? await isDaemonRunning()) ?? false

        if isRunning {
            let health = await healthMonitor.getHealthStatus()
            return DaemonStatus(
                isRunning: true,
                pid: try? await launchdManager.getAgentPID(),
                health: health
            )
        } else {
            return DaemonStatus(isRunning: false, pid: nil, health: nil)
        }
    }

    // MARK: - Configuration

    private func createDefaultConfig() throws {
        let defaultConfig = """
        # MLXR Server Configuration

        # Daemon settings
        uds_path: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock
        http_port: 0  # Disabled by default

        # Performance settings
        max_batch_tokens: 2048
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
        """

        try defaultConfig.write(to: configPath, atomically: true, encoding: .utf8)
    }
}

// MARK: - Types

struct DaemonStatus {
    let isRunning: Bool
    let pid: Int?
    let health: HealthStatus?
}

// MARK: - Errors

enum DaemonError: Error, LocalizedError {
    case binaryNotFound
    case configNotFound
    case startFailed
    case stopFailed
    case notRunning

    var errorDescription: String? {
        switch self {
        case .binaryNotFound:
            return "Daemon binary not found. Please reinstall MLXR."
        case .configNotFound:
            return "Configuration file not found."
        case .startFailed:
            return "Failed to start daemon. Check logs for details."
        case .stopFailed:
            return "Failed to stop daemon."
        case .notRunning:
            return "Daemon is not running."
        }
    }
}
