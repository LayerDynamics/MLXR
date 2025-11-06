//
//  LaunchdManager.swift
//  MLXR
//
//  Manages launchd agent for running daemon as background service.
//

import Foundation

class LaunchdManager {

    // MARK: - Properties

    private let fileManager = FileManager.default
    private let agentLabel = "com.mlxr.mlxrunnerd"

    private var agentPlistPath: URL {
        let home = fileManager.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Library/LaunchAgents/\(agentLabel).plist")
    }

    private var logsDir: URL {
        let home = fileManager.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Library/Logs/")
    }

    // MARK: - Installation

    /// Check if agent is installed
    func isAgentInstalled() async throws -> Bool {
        return fileManager.fileExists(atPath: agentPlistPath.path)
    }

    /// Install launchd agent
    func installAgent(daemonPath: URL, configPath: URL) async throws {
        print("[LaunchdManager] Installing launchd agent...")

        // Create LaunchAgents directory if needed
        let launchAgentsDir = agentPlistPath.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: launchAgentsDir.path) {
            try fileManager.createDirectory(at: launchAgentsDir, withIntermediateDirectories: true)
        }

        // Create logs directory if needed
        if !fileManager.fileExists(atPath: logsDir.path) {
            try fileManager.createDirectory(at: logsDir, withIntermediateDirectories: true)
        }

        // Generate plist content
        let plist = generatePlist(daemonPath: daemonPath, configPath: configPath)

        // Write plist
        try plist.write(to: agentPlistPath, atomically: true, encoding: .utf8)

        print("[LaunchdManager] Agent plist created at: \(agentPlistPath.path)")
    }

    /// Uninstall launchd agent
    func uninstallAgent() async throws {
        print("[LaunchdManager] Uninstalling launchd agent...")

        // Stop agent first
        try await stopAgent()

        // Remove plist
        if fileManager.fileExists(atPath: agentPlistPath.path) {
            try fileManager.removeItem(at: agentPlistPath)
        }

        print("[LaunchdManager] Agent uninstalled")
    }

    // MARK: - Control

    /// Start launchd agent
    func startAgent() async throws {
        print("[LaunchdManager] Starting agent...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["load", "-w", agentPlistPath.path]

        let pipe = Pipe()
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let error = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw LaunchdError.startFailed(error)
        }

        print("[LaunchdManager] Agent started")
    }

    /// Stop launchd agent
    func stopAgent() async throws {
        print("[LaunchdManager] Stopping agent...")

        guard try await isAgentInstalled() else {
            print("[LaunchdManager] Agent not installed, nothing to stop")
            return
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["unload", "-w", agentPlistPath.path]

        let pipe = Pipe()
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        // Exit code 0 = success, 3 = already stopped (acceptable)
        if process.terminationStatus != 0 && process.terminationStatus != 3 {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let error = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw LaunchdError.stopFailed(error)
        }

        print("[LaunchdManager] Agent stopped")
    }

    /// Get agent PID
    func getAgentPID() async throws -> Int? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["list", agentLabel]

        let pipe = Pipe()
        process.standardOutput = pipe

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            return nil
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let output = String(data: data, encoding: .utf8) else {
            return nil
        }

        // Parse PID from output
        // Format: "PID    Status    Label"
        let lines = output.components(separatedBy: "\n")
        for line in lines {
            if line.contains(agentLabel) {
                let components = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                if let pidString = components.first, let pid = Int(pidString) {
                    return pid
                }
            }
        }

        return nil
    }

    // MARK: - Plist Generation

    private func generatePlist(daemonPath: URL, configPath: URL) -> String {
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>\(agentLabel)</string>
            <key>ProgramArguments</key>
            <array>
                <string>\(daemonPath.path)</string>
                <string>--config</string>
                <string>\(configPath.path)</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>\(logsDir.path)/mlxrunnerd.out.log</string>
            <key>StandardErrorPath</key>
            <string>\(logsDir.path)/mlxrunnerd.err.log</string>
            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
            </dict>
        </dict>
        </plist>
        """
    }
}

// MARK: - Errors

enum LaunchdError: Error, LocalizedError {
    case startFailed(String)
    case stopFailed(String)
    case notInstalled

    var errorDescription: String? {
        switch self {
        case .startFailed(let error):
            return "Failed to start agent: \(error)"
        case .stopFailed(let error):
            return "Failed to stop agent: \(error)"
        case .notInstalled:
            return "Agent is not installed"
        }
    }
}
