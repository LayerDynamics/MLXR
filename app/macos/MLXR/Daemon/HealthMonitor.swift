//
//  HealthMonitor.swift
//  MLXR
//
//  Monitors daemon health status via health check endpoint.
//

import Foundation

class HealthMonitor {

    // MARK: - Properties

    private let socketClient: UnixSocketClient

    // MARK: - Initialization

    init() {
        self.socketClient = UnixSocketClient()
    }

    // MARK: - Health Check

    /// Check if daemon is healthy
    func checkHealth() async -> Bool {
        do {
            let _ = try await socketClient.request(
                method: "GET",
                path: "/health",
                headers: [:],
                body: nil
            )
            return true
        } catch {
            return false
        }
    }

    /// Get detailed health status
    func getHealthStatus() async -> HealthStatus {
        do {
            let response = try await socketClient.request(
                method: "GET",
                path: "/health",
                headers: [:],
                body: nil
            )

            // Parse health response
            if response.status == 200, let body = response.body {
                if let data = body.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    return parseHealthStatus(from: json)
                }
            }

            return HealthStatus(
                status: "unknown",
                uptime: 0,
                requests: 0,
                modelsLoaded: []
            )
        } catch {
            return HealthStatus(
                status: "error",
                uptime: 0,
                requests: 0,
                modelsLoaded: []
            )
        }
    }

    // MARK: - Parsing

    private func parseHealthStatus(from json: [String: Any]) -> HealthStatus {
        let status = json["status"] as? String ?? "unknown"
        let uptime = json["uptime"] as? Double ?? 0
        let requests = json["requests"] as? Int ?? 0
        let modelsLoaded = json["models_loaded"] as? [String] ?? []

        return HealthStatus(
            status: status,
            uptime: uptime,
            requests: requests,
            modelsLoaded: modelsLoaded
        )
    }
}

// MARK: - Types

struct HealthStatus {
    let status: String
    let uptime: TimeInterval
    let requests: Int
    let modelsLoaded: [String]

    var isHealthy: Bool {
        return status == "ok" || status == "healthy"
    }
}
