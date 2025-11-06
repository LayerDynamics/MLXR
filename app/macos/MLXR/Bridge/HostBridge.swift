//
//  HostBridge.swift
//  MLXR
//
//  Protocol defining the bridge interface between JavaScript and Swift.
//  Matches the TypeScript interface in app/ui/src/types/bridge.ts
//

import Foundation

/// Protocol for handling bridge messages from JavaScript
protocol HostBridge {

    /// Make an HTTP request to the daemon via Unix Domain Socket
    /// - Parameters:
    ///   - path: Request path (e.g., '/v1/models')
    ///   - init: Request options (method, headers, body)
    /// - Returns: Response with status, headers, and body
    func request(path: String, init: [String: Any]?) async throws -> BridgeResponse

    /// Open file/folder picker dialog
    /// - Parameter type: 'models' or 'cache'
    /// - Returns: Selected path or nil if cancelled
    func openPathDialog(type: String) async throws -> String?

    /// Read server configuration
    /// - Returns: YAML config content
    func readConfig() async throws -> String

    /// Write server configuration
    /// - Parameter yaml: YAML config content
    func writeConfig(yaml: String) async throws

    /// Start the daemon
    func startDaemon() async throws

    /// Stop the daemon
    func stopDaemon() async throws

    /// Get app and daemon versions
    /// - Returns: Dictionary with 'app' and 'daemon' version strings
    func getVersion() async throws -> [String: String]
}

// MARK: - Bridge Types

/// Response from bridge request
struct BridgeResponse: Codable {
    let status: Int
    let statusText: String?
    let headers: [String: String]?
    let body: String?
}

/// Bridge message from JavaScript
struct BridgeMessage: Codable {
    let id: Int
    let method: String
    let params: [String: AnyCodable]?
}

/// Bridge response to JavaScript
struct BridgeResult: Codable {
    let id: Int
    let error: String?
    let result: AnyCodable?
}

// MARK: - AnyCodable Helper

/// Type-erased Codable wrapper for handling dynamic JSON
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else if container.decodeNil() {
            value = NSNull()
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "AnyCodable cannot decode value"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let bool as Bool:
            try container.encode(bool)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        case is NSNull:
            try container.encodeNil()
        default:
            throw EncodingError.invalidValue(
                value,
                EncodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription: "AnyCodable cannot encode value"
                )
            )
        }
    }
}

// MARK: - Bridge Errors

enum BridgeError: Error, LocalizedError {
    case invalidMessage
    case invalidMethod(String)
    case invalidParams
    case daemonNotRunning
    case requestFailed(String)
    case filePickerCancelled

    var errorDescription: String? {
        switch self {
        case .invalidMessage:
            return "Invalid bridge message format"
        case .invalidMethod(let method):
            return "Invalid bridge method: \(method)"
        case .invalidParams:
            return "Invalid bridge parameters"
        case .daemonNotRunning:
            return "Daemon is not running"
        case .requestFailed(let error):
            return "Request failed: \(error)"
        case .filePickerCancelled:
            return "File picker was cancelled"
        }
    }
}
