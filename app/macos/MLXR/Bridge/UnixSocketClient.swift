//
//  UnixSocketClient.swift
//  MLXR
//
//  HTTP client that communicates with daemon over Unix Domain Socket.
//

import Foundation

class UnixSocketClient {

    // MARK: - Properties

    private let socketPath: String
    private let timeout: TimeInterval = 30.0

    // MARK: - Initialization

    init() {
        // Default socket path
        let home = FileManager.default.homeDirectoryForCurrentUser
        self.socketPath = home.appendingPathComponent("Library/Application Support/MLXRunner/run/mlxrunner.sock").path
    }

    init(socketPath: String) {
        self.socketPath = socketPath
    }

    // MARK: - Request

    func request(method: String, path: String, headers: [String: String], body: String?) async throws -> BridgeResponse {
        // Check if socket exists
        guard FileManager.default.fileExists(atPath: socketPath) else {
            throw BridgeError.daemonNotRunning
        }

        // Create socket
        let socketFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw NSError(domain: "UnixSocketClient", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to create socket"
            ])
        }

        defer { close(socketFD) }

        // Set socket timeout
        var tv = timeval()
        tv.tv_sec = Int(timeout)
        tv.tv_usec = 0
        setsockopt(socketFD, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(socketFD, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        // Connect to socket
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)

        let pathLength = min(socketPath.utf8.count, MemoryLayout.size(ofValue: addr.sun_path) - 1)
        _ = withUnsafeMutablePointer(to: &addr.sun_path.0) { ptr in
            socketPath.withCString { cString in
                strncpy(UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self), cString, pathLength)
            }
        }

        let connectResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(socketFD, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }

        guard connectResult >= 0 else {
            throw NSError(domain: "UnixSocketClient", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Failed to connect to daemon socket"
            ])
        }

        // Build HTTP request
        let httpRequest = buildHTTPRequest(method: method, path: path, headers: headers, body: body)

        // Send request
        try httpRequest.withCString { cString in
            let length = strlen(cString)
            let sent = send(socketFD, cString, length, 0)
            if sent < 0 {
                throw NSError(domain: "UnixSocketClient", code: -3, userInfo: [
                    NSLocalizedDescriptionKey: "Failed to send request"
                ])
            }
        }

        // Read response
        let responseData = try readResponse(from: socketFD)

        // Parse HTTP response
        return try parseHTTPResponse(responseData)
    }

    // MARK: - HTTP Building

    private func buildHTTPRequest(method: String, path: String, headers: [String: String], body: String?) -> String {
        var request = "\(method) \(path) HTTP/1.1\r\n"
        request += "Host: localhost\r\n"
        request += "Connection: close\r\n"

        // Add headers
        for (key, value) in headers {
            request += "\(key): \(value)\r\n"
        }

        // Add body if present
        if let body = body {
            let bodyData = body.data(using: .utf8) ?? Data()
            request += "Content-Length: \(bodyData.count)\r\n"
            request += "\r\n"
            request += body
        } else {
            request += "\r\n"
        }

        return request
    }

    // MARK: - Response Reading

    private func readResponse(from socket: Int32) throws -> Data {
        var responseData = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)

        while true {
            let bytesRead = recv(socket, &buffer, buffer.count, 0)

            if bytesRead < 0 {
                throw NSError(domain: "UnixSocketClient", code: -4, userInfo: [
                    NSLocalizedDescriptionKey: "Failed to read response"
                ])
            }

            if bytesRead == 0 {
                // Connection closed
                break
            }

            responseData.append(contentsOf: buffer[..<bytesRead])
        }

        return responseData
    }

    // MARK: - HTTP Parsing

    private func parseHTTPResponse(_ data: Data) throws -> BridgeResponse {
        guard let responseString = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "UnixSocketClient", code: -5, userInfo: [
                NSLocalizedDescriptionKey: "Invalid response encoding"
            ])
        }

        // Split headers and body
        let parts = responseString.components(separatedBy: "\r\n\r\n")
        guard parts.count >= 1 else {
            throw NSError(domain: "UnixSocketClient", code: -6, userInfo: [
                NSLocalizedDescriptionKey: "Invalid HTTP response format"
            ])
        }

        let headerSection = parts[0]
        let body = parts.count > 1 ? parts[1...].joined(separator: "\r\n\r\n") : ""

        // Parse status line
        let lines = headerSection.components(separatedBy: "\r\n")
        guard let statusLine = lines.first else {
            throw NSError(domain: "UnixSocketClient", code: -7, userInfo: [
                NSLocalizedDescriptionKey: "Missing status line"
            ])
        }

        let statusParts = statusLine.components(separatedBy: " ")
        guard statusParts.count >= 3,
              let status = Int(statusParts[1]) else {
            throw NSError(domain: "UnixSocketClient", code: -8, userInfo: [
                NSLocalizedDescriptionKey: "Invalid status line"
            ])
        }

        let statusText = statusParts[2...].joined(separator: " ")

        // Parse headers
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let headerParts = line.components(separatedBy: ": ")
            if headerParts.count == 2 {
                headers[headerParts[0]] = headerParts[1]
            }
        }

        return BridgeResponse(
            status: status,
            statusText: statusText,
            headers: headers,
            body: body
        )
    }
}
