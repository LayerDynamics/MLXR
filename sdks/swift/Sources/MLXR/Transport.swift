// Copyright © 2025 MLXR Development
// Transport layer for communicating with the MLXR daemon

import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// Protocol for transport implementations
protocol Transport {
    func request(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?,
        timeout: TimeInterval
    ) async throws -> (Data, HTTPURLResponse)

    func streamRequest(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?,
        timeout: TimeInterval
    ) async throws -> AsyncThrowingStream<Data, Error>
}

/// HTTP transport using URLSession
final class HTTPTransport: Transport {
    private let baseURL: URL
    private let session: URLSession

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    func request(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?,
        timeout: TimeInterval
    ) async throws -> (Data, HTTPURLResponse) {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw MLXRError.invalidURL("\(baseURL)\(path)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = timeout

        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }

        if let body = body {
            request.httpBody = body
        }

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw MLXRError.invalidResponse("Not an HTTP response")
            }

            // Handle HTTP errors
            if httpResponse.statusCode >= 400 {
                let message = String(data: data, encoding: .utf8) ?? "Unknown error"
                throw MLXRError.httpError(statusCode: httpResponse.statusCode, message: message)
            }

            return (data, httpResponse)
        } catch let error as MLXRError {
            throw error
        } catch {
            throw MLXRError.networkError(error)
        }
    }

    func streamRequest(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?,
        timeout: TimeInterval
    ) async throws -> AsyncThrowingStream<Data, Error> {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw MLXRError.invalidURL("\(baseURL)\(path)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = timeout

        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }

        if let body = body {
            request.httpBody = body
        }

        let (bytes, response) = try await session.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw MLXRError.invalidResponse("Not an HTTP response")
        }

        if httpResponse.statusCode >= 400 {
            throw MLXRError.httpError(statusCode: httpResponse.statusCode, message: "HTTP \(httpResponse.statusCode)")
        }

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await byte in bytes {
                        var data = Data()
                        data.append(byte)
                        continuation.yield(data)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: MLXRError.networkError(error))
                }
            }
        }
    }
}

#if canImport(Darwin)
/// Unix Domain Socket transport (macOS only)
final class UnixSocketTransport: Transport {
    private let socketPath: String

    init(socketPath: String) {
        self.socketPath = socketPath
    }

    func request(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?,
        timeout: TimeInterval
    ) async throws -> (Data, HTTPURLResponse) {
        let socket = try createSocket()
        defer { close(socket) }

        try connectSocket(socket)

        // Build HTTP request
        let httpRequest = buildHTTPRequest(method: method, path: path, headers: headers, body: body)
        try sendData(socket: socket, data: httpRequest)

        // Read response
        let responseData = try receiveData(socket: socket)
        let (response, body) = try parseHTTPResponse(responseData)

        return (body, response)
    }

    func streamRequest(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?,
        timeout: TimeInterval
    ) async throws -> AsyncThrowingStream<Data, Error> {
        let socket = try createSocket()

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try self.connectSocket(socket)

                    // Build and send HTTP request
                    let httpRequest = self.buildHTTPRequest(
                        method: method,
                        path: path,
                        headers: headers,
                        body: body
                    )
                    try self.sendData(socket: socket, data: httpRequest)

                    // Read response headers
                    var buffer = Data()
                    let headerEndMarker = "\r\n\r\n".data(using: .utf8)!

                    // Read until we find end of headers
                    while !buffer.contains(headerEndMarker) {
                        let chunk = try self.receiveChunk(socket: socket)
                        buffer.append(chunk)
                    }

                    // Parse headers and skip to body
                    guard let headerEndRange = buffer.range(of: headerEndMarker) else {
                        throw MLXRError.invalidResponse("Invalid HTTP response headers")
                    }

                    let bodyStart = headerEndRange.upperBound
                    var bodyBuffer = buffer[bodyStart...]

                    // Stream body chunks
                    while true {
                        if !bodyBuffer.isEmpty {
                            continuation.yield(Data(bodyBuffer))
                            bodyBuffer.removeAll()
                        }

                        do {
                            let chunk = try self.receiveChunk(socket: socket)
                            if chunk.isEmpty { break }
                            bodyBuffer.append(chunk)
                        } catch {
                            break
                        }
                    }

                    continuation.finish()
                    close(socket)
                } catch {
                    close(socket)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private helpers

    private func createSocket() throws -> Int32 {
        let socket = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard socket >= 0 else {
            throw MLXRError.socketNotAvailable(socketPath)
        }
        return socket
    }

    private func connectSocket(_ socket: Int32) throws {
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)

        let pathCString = (socketPath as NSString).utf8String!
        let pathLength = Int(strlen(pathCString))

        guard pathLength < MemoryLayout.size(ofValue: addr.sun_path) else {
            throw MLXRError.socketNotAvailable(socketPath)
        }

        // Use memcpy for safe and portable assignment, ensure null-termination
        let sunPathSize = MemoryLayout.size(ofValue: addr.sun_path)
        withUnsafeMutableBytes(of: &addr.sun_path) { sunPathPtr in
            // Copy up to sunPathSize - 1 to leave space for null terminator
            let copyLength = min(pathLength, sunPathSize - 1)
            memcpy(sunPathPtr.baseAddress, pathCString, copyLength)
            // Ensure null-termination
            sunPathPtr[copyLength] = 0
        }

        let connectResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(socket, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }

        guard connectResult >= 0 else {
            throw MLXRError.socketNotAvailable(socketPath)
        }
    }

    private func buildHTTPRequest(
        method: String,
        path: String,
        headers: [String: String],
        body: Data?
    ) -> Data {
        var request = "\(method) \(path) HTTP/1.1\r\n"

        var allHeaders = headers
        if let body = body {
            allHeaders["Content-Length"] = "\(body.count)"
        }
        allHeaders["Host"] = "localhost"
        allHeaders["Connection"] = "close"

        for (key, value) in allHeaders {
            request += "\(key): \(value)\r\n"
        }

        request += "\r\n"

        var data = request.data(using: .utf8)!
        if let body = body {
            data.append(body)
        }

        return data
    }

    private func sendData(socket: Int32, data: Data) throws {
        let sent = data.withUnsafeBytes { bufferPtr in
            Darwin.send(socket, bufferPtr.baseAddress, data.count, 0)
        }

        guard sent == data.count else {
            throw MLXRError.networkError(NSError(domain: "UnixSocket", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to send data"
            ]))
        }
    }

    private func receiveData(socket: Int32) throws -> Data {
        var buffer = Data()
        var chunk = Data(count: 4096)

        while true {
            let received = chunk.withUnsafeMutableBytes { bufferPtr in
                Darwin.recv(socket, bufferPtr.baseAddress, chunk.count, 0)
            }

            guard received > 0 else { break }
            buffer.append(chunk.prefix(received))
        }

        return buffer
    }

    private func receiveChunk(socket: Int32) throws -> Data {
        var chunk = Data(count: 4096)

        let received = chunk.withUnsafeMutableBytes { bufferPtr in
            Darwin.recv(socket, bufferPtr.baseAddress, chunk.count, 0)
        }

        guard received >= 0 else {
            throw MLXRError.networkError(NSError(domain: "UnixSocket", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to receive data"
            ]))
        }

        return chunk.prefix(received)
    }

    private func parseHTTPResponse(_ data: Data) throws -> (HTTPURLResponse, Data) {
        guard let responseString = String(data: data, encoding: .utf8) else {
            throw MLXRError.invalidResponse("Invalid UTF-8 in response")
        }

        let lines = responseString.components(separatedBy: "\r\n")
        guard !lines.isEmpty else {
            throw MLXRError.invalidResponse("Empty response")
        }

        // Parse status line
        let statusLine = lines[0]
        let statusParts = statusLine.components(separatedBy: " ")
        guard statusParts.count >= 3,
              let statusCode = Int(statusParts[1]) else {
            throw MLXRError.invalidResponse("Invalid status line: \(statusLine)")
        }

        // Parse headers
        var headerDict: [String: String] = [:]
        var bodyStartIndex = 0

        for (index, line) in lines.enumerated().dropFirst() {
            if line.isEmpty {
                bodyStartIndex = index + 1
                break
            }

            let headerParts = line.components(separatedBy: ": ")
            if headerParts.count == 2 {
                headerDict[headerParts[0]] = headerParts[1]
            }
        }

        // Extract body
        let bodyLines = lines[bodyStartIndex...]
        let bodyString = bodyLines.joined(separator: "\r\n")
        let bodyData = bodyString.data(using: .utf8) ?? Data()

        // Create HTTPURLResponse
        guard let url = URL(string: "http://localhost") else {
            throw MLXRError.invalidResponse("Failed to create URL")
        }

        guard let response = HTTPURLResponse(
            url: url,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: headerDict
        ) else {
            throw MLXRError.invalidResponse("Failed to create HTTPURLResponse")
        }

        // Handle HTTP errors
        if statusCode >= 400 {
            let message = String(data: bodyData, encoding: .utf8) ?? "Unknown error"
            throw MLXRError.httpError(statusCode: statusCode, message: message)
        }

        return (response, bodyData)
    }
}
#endif
