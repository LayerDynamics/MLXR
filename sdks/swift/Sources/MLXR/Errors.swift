// Copyright © 2025 MLXR Development
// Error types for the MLXR Swift SDK

import Foundation

/// Errors that can occur when using the MLXR client
public enum MLXRError: LocalizedError {
    /// Network or connection error
    case networkError(Error)

    /// Invalid URL or socket path
    case invalidURL(String)

    /// HTTP error with status code and message
    case httpError(statusCode: Int, message: String)

    /// Failed to encode request
    case encodingError(Error)

    /// Failed to decode response
    case decodingError(Error)

    /// Invalid response format
    case invalidResponse(String)

    /// Server returned an error
    case serverError(message: String, type: String, code: String?)

    /// Unix domain socket not available
    case socketNotAvailable(path: String)

    /// Streaming error
    case streamingError(String)

    /// Model not found
    case modelNotFound(String)

    /// Operation timeout
    case timeout

    /// Operation cancelled
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .httpError(let statusCode, let message):
            return "HTTP \(statusCode): \(message)"
        case .encodingError(let error):
            return "Encoding error: \(error.localizedDescription)"
        case .decodingError(let error):
            return "Decoding error: \(error.localizedDescription)"
        case .invalidResponse(let message):
            return "Invalid response: \(message)"
        case .serverError(let message, let type, let code):
            if let code = code {
                return "Server error [\(type)/\(code)]: \(message)"
            } else {
                return "Server error [\(type)]: \(message)"
            }
        case .socketNotAvailable(let path):
            return "Unix domain socket not available at: \(path)"
        case .streamingError(let message):
            return "Streaming error: \(message)"
        case .modelNotFound(let model):
            return "Model not found: \(model)"
        case .timeout:
            return "Operation timeout"
        case .cancelled:
            return "Operation cancelled"
        }
    }
}

/// Error response from the server (OpenAI format)
public struct ErrorResponse: Codable {
    public struct ErrorDetail: Codable {
        public let message: String
        public let type: String
        public let code: String?
    }

    public let error: ErrorDetail
}
