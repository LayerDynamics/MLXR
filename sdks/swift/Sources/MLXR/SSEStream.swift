// Copyright © 2025 MLXR Development
// Server-Sent Events (SSE) streaming support

import Foundation

/// SSE event parser
struct SSEParser {
    /// Parse SSE events from a stream, returning events and the number of bytes consumed
    static func parseEvents(from data: Data) -> (events: [SSEEvent], bytesConsumed: Int) {
        guard let text = String(data: data, encoding: .utf8) else {
            return ([], 0)
        }

        var events: [SSEEvent] = []
        var currentEvent = SSEEvent()
        var position = 0
        var lastCompleteEventEnd = 0

        let lines = text.components(separatedBy: .newlines)

        for line in lines {
            let lineLength = line.utf8.count + 1  // +1 for newline

            if line.isEmpty {
                // Empty line marks end of event
                if !currentEvent.isEmpty {
                    events.append(currentEvent)
                    currentEvent = SSEEvent()
                    lastCompleteEventEnd = position + lineLength
                }
                position += lineLength
                continue
            }

            if line.hasPrefix(":") {
                // Comment line, skip
                position += lineLength
                continue
            }

            guard let colonIndex = line.firstIndex(of: ":") else {
                // Invalid line, skip
                position += lineLength
                continue
            }

            let fieldName = String(line[..<colonIndex])
            var fieldValue = String(line[line.index(after: colonIndex)...])

            // Remove leading space after colon
            if fieldValue.hasPrefix(" ") {
                fieldValue = String(fieldValue.dropFirst())
            }

            switch fieldName {
            case "event":
                currentEvent.event = fieldValue
            case "data":
                if currentEvent.data == nil {
                    currentEvent.data = fieldValue
                } else {
                    currentEvent.data! += "\n" + fieldValue
                }
            case "id":
                currentEvent.id = fieldValue
            case "retry":
                if let retryMs = Int(fieldValue) {
                    currentEvent.retry = retryMs
                }
            default:
                break
            }

            position += lineLength
        }

        // Only return complete events and the position up to the last complete event
        return (events, lastCompleteEventEnd)
    }
}

/// SSE event structure
struct SSEEvent {
    var event: String?
    var data: String?
    var id: String?
    var retry: Int?

    var isEmpty: Bool {
        return event == nil && data == nil && id == nil && retry == nil
    }
}

/// Stream SSE events and decode JSON
public struct SSEStream<T: Decodable> {
    private let dataStream: AsyncThrowingStream<Data, Error>
    private let decoder: JSONDecoder

    init(dataStream: AsyncThrowingStream<Data, Error>, decoder: JSONDecoder = JSONDecoder()) {
        self.dataStream = dataStream
        self.decoder = decoder
    }

    /// Create an async sequence of decoded events
    public func events() -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var buffer = Data()

                do {
                    for try await chunk in dataStream {
                        buffer.append(chunk)

                        // Process complete SSE events
                        let (events, bytesConsumed) = SSEParser.parseEvents(from: buffer)

                        for event in events {
                            // Handle special [DONE] marker for OpenAI
                            if event.data == "[DONE]" {
                                continuation.finish()
                                return
                            }

                            guard let data = event.data else {
                                continue
                            }

                            // Try to decode the data as JSON
                            guard let jsonData = data.data(using: .utf8) else {
                                continue
                            }

                            do {
                                let decoded = try decoder.decode(T.self, from: jsonData)
                                continuation.yield(decoded)
                            } catch {
                                // Skip invalid JSON chunks - may be partial data
                                continue
                            }
                        }

                        // Only remove processed data from buffer, keep partial events
                        if bytesConsumed > 0 {
                            buffer.removeFirst(bytesConsumed)
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: MLXRError.streamingError(error.localizedDescription))
                }
            }
        }
    }
}

/// Raw SSE stream without decoding
public struct RawSSEStream {
    private let dataStream: AsyncThrowingStream<Data, Error>

    init(dataStream: AsyncThrowingStream<Data, Error>) {
        self.dataStream = dataStream
    }

    /// Create an async sequence of raw SSE events
    public func events() -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var buffer = Data()

                do {
                    for try await chunk in dataStream {
                        buffer.append(chunk)

                        let (events, bytesConsumed) = SSEParser.parseEvents(from: buffer)

                        for event in events {
                            continuation.yield(event)
                        }

                        // Only remove processed data from buffer, keep partial events
                        if bytesConsumed > 0 {
                            buffer.removeFirst(bytesConsumed)
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: MLXRError.streamingError(error.localizedDescription))
                }
            }
        }
    }
}
