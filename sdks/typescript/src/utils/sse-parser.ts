/**
 * SSE (Server-Sent Events) parsing utilities
 */

export interface SSEChunk {
  data: string;
  done: boolean;
}

/**
 * Parse SSE stream from HTTP response
 * Handles both Unix (\n) and Windows (\r\n) line endings
 */
export async function* parseSSEStream(
  response: AsyncIterable<Buffer>
): AsyncGenerator<SSEChunk, void, unknown> {
  let buffer = '';

  for await (const chunk of response) {
    buffer += chunk.toString();
    // Split on both \n and \r\n to support all line endings
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          yield { data: '', done: true };
          return;
        }
        if (data) {
          yield { data, done: false };
        }
      }
    }
  }

  // Process any remaining data in buffer
  if (buffer.startsWith('data: ')) {
    const data = buffer.slice(6).trim();
    if (data && data !== '[DONE]') {
      yield { data, done: false };
    }
  }
}

/**
 * Parse and yield JSON objects from SSE stream
 */
export async function* parseJSONStream<T>(
  sseStream: AsyncGenerator<SSEChunk, void, unknown>,
  onParseError?: (error: Error, data: string) => void
): AsyncGenerator<T, void, unknown> {
  for await (const chunk of sseStream) {
    if (chunk.done) {
      break;
    }
    try {
      const parsed = JSON.parse(chunk.data);
      yield parsed as T;
    } catch (err) {
      if (onParseError) {
        onParseError(err as Error, chunk.data);
      }
      // Propagate error if no handler provided
      if (!onParseError) {
        throw new Error(`Failed to parse SSE chunk: ${err}`);
      }
    }
  }
}
