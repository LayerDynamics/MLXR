/**
 * MLXR TypeScript SDK
 *
 * OpenAI and Ollama-compatible client for MLXR inference engine
 */

// Export main client
export { MLXRClient } from './clients/mlxr-client';
export { OpenAIClient } from './clients/openai-client';
export { OllamaClient } from './clients/ollama-client';

// Export all types
export * from './types';

// Export utilities
export { HttpClient } from './utils/http-client';

// Re-export for convenience
import { MLXRClient } from './clients/mlxr-client';
export default MLXRClient;
