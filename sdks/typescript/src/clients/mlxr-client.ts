/**
 * Main MLXR client that combines OpenAI and Ollama APIs
 */

import { OpenAIClient } from './openai-client';
import { OllamaClient } from './ollama-client';
import type { MLXRConfig } from '../types';

/**
 * Unified MLXR client providing both OpenAI and Ollama APIs
 */
export class MLXRClient {
  /**
   * OpenAI-compatible API client
   */
  public readonly openai: OpenAIClient;

  /**
   * Ollama-compatible API client
   */
  public readonly ollama: OllamaClient;

  constructor(config?: MLXRConfig) {
    this.openai = new OpenAIClient(config);
    this.ollama = new OllamaClient(config);
  }

  /**
   * Update configuration for both clients
   */
  updateConfig(config: Partial<MLXRConfig>): void {
    this.openai.updateConfig(config);
    this.ollama.updateConfig(config);
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<MLXRConfig> {
    return this.openai.getConfig();
  }

  /**
   * Check if Unix socket is available
   */
  isUnixSocketAvailable(): boolean {
    return this.openai.isUnixSocketAvailable();
  }

  /**
   * Health check endpoint
   */
  async health(): Promise<{ status: string }> {
    try {
      // Try to list models as a health check
      await this.openai.listModels();
      return { status: 'ok' };
    } catch (err) {
      throw new Error(`Health check failed: ${err}`);
    }
  }
}
