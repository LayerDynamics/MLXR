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
   * @param apis Array of APIs to check. Supported values: 'openai', 'ollama'. Defaults to ['openai'].
   */
  async health(
    apis: Array<'openai' | 'ollama'> = ['openai']
  ): Promise<{ status: string; details: Record<string, string> }> {
    const details: Record<string, string> = {};
    let overallStatus = 'ok';

    for (const api of apis) {
      try {
        if (api === 'openai') {
          await this.openai.listModels();
          details['openai'] = 'ok';
        } else if (api === 'ollama') {
          await this.ollama.tags();
          details['ollama'] = 'ok';
        }
      } catch (err) {
        details[api] = `error: ${err instanceof Error ? err.message : String(err)}`;
        overallStatus = 'error';
      }
    }

    return { status: overallStatus, details };
  }
}
