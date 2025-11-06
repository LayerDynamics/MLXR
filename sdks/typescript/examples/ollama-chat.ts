/**
 * Example: Ollama-compatible chat
 */

import { MLXRClient } from '../src';

async function main() {
  const client = new MLXRClient();

  console.log('MLXR TypeScript SDK - Ollama Chat Example\n');

  // Non-streaming chat
  console.log('1. Non-streaming chat:');
  try {
    const response = await client.ollama.chat({
      model: 'tinyllama',
      messages: [
        { role: 'system', content: 'You are a helpful AI assistant.' },
        { role: 'user', content: 'Explain quantum computing in simple terms.' },
      ],
      temperature: 0.7,
    });

    console.log('Response:', response.message.content);
    console.log('Eval count:', response.eval_count);
    console.log('Eval duration:', response.eval_duration, 'ns');
  } catch (error) {
    console.error('Error:', error);
  }

  // Streaming chat
  console.log('\n2. Streaming chat:');
  try {
    const stream = client.ollama.streamChat({
      model: 'tinyllama',
      messages: [
        { role: 'user', content: 'Tell me a short joke about programming.' },
      ],
      temperature: 0.9,
    });

    process.stdout.write('Response: ');
    for await (const chunk of stream) {
      const content = chunk.message?.content;
      if (content) {
        process.stdout.write(content);
      }
    }
    console.log('\n');
  } catch (error) {
    console.error('Error:', error);
  }
}

main().catch(console.error);
