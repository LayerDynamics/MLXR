/**
 * Example: OpenAI-compatible chat completion
 */

import { MLXRClient } from '../src';

async function main() {
  // Create client (will auto-detect Unix socket on macOS)
  const client = new MLXRClient({
    // Optional: specify base URL for HTTP
    // baseUrl: 'http://localhost:11434',

    // Optional: specify API key
    // apiKey: 'your-api-key',
  });

  console.log('MLXR TypeScript SDK - OpenAI Chat Example\n');

  // Non-streaming chat completion
  console.log('1. Non-streaming chat completion:');
  try {
    const response = await client.openai.createChatCompletion({
      model: 'tinyllama',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is machine learning?' },
      ],
      temperature: 0.7,
      max_tokens: 100,
    });

    console.log('Response:', response.choices[0].message.content);
    console.log('Usage:', response.usage);
  } catch (error) {
    console.error('Error:', error);
  }

  // Streaming chat completion
  console.log('\n2. Streaming chat completion:');
  try {
    const stream = client.openai.streamChatCompletion({
      model: 'tinyllama',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Write a haiku about coding.' },
      ],
      temperature: 0.8,
      max_tokens: 50,
    });

    process.stdout.write('Response: ');
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
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
