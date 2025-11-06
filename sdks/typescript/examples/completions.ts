/**
 * Example: Text completions (non-chat)
 */

import { MLXRClient } from '../src';

async function main() {
  const client = new MLXRClient();

  console.log('MLXR TypeScript SDK - Completions Example\n');

  // OpenAI-style completion
  console.log('1. OpenAI-style completion:');
  try {
    const response = await client.openai.createCompletion({
      model: 'tinyllama',
      prompt: 'Once upon a time in a land far away,',
      max_tokens: 100,
      temperature: 0.8,
      stop: ['\n\n', 'The End'],
    });

    console.log('Completion:', response.choices[0].text);
    console.log('Finish reason:', response.choices[0].finish_reason);
    console.log('Usage:', response.usage);
  } catch (error) {
    console.error('Error:', error);
  }

  // OpenAI-style streaming completion
  console.log('\n2. OpenAI-style streaming completion:');
  try {
    const stream = client.openai.streamCompletion({
      model: 'tinyllama',
      prompt: 'The three laws of robotics are:',
      max_tokens: 150,
      temperature: 0.7,
    });

    process.stdout.write('Completion: ');
    for await (const chunk of stream) {
      const text = chunk.choices[0]?.text;
      if (text) {
        process.stdout.write(text);
      }
    }
    console.log('\n');
  } catch (error) {
    console.error('Error:', error);
  }

  // Ollama-style generation
  console.log('3. Ollama-style generation:');
  try {
    const response = await client.ollama.generate({
      model: 'tinyllama',
      prompt: 'Explain the concept of recursion in programming:',
      num_predict: 100,
      temperature: 0.7,
    });

    console.log('Response:', response.response);
    console.log('Done:', response.done);
    console.log('Eval count:', response.eval_count);
  } catch (error) {
    console.error('Error:', error);
  }

  // Ollama-style streaming generation
  console.log('\n4. Ollama-style streaming generation:');
  try {
    const stream = client.ollama.streamGenerate({
      model: 'tinyllama',
      prompt: 'Write a poem about technology:',
      num_predict: 80,
      temperature: 0.9,
    });

    process.stdout.write('Response: ');
    for await (const chunk of stream) {
      if (chunk.response) {
        process.stdout.write(chunk.response);
      }
    }
    console.log('\n');
  } catch (error) {
    console.error('Error:', error);
  }
}

main().catch(console.error);
