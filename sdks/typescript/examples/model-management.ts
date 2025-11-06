/**
 * Example: Model management with Ollama API
 */

import { MLXRClient } from '../src';

async function main() {
  const client = new MLXRClient();

  console.log('MLXR TypeScript SDK - Model Management Example\n');

  // List all models
  console.log('1. List all models:');
  try {
    const tags = await client.ollama.tags();
    console.log('Available models:', tags.models.length);
    tags.models.forEach((model) => {
      console.log(`  - ${model.name}`);
      console.log(`    Size: ${(model.size / 1024 / 1024).toFixed(2)} MB`);
      console.log(`    Modified: ${model.modified_at}`);
    });
  } catch (error) {
    console.error('Error:', error);
  }

  // Show model details
  console.log('\n2. Show model details:');
  try {
    const details = await client.ollama.show({ name: 'tinyllama' });
    console.log('Model details:');
    console.log('  Format:', details.details.format);
    console.log('  Family:', details.details.family);
    console.log('  Parameters:', details.details.parameter_size);
    console.log('  Quantization:', details.details.quantization_level);
  } catch (error) {
    console.error('Error:', error);
  }

  // List running models
  console.log('\n3. List running models:');
  try {
    const running = await client.ollama.ps();
    console.log('Running models:', running.models.length);
    running.models.forEach((model) => {
      console.log(`  - ${model.name}`);
      console.log(`    VRAM: ${((model.size_vram || 0) / 1024 / 1024).toFixed(2)} MB`);
    });
  } catch (error) {
    console.error('Error:', error);
  }

  // Pull a model (with streaming progress)
  console.log('\n4. Pull a model:');
  try {
    console.log('Pulling tinyllama (this may take a while)...');
    const stream = client.ollama.streamPull({
      name: 'tinyllama',
      stream: true,
    });

    for await (const progress of stream) {
      if (progress.total && progress.completed) {
        const percent = ((progress.completed / progress.total) * 100).toFixed(1);
        process.stdout.write(`\r  Progress: ${percent}%`);
      } else {
        console.log(`  Status: ${progress.status}`);
      }
    }
    console.log('\n  Model pulled successfully!');
  } catch (error) {
    console.error('Error:', error);
  }
}

main().catch(console.error);
