/**
 * Example: Generate embeddings
 */

import { MLXRClient } from '../src';

async function main() {
  const client = new MLXRClient();

  console.log('MLXR TypeScript SDK - Embeddings Example\n');

  // OpenAI-style embeddings
  console.log('1. OpenAI-style embeddings:');
  try {
    const response = await client.openai.createEmbedding({
      model: 'tinyllama',
      input: 'Machine learning is transforming the world.',
    });

    const {embedding} = response.data[0];
    console.log('Embedding dimensions:', embedding.length);
    console.log('First 10 values:', embedding.slice(0, 10));
    console.log('Usage:', response.usage);
  } catch (error) {
    console.error('Error:', error);
  }

  // Ollama-style embeddings
  console.log('\n2. Ollama-style embeddings:');
  try {
    const response = await client.ollama.embeddings({
      model: 'tinyllama',
      prompt: 'Artificial intelligence is the future.',
    });

    console.log('Embedding dimensions:', response.embedding.length);
    console.log('First 10 values:', response.embedding.slice(0, 10));
  } catch (error) {
    console.error('Error:', error);
  }

  // Compute similarity between two texts
  console.log('\n3. Compute text similarity:');
  try {
    const text1 = 'I love programming in TypeScript.';
    const text2 = 'TypeScript is great for building applications.';
    const text3 = 'The weather is nice today.';

    const [emb1, emb2, emb3] = await Promise.all([
      client.ollama.embeddings({ model: 'tinyllama', prompt: text1 }),
      client.ollama.embeddings({ model: 'tinyllama', prompt: text2 }),
      client.ollama.embeddings({ model: 'tinyllama', prompt: text3 }),
    ]);

    const similarity12 = cosineSimilarity(emb1.embedding, emb2.embedding);
    const similarity13 = cosineSimilarity(emb1.embedding, emb3.embedding);

    console.log(`Similarity between text1 and text2: ${similarity12.toFixed(4)}`);
    console.log(`Similarity between text1 and text3: ${similarity13.toFixed(4)}`);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Helper function to compute cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

main().catch(console.error);
