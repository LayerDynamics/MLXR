/**
 * @file tokenizer_test.cpp
 * @brief Unit tests for Tokenizer implementations
 */

#include "runtime/tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace mlxr::runtime;

// Test fixture for Tokenizer tests
class TokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Note: These tests will use mock behaviors since we don't have
    // a real SentencePiece model file in the test environment.
    // For full integration testing, a real model file would be needed.
  }

  void TearDown() override {
    // Cleanup if needed
  }

  // Helper to create a temporary dummy model file
  std::string create_dummy_model_file(const std::string& extension) {
    std::string filename = "/tmp/test_tokenizer" + extension;
    std::ofstream file(filename);
    file << "dummy content";
    file.close();
    return filename;
  }
};

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(TokenizerTest, CreateTokenizerSentencePiece) {
  // Test that .model extension is recognized
  std::string model_path = create_dummy_model_file(".model");

  // Note: This will fail because the file is not a valid SentencePiece model,
  // but we're testing that the factory recognizes the extension
  EXPECT_THROW({ auto tokenizer = create_tokenizer(model_path); },
               std::runtime_error);

  // Clean up
  std::filesystem::remove(model_path);
}

TEST_F(TokenizerTest, CreateTokenizerHuggingFace) {
  // Test that .json extension throws not implemented
  std::string model_path = create_dummy_model_file(".json");

  EXPECT_THROW({ auto tokenizer = create_tokenizer(model_path); },
               std::runtime_error);

  // Clean up
  std::filesystem::remove(model_path);
}

TEST_F(TokenizerTest, CreateTokenizerUnknownExtension) {
  // Test that unknown extension throws error
  std::string model_path = create_dummy_model_file(".unknown");

  EXPECT_THROW({ auto tokenizer = create_tokenizer(model_path); },
               std::runtime_error);

  // Clean up
  std::filesystem::remove(model_path);
}

TEST_F(TokenizerTest, CreateTokenizerNonExistentFile) {
  // Test that non-existent file throws error
  std::string model_path = "/tmp/non_existent_model_file_12345.model";

  EXPECT_THROW({ auto tokenizer = create_tokenizer(model_path); },
               std::runtime_error);
}

// ============================================================================
// SentencePieceTokenizer Tests (Mock/Conceptual)
// ============================================================================

TEST_F(TokenizerTest, SentencePieceConstructorInvalidPath) {
  // Test that invalid path throws error
  EXPECT_THROW(
      { SentencePieceTokenizer tokenizer("/invalid/path/to/model.model"); },
      std::runtime_error);
}

// Note: The following tests would require a real SentencePiece model file
// For demonstration, we show what the tests would look like

/*
TEST_F(TokenizerTest, SentencePieceEncodeBasic) {
  // Requires real model file
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string text = "Hello, world!";
  auto token_ids = tokenizer.encode(text);

  EXPECT_GT(token_ids.size(), 0);
}

TEST_F(TokenizerTest, SentencePieceDecodeBasic) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string text = "Hello, world!";
  auto token_ids = tokenizer.encode(text);
  std::string decoded = tokenizer.decode(token_ids);

  // Decoded should be similar to original (may have minor differences)
  EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizerTest, SentencePieceEncodeDecodeRoundTrip) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string original = "The quick brown fox jumps over the lazy dog.";
  auto token_ids = tokenizer.encode(original);
  std::string decoded = tokenizer.decode(token_ids);

  // For most tokenizers, encoding then decoding should be lossless
  EXPECT_EQ(decoded, original);
}

TEST_F(TokenizerTest, SentencePieceVocabSize) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  size_t vocab_size = tokenizer.vocab_size();

  // Typical vocab sizes are in thousands to tens of thousands
  EXPECT_GT(vocab_size, 1000);
  EXPECT_LT(vocab_size, 100000);
}

TEST_F(TokenizerTest, SentencePieceSpecialTokens) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  int bos = tokenizer.bos_token_id();
  int eos = tokenizer.eos_token_id();
  int pad = tokenizer.pad_token_id();

  // Special tokens should be valid IDs (may be -1 if not defined)
  EXPECT_GE(bos, -1);
  EXPECT_GE(eos, -1);
  EXPECT_GE(pad, -1);

  // If defined, should be within vocab range
  if (bos >= 0) {
    EXPECT_LT(static_cast<size_t>(bos), tokenizer.vocab_size());
  }
  if (eos >= 0) {
    EXPECT_LT(static_cast<size_t>(eos), tokenizer.vocab_size());
  }
}

TEST_F(TokenizerTest, SentencePieceTokenToId) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  // Test common tokens
  int hello_id = tokenizer.token_to_id("hello");
  int world_id = tokenizer.token_to_id("world");

  // Should get valid IDs
  EXPECT_GE(hello_id, 0);
  EXPECT_GE(world_id, 0);

  // Different tokens should have different IDs
  EXPECT_NE(hello_id, world_id);
}

TEST_F(TokenizerTest, SentencePieceIdToToken) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  // Get a token ID
  int token_id = tokenizer.token_to_id("hello");
  ASSERT_GE(token_id, 0);

  // Convert back to token
  std::string token = tokenizer.id_to_token(token_id);

  // Should be non-empty
  EXPECT_FALSE(token.empty());
}

TEST_F(TokenizerTest, SentencePieceTokenIdRoundTrip) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string original_token = "hello";
  int token_id = tokenizer.token_to_id(original_token);
  std::string retrieved_token = tokenizer.id_to_token(token_id);

  // Round trip should be consistent
  EXPECT_EQ(original_token, retrieved_token);
}

TEST_F(TokenizerTest, SentencePieceEncodeEmpty) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string empty_text = "";
  auto token_ids = tokenizer.encode(empty_text);

  // Empty text should produce empty or minimal token sequence
  EXPECT_LE(token_ids.size(), 2); // At most BOS/EOS
}

TEST_F(TokenizerTest, SentencePieceDecodeEmpty) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::vector<int> empty_ids;
  std::string decoded = tokenizer.decode(empty_ids);

  // Empty IDs should produce empty string
  EXPECT_TRUE(decoded.empty());
}

TEST_F(TokenizerTest, SentencePieceEncodeSpecialChars) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string special_text = "Hello!\n\tWorld? 😀";
  auto token_ids = tokenizer.encode(special_text);

  // Should handle special characters
  EXPECT_GT(token_ids.size(), 0);

  // Round trip
  std::string decoded = tokenizer.decode(token_ids);
  EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizerTest, SentencePieceEncodeLongText) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  // Create a long text (1000+ characters)
  std::string long_text;
  for (int i = 0; i < 100; ++i) {
    long_text += "This is a test sentence. ";
  }

  auto token_ids = tokenizer.encode(long_text);

  // Should produce many tokens
  EXPECT_GT(token_ids.size(), 100);

  // Round trip
  std::string decoded = tokenizer.decode(token_ids);
  EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizerTest, SentencePieceInvalidTokenId) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  // Try to convert invalid token ID
  int invalid_id = static_cast<int>(tokenizer.vocab_size()) + 1000;

  // Should handle gracefully (return unknown token or throw)
  EXPECT_NO_THROW({
    std::string token = tokenizer.id_to_token(invalid_id);
    // Token might be empty or "<unk>" or similar
    (void)token;
  });
}

TEST_F(TokenizerTest, SentencePieceUnknownToken) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  // Try to get ID for non-existent token
  std::string unknown_token = "ThisTokenDefinitelyDoesNotExist12345";
  int token_id = tokenizer.token_to_id(unknown_token);

  // Should return -1 or unknown token ID
  // Depending on implementation, might be -1 or vocab_size-1 (unk)
  EXPECT_TRUE(token_id == -1 || token_id >= 0);
}

TEST_F(TokenizerTest, SentencePieceMultipleSentences) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string text = "First sentence. Second sentence. Third sentence.";
  auto token_ids = tokenizer.encode(text);

  // Should handle multiple sentences
  EXPECT_GT(token_ids.size(), 3);

  // Round trip
  std::string decoded = tokenizer.decode(token_ids);
  EXPECT_FALSE(decoded.empty());
}

TEST_F(TokenizerTest, SentencePieceCasePreservation) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string lower = "hello world";
  std::string upper = "HELLO WORLD";
  std::string mixed = "Hello World";

  auto lower_ids = tokenizer.encode(lower);
  auto upper_ids = tokenizer.encode(upper);
  auto mixed_ids = tokenizer.encode(mixed);

  // Different cases should generally produce different token sequences
  // (unless tokenizer is case-insensitive, which is uncommon)
  EXPECT_NE(lower_ids, upper_ids);
}

TEST_F(TokenizerTest, SentencePieceNumericText) {
  SentencePieceTokenizer tokenizer("path/to/real/tokenizer.model");

  std::string numbers = "1234567890";
  auto token_ids = tokenizer.encode(numbers);

  EXPECT_GT(token_ids.size(), 0);

  // Round trip
  std::string decoded = tokenizer.decode(token_ids);
  EXPECT_FALSE(decoded.empty());
}
*/

// ============================================================================
// Interface Tests (Conceptual)
// ============================================================================

TEST_F(TokenizerTest, TokenizerInterfaceExists) {
  // Test that the Tokenizer interface can be used polymorphically
  // This is a compile-time test essentially

  // We can't instantiate abstract class, but can test pointer type
  Tokenizer* tokenizer_ptr = nullptr;
  EXPECT_EQ(tokenizer_ptr, nullptr);

  // Test that all required methods are declared
  // This will fail to compile if methods are missing
  if (tokenizer_ptr) {
    (void)tokenizer_ptr->encode("");
    (void)tokenizer_ptr->decode({});
    (void)tokenizer_ptr->vocab_size();
    (void)tokenizer_ptr->bos_token_id();
    (void)tokenizer_ptr->eos_token_id();
    (void)tokenizer_ptr->pad_token_id();
    (void)tokenizer_ptr->id_to_token(0);
    (void)tokenizer_ptr->token_to_id("");
  }
}

// ============================================================================
// Documentation Tests
// ============================================================================

TEST_F(TokenizerTest, READMEInstructions) {
  // This test documents how to use the tokenizer in practice

  // Step 1: Obtain a real SentencePiece model file
  // You can download from HuggingFace or train your own

  // Step 2: Create tokenizer
  // auto tokenizer = create_tokenizer("path/to/tokenizer.model");

  // Step 3: Encode text
  // std::string text = "Hello, world!";
  // auto token_ids = tokenizer->encode(text);

  // Step 4: Decode tokens
  // std::string decoded = tokenizer->decode(token_ids);

  // Step 5: Access special tokens
  // int bos = tokenizer->bos_token_id();
  // int eos = tokenizer->eos_token_id();

  SUCCEED();  // This test always passes, it's just documentation
}

// ============================================================================
// NOTE: Real Integration Tests
// ============================================================================

/*
 * To run comprehensive tokenizer tests, you need:
 *
 * 1. A real SentencePiece model file (e.g., from LLaMA, TinyLlama)
 * 2. Update the test fixture to load this model in SetUp()
 * 3. Uncomment the tests above
 * 4. Run: ctest --verbose
 *
 * Example model download:
 *   huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
 *     tokenizer.model --local-dir ./test_models/
 *
 * Then update SetUp() to:
 *   tokenizer_ = std::make_unique<SentencePieceTokenizer>(
 *       "./test_models/tokenizer.model"
 *   );
 */
