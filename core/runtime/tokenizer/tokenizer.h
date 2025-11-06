/**
 * @file tokenizer.h
 * @brief Tokenizer interface for MLXR
 *
 * Provides a unified interface for different tokenizer backends
 * (SentencePiece, HuggingFace tokenizers, etc.)
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mlxr {
namespace runtime {

/**
 * @brief Abstract tokenizer interface
 *
 * Base class for all tokenizer implementations.
 */
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  /**
   * @brief Encode text to token IDs
   * @param text Input text
   * @return Vector of token IDs
   */
  virtual std::vector<int> encode(const std::string& text) = 0;

  /**
   * @brief Decode token IDs to text
   * @param ids Vector of token IDs
   * @return Decoded text
   */
  virtual std::string decode(const std::vector<int>& ids) = 0;

  /**
   * @brief Get vocabulary size
   * @return Number of tokens in vocabulary
   */
  virtual size_t vocab_size() const = 0;

  /**
   * @brief Get BOS (beginning of sequence) token ID
   * @return BOS token ID, or -1 if not defined
   */
  virtual int bos_token_id() const = 0;

  /**
   * @brief Get EOS (end of sequence) token ID
   * @return EOS token ID, or -1 if not defined
   */
  virtual int eos_token_id() const = 0;

  /**
   * @brief Get PAD (padding) token ID
   * @return PAD token ID, or -1 if not defined
   */
  virtual int pad_token_id() const = 0;

  /**
   * @brief Convert token ID to string
   * @param id Token ID
   * @return Token string
   */
  virtual std::string id_to_token(int id) const = 0;

  /**
   * @brief Convert token string to ID
   * @param token Token string
   * @return Token ID, or -1 if not found
   */
  virtual int token_to_id(const std::string& token) const = 0;
};

/**
 * @brief SentencePiece tokenizer implementation
 *
 * Wrapper around the SentencePiece library for tokenization.
 */
class SentencePieceTokenizer : public Tokenizer {
 public:
  /**
   * @brief Construct tokenizer from model file
   * @param model_path Path to SentencePiece model file (.model)
   */
  explicit SentencePieceTokenizer(const std::string& model_path);

  /**
   * @brief Destructor
   */
  ~SentencePieceTokenizer() override;

  // Disable copy and move
  SentencePieceTokenizer(const SentencePieceTokenizer&) = delete;
  SentencePieceTokenizer& operator=(const SentencePieceTokenizer&) = delete;
  SentencePieceTokenizer(SentencePieceTokenizer&&) = delete;
  SentencePieceTokenizer& operator=(SentencePieceTokenizer&&) = delete;

  std::vector<int> encode(const std::string& text) override;
  std::string decode(const std::vector<int>& ids) override;
  size_t vocab_size() const override;
  int bos_token_id() const override;
  int eos_token_id() const override;
  int pad_token_id() const override;
  std::string id_to_token(int id) const override;
  int token_to_id(const std::string& token) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Factory function to create tokenizer from path
 * @param model_path Path to tokenizer model file
 * @return Unique pointer to tokenizer
 *
 * Automatically detects tokenizer type based on file extension:
 * - .model -> SentencePiece
 * - .json -> HuggingFace tokenizers (future)
 */
std::unique_ptr<Tokenizer> create_tokenizer(const std::string& model_path);

}  // namespace runtime
}  // namespace mlxr
