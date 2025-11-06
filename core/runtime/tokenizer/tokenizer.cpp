/**
 * @file tokenizer.cpp
 * @brief Implementation of tokenizer wrappers
 */

#include "tokenizer.h"

#include <sentencepiece_processor.h>

#include <stdexcept>

namespace mlxr {
namespace runtime {

// SentencePiece implementation using PIMPL pattern
class SentencePieceTokenizer::Impl {
 public:
  explicit Impl(const std::string& model_path) {
    const auto status = processor_.Load(model_path);
    if (!status.ok()) {
      throw std::runtime_error("Failed to load SentencePiece model from " +
                               model_path + ": " + status.ToString());
    }
  }

  std::vector<int> encode(const std::string& text) {
    std::vector<int> ids;
    processor_.Encode(text, &ids);
    return ids;
  }

  std::string decode(const std::vector<int>& ids) {
    std::string text;
    processor_.Decode(ids, &text);
    return text;
  }

  size_t vocab_size() const { return processor_.GetPieceSize(); }

  int bos_token_id() const { return processor_.bos_id(); }

  int eos_token_id() const { return processor_.eos_id(); }

  int pad_token_id() const { return processor_.pad_id(); }

  std::string id_to_token(int id) const {
    if (id < 0 || id >= static_cast<int>(processor_.GetPieceSize())) {
      return "";
    }
    return processor_.IdToPiece(id);
  }

  int token_to_id(const std::string& token) const {
    return processor_.PieceToId(token);
  }

 private:
  sentencepiece::SentencePieceProcessor processor_;
};

// SentencePieceTokenizer implementation
SentencePieceTokenizer::SentencePieceTokenizer(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

std::vector<int> SentencePieceTokenizer::encode(const std::string& text) {
  return impl_->encode(text);
}

std::string SentencePieceTokenizer::decode(const std::vector<int>& ids) {
  return impl_->decode(ids);
}

size_t SentencePieceTokenizer::vocab_size() const {
  return impl_->vocab_size();
}

int SentencePieceTokenizer::bos_token_id() const {
  return impl_->bos_token_id();
}

int SentencePieceTokenizer::eos_token_id() const {
  return impl_->eos_token_id();
}

int SentencePieceTokenizer::pad_token_id() const {
  return impl_->pad_token_id();
}

std::string SentencePieceTokenizer::id_to_token(int id) const {
  return impl_->id_to_token(id);
}

int SentencePieceTokenizer::token_to_id(const std::string& token) const {
  return impl_->token_to_id(token);
}

// C++17 compatible string suffix check
static bool ends_with(const std::string& str, const std::string& suffix) {
  if (suffix.length() > str.length()) return false;
  return str.compare(str.length() - suffix.length(), suffix.length(), suffix) ==
         0;
}

// Factory function
std::unique_ptr<Tokenizer> create_tokenizer(const std::string& model_path) {
  // Check file extension to determine tokenizer type
  if (ends_with(model_path, ".model")) {
    return std::make_unique<SentencePieceTokenizer>(model_path);
  } else if (ends_with(model_path, ".json")) {
    throw std::runtime_error(
        "HuggingFace tokenizers not yet implemented. "
        "Please use SentencePiece (.model) format.");
  } else {
    throw std::runtime_error("Unknown tokenizer format: " + model_path +
                             ". Supported: .model (SentencePiece)");
  }
}

}  // namespace runtime
}  // namespace mlxr
