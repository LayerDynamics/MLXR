// Copyright © 2025 MLXR Development
// Ollama API unit tests

#include "server/ollama_api.h"

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <vector>

using namespace mlxr::server;

namespace {

// ==============================================================================
// Test Helpers
// ==============================================================================

class OllamaAPITest : public ::testing::Test {
 protected:
  void SetUp() override {
    handler_ = std::make_unique<OllamaAPIHandler>();
    received_chunks_.clear();
    callback_called_ = 0;
  }

  void TearDown() override { handler_.reset(); }

  // Mock stream callback
  bool test_stream_callback(const std::string& chunk) {
    callback_called_++;
    received_chunks_.push_back(chunk);
    return true;  // Continue streaming
  }

  std::unique_ptr<OllamaAPIHandler> handler_;
  std::vector<std::string> received_chunks_;
  std::atomic<int> callback_called_{0};
};

// ==============================================================================
// Generate Endpoint Tests
// ==============================================================================

TEST_F(OllamaAPITest, GenerateNonStreaming) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Hello, world!",
    "stream": false
  })";

  std::string response = handler_->handle_generate(request, nullptr);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"model\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"response\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"done\":true") != std::string::npos);
}

TEST_F(OllamaAPITest, GenerateStreaming) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Tell me a story",
    "stream": true
  })";

  auto callback = [this](const std::string& chunk) {
    return test_stream_callback(chunk);
  };

  std::string response = handler_->handle_generate(request, callback);

  // Non-streaming response should be empty when streaming
  EXPECT_TRUE(response.empty());

  // Should have received multiple chunks
  EXPECT_GT(callback_called_, 0);
  EXPECT_GT(received_chunks_.size(), 0);

  // Last chunk should have done:true
  if (!received_chunks_.empty()) {
    const auto& last_chunk = received_chunks_.back();
    EXPECT_TRUE(last_chunk.find("\"done\":true") != std::string::npos);
  }
}

// ==============================================================================
// Chat Endpoint Tests
// ==============================================================================

TEST_F(OllamaAPITest, ChatNonStreaming) {
  std::string request = R"({
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  })";

  std::string response = handler_->handle_chat(request, nullptr);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"model\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"message\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"role\":\"assistant\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"content\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"done\":true") != std::string::npos);
}

TEST_F(OllamaAPITest, ChatStreaming) {
  std::string request = R"({
    "model": "llama3",
    "messages": [
      {"role": "user", "content": "Write a poem"}
    ],
    "stream": true
  })";

  auto callback = [this](const std::string& chunk) {
    return test_stream_callback(chunk);
  };

  std::string response = handler_->handle_chat(request, callback);

  EXPECT_TRUE(response.empty());
  EXPECT_GT(callback_called_, 0);
  EXPECT_GT(received_chunks_.size(), 0);

  // Last chunk should have done:true
  if (!received_chunks_.empty()) {
    const auto& last_chunk = received_chunks_.back();
    EXPECT_TRUE(last_chunk.find("\"done\":true") != std::string::npos);
  }
}

TEST_F(OllamaAPITest, ChatMultipleMessages) {
  std::string request = R"({
    "model": "llama3",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What's the weather?"},
      {"role": "assistant", "content": "I don't have weather info."},
      {"role": "user", "content": "That's okay."}
    ],
    "stream": false
  })";

  std::string response = handler_->handle_chat(request, nullptr);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"done\":true") != std::string::npos);
}

// ==============================================================================
// Embeddings Endpoint Tests
// ==============================================================================

TEST_F(OllamaAPITest, Embeddings) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Embed this text"
  })";

  std::string response = handler_->handle_embeddings(request);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"embedding\"") != std::string::npos);
  EXPECT_TRUE(response.find("[") != std::string::npos);
  EXPECT_TRUE(response.find("]") != std::string::npos);
}

TEST_F(OllamaAPITest, EmbeddingsLongText) {
  std::string long_text(1000, 'a');
  std::string request = R"({
    "model": "llama3",
    "prompt": ")" + long_text +
                        R"("
  })";

  std::string response = handler_->handle_embeddings(request);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"embedding\"") != std::string::npos);
}

// ==============================================================================
// Model Management Tests
// ==============================================================================

TEST_F(OllamaAPITest, Tags) {
  std::string response = handler_->handle_tags();

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"models\"") != std::string::npos);
  EXPECT_TRUE(response.find("[") != std::string::npos);
}

TEST_F(OllamaAPITest, TagsModelDetails) {
  std::string response = handler_->handle_tags();

  EXPECT_TRUE(response.find("\"name\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"modified_at\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"size\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"digest\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"details\"") != std::string::npos);
}

TEST_F(OllamaAPITest, ProcessList) {
  std::string response = handler_->handle_ps();

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"models\"") != std::string::npos);
}

TEST_F(OllamaAPITest, Show) {
  std::string request = R"({
    "name": "llama3:latest"
  })";

  std::string response = handler_->handle_show(request);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"modelfile\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"parameters\"") != std::string::npos);
  EXPECT_TRUE(response.find("\"template\"") != std::string::npos);
}

TEST_F(OllamaAPITest, Copy) {
  std::string request = R"({
    "source": "llama3:latest",
    "destination": "llama3:backup"
  })";

  std::string response = handler_->handle_copy(request);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("{") != std::string::npos);
}

TEST_F(OllamaAPITest, Delete) {
  std::string request = R"({
    "name": "llama3:latest"
  })";

  std::string response = handler_->handle_delete(request);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("{") != std::string::npos);
}

// ==============================================================================
// Pull Endpoint Tests
// ==============================================================================

TEST_F(OllamaAPITest, PullStreaming) {
  std::string request = R"({
    "name": "llama3:latest",
    "stream": true
  })";

  auto callback = [this](const std::string& chunk) {
    return test_stream_callback(chunk);
  };

  std::string response = handler_->handle_pull(request, callback);

  EXPECT_TRUE(response.empty());
  EXPECT_GT(callback_called_, 0);
  EXPECT_GT(received_chunks_.size(), 0);

  // Should have status updates
  bool found_status = false;
  for (const auto& chunk : received_chunks_) {
    if (chunk.find("\"status\"") != std::string::npos) {
      found_status = true;
      break;
    }
  }
  EXPECT_TRUE(found_status);
}

TEST_F(OllamaAPITest, PullProgress) {
  std::string request = R"({
    "name": "llama3:latest",
    "stream": true
  })";

  auto callback = [this](const std::string& chunk) {
    return test_stream_callback(chunk);
  };

  handler_->handle_pull(request, callback);

  // Check for various status messages
  bool found_downloading = false;
  for (const auto& chunk : received_chunks_) {
    if (chunk.find("downloading") != std::string::npos) {
      found_downloading = true;
      break;
    }
  }
  EXPECT_TRUE(found_downloading);
}

// ==============================================================================
// Create Endpoint Tests
// ==============================================================================

TEST_F(OllamaAPITest, CreateStreaming) {
  std::string request = R"({
    "name": "custom-model",
    "modelfile": "FROM llama3\nPARAMETER temperature 0.8",
    "stream": true
  })";

  auto callback = [this](const std::string& chunk) {
    return test_stream_callback(chunk);
  };

  std::string response = handler_->handle_create(request, callback);

  EXPECT_TRUE(response.empty());
  EXPECT_GT(callback_called_, 0);
  EXPECT_GT(received_chunks_.size(), 0);

  // Last chunk should indicate success
  if (!received_chunks_.empty()) {
    const auto& last_chunk = received_chunks_.back();
    EXPECT_TRUE(last_chunk.find("success") != std::string::npos);
  }
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(OllamaAPITest, InvalidGenerateRequest) {
  std::string request = "invalid json {{{";

  std::string response = handler_->handle_generate(request, nullptr);

  // Should still return something (placeholder implementation returns valid
  // JSON)
  EXPECT_FALSE(response.empty());
}

TEST_F(OllamaAPITest, InvalidChatRequest) {
  std::string request = "not json at all";

  std::string response = handler_->handle_chat(request, nullptr);

  // Should still return something (placeholder implementation returns valid
  // JSON)
  EXPECT_FALSE(response.empty());
}

// ==============================================================================
// Streaming Cancellation Tests
// ==============================================================================

TEST_F(OllamaAPITest, StreamingCancellation) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Long story",
    "stream": true
  })";

  int chunks_before_cancel = 3;
  int chunks_received = 0;

  auto callback = [&](const std::string& chunk) {
    chunks_received++;
    if (chunks_received >= chunks_before_cancel) {
      return false;  // Cancel streaming
    }
    return true;
  };

  handler_->handle_generate(request, callback);

  EXPECT_EQ(chunks_received, chunks_before_cancel);
}

// ==============================================================================
// Model Parameters Tests
// ==============================================================================

TEST_F(OllamaAPITest, GenerateWithParameters) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Test",
    "temperature": 0.8,
    "top_p": 0.9,
    "num_predict": 100,
    "seed": 42,
    "stop": ["STOP", "END"]
  })";

  std::string response = handler_->handle_generate(request, nullptr);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"done\":true") != std::string::npos);
}

TEST_F(OllamaAPITest, ChatWithParameters) {
  std::string request = R"({
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.7,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "format": "json"
  })";

  std::string response = handler_->handle_chat(request, nullptr);

  EXPECT_FALSE(response.empty());
  EXPECT_TRUE(response.find("\"done\":true") != std::string::npos);
}

// ==============================================================================
// JSON Format Tests
// ==============================================================================

TEST_F(OllamaAPITest, GenerateResponseFormat) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Test",
    "stream": false
  })";

  std::string response = handler_->handle_generate(request, nullptr);

  // Basic JSON structure checks
  EXPECT_TRUE(response.find("{") != std::string::npos);
  EXPECT_TRUE(response.find("}") != std::string::npos);
  EXPECT_TRUE(response.find("\"model\":") != std::string::npos);
  EXPECT_TRUE(response.find("\"created_at\":") != std::string::npos);
  EXPECT_TRUE(response.find("\"response\":") != std::string::npos);
  EXPECT_TRUE(response.find("\"done\":") != std::string::npos);
}

TEST_F(OllamaAPITest, ChatResponseFormat) {
  std::string request = R"({
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": false
  })";

  std::string response = handler_->handle_chat(request, nullptr);

  // Basic JSON structure checks
  EXPECT_TRUE(response.find("{") != std::string::npos);
  EXPECT_TRUE(response.find("}") != std::string::npos);
  EXPECT_TRUE(response.find("\"model\":") != std::string::npos);
  EXPECT_TRUE(response.find("\"created_at\":") != std::string::npos);
  EXPECT_TRUE(response.find("\"message\":") != std::string::npos);
  EXPECT_TRUE(response.find("\"done\":") != std::string::npos);
}

TEST_F(OllamaAPITest, EmbeddingsResponseFormat) {
  std::string request = R"({
    "model": "llama3",
    "prompt": "Test"
  })";

  std::string response = handler_->handle_embeddings(request);

  // Should have array of floats
  EXPECT_TRUE(response.find("\"embedding\":[") != std::string::npos);
  EXPECT_TRUE(response.find("]") != std::string::npos);

  // Should have numeric values
  bool has_numeric = false;
  for (char c : response) {
    if (std::isdigit(c)) {
      has_numeric = true;
      break;
    }
  }
  EXPECT_TRUE(has_numeric);
}

}  // namespace
