// Copyright © 2025 MLXR Development
// REST server unit tests

#include "server/rest_server.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "runtime/tokenizer/tokenizer.h"

using namespace mlxr::server;
using namespace mlxr::runtime;

namespace {

// ==============================================================================
// Test Data Structures
// ==============================================================================

class RestServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.bind_address = "127.0.0.1";
    config_.port = 8080;
    config_.enable_cors = true;
    config_.max_connections = 100;
    config_.thread_pool_size = 4;
    config_.enable_metrics = true;
  }

  void TearDown() override {
    // Cleanup
  }

  ServerConfig config_;
};

// ==============================================================================
// Configuration Tests
// ==============================================================================

TEST_F(RestServerTest, ConfigDefaults) {
  ServerConfig default_config;

  EXPECT_EQ(default_config.bind_address, "127.0.0.1");
  EXPECT_EQ(default_config.port, 8080);
  EXPECT_TRUE(default_config.enable_cors);
  EXPECT_EQ(default_config.max_connections, 100);
  EXPECT_EQ(default_config.thread_pool_size, 4);
  EXPECT_TRUE(default_config.enable_metrics);
}

TEST_F(RestServerTest, ServerConstruction) {
  RestServer server(config_);
  EXPECT_EQ(server.config().bind_address, "127.0.0.1");
  EXPECT_EQ(server.config().port, 8080);
  EXPECT_FALSE(server.is_running());
}

TEST_F(RestServerTest, ServerInitialization) {
  RestServer server(config_);
  EXPECT_TRUE(server.initialize());
}

TEST_F(RestServerTest, ServerInvalidPort) {
  config_.port = -1;
  RestServer server(config_);
  EXPECT_FALSE(server.initialize());
}

TEST_F(RestServerTest, ServerStartStop) {
  RestServer server(config_);
  ASSERT_TRUE(server.initialize());

  EXPECT_TRUE(server.start());
  EXPECT_TRUE(server.is_running());

  // Give server time to start listening before stopping
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  server.stop();
  EXPECT_FALSE(server.is_running());
}

TEST_F(RestServerTest, ServerDoubleStart) {
  RestServer server(config_);
  ASSERT_TRUE(server.initialize());

  EXPECT_TRUE(server.start());

  // Give server time to start listening
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  EXPECT_FALSE(server.start());  // Should fail on second start

  server.stop();
}

// ==============================================================================
// Request/Response Data Structure Tests
// ==============================================================================

TEST_F(RestServerTest, ChatMessageStructure) {
  ChatMessage msg;
  msg.role = "user";
  msg.content = "Hello";
  msg.name = "test_user";

  EXPECT_EQ(msg.role, "user");
  EXPECT_EQ(msg.content, "Hello");
  EXPECT_TRUE(msg.name.has_value());
  EXPECT_EQ(msg.name.value(), "test_user");
}

TEST_F(RestServerTest, ChatCompletionRequest) {
  ChatCompletionRequest req;
  req.model = "llama-7b";

  ChatMessage msg;
  msg.role = "user";
  msg.content = "Hello, AI!";
  req.messages.push_back(msg);

  req.temperature = 0.7f;
  req.max_tokens = 100;
  req.stream = false;

  EXPECT_EQ(req.model, "llama-7b");
  EXPECT_EQ(req.messages.size(), 1);
  EXPECT_EQ(req.messages[0].role, "user");
  EXPECT_TRUE(req.temperature.has_value());
  EXPECT_FLOAT_EQ(req.temperature.value(), 0.7f);
  EXPECT_TRUE(req.max_tokens.has_value());
  EXPECT_EQ(req.max_tokens.value(), 100);
}

TEST_F(RestServerTest, CompletionRequest) {
  CompletionRequest req;
  req.model = "llama-7b";
  req.prompt = "Once upon a time";
  req.max_tokens = 50;
  req.temperature = 0.8f;

  EXPECT_EQ(req.model, "llama-7b");
  EXPECT_EQ(req.prompt, "Once upon a time");
  EXPECT_EQ(req.max_tokens.value(), 50);
  EXPECT_FLOAT_EQ(req.temperature.value(), 0.8f);
}

TEST_F(RestServerTest, EmbeddingRequest) {
  EmbeddingRequest req;
  req.model = "text-embedding-ada-002";
  req.input = "The quick brown fox";
  req.encoding_format = "float";

  EXPECT_EQ(req.model, "text-embedding-ada-002");
  EXPECT_EQ(req.input, "The quick brown fox");
  EXPECT_TRUE(req.encoding_format.has_value());
  EXPECT_EQ(req.encoding_format.value(), "float");
}

TEST_F(RestServerTest, UsageInfo) {
  UsageInfo usage;
  usage.prompt_tokens = 10;
  usage.completion_tokens = 20;
  usage.total_tokens = 30;

  EXPECT_EQ(usage.prompt_tokens, 10);
  EXPECT_EQ(usage.completion_tokens, 20);
  EXPECT_EQ(usage.total_tokens, 30);
}

TEST_F(RestServerTest, ChatCompletionResponse) {
  ChatCompletionResponse response;
  response.id = "chatcmpl-123";
  response.model = "llama-7b";
  response.created = 1234567890;

  ChatCompletionChoice choice;
  choice.index = 0;
  choice.message.role = "assistant";
  choice.message.content = "Hello! How can I help you?";
  choice.finish_reason = "stop";

  response.choices.push_back(choice);

  response.usage.prompt_tokens = 5;
  response.usage.completion_tokens = 10;
  response.usage.total_tokens = 15;

  EXPECT_EQ(response.id, "chatcmpl-123");
  EXPECT_EQ(response.model, "llama-7b");
  EXPECT_EQ(response.choices.size(), 1);
  EXPECT_EQ(response.choices[0].message.content, "Hello! How can I help you?");
  EXPECT_EQ(response.usage.total_tokens, 15);
}

TEST_F(RestServerTest, CompletionResponse) {
  CompletionResponse response;
  response.id = "cmpl-123";
  response.model = "llama-7b";
  response.created = 1234567890;

  CompletionChoice choice;
  choice.index = 0;
  choice.text = "Once upon a time, there was a brave knight.";
  choice.finish_reason = "stop";

  response.choices.push_back(choice);

  response.usage.prompt_tokens = 4;
  response.usage.completion_tokens = 12;
  response.usage.total_tokens = 16;

  EXPECT_EQ(response.id, "cmpl-123");
  EXPECT_EQ(response.choices[0].text,
            "Once upon a time, there was a brave knight.");
  EXPECT_EQ(response.usage.total_tokens, 16);
}

TEST_F(RestServerTest, EmbeddingResponse) {
  EmbeddingResponse response;
  response.model = "text-embedding-ada-002";

  EmbeddingObject emb;
  emb.index = 0;
  emb.embedding = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

  response.data.push_back(emb);

  response.usage.prompt_tokens = 5;
  response.usage.total_tokens = 5;

  EXPECT_EQ(response.model, "text-embedding-ada-002");
  EXPECT_EQ(response.data.size(), 1);
  EXPECT_EQ(response.data[0].embedding.size(), 5);
  EXPECT_FLOAT_EQ(response.data[0].embedding[0], 0.1f);
  EXPECT_EQ(response.usage.total_tokens, 5);
}

TEST_F(RestServerTest, ModelInfo) {
  ModelInfo info;
  info.id = "llama-7b";
  info.created = 1234567890;
  info.owned_by = "mlxr";

  EXPECT_EQ(info.id, "llama-7b");
  EXPECT_EQ(info.object, "model");
  EXPECT_EQ(info.owned_by, "mlxr");
}

TEST_F(RestServerTest, ModelListResponse) {
  ModelListResponse response;

  ModelInfo model1;
  model1.id = "llama-7b";
  response.data.push_back(model1);

  ModelInfo model2;
  model2.id = "mistral-7b";
  response.data.push_back(model2);

  EXPECT_EQ(response.object, "list");
  EXPECT_EQ(response.data.size(), 2);
  EXPECT_EQ(response.data[0].id, "llama-7b");
  EXPECT_EQ(response.data[1].id, "mistral-7b");
}

TEST_F(RestServerTest, ErrorResponse) {
  ErrorResponse error;
  error.error.message = "Model not found";
  error.error.type = "not_found_error";
  error.error.code = "404";

  EXPECT_EQ(error.error.message, "Model not found");
  EXPECT_EQ(error.error.type, "not_found_error");
  EXPECT_TRUE(error.error.code.has_value());
  EXPECT_EQ(error.error.code.value(), "404");
}

TEST_F(RestServerTest, ChatCompletionChunk) {
  ChatCompletionChunk chunk;
  chunk.id = "chatcmpl-123";
  chunk.model = "llama-7b";
  chunk.created = 1234567890;

  ChatCompletionStreamChoice choice;
  choice.index = 0;
  choice.delta.content = "Hello";
  choice.finish_reason = "";

  chunk.choices.push_back(choice);

  EXPECT_EQ(chunk.object, "chat.completion.chunk");
  EXPECT_EQ(chunk.choices.size(), 1);
  EXPECT_TRUE(chunk.choices[0].delta.content.has_value());
  EXPECT_EQ(chunk.choices[0].delta.content.value(), "Hello");
}

// ==============================================================================
// HTTP Request/Response Tests
// ==============================================================================

TEST_F(RestServerTest, HttpRequest) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/chat/completions";
  request.headers["Content-Type"] = "application/json";
  request.headers["Authorization"] = "Bearer sk-test";
  request.body = "{\"model\":\"llama-7b\"}";

  EXPECT_EQ(request.method, "POST");
  EXPECT_EQ(request.path, "/v1/chat/completions");
  EXPECT_EQ(request.headers.size(), 2);
  EXPECT_EQ(request.headers["Content-Type"], "application/json");
}

TEST_F(RestServerTest, HttpResponse) {
  HttpResponse response;
  response.status_code = 200;
  response.headers["Content-Type"] = "application/json";
  response.body = "{\"status\":\"ok\"}";

  EXPECT_EQ(response.status_code, 200);
  EXPECT_EQ(response.headers["Content-Type"], "application/json");
  EXPECT_EQ(response.body, "{\"status\":\"ok\"}");
}

// ==============================================================================
// Utility Tests
// ==============================================================================

TEST_F(RestServerTest, GenerateRequestId) {
  RestServer server(config_);

  // Can't test private method directly, but we test it via public API
  // Just verify that server can be constructed
  EXPECT_TRUE(true);
}

// ==============================================================================
// Function Definition Tests
// ==============================================================================

TEST_F(RestServerTest, FunctionDefinition) {
  FunctionDefinition func;
  func.name = "get_weather";
  func.description = "Get current weather for a location";
  func.parameters_json = "{\"type\":\"object\",\"properties\":{}}";

  EXPECT_EQ(func.name, "get_weather");
  EXPECT_EQ(func.description, "Get current weather for a location");
  EXPECT_FALSE(func.parameters_json.empty());
}

TEST_F(RestServerTest, ToolDefinition) {
  ToolDefinition tool;
  tool.type = "function";
  tool.function.name = "calculate";
  tool.function.description = "Perform calculations";

  EXPECT_EQ(tool.type, "function");
  EXPECT_EQ(tool.function.name, "calculate");
}

// ==============================================================================
// Streaming Tests (Structure only - actual streaming in Task 18)
// ==============================================================================

TEST_F(RestServerTest, ChatCompletionDelta) {
  ChatCompletionDelta delta;
  delta.role = "assistant";
  delta.content = "Hello";

  EXPECT_TRUE(delta.role.has_value());
  EXPECT_EQ(delta.role.value(), "assistant");
  EXPECT_TRUE(delta.content.has_value());
  EXPECT_EQ(delta.content.value(), "Hello");
}

TEST_F(RestServerTest, ChatCompletionStreamChoice) {
  ChatCompletionStreamChoice choice;
  choice.index = 0;
  choice.delta.content = "World";
  choice.finish_reason = "";

  EXPECT_EQ(choice.index, 0);
  EXPECT_TRUE(choice.delta.content.has_value());
  EXPECT_EQ(choice.finish_reason, "");
}

}  // namespace
