// Copyright © 2025 MLXR Development
// SSE streaming unit tests

#include "server/sse_stream.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace mlxr::server;

namespace {

// ==============================================================================
// Test Helpers
// ==============================================================================

class SSEStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    received_events.clear();
    callback_called = 0;
    callback_should_fail = false;
  }

  void TearDown() override {
    // Cleanup
  }

  // Mock callback for SSE stream
  bool test_callback(const std::string& event) {
    callback_called++;
    received_events.push_back(event);
    return !callback_should_fail;
  }

  std::vector<std::string> received_events;
  std::atomic<int> callback_called{0};
  bool callback_should_fail{false};
};

// ==============================================================================
// SSEEvent Tests
// ==============================================================================

TEST_F(SSEStreamTest, EventFormatDataOnly) {
  SSEEvent event;
  event.data = "Hello World";

  std::string formatted = event.format();

  EXPECT_TRUE(formatted.find("data: Hello World\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("\n\n") != std::string::npos);  // End marker
}

TEST_F(SSEStreamTest, EventFormatWithType) {
  SSEEvent event;
  event.event = "custom";
  event.data = "Test data";

  std::string formatted = event.format();

  EXPECT_TRUE(formatted.find("event: custom\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("data: Test data\n") != std::string::npos);
}

TEST_F(SSEStreamTest, EventFormatWithId) {
  SSEEvent event;
  event.data = "Message";
  event.id = "123";

  std::string formatted = event.format();

  EXPECT_TRUE(formatted.find("id: 123\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("data: Message\n") != std::string::npos);
}

TEST_F(SSEStreamTest, EventFormatWithRetry) {
  SSEEvent event;
  event.data = "Retry test";
  event.retry = 5000;

  std::string formatted = event.format();

  EXPECT_TRUE(formatted.find("retry: 5000\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("data: Retry test\n") != std::string::npos);
}

TEST_F(SSEStreamTest, EventFormatMultilineData) {
  SSEEvent event;
  event.data = "Line 1\nLine 2\nLine 3";

  std::string formatted = event.format();

  // Each line should be prefixed with "data: "
  EXPECT_TRUE(formatted.find("data: Line 1\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("data: Line 2\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("data: Line 3\n") != std::string::npos);
}

// ==============================================================================
// SSEStream Tests
// ==============================================================================

TEST_F(SSEStreamTest, StreamCreation) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  EXPECT_TRUE(stream.is_open());
  EXPECT_EQ(stream.event_count(), 0);
}

TEST_F(SSEStreamTest, SendData) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  bool success = stream.send_data("Hello");

  EXPECT_TRUE(success);
  EXPECT_EQ(callback_called, 1);
  EXPECT_EQ(stream.event_count(), 1);
  EXPECT_TRUE(received_events[0].find("data: Hello\n") != std::string::npos);
}

TEST_F(SSEStreamTest, SendEvent) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  bool success = stream.send_event("custom", "Test message");

  EXPECT_TRUE(success);
  EXPECT_EQ(callback_called, 1);
  EXPECT_TRUE(received_events[0].find("event: custom\n") != std::string::npos);
  EXPECT_TRUE(received_events[0].find("data: Test message\n") !=
              std::string::npos);
}

TEST_F(SSEStreamTest, SendComment) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  bool success = stream.send_comment("Keep-alive ping");

  EXPECT_TRUE(success);
  EXPECT_EQ(callback_called, 1);
  EXPECT_TRUE(received_events[0].find(": Keep-alive ping\n") !=
              std::string::npos);
}

TEST_F(SSEStreamTest, SendDone) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  stream.send_done();

  EXPECT_EQ(callback_called, 1);
  EXPECT_TRUE(received_events[0].find("data: [DONE]\n") != std::string::npos);
}

TEST_F(SSEStreamTest, SendMultipleEvents) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  stream.send_data("Event 1");
  stream.send_data("Event 2");
  stream.send_data("Event 3");

  EXPECT_EQ(callback_called, 3);
  EXPECT_EQ(stream.event_count(), 3);
}

TEST_F(SSEStreamTest, CloseStream) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  EXPECT_TRUE(stream.is_open());

  stream.close();

  EXPECT_FALSE(stream.is_open());

  // Should not send after close
  bool success = stream.send_data("After close");
  EXPECT_FALSE(success);
  EXPECT_EQ(callback_called, 0);  // Callback not called
}

TEST_F(SSEStreamTest, CallbackFailureClosesStream) {
  SSEStream stream(
      [this](const std::string& event) { return test_callback(event); });

  callback_should_fail = true;

  bool success = stream.send_data("This should fail");

  EXPECT_FALSE(success);
  EXPECT_FALSE(stream.is_open());
}

// ==============================================================================
// SSEResponseBuilder Tests
// ==============================================================================

TEST_F(SSEStreamTest, ResponseBuilderHeaders) {
  auto headers = SSEResponseBuilder::create_headers();

  EXPECT_EQ(headers["Content-Type"], "text/event-stream");
  EXPECT_EQ(headers["Cache-Control"], "no-cache");
  EXPECT_EQ(headers["Connection"], "keep-alive");
  EXPECT_EQ(headers["X-Accel-Buffering"], "no");
}

TEST_F(SSEStreamTest, ResponseBuilderFormatData) {
  std::string formatted = SSEResponseBuilder::format_data("Test");

  EXPECT_TRUE(formatted.find("data: Test\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("\n\n") != std::string::npos);
}

TEST_F(SSEStreamTest, ResponseBuilderFormatEvent) {
  std::string formatted = SSEResponseBuilder::format_event("message", "Hello");

  EXPECT_TRUE(formatted.find("event: message\n") != std::string::npos);
  EXPECT_TRUE(formatted.find("data: Hello\n") != std::string::npos);
}

TEST_F(SSEStreamTest, ResponseBuilderFormatComment) {
  std::string formatted = SSEResponseBuilder::format_comment("Ping");

  EXPECT_EQ(formatted, ": Ping\n\n");
}

TEST_F(SSEStreamTest, ResponseBuilderDoneMarker) {
  std::string done = SSEResponseBuilder::create_done_marker();

  EXPECT_TRUE(done.find("data: [DONE]\n") != std::string::npos);
}

// ==============================================================================
// StreamingGenerator Tests
// ==============================================================================

TEST_F(SSEStreamTest, GeneratorCreation) {
  auto stream = std::make_shared<SSEStream>(
      [this](const std::string& event) { return test_callback(event); });

  StreamingGenerator generator(stream);

  EXPECT_FALSE(generator.is_generating());
  EXPECT_EQ(generator.tokens_generated(), 0);
}

TEST_F(SSEStreamTest, GeneratorGenerate) {
  auto stream = std::make_shared<SSEStream>(
      [this](const std::string& event) { return test_callback(event); });

  StreamingGenerator generator(stream);

  std::vector<int> prompt_tokens = {1, 2, 3};
  bool success = generator.generate(prompt_tokens, 5);

  EXPECT_TRUE(success);
  EXPECT_EQ(generator.tokens_generated(), 5);
}

TEST_F(SSEStreamTest, GeneratorStop) {
  auto stream = std::make_shared<SSEStream>(
      [this](const std::string& event) { return test_callback(event); });

  StreamingGenerator generator(stream);

  // Start generation in background
  std::thread gen_thread([&]() {
    std::vector<int> prompt_tokens = {1, 2, 3};
    generator.generate(prompt_tokens, 100);
  });

  // Wait a bit then stop
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  generator.stop();

  gen_thread.join();

  // Should have generated some but not all tokens
  EXPECT_GT(generator.tokens_generated(), 0);
  EXPECT_LT(generator.tokens_generated(), 100);
}

// ==============================================================================
// ChatCompletionStreamFormatter Tests
// ==============================================================================

TEST_F(SSEStreamTest, ChatFormatterRole) {
  ChatCompletionStreamFormatter formatter("req-123", "llama-7b");

  std::string chunk = formatter.format_role("assistant");

  EXPECT_TRUE(chunk.find("\"id\":\"req-123\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"model\":\"llama-7b\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"role\":\"assistant\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"finish_reason\":null") != std::string::npos);
}

TEST_F(SSEStreamTest, ChatFormatterContent) {
  ChatCompletionStreamFormatter formatter("req-123", "llama-7b");

  std::string chunk = formatter.format_content("Hello");

  EXPECT_TRUE(chunk.find("\"content\":\"Hello\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"finish_reason\":null") != std::string::npos);
}

TEST_F(SSEStreamTest, ChatFormatterFinish) {
  ChatCompletionStreamFormatter formatter("req-123", "llama-7b");

  std::string chunk = formatter.format_finish("stop");

  EXPECT_TRUE(chunk.find("\"finish_reason\":\"stop\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"delta\":{}") != std::string::npos);
}

TEST_F(SSEStreamTest, ChatFormatterFunctionCall) {
  ChatCompletionStreamFormatter formatter("req-123", "llama-7b");

  std::string chunk =
      formatter.format_function_call("get_weather", "{\"location\":\"NYC\"}");

  EXPECT_TRUE(chunk.find("\"function_call\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"name\":\"get_weather\"") != std::string::npos);
}

// ==============================================================================
// CompletionStreamFormatter Tests
// ==============================================================================

TEST_F(SSEStreamTest, CompletionFormatterText) {
  CompletionStreamFormatter formatter("req-456", "llama-7b");

  std::string chunk = formatter.format_text("Generated text");

  EXPECT_TRUE(chunk.find("\"id\":\"req-456\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"text\":\"Generated text\"") != std::string::npos);
  EXPECT_TRUE(chunk.find("\"finish_reason\":null") != std::string::npos);
}

TEST_F(SSEStreamTest, CompletionFormatterFinish) {
  CompletionStreamFormatter formatter("req-456", "llama-7b");

  std::string chunk = formatter.format_finish("length");

  EXPECT_TRUE(chunk.find("\"finish_reason\":\"length\"") != std::string::npos);
}

// ==============================================================================
// Utility Functions Tests
// ==============================================================================

TEST_F(SSEStreamTest, GetCurrentTimestamp) {
  int64_t timestamp = get_current_timestamp();

  EXPECT_GT(timestamp, 0);
  // Should be roughly current time (after 2020)
  EXPECT_GT(timestamp, 1577836800);  // Jan 1, 2020
}

TEST_F(SSEStreamTest, EscapeJsonBasic) {
  std::string result = escape_json("Hello World");
  EXPECT_EQ(result, "Hello World");
}

TEST_F(SSEStreamTest, EscapeJsonQuotes) {
  std::string result = escape_json("Say \"Hello\"");
  EXPECT_EQ(result, "Say \\\"Hello\\\"");
}

TEST_F(SSEStreamTest, EscapeJsonNewline) {
  std::string result = escape_json("Line 1\nLine 2");
  EXPECT_EQ(result, "Line 1\\nLine 2");
}

TEST_F(SSEStreamTest, EscapeJsonBackslash) {
  std::string result = escape_json("Path\\to\\file");
  EXPECT_EQ(result, "Path\\\\to\\\\file");
}

TEST_F(SSEStreamTest, EscapeJsonTab) {
  std::string result = escape_json("Column1\tColumn2");
  EXPECT_EQ(result, "Column1\\tColumn2");
}

}  // namespace
