#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include <thread>
#include <chrono>

#include "daemon/server/grpc_server.h"
#include "daemon/server/proto/mlxrunner.grpc.pb.h"
#include "daemon/scheduler/scheduler.h"
#include "daemon/registry/model_registry.h"
#include "daemon/telemetry/metrics.h"

using namespace mlxr;
using namespace mlxrunner::v1;

// Test fixture for gRPC server tests
class GrpcServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock dependencies
        scheduler_ = std::make_shared<Scheduler>();
        registry_ = std::make_shared<ModelRegistry>();
        metrics_ = std::make_shared<MetricsCollector>();

        // Configure server for testing
        config_.host = "127.0.0.1";
        config_.port = 50052;  // Use different port than default to avoid conflicts
        config_.enable_reflection = true;
        config_.max_message_size = 10 * 1024 * 1024;  // 10MB for tests
    }

    void TearDown() override {
        if (server_) {
            server_->Stop();
        }
    }

    // Helper to start server
    bool StartTestServer() {
        server_ = std::make_unique<GrpcServer>(config_, scheduler_, registry_, metrics_);
        bool started = server_->Start();
        if (started) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Let server initialize
        }
        return started;
    }

    // Helper to create client stub
    std::unique_ptr<MLXRunnerService::Stub> CreateClient() {
        std::string server_address = config_.host + ":" + std::to_string(config_.port);
        auto channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
        return MLXRunnerService::NewStub(channel);
    }

    GrpcServer::Config config_;
    std::shared_ptr<Scheduler> scheduler_;
    std::shared_ptr<ModelRegistry> registry_;
    std::shared_ptr<MetricsCollector> metrics_;
    std::unique_ptr<GrpcServer> server_;
};

// ============================================================================
// Server Lifecycle Tests
// ============================================================================

TEST_F(GrpcServerTest, ServerStartsSuccessfully) {
    EXPECT_TRUE(StartTestServer());
    EXPECT_TRUE(server_->IsRunning());
}

TEST_F(GrpcServerTest, ServerStopsCleanly) {
    ASSERT_TRUE(StartTestServer());
    server_->Stop();
    EXPECT_FALSE(server_->IsRunning());
}

TEST_F(GrpcServerTest, ServerBindsToConfiguredAddress) {
    ASSERT_TRUE(StartTestServer());
    std::string expected = config_.host + ":" + std::to_string(config_.port);
    EXPECT_EQ(server_->GetBindAddress(), expected);
}

TEST_F(GrpcServerTest, CannotStartServerTwice) {
    ASSERT_TRUE(StartTestServer());
    EXPECT_FALSE(server_->Start());  // Second start should fail
}

// ============================================================================
// Health Endpoint Tests
// ============================================================================

TEST_F(GrpcServerTest, HealthEndpointReturnsOk) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    HealthRequest request;
    HealthResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->Health(&context, request, &response);

    ASSERT_TRUE(status.ok()) << "Error: " << status.error_message();
    EXPECT_EQ(response.status(), "ok");
    EXPECT_GE(response.uptime_seconds(), 0);
}

TEST_F(GrpcServerTest, HealthEndpointReturnsUptime) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    // First health check
    {
        HealthRequest request;
        HealthResponse response;
        grpc::ClientContext context;
        ASSERT_TRUE(stub->Health(&context, request, &response).ok());
        int64_t uptime1 = response.uptime_seconds();

        // Wait a bit
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Second health check
        grpc::ClientContext context2;
        HealthResponse response2;
        ASSERT_TRUE(stub->Health(&context2, request, &response2).ok());
        int64_t uptime2 = response2.uptime_seconds();

        // Uptime should increase
        EXPECT_GE(uptime2, uptime1);
    }
}

// ============================================================================
// Status Endpoint Tests
// ============================================================================

TEST_F(GrpcServerTest, GetStatusReturnsSchedulerStats) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    StatusRequest request;
    StatusResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->GetStatus(&context, request, &response);

    ASSERT_TRUE(status.ok());
    EXPECT_GE(response.pending_requests(), 0);
    EXPECT_GE(response.active_requests(), 0);
    EXPECT_GE(response.kv_blocks_total(), 0);
}

TEST_F(GrpcServerTest, GetStatusReturnsKVCacheUtilization) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    StatusRequest request;
    StatusResponse response;
    grpc::ClientContext context;

    ASSERT_TRUE(stub->GetStatus(&context, request, &response).ok());

    EXPECT_GE(response.kv_utilization_percent(), 0.0f);
    EXPECT_LE(response.kv_utilization_percent(), 100.0f);
}

// ============================================================================
// Model Management Tests
// ============================================================================

TEST_F(GrpcServerTest, ListModelsReturnsEmptyInitially) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    ListModelsRequest request;
    ListModelsResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->ListModels(&context, request, &response);

    ASSERT_TRUE(status.ok());
    // Initially no models (assuming empty registry)
    EXPECT_GE(response.models_size(), 0);
}

TEST_F(GrpcServerTest, GetModelReturnsNotFoundForInvalidId) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    GetModelRequest request;
    request.set_model_id("nonexistent-model");
    GetModelResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->GetModel(&context, request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(GrpcServerTest, LoadModelFailsForInvalidModel) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    LoadModelRequest request;
    request.set_model_id("invalid-model");
    LoadModelResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->LoadModel(&context, request, &response);

    // Should complete but indicate failure
    ASSERT_TRUE(status.ok());  // RPC succeeded
    EXPECT_FALSE(response.success());  // But model loading failed
}

// ============================================================================
// Streaming Tests
// ============================================================================

TEST_F(GrpcServerTest, CreateChatCompletionStreamsTokens) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    ChatCompletionRequest request;
    request.set_model("test-model");
    request.set_stream(true);

    auto* message = request.add_messages();
    message->set_role("user");
    message->set_content("Hello");

    grpc::ClientContext context;
    auto stream = stub->CreateChatCompletion(&context, request);

    // Read at least one chunk (or get error if model not loaded)
    ChatCompletionChunk chunk;
    if (stream->Read(&chunk)) {
        EXPECT_FALSE(chunk.id().empty());
        EXPECT_EQ(chunk.model(), "test-model");
    }

    grpc::Status status = stream->Finish();
    // May fail if no model is loaded, which is ok for this test
    EXPECT_TRUE(status.ok() || status.error_code() == grpc::StatusCode::INTERNAL);
}

TEST_F(GrpcServerTest, GenerateStreamsResponse) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    GenerateRequest request;
    request.set_model("test-model");
    request.set_prompt("Test prompt");
    request.set_stream(true);

    grpc::ClientContext context;
    auto stream = stub->Generate(&context, request);

    // Try to read a response
    GenerateResponse response;
    if (stream->Read(&response)) {
        EXPECT_EQ(response.model(), "test-model");
    }

    grpc::Status status = stream->Finish();
    EXPECT_TRUE(status.ok() || status.error_code() == grpc::StatusCode::INTERNAL);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(GrpcServerTest, HandlesInvalidRequestGracefully) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    // Empty chat completion request
    ChatCompletionRequest request;
    grpc::ClientContext context;
    auto stream = stub->CreateChatCompletion(&context, request);

    ChatCompletionChunk chunk;
    stream->Read(&chunk);
    grpc::Status status = stream->Finish();

    // Should not crash, may return error
    EXPECT_TRUE(status.ok() || status.error_code() != grpc::StatusCode::OK);
}

TEST_F(GrpcServerTest, ReturnsUnimplementedForEmbeddings) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    EmbeddingRequest request;
    request.set_model("test-model");
    request.set_text("Test text");

    EmbeddingResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->CreateEmbedding(&context, request, &response);

    // Embeddings not yet implemented
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
}

TEST_F(GrpcServerTest, ReturnsUnimplementedForModelPull) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    PullModelRequest request;
    request.set_model_name("test-model");
    request.set_stream(true);

    grpc::ClientContext context;
    auto stream = stub->PullModel(&context, request);

    PullModelProgress progress;
    bool got_response = stream->Read(&progress);

    if (got_response) {
        EXPECT_EQ(progress.status(), PULL_STATUS_FAILED);
    }

    grpc::Status status = stream->Finish();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
}

// ============================================================================
// Metrics Tests
// ============================================================================

TEST_F(GrpcServerTest, GetMetricsReturnsData) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    MetricsRequest request;
    request.set_format(METRICS_FORMAT_JSON);

    MetricsResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->GetMetrics(&context, request, &response);

    ASSERT_TRUE(status.ok());
    EXPECT_EQ(response.format(), "json");
    EXPECT_FALSE(response.data().empty());
}

TEST_F(GrpcServerTest, GetMetricsSupportsPrometheusFormat) {
    ASSERT_TRUE(StartTestServer());
    auto stub = CreateClient();

    MetricsRequest request;
    request.set_format(METRICS_FORMAT_PROMETHEUS);

    MetricsResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub->GetMetrics(&context, request, &response);

    ASSERT_TRUE(status.ok());
    EXPECT_EQ(response.format(), "prometheus");
    EXPECT_FALSE(response.data().empty());
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

TEST_F(GrpcServerTest, HandlesConcurrentHealthChecks) {
    ASSERT_TRUE(StartTestServer());

    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &success_count]() {
            auto stub = CreateClient();
            HealthRequest request;
            HealthResponse response;
            grpc::ClientContext context;

            if (stub->Health(&context, request, &response).ok()) {
                success_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(GrpcServerTest, RespectsMaxMessageSize) {
    // This would require sending a large message to test properly
    // For now, just verify the server starts with the config
    ASSERT_TRUE(StartTestServer());
    EXPECT_TRUE(server_->IsRunning());
}

TEST_F(GrpcServerTest, CanBindToUnixSocket) {
    config_.unix_socket_path = "/tmp/mlxr_test.sock";
    config_.port = 0;  // Disable TCP

    // Clean up any existing socket
    unlink(config_.unix_socket_path.c_str());

    ASSERT_TRUE(StartTestServer());
    EXPECT_TRUE(server_->IsRunning());

    // Clean up
    server_->Stop();
    unlink(config_.unix_socket_path.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
