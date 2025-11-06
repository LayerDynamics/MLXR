#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <thread>

// Include generated protobuf/gRPC headers
#include "mlxrunner.grpc.pb.h"

// Include headers for types used in method signatures
#include "scheduler/request.h"      // for mlxr::scheduler::SamplingParams
#include "registry/model_registry.h" // for ModelRegistry::ModelInfo

namespace mlxr {

// Forward declarations
class Scheduler;

namespace telemetry {
class MetricsRegistry;
}

class GrpcServiceImpl;

/**
 * @brief gRPC server for MLXR daemon
 *
 * Provides gRPC endpoints alongside REST API:
 * - OpenAI-compatible streaming chat/completions
 * - Ollama-compatible generate/chat
 * - Model management RPCs
 * - Health and metrics endpoints
 *
 * Integrates with existing Scheduler and ModelRegistry.
 */
class GrpcServer {
public:
    struct Config {
        std::string host = "127.0.0.1";
        int port = 50051;
        int max_message_size = 100 * 1024 * 1024;  // 100MB
        bool enable_reflection = true;              // For grpc_cli
        std::string unix_socket_path = "";          // Optional UDS binding

        // TLS/SSL configuration (optional)
        bool enable_tls = false;
        std::string server_cert_path;
        std::string server_key_path;
        std::string client_ca_cert_path;  // For mTLS
    };

    GrpcServer(const Config& config,
               std::shared_ptr<Scheduler> scheduler,
               std::shared_ptr<registry::ModelRegistry> registry,
               std::shared_ptr<telemetry::MetricsRegistry> metrics);
    ~GrpcServer();

    // Lifecycle
    bool Start();
    void Stop();
    void Wait();  // Block until server shutdown
    bool IsRunning() const { return running_; }

    // Server info
    std::string GetBindAddress() const;
    int GetPort() const { return config_.port; }

private:
    Config config_;
    std::shared_ptr<Scheduler> scheduler_;
    std::shared_ptr<registry::ModelRegistry> registry_;
    std::shared_ptr<telemetry::MetricsRegistry> metrics_;

    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<GrpcServiceImpl> service_;

    std::thread server_thread_;
    std::atomic<bool> running_{false};

    // Build server credentials
    std::shared_ptr<grpc::ServerCredentials> BuildCredentials();

    // Build server builder with options
    void ConfigureBuilder(grpc::ServerBuilder& builder);
};

/**
 * @brief Implementation of MLXRunnerService gRPC service
 *
 * Implements all RPC methods defined in mlxrunner.proto
 */
class GrpcServiceImpl final : public mlxrunner::v1::MLXRunnerService::Service {
public:
    GrpcServiceImpl(std::shared_ptr<Scheduler> scheduler,
                    std::shared_ptr<registry::ModelRegistry> registry,
                    std::shared_ptr<telemetry::MetricsRegistry> metrics);

    // Health and Status
    grpc::Status Health(grpc::ServerContext* context,
                       const mlxrunner::v1::HealthRequest* request,
                       mlxrunner::v1::HealthResponse* response) override;

    grpc::Status GetStatus(grpc::ServerContext* context,
                          const mlxrunner::v1::StatusRequest* request,
                          mlxrunner::v1::StatusResponse* response) override;

    // Model Management
    grpc::Status ListModels(grpc::ServerContext* context,
                           const mlxrunner::v1::ListModelsRequest* request,
                           mlxrunner::v1::ListModelsResponse* response) override;

    grpc::Status GetModel(grpc::ServerContext* context,
                         const mlxrunner::v1::GetModelRequest* request,
                         mlxrunner::v1::GetModelResponse* response) override;

    grpc::Status LoadModel(grpc::ServerContext* context,
                          const mlxrunner::v1::LoadModelRequest* request,
                          mlxrunner::v1::LoadModelResponse* response) override;

    grpc::Status UnloadModel(grpc::ServerContext* context,
                            const mlxrunner::v1::UnloadModelRequest* request,
                            mlxrunner::v1::UnloadModelResponse* response) override;

    grpc::Status PullModel(grpc::ServerContext* context,
                          const mlxrunner::v1::PullModelRequest* request,
                          grpc::ServerWriter<mlxrunner::v1::PullModelProgress>* writer) override;

    // OpenAI-Compatible Endpoints
    grpc::Status CreateChatCompletion(
        grpc::ServerContext* context,
        const mlxrunner::v1::ChatCompletionRequest* request,
        grpc::ServerWriter<mlxrunner::v1::ChatCompletionChunk>* writer) override;

    grpc::Status CreateCompletion(
        grpc::ServerContext* context,
        const mlxrunner::v1::CompletionRequest* request,
        grpc::ServerWriter<mlxrunner::v1::CompletionChunk>* writer) override;

    grpc::Status CreateEmbedding(
        grpc::ServerContext* context,
        const mlxrunner::v1::EmbeddingRequest* request,
        mlxrunner::v1::EmbeddingResponse* response) override;

    // Ollama-Compatible Endpoints
    grpc::Status Generate(
        grpc::ServerContext* context,
        const mlxrunner::v1::GenerateRequest* request,
        grpc::ServerWriter<mlxrunner::v1::GenerateResponse>* writer) override;

    grpc::Status Chat(
        grpc::ServerContext* context,
        const mlxrunner::v1::ChatRequest* request,
        grpc::ServerWriter<mlxrunner::v1::ChatResponse>* writer) override;

    grpc::Status Embeddings(
        grpc::ServerContext* context,
        const mlxrunner::v1::EmbeddingsRequest* request,
        mlxrunner::v1::EmbeddingsResponse* response) override;

    grpc::Status CreateBlob(
        grpc::ServerContext* context,
        const mlxrunner::v1::CreateBlobRequest* request,
        mlxrunner::v1::CreateBlobResponse* response) override;

    grpc::Status CheckBlob(
        grpc::ServerContext* context,
        const mlxrunner::v1::CheckBlobRequest* request,
        mlxrunner::v1::CheckBlobResponse* response) override;

    // Metrics
    grpc::Status GetMetrics(
        grpc::ServerContext* context,
        const mlxrunner::v1::MetricsRequest* request,
        mlxrunner::v1::MetricsResponse* response) override;

private:
    std::shared_ptr<Scheduler> scheduler_;
    std::shared_ptr<registry::ModelRegistry> registry_;
    std::shared_ptr<telemetry::MetricsRegistry> metrics_;

    std::atomic<int64_t> requests_processed_{0};
    std::chrono::steady_clock::time_point start_time_;

    // Helper methods
    std::string GenerateRequestId() const;
    std::string GetTimestamp() const;

    // Conversion helpers
    void ConvertModelInfo(const registry::ModelInfo& src,
                         mlxrunner::v1::ModelInfo* dst);

    mlxr::scheduler::SamplingParams ConvertSamplingParams(const mlxrunner::v1::GenerateOptions& opts);
};

} // namespace mlxr
