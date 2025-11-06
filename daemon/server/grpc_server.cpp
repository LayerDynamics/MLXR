#include "grpc_server.h"
#include "mlxrunner.grpc.pb.h"
#include "scheduler/scheduler.h"
#include "scheduler/request.h"
#include "registry/model_registry.h"
#include "telemetry/metrics.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <sstream>
#include <uuid/uuid.h>
#include <chrono>

namespace mlxr {

// ============================================================================
// GrpcServer Implementation
// ============================================================================

GrpcServer::GrpcServer(const Config& config,
                       std::shared_ptr<Scheduler> scheduler,
                       std::shared_ptr<registry::ModelRegistry> registry,
                       std::shared_ptr<telemetry::MetricsRegistry> metrics)
    : config_(config),
      scheduler_(scheduler),
      registry_(registry),
      metrics_(metrics) {
}

GrpcServer::~GrpcServer() {
    Stop();
}

bool GrpcServer::Start() {
    if (running_.load()) {
        return false;  // Already running
    }

    try {
        grpc::ServerBuilder builder;

        // Bind address
        std::string server_address;
        if (!config_.unix_socket_path.empty()) {
            // Unix Domain Socket
            server_address = "unix:" + config_.unix_socket_path;
        } else {
            // TCP
            server_address = config_.host + ":" + std::to_string(config_.port);
        }

        // Add listening port
        auto creds = BuildCredentials();
        builder.AddListeningPort(server_address, creds);

        // Register service
        service_ = std::make_unique<GrpcServiceImpl>(scheduler_, registry_, metrics_);
        builder.RegisterService(service_.get());

        // Configure options
        ConfigureBuilder(builder);

        // Build and start server
        server_ = builder.BuildAndStart();
        if (!server_) {
            return false;
        }

        running_.store(true);
        std::cout << "gRPC server listening on " << server_address << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start gRPC server: " << e.what() << std::endl;
        return false;
    }
}

void GrpcServer::Stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (server_) {
        server_->Shutdown();
        server_.reset();
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void GrpcServer::Wait() {
    if (server_) {
        server_->Wait();
    }
}

std::string GrpcServer::GetBindAddress() const {
    if (!config_.unix_socket_path.empty()) {
        return "unix:" + config_.unix_socket_path;
    }
    return config_.host + ":" + std::to_string(config_.port);
}

std::shared_ptr<grpc::ServerCredentials> GrpcServer::BuildCredentials() {
    if (config_.enable_tls) {
        grpc::SslServerCredentialsOptions ssl_opts;
        ssl_opts.pem_root_certs = "";  // No client verification by default

        grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert_pair;
        // Load server cert and key from files
        // TODO: Implement file reading
        key_cert_pair.private_key = "";  // Read from config_.server_key_path
        key_cert_pair.cert_chain = "";    // Read from config_.server_cert_path

        ssl_opts.pem_key_cert_pairs.push_back(key_cert_pair);

        return grpc::SslServerCredentials(ssl_opts);
    } else {
        return grpc::InsecureServerCredentials();
    }
}

void GrpcServer::ConfigureBuilder(grpc::ServerBuilder& builder) {
    // Set max message size
    builder.SetMaxReceiveMessageSize(config_.max_message_size);
    builder.SetMaxSendMessageSize(config_.max_message_size);

    // Enable health checking
    grpc::EnableDefaultHealthCheckService(true);

    // Enable reflection for debugging with grpc_cli
    if (config_.enable_reflection) {
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    }

    // Performance options
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);  // 30 seconds
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);  // 10 seconds
    builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 5000);
}

// ============================================================================
// GrpcServiceImpl Implementation
// ============================================================================

GrpcServiceImpl::GrpcServiceImpl(std::shared_ptr<Scheduler> scheduler,
                                 std::shared_ptr<registry::ModelRegistry> registry,
                                 std::shared_ptr<telemetry::MetricsRegistry> metrics)
    : scheduler_(scheduler),
      registry_(registry),
      metrics_(metrics),
      start_time_(std::chrono::steady_clock::now()) {
}

// ----------------------------------------------------------------------------
// Health and Status
// ----------------------------------------------------------------------------

grpc::Status GrpcServiceImpl::Health(
    grpc::ServerContext* context,
    const mlxrunner::v1::HealthRequest* request,
    mlxrunner::v1::HealthResponse* response) {

    response->set_status("ok");

    // Uptime
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    response->set_uptime_seconds(uptime.count());

    // Requests processed
    response->set_requests_processed(requests_processed_.load());

    // Loaded models
    auto models = registry_->list_models();
    for (const auto& model : models) {
        if (model.is_loaded) {
            response->add_loaded_models(model.name);
        }
    }

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::GetStatus(
    grpc::ServerContext* context,
    const mlxrunner::v1::StatusRequest* request,
    mlxrunner::v1::StatusResponse* response) {

    auto stats = scheduler_->get_stats();

    response->set_pending_requests(stats.waiting_requests);
    response->set_active_requests(stats.prefilling_requests + stats.decoding_requests);
    response->set_current_batch_size(stats.prefilling_requests + stats.decoding_requests);

    response->set_kv_blocks_used(stats.used_kv_blocks);
    response->set_kv_blocks_total(stats.available_kv_blocks + stats.used_kv_blocks);
    response->set_kv_utilization_percent(stats.kv_utilization * 100.0f);

    response->set_avg_prefill_latency_ms(stats.avg_prefill_time_ms);
    response->set_avg_decode_latency_ms(stats.avg_decode_latency_ms);
    response->set_tokens_per_second(stats.tokens_per_second);

    return grpc::Status::OK;
}

// ----------------------------------------------------------------------------
// Model Management
// ----------------------------------------------------------------------------

grpc::Status GrpcServiceImpl::ListModels(
    grpc::ServerContext* context,
    const mlxrunner::v1::ListModelsRequest* request,
    mlxrunner::v1::ListModelsResponse* response) {

    auto models = registry_->list_models();

    int offset = request->offset();
    int limit = request->limit() > 0 ? request->limit() : models.size();

    for (size_t i = offset; i < models.size() && i < offset + limit; ++i) {
        auto* model_info = response->add_models();
        ConvertModelInfo(models[i], model_info);
    }

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::GetModel(
    grpc::ServerContext* context,
    const mlxrunner::v1::GetModelRequest* request,
    mlxrunner::v1::GetModelResponse* response) {

    auto model_opt = registry_->get_model_by_identifier(request->model_id());
    if (!model_opt.has_value()) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                          "Model not found: " + request->model_id());
    }

    ConvertModelInfo(model_opt.value(), response->mutable_model());
    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::LoadModel(
    grpc::ServerContext* context,
    const mlxrunner::v1::LoadModelRequest* request,
    mlxrunner::v1::LoadModelResponse* response) {

    // TODO: Implement model loading
    // ModelRegistry doesn't have a LoadModel method yet
    // This requires integration with Engine for actual loading

    response->set_success(false);
    response->set_load_time_ms(0);
    response->set_message("Model loading not yet implemented");

    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                       "Model loading not yet implemented");
}

grpc::Status GrpcServiceImpl::UnloadModel(
    grpc::ServerContext* context,
    const mlxrunner::v1::UnloadModelRequest* request,
    mlxrunner::v1::UnloadModelResponse* response) {

    // TODO: Implement model unloading
    // ModelRegistry doesn't have an UnloadModel method yet
    // This requires integration with Engine for actual unloading

    response->set_success(false);
    response->set_message("Model unloading not yet implemented");

    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                       "Model unloading not yet implemented");
}

grpc::Status GrpcServiceImpl::PullModel(
    grpc::ServerContext* context,
    const mlxrunner::v1::PullModelRequest* request,
    grpc::ServerWriter<mlxrunner::v1::PullModelProgress>* writer) {

    // TODO: Implement model pulling from Hugging Face/Ollama
    // For now, return not implemented

    mlxrunner::v1::PullModelProgress progress;
    progress.set_status(mlxrunner::v1::PULL_STATUS_FAILED);
    progress.set_message("Model pulling not yet implemented");
    progress.set_percent_complete(0);

    writer->Write(progress);

    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                       "Model pulling not yet implemented");
}

// ----------------------------------------------------------------------------
// OpenAI-Compatible Endpoints
// ----------------------------------------------------------------------------

grpc::Status GrpcServiceImpl::CreateChatCompletion(
    grpc::ServerContext* context,
    const mlxrunner::v1::ChatCompletionRequest* request,
    grpc::ServerWriter<mlxrunner::v1::ChatCompletionChunk>* writer) {

    using namespace mlxr::scheduler;

    requests_processed_.fetch_add(1);

    // Build prompt from messages
    std::string prompt;
    for (const auto& msg : request->messages()) {
        prompt += msg.role() + ": " + msg.content() + "\n";
    }
    prompt += "assistant: ";

    // TODO: Get tokenizer from model registry and tokenize prompt
    // For now, use placeholder tokens (this needs model-specific tokenizer)
    std::vector<int> prompt_tokens;  // Would be: tokenizer->encode(prompt);

    // Create sampling parameters
    SamplingParams sampling_params;
    sampling_params.temperature = request->temperature() > 0 ? request->temperature() : 0.7f;
    sampling_params.top_p = request->top_p() > 0 ? request->top_p() : 0.9f;
    sampling_params.max_tokens = request->max_tokens() > 0 ? request->max_tokens() : 512;

    // Create scheduler request
    std::string request_id = GenerateRequestId();
    auto sched_req = std::make_shared<Request>(
        request_id,
        prompt,
        prompt_tokens,
        sampling_params
    );

    // Token callback for streaming
    // Note: The callback signature is (int token_id, bool finished)
    // We need to decode token_id to string, which requires a tokenizer
    sched_req->token_callback = [writer, request_id, model = request->model()](int token_id, bool finished) {
        mlxrunner::v1::ChatCompletionChunk chunk;
        chunk.set_id(request_id);
        chunk.set_object("chat.completion.chunk");
        chunk.set_created(std::chrono::system_clock::now().time_since_epoch().count());
        chunk.set_model(model);

        auto* choice = chunk.add_choices();
        choice->set_index(0);

        auto* delta = choice->mutable_delta();
        // TODO: Decode token_id to text using tokenizer
        delta->set_content(std::to_string(token_id));  // Placeholder

        if (finished) {
            choice->set_finish_reason("stop");
        }

        writer->Write(chunk);
    };

    // Submit to scheduler
    if (!scheduler_->submit_request(sched_req)) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Scheduler queue is full");
    }

    // TODO: Wait for completion asynchronously
    // Current implementation returns immediately
    // Production version should wait for request to complete or stream tokens
    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::CreateCompletion(
    grpc::ServerContext* context,
    const mlxrunner::v1::CompletionRequest* request,
    grpc::ServerWriter<mlxrunner::v1::CompletionChunk>* writer) {

    using namespace mlxr::scheduler;

    requests_processed_.fetch_add(1);

    // TODO: Get tokenizer from model registry and tokenize prompt
    std::vector<int> prompt_tokens;  // Would be: tokenizer->encode(request->prompt());

    // Create sampling parameters
    SamplingParams sampling_params;
    sampling_params.temperature = request->temperature() > 0 ? request->temperature() : 0.7f;
    sampling_params.top_p = request->top_p() > 0 ? request->top_p() : 0.9f;
    sampling_params.max_tokens = request->max_tokens() > 0 ? request->max_tokens() : 512;

    // Create scheduler request
    std::string request_id = GenerateRequestId();
    auto sched_req = std::make_shared<Request>(
        request_id,
        request->prompt(),
        prompt_tokens,
        sampling_params
    );

    // Token callback for streaming
    sched_req->token_callback = [writer, request_id, model = request->model()](int token_id, bool finished) {
        mlxrunner::v1::CompletionChunk chunk;
        chunk.set_id(request_id);
        chunk.set_object("text_completion");
        chunk.set_created(std::chrono::system_clock::now().time_since_epoch().count());
        chunk.set_model(model);

        auto* choice = chunk.add_choices();
        // TODO: Decode token_id to text using tokenizer
        choice->set_text(std::to_string(token_id));  // Placeholder
        choice->set_index(0);

        if (finished) {
            choice->set_finish_reason("stop");
        }

        writer->Write(chunk);
    };

    // Submit to scheduler
    if (!scheduler_->submit_request(sched_req)) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Scheduler queue is full");
    }

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::CreateEmbedding(
    grpc::ServerContext* context,
    const mlxrunner::v1::EmbeddingRequest* request,
    mlxrunner::v1::EmbeddingResponse* response) {

    // TODO: Implement embedding generation
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                       "Embeddings not yet implemented");
}

// ----------------------------------------------------------------------------
// Ollama-Compatible Endpoints
// ----------------------------------------------------------------------------

grpc::Status GrpcServiceImpl::Generate(
    grpc::ServerContext* context,
    const mlxrunner::v1::GenerateRequest* request,
    grpc::ServerWriter<mlxrunner::v1::GenerateResponse>* writer) {

    using namespace mlxr::scheduler;

    requests_processed_.fetch_add(1);

    // TODO: Get tokenizer from model registry and tokenize prompt
    std::vector<int> prompt_tokens;  // Would be: tokenizer->encode(request->prompt());

    // Get sampling parameters
    SamplingParams sampling_params;
    if (request->has_options()) {
        sampling_params = ConvertSamplingParams(request->options());
    } else {
        // Use defaults
        sampling_params.temperature = 0.7f;
        sampling_params.top_p = 0.9f;
        sampling_params.max_tokens = 512;
    }

    // Create scheduler request
    std::string request_id = GenerateRequestId();
    auto sched_req = std::make_shared<Request>(
        request_id,
        request->prompt(),
        prompt_tokens,
        sampling_params
    );

    // Token callback for streaming
    auto start_time = std::chrono::steady_clock::now();
    sched_req->token_callback = [this, writer, model = request->model(), start_time](int token_id, bool finished) {
        mlxrunner::v1::GenerateResponse resp;
        resp.set_model(model);
        resp.set_created_at(GetTimestamp());
        // TODO: Decode token_id to text using tokenizer
        resp.set_response(std::to_string(token_id));  // Placeholder
        resp.set_done(finished);

        if (finished) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time);
            resp.set_total_duration(duration.count());
        }

        writer->Write(resp);
    };

    // Submit to scheduler
    if (!scheduler_->submit_request(sched_req)) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Scheduler queue is full");
    }

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::Chat(
    grpc::ServerContext* context,
    const mlxrunner::v1::ChatRequest* request,
    grpc::ServerWriter<mlxrunner::v1::ChatResponse>* writer) {

    using namespace mlxr::scheduler;

    requests_processed_.fetch_add(1);

    // Build prompt from messages
    std::string prompt;
    for (const auto& msg : request->messages()) {
        prompt += msg.role() + ": " + msg.content() + "\n";
    }
    prompt += "assistant: ";

    // TODO: Get tokenizer from model registry and tokenize prompt
    std::vector<int> prompt_tokens;  // Would be: tokenizer->encode(prompt);

    // Get sampling parameters
    SamplingParams sampling_params;
    if (request->has_options()) {
        sampling_params = ConvertSamplingParams(request->options());
    } else {
        // Use defaults
        sampling_params.temperature = 0.7f;
        sampling_params.top_p = 0.9f;
        sampling_params.max_tokens = 512;
    }

    // Create scheduler request
    std::string request_id = GenerateRequestId();
    auto sched_req = std::make_shared<Request>(
        request_id,
        prompt,
        prompt_tokens,
        sampling_params
    );

    // Token callback for streaming
    auto start_time = std::chrono::steady_clock::now();
    auto accumulated = std::make_shared<std::string>();  // Shared for lambda capture
    sched_req->token_callback = [this, writer, model = request->model(), start_time, accumulated](int token_id, bool finished) {
        // TODO: Decode token_id to text using tokenizer
        *accumulated += std::to_string(token_id);  // Placeholder

        mlxrunner::v1::ChatResponse resp;
        resp.set_model(model);
        resp.set_created_at(GetTimestamp());
        resp.set_done(finished);

        auto* msg = resp.mutable_message();
        msg->set_role("assistant");
        msg->set_content(*accumulated);

        if (finished) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time);
            resp.set_total_duration(duration.count());
        }

        writer->Write(resp);
    };

    // Submit to scheduler
    if (!scheduler_->submit_request(sched_req)) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Scheduler queue is full");
    }

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::Embeddings(
    grpc::ServerContext* context,
    const mlxrunner::v1::EmbeddingsRequest* request,
    mlxrunner::v1::EmbeddingsResponse* response) {

    // TODO: Implement embeddings
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                       "Embeddings not yet implemented");
}

grpc::Status GrpcServiceImpl::CreateBlob(
    grpc::ServerContext* context,
    const mlxrunner::v1::CreateBlobRequest* request,
    mlxrunner::v1::CreateBlobResponse* response) {

    // TODO: Implement blob storage (for Ollama model uploads)
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                       "Blob storage not yet implemented");
}

grpc::Status GrpcServiceImpl::CheckBlob(
    grpc::ServerContext* context,
    const mlxrunner::v1::CheckBlobRequest* request,
    mlxrunner::v1::CheckBlobResponse* response) {

    response->set_exists(false);
    return grpc::Status::OK;
}

// ----------------------------------------------------------------------------
// Metrics
// ----------------------------------------------------------------------------

grpc::Status GrpcServiceImpl::GetMetrics(
    grpc::ServerContext* context,
    const mlxrunner::v1::MetricsRequest* request,
    mlxrunner::v1::MetricsResponse* response) {

    if (request->format() == mlxrunner::v1::METRICS_FORMAT_PROMETHEUS) {
        response->set_format("prometheus");
        response->set_data(metrics_->export_prometheus());
    } else {
        response->set_format("json");
        response->set_data(metrics_->export_json());
    }

    return grpc::Status::OK;
}

// ----------------------------------------------------------------------------
// Helper Methods
// ----------------------------------------------------------------------------

std::string GrpcServiceImpl::GenerateRequestId() const {
    uuid_t uuid;
    uuid_generate(uuid);
    char uuid_str[37];
    uuid_unparse(uuid, uuid_str);
    return std::string(uuid_str);
}

std::string GrpcServiceImpl::GetTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

void GrpcServiceImpl::ConvertModelInfo(const registry::ModelInfo& src,
                                       mlxrunner::v1::ModelInfo* dst) {
    dst->set_id(std::to_string(src.id));
    dst->set_name(src.name);

    // Convert architecture enum to string
    // For now, just use a placeholder - proper conversion would use a lookup table
    dst->set_family("llama");  // TODO: Convert from src.architecture enum
    dst->set_architecture("llama");  // TODO: Convert from src.architecture enum

    // Convert format enum to proto enum
    switch (src.format) {
        case registry::ModelFormat::GGUF:
            dst->set_format(mlxrunner::v1::MODEL_FORMAT_GGUF);
            break;
        case registry::ModelFormat::SAFETENSORS:
            dst->set_format(mlxrunner::v1::MODEL_FORMAT_SAFETENSORS);
            break;
        case registry::ModelFormat::MLX:
            dst->set_format(mlxrunner::v1::MODEL_FORMAT_MLX);
            break;
        default:
            dst->set_format(mlxrunner::v1::MODEL_FORMAT_UNKNOWN);
            break;
    }

    dst->set_path(src.file_path);
    dst->set_dtype("fp16");  // TODO: Determine from model info
    dst->set_quantization(src.quant_details);
    dst->set_parameters(src.param_count);
    dst->set_max_context_length(src.context_length);
    dst->set_num_layers(src.num_layers);
    dst->set_vocab_size(src.vocab_size);
    dst->set_file_size_bytes(src.file_size);
    dst->set_created_at(src.created_timestamp);

    for (const auto& tag : src.tags) {
        dst->add_tags(tag);
    }
}

mlxr::scheduler::SamplingParams GrpcServiceImpl::ConvertSamplingParams(
    const mlxrunner::v1::GenerateOptions& opts) {

    mlxr::scheduler::SamplingParams params;
    params.temperature = opts.temperature() > 0 ? opts.temperature() : 0.7f;
    params.top_p = opts.top_p() > 0 ? opts.top_p() : 0.9f;
    params.top_k = opts.top_k() > 0 ? opts.top_k() : 40;
    params.repetition_penalty = opts.repeat_penalty() > 0 ? opts.repeat_penalty() : 1.1f;
    params.max_tokens = opts.num_predict() > 0 ? opts.num_predict() : 512;

    return params;
}

} // namespace mlxr
