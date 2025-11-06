#include "grpc_server.h"
#include "mlxrunner.grpc.pb.h"
#include "scheduler/scheduler.h"
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
                       std::shared_ptr<ModelRegistry> registry,
                       std::shared_ptr<MetricsCollector> metrics)
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
                                 std::shared_ptr<ModelRegistry> registry,
                                 std::shared_ptr<MetricsCollector> metrics)
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
    auto models = registry_->ListModels();
    for (const auto& model : models) {
        if (model.loaded) {
            response->add_loaded_models(model.name);
        }
    }

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::GetStatus(
    grpc::ServerContext* context,
    const mlxrunner::v1::StatusRequest* request,
    mlxrunner::v1::StatusResponse* response) {

    auto stats = scheduler_->GetStats();

    response->set_pending_requests(stats.pending_requests);
    response->set_active_requests(stats.running_requests);
    response->set_current_batch_size(stats.current_batch_size);

    response->set_kv_blocks_used(stats.kv_blocks_used);
    response->set_kv_blocks_total(stats.kv_blocks_total);
    response->set_kv_utilization_percent(
        stats.kv_blocks_total > 0 ?
        (100.0f * stats.kv_blocks_used / stats.kv_blocks_total) : 0.0f
    );

    response->set_avg_prefill_latency_ms(stats.avg_prefill_latency_ms);
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

    auto models = registry_->ListModels();

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

    auto model_opt = registry_->GetModel(request->model_id());
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

    auto start = std::chrono::steady_clock::now();

    bool success = registry_->LoadModel(request->model_id());

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    response->set_success(success);
    response->set_load_time_ms(duration.count());

    if (success) {
        response->set_message("Model loaded successfully");
        return grpc::Status::OK;
    } else {
        response->set_message("Failed to load model");
        return grpc::Status(grpc::StatusCode::INTERNAL, "Model loading failed");
    }
}

grpc::Status GrpcServiceImpl::UnloadModel(
    grpc::ServerContext* context,
    const mlxrunner::v1::UnloadModelRequest* request,
    mlxrunner::v1::UnloadModelResponse* response) {

    bool success = registry_->UnloadModel(request->model_id());

    response->set_success(success);
    response->set_message(success ? "Model unloaded" : "Failed to unload");

    return grpc::Status::OK;
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

    requests_processed_.fetch_add(1);

    // Build prompt from messages
    std::string prompt;
    for (const auto& msg : request->messages()) {
        prompt += msg.role() + ": " + msg.content() + "\n";
    }
    prompt += "assistant: ";

    // Create scheduler request
    SchedulerRequest sched_req;
    sched_req.request_id = GenerateRequestId();
    sched_req.model = request->model();
    sched_req.prompt = prompt;
    sched_req.max_tokens = request->max_tokens() > 0 ? request->max_tokens() : 512;

    // Sampling params
    sched_req.sampling_params.temperature = request->temperature();
    sched_req.sampling_params.top_p = request->top_p();

    // Token callback for streaming
    std::string accumulated_text;
    sched_req.token_callback = [&](const std::string& token, bool is_final) {
        mlxrunner::v1::ChatCompletionChunk chunk;
        chunk.set_id(sched_req.request_id);
        chunk.set_object("chat.completion.chunk");
        chunk.set_created(std::chrono::system_clock::now().time_since_epoch().count());
        chunk.set_model(request->model());

        auto* choice = chunk.add_choices();
        choice->set_index(0);

        auto* delta = choice->mutable_delta();
        delta->set_content(token);

        if (is_final) {
            choice->set_finish_reason("stop");
        }

        writer->Write(chunk);
        accumulated_text += token;
    };

    // Submit to scheduler
    scheduler_->SubmitRequest(std::move(sched_req));

    // Wait for completion (in a real implementation, this would be async)
    // For now, we just return after submission
    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::CreateCompletion(
    grpc::ServerContext* context,
    const mlxrunner::v1::CompletionRequest* request,
    grpc::ServerWriter<mlxrunner::v1::CompletionChunk>* writer) {

    requests_processed_.fetch_add(1);

    // Create scheduler request
    SchedulerRequest sched_req;
    sched_req.request_id = GenerateRequestId();
    sched_req.model = request->model();
    sched_req.prompt = request->prompt();
    sched_req.max_tokens = request->max_tokens() > 0 ? request->max_tokens() : 512;

    // Sampling params
    sched_req.sampling_params.temperature = request->temperature();
    sched_req.sampling_params.top_p = request->top_p();

    // Token callback
    sched_req.token_callback = [&](const std::string& token, bool is_final) {
        mlxrunner::v1::CompletionChunk chunk;
        chunk.set_id(sched_req.request_id);
        chunk.set_object("text_completion");
        chunk.set_created(std::chrono::system_clock::now().time_since_epoch().count());
        chunk.set_model(request->model());

        auto* choice = chunk.add_choices();
        choice->set_text(token);
        choice->set_index(0);

        if (is_final) {
            choice->set_finish_reason("stop");
        }

        writer->Write(chunk);
    };

    scheduler_->SubmitRequest(std::move(sched_req));

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

    requests_processed_.fetch_add(1);

    // Create scheduler request
    SchedulerRequest sched_req;
    sched_req.request_id = GenerateRequestId();
    sched_req.model = request->model();
    sched_req.prompt = request->prompt();

    if (request->has_options()) {
        sched_req.sampling_params = ConvertSamplingParams(request->options());
    }

    // Token callback
    auto start_time = std::chrono::steady_clock::now();
    sched_req.token_callback = [&](const std::string& token, bool is_final) {
        mlxrunner::v1::GenerateResponse resp;
        resp.set_model(request->model());
        resp.set_created_at(GetTimestamp());
        resp.set_response(token);
        resp.set_done(is_final);

        if (is_final) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time);
            resp.set_total_duration(duration.count());
        }

        writer->Write(resp);
    };

    scheduler_->SubmitRequest(std::move(sched_req));

    return grpc::Status::OK;
}

grpc::Status GrpcServiceImpl::Chat(
    grpc::ServerContext* context,
    const mlxrunner::v1::ChatRequest* request,
    grpc::ServerWriter<mlxrunner::v1::ChatResponse>* writer) {

    requests_processed_.fetch_add(1);

    // Build prompt from messages
    std::string prompt;
    for (const auto& msg : request->messages()) {
        prompt += msg.role() + ": " + msg.content() + "\n";
    }
    prompt += "assistant: ";

    // Create scheduler request
    SchedulerRequest sched_req;
    sched_req.request_id = GenerateRequestId();
    sched_req.model = request->model();
    sched_req.prompt = prompt;

    if (request->has_options()) {
        sched_req.sampling_params = ConvertSamplingParams(request->options());
    }

    // Token callback
    auto start_time = std::chrono::steady_clock::now();
    std::string accumulated;
    sched_req.token_callback = [&](const std::string& token, bool is_final) {
        accumulated += token;

        mlxrunner::v1::ChatResponse resp;
        resp.set_model(request->model());
        resp.set_created_at(GetTimestamp());
        resp.set_done(is_final);

        auto* msg = resp.mutable_message();
        msg->set_role("assistant");
        msg->set_content(accumulated);

        if (is_final) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time);
            resp.set_total_duration(duration.count());
        }

        writer->Write(resp);
    };

    scheduler_->SubmitRequest(std::move(sched_req));

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
        response->set_data(metrics_->ExportPrometheus());
    } else {
        response->set_format("json");
        response->set_data(metrics_->ExportJSON());
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

void GrpcServiceImpl::ConvertModelInfo(const ModelRegistry::ModelInfo& src,
                                       mlxrunner::v1::ModelInfo* dst) {
    dst->set_id(src.id);
    dst->set_name(src.name);
    dst->set_family(src.family);
    dst->set_architecture(src.architecture);

    // Convert format
    if (src.format == "gguf") {
        dst->set_format(mlxrunner::v1::MODEL_FORMAT_GGUF);
    } else if (src.format == "safetensors") {
        dst->set_format(mlxrunner::v1::MODEL_FORMAT_SAFETENSORS);
    } else if (src.format == "mlx") {
        dst->set_format(mlxrunner::v1::MODEL_FORMAT_MLX);
    }

    dst->set_path(src.path);
    dst->set_dtype(src.dtype);
    dst->set_quantization(src.quantization);
    dst->set_parameters(src.parameters);
    dst->set_max_context_length(src.max_context_length);
    dst->set_num_layers(src.num_layers);
    dst->set_vocab_size(src.vocab_size);
    dst->set_file_size_bytes(src.file_size_bytes);
    dst->set_created_at(src.created_at);

    for (const auto& tag : src.tags) {
        dst->add_tags(tag);
    }
}

SamplingParams GrpcServiceImpl::ConvertSamplingParams(
    const mlxrunner::v1::GenerateOptions& opts) {

    SamplingParams params;
    params.temperature = opts.temperature();
    params.top_p = opts.top_p();
    params.top_k = opts.top_k();
    params.repetition_penalty = opts.repeat_penalty();
    params.max_tokens = opts.num_predict();

    return params;
}

} // namespace mlxr
