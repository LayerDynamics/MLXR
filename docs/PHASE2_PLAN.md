# Phase 2: Scheduler-Engine Integration Plan

## Overview

Connect the REST server to the inference engine via the scheduler to enable real text generation.

## Current State

✅ **Phase 1 Complete**:
- HTTP server running with cpp-httplib
- Routes registered for OpenAI/Ollama endpoints
- Basic request/response handling with mock data
- Health and model list endpoints functional

## Phase 2 Goals

Wire the scheduler and engine to the REST server to enable:
1. Real token generation (not mocked responses)
2. SSE streaming of generated tokens
3. Continuous batching via the scheduler
4. Multiple concurrent requests

## Architecture

```
┌─────────────┐         ┌───────────┐         ┌──────────┐
│ HTTP Client │────────▶│ RestServer│────────▶│Scheduler │
└─────────────┘   POST  └───────────┘ submit  └──────────┘
                                                     │
                                                     │ get_batch
                                                     ▼
                                              ┌──────────────┐
                                              │ Worker Thread│
                                              └──────────────┘
                                                     │
                                                     │ execute
                                                     ▼
                                              ┌──────────┐
                                              │  Engine  │
                                              └──────────┘
                                                     │
                                                     │ token_callback
                                                     ▼
                                              ┌──────────┐
                                              │   SSE    │───────▶ Client
                                              └──────────┘
```

## Implementation Steps

### Step 1: Add Scheduler to RestServer

**Files to modify**:
- `daemon/server/rest_server.h`: Add scheduler member
- `daemon/server/rest_server.cpp`: Accept scheduler in constructor

**Changes**:
```cpp
// In rest_server.h
class RestServer {
  // ...
  void set_scheduler(std::shared_ptr<scheduler::Scheduler> scheduler);

private:
  std::shared_ptr<scheduler::Scheduler> scheduler_;
  // ...
};
```

### Step 2: Create Scheduler Worker Thread

**New file**: `daemon/server/scheduler_worker.h` and `.cpp`

**Purpose**: Continuously poll scheduler for batches and execute them

```cpp
class SchedulerWorker {
public:
  SchedulerWorker(Scheduler* scheduler, Engine* engine);
  void start();
  void stop();

private:
  void run_loop();
  void execute_batch(const Batch& batch);
};
```

**Logic**:
1. Loop: call `scheduler->get_next_batch()`
2. If batch not empty, execute prefill/decode for each request
3. Call request's `token_callback` for each generated token
4. Call `scheduler->complete_batch()` when done

### Step 3: Implement Real handle_chat_completion()

**File**: `daemon/server/rest_server.cpp`

**Logic**:
1. Parse request JSON to extract:
   - model name
   - messages array
   - sampling params (temperature, top_p, max_tokens, etc.)
   - stream flag
2. Tokenize the messages using tokenizer
3. Create `scheduler::Request` with:
   - Unique request_id
   - Tokenized prompt
   - Sampling params
4. If streaming:
   - Set up SSEStream
   - Set token_callback to send SSE chunks
   - Submit request to scheduler
   - Block until request completes (or timeout)
5. If not streaming:
   - Submit request
   - Wait for completion
   - Return full response

**Token Callback for Streaming**:
```cpp
request->token_callback = [sse_stream, tokenizer](int token_id, bool finished) {
  std::string token_text = tokenizer->decode({token_id});

  ChatCompletionChunk chunk;
  chunk.choices[0].delta.content = token_text;
  chunk.choices[0].finish_reason = finished ? "stop" : "";

  std::string sse_data = serialize_chat_completion_chunk(chunk);
  sse_stream->send(sse_data);

  if (finished) {
    sse_stream->send("[DONE]");
    sse_stream->close();
  }
};
```

### Step 4: Implement Real handle_completion()

Similar to chat_completion but with simpler message format (just prompt string instead of messages array).

### Step 5: Implement handle_embedding()

**Logic**:
1. Parse embedding request
2. Tokenize input text(s)
3. Run forward pass through model to get hidden states
4. Extract final layer embeddings
5. Return embeddings as float arrays

**Note**: This doesn't use the scheduler (not generative), just direct model forward pass.

### Step 6: Wire Ollama API Handlers

**File**: `daemon/server/ollama_api.cpp`

Similar logic to OpenAI endpoints but with Ollama's request/response format:
- `/api/generate` → maps to completion
- `/api/chat` → maps to chat completion
- `/api/embeddings` → maps to embeddings

### Step 7: Update test_daemon

Add scheduler and engine initialization:
```cpp
// Create scheduler
SchedulerConfig sched_config;
auto scheduler = std::make_shared<Scheduler>(sched_config);

// Load model and engine
auto engine = load_engine("./models/TinyLlama-1.1B", "./models/tokenizer.model");

// Create worker
SchedulerWorker worker(scheduler.get(), engine.get());
worker.start();

// Set components on server
server.set_scheduler(scheduler);
server.set_engine(engine);
```

## Testing Strategy

### Unit Tests
- `scheduler_worker_test.cpp`: Test batch execution
- `rest_server_integration_test.cpp`: Test full request flow

### Integration Tests
1. Single request: verify correct token generation
2. Concurrent requests: verify batching works
3. Streaming: verify SSE chunks arrive in order
4. Cancellation: verify requests can be cancelled mid-generation

### Manual Testing
```bash
# Start daemon
./bin/test_daemon

# Test streaming
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role":"user","content":"Hello!"}],
    "stream": true
  }'

# Test non-streaming
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role":"user","content":"Hello!"}],
    "stream": false,
    "max_tokens": 50
  }'
```

## Success Criteria

- ✅ Real token generation (not mocked)
- ✅ SSE streaming works correctly
- ✅ Multiple concurrent requests are batched
- ✅ Requests complete successfully
- ✅ Cancellation works
- ✅ Error handling for failed requests
- ✅ Latency < 100ms per token for small models

## Next Steps After Phase 2

**Phase 3**: Model loading from registry
- Load models on daemon startup
- Support model switching
- Model download/management endpoints

**Phase 4**: System monitoring
- CPU/GPU metrics
- Request queue depth
- Throughput stats

**Phase 5**: Configuration and production readiness
- YAML config loading
- Logging improvements
- Graceful shutdown

## Estimated Timeline

- Step 1-2: Add scheduler worker thread (2-3 hours)
- Step 3: Implement real chat_completion (2-3 hours)
- Step 4-6: Other endpoints (2-3 hours)
- Step 7: Testing and fixes (2-3 hours)

**Total**: 8-12 hours of implementation time
