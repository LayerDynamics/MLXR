// Copyright © 2025 MLXR Development
// MLX Primitive-based custom attention decode kernel implementation
//
// This implementation uses MLX's Primitive API with direct Metal buffer access
// via MLX's compute encoder. Metal-cpp headers are bundled with MLX and provide
// the Metal C++ API for custom kernel dispatch.

#include "attention_decode_primitive.h"

#include <mlx/ops.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/transforms.h>  // For eval

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mlxr {
namespace kernels {

// ============================================================================
// Metal Kernel Argument Structure
// ============================================================================

/**
 * AttentionDecodeArgs structure matching Metal shader
 * Must maintain exact same layout as Metal struct (UPDATED for zero-copy)
 */
struct AttentionDecodeArgs {
  void* q;                    // device const half*
  void* k_cache;              // device const half*
  void* v_cache;              // device const half*
  void* output;               // device half*
  void* page_table;           // device const int*
  void* seq_lengths;          // device const int*

  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t block_size;
  uint32_t max_blocks_per_seq;
  uint32_t num_layers;        // NEW: for block format
  uint32_t layer_idx;         // NEW: for block format
  bool use_block_format;      // NEW: format selector
  float scale;
  bool use_sliding_window;
  uint32_t sliding_window_size;
};

// ============================================================================
// Constructor & Destructor
// ============================================================================

AttentionDecodePrimitive::AttentionDecodePrimitive(
    mlx::core::Stream stream,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,
    bool use_block_format,
    bool use_sliding_window,
    int sliding_window_size)
    : mlx::core::Primitive(stream),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      block_size_(block_size),
      max_blocks_per_seq_(max_blocks_per_seq),
      num_layers_(num_layers),
      layer_idx_(layer_idx),
      use_block_format_(use_block_format),
      use_sliding_window_(use_sliding_window),
      sliding_window_size_(sliding_window_size),
      library_(nullptr) {
}

AttentionDecodePrimitive::~AttentionDecodePrimitive() {
  // Metal library is managed by MLX's device, no explicit cleanup needed
}

// ============================================================================
// Metal Kernel Loading
// ============================================================================

void* AttentionDecodePrimitive::load_metal_library() {
  if (library_) {
    return library_;
  }

  @autoreleasepool {
    // Get Metal device
    auto& d = mlx::core::metal::device(stream().device);

    // Find metallib file - search multiple paths
    NSArray<NSString*>* search_paths = @[
      @"build/lib/attention_decode.metallib",
      @"../../lib/attention_decode.metallib",
      @"../lib/attention_decode.metallib",
      @"lib/attention_decode.metallib",
      [@(getenv("PWD") ?: ".") stringByAppendingString:@"/build/lib/attention_decode.metallib"]
    ];

    NSString* metallib_path = nil;
    for (NSString* path in search_paths) {
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        metallib_path = path;
        NSLog(@"Found attention_decode.metallib at: %@", path);
        break;
      }
    }

    if (!metallib_path) {
      NSLog(@"Failed to find attention_decode.metallib in any of these paths:");
      for (NSString* path in search_paths) {
        NSLog(@"  - %@", path);
      }
      throw std::runtime_error(
          "Failed to find attention_decode.metallib. Please run 'make metal'");
    }

    // Load Metal library via MLX's device
    NSURL* url = [NSURL fileURLWithPath:metallib_path];
    NSError* error = nil;

    // Get the raw MTL::Device pointer
    auto mtl_device = d.mtl_device();

    // Load library using Objective-C bridge
    id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
    id<MTLLibrary> library_objc = [device_objc newLibraryWithURL:url error:&error];

    if (!library_objc) {
      NSString* err_msg = error ? [error localizedDescription] : @"Unknown error";
      throw std::runtime_error(
          "Failed to load Metal library: " +
          std::string([err_msg UTF8String]));
    }

    // Convert to metal-cpp type and store
    MTL::Library* library_cpp = (__bridge MTL::Library*)library_objc;
    library_ = static_cast<void*>(library_cpp);

    return library_;
  }
}

// ============================================================================
// CPU Evaluation (Fallback)
// ============================================================================

void AttentionDecodePrimitive::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size() == 5);
  assert(outputs.size() == 1);

  auto& q = inputs[0];            // [batch, num_heads, head_dim]
  auto& k_cache = inputs[1];      // [num_pages, block_size, num_kv_heads, head_dim]
  auto& v_cache = inputs[2];      // [num_pages, block_size, num_kv_heads, head_dim]
  auto& page_table = inputs[3];   // [batch, max_blocks_per_seq]
  auto& seq_lengths = inputs[4];  // [batch]
  auto& output = outputs[0];      // [batch, num_heads, head_dim]

  // Allocate output buffer
  auto buffer = mlx::core::allocator::malloc(output.nbytes());
  output.set_data(buffer);

  // For CPU fallback, we implement a simple reference version
  // This is not optimized but ensures correctness for testing

  int batch_size = q.shape(0);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  // Process each batch element
  for (int b = 0; b < batch_size; b++) {
    int seq_len = seq_lengths.data<int>()[b];

    if (seq_len <= 0) {
      // Empty sequence, write zeros
      for (int h = 0; h < num_heads_; h++) {
        for (int d = 0; d < head_dim_; d++) {
          int out_idx = b * num_heads_ * head_dim_ + h * head_dim_ + d;
          output.data<float>()[out_idx] = 0.0f;
        }
      }
      continue;
    }

    // Process each query head
    for (int h = 0; h < num_heads_; h++) {
      int kv_h = h / (num_heads_ / num_kv_heads_);  // GQA mapping

      // Compute attention scores for all past tokens
      std::vector<float> scores(seq_len);
      float max_score = -std::numeric_limits<float>::infinity();

      for (int t = 0; t < seq_len; t++) {
        int block_idx = t / block_size_;
        int block_offset = t % block_size_;
        int page_id = page_table.data<int>()[b * max_blocks_per_seq_ + block_idx];

        if (page_id < 0) {
          scores[t] = -std::numeric_limits<float>::infinity();
          continue;
        }

        // Compute Q @ K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim_; d++) {
          int q_idx = b * num_heads_ * head_dim_ + h * head_dim_ + d;
          int k_idx = page_id * block_size_ * num_kv_heads_ * head_dim_ +
                     block_offset * num_kv_heads_ * head_dim_ +
                     kv_h * head_dim_ + d;

          float q_val = q.data<float>()[q_idx];
          float k_val = k_cache.data<float>()[k_idx];
          score += q_val * k_val;
        }

        score *= scale;
        scores[t] = score;
        max_score = std::max(max_score, score);
      }

      // Softmax: exp and normalize
      float sum_exp = 0.0f;
      for (int t = 0; t < seq_len; t++) {
        scores[t] = std::exp(scores[t] - max_score);
        sum_exp += scores[t];
      }

      float inv_sum = 1.0f / (sum_exp + 1e-8f);
      for (int t = 0; t < seq_len; t++) {
        scores[t] *= inv_sum;
      }

      // Compute context = softmax(scores) @ V
      for (int d = 0; d < head_dim_; d++) {
        float accum = 0.0f;

        for (int t = 0; t < seq_len; t++) {
          int block_idx = t / block_size_;
          int block_offset = t % block_size_;
          int page_id = page_table.data<int>()[b * max_blocks_per_seq_ + block_idx];

          if (page_id < 0) {
            continue;
          }

          int v_idx = page_id * block_size_ * num_kv_heads_ * head_dim_ +
                     block_offset * num_kv_heads_ * head_dim_ +
                     kv_h * head_dim_ + d;

          float v_val = v_cache.data<float>()[v_idx];
          accum += scores[t] * v_val;
        }

        int out_idx = b * num_heads_ * head_dim_ + h * head_dim_ + d;
        output.data<float>()[out_idx] = accum;
      }
    }
  }
}

// ============================================================================
// GPU Evaluation (Custom Metal Kernel)
// ============================================================================

void AttentionDecodePrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  NSLog(@"[AttentionDecode] eval_gpu() called - using Metal kernel");

  assert(inputs.size() == 5);
  assert(outputs.size() == 1);

  auto& q = inputs[0];            // [batch, num_heads, head_dim]
  auto& k_cache = inputs[1];      // [num_pages, block_size, num_kv_heads, head_dim]
  auto& v_cache = inputs[2];      // [num_pages, block_size, num_kv_heads, head_dim]
  auto& page_table = inputs[3];   // [batch, max_blocks_per_seq]
  auto& seq_lengths = inputs[4];  // [batch]
  auto& output = outputs[0];      // [batch, num_heads, head_dim]

  // Check contiguity (Phase 1 limitation)
  if (!q.flags().row_contiguous ||
      !k_cache.flags().row_contiguous ||
      !v_cache.flags().row_contiguous ||
      !page_table.flags().row_contiguous ||
      !seq_lengths.flags().row_contiguous) {
    throw std::runtime_error(
        "AttentionDecodePrimitive requires contiguous inputs. "
        "This is a Phase 1 limitation.");
  }

  // Allocate output buffer
  auto buffer = mlx::core::allocator::malloc(output.nbytes());
  output.set_data(buffer);

  // Get stream and device
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  // Load our custom Metal library
  auto* mtl_lib = static_cast<MTL::Library*>(load_metal_library());

  // For Phase 1, use single general-purpose kernel
  // Phase 2 will add optimized variants for different head dimensions
  std::string kernel_name = "attention_decode_fused";

  // Get compiled kernel from MLX's device (this caches it)
  auto* kernel = d.get_kernel(kernel_name, mtl_lib);

  // Get MLX's command encoder and set pipeline
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Prepare kernel arguments
  int batch_size = q.shape(0);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  // Bind buffers via argument buffer (index 0)
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_cache, 1);
  compute_encoder.set_input_array(v_cache, 2);
  compute_encoder.set_output_array(output, 3);
  compute_encoder.set_input_array(page_table, 4);
  compute_encoder.set_input_array(seq_lengths, 5);

  // Set scalar parameters
  compute_encoder.set_bytes(static_cast<uint32_t>(batch_size), 6);
  compute_encoder.set_bytes(static_cast<uint32_t>(num_heads_), 7);
  compute_encoder.set_bytes(static_cast<uint32_t>(num_kv_heads_), 8);
  compute_encoder.set_bytes(static_cast<uint32_t>(head_dim_), 9);
  compute_encoder.set_bytes(static_cast<uint32_t>(block_size_), 10);
  compute_encoder.set_bytes(static_cast<uint32_t>(max_blocks_per_seq_), 11);
  compute_encoder.set_bytes(static_cast<uint32_t>(num_layers_), 12);     // NEW
  compute_encoder.set_bytes(static_cast<uint32_t>(layer_idx_), 13);      // NEW
  compute_encoder.set_bytes(use_block_format_, 14);                      // NEW
  compute_encoder.set_bytes(scale, 15);
  compute_encoder.set_bytes(use_sliding_window_, 16);
  compute_encoder.set_bytes(static_cast<uint32_t>(sliding_window_size_), 17);

  // Dispatch configuration
  // Each threadgroup handles one query head
  size_t num_threadgroups = batch_size * num_heads_;
  size_t threads_per_group = 256;  // Configurable, but 256 is good default

  MTL::Size grid_dims(num_threadgroups, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);

  NSLog(@"[AttentionDecode] Dispatch params: batch=%d, heads=%d, kv_heads=%d, head_dim=%d",
        batch_size, num_heads_, num_kv_heads_, head_dim_);
  NSLog(@"[AttentionDecode] Grid dims: (%zu, %zu, %zu), Group dims: (%zu, %zu, %zu)",
        grid_dims.width, grid_dims.height, grid_dims.depth,
        group_dims.width, group_dims.height, group_dims.depth);

  // Allocate threadgroup memory
  // shared_scores: STRIPE_SIZE (64) * sizeof(float)
  // shared_reduce: threads_per_group * sizeof(float)
  size_t stripe_size = 64;
  size_t shared_scores_size = stripe_size * sizeof(float);
  size_t shared_reduce_size = threads_per_group * sizeof(float);

  compute_encoder.set_threadgroup_memory_length(shared_scores_size, 0);
  compute_encoder.set_threadgroup_memory_length(shared_reduce_size, 1);

  NSLog(@"[AttentionDecode] Threadgroup memory: scores=%zu bytes, reduce=%zu bytes",
        shared_scores_size, shared_reduce_size);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  NSLog(@"[AttentionDecode] Dispatch complete");
}

// ============================================================================
// Function Transformations
// ============================================================================

std::pair<std::vector<mlx::core::array>, std::vector<int>>
AttentionDecodePrimitive::vmap(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<int>& axes) {

  // Attention decode can be vmapped over batch dimension
  auto out = attention_decode_fused(
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
      num_heads_, num_kv_heads_, head_dim_, block_size_,
      max_blocks_per_seq_, num_layers_, layer_idx_, use_block_format_,
      use_sliding_window_, sliding_window_size_,
      stream());

  return {{out}, {axes[0]}};
}

std::vector<mlx::core::array> AttentionDecodePrimitive::jvp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& tangents,
    const std::vector<int>& argnums) {

  // For Phase 1, fall back to MLX's autodiff
  // In Phase 2, implement custom JVP for efficiency
  auto out = attention_decode_fused(
      primals[0], primals[1], primals[2], primals[3], primals[4],
      num_heads_, num_kv_heads_, head_dim_, block_size_,
      max_blocks_per_seq_, num_layers_, layer_idx_, use_block_format_,
      use_sliding_window_, sliding_window_size_,
      stream());

  return {out};  // Placeholder
}

std::vector<mlx::core::array> AttentionDecodePrimitive::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {

  // For Phase 1, fall back to MLX's autodiff
  // In Phase 2, implement custom VJP for efficiency
  return {cotangents[0]};  // Placeholder
}

bool AttentionDecodePrimitive::is_equivalent(const mlx::core::Primitive& other) const {
  const auto* other_attn = dynamic_cast<const AttentionDecodePrimitive*>(&other);
  if (!other_attn) {
    return false;
  }
  return num_heads_ == other_attn->num_heads_ &&
         num_kv_heads_ == other_attn->num_kv_heads_ &&
         head_dim_ == other_attn->head_dim_ &&
         block_size_ == other_attn->block_size_ &&
         max_blocks_per_seq_ == other_attn->max_blocks_per_seq_ &&
         use_sliding_window_ == other_attn->use_sliding_window_ &&
         sliding_window_size_ == other_attn->sliding_window_size_;
}

// ============================================================================
// Public API
// ============================================================================

mlx::core::array attention_decode_fused(
    const mlx::core::array& q,
    const mlx::core::array& k_cache,
    const mlx::core::array& v_cache,
    const mlx::core::array& page_table,
    const mlx::core::array& seq_lengths,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,
    bool use_block_format,
    bool use_sliding_window,
    int sliding_window_size,
    mlx::core::StreamOrDevice s) {

  // Validate inputs
  if (q.ndim() != 3) {
    throw std::invalid_argument("q must be 3-dimensional [batch, num_heads, head_dim]");
  }

  // Validate cache dimensions based on format
  if (use_block_format) {
    // Block format: [num_pages, num_layers, block_size, num_kv_heads, head_dim]
    if (k_cache.ndim() != 5) {
      throw std::invalid_argument(
          "k_cache must be 5-dimensional [num_pages, num_layers, block_size, num_kv_heads, head_dim] when use_block_format=true");
    }
    if (v_cache.ndim() != 5) {
      throw std::invalid_argument(
          "v_cache must be 5-dimensional [num_pages, num_layers, block_size, num_kv_heads, head_dim] when use_block_format=true");
    }
  } else {
    // Stacked format: [num_pages, block_size, num_kv_heads, head_dim]
    if (k_cache.ndim() != 4) {
      throw std::invalid_argument(
          "k_cache must be 4-dimensional [num_pages, block_size, num_kv_heads, head_dim] when use_block_format=false");
    }
    if (v_cache.ndim() != 4) {
      throw std::invalid_argument(
          "v_cache must be 4-dimensional [num_pages, block_size, num_kv_heads, head_dim] when use_block_format=false");
    }
  }

  if (page_table.ndim() != 2) {
    throw std::invalid_argument(
        "page_table must be 2-dimensional [batch, max_blocks_per_seq]");
  }

  if (seq_lengths.ndim() != 1) {
    throw std::invalid_argument("seq_lengths must be 1-dimensional [batch]");
  }

  // Check shapes match
  int batch_size = q.shape(0);
  if (q.shape(1) != num_heads || q.shape(2) != head_dim) {
    throw std::invalid_argument("q shape mismatch with num_heads/head_dim");
  }

  if (page_table.shape(0) != batch_size) {
    throw std::invalid_argument("page_table batch size mismatch");
  }

  if (seq_lengths.shape(0) != batch_size) {
    throw std::invalid_argument("seq_lengths batch size mismatch");
  }

  // Get stream
  auto stream = mlx::core::to_stream(s);

  // Ensure inputs are contiguous (Phase 1 requirement)
  auto q_contig = q;
  auto k_cache_contig = k_cache;
  auto v_cache_contig = v_cache;
  auto page_table_contig = page_table;
  auto seq_lengths_contig = seq_lengths;

  if (!q.flags().row_contiguous) {
    q_contig = mlx::core::reshape(mlx::core::reshape(q, {-1}, stream), q.shape(), stream);
    mlx::core::eval(q_contig);
  }

  if (!k_cache.flags().row_contiguous) {
    k_cache_contig = mlx::core::reshape(
        mlx::core::reshape(k_cache, {-1}, stream), k_cache.shape(), stream);
    mlx::core::eval(k_cache_contig);
  }

  if (!v_cache.flags().row_contiguous) {
    v_cache_contig = mlx::core::reshape(
        mlx::core::reshape(v_cache, {-1}, stream), v_cache.shape(), stream);
    mlx::core::eval(v_cache_contig);
  }

  if (!page_table.flags().row_contiguous) {
    page_table_contig = mlx::core::reshape(
        mlx::core::reshape(page_table, {-1}, stream), page_table.shape(), stream);
    mlx::core::eval(page_table_contig);
  }

  if (!seq_lengths.flags().row_contiguous) {
    seq_lengths_contig = mlx::core::reshape(
        mlx::core::reshape(seq_lengths, {-1}, stream), seq_lengths.shape(), stream);
    mlx::core::eval(seq_lengths_contig);
  }

  // Create primitive
  auto primitive = std::make_shared<AttentionDecodePrimitive>(
      stream, num_heads, num_kv_heads, head_dim, block_size,
      max_blocks_per_seq, num_layers, layer_idx, use_block_format,
      use_sliding_window, sliding_window_size);

  // Create output array using MLX's array factory with primitive
  auto outputs = mlx::core::array::make_arrays(
      {q_contig.shape()},                   // output shape
      {q_contig.dtype()},                   // output dtype
      primitive,                            // the primitive
      {q_contig, k_cache_contig, v_cache_contig,
       page_table_contig, seq_lengths_contig}  // inputs (contiguous)
  );

  return outputs[0];
}

}  // namespace kernels
}  // namespace mlxr
