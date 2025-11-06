// Copyright © 2025 MLXR Development
// MLX Primitive-based custom attention prefill kernel implementation
//
// This implementation uses MLX's Primitive API with direct Metal buffer access
// via MLX's compute encoder. Metal-cpp headers are bundled with MLX and provide
// the Metal C++ API for custom kernel dispatch.

#include "attention_prefill_primitive.h"

#include <mlx/ops.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/transforms.h>  // For eval

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mlxr {
namespace kernels {

// ============================================================================
// Constructor & Destructor
// ============================================================================

AttentionPrefillPrimitive::AttentionPrefillPrimitive(
    mlx::core::Stream stream,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,
    bool use_block_format,
    int position_offset)
    : mlx::core::Primitive(stream),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      hidden_size_(hidden_size),
      block_size_(block_size),
      max_blocks_per_seq_(max_blocks_per_seq),
      num_layers_(num_layers),
      layer_idx_(layer_idx),
      use_block_format_(use_block_format),
      position_offset_(position_offset),
      library_(nullptr) {
}

AttentionPrefillPrimitive::~AttentionPrefillPrimitive() {
  // Metal library is managed by MLX's device, no explicit cleanup needed
}

// ============================================================================
// Metal Kernel Loading
// ============================================================================

void* AttentionPrefillPrimitive::load_metal_library() {
  if (library_) {
    return library_;
  }

  @autoreleasepool {
    // Get Metal device
    auto& d = mlx::core::metal::device(stream().device);

    // Find metallib file - search multiple paths
    NSArray<NSString*>* search_paths = @[
      @"build/lib/attention_prefill.metallib",
      @"../../lib/attention_prefill.metallib",
      @"../lib/attention_prefill.metallib",
      @"lib/attention_prefill.metallib",
      [@(getenv("PWD") ?: ".") stringByAppendingString:@"/build/lib/attention_prefill.metallib"]
    ];

    NSString* metallib_path = nil;
    for (NSString* path in search_paths) {
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        metallib_path = path;
        NSLog(@"Found attention_prefill.metallib at: %@", path);
        break;
      }
    }

    if (!metallib_path) {
      NSLog(@"Failed to find attention_prefill.metallib in any of these paths:");
      for (NSString* path in search_paths) {
        NSLog(@"  - %@", path);
      }
      throw std::runtime_error(
          "Failed to find attention_prefill.metallib. Please run 'make metal'");
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

void AttentionPrefillPrimitive::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size() == 9);
  assert(outputs.size() == 1);

  const auto& input = inputs[0];        // [batch, seq_len, hidden_size]
  const auto& q = inputs[1];            // [batch, seq_len, num_heads, head_dim]
  const auto& k = inputs[2];            // [batch, seq_len, num_kv_heads, head_dim]
  const auto& v = inputs[3];            // [batch, seq_len, num_kv_heads, head_dim]
  const auto& rope_cos = inputs[4];     // [max_seq_len, head_dim/2]
  const auto& rope_sin = inputs[5];     // [max_seq_len, head_dim/2]
  // k_cache and v_cache are modified in-place
  auto& k_cache = const_cast<mlx::core::array&>(inputs[6]);  // [num_pages, block_size, num_kv_heads, head_dim]
  auto& v_cache = const_cast<mlx::core::array&>(inputs[7]);  // [num_pages, block_size, num_kv_heads, head_dim]
  const auto& page_table = inputs[8];   // [batch, max_blocks_per_seq]
  auto& output = outputs[0];            // [batch, seq_len, num_heads, head_dim]

  // Allocate output buffer
  auto buffer = mlx::core::allocator::malloc(output.nbytes());
  output.set_data(buffer);

  // CPU reference implementation for testing
  int batch_size = q.shape(0);
  int seq_len = q.shape(1);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  // Process each batch, sequence position, and head
  for (int b = 0; b < batch_size; b++) {
    for (int t = 0; t < seq_len; t++) {
      for (int h = 0; h < num_heads_; h++) {
        int kv_h = h / (num_heads_ / num_kv_heads_);  // GQA mapping

        // Apply RoPE to Q (simplified - real implementation uses cos/sin tables)
        std::vector<float> q_rope(head_dim_);
        for (int d = 0; d < head_dim_; d++) {
          int q_idx = b * seq_len * num_heads_ * head_dim_ +
                     t * num_heads_ * head_dim_ +
                     h * head_dim_ + d;
          q_rope[d] = q.data<float>()[q_idx];
        }

        // Compute attention scores for tokens [0..t] (causal)
        int num_context = t + 1;
        std::vector<float> scores(num_context);
        float max_score = -std::numeric_limits<float>::infinity();

        for (int ctx = 0; ctx < num_context; ctx++) {
          // Apply RoPE to K and store in cache
          int k_idx_base = b * seq_len * num_kv_heads_ * head_dim_ +
                          ctx * num_kv_heads_ * head_dim_ +
                          kv_h * head_dim_;

          // Compute Q @ K^T
          float score = 0.0f;
          for (int d = 0; d < head_dim_; d++) {
            float k_val = k.data<float>()[k_idx_base + d];
            score += q_rope[d] * k_val;

            // Store K in cache (simplified - real impl uses page table)
            int block_idx = ctx / block_size_;
            int block_offset = ctx % block_size_;
            int page_id = page_table.data<int>()[b * max_blocks_per_seq_ + block_idx];

            if (page_id >= 0) {
              int k_cache_idx = page_id * block_size_ * num_kv_heads_ * head_dim_ +
                               block_offset * num_kv_heads_ * head_dim_ +
                               kv_h * head_dim_ + d;
              k_cache.data<float>()[k_cache_idx] = k_val;
            }
          }

          score *= scale;
          scores[ctx] = score;
          max_score = std::max(max_score, score);
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int ctx = 0; ctx < num_context; ctx++) {
          scores[ctx] = std::exp(scores[ctx] - max_score);
          sum_exp += scores[ctx];
        }

        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int ctx = 0; ctx < num_context; ctx++) {
          scores[ctx] *= inv_sum;
        }

        // Compute context = softmax(scores) @ V
        for (int d = 0; d < head_dim_; d++) {
          float accum = 0.0f;

          for (int ctx = 0; ctx < num_context; ctx++) {
            int v_idx = b * seq_len * num_kv_heads_ * head_dim_ +
                       ctx * num_kv_heads_ * head_dim_ +
                       kv_h * head_dim_ + d;
            float v_val = v.data<float>()[v_idx];

            // Store V in cache
            int block_idx = ctx / block_size_;
            int block_offset = ctx % block_size_;
            int page_id = page_table.data<int>()[b * max_blocks_per_seq_ + block_idx];

            if (page_id >= 0) {
              int v_cache_idx = page_id * block_size_ * num_kv_heads_ * head_dim_ +
                               block_offset * num_kv_heads_ * head_dim_ +
                               kv_h * head_dim_ + d;
              v_cache.data<float>()[v_cache_idx] = v_val;
            }

            accum += scores[ctx] * v_val;
          }

          int out_idx = b * seq_len * num_heads_ * head_dim_ +
                       t * num_heads_ * head_dim_ +
                       h * head_dim_ + d;
          output.data<float>()[out_idx] = accum;
        }
      }
    }
  }
}

// ============================================================================
// GPU Evaluation (Custom Metal Kernel)
// ============================================================================

void AttentionPrefillPrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  NSLog(@"[AttentionPrefill] eval_gpu() called - using Metal kernel");

  assert(inputs.size() == 9);
  assert(outputs.size() == 1);

  const auto& input = inputs[0];        // [batch, seq_len, hidden_size]
  const auto& q = inputs[1];            // [batch, seq_len, num_heads, head_dim]
  const auto& k = inputs[2];            // [batch, seq_len, num_kv_heads, head_dim]
  const auto& v = inputs[3];            // [batch, seq_len, num_kv_heads, head_dim]
  const auto& rope_cos = inputs[4];     // [max_seq_len, head_dim/2]
  const auto& rope_sin = inputs[5];     // [max_seq_len, head_dim/2]
  // k_cache and v_cache are modified in-place by the Metal kernel
  auto& k_cache = const_cast<mlx::core::array&>(inputs[6]);  // [num_pages, block_size, num_kv_heads, head_dim]
  auto& v_cache = const_cast<mlx::core::array&>(inputs[7]);  // [num_pages, block_size, num_kv_heads, head_dim]
  const auto& page_table = inputs[8];   // [batch, max_blocks_per_seq]
  auto& output = outputs[0];            // [batch, seq_len, num_heads, head_dim]

  // Check contiguity (Phase 1 limitation)
  if (!input.flags().row_contiguous ||
      !q.flags().row_contiguous ||
      !k.flags().row_contiguous ||
      !v.flags().row_contiguous ||
      !rope_cos.flags().row_contiguous ||
      !rope_sin.flags().row_contiguous ||
      !k_cache.flags().row_contiguous ||
      !v_cache.flags().row_contiguous ||
      !page_table.flags().row_contiguous) {
    throw std::runtime_error(
        "AttentionPrefillPrimitive requires contiguous inputs. "
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
  std::string kernel_name = "attention_prefill_fused";

  // Get compiled kernel from MLX's device (this caches it)
  auto* kernel = d.get_kernel(kernel_name, mtl_lib);

  // Get MLX's command encoder and set pipeline
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Prepare kernel arguments
  int batch_size = q.shape(0);
  int seq_len = q.shape(1);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  // Bind buffers
  compute_encoder.set_input_array(input, 0);
  compute_encoder.set_output_array(output, 1);
  compute_encoder.set_input_array(q, 2);
  compute_encoder.set_input_array(k, 3);
  compute_encoder.set_input_array(v, 4);
  compute_encoder.set_input_array(rope_cos, 5);
  compute_encoder.set_input_array(rope_sin, 6);
  compute_encoder.set_input_array(k_cache, 7);
  compute_encoder.set_input_array(v_cache, 8);
  compute_encoder.set_input_array(page_table, 9);

  // Set scalar parameters
  compute_encoder.set_bytes(static_cast<uint32_t>(batch_size), 10);
  compute_encoder.set_bytes(static_cast<uint32_t>(seq_len), 11);
  compute_encoder.set_bytes(static_cast<uint32_t>(num_heads_), 12);
  compute_encoder.set_bytes(static_cast<uint32_t>(num_kv_heads_), 13);
  compute_encoder.set_bytes(static_cast<uint32_t>(head_dim_), 14);
  compute_encoder.set_bytes(static_cast<uint32_t>(hidden_size_), 15);
  compute_encoder.set_bytes(static_cast<uint32_t>(block_size_), 16);
  compute_encoder.set_bytes(static_cast<uint32_t>(max_blocks_per_seq_), 17);
  compute_encoder.set_bytes(static_cast<uint32_t>(num_layers_), 18);      // NEW
  compute_encoder.set_bytes(static_cast<uint32_t>(layer_idx_), 19);       // NEW
  compute_encoder.set_bytes(use_block_format_, 20);                        // NEW
  compute_encoder.set_bytes(scale, 21);
  compute_encoder.set_bytes(static_cast<uint32_t>(position_offset_), 22);

  // Dispatch configuration
  // Each threadgroup handles one query head for one token
  size_t num_threadgroups = batch_size * seq_len * num_heads_;
  size_t threads_per_group = 256;

  MTL::Size grid_dims(num_threadgroups, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);

  NSLog(@"[AttentionPrefill] Dispatch params: batch=%d, seq_len=%d, heads=%d, kv_heads=%d, head_dim=%d",
        batch_size, seq_len, num_heads_, num_kv_heads_, head_dim_);
  NSLog(@"[AttentionPrefill] Grid dims: (%zu, %zu, %zu), Group dims: (%zu, %zu, %zu)",
        grid_dims.width, grid_dims.height, grid_dims.depth,
        group_dims.width, group_dims.height, group_dims.depth);

  // Allocate threadgroup memory
  // shared_scores: 64 * sizeof(float) for stripe processing
  // shared_reduce: threads_per_group * sizeof(float) for reductions
  // shared_q: head_dim * sizeof(half) for Q storage
  // shared_k: head_dim * sizeof(half) for K storage
  size_t stripe_size = 64;
  size_t shared_scores_size = stripe_size * sizeof(float);
  size_t shared_reduce_size = threads_per_group * sizeof(float);
  size_t shared_q_size = head_dim_ * sizeof(uint16_t);  // half = 16-bit
  size_t shared_k_size = head_dim_ * sizeof(uint16_t);

  compute_encoder.set_threadgroup_memory_length(shared_scores_size, 0);
  compute_encoder.set_threadgroup_memory_length(shared_reduce_size, 1);
  compute_encoder.set_threadgroup_memory_length(shared_q_size, 2);
  compute_encoder.set_threadgroup_memory_length(shared_k_size, 3);

  NSLog(@"[AttentionPrefill] Threadgroup memory: scores=%zu, reduce=%zu, q=%zu, k=%zu bytes",
        shared_scores_size, shared_reduce_size, shared_q_size, shared_k_size);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  NSLog(@"[AttentionPrefill] Dispatch complete");
}

// ============================================================================
// Function Transformations
// ============================================================================

std::pair<std::vector<mlx::core::array>, std::vector<int>>
AttentionPrefillPrimitive::vmap(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<int>& axes) {

  // Attention prefill can be vmapped over batch dimension
  auto out = attention_prefill_fused(
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
      const_cast<mlx::core::array&>(inputs[6]),
      const_cast<mlx::core::array&>(inputs[7]),
      inputs[8],
      num_heads_, num_kv_heads_, head_dim_, hidden_size_,
      block_size_, max_blocks_per_seq_, num_layers_, layer_idx_,
      use_block_format_, position_offset_,
      stream());

  return {{out}, {axes[0]}};
}

std::vector<mlx::core::array> AttentionPrefillPrimitive::jvp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& tangents,
    const std::vector<int>& argnums) {

  // For Phase 1, fall back to MLX's autodiff
  auto out = attention_prefill_fused(
      primals[0], primals[1], primals[2], primals[3], primals[4], primals[5],
      const_cast<mlx::core::array&>(primals[6]),
      const_cast<mlx::core::array&>(primals[7]),
      primals[8],
      num_heads_, num_kv_heads_, head_dim_, hidden_size_,
      block_size_, max_blocks_per_seq_, num_layers_, layer_idx_,
      use_block_format_, position_offset_,
      stream());

  return {out};  // Placeholder
}

std::vector<mlx::core::array> AttentionPrefillPrimitive::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {

  // For Phase 1, fall back to MLX's autodiff
  return {cotangents[0]};  // Placeholder
}

bool AttentionPrefillPrimitive::is_equivalent(const mlx::core::Primitive& other) const {
  const auto* other_attn = dynamic_cast<const AttentionPrefillPrimitive*>(&other);
  if (!other_attn) {
    return false;
  }
  return num_heads_ == other_attn->num_heads_ &&
         num_kv_heads_ == other_attn->num_kv_heads_ &&
         head_dim_ == other_attn->head_dim_ &&
         hidden_size_ == other_attn->hidden_size_ &&
         block_size_ == other_attn->block_size_ &&
         max_blocks_per_seq_ == other_attn->max_blocks_per_seq_ &&
         position_offset_ == other_attn->position_offset_;
}

// ============================================================================
// Public API
// ============================================================================

mlx::core::array attention_prefill_fused(
    const mlx::core::array& input,
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& rope_cos,
    const mlx::core::array& rope_sin,
    mlx::core::array& k_cache,
    mlx::core::array& v_cache,
    const mlx::core::array& page_table,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int hidden_size,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,
    bool use_block_format,
    int position_offset,
    mlx::core::StreamOrDevice s) {

  // Validate inputs
  if (input.ndim() != 3) {
    throw std::invalid_argument("input must be 3-dimensional [batch, seq_len, hidden_size]");
  }

  if (q.ndim() != 4) {
    throw std::invalid_argument("q must be 4-dimensional [batch, seq_len, num_heads, head_dim]");
  }

  if (k.ndim() != 4) {
    throw std::invalid_argument("k must be 4-dimensional [batch, seq_len, num_kv_heads, head_dim]");
  }

  if (v.ndim() != 4) {
    throw std::invalid_argument("v must be 4-dimensional [batch, seq_len, num_kv_heads, head_dim]");
  }

  if (rope_cos.ndim() != 2) {
    throw std::invalid_argument("rope_cos must be 2-dimensional [max_seq_len, head_dim/2]");
  }

  if (rope_sin.ndim() != 2) {
    throw std::invalid_argument("rope_sin must be 2-dimensional [max_seq_len, head_dim/2]");
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
    throw std::invalid_argument("page_table must be 2-dimensional [batch, max_blocks_per_seq]");
  }

  // Check shapes match
  int batch_size = q.shape(0);
  int seq_len = q.shape(1);

  if (q.shape(2) != num_heads || q.shape(3) != head_dim) {
    throw std::invalid_argument("q shape mismatch with num_heads/head_dim");
  }

  if (k.shape(0) != batch_size || k.shape(1) != seq_len) {
    throw std::invalid_argument("k batch/seq_len mismatch");
  }

  if (v.shape(0) != batch_size || v.shape(1) != seq_len) {
    throw std::invalid_argument("v batch/seq_len mismatch");
  }

  if (page_table.shape(0) != batch_size) {
    throw std::invalid_argument("page_table batch size mismatch");
  }

  // Get stream
  auto stream = mlx::core::to_stream(s);

  // Ensure inputs are contiguous (Phase 1 requirement)
  auto input_contig = input;
  auto q_contig = q;
  auto k_contig = k;
  auto v_contig = v;
  auto rope_cos_contig = rope_cos;
  auto rope_sin_contig = rope_sin;
  auto k_cache_contig = k_cache;
  auto v_cache_contig = v_cache;
  auto page_table_contig = page_table;

  if (!input.flags().row_contiguous) {
    input_contig = mlx::core::reshape(
        mlx::core::reshape(input, {-1}, stream), input.shape(), stream);
    mlx::core::eval(input_contig);
  }

  if (!q.flags().row_contiguous) {
    q_contig = mlx::core::reshape(
        mlx::core::reshape(q, {-1}, stream), q.shape(), stream);
    mlx::core::eval(q_contig);
  }

  if (!k.flags().row_contiguous) {
    k_contig = mlx::core::reshape(
        mlx::core::reshape(k, {-1}, stream), k.shape(), stream);
    mlx::core::eval(k_contig);
  }

  if (!v.flags().row_contiguous) {
    v_contig = mlx::core::reshape(
        mlx::core::reshape(v, {-1}, stream), v.shape(), stream);
    mlx::core::eval(v_contig);
  }

  if (!rope_cos.flags().row_contiguous) {
    rope_cos_contig = mlx::core::reshape(
        mlx::core::reshape(rope_cos, {-1}, stream), rope_cos.shape(), stream);
    mlx::core::eval(rope_cos_contig);
  }

  if (!rope_sin.flags().row_contiguous) {
    rope_sin_contig = mlx::core::reshape(
        mlx::core::reshape(rope_sin, {-1}, stream), rope_sin.shape(), stream);
    mlx::core::eval(rope_sin_contig);
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

  // Create primitive
  auto primitive = std::make_shared<AttentionPrefillPrimitive>(
      stream, num_heads, num_kv_heads, head_dim, hidden_size,
      block_size, max_blocks_per_seq, num_layers, layer_idx,
      use_block_format, position_offset);

  // Create output array using MLX's array factory with primitive
  auto outputs = mlx::core::array::make_arrays(
      {q_contig.shape()},                   // output shape (same as Q)
      {q_contig.dtype()},                   // output dtype
      primitive,                            // the primitive
      {input_contig, q_contig, k_contig, v_contig,
       rope_cos_contig, rope_sin_contig,
       k_cache_contig, v_cache_contig, page_table_contig}  // inputs (contiguous)
  );

  return outputs[0];
}

}  // namespace kernels
}  // namespace mlxr
