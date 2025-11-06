/**
 * @file arena.cpp
 * @brief Implementation of paged KV cache arena
 */

#include "arena.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace mlxr {
namespace runtime {
namespace kv {

// ============================================================================
// Block Implementation
// ============================================================================

Block::Block(int id, int location, const ArenaConfig& config)
    : block_id(id),
      ref_count(0),
      location(location),
      dirty(false),
      last_access_time(0) {
  // Allocate K and V tensors
  // Shape: [num_layers, block_size_tokens, num_kv_heads, head_dim]
  std::vector<int> shape = {config.num_layers, config.block_size_tokens,
                            config.num_kv_heads, config.head_dim};

  // Allocate on GPU or CPU based on location
  if (location == 0) {
    // GPU allocation via MLX (default device)
    k_data = graph::zeros(shape, config.dtype);
    v_data = graph::zeros(shape, config.dtype);
  } else {
    // CPU allocation
    // MLX arrays can be created and then moved to CPU if needed
    k_data = graph::zeros(shape, config.dtype);
    v_data = graph::zeros(shape, config.dtype);
    // Note: In MLX, arrays are lazy - actual allocation happens on first use
    // For CPU, we'd use mlx::core::eval to materialize and then copy to CPU
    // For now, keep on default device and rely on MLX's unified memory
  }
}

// ============================================================================
// Arena Implementation
// ============================================================================

Arena::Arena(const ArenaConfig& config)
    : config_(config),
      next_block_id_(0),
      num_gpu_to_cpu_moves_(0),
      num_cpu_to_gpu_moves_(0),
      timestamp_counter_(0) {
  initialize();
}

Arena::~Arena() { clear(); }

void Arena::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Calculate capacity but use LAZY allocation
  // Blocks will be allocated on-demand in allocate_block()
  int num_gpu_blocks = config_.num_blocks;
  if (config_.allow_cpu_overflow) {
    // Reserve some capacity for CPU overflow
    num_gpu_blocks = std::max(64, config_.num_blocks - config_.max_cpu_blocks);
  }

  // Reserve capacity but don't pre-allocate (lazy allocation)
  free_gpu_blocks_.reserve(num_gpu_blocks);
  blocks_.reserve(num_gpu_blocks);

  std::cout << "KV Arena initialized (lazy allocation): capacity="
            << num_gpu_blocks << " GPU blocks, "
            << "block_size=" << config_.block_size_tokens << " tokens"
            << std::endl;
}

int Arena::allocate_physical_block(int location) {
  // Create new block (no lock needed - called from locked context)
  int block_id = next_block_id_++;
  auto block = std::make_unique<Block>(block_id, location, config_);

  int block_index = static_cast<int>(blocks_.size());
  block_id_to_index_[block_id] = block_index;
  blocks_.push_back(std::move(block));

  return block_id;
}

int Arena::allocate_block() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Try to allocate from GPU free list first
  if (!free_gpu_blocks_.empty()) {
    int block_id = free_gpu_blocks_.back();
    free_gpu_blocks_.pop_back();

    Block* block = get_block(block_id);
    if (block) {
      block->ref_count = 1;
      block->dirty = false;
      block->last_access_time = get_timestamp();
    }

    return block_id;
  }

  // LAZY ALLOCATION: If free list is empty but we're below capacity, allocate a
  // new block
  int num_gpu_capacity = config_.num_blocks;
  if (config_.allow_cpu_overflow) {
    num_gpu_capacity =
        std::max(64, config_.num_blocks - config_.max_cpu_blocks);
  }

  int num_gpu_blocks = 0;
  for (const auto& block : blocks_) {
    if (block->location == 0) {  // GPU
      num_gpu_blocks++;
    }
  }

  if (num_gpu_blocks < num_gpu_capacity) {
    // Allocate new GPU block lazily
    int block_id = allocate_physical_block(0);  // GPU
    Block* block = get_block(block_id);
    if (block) {
      block->ref_count = 1;
      block->dirty = false;
      block->last_access_time = get_timestamp();
    }
    return block_id;
  }

  // Try CPU overflow if enabled
  if (config_.allow_cpu_overflow) {
    if (!free_cpu_blocks_.empty()) {
      int block_id = free_cpu_blocks_.back();
      free_cpu_blocks_.pop_back();

      Block* block = get_block(block_id);
      if (block) {
        block->ref_count = 1;
        block->dirty = false;
        block->last_access_time = get_timestamp();
      }

      return block_id;
    }

    // Allocate new CPU block if under limit
    if (static_cast<int>(free_cpu_blocks_.size()) + num_allocated_blocks() <
        config_.num_blocks + config_.max_cpu_blocks) {
      int block_id = allocate_physical_block(1);  // CPU

      Block* block = get_block(block_id);
      if (block) {
        block->ref_count = 1;
        block->dirty = false;
        block->last_access_time = get_timestamp();
      }

      return block_id;
    }
  }

  // No blocks available
  return -1;
}

std::vector<int> Arena::allocate_blocks(int num_blocks) {
  std::vector<int> allocated;
  allocated.reserve(num_blocks);

  for (int i = 0; i < num_blocks; ++i) {
    int block_id = allocate_block();
    if (block_id < 0) {
      // Allocation failed - free already allocated blocks
      free_blocks(allocated);
      return {};
    }
    allocated.push_back(block_id);
  }

  return allocated;
}

void Arena::free_block(int block_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  Block* block = get_block(block_id);
  if (!block) {
    std::cerr << "Warning: Attempting to free invalid block " << block_id
              << std::endl;
    return;
  }

  // Decrement reference count
  block->ref_count--;

  // Only add to free list if ref count reaches 0
  if (block->ref_count <= 0) {
    block->ref_count = 0;
    block->dirty = false;

    if (block->location == 0) {
      free_gpu_blocks_.push_back(block_id);
    } else {
      free_cpu_blocks_.push_back(block_id);
    }
  }
}

void Arena::free_blocks(const std::vector<int>& block_ids) {
  for (int block_id : block_ids) {
    free_block(block_id);
  }
}

Block* Arena::get_block(int block_id) {
  // No lock - assumes caller has lock or is read-only
  auto it = block_id_to_index_.find(block_id);
  if (it == block_id_to_index_.end()) {
    return nullptr;
  }

  int index = it->second;
  if (index < 0 || index >= static_cast<int>(blocks_.size())) {
    return nullptr;
  }

  return blocks_[index].get();
}

const Block* Arena::get_block(int block_id) const {
  auto it = block_id_to_index_.find(block_id);
  if (it == block_id_to_index_.end()) {
    return nullptr;
  }

  int index = it->second;
  if (index < 0 || index >= static_cast<int>(blocks_.size())) {
    return nullptr;
  }

  return blocks_[index].get();
}

void Arena::ref_block(int block_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  Block* block = get_block(block_id);
  if (block) {
    block->ref_count++;
    block->last_access_time = get_timestamp();
  }
}

void Arena::unref_block(int block_id) {
  // Just calls free_block which handles ref counting
  free_block(block_id);
}

void Arena::touch_block(int block_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  Block* block = get_block(block_id);
  if (block) {
    block->last_access_time = get_timestamp();
  }
}

bool Arena::move_to_cpu(int block_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  Block* block = get_block(block_id);
  if (!block || block->location != 0) {
    return false;  // Not on GPU or invalid
  }

  // In MLX with unified memory, this is mostly a logical move
  // The actual data might stay in unified memory accessible to both
  // For explicit CPU copy:
  // 1. Evaluate tensors to materialize
  mlx::core::eval(block->k_data.array());
  mlx::core::eval(block->v_data.array());

  // 2. Mark as CPU location
  block->location = 1;

  num_gpu_to_cpu_moves_++;

  return true;
}

bool Arena::move_to_gpu(int block_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  Block* block = get_block(block_id);
  if (!block || block->location != 1) {
    return false;  // Not on CPU or invalid
  }

  // In MLX with unified memory, data is already accessible
  // Just mark as GPU location
  block->location = 0;

  num_cpu_to_gpu_moves_++;

  return true;
}

int Arena::num_free_gpu_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(free_gpu_blocks_.size());
}

int Arena::num_free_cpu_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(free_cpu_blocks_.size());
}

int Arena::num_allocated_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);

  int allocated = 0;
  for (const auto& block : blocks_) {
    if (block->ref_count > 0) {
      allocated++;
    }
  }

  return allocated;
}

size_t Arena::memory_usage() const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Size per block: 2 tensors (K, V) * layers * block_size * heads * head_dim *
  // dtype_size
  size_t dtype_size = 2;  // float16
  if (config_.dtype == mlx::core::float32) {
    dtype_size = 4;
  }

  size_t block_size = 2 * config_.num_layers * config_.block_size_tokens *
                      config_.num_kv_heads * config_.head_dim * dtype_size;

  return blocks_.size() * block_size;
}

size_t Arena::gpu_memory_usage() const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t dtype_size = (config_.dtype == mlx::core::float32) ? 4 : 2;
  size_t block_size = 2 * config_.num_layers * config_.block_size_tokens *
                      config_.num_kv_heads * config_.head_dim * dtype_size;

  size_t gpu_mem = 0;
  for (const auto& block : blocks_) {
    if (block->location == 0) {
      gpu_mem += block_size;
    }
  }

  return gpu_mem;
}

size_t Arena::cpu_memory_usage() const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t dtype_size = (config_.dtype == mlx::core::float32) ? 4 : 2;
  size_t block_size = 2 * config_.num_layers * config_.block_size_tokens *
                      config_.num_kv_heads * config_.head_dim * dtype_size;

  size_t cpu_mem = 0;
  for (const auto& block : blocks_) {
    if (block->location == 1) {
      cpu_mem += block_size;
    }
  }

  return cpu_mem;
}

void Arena::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  blocks_.clear();
  free_gpu_blocks_.clear();
  free_cpu_blocks_.clear();
  block_id_to_index_.clear();
  next_block_id_ = 0;
  num_gpu_to_cpu_moves_ = 0;
  num_cpu_to_gpu_moves_ = 0;
  timestamp_counter_ = 0;
}

Arena::Stats Arena::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);

  Stats stats;
  stats.total_blocks = static_cast<int>(blocks_.size());
  stats.free_gpu_blocks = static_cast<int>(free_gpu_blocks_.size());
  stats.free_cpu_blocks = static_cast<int>(free_cpu_blocks_.size());
  stats.allocated_blocks = num_allocated_blocks();
  stats.total_memory_bytes = memory_usage();
  stats.gpu_memory_bytes = gpu_memory_usage();
  stats.cpu_memory_bytes = cpu_memory_usage();
  stats.num_gpu_to_cpu_moves = num_gpu_to_cpu_moves_;
  stats.num_cpu_to_gpu_moves = num_cpu_to_gpu_moves_;

  return stats;
}

uint64_t Arena::get_timestamp() const {
  // Simple monotonic counter for LRU
  return ++timestamp_counter_;
}

// ============================================================================
// Metal Primitive Bridge Methods
// ============================================================================

graph::Tensor Arena::build_k_cache_array(int layer_idx,
                                         const std::vector<int>& block_ids) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (block_ids.empty()) {
    // Return empty tensor if no blocks
    return graph::zeros(
        {0, config_.block_size_tokens, config_.num_kv_heads, config_.head_dim},
        config_.dtype);
  }

  // Extract layer slice from each block and stack
  // Each block's k_data shape: [num_layers, block_size, num_kv_heads, head_dim]
  // We want to extract layer_idx and stack: [num_pages, block_size,
  // num_kv_heads, head_dim]

  std::vector<graph::Tensor> layer_slices;
  layer_slices.reserve(block_ids.size());

  for (int block_id : block_ids) {
    Block* block = get_block(block_id);
    if (!block) {
      throw std::runtime_error("Invalid block ID in build_k_cache_array: " +
                               std::to_string(block_id));
    }

    // Extract layer slice: k_data[layer_idx, :, :, :]
    // Shape: [block_size, num_kv_heads, head_dim]
    auto k_arr = block->k_data.array();
    auto layer_slice =
        mlx::core::slice(k_arr, {layer_idx, 0, 0, 0},
                         {layer_idx + 1, config_.block_size_tokens,
                          config_.num_kv_heads, config_.head_dim},
                         {1, 1, 1, 1});

    // Remove layer dimension: [1, block_size, num_kv_heads, head_dim] ->
    // [block_size, num_kv_heads, head_dim]
    layer_slice = mlx::core::squeeze(layer_slice, 0);

    layer_slices.push_back(graph::Tensor(layer_slice));
  }

  // Stack along new dimension 0 to get [num_pages, block_size, num_kv_heads,
  // head_dim] Convert Tensors to arrays for MLX stack
  std::vector<mlx::core::array> arrays;
  arrays.reserve(layer_slices.size());
  for (const auto& t : layer_slices) {
    arrays.push_back(t.array());
  }

  auto stacked_arr = mlx::core::stack(arrays, 0);
  return graph::Tensor(stacked_arr);
}

graph::Tensor Arena::build_v_cache_array(int layer_idx,
                                         const std::vector<int>& block_ids) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (block_ids.empty()) {
    // Return empty tensor if no blocks
    return graph::zeros(
        {0, config_.block_size_tokens, config_.num_kv_heads, config_.head_dim},
        config_.dtype);
  }

  // Extract layer slice from each block and stack
  std::vector<graph::Tensor> layer_slices;
  layer_slices.reserve(block_ids.size());

  for (int block_id : block_ids) {
    Block* block = get_block(block_id);
    if (!block) {
      throw std::runtime_error("Invalid block ID in build_v_cache_array: " +
                               std::to_string(block_id));
    }

    // Extract layer slice: v_data[layer_idx, :, :, :]
    auto v_arr = block->v_data.array();
    auto layer_slice =
        mlx::core::slice(v_arr, {layer_idx, 0, 0, 0},
                         {layer_idx + 1, config_.block_size_tokens,
                          config_.num_kv_heads, config_.head_dim},
                         {1, 1, 1, 1});

    // Remove layer dimension
    layer_slice = mlx::core::squeeze(layer_slice, 0);

    layer_slices.push_back(graph::Tensor(layer_slice));
  }

  // Stack along new dimension 0
  std::vector<mlx::core::array> arrays;
  arrays.reserve(layer_slices.size());
  for (const auto& t : layer_slices) {
    arrays.push_back(t.array());
  }

  auto stacked_arr = mlx::core::stack(arrays, 0);
  return graph::Tensor(stacked_arr);
}

void Arena::write_k_cache_array(int layer_idx,
                                const std::vector<int>& block_ids,
                                const graph::Tensor& k_cache) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (block_ids.empty()) {
    return;
  }

  // k_cache shape: [num_pages, block_size, num_kv_heads, head_dim]
  // Need to write each page back to corresponding block's layer slice

  auto k_cache_arr = k_cache.array();
  int num_pages = k_cache.shape()[0];

  if (num_pages != static_cast<int>(block_ids.size())) {
    throw std::runtime_error(
        "Mismatch between k_cache pages and block_ids in write_k_cache_array");
  }

  for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
    int block_id = block_ids[page_idx];
    Block* block = get_block(block_id);
    if (!block) {
      throw std::runtime_error("Invalid block ID in write_k_cache_array: " +
                               std::to_string(block_id));
    }

    // Extract page: k_cache[page_idx, :, :, :]
    auto page_slice = mlx::core::slice(k_cache_arr, {page_idx, 0, 0, 0},
                                       {page_idx + 1, config_.block_size_tokens,
                                        config_.num_kv_heads, config_.head_dim},
                                       {1, 1, 1, 1});

    // Remove page dimension and add layer dimension
    // [1, block_size, num_kv_heads, head_dim] -> [block_size, num_kv_heads,
    // head_dim]
    page_slice = mlx::core::squeeze(page_slice, 0);
    // [block_size, num_kv_heads, head_dim] -> [1, block_size, num_kv_heads,
    // head_dim]
    page_slice = mlx::core::expand_dims(page_slice, 0);

    // Write to block's k_data[layer_idx, :, :, :]
    // We need to update the existing tensor
    // For simplicity, reconstruct the full block tensor with the updated layer
    auto k_arr = block->k_data.array();

    // Create update indices
    std::vector<mlx::core::array> indices;
    // Layer index
    indices.push_back(mlx::core::array(layer_idx));

    // Update the layer slice in place
    // This is simplified - in practice might need more sophisticated update
    // For now, we'll reconstruct by concatenating slices before and after
    if (layer_idx == 0) {
      // Replace first layer
      auto after =
          mlx::core::slice(k_arr, {1, 0, 0, 0},
                           {config_.num_layers, config_.block_size_tokens,
                            config_.num_kv_heads, config_.head_dim},
                           {1, 1, 1, 1});
      auto updated = mlx::core::concatenate({page_slice, after}, 0);
      block->k_data = graph::Tensor(updated);
    } else if (layer_idx == config_.num_layers - 1) {
      // Replace last layer
      auto before =
          mlx::core::slice(k_arr, {0, 0, 0, 0},
                           {config_.num_layers - 1, config_.block_size_tokens,
                            config_.num_kv_heads, config_.head_dim},
                           {1, 1, 1, 1});
      auto updated = mlx::core::concatenate({before, page_slice}, 0);
      block->k_data = graph::Tensor(updated);
    } else {
      // Replace middle layer
      auto before = mlx::core::slice(k_arr, {0, 0, 0, 0},
                                     {layer_idx, config_.block_size_tokens,
                                      config_.num_kv_heads, config_.head_dim},
                                     {1, 1, 1, 1});
      auto after =
          mlx::core::slice(k_arr, {layer_idx + 1, 0, 0, 0},
                           {config_.num_layers, config_.block_size_tokens,
                            config_.num_kv_heads, config_.head_dim},
                           {1, 1, 1, 1});
      auto updated = mlx::core::concatenate({before, page_slice, after}, 0);
      block->k_data = graph::Tensor(updated);
    }

    block->dirty = true;
  }
}

void Arena::write_v_cache_array(int layer_idx,
                                const std::vector<int>& block_ids,
                                const graph::Tensor& v_cache) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (block_ids.empty()) {
    return;
  }

  auto v_cache_arr = v_cache.array();
  int num_pages = v_cache.shape()[0];

  if (num_pages != static_cast<int>(block_ids.size())) {
    throw std::runtime_error(
        "Mismatch between v_cache pages and block_ids in write_v_cache_array");
  }

  for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
    int block_id = block_ids[page_idx];
    Block* block = get_block(block_id);
    if (!block) {
      throw std::runtime_error("Invalid block ID in write_v_cache_array: " +
                               std::to_string(block_id));
    }

    // Extract page
    auto page_slice = mlx::core::slice(v_cache_arr, {page_idx, 0, 0, 0},
                                       {page_idx + 1, config_.block_size_tokens,
                                        config_.num_kv_heads, config_.head_dim},
                                       {1, 1, 1, 1});

    page_slice = mlx::core::squeeze(page_slice, 0);
    page_slice = mlx::core::expand_dims(page_slice, 0);

    // Write to block's v_data[layer_idx, :, :, :]
    auto v_arr = block->v_data.array();

    if (layer_idx == 0) {
      auto after =
          mlx::core::slice(v_arr, {1, 0, 0, 0},
                           {config_.num_layers, config_.block_size_tokens,
                            config_.num_kv_heads, config_.head_dim},
                           {1, 1, 1, 1});
      auto updated = mlx::core::concatenate({page_slice, after}, 0);
      block->v_data = graph::Tensor(updated);
    } else if (layer_idx == config_.num_layers - 1) {
      auto before =
          mlx::core::slice(v_arr, {0, 0, 0, 0},
                           {config_.num_layers - 1, config_.block_size_tokens,
                            config_.num_kv_heads, config_.head_dim},
                           {1, 1, 1, 1});
      auto updated = mlx::core::concatenate({before, page_slice}, 0);
      block->v_data = graph::Tensor(updated);
    } else {
      auto before = mlx::core::slice(v_arr, {0, 0, 0, 0},
                                     {layer_idx, config_.block_size_tokens,
                                      config_.num_kv_heads, config_.head_dim},
                                     {1, 1, 1, 1});
      auto after =
          mlx::core::slice(v_arr, {layer_idx + 1, 0, 0, 0},
                           {config_.num_layers, config_.block_size_tokens,
                            config_.num_kv_heads, config_.head_dim},
                           {1, 1, 1, 1});
      auto updated = mlx::core::concatenate({before, page_slice, after}, 0);
      block->v_data = graph::Tensor(updated);
    }

    block->dirty = true;
  }
}

std::vector<mlx::core::array> Arena::get_k_block_arrays(
    const std::vector<int>& block_ids) {
  std::vector<mlx::core::array> result;
  result.reserve(block_ids.size());

  for (int block_id : block_ids) {
    Block* block = get_block(block_id);
    if (!block) {
      throw std::runtime_error("Invalid block ID: " + std::to_string(block_id));
    }

    // Return raw MLX array reference (zero-copy)
    result.push_back(block->k_data.array());

    // Update access time for LRU
    touch_block(block_id);
  }

  return result;
}

std::vector<mlx::core::array> Arena::get_v_block_arrays(
    const std::vector<int>& block_ids) {
  std::vector<mlx::core::array> result;
  result.reserve(block_ids.size());

  for (int block_id : block_ids) {
    Block* block = get_block(block_id);
    if (!block) {
      throw std::runtime_error("Invalid block ID: " + std::to_string(block_id));
    }

    // Return raw MLX array reference (zero-copy)
    result.push_back(block->v_data.array());

    // Update access time for LRU
    touch_block(block_id);
  }

  return result;
}

}  // namespace kv
}  // namespace runtime
}  // namespace mlxr
