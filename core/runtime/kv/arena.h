/**
 * @file arena.h
 * @brief Paged KV cache arena for efficient memory management
 *
 * Implements a block-based memory allocator for KV cache with:
 * - Fixed-size blocks (pages) for predictable allocation
 * - Free list management for fast allocation/deallocation
 * - Support for GPU and CPU memory
 * - Unified memory optimization for Apple Silicon
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../../graph/tensor.h"
#include "mlx/mlx.h"

namespace mlxr {
namespace runtime {
namespace kv {

/**
 * @brief Configuration for KV cache arena
 */
struct ArenaConfig {
  // Block size in tokens (16 or 32 recommended)
  int block_size_tokens = 32;

  // Number of blocks to pre-allocate
  int num_blocks = 1024;

  // Whether to allow CPU overflow when GPU memory is full
  bool allow_cpu_overflow = true;

  // Maximum blocks on CPU before eviction required
  int max_cpu_blocks = 256;

  // Model dimensions
  int num_layers = 32;
  int num_kv_heads = 4;
  int head_dim = 128;

  // Data type for KV cache
  mlx::core::Dtype dtype = mlx::core::float16;
};

/**
 * @brief Represents a single KV cache block (page)
 *
 * Each block stores K and V tensors for a fixed number of tokens
 * across all layers and heads.
 */
struct Block {
  // Unique block ID
  int block_id;

  // Reference count (number of sequences using this block)
  int ref_count;

  // Location: 0=GPU, 1=CPU
  int location;

  // Whether block is dirty (needs persistence)
  bool dirty;

  // Last access timestamp for LRU eviction
  uint64_t last_access_time;

  // K tensor storage: [num_layers, block_size_tokens, num_kv_heads, head_dim]
  graph::Tensor k_data;

  // V tensor storage: [num_layers, block_size_tokens, num_kv_heads, head_dim]
  graph::Tensor v_data;

  Block(int id, int location, const ArenaConfig& config);
};

/**
 * @brief KV cache arena with paged memory management
 *
 * Manages a pool of fixed-size blocks for storing KV cache across
 * multiple sequences. Supports both GPU and CPU memory with overflow.
 */
class Arena {
 public:
  /**
   * @brief Construct KV cache arena
   * @param config Arena configuration
   */
  explicit Arena(const ArenaConfig& config);

  /**
   * @brief Destructor - releases all blocks
   */
  ~Arena();

  // Disable copy and move
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  /**
   * @brief Allocate a new block from the free list
   * @return Block ID, or -1 if no blocks available
   */
  int allocate_block();

  /**
   * @brief Allocate multiple contiguous blocks
   * @param num_blocks Number of blocks to allocate
   * @return Vector of block IDs, empty if allocation fails
   */
  std::vector<int> allocate_blocks(int num_blocks);

  /**
   * @brief Free a block back to the free list
   * @param block_id Block to free
   */
  void free_block(int block_id);

  /**
   * @brief Free multiple blocks
   * @param block_ids Blocks to free
   */
  void free_blocks(const std::vector<int>& block_ids);

  /**
   * @brief Get reference to a block
   * @param block_id Block ID
   * @return Pointer to block, or nullptr if invalid
   */
  Block* get_block(int block_id);
  const Block* get_block(int block_id) const;

  /**
   * @brief Increment reference count for a block
   * @param block_id Block ID
   */
  void ref_block(int block_id);

  /**
   * @brief Decrement reference count for a block
   * @param block_id Block ID
   */
  void unref_block(int block_id);

  /**
   * @brief Touch a block (update last access time)
   * @param block_id Block ID
   */
  void touch_block(int block_id);

  /**
   * @brief Move block from GPU to CPU
   * @param block_id Block to move
   * @return True if successful
   */
  bool move_to_cpu(int block_id);

  /**
   * @brief Move block from CPU to GPU
   * @param block_id Block to move
   * @return True if successful
   */
  bool move_to_gpu(int block_id);

  /**
   * @brief Get number of free blocks on GPU
   */
  int num_free_gpu_blocks() const;

  /**
   * @brief Get number of free blocks on CPU
   */
  int num_free_cpu_blocks() const;

  /**
   * @brief Get total number of allocated blocks
   */
  int num_allocated_blocks() const;

  /**
   * @brief Get memory usage in bytes
   */
  size_t memory_usage() const;

  /**
   * @brief Get GPU memory usage in bytes
   */
  size_t gpu_memory_usage() const;

  /**
   * @brief Get CPU memory usage in bytes
   */
  size_t cpu_memory_usage() const;

  /**
   * @brief Get configuration
   */
  const ArenaConfig& config() const { return config_; }

  /**
   * @brief Clear all blocks and reset arena
   */
  void clear();

  /**
   * @brief Get statistics for monitoring
   */
  struct Stats {
    int total_blocks;
    int free_gpu_blocks;
    int free_cpu_blocks;
    int allocated_blocks;
    size_t total_memory_bytes;
    size_t gpu_memory_bytes;
    size_t cpu_memory_bytes;
    int num_gpu_to_cpu_moves;
    int num_cpu_to_gpu_moves;
  };

  Stats get_stats() const;

  /**
   * @brief Build K cache array for Metal primitives (for a specific layer)
   * @param layer_idx Layer index to extract
   * @param block_ids Block IDs to include (page table)
   * @return K cache array [num_pages, block_size, num_kv_heads, head_dim]
   *
   * Extracts layer_idx slice from each block and stacks them into a single
   * array suitable for Metal attention primitive input.
   */
  graph::Tensor build_k_cache_array(int layer_idx,
                                    const std::vector<int>& block_ids);

  /**
   * @brief Build V cache array for Metal primitives (for a specific layer)
   * @param layer_idx Layer index to extract
   * @param block_ids Block IDs to include (page table)
   * @return V cache array [num_pages, block_size, num_kv_heads, head_dim]
   *
   * Extracts layer_idx slice from each block and stacks them into a single
   * array suitable for Metal attention primitive input.
   */
  graph::Tensor build_v_cache_array(int layer_idx,
                                    const std::vector<int>& block_ids);

  /**
   * @brief Write K cache array back to blocks after Metal primitive execution
   * @param layer_idx Layer index to write to
   * @param block_ids Block IDs to update
   * @param k_cache K cache array [num_pages, block_size, num_kv_heads,
   * head_dim]
   */
  void write_k_cache_array(int layer_idx, const std::vector<int>& block_ids,
                           const graph::Tensor& k_cache);

  /**
   * @brief Write V cache array back to blocks after Metal primitive execution
   * @param layer_idx Layer index to write to
   * @param block_ids Block IDs to update
   * @param v_cache V cache array [num_pages, block_size, num_kv_heads,
   * head_dim]
   */
  void write_v_cache_array(int layer_idx, const std::vector<int>& block_ids,
                           const graph::Tensor& v_cache);

  /**
   * @brief Get raw MLX arrays for K cache blocks (ZERO-COPY)
   * @param block_ids Block IDs in page table order
   * @return Vector of K data arrays, one per block
   *         Each array shape: [num_layers, block_size, num_kv_heads, head_dim]
   *
   * Returns direct references to block storage without copying.
   * Metal kernels can access via page table: k_blocks[page_table[i]][layer_idx]
   */
  std::vector<mlx::core::array> get_k_block_arrays(
      const std::vector<int>& block_ids);

  /**
   * @brief Get raw MLX arrays for V cache blocks (ZERO-COPY)
   * @param block_ids Block IDs in page table order
   * @return Vector of V data arrays, one per block
   *         Each array shape: [num_layers, block_size, num_kv_heads, head_dim]
   *
   * Returns direct references to block storage without copying.
   * Metal kernels can access via page table: v_blocks[page_table[i]][layer_idx]
   */
  std::vector<mlx::core::array> get_v_block_arrays(
      const std::vector<int>& block_ids);

 private:
  /**
   * @brief Initialize arena by pre-allocating blocks
   */
  void initialize();

  /**
   * @brief Allocate a new physical block
   * @param location 0=GPU, 1=CPU
   * @return Block ID
   */
  int allocate_physical_block(int location);

  /**
   * @brief Get current timestamp for LRU tracking
   */
  uint64_t get_timestamp() const;

  // Configuration
  ArenaConfig config_;

  // All blocks (both free and allocated)
  std::vector<std::unique_ptr<Block>> blocks_;

  // Free block IDs on GPU
  std::vector<int> free_gpu_blocks_;

  // Free block IDs on CPU
  std::vector<int> free_cpu_blocks_;

  // Mapping from block ID to block index
  std::unordered_map<int, int> block_id_to_index_;

  // Next block ID to assign
  int next_block_id_;

  // Statistics
  mutable std::mutex stats_mutex_;
  int num_gpu_to_cpu_moves_;
  int num_cpu_to_gpu_moves_;

  // Timestamp counter
  mutable uint64_t timestamp_counter_;

  // Thread safety
  mutable std::mutex mutex_;
};

}  // namespace kv
}  // namespace runtime
}  // namespace mlxr
