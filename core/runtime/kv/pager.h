/**
 * @file pager.h
 * @brief Page table management for KV cache sequences
 *
 * Manages per-sequence page tables that map logical token positions
 * to physical KV cache blocks. Supports:
 * - Dynamic growth as sequences extend
 * - Copy-on-write for sequence forking
 * - Efficient block sharing between sequences
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "arena.h"

namespace mlxr {
namespace runtime {
namespace kv {

/**
 * @brief Represents a logical sequence with its page table
 *
 * Maps logical token positions to physical block IDs.
 * Each block covers block_size_tokens contiguous tokens.
 */
class Sequence {
 public:
  /**
   * @brief Construct a new sequence
   * @param seq_id Unique sequence ID
   * @param block_size_tokens Tokens per block
   */
  Sequence(int seq_id, int block_size_tokens);

  /**
   * @brief Get sequence ID
   */
  int id() const { return seq_id_; }

  /**
   * @brief Get number of tokens in sequence
   */
  int num_tokens() const { return num_tokens_; }

  /**
   * @brief Set number of tokens (grows page table if needed)
   * @param num_tokens New token count
   */
  void set_num_tokens(int num_tokens) { num_tokens_ = num_tokens; }

  /**
   * @brief Get block size in tokens
   */
  int block_size() const { return block_size_tokens_; }

  /**
   * @brief Get number of blocks required for current tokens
   */
  int num_blocks_required() const;

  /**
   * @brief Get page table (block IDs)
   */
  const std::vector<int>& page_table() const { return page_table_; }

  /**
   * @brief Set page table
   */
  void set_page_table(const std::vector<int>& page_table) {
    page_table_ = page_table;
  }

  /**
   * @brief Append a block to page table
   */
  void append_block(int block_id);

  /**
   * @brief Get block ID for a given logical block index
   * @param block_idx Logical block index
   * @return Block ID, or -1 if not allocated
   */
  int get_block_id(int block_idx) const;

  /**
   * @brief Get block ID for a given token position
   * @param token_pos Token position
   * @return Block ID, or -1 if not allocated
   */
  int get_block_id_for_token(int token_pos) const;

  /**
   * @brief Get last access time
   */
  uint64_t last_access_time() const { return last_access_time_; }

  /**
   * @brief Update last access time
   */
  void touch() { last_access_time_ = get_current_time(); }

  /**
   * @brief Whether sequence is active (not finished)
   */
  bool is_active() const { return is_active_; }

  /**
   * @brief Mark sequence as finished
   */
  void finish() { is_active_ = false; }

  /**
   * @brief Get parent sequence ID (for forking/beam search)
   */
  int parent_id() const { return parent_id_; }

  /**
   * @brief Set parent sequence ID
   */
  void set_parent_id(int parent_id) { parent_id_ = parent_id; }

 private:
  static uint64_t get_current_time();

  int seq_id_;
  int block_size_tokens_;
  int num_tokens_;
  std::vector<int> page_table_;  // Maps block_idx -> block_id
  uint64_t last_access_time_;
  bool is_active_;
  int parent_id_;  // For beam search/forking (-1 if root)
};

/**
 * @brief Manages page tables for multiple sequences
 *
 * Coordinates between sequences and the KV cache arena, handling:
 * - Sequence creation and deletion
 * - Block allocation for growing sequences
 * - Block sharing and reference counting
 * - Copy-on-write for sequence forking
 */
class Pager {
 public:
  /**
   * @brief Construct pager with arena
   * @param arena KV cache arena
   */
  explicit Pager(std::shared_ptr<Arena> arena);

  /**
   * @brief Destructor
   */
  ~Pager();

  // Disable copy and move
  Pager(const Pager&) = delete;
  Pager& operator=(const Pager&) = delete;

  /**
   * @brief Create a new sequence
   * @param seq_id Unique sequence ID
   * @return True if successful
   */
  bool create_sequence(int seq_id);

  /**
   * @brief Delete a sequence and free its blocks
   * @param seq_id Sequence ID
   */
  void delete_sequence(int seq_id);

  /**
   * @brief Fork a sequence (for beam search)
   * @param parent_seq_id Parent sequence
   * @param child_seq_id New child sequence
   * @return True if successful
   */
  bool fork_sequence(int parent_seq_id, int child_seq_id);

  /**
   * @brief Get sequence by ID
   * @param seq_id Sequence ID
   * @return Pointer to sequence, or nullptr if not found
   */
  Sequence* get_sequence(int seq_id);
  const Sequence* get_sequence(int seq_id) const;

  /**
   * @brief Allocate blocks for a sequence to reach target token count
   * @param seq_id Sequence ID
   * @param target_num_tokens Target token count
   * @return True if allocation successful
   */
  bool allocate_blocks_for_sequence(int seq_id, int target_num_tokens);

  /**
   * @brief Get block pointer from arena
   * @param block_id Block ID
   * @return Block pointer, or nullptr if invalid
   */
  Block* get_block(int block_id);
  const Block* get_block(int block_id) const;

  /**
   * @brief Touch all blocks in a sequence (for LRU)
   * @param seq_id Sequence ID
   */
  void touch_sequence(int seq_id);

  /**
   * @brief Get number of sequences
   */
  int num_sequences() const;

  /**
   * @brief Get all sequence IDs
   */
  std::vector<int> get_sequence_ids() const;

  /**
   * @brief Get arena
   */
  Arena& arena() { return *arena_; }
  const Arena& arena() const { return *arena_; }

  /**
   * @brief Clear all sequences
   */
  void clear();

  /**
   * @brief Build page table array for Metal primitives
   * @param seq_id Sequence ID
   * @param max_blocks Maximum blocks to include
   * @return Page table array [batch=1, max_blocks] with block IDs (-1 for
   * empty)
   *
   * Creates a dense 2D array of block IDs for the sequence, padded with -1.
   * Shape is [1, max_blocks] to match Metal kernel expectations.
   * Used by Metal attention primitives.
   */
  graph::Tensor build_page_table_array(int seq_id, int max_blocks);

  /**
   * @brief Get statistics
   */
  struct Stats {
    int num_sequences;
    int num_active_sequences;
    int total_tokens;
    int total_blocks_allocated;
    int num_forks;
  };

  Stats get_stats() const;

 private:
  /**
   * @brief Allocate blocks for sequence growth
   * @param seq Sequence to grow
   * @param num_new_blocks Number of blocks to allocate
   * @return True if successful
   */
  bool allocate_blocks_for_sequence_impl(Sequence* seq, int num_new_blocks);

  std::shared_ptr<Arena> arena_;
  std::unordered_map<int, std::unique_ptr<Sequence>> sequences_;
  mutable std::mutex mutex_;
  int num_forks_;
};

}  // namespace kv
}  // namespace runtime
}  // namespace mlxr
