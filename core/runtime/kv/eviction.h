/**
 * @file eviction.h
 * @brief KV cache block eviction policies
 *
 * Implements eviction strategies to free blocks when memory pressure is high:
 * - LRU (Least Recently Used) - default policy
 * - Working set aware eviction
 * - Persistence support for evicted blocks
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "pager.h"

namespace mlxr {
namespace runtime {
namespace kv {

/**
 * @brief Eviction policy configuration
 */
struct EvictionConfig {
  // Eviction trigger threshold (fraction of blocks used)
  float eviction_threshold = 0.9f;

  // Target usage after eviction (fraction of blocks)
  float target_usage = 0.7f;

  // Enable persistence of evicted blocks to disk
  bool enable_persistence = true;

  // Persistence directory
  std::string persistence_dir = "~/.mlxr/kv_cache";

  // Minimum blocks to keep per sequence
  int min_blocks_per_sequence = 1;
};

/**
 * @brief Eviction policy interface
 */
class EvictionPolicy {
 public:
  virtual ~EvictionPolicy() = default;

  /**
   * @brief Select blocks to evict
   * @param pager Pager with sequences and blocks
   * @param num_blocks_to_evict Number of blocks to free
   * @return Vector of (seq_id, block_idx) pairs to evict
   */
  virtual std::vector<std::pair<int, int>> select_blocks_to_evict(
      const Pager& pager, int num_blocks_to_evict) = 0;

  /**
   * @brief Check if eviction is needed
   * @param pager Pager to check
   * @return True if eviction should be triggered
   */
  virtual bool should_evict(const Pager& pager) const = 0;
};

/**
 * @brief LRU (Least Recently Used) eviction policy
 *
 * Evicts blocks that haven't been accessed recently, preserving
 * working set of active sequences.
 */
class LRUEvictionPolicy : public EvictionPolicy {
 public:
  explicit LRUEvictionPolicy(const EvictionConfig& config);

  std::vector<std::pair<int, int>> select_blocks_to_evict(
      const Pager& pager, int num_blocks_to_evict) override;

  bool should_evict(const Pager& pager) const override;

 private:
  EvictionConfig config_;
};

/**
 * @brief Working-set aware eviction policy
 *
 * Considers sequence importance and working set size when evicting.
 * Prefers to evict from inactive or low-priority sequences.
 */
class WorkingSetEvictionPolicy : public EvictionPolicy {
 public:
  explicit WorkingSetEvictionPolicy(const EvictionConfig& config);

  std::vector<std::pair<int, int>> select_blocks_to_evict(
      const Pager& pager, int num_blocks_to_evict) override;

  bool should_evict(const Pager& pager) const override;

  /**
   * @brief Set priority for a sequence (higher = keep longer)
   * @param seq_id Sequence ID
   * @param priority Priority value
   */
  void set_sequence_priority(int seq_id, float priority);

 private:
  EvictionConfig config_;
  std::unordered_map<int, float> sequence_priorities_;
};

/**
 * @brief Eviction manager with persistence support
 *
 * Coordinates eviction and optional persistence of KV cache blocks.
 */
class EvictionManager {
 public:
  /**
   * @brief Construct eviction manager
   * @param pager Pager to manage
   * @param config Eviction configuration
   */
  EvictionManager(std::shared_ptr<Pager> pager, const EvictionConfig& config);

  /**
   * @brief Set eviction policy
   * @param policy Eviction policy to use
   */
  void set_policy(std::unique_ptr<EvictionPolicy> policy);

  /**
   * @brief Check and perform eviction if needed
   * @return Number of blocks evicted
   */
  int maybe_evict();

  /**
   * @brief Force eviction of N blocks
   * @param num_blocks Number of blocks to evict
   * @return Number of blocks actually evicted
   */
  int evict_blocks(int num_blocks);

  /**
   * @brief Persist a block to disk
   * @param seq_id Sequence ID
   * @param block_idx Block index in sequence
   * @return True if successful
   */
  bool persist_block(int seq_id, int block_idx);

  /**
   * @brief Restore a block from disk
   * @param seq_id Sequence ID
   * @param block_idx Block index in sequence
   * @return True if successful
   */
  bool restore_block(int seq_id, int block_idx);

  /**
   * @brief Get eviction statistics
   */
  struct Stats {
    int num_evictions;
    int total_blocks_evicted;
    int blocks_persisted;
    int blocks_restored;
    size_t persistence_bytes;
  };

  Stats get_stats() const;

  /**
   * @brief Clear eviction statistics
   */
  void clear_stats();

 private:
  /**
   * @brief Evict a specific block
   * @param seq_id Sequence ID
   * @param block_idx Block index
   * @return True if successful
   */
  bool evict_block_impl(int seq_id, int block_idx);

  /**
   * @brief Get persistence path for a block
   */
  std::string get_persistence_path(int seq_id, int block_idx) const;

  std::shared_ptr<Pager> pager_;
  EvictionConfig config_;
  std::unique_ptr<EvictionPolicy> policy_;

  // Statistics
  mutable std::mutex stats_mutex_;
  Stats stats_;
};

}  // namespace kv
}  // namespace runtime
}  // namespace mlxr
