/**
 * @file eviction.cpp
 * @brief Implementation of KV cache eviction policies
 */

#include "eviction.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>

namespace mlxr {
namespace runtime {
namespace kv {

// ============================================================================
// LRU Eviction Policy
// ============================================================================

LRUEvictionPolicy::LRUEvictionPolicy(const EvictionConfig& config)
    : config_(config) {}

bool LRUEvictionPolicy::should_evict(const Pager& pager) const {
  const Arena& arena = pager.arena();
  int total_blocks = arena.config().num_blocks;
  int free_blocks = arena.num_free_gpu_blocks() + arena.num_free_cpu_blocks();
  float usage = 1.0f - static_cast<float>(free_blocks) / total_blocks;

  return usage >= config_.eviction_threshold;
}

std::vector<std::pair<int, int>> LRUEvictionPolicy::select_blocks_to_evict(
    const Pager& pager, int num_blocks_to_evict) {
  std::vector<std::pair<int, int>> blocks_to_evict;
  if (num_blocks_to_evict <= 0) {
    return blocks_to_evict;
  }

  // Build list of all (seq_id, block_idx, block_id, timestamp) tuples
  struct BlockInfo {
    int seq_id;
    int block_idx;
    int block_id;
    uint64_t timestamp;

    bool operator<(const BlockInfo& other) const {
      return timestamp > other.timestamp;  // Min heap (oldest first)
    }
  };

  std::priority_queue<BlockInfo> block_queue;

  // Iterate through all sequences
  std::vector<int> seq_ids = pager.get_sequence_ids();

  for (int seq_id : seq_ids) {
    const Sequence* seq = pager.get_sequence(seq_id);
    if (!seq) continue;

    const auto& page_table = seq->page_table();

    // Skip sequences that are at minimum blocks
    if (static_cast<int>(page_table.size()) <=
        config_.min_blocks_per_sequence) {
      continue;
    }

    // Add blocks to priority queue, starting from oldest
    for (int block_idx = 0; block_idx < static_cast<int>(page_table.size());
         ++block_idx) {
      int block_id = page_table[block_idx];
      if (block_id < 0) continue;

      const Block* block = pager.get_block(block_id);
      if (!block || block->ref_count > 1) {
        // Skip blocks that are shared (COW) or invalid
        continue;
      }

      BlockInfo info;
      info.seq_id = seq_id;
      info.block_idx = block_idx;
      info.block_id = block_id;
      info.timestamp = block->last_access_time;

      block_queue.push(info);
    }
  }

  // Select oldest blocks
  while (!block_queue.empty() &&
         static_cast<int>(blocks_to_evict.size()) < num_blocks_to_evict) {
    BlockInfo info = block_queue.top();
    block_queue.pop();

    // Double-check we're not evicting below minimum
    const Sequence* seq = pager.get_sequence(info.seq_id);
    if (seq) {
      int remaining_blocks = static_cast<int>(seq->page_table().size()) - 1;

      // Count how many blocks from this sequence are already in eviction list
      int blocks_evicting_from_seq = 0;
      for (const auto& pair : blocks_to_evict) {
        if (pair.first == info.seq_id) {
          blocks_evicting_from_seq++;
        }
      }

      if (remaining_blocks - blocks_evicting_from_seq >=
          config_.min_blocks_per_sequence) {
        blocks_to_evict.push_back({info.seq_id, info.block_idx});
      }
    }
  }

  return blocks_to_evict;
}

// ============================================================================
// Working Set Eviction Policy
// ============================================================================

WorkingSetEvictionPolicy::WorkingSetEvictionPolicy(const EvictionConfig& config)
    : config_(config) {}

bool WorkingSetEvictionPolicy::should_evict(const Pager& pager) const {
  const Arena& arena = pager.arena();
  int total_blocks = arena.config().num_blocks;
  int free_blocks = arena.num_free_gpu_blocks() + arena.num_free_cpu_blocks();
  float usage = 1.0f - static_cast<float>(free_blocks) / total_blocks;

  return usage >= config_.eviction_threshold;
}

void WorkingSetEvictionPolicy::set_sequence_priority(int seq_id,
                                                     float priority) {
  sequence_priorities_[seq_id] = priority;
}

std::vector<std::pair<int, int>>
WorkingSetEvictionPolicy::select_blocks_to_evict(const Pager& pager,
                                                 int num_blocks_to_evict) {
  std::vector<std::pair<int, int>> blocks_to_evict;
  if (num_blocks_to_evict <= 0) {
    return blocks_to_evict;
  }

  // Build list with priority consideration
  struct BlockInfo {
    int seq_id;
    int block_idx;
    int block_id;
    uint64_t timestamp;
    float priority;
    bool is_active;

    // Lower score = higher priority for eviction
    float eviction_score() const {
      float score = static_cast<float>(timestamp);

      // Boost score for inactive sequences (evict them first)
      if (!is_active) {
        score *= 0.1f;
      }

      // Apply priority (lower priority = higher eviction score)
      score *= (1.0f / (priority + 0.1f));

      return score;
    }

    bool operator<(const BlockInfo& other) const {
      return eviction_score() > other.eviction_score();  // Min heap
    }
  };

  std::priority_queue<BlockInfo> block_queue;

  // Iterate through all sequences
  std::vector<int> seq_ids = pager.get_sequence_ids();

  for (int seq_id : seq_ids) {
    const Sequence* seq = pager.get_sequence(seq_id);
    if (!seq) continue;

    float priority = 1.0f;
    auto it = sequence_priorities_.find(seq_id);
    if (it != sequence_priorities_.end()) {
      priority = it->second;
    }

    const auto& page_table = seq->page_table();

    // Skip if at minimum
    if (static_cast<int>(page_table.size()) <=
        config_.min_blocks_per_sequence) {
      continue;
    }

    // Add blocks to queue
    for (int block_idx = 0; block_idx < static_cast<int>(page_table.size());
         ++block_idx) {
      int block_id = page_table[block_idx];
      if (block_id < 0) continue;

      const Block* block = pager.get_block(block_id);
      if (!block || block->ref_count > 1) {
        continue;
      }

      BlockInfo info;
      info.seq_id = seq_id;
      info.block_idx = block_idx;
      info.block_id = block_id;
      info.timestamp = block->last_access_time;
      info.priority = priority;
      info.is_active = seq->is_active();

      block_queue.push(info);
    }
  }

  // Select blocks with lowest eviction score
  while (!block_queue.empty() &&
         static_cast<int>(blocks_to_evict.size()) < num_blocks_to_evict) {
    BlockInfo info = block_queue.top();
    block_queue.pop();

    const Sequence* seq = pager.get_sequence(info.seq_id);
    if (seq) {
      int remaining_blocks = static_cast<int>(seq->page_table().size()) - 1;

      int blocks_evicting_from_seq = 0;
      for (const auto& pair : blocks_to_evict) {
        if (pair.first == info.seq_id) {
          blocks_evicting_from_seq++;
        }
      }

      if (remaining_blocks - blocks_evicting_from_seq >=
          config_.min_blocks_per_sequence) {
        blocks_to_evict.push_back({info.seq_id, info.block_idx});
      }
    }
  }

  return blocks_to_evict;
}

// ============================================================================
// Eviction Manager
// ============================================================================

EvictionManager::EvictionManager(std::shared_ptr<Pager> pager,
                                 const EvictionConfig& config)
    : pager_(pager), config_(config) {
  // Default to LRU policy
  policy_ = std::make_unique<LRUEvictionPolicy>(config);

  // Clear stats
  stats_ = {};
}

void EvictionManager::set_policy(std::unique_ptr<EvictionPolicy> policy) {
  policy_ = std::move(policy);
}

int EvictionManager::maybe_evict() {
  if (!policy_) {
    return 0;
  }

  if (!policy_->should_evict(*pager_)) {
    return 0;
  }

  // Calculate how many blocks to evict
  const Arena& arena = pager_->arena();
  int total_blocks = arena.config().num_blocks;
  int free_blocks = arena.num_free_gpu_blocks() + arena.num_free_cpu_blocks();
  float current_usage = 1.0f - static_cast<float>(free_blocks) / total_blocks;

  int target_free =
      static_cast<int>(total_blocks * (1.0f - config_.target_usage));
  int num_blocks_to_evict = target_free - free_blocks;

  if (num_blocks_to_evict <= 0) {
    return 0;
  }

  return evict_blocks(num_blocks_to_evict);
}

int EvictionManager::evict_blocks(int num_blocks) {
  if (!policy_) {
    return 0;
  }

  // Select blocks to evict
  auto blocks_to_evict = policy_->select_blocks_to_evict(*pager_, num_blocks);

  int evicted = 0;
  for (const auto& [seq_id, block_idx] : blocks_to_evict) {
    if (evict_block_impl(seq_id, block_idx)) {
      evicted++;
    }
  }

  if (evicted > 0) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.num_evictions++;
    stats_.total_blocks_evicted += evicted;
  }

  return evicted;
}

bool EvictionManager::evict_block_impl(int seq_id, int block_idx) {
  Sequence* seq = pager_->get_sequence(seq_id);
  if (!seq) {
    return false;
  }

  // Get block ID
  int block_id = seq->get_block_id(block_idx);
  if (block_id < 0) {
    return false;
  }

  // Persist if enabled
  if (config_.enable_persistence) {
    if (!persist_block(seq_id, block_idx)) {
      std::cerr << "Warning: Failed to persist block before eviction"
                << std::endl;
    }
  }

  // Remove block from page table (set to -1 to mark as evicted)
  auto page_table = seq->page_table();
  if (block_idx >= 0 && block_idx < static_cast<int>(page_table.size())) {
    page_table[block_idx] = -1;
    seq->set_page_table(page_table);
  }

  // Free the block
  pager_->arena().free_block(block_id);

  return true;
}

std::string EvictionManager::get_persistence_path(int seq_id,
                                                  int block_idx) const {
  std::filesystem::path dir(config_.persistence_dir);

  // Create directory if it doesn't exist
  std::filesystem::create_directories(dir);

  std::string filename =
      "kv_" + std::to_string(seq_id) + "_" + std::to_string(block_idx) + ".bin";

  return (dir / filename).string();
}

bool EvictionManager::persist_block(int seq_id, int block_idx) {
  Sequence* seq = pager_->get_sequence(seq_id);
  if (!seq) {
    return false;
  }

  int block_id = seq->get_block_id(block_idx);
  if (block_id < 0) {
    return false;
  }

  Block* block = pager_->get_block(block_id);
  if (!block) {
    return false;
  }

  // Get persistence path
  std::string path = get_persistence_path(seq_id, block_idx);

  try {
    // Evaluate tensors to materialize data
    mlx::core::eval(block->k_data.array());
    mlx::core::eval(block->v_data.array());

    // For now, just create placeholder file
    // Full implementation would serialize MLX arrays to disk
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    // Write metadata
    file.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    file.write(reinterpret_cast<const char*>(&block_idx), sizeof(block_idx));
    file.write(reinterpret_cast<const char*>(&block_id), sizeof(block_id));

    // TODO: Serialize K and V tensors
    // For now, just mark as persisted
    file.close();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.blocks_persisted++;
    stats_.persistence_bytes += std::filesystem::file_size(path);

    return true;

  } catch (const std::exception& e) {
    std::cerr << "Failed to persist block: " << e.what() << std::endl;
    return false;
  }
}

bool EvictionManager::restore_block(int seq_id, int block_idx) {
  std::string path = get_persistence_path(seq_id, block_idx);

  if (!std::filesystem::exists(path)) {
    return false;
  }

  try {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    // Read metadata
    int stored_seq_id, stored_block_idx, stored_block_id;
    file.read(reinterpret_cast<char*>(&stored_seq_id), sizeof(stored_seq_id));
    file.read(reinterpret_cast<char*>(&stored_block_idx),
              sizeof(stored_block_idx));
    file.read(reinterpret_cast<char*>(&stored_block_id),
              sizeof(stored_block_id));

    // Verify metadata
    if (stored_seq_id != seq_id || stored_block_idx != block_idx) {
      return false;
    }

    // TODO: Deserialize K and V tensors and restore to block

    file.close();

    // Remove persisted file
    std::filesystem::remove(path);

    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.blocks_restored++;

    return true;

  } catch (const std::exception& e) {
    std::cerr << "Failed to restore block: " << e.what() << std::endl;
    return false;
  }
}

EvictionManager::Stats EvictionManager::get_stats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

void EvictionManager::clear_stats() {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  stats_ = {};
}

}  // namespace kv
}  // namespace runtime
}  // namespace mlxr
