/**
 * @file pager.cpp
 * @brief Implementation of KV cache page table management
 */

#include "pager.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace mlxr {
namespace runtime {
namespace kv {

// ============================================================================
// Sequence Implementation
// ============================================================================

Sequence::Sequence(int seq_id, int block_size_tokens)
    : seq_id_(seq_id),
      block_size_tokens_(block_size_tokens),
      num_tokens_(0),
      last_access_time_(get_current_time()),
      is_active_(true),
      parent_id_(-1) {}

int Sequence::num_blocks_required() const {
  if (num_tokens_ == 0) {
    return 0;
  }
  return (num_tokens_ + block_size_tokens_ - 1) / block_size_tokens_;
}

void Sequence::append_block(int block_id) {
  page_table_.push_back(block_id);
  touch();
}

int Sequence::get_block_id(int block_idx) const {
  if (block_idx < 0 || block_idx >= static_cast<int>(page_table_.size())) {
    return -1;
  }
  return page_table_[block_idx];
}

int Sequence::get_block_id_for_token(int token_pos) const {
  if (token_pos < 0 || token_pos >= num_tokens_) {
    return -1;
  }
  int block_idx = token_pos / block_size_tokens_;
  return get_block_id(block_idx);
}

uint64_t Sequence::get_current_time() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             now.time_since_epoch())
      .count();
}

// ============================================================================
// Pager Implementation
// ============================================================================

Pager::Pager(std::shared_ptr<Arena> arena) : arena_(arena), num_forks_(0) {}

Pager::~Pager() { clear(); }

bool Pager::create_sequence(int seq_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if sequence already exists
  if (sequences_.find(seq_id) != sequences_.end()) {
    std::cerr << "Sequence " << seq_id << " already exists" << std::endl;
    return false;
  }

  // Create new sequence
  auto seq =
      std::make_unique<Sequence>(seq_id, arena_->config().block_size_tokens);
  sequences_[seq_id] = std::move(seq);

  return true;
}

void Pager::delete_sequence(int seq_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = sequences_.find(seq_id);
  if (it == sequences_.end()) {
    return;
  }

  Sequence* seq = it->second.get();

  // Free all blocks in the page table
  for (int block_id : seq->page_table()) {
    arena_->unref_block(block_id);
  }

  // Remove sequence
  sequences_.erase(it);
}

bool Pager::fork_sequence(int parent_seq_id, int child_seq_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Find parent sequence
  auto parent_it = sequences_.find(parent_seq_id);
  if (parent_it == sequences_.end()) {
    std::cerr << "Parent sequence " << parent_seq_id << " not found"
              << std::endl;
    return false;
  }

  // Check if child already exists
  if (sequences_.find(child_seq_id) != sequences_.end()) {
    std::cerr << "Child sequence " << child_seq_id << " already exists"
              << std::endl;
    return false;
  }

  Sequence* parent = parent_it->second.get();

  // Create child sequence
  auto child = std::make_unique<Sequence>(child_seq_id,
                                          arena_->config().block_size_tokens);
  child->set_parent_id(parent_seq_id);
  child->set_num_tokens(parent->num_tokens());

  // Copy page table and increment ref counts (COW - copy-on-write)
  const auto& parent_page_table = parent->page_table();
  child->set_page_table(parent_page_table);

  for (int block_id : parent_page_table) {
    arena_->ref_block(block_id);
  }

  sequences_[child_seq_id] = std::move(child);
  num_forks_++;

  return true;
}

Sequence* Pager::get_sequence(int seq_id) {
  // No lock for read access to existing sequence
  auto it = sequences_.find(seq_id);
  if (it == sequences_.end()) {
    return nullptr;
  }
  return it->second.get();
}

const Sequence* Pager::get_sequence(int seq_id) const {
  auto it = sequences_.find(seq_id);
  if (it == sequences_.end()) {
    return nullptr;
  }
  return it->second.get();
}

bool Pager::allocate_blocks_for_sequence(int seq_id, int target_num_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);

  Sequence* seq = get_sequence(seq_id);
  if (!seq) {
    std::cerr << "Sequence " << seq_id << " not found" << std::endl;
    return false;
  }

  int current_tokens = seq->num_tokens();
  if (target_num_tokens <= current_tokens) {
    // No allocation needed
    seq->set_num_tokens(target_num_tokens);
    return true;
  }

  // Calculate number of new blocks needed
  int current_blocks = seq->page_table().size();
  int target_blocks =
      (target_num_tokens + seq->block_size() - 1) / seq->block_size();
  int num_new_blocks = target_blocks - current_blocks;

  if (num_new_blocks <= 0) {
    // No new blocks needed, just growing within existing blocks
    seq->set_num_tokens(target_num_tokens);
    return true;
  }

  // Allocate new blocks
  return allocate_blocks_for_sequence_impl(seq, num_new_blocks);
}

bool Pager::allocate_blocks_for_sequence_impl(Sequence* seq,
                                              int num_new_blocks) {
  // Allocate blocks from arena
  std::vector<int> new_block_ids = arena_->allocate_blocks(num_new_blocks);

  if (new_block_ids.empty()) {
    std::cerr << "Failed to allocate " << num_new_blocks
              << " blocks for sequence " << seq->id() << std::endl;
    return false;
  }

  // Append blocks to page table
  for (int block_id : new_block_ids) {
    seq->append_block(block_id);
  }

  return true;
}

Block* Pager::get_block(int block_id) { return arena_->get_block(block_id); }

const Block* Pager::get_block(int block_id) const {
  return arena_->get_block(block_id);
}

void Pager::touch_sequence(int seq_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  Sequence* seq = get_sequence(seq_id);
  if (!seq) {
    return;
  }

  seq->touch();

  // Touch all blocks in the sequence
  for (int block_id : seq->page_table()) {
    arena_->touch_block(block_id);
  }
}

int Pager::num_sequences() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(sequences_.size());
}

std::vector<int> Pager::get_sequence_ids() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int> ids;
  ids.reserve(sequences_.size());

  for (const auto& pair : sequences_) {
    ids.push_back(pair.first);
  }

  return ids;
}

void Pager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Delete all sequences (this frees their blocks)
  for (auto& pair : sequences_) {
    Sequence* seq = pair.second.get();
    for (int block_id : seq->page_table()) {
      arena_->unref_block(block_id);
    }
  }

  sequences_.clear();
  num_forks_ = 0;
}

Pager::Stats Pager::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);

  Stats stats;
  stats.num_sequences = static_cast<int>(sequences_.size());
  stats.num_active_sequences = 0;
  stats.total_tokens = 0;
  stats.total_blocks_allocated = 0;
  stats.num_forks = num_forks_;

  for (const auto& pair : sequences_) {
    const Sequence* seq = pair.second.get();
    if (seq->is_active()) {
      stats.num_active_sequences++;
    }
    stats.total_tokens += seq->num_tokens();
    stats.total_blocks_allocated += static_cast<int>(seq->page_table().size());
  }

  return stats;
}

graph::Tensor Pager::build_page_table_array(int seq_id, int max_blocks) {
  std::lock_guard<std::mutex> lock(mutex_);

  Sequence* seq = get_sequence(seq_id);
  if (!seq) {
    throw std::runtime_error("Sequence not found in build_page_table_array: " +
                             std::to_string(seq_id));
  }

  const auto& page_table = seq->page_table();

  // Create array of block IDs, padded with -1
  std::vector<int> table_data;
  table_data.reserve(max_blocks);

  // Copy actual block IDs
  for (size_t i = 0;
       i < page_table.size() && i < static_cast<size_t>(max_blocks); ++i) {
    table_data.push_back(page_table[i]);
  }

  // Pad with -1 for unused slots
  while (table_data.size() < static_cast<size_t>(max_blocks)) {
    table_data.push_back(-1);
  }

  // Create tensor from data
  // Shape: [batch=1, max_blocks] - Metal kernels expect 2D page table
  auto arr =
      mlx::core::array(table_data.data(), {1, max_blocks}, mlx::core::int32);

  return graph::Tensor(arr);
}

}  // namespace kv
}  // namespace runtime
}  // namespace mlxr
