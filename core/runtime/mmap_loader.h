// Copyright © 2025 MLXR Development
// Memory-mapped weight loader for efficient model loading

#pragma once

#include <sys/mman.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlxr {

/**
 * Memory-mapped file region
 * Represents a contiguous region of memory-mapped file data
 */
struct MappedRegion {
  void* data;          // Pointer to mapped data
  size_t size;         // Size of mapped region in bytes
  size_t file_offset;  // Offset in file where region starts
  bool is_valid;       // Whether mapping is valid

  MappedRegion() : data(nullptr), size(0), file_offset(0), is_valid(false) {}

  MappedRegion(void* d, size_t s, size_t offset)
      : data(d), size(s), file_offset(offset), is_valid(true) {}

  // Get typed pointer to data
  template <typename T>
  T* as() const {
    return static_cast<T*>(data);
  }

  // Get data at byte offset
  void* at(size_t byte_offset) const {
    if (byte_offset >= size) {
      return nullptr;
    }
    return static_cast<char*>(data) + byte_offset;
  }
};

/**
 * Weight tensor metadata
 * Describes a tensor's location and properties within a mapped file
 */
struct WeightTensor {
  std::string name;  // Tensor name (e.g., "model.layers.0.attn.q_proj.weight")
  std::vector<int64_t> shape;  // Tensor dimensions
  size_t file_offset;          // Offset in file where tensor data starts
  size_t data_size;            // Size of tensor data in bytes
  std::string dtype;           // Data type (fp32, fp16, q4_0, etc.)

  // Quantization metadata (if applicable)
  int quant_block_size = 0;  // Block size for quantized tensors
  std::string quant_type;    // Quantization type string
};

/**
 * Memory-mapped weight loader
 *
 * Provides efficient read-only access to model weights via mmap.
 * Supports:
 * - Zero-copy weight access
 * - Page-aligned offsets for optimal performance
 * - Multiple concurrent mappings
 * - Lazy loading (map on first access)
 * - Automatic unmapping on destruction
 */
class MMapWeightLoader {
 public:
  /**
   * Create weight loader for a model file
   * @param file_path Path to model file (GGUF, safetensors, etc.)
   * @param read_only Whether to open file read-only (default: true)
   */
  explicit MMapWeightLoader(const std::string& file_path,
                            bool read_only = true);

  ~MMapWeightLoader();

  // Non-copyable, movable
  MMapWeightLoader(const MMapWeightLoader&) = delete;
  MMapWeightLoader& operator=(const MMapWeightLoader&) = delete;
  MMapWeightLoader(MMapWeightLoader&&) noexcept;
  MMapWeightLoader& operator=(MMapWeightLoader&&) noexcept;

  /**
   * Initialize loader (open file and read metadata)
   * @return true if successful
   */
  bool initialize();

  /**
   * Register a tensor with its location in the file
   * @param tensor Tensor metadata
   */
  void register_tensor(const WeightTensor& tensor);

  /**
   * Map a specific tensor into memory
   * @param tensor_name Name of tensor to map
   * @param prefetch Whether to advise kernel to prefetch pages
   * @return Mapped region, or invalid region on failure
   */
  MappedRegion map_tensor(const std::string& tensor_name,
                          bool prefetch = false);

  /**
   * Map a specific file region
   * @param offset Offset in file (will be page-aligned)
   * @param size Size to map
   * @param prefetch Whether to prefetch pages
   * @return Mapped region, or invalid region on failure
   */
  MappedRegion map_region(size_t offset, size_t size, bool prefetch = false);

  /**
   * Map entire file into memory
   * @param prefetch Whether to prefetch all pages
   * @return Mapped region, or invalid region on failure
   */
  MappedRegion map_all(bool prefetch = false);

  /**
   * Unmap a specific region
   * @param region Region to unmap
   */
  void unmap_region(const MappedRegion& region);

  /**
   * Get tensor metadata by name
   * @param tensor_name Name of tensor
   * @return Tensor metadata if found
   */
  std::optional<WeightTensor> get_tensor_info(
      const std::string& tensor_name) const;

  /**
   * List all registered tensor names
   */
  std::vector<std::string> list_tensors() const;

  /**
   * Get file size
   */
  size_t file_size() const { return file_size_; }

  /**
   * Get file path
   */
  const std::string& file_path() const { return file_path_; }

  /**
   * Check if file is currently mapped
   */
  bool is_mapped() const { return full_mapping_.is_valid; }

  /**
   * Get total bytes currently mapped
   */
  size_t total_mapped_bytes() const { return total_mapped_bytes_; }

  /**
   * Advise kernel about memory access patterns
   */
  enum class AdvicePattern {
    NORMAL,      // No specific pattern
    RANDOM,      // Random access (disable readahead)
    SEQUENTIAL,  // Sequential access (enable aggressive readahead)
    WILLNEED,    // Will need soon (prefetch pages)
    DONTNEED     // Won't need (free pages)
  };

  /**
   * Give advice to kernel about region access pattern
   * @param region Region to advise on
   * @param pattern Access pattern advice
   * @return true if successful
   */
  bool advise(const MappedRegion& region, AdvicePattern pattern);

  /**
   * Lock region in physical memory (prevent swapping)
   * @param region Region to lock
   * @return true if successful
   */
  bool lock_memory(const MappedRegion& region);

  /**
   * Unlock region (allow swapping)
   * @param region Region to unlock
   * @return true if successful
   */
  bool unlock_memory(const MappedRegion& region);

  /**
   * Get statistics about memory usage
   */
  struct Stats {
    size_t total_file_size;
    size_t total_mapped_bytes;
    size_t num_active_mappings;
    size_t num_registered_tensors;
    size_t page_size;
  };

  Stats get_stats() const;

 private:
  std::string file_path_;
  int fd_;  // File descriptor
  bool read_only_;
  size_t file_size_;
  size_t page_size_;  // System page size

  // Tensor registry
  std::unordered_map<std::string, WeightTensor> tensors_;

  // Active mappings
  std::vector<MappedRegion> active_mappings_;
  MappedRegion full_mapping_;  // Full file mapping (if mapped)
  size_t total_mapped_bytes_;

  /**
   * Align offset down to page boundary
   */
  size_t align_down_to_page(size_t offset) const;

  /**
   * Align size up to page boundary
   */
  size_t align_up_to_page(size_t size) const;

  /**
   * Get system page size
   */
  static size_t get_page_size();

  /**
   * Close file descriptor
   */
  void close_file();

  /**
   * Unmap all active mappings
   */
  void unmap_all();
};

}  // namespace mlxr
