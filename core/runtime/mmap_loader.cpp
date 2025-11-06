// Copyright © 2025 MLXR Development
// Memory-mapped weight loader implementation

#include "mmap_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>

namespace mlxr {

MMapWeightLoader::MMapWeightLoader(const std::string& file_path, bool read_only)
    : file_path_(file_path),
      fd_(-1),
      read_only_(read_only),
      file_size_(0),
      page_size_(get_page_size()),
      total_mapped_bytes_(0) {}

MMapWeightLoader::~MMapWeightLoader() {
  unmap_all();
  close_file();
}

MMapWeightLoader::MMapWeightLoader(MMapWeightLoader&& other) noexcept
    : file_path_(std::move(other.file_path_)),
      fd_(other.fd_),
      read_only_(other.read_only_),
      file_size_(other.file_size_),
      page_size_(other.page_size_),
      tensors_(std::move(other.tensors_)),
      active_mappings_(std::move(other.active_mappings_)),
      full_mapping_(other.full_mapping_),
      total_mapped_bytes_(other.total_mapped_bytes_) {
  other.fd_ = -1;
  other.file_size_ = 0;
  other.total_mapped_bytes_ = 0;
  other.full_mapping_ = MappedRegion();
}

MMapWeightLoader& MMapWeightLoader::operator=(
    MMapWeightLoader&& other) noexcept {
  if (this != &other) {
    unmap_all();
    close_file();

    file_path_ = std::move(other.file_path_);
    fd_ = other.fd_;
    read_only_ = other.read_only_;
    file_size_ = other.file_size_;
    page_size_ = other.page_size_;
    tensors_ = std::move(other.tensors_);
    active_mappings_ = std::move(other.active_mappings_);
    full_mapping_ = other.full_mapping_;
    total_mapped_bytes_ = other.total_mapped_bytes_;

    other.fd_ = -1;
    other.file_size_ = 0;
    other.total_mapped_bytes_ = 0;
    other.full_mapping_ = MappedRegion();
  }
  return *this;
}

bool MMapWeightLoader::initialize() {
  // Open file
  int flags = read_only_ ? O_RDONLY : O_RDWR;
  fd_ = open(file_path_.c_str(), flags);

  if (fd_ < 0) {
    std::cerr << "Failed to open file: " << file_path_ << std::endl;
    return false;
  }

  // Get file size
  struct stat st;
  if (fstat(fd_, &st) != 0) {
    std::cerr << "Failed to stat file: " << file_path_ << std::endl;
    close_file();
    return false;
  }

  file_size_ = st.st_size;

  if (file_size_ == 0) {
    std::cerr << "File is empty: " << file_path_ << std::endl;
    close_file();
    return false;
  }

  return true;
}

void MMapWeightLoader::register_tensor(const WeightTensor& tensor) {
  tensors_[tensor.name] = tensor;
}

MappedRegion MMapWeightLoader::map_tensor(const std::string& tensor_name,
                                          bool prefetch) {
  auto it = tensors_.find(tensor_name);
  if (it == tensors_.end()) {
    std::cerr << "Tensor not found: " << tensor_name << std::endl;
    return MappedRegion();
  }

  const WeightTensor& tensor = it->second;
  return map_region(tensor.file_offset, tensor.data_size, prefetch);
}

MappedRegion MMapWeightLoader::map_region(size_t offset, size_t size,
                                          bool prefetch) {
  if (fd_ < 0) {
    std::cerr << "File not open" << std::endl;
    return MappedRegion();
  }

  if (offset + size > file_size_) {
    std::cerr << "Region exceeds file size" << std::endl;
    return MappedRegion();
  }

  // Align offset to page boundary
  size_t page_offset = align_down_to_page(offset);
  size_t offset_adjustment = offset - page_offset;
  size_t aligned_size = align_up_to_page(size + offset_adjustment);

  // Map region
  int prot = PROT_READ;
  if (!read_only_) {
    prot |= PROT_WRITE;
  }

  int flags = MAP_SHARED;

  void* addr = mmap(nullptr, aligned_size, prot, flags, fd_, page_offset);

  if (addr == MAP_FAILED) {
    std::cerr << "mmap failed for region at offset " << offset << " size "
              << size << std::endl;
    return MappedRegion();
  }

  // Adjust pointer to actual data start
  void* data_ptr = static_cast<char*>(addr) + offset_adjustment;

  MappedRegion region(data_ptr, size, offset);

  // Store the actual mapping info for unmapping
  MappedRegion full_region(addr, aligned_size, page_offset);
  active_mappings_.push_back(full_region);
  total_mapped_bytes_ += aligned_size;

  // Prefetch if requested
  if (prefetch) {
    advise(full_region, AdvicePattern::WILLNEED);
  }

  return region;
}

MappedRegion MMapWeightLoader::map_all(bool prefetch) {
  if (full_mapping_.is_valid) {
    return full_mapping_;
  }

  if (fd_ < 0) {
    std::cerr << "File not open" << std::endl;
    return MappedRegion();
  }

  // Map entire file
  int prot = PROT_READ;
  if (!read_only_) {
    prot |= PROT_WRITE;
  }

  int flags = MAP_SHARED;

  void* addr = mmap(nullptr, file_size_, prot, flags, fd_, 0);

  if (addr == MAP_FAILED) {
    std::cerr << "mmap failed for entire file" << std::endl;
    return MappedRegion();
  }

  full_mapping_ = MappedRegion(addr, file_size_, 0);
  total_mapped_bytes_ += file_size_;

  // Prefetch if requested
  if (prefetch) {
    advise(full_mapping_, AdvicePattern::WILLNEED);
  }

  return full_mapping_;
}

void MMapWeightLoader::unmap_region(const MappedRegion& region) {
  if (!region.is_valid || region.data == nullptr) {
    return;
  }

  // Align to find actual mapping start
  size_t page_offset = align_down_to_page(region.file_offset);
  size_t offset_adjustment = region.file_offset - page_offset;
  void* map_start = static_cast<char*>(region.data) - offset_adjustment;
  size_t map_size = align_up_to_page(region.size + offset_adjustment);

  if (munmap(map_start, map_size) == 0) {
    total_mapped_bytes_ -= map_size;

    // Remove from active mappings
    active_mappings_.erase(
        std::remove_if(
            active_mappings_.begin(), active_mappings_.end(),
            [&](const MappedRegion& r) { return r.data == map_start; }),
        active_mappings_.end());
  }
}

std::optional<WeightTensor> MMapWeightLoader::get_tensor_info(
    const std::string& tensor_name) const {
  auto it = tensors_.find(tensor_name);
  if (it == tensors_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<std::string> MMapWeightLoader::list_tensors() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());

  for (const auto& [name, tensor] : tensors_) {
    names.push_back(name);
  }

  return names;
}

bool MMapWeightLoader::advise(const MappedRegion& region,
                              AdvicePattern pattern) {
  if (!region.is_valid) {
    return false;
  }

  int advice;
  switch (pattern) {
    case AdvicePattern::NORMAL:
      advice = MADV_NORMAL;
      break;
    case AdvicePattern::RANDOM:
      advice = MADV_RANDOM;
      break;
    case AdvicePattern::SEQUENTIAL:
      advice = MADV_SEQUENTIAL;
      break;
    case AdvicePattern::WILLNEED:
      advice = MADV_WILLNEED;
      break;
    case AdvicePattern::DONTNEED:
      advice = MADV_DONTNEED;
      break;
    default:
      return false;
  }

  // Align region for madvise
  size_t page_offset = align_down_to_page(region.file_offset);
  size_t offset_adjustment = region.file_offset - page_offset;
  void* map_start = static_cast<char*>(region.data) - offset_adjustment;
  size_t map_size = align_up_to_page(region.size + offset_adjustment);

  return madvise(map_start, map_size, advice) == 0;
}

bool MMapWeightLoader::lock_memory(const MappedRegion& region) {
  if (!region.is_valid) {
    return false;
  }

  // Align region
  size_t page_offset = align_down_to_page(region.file_offset);
  size_t offset_adjustment = region.file_offset - page_offset;
  void* map_start = static_cast<char*>(region.data) - offset_adjustment;
  size_t map_size = align_up_to_page(region.size + offset_adjustment);

  return mlock(map_start, map_size) == 0;
}

bool MMapWeightLoader::unlock_memory(const MappedRegion& region) {
  if (!region.is_valid) {
    return false;
  }

  // Align region
  size_t page_offset = align_down_to_page(region.file_offset);
  size_t offset_adjustment = region.file_offset - page_offset;
  void* map_start = static_cast<char*>(region.data) - offset_adjustment;
  size_t map_size = align_up_to_page(region.size + offset_adjustment);

  return munlock(map_start, map_size) == 0;
}

MMapWeightLoader::Stats MMapWeightLoader::get_stats() const {
  Stats stats;
  stats.total_file_size = file_size_;
  stats.total_mapped_bytes = total_mapped_bytes_;
  stats.num_active_mappings = active_mappings_.size();
  stats.num_registered_tensors = tensors_.size();
  stats.page_size = page_size_;
  return stats;
}

// Private methods

size_t MMapWeightLoader::align_down_to_page(size_t offset) const {
  return (offset / page_size_) * page_size_;
}

size_t MMapWeightLoader::align_up_to_page(size_t size) const {
  return ((size + page_size_ - 1) / page_size_) * page_size_;
}

size_t MMapWeightLoader::get_page_size() {
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

void MMapWeightLoader::close_file() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

void MMapWeightLoader::unmap_all() {
  // Unmap full mapping if present
  if (full_mapping_.is_valid && full_mapping_.data != nullptr) {
    munmap(full_mapping_.data, full_mapping_.size);
    total_mapped_bytes_ -= full_mapping_.size;
    full_mapping_ = MappedRegion();
  }

  // Unmap all active mappings
  for (const auto& region : active_mappings_) {
    if (region.is_valid && region.data != nullptr) {
      munmap(region.data, region.size);
    }
  }

  active_mappings_.clear();
  total_mapped_bytes_ = 0;
}

}  // namespace mlxr
