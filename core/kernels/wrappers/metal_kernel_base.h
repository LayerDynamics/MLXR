/**
 * @file metal_kernel_base.h
 * @brief Base class for Metal kernel wrappers
 *
 * Provides infrastructure for loading and executing custom Metal kernels
 * with MLX tensor integration.
 */

#pragma once

#include <memory>
#include <string>

namespace mlxr {
namespace kernels {

// Forward declarations to avoid exposing Metal types in header
struct MetalDeviceImpl;
struct MetalPipelineImpl;
struct ThreadgroupSize;

/**
 * @brief Thread configuration structure
 */
struct ThreadgroupSize {
  unsigned long width;
  unsigned long height;
  unsigned long depth;
};

/**
 * @brief Base class for Metal kernel management
 *
 * Handles Metal device initialization, library loading, and
 * provides utilities for kernel execution with MLX tensors.
 * Uses PIMPL pattern to hide Metal-specific types from C++ headers.
 */
class MetalKernelBase {
 public:
  /**
   * @brief Get the shared Metal device implementation
   * @return Opaque pointer to Metal device
   */
  static MetalDeviceImpl* device();

  /**
   * @brief Get the shared Metal command queue
   * @return Opaque pointer to Metal command queue
   */
  static void* command_queue();

  /**
   * @brief Load a Metal library from file
   * @param library_path Path to .metallib file
   * @return Opaque pointer to Metal library
   */
  static void* load_library(const std::string& library_path);

  /**
   * @brief Get a compute pipeline for a kernel function
   * @param library Opaque pointer to Metal library
   * @param function_name Name of the kernel function
   * @return Opaque pointer to compute pipeline state
   */
  static void* get_pipeline(void* library, const std::string& function_name);

  /**
   * @brief Calculate optimal threadgroup size for a kernel
   * @param pipeline Opaque pointer to compute pipeline state
   * @param total_threads Total number of threads needed
   * @return Threadgroup size
   */
  static ThreadgroupSize calculate_threadgroup_size(void* pipeline,
                                                    size_t total_threads);

  /**
   * @brief Calculate grid size from total threads and threadgroup size
   * @param total_threads Total number of threads needed
   * @param threadgroup_size Threadgroup size
   * @return Grid size
   */
  static ThreadgroupSize calculate_grid_size(size_t total_threads,
                                             ThreadgroupSize threadgroup_size);

 protected:
  MetalKernelBase() = default;
  virtual ~MetalKernelBase() = default;

 private:
  // Initialize Metal device (called lazily)
  static void initialize();
};

}  // namespace kernels
}  // namespace mlxr
