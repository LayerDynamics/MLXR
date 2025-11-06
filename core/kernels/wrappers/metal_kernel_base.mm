/**
 * @file metal_kernel_base.mm
 * @brief Implementation of Metal kernel base class
 *
 * Note: Using .mm extension for Objective-C++ to work with Metal API
 */

#include "metal_kernel_base.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

namespace mlxr {
namespace kernels {

// Internal state (singleton pattern)
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static std::unordered_map<std::string, id<MTLLibrary>> g_library_cache;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipeline_cache;

void MetalKernelBase::initialize() {
  if (g_device != nil) {
    return;  // Already initialized
  }

  // Get the default Metal device
  g_device = MTLCreateSystemDefaultDevice();
  if (g_device == nil) {
    throw std::runtime_error("Failed to create Metal device");
  }

  // Create command queue
  g_command_queue = [g_device newCommandQueue];
  if (g_command_queue == nil) {
    throw std::runtime_error("Failed to create Metal command queue");
  }

  std::cout << "Metal device initialized: "
            << [[g_device name] UTF8String] << std::endl;
}

MetalDeviceImpl* MetalKernelBase::device() {
  initialize();
  return reinterpret_cast<MetalDeviceImpl*>((__bridge void*)g_device);
}

void* MetalKernelBase::command_queue() {
  initialize();
  return (__bridge void*)g_command_queue;
}

void* MetalKernelBase::load_library(const std::string& library_path) {
  // Check cache first
  auto it = g_library_cache.find(library_path);
  if (it != g_library_cache.end()) {
    return (__bridge void*)it->second;
  }

  initialize();

  // Convert path to NSString
  NSString* path = [NSString stringWithUTF8String:library_path.c_str()];

  // Load library from file (use URL-based API to avoid deprecation)
  NSURL* url = [NSURL fileURLWithPath:path];
  NSError* error = nil;
  id<MTLLibrary> library = [g_device newLibraryWithURL:url error:&error];

  if (library == nil) {
    std::string error_msg = "Failed to load Metal library: " + library_path;
    if (error != nil) {
      error_msg += " - ";
      error_msg += [[error localizedDescription] UTF8String];
    }
    throw std::runtime_error(error_msg);
  }

  // Cache the library
  g_library_cache[library_path] = library;

  std::cout << "Loaded Metal library: " << library_path << std::endl;

  return (__bridge void*)library;
}

void* MetalKernelBase::get_pipeline(
    void* library_ptr,
    const std::string& function_name) {
  // Create cache key
  std::string cache_key = std::to_string(reinterpret_cast<uintptr_t>(library_ptr)) +
                          ":" + function_name;

  // Check cache
  auto it = g_pipeline_cache.find(cache_key);
  if (it != g_pipeline_cache.end()) {
    return (__bridge void*)it->second;
  }

  initialize();

  // Cast back to Metal library
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_ptr;

  // Get kernel function
  NSString* name = [NSString stringWithUTF8String:function_name.c_str()];
  id<MTLFunction> function = [library newFunctionWithName:name];

  if (function == nil) {
    throw std::runtime_error("Failed to find kernel function: " + function_name);
  }

  // Create compute pipeline state
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [g_device newComputePipelineStateWithFunction:function error:&error];

  if (pipeline == nil) {
    std::string error_msg = "Failed to create pipeline for function: " + function_name;
    if (error != nil) {
      error_msg += " - ";
      error_msg += [[error localizedDescription] UTF8String];
    }
    throw std::runtime_error(error_msg);
  }

  // Cache the pipeline
  g_pipeline_cache[cache_key] = pipeline;

  std::cout << "Created pipeline for kernel: " << function_name << std::endl;

  return (__bridge void*)pipeline;
}

ThreadgroupSize MetalKernelBase::calculate_threadgroup_size(
    void* pipeline_ptr,
    size_t total_threads) {
  id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;

  // Get maximum threads per threadgroup
  NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];

  // Calculate optimal size (power of 2, up to max)
  size_t size = 1;
  while (size * 2 <= max_threads && size * 2 <= total_threads) {
    size *= 2;
  }

  // Ensure at least 32 threads (one warp/simdgroup)
  if (size < 32) {
    size = std::min<size_t>(32, max_threads);
  }

  ThreadgroupSize result;
  result.width = size;
  result.height = 1;
  result.depth = 1;

  return result;
}

ThreadgroupSize MetalKernelBase::calculate_grid_size(
    size_t total_threads,
    ThreadgroupSize threadgroup_size) {
  size_t threadgroup_threads = threadgroup_size.width *
                                threadgroup_size.height *
                                threadgroup_size.depth;

  size_t num_threadgroups = (total_threads + threadgroup_threads - 1) / threadgroup_threads;

  ThreadgroupSize result;
  result.width = num_threadgroups;
  result.height = 1;
  result.depth = 1;

  return result;
}

}  // namespace kernels
}  // namespace mlxr
