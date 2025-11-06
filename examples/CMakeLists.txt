# Examples

# Simple generation example
add_executable(simple_generation simple_generation.cpp)

target_link_libraries(simple_generation PRIVATE
    mlxr_core
)

# Set C++17 standard
set_property(TARGET simple_generation PROPERTY CXX_STANDARD 17)
set_property(TARGET simple_generation PROPERTY CXX_STANDARD_REQUIRED ON)

# Install to build/bin
install(TARGETS simple_generation DESTINATION bin)

# KV cache test example
add_executable(kv_cache_test kv_cache_test.cpp)

target_link_libraries(kv_cache_test PRIVATE
    mlxr_core
)

# Set C++17 standard
set_property(TARGET kv_cache_test PROPERTY CXX_STANDARD 17)
set_property(TARGET kv_cache_test PROPERTY CXX_STANDARD_REQUIRED ON)

# Install to build/bin
install(TARGETS kv_cache_test DESTINATION bin)

# Metal kernel test (validates Metal attention kernels)
add_executable(metal_kernel_test metal_kernel_test.cpp)

target_link_libraries(metal_kernel_test PRIVATE
    mlxr_core
)

# Set C++17 standard
set_property(TARGET metal_kernel_test PROPERTY CXX_STANDARD 17)
set_property(TARGET metal_kernel_test PROPERTY CXX_STANDARD_REQUIRED ON)

# Install to build/bin
install(TARGETS metal_kernel_test DESTINATION bin)

# Cached model test (validates zero-copy optimization with CachedLlamaModel)
add_executable(cached_model_test cached_model_test.cpp)

target_link_libraries(cached_model_test PRIVATE
    mlxr_core
)

# Set C++17 standard
set_property(TARGET cached_model_test PROPERTY CXX_STANDARD 17)
set_property(TARGET cached_model_test PROPERTY CXX_STANDARD_REQUIRED ON)

# Install to build/bin
install(TARGETS cached_model_test DESTINATION bin)

message(STATUS "Examples configured")
