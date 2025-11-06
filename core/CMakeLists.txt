# Core inference engine library

# Option to enable custom Metal kernels for performance
option(USE_CUSTOM_KERNELS "Enable custom Metal kernels for optimized performance" ON)

set(CORE_SOURCES
    # Graph components (MLX integration)
    graph/tensor.cpp
    graph/layers.cpp
    graph/model.cpp
    graph/attention_cached.cpp

    # Runtime components
    runtime/engine.cpp
    runtime/mmap_loader.cpp

    # Tokenizer
    runtime/tokenizer/tokenizer.cpp

    # Sampler
    runtime/sampler.cpp

    # KV cache
    runtime/kv/arena.cpp
    runtime/kv/pager.cpp
    runtime/kv/eviction.cpp

    # Speculative decoding
    runtime/spec/speculative_decoder.cpp
)

# Add custom Metal kernel primitives if enabled
if(USE_CUSTOM_KERNELS)
    list(APPEND CORE_SOURCES
        # MLX Primitive-based custom kernels
        kernels/primitives/rmsnorm_primitive.mm
        kernels/primitives/attention_decode_primitive.mm
        kernels/primitives/attention_prefill_primitive.mm
        kernels/primitives/q_gemm_dequant_primitive.mm
        kernels/primitives/rope_apply_primitive.mm
        kernels/primitives/swiglu_mlp_primitive.mm

        # Legacy wrappers (deprecated, kept for reference)
        # kernels/wrappers/metal_kernel_base.mm
        # kernels/wrappers/rmsnorm_kernel.mm
    )
    message(STATUS "Custom Metal kernels enabled (Primitive-based)")
endif()

# Metal kernel library will be built separately via scripts/build_metal.sh
# and linked as a resource

# Create core library (will be static for now)
add_library(mlxr_core STATIC ${CORE_SOURCES})

target_include_directories(mlxr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/graph
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# Add metal-cpp headers for custom Metal kernel primitives
if(USE_CUSTOM_KERNELS)
    # MLX bundles metal-cpp headers alongside its own headers
    get_target_property(MLX_INCLUDE_DIRS MLX::mlx INTERFACE_INCLUDE_DIRECTORIES)
    foreach(MLX_INCLUDE_DIR ${MLX_INCLUDE_DIRS})
        if(EXISTS "${MLX_INCLUDE_DIR}/metal_cpp")
            target_include_directories(mlxr_core PRIVATE "${MLX_INCLUDE_DIR}/metal_cpp")
            message(STATUS "Added metal-cpp include path: ${MLX_INCLUDE_DIR}/metal_cpp")
        endif()
    endforeach()
endif()

# Add compile definition if custom kernels are enabled
if(USE_CUSTOM_KERNELS)
    target_compile_definitions(mlxr_core PRIVATE USE_CUSTOM_KERNELS)
endif()

target_link_libraries(mlxr_core PUBLIC
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
    MLX::mlx
    SentencePiece::sentencepiece
)

# Set C++17 standard
set_property(TARGET mlxr_core PROPERTY CXX_STANDARD 17)
set_property(TARGET mlxr_core PROPERTY CXX_STANDARD_REQUIRED ON)

# Create subdirectory markers for future components
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/.gitkeep "")

message(STATUS "Core library created with MLX tensor wrapper")
