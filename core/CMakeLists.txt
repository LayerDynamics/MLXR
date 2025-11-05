# Core inference engine library

set(CORE_SOURCES
    # Graph components (MLX integration)
    graph/tensor.cpp
    graph/layers.cpp

    # Runtime components
    # runtime/engine.cpp
    # runtime/memory.cpp

    # Tokenizer
    runtime/tokenizer.cpp

    # KV cache
    # runtime/kv/arena.cpp
    # runtime/kv/pager.cpp
    # runtime/kv/eviction.cpp

    # Speculative decoding
    # runtime/spec/draft_model.cpp
    # runtime/spec/verifier.cpp
)

# Metal kernel library will be built separately via scripts/build_metal.sh
# and linked as a resource

# Create core library (will be static for now)
add_library(mlxr_core STATIC ${CORE_SOURCES})

target_include_directories(mlxr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/graph
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

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
