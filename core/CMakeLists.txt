# Core inference engine library

set(CORE_SOURCES
    # Graph components (MLX integration)
    # graph/layers.cpp
    # graph/attention.cpp
    # graph/mlp.cpp

    # Runtime components
    # runtime/engine.cpp
    # runtime/memory.cpp

    # Tokenizer
    # runtime/tokenizer/sentencepiece.cpp
    # runtime/tokenizer/hf_tokenizers.cpp

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

# For now, create a minimal placeholder library
# We'll populate this as we implement Phase 1

# Create core library (will be static for now)
# add_library(mlxr_core STATIC ${CORE_SOURCES})

# target_include_directories(mlxr_core PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/graph
#     ${CMAKE_CURRENT_SOURCE_DIR}/runtime
# )

# target_link_libraries(mlxr_core PUBLIC
#     ${FOUNDATION_FRAMEWORK}
#     ${METAL_FRAMEWORK}
#     ${METALKIT_FRAMEWORK}
#     ${ACCELERATE_FRAMEWORK}
# )

# For Phase 0, just create subdirectory structure markers
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/graph/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/.gitkeep "")

message(STATUS "Core library structure created (implementation pending)")
