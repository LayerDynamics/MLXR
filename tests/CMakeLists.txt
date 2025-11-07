# Test suite

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Ensure we can find core and daemon headers
include_directories(${CMAKE_SOURCE_DIR}/core)
include_directories(${CMAKE_SOURCE_DIR}/daemon)

# Collect all unit test source files
set(UNIT_TEST_SOURCES
    unit/tensor_test.cpp
    unit/layers_test.cpp
    unit/tokenizer_test.cpp
    unit/rmsnorm_primitive_test.cpp
    unit/gguf_parser_test.cpp
    unit/model_registry_test.cpp
    unit/speculative_decoder_test.cpp
    unit/mmap_loader_test.cpp
    unit/rest_server_test.cpp
    unit/sse_stream_test.cpp
    unit/ollama_api_test.cpp
    unit/metrics_test.cpp
    unit/scheduler_test.cpp
    unit/scheduler_worker_test.cpp
    unit/test_model_loader_weights.cpp
    unit/test_model_loader_gguf.cpp
    unit/test_model_loader_pager.cpp
    unit/test_model_loader_integration.cpp
)

# Create unit test executable
add_executable(mlxr_unit_tests ${UNIT_TEST_SOURCES})

# Link against Google Test, mlxr_core, mlxr_daemon, and required frameworks
target_link_libraries(mlxr_unit_tests
    PRIVATE
    mlxr_core
    mlxr_daemon
    GTest::gtest
    GTest::gtest_main
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

# Set C++17 standard
set_property(TARGET mlxr_unit_tests PROPERTY CXX_STANDARD 17)
set_property(TARGET mlxr_unit_tests PROPERTY CXX_STANDARD_REQUIRED ON)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(mlxr_unit_tests)

message(STATUS "Unit tests configured with Google Test")
