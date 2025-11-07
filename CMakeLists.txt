cmake_minimum_required(VERSION 3.20)
project(MLXR VERSION 0.1.0 LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform checks
if(NOT APPLE)
    message(FATAL_ERROR "MLXR is designed for macOS/Apple Silicon only")
endif()

# Check for Apple Silicon
execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT ARCH STREQUAL "arm64")
    message(WARNING "MLXR is optimized for Apple Silicon (arm64), detected: ${ARCH}")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Find required frameworks
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Find Python (needed for MLX detection)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find MLX framework
find_package(MLX REQUIRED)

# Find SentencePiece for tokenization
find_package(SentencePiece REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/core)
include_directories(${CMAKE_SOURCE_DIR}/daemon)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_DAEMON "Build daemon executable" ON)
option(BUILD_TOOLS "Build conversion and utility tools" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_CUSTOM_KERNELS "Enable custom Metal kernels for attention/RoPE/MLP" ON)
option(ENABLE_METAL_VALIDATION "Enable Metal API validation (slower, for debugging)" OFF)

# Add USE_CUSTOM_KERNELS definition if enabled
if(USE_CUSTOM_KERNELS)
    add_compile_definitions(USE_CUSTOM_KERNELS)
    message(STATUS "Custom Metal kernels: ENABLED")
else()
    message(STATUS "Custom Metal kernels: DISABLED (using MLX fallback)")
endif()

# Subdirectories
add_subdirectory(core)

if(BUILD_DAEMON)
    add_subdirectory(daemon)
endif()

if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install targets
install(DIRECTORY ${CMAKE_SOURCE_DIR}/configs/
        DESTINATION share/mlxr/configs
        FILES_MATCHING PATTERN "*.yaml")

# Print configuration summary
message(STATUS "")
message(STATUS "MLXR Configuration Summary:")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  Architecture:         ${ARCH}")
message(STATUS "  C++ Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:               ${Python3_EXECUTABLE}")
message(STATUS "  MLX version:          ${MLX_VERSION}")
message(STATUS "  Build tests:          ${BUILD_TESTS}")
message(STATUS "  Build daemon:         ${BUILD_DAEMON}")
message(STATUS "  Build tools:          ${BUILD_TOOLS}")
message(STATUS "  Build examples:       ${BUILD_EXAMPLES}")
message(STATUS "  Custom Metal kernels: ${USE_CUSTOM_KERNELS}")
message(STATUS "  Metal validation:     ${ENABLE_METAL_VALIDATION}")
message(STATUS "")
