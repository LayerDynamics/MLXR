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

# Find required frameworks
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Try to find MLX (will provide more robust detection later)
# For now, assume MLX is installed via pip and accessible
set(MLX_INCLUDE_DIR "/opt/homebrew/lib/python3.11/site-packages/mlx/include" CACHE PATH "MLX include directory")
set(MLX_LIBRARY_DIR "/opt/homebrew/lib/python3.11/site-packages/mlx/lib" CACHE PATH "MLX library directory")

# Check if MLX paths exist
if(EXISTS ${MLX_INCLUDE_DIR})
    message(STATUS "Found MLX include directory: ${MLX_INCLUDE_DIR}")
    include_directories(${MLX_INCLUDE_DIR})
else()
    message(WARNING "MLX include directory not found at ${MLX_INCLUDE_DIR}")
    message(WARNING "You may need to set MLX_INCLUDE_DIR manually or install MLX via: pip install mlx")
endif()

if(EXISTS ${MLX_LIBRARY_DIR})
    message(STATUS "Found MLX library directory: ${MLX_LIBRARY_DIR}")
    link_directories(${MLX_LIBRARY_DIR})
else()
    message(WARNING "MLX library directory not found at ${MLX_LIBRARY_DIR}")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/core)
include_directories(${CMAKE_SOURCE_DIR}/daemon)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_DAEMON "Build daemon executable" ON)
option(BUILD_TOOLS "Build conversion and utility tools" ON)
option(ENABLE_METAL_VALIDATION "Enable Metal API validation (slower, for debugging)" OFF)

# Subdirectories
add_subdirectory(core)

if(BUILD_DAEMON)
    add_subdirectory(daemon)
endif()

if(BUILD_TOOLS)
    add_subdirectory(tools)
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
message(STATUS "  Build tests:          ${BUILD_TESTS}")
message(STATUS "  Build daemon:         ${BUILD_DAEMON}")
message(STATUS "  Build tools:          ${BUILD_TOOLS}")
message(STATUS "  Metal validation:     ${ENABLE_METAL_VALIDATION}")
message(STATUS "")
