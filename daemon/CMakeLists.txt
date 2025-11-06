# Daemon process (background server)

# Daemon will include:
# - REST/gRPC server
# - Request scheduler
# - Model registry (GGUF parser, model catalog)
# - Telemetry

# Find gRPC and Protobuf
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Generate protobuf and gRPC files
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/server/proto")
set(PROTO_FILE "${PROTO_PATH}/mlxrunner.proto")
set(GENERATED_PROTOBUF_PATH "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})

# Generated files
set(PROTO_SRCS "${GENERATED_PROTOBUF_PATH}/mlxrunner.pb.cc")
set(PROTO_HDRS "${GENERATED_PROTOBUF_PATH}/mlxrunner.pb.h")
set(GRPC_SRCS "${GENERATED_PROTOBUF_PATH}/mlxrunner.grpc.pb.cc")
set(GRPC_HDRS "${GENERATED_PROTOBUF_PATH}/mlxrunner.grpc.pb.h")

# Find protoc and grpc_cpp_plugin
find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# Custom command to generate protobuf files
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --proto_path=${PROTO_PATH}
         --cpp_out=${GENERATED_PROTOBUF_PATH}
         --grpc_out=${GENERATED_PROTOBUF_PATH}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf and gRPC sources from ${PROTO_FILE}"
    VERBATIM
)

set(DAEMON_SOURCES
    # Registry components
    registry/gguf_parser.cpp
    registry/model_registry.cpp

    # Scheduler components
    scheduler/scheduler.cpp

    # Server components
    server/rest_server.cpp
    server/sse_stream.cpp
    server/ollama_api.cpp
    server/scheduler_worker.cpp
    server/grpc_server.cpp

    # Telemetry components
    telemetry/metrics.cpp

    # Generated protobuf sources
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# Create daemon library (will be linked into mlxrunnerd binary)
add_library(mlxr_daemon STATIC ${DAEMON_SOURCES})

target_include_directories(mlxr_daemon PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler
    ${CMAKE_CURRENT_SOURCE_DIR}/server
    ${CMAKE_CURRENT_SOURCE_DIR}/telemetry
    ${GENERATED_PROTOBUF_PATH}  # Include generated protobuf headers
    ${Protobuf_INCLUDE_DIRS}
)

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)

# Find cpp-httplib (header-only HTTP server library)
find_path(HTTPLIB_INCLUDE_DIR NAMES httplib.h
    PATHS
    /opt/homebrew/opt/cpp-httplib/include
    /usr/local/include
    /usr/include
)

if(NOT HTTPLIB_INCLUDE_DIR)
    message(FATAL_ERROR "cpp-httplib not found. Install with: brew install cpp-httplib")
endif()

message(STATUS "Found cpp-httplib: ${HTTPLIB_INCLUDE_DIR}")

# Find OpenSSL (required for CPPHTTPLIB_OPENSSL_SUPPORT)
find_package(OpenSSL REQUIRED)
message(STATUS "Found OpenSSL: ${OPENSSL_VERSION}")

# Link dependencies
target_link_libraries(mlxr_daemon PUBLIC
    mlxr_core
    SQLite::SQLite3
    nlohmann_json::nlohmann_json
    OpenSSL::SSL       # Required by CPPHTTPLIB_OPENSSL_SUPPORT
    OpenSSL::Crypto    # Required by CPPHTTPLIB_OPENSSL_SUPPORT
    pthread            # Required by cpp-httplib ThreadPool
    gRPC::grpc++       # gRPC C++ library
    gRPC::grpc++_reflection  # gRPC reflection (optional)
    protobuf::libprotobuf    # Protobuf library
)

# Add cpp-httplib include directory
target_include_directories(mlxr_daemon PUBLIC ${HTTPLIB_INCLUDE_DIR})

# Set C++17 standard
set_property(TARGET mlxr_daemon PROPERTY CXX_STANDARD 17)
set_property(TARGET mlxr_daemon PROPERTY CXX_STANDARD_REQUIRED ON)

# Create subdirectory markers for future components
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/server/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/.gitkeep "")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/telemetry/.gitkeep "")

# Test daemon executable (for testing HTTP server)
add_executable(test_daemon test_daemon_main.cpp)
target_link_libraries(test_daemon PRIVATE mlxr_daemon)
set_target_properties(test_daemon PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "Daemon library created with GGUF parser")
message(STATUS "gRPC and Protobuf configured")
message(STATUS "Generated protobuf path: ${GENERATED_PROTOBUF_PATH}")
message(STATUS "Test daemon executable configured")
