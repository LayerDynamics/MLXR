// Configuration validation utility
// Validates server.yaml and model config files

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

bool validate_server_config(const std::string& path) {
    std::cout << "Validating server config: " << path << std::endl;

    try {
        YAML::Node config = YAML::LoadFile(path);

        // Check required sections
        if (!config["server"]) {
            std::cerr << "  ERROR: Missing 'server' section" << std::endl;
            return false;
        }

        if (!config["scheduler"]) {
            std::cerr << "  ERROR: Missing 'scheduler' section" << std::endl;
            return false;
        }

        if (!config["kv_cache"]) {
            std::cerr << "  ERROR: Missing 'kv_cache' section" << std::endl;
            return false;
        }

        // Validate server section
        auto server = config["server"];
        if (!server["uds_path"]) {
            std::cerr << "  WARNING: Missing server.uds_path" << std::endl;
        }

        // Validate scheduler section
        auto scheduler = config["scheduler"];
        if (!scheduler["max_batch_tokens"]) {
            std::cerr << "  WARNING: Missing scheduler.max_batch_tokens" << std::endl;
        }
        if (!scheduler["max_batch_size"]) {
            std::cerr << "  WARNING: Missing scheduler.max_batch_size" << std::endl;
        }

        // Validate gRPC section if present
        if (config["grpc"]) {
            auto grpc = config["grpc"];
            if (grpc["enabled"] && grpc["enabled"].as<bool>()) {
                if (!grpc["port"]) {
                    std::cerr << "  WARNING: gRPC enabled but no port specified" << std::endl;
                }
            }
        }

        std::cout << "  ✓ Server config is valid" << std::endl;
        return true;

    } catch (const YAML::Exception& e) {
        std::cerr << "  ERROR: YAML parse error: " << e.what() << std::endl;
        return false;
    }
}

bool validate_model_config(const std::string& path) {
    std::cout << "Validating model config: " << path << std::endl;

    try {
        YAML::Node config = YAML::LoadFile(path);

        // Check required sections
        if (!config["model"]) {
            std::cerr << "  ERROR: Missing 'model' section" << std::endl;
            return false;
        }

        auto model = config["model"];

        // Required fields
        if (!model["name"]) {
            std::cerr << "  ERROR: Missing model.name" << std::endl;
            return false;
        }
        if (!model["family"]) {
            std::cerr << "  ERROR: Missing model.family" << std::endl;
            return false;
        }
        if (!model["path"]) {
            std::cerr << "  ERROR: Missing model.path" << std::endl;
            return false;
        }
        if (!model["format"]) {
            std::cerr << "  ERROR: Missing model.format" << std::endl;
            return false;
        }

        // Validate format
        std::string format = model["format"].as<std::string>();
        if (format != "gguf" && format != "safetensors" && format != "mlx") {
            std::cerr << "  WARNING: Unknown format: " << format << std::endl;
        }

        // Check architecture section
        if (!config["architecture"]) {
            std::cerr << "  WARNING: Missing 'architecture' section" << std::endl;
        } else {
            auto arch = config["architecture"];
            if (!arch["vocab_size"]) {
                std::cerr << "  WARNING: Missing architecture.vocab_size" << std::endl;
            }
            if (!arch["hidden_size"]) {
                std::cerr << "  WARNING: Missing architecture.hidden_size" << std::endl;
            }
            if (!arch["num_hidden_layers"]) {
                std::cerr << "  WARNING: Missing architecture.num_hidden_layers" << std::endl;
            }
        }

        // Check tokenizer section
        if (!config["tokenizer"]) {
            std::cerr << "  WARNING: Missing 'tokenizer' section" << std::endl;
        }

        std::cout << "  ✓ Model config is valid" << std::endl;
        return true;

    } catch (const YAML::Exception& e) {
        std::cerr << "  ERROR: YAML parse error: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "MLXR Configuration Validator\n" << std::endl;

    std::string config_dir = "configs";
    if (argc > 1) {
        config_dir = argv[1];
    }

    int errors = 0;

    // Validate server config
    std::string server_config = config_dir + "/server.yaml";
    if (fs::exists(server_config)) {
        if (!validate_server_config(server_config)) {
            errors++;
        }
    } else {
        std::cerr << "ERROR: Server config not found: " << server_config << std::endl;
        errors++;
    }

    std::cout << std::endl;

    // Validate model configs
    std::string models_dir = config_dir + "/models";
    if (fs::exists(models_dir) && fs::is_directory(models_dir)) {
        for (const auto& entry : fs::directory_iterator(models_dir)) {
            if (entry.path().extension() == ".yaml") {
                if (!validate_model_config(entry.path().string())) {
                    errors++;
                }
                std::cout << std::endl;
            }
        }
    } else {
        std::cerr << "WARNING: Models directory not found: " << models_dir << std::endl;
    }

    if (errors == 0) {
        std::cout << "✓ All configurations are valid!" << std::endl;
        return 0;
    } else {
        std::cerr << "✗ Found " << errors << " configuration error(s)" << std::endl;
        return 1;
    }
}
