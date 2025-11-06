// Copyright © 2025 MLXR Development
// Model registry implementation

#include "model_registry.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace mlxr {
namespace registry {

namespace {

// Get current Unix timestamp
int64_t current_timestamp() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// SQL schema for model registry
const char* SCHEMA_SQL = R"(
-- Models table
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  model_id TEXT UNIQUE NOT NULL,
  architecture TEXT NOT NULL,
  file_path TEXT NOT NULL,
  format TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  sha256 TEXT,
  param_count INTEGER,
  context_length INTEGER,
  hidden_size INTEGER,
  num_layers INTEGER,
  num_heads INTEGER,
  num_kv_heads INTEGER,
  intermediate_size INTEGER,
  vocab_size INTEGER,
  quant_type TEXT,
  quant_details TEXT,
  tokenizer_type TEXT,
  tokenizer_path TEXT,
  rope_freq_base REAL,
  rope_scale REAL,
  rope_scaling_type TEXT,
  description TEXT,
  license TEXT,
  source_url TEXT,
  is_loaded INTEGER DEFAULT 0,
  last_used_timestamp INTEGER,
  created_timestamp INTEGER,
  chat_template TEXT
);

CREATE INDEX IF NOT EXISTS idx_models_model_id ON models(model_id);
CREATE INDEX IF NOT EXISTS idx_models_architecture ON models(architecture);
CREATE INDEX IF NOT EXISTS idx_models_last_used ON models(last_used_timestamp DESC);

-- Model tags table
CREATE TABLE IF NOT EXISTS model_tags (
  model_id INTEGER NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY (model_id, key),
  FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_key_value ON model_tags(key, value);

-- Adapters table
CREATE TABLE IF NOT EXISTS adapters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  base_model_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  adapter_id TEXT UNIQUE NOT NULL,
  file_path TEXT NOT NULL,
  adapter_type TEXT NOT NULL,
  rank INTEGER,
  scale REAL,
  target_modules TEXT,
  created_timestamp INTEGER,
  FOREIGN KEY (base_model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_adapters_base_model ON adapters(base_model_id);
)";

}  // anonymous namespace

ModelRegistry::ModelRegistry(const std::string& db_path, bool create_if_missing)
    : db_(nullptr),
      db_path_(db_path),
      stmt_insert_model_(nullptr),
      stmt_update_model_(nullptr),
      stmt_get_model_(nullptr),
      stmt_touch_model_(nullptr) {
  (void)create_if_missing;  // Reserved for future use
}

ModelRegistry::~ModelRegistry() { close(); }

bool ModelRegistry::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Always allow creation if database doesn't exist
  int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;

  int rc = sqlite3_open_v2(db_path_.c_str(), &db_, flags, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "Failed to open database: " << sqlite3_errmsg(db_)
              << std::endl;
    return false;
  }

  // Enable foreign keys
  sqlite3_exec(db_, "PRAGMA foreign_keys = ON;", nullptr, nullptr, nullptr);

  // Create schema (idempotent - uses CREATE TABLE IF NOT EXISTS)
  if (!create_schema()) {
    return false;
  }

  // Prepare statements
  if (!prepare_statements()) {
    return false;
  }

  return true;
}

bool ModelRegistry::create_schema() {
  char* err_msg = nullptr;
  int rc = sqlite3_exec(db_, SCHEMA_SQL, nullptr, nullptr, &err_msg);

  if (rc != SQLITE_OK) {
    std::cerr << "Failed to create schema: " << err_msg << std::endl;
    sqlite3_free(err_msg);
    return false;
  }

  return true;
}

bool ModelRegistry::prepare_statements() {
  // Insert model statement
  const char* insert_sql = R"(
    INSERT INTO models (
      name, model_id, architecture, file_path, format, file_size, sha256,
      param_count, context_length, hidden_size, num_layers, num_heads, num_kv_heads,
      intermediate_size, vocab_size, quant_type, quant_details, tokenizer_type,
      tokenizer_path, rope_freq_base, rope_scale, rope_scaling_type, description,
      license, source_url, is_loaded, last_used_timestamp, created_timestamp, chat_template
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  )";

  int rc =
      sqlite3_prepare_v2(db_, insert_sql, -1, &stmt_insert_model_, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  // Get model statement
  const char* get_sql = "SELECT * FROM models WHERE id = ?";
  rc = sqlite3_prepare_v2(db_, get_sql, -1, &stmt_get_model_, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  // Touch model statement
  const char* touch_sql =
      "UPDATE models SET last_used_timestamp = ? WHERE id = ?";
  rc = sqlite3_prepare_v2(db_, touch_sql, -1, &stmt_touch_model_, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  return true;
}

void ModelRegistry::finalize_statements() {
  if (stmt_insert_model_) sqlite3_finalize(stmt_insert_model_);
  if (stmt_update_model_) sqlite3_finalize(stmt_update_model_);
  if (stmt_get_model_) sqlite3_finalize(stmt_get_model_);
  if (stmt_touch_model_) sqlite3_finalize(stmt_touch_model_);

  stmt_insert_model_ = nullptr;
  stmt_update_model_ = nullptr;
  stmt_get_model_ = nullptr;
  stmt_touch_model_ = nullptr;
}

int64_t ModelRegistry::register_model(const ModelInfo& info) {
  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(stmt_insert_model_);

  int idx = 1;
  sqlite3_bind_text(stmt_insert_model_, idx++, info.name.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.model_id.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++,
                    architecture_to_string(info.architecture).c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.file_path.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++,
                    format_to_string(info.format).c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt_insert_model_, idx++, info.file_size);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.sha256.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt_insert_model_, idx++, info.param_count);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.context_length);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.hidden_size);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.num_layers);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.num_heads);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.num_kv_heads);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.intermediate_size);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.vocab_size);
  sqlite3_bind_text(stmt_insert_model_, idx++,
                    quant_type_to_string(info.quant_type).c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.quant_details.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.tokenizer_type.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.tokenizer_path.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt_insert_model_, idx++, info.rope_freq_base);
  sqlite3_bind_double(stmt_insert_model_, idx++, info.rope_scale);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.rope_scaling_type.c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.description.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.license.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt_insert_model_, idx++, info.source_url.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt_insert_model_, idx++, info.is_loaded ? 1 : 0);
  sqlite3_bind_int64(stmt_insert_model_, idx++, current_timestamp());
  sqlite3_bind_int64(stmt_insert_model_, idx++, current_timestamp());
  sqlite3_bind_text(stmt_insert_model_, idx++, info.chat_template.c_str(), -1,
                    SQLITE_TRANSIENT);

  int rc = sqlite3_step(stmt_insert_model_);
  if (rc != SQLITE_DONE) {
    std::cerr << "Failed to insert model: " << sqlite3_errmsg(db_) << std::endl;
    return -1;
  }

  int64_t model_id = sqlite3_last_insert_rowid(db_);

  // Insert tags
  if (!info.tags.empty()) {
    std::unordered_map<std::string, std::string> tags;
    for (size_t i = 0; i < info.tags.size(); i++) {
      tags["tag_" + std::to_string(i)] = info.tags[i];
    }
    add_tags(model_id, tags);
  }

  return model_id;
}

bool ModelRegistry::update_model(const ModelInfo& info) {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* update_sql = R"(
    UPDATE models SET
      name = ?, model_id = ?, architecture = ?, file_path = ?, format = ?,
      file_size = ?, sha256 = ?, param_count = ?, context_length = ?,
      hidden_size = ?, num_layers = ?, num_heads = ?, num_kv_heads = ?,
      intermediate_size = ?, vocab_size = ?, quant_type = ?, quant_details = ?,
      tokenizer_type = ?, tokenizer_path = ?, rope_freq_base = ?, rope_scale = ?,
      rope_scaling_type = ?, description = ?, license = ?, source_url = ?,
      chat_template = ?
    WHERE id = ?
  )";

  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(db_, update_sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  int idx = 1;
  sqlite3_bind_text(stmt, idx++, info.name.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.model_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++,
                    architecture_to_string(info.architecture).c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.file_path.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, format_to_string(info.format).c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, idx++, info.file_size);
  sqlite3_bind_text(stmt, idx++, info.sha256.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, idx++, info.param_count);
  sqlite3_bind_int(stmt, idx++, info.context_length);
  sqlite3_bind_int(stmt, idx++, info.hidden_size);
  sqlite3_bind_int(stmt, idx++, info.num_layers);
  sqlite3_bind_int(stmt, idx++, info.num_heads);
  sqlite3_bind_int(stmt, idx++, info.num_kv_heads);
  sqlite3_bind_int(stmt, idx++, info.intermediate_size);
  sqlite3_bind_int(stmt, idx++, info.vocab_size);
  sqlite3_bind_text(stmt, idx++, quant_type_to_string(info.quant_type).c_str(),
                    -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.quant_details.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.tokenizer_type.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.tokenizer_path.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt, idx++, info.rope_freq_base);
  sqlite3_bind_double(stmt, idx++, info.rope_scale);
  sqlite3_bind_text(stmt, idx++, info.rope_scaling_type.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.description.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.license.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.source_url.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, idx++, info.chat_template.c_str(), -1,
                    SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, idx++, info.id);

  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  return rc == SQLITE_DONE;
}

bool ModelRegistry::remove_model(int64_t model_id, bool delete_file) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (delete_file) {
    auto model = get_model(model_id);
    if (model && !model->file_path.empty()) {
      std::remove(model->file_path.c_str());
    }
  }

  const char* delete_sql = "DELETE FROM models WHERE id = ?";
  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(db_, delete_sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  sqlite3_bind_int64(stmt, 1, model_id);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  return rc == SQLITE_DONE;
}

std::optional<ModelInfo> ModelRegistry::get_model(int64_t model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(stmt_get_model_);
  sqlite3_bind_int64(stmt_get_model_, 1, model_id);

  int rc = sqlite3_step(stmt_get_model_);
  if (rc == SQLITE_ROW) {
    return row_to_model_info(stmt_get_model_);
  }

  return std::nullopt;
}

std::optional<ModelInfo> ModelRegistry::get_model_by_identifier(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql = "SELECT * FROM models WHERE model_id = ?";
  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return std::nullopt;
  }

  sqlite3_bind_text(stmt, 1, model_id.c_str(), -1, SQLITE_TRANSIENT);
  rc = sqlite3_step(stmt);

  std::optional<ModelInfo> result;
  if (rc == SQLITE_ROW) {
    result = row_to_model_info(stmt);
  }

  sqlite3_finalize(stmt);
  return result;
}

std::vector<ModelInfo> ModelRegistry::list_models(
    const QueryOptions& options) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream sql;
  sql << "SELECT * FROM models WHERE 1=1";

  if (options.architecture) {
    sql << " AND architecture = '"
        << architecture_to_string(*options.architecture) << "'";
  }

  if (options.format) {
    sql << " AND format = '" << format_to_string(*options.format) << "'";
  }

  if (options.quant_type) {
    sql << " AND quant_type = '" << quant_type_to_string(*options.quant_type)
        << "'";
  }

  if (options.search_term) {
    sql << " AND (name LIKE '%" << *options.search_term << "%' "
        << "OR description LIKE '%" << *options.search_term << "%')";
  }

  sql << " ORDER BY " << options.order_by;
  sql << " LIMIT " << options.limit << " OFFSET " << options.offset;

  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(db_, sql.str().c_str(), -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return {};
  }

  std::vector<ModelInfo> results;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    results.push_back(row_to_model_info(stmt));
  }

  sqlite3_finalize(stmt);
  return results;
}

void ModelRegistry::touch_model(int64_t model_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  sqlite3_reset(stmt_touch_model_);
  sqlite3_bind_int64(stmt_touch_model_, 1, current_timestamp());
  sqlite3_bind_int64(stmt_touch_model_, 2, model_id);
  sqlite3_step(stmt_touch_model_);
}

void ModelRegistry::set_model_loaded(int64_t model_id, bool loaded) {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql = "UPDATE models SET is_loaded = ? WHERE id = ?";
  sqlite3_stmt* stmt;
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_int(stmt, 1, loaded ? 1 : 0);
  sqlite3_bind_int64(stmt, 2, model_id);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

bool ModelRegistry::add_tags(
    int64_t model_id,
    const std::unordered_map<std::string, std::string>& tags) {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql =
      "INSERT OR REPLACE INTO model_tags (model_id, key, value) VALUES (?, ?, "
      "?)";
  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return false;
  }

  for (const auto& [key, value] : tags) {
    sqlite3_reset(stmt);
    sqlite3_bind_int64(stmt, 1, model_id);
    sqlite3_bind_text(stmt, 2, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, value.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
  }

  sqlite3_finalize(stmt);
  return true;
}

std::unordered_map<std::string, std::string> ModelRegistry::get_tags(
    int64_t model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql = "SELECT key, value FROM model_tags WHERE model_id = ?";
  sqlite3_stmt* stmt;
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_int64(stmt, 1, model_id);

  std::unordered_map<std::string, std::string> tags;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* key =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    const char* value =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    tags[key] = value;
  }

  sqlite3_finalize(stmt);
  return tags;
}

int64_t ModelRegistry::register_adapter(const AdapterInfo& info) {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql = R"(
    INSERT INTO adapters (base_model_id, name, adapter_id, file_path, adapter_type, rank, scale, target_modules, created_timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  )";

  sqlite3_stmt* stmt;
  int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    return -1;
  }

  std::string target_modules_str;
  for (size_t i = 0; i < info.target_modules.size(); i++) {
    if (i > 0) target_modules_str += ",";
    target_modules_str += info.target_modules[i];
  }

  sqlite3_bind_int64(stmt, 1, info.base_model_id);
  sqlite3_bind_text(stmt, 2, info.name.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, info.adapter_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 4, info.file_path.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 5, info.adapter_type.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 6, info.rank);
  sqlite3_bind_double(stmt, 7, info.scale);
  sqlite3_bind_text(stmt, 8, target_modules_str.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 9, current_timestamp());

  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  if (rc != SQLITE_DONE) {
    return -1;
  }

  return sqlite3_last_insert_rowid(db_);
}

std::vector<AdapterInfo> ModelRegistry::get_adapters(
    int64_t base_model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql = "SELECT * FROM adapters WHERE base_model_id = ?";
  sqlite3_stmt* stmt;
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_int64(stmt, 1, base_model_id);

  std::vector<AdapterInfo> adapters;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    adapters.push_back(row_to_adapter_info(stmt));
  }

  sqlite3_finalize(stmt);
  return adapters;
}

bool ModelRegistry::remove_adapter(int64_t adapter_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  const char* sql = "DELETE FROM adapters WHERE id = ?";
  sqlite3_stmt* stmt;
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_int64(stmt, 1, adapter_id);
  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  return rc == SQLITE_DONE;
}

std::unordered_map<std::string, int64_t> ModelRegistry::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::unordered_map<std::string, int64_t> stats;

  const char* count_sql = "SELECT COUNT(*) FROM models";
  sqlite3_stmt* stmt;
  sqlite3_prepare_v2(db_, count_sql, -1, &stmt, nullptr);
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    stats["total_models"] = sqlite3_column_int64(stmt, 0);
  }
  sqlite3_finalize(stmt);

  const char* loaded_sql = "SELECT COUNT(*) FROM models WHERE is_loaded = 1";
  sqlite3_prepare_v2(db_, loaded_sql, -1, &stmt, nullptr);
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    stats["loaded_models"] = sqlite3_column_int64(stmt, 0);
  }
  sqlite3_finalize(stmt);

  return stats;
}

bool ModelRegistry::health_check() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return db_ != nullptr &&
         sqlite3_exec(db_, "SELECT 1", nullptr, nullptr, nullptr) == SQLITE_OK;
}

void ModelRegistry::close() {
  std::lock_guard<std::mutex> lock(mutex_);

  finalize_statements();

  if (db_) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
}

// Helper implementations

ModelInfo ModelRegistry::row_to_model_info(sqlite3_stmt* stmt) const {
  ModelInfo info;

  info.id = sqlite3_column_int64(stmt, 0);
  info.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
  info.model_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
  info.architecture = string_to_architecture(
      reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
  info.file_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
  info.format = string_to_format(
      reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
  info.file_size = sqlite3_column_int64(stmt, 6);
  info.sha256 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
  info.param_count = sqlite3_column_int64(stmt, 8);
  info.context_length = sqlite3_column_int(stmt, 9);
  info.hidden_size = sqlite3_column_int(stmt, 10);
  info.num_layers = sqlite3_column_int(stmt, 11);
  info.num_heads = sqlite3_column_int(stmt, 12);
  info.num_kv_heads = sqlite3_column_int(stmt, 13);
  info.intermediate_size = sqlite3_column_int(stmt, 14);
  info.vocab_size = sqlite3_column_int(stmt, 15);
  info.quant_type = string_to_quant_type(
      reinterpret_cast<const char*>(sqlite3_column_text(stmt, 16)));
  if (sqlite3_column_text(stmt, 17)) {
    info.quant_details =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 17));
  }
  if (sqlite3_column_text(stmt, 18)) {
    info.tokenizer_type =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 18));
  }
  if (sqlite3_column_text(stmt, 19)) {
    info.tokenizer_path =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 19));
  }
  info.rope_freq_base = sqlite3_column_double(stmt, 20);
  info.rope_scale = sqlite3_column_double(stmt, 21);
  if (sqlite3_column_text(stmt, 22)) {
    info.rope_scaling_type =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 22));
  }
  if (sqlite3_column_text(stmt, 23)) {
    info.description =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 23));
  }
  if (sqlite3_column_text(stmt, 24)) {
    info.license = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 24));
  }
  if (sqlite3_column_text(stmt, 25)) {
    info.source_url =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 25));
  }
  info.is_loaded = sqlite3_column_int(stmt, 26) != 0;
  info.last_used_timestamp = sqlite3_column_int64(stmt, 27);
  info.created_timestamp = sqlite3_column_int64(stmt, 28);
  if (sqlite3_column_text(stmt, 29)) {
    info.chat_template =
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 29));
  }

  return info;
}

AdapterInfo ModelRegistry::row_to_adapter_info(sqlite3_stmt* stmt) const {
  AdapterInfo info;

  info.id = sqlite3_column_int64(stmt, 0);
  info.base_model_id = sqlite3_column_int64(stmt, 1);
  info.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
  info.adapter_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
  info.file_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
  info.adapter_type =
      reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
  info.rank = sqlite3_column_int(stmt, 6);
  info.scale = sqlite3_column_double(stmt, 7);

  // Parse target_modules
  const char* modules_str =
      reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8));
  if (modules_str) {
    std::string modules(modules_str);
    size_t pos = 0;
    while ((pos = modules.find(',')) != std::string::npos) {
      info.target_modules.push_back(modules.substr(0, pos));
      modules.erase(0, pos + 1);
    }
    if (!modules.empty()) {
      info.target_modules.push_back(modules);
    }
  }

  info.created_timestamp = sqlite3_column_int64(stmt, 9);

  return info;
}

std::string ModelRegistry::architecture_to_string(
    ModelArchitecture arch) const {
  switch (arch) {
    case ModelArchitecture::LLAMA:
      return "llama";
    case ModelArchitecture::MISTRAL:
      return "mistral";
    case ModelArchitecture::MIXTRAL:
      return "mixtral";
    case ModelArchitecture::GEMMA:
      return "gemma";
    case ModelArchitecture::PHI:
      return "phi";
    case ModelArchitecture::QWEN:
      return "qwen";
    default:
      return "unknown";
  }
}

ModelArchitecture ModelRegistry::string_to_architecture(
    const std::string& str) const {
  if (str == "llama") return ModelArchitecture::LLAMA;
  if (str == "mistral") return ModelArchitecture::MISTRAL;
  if (str == "mixtral") return ModelArchitecture::MIXTRAL;
  if (str == "gemma") return ModelArchitecture::GEMMA;
  if (str == "phi") return ModelArchitecture::PHI;
  if (str == "qwen") return ModelArchitecture::QWEN;
  return ModelArchitecture::UNKNOWN;
}

std::string ModelRegistry::format_to_string(ModelFormat format) const {
  switch (format) {
    case ModelFormat::GGUF:
      return "gguf";
    case ModelFormat::SAFETENSORS:
      return "safetensors";
    case ModelFormat::MLX_NATIVE:
      return "mlx";
    default:
      return "unknown";
  }
}

ModelFormat ModelRegistry::string_to_format(const std::string& str) const {
  if (str == "gguf") return ModelFormat::GGUF;
  if (str == "safetensors") return ModelFormat::SAFETENSORS;
  if (str == "mlx") return ModelFormat::MLX_NATIVE;
  return ModelFormat::UNKNOWN;
}

std::string ModelRegistry::quant_type_to_string(QuantizationType type) const {
  switch (type) {
    case QuantizationType::NONE:
      return "none";
    case QuantizationType::Q4_0:
      return "Q4_0";
    case QuantizationType::Q4_1:
      return "Q4_1";
    case QuantizationType::Q5_0:
      return "Q5_0";
    case QuantizationType::Q5_1:
      return "Q5_1";
    case QuantizationType::Q8_0:
      return "Q8_0";
    case QuantizationType::Q2_K:
      return "Q2_K";
    case QuantizationType::Q3_K:
      return "Q3_K";
    case QuantizationType::Q4_K:
      return "Q4_K";
    case QuantizationType::Q5_K:
      return "Q5_K";
    case QuantizationType::Q6_K:
      return "Q6_K";
    case QuantizationType::Q8_K:
      return "Q8_K";
    case QuantizationType::IQ2_XXS:
      return "IQ2_XXS";
    case QuantizationType::IQ2_XS:
      return "IQ2_XS";
    case QuantizationType::IQ3_XXS:
      return "IQ3_XXS";
    case QuantizationType::MIXED:
      return "mixed";
    default:
      return "none";
  }
}

QuantizationType ModelRegistry::string_to_quant_type(
    const std::string& str) const {
  if (str == "Q4_0") return QuantizationType::Q4_0;
  if (str == "Q4_1") return QuantizationType::Q4_1;
  if (str == "Q5_0") return QuantizationType::Q5_0;
  if (str == "Q5_1") return QuantizationType::Q5_1;
  if (str == "Q8_0") return QuantizationType::Q8_0;
  if (str == "Q2_K") return QuantizationType::Q2_K;
  if (str == "Q3_K") return QuantizationType::Q3_K;
  if (str == "Q4_K") return QuantizationType::Q4_K;
  if (str == "Q5_K") return QuantizationType::Q5_K;
  if (str == "Q6_K") return QuantizationType::Q6_K;
  if (str == "Q8_K") return QuantizationType::Q8_K;
  if (str == "IQ2_XXS") return QuantizationType::IQ2_XXS;
  if (str == "IQ2_XS") return QuantizationType::IQ2_XS;
  if (str == "IQ3_XXS") return QuantizationType::IQ3_XXS;
  if (str == "mixed") return QuantizationType::MIXED;
  return QuantizationType::NONE;
}

}  // namespace registry
}  // namespace mlxr
