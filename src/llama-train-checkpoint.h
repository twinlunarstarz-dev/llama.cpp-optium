#pragma once

// P2: Transactional checkpoint store
//
// Versioned directory-based checkpoint format with crash-consistent saves,
// integrity verification, and safe concurrent access.
//
// Directory layout:
//   <root>/
//     manifest.json       - human-readable manifest (metadata, shard inventory)
//     data-00000000.bin  - shard files (typed tensor/state records)
//     data-00000001.bin
//     ...
//     COMMIT              - atomic commit marker with generation number
//
// Save protocol (atomic/crash-consistent):
//   1. Create staging directory <root>.staging.NNNNNN/
//   2. Write manifest.json and data shards to staging
//   3. fsync all data files
//   4. Write COMMIT.tmp with generation number
//   5. fsync COMMIT.tmp
//   6. fsync staging directory
//   7. Atomic rename staging -> <root>.N (versioned checkpoint)
//   8. Update <root>/COMMIT via atomic rename
//
// Load protocol:
//   1. Read COMMIT to get active generation
//   2. Open <root>.<generation>/ or <root>/ if no versioning
//   3. Verify manifest integrity (hash + schema version)
//   4. Verify source fingerprint matches expected
//   5. Verify each shard hash
//   6. Deserialize state records
//   7. Verify commit marker
//   8. All-or-nothing: reject any partial load
//
// Concurrency:
//   - Two writers cannot corrupt each other (staging + atomic rename)
//   - Readers see last committed generation
//   - Cross-device atomic rename must fail explicitly

#include "llama-train-state.h"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
  #include <sys/stat.h>   // mode_t
#endif

namespace llama_train {

// ---------------------------------------------------------------------------
// Syscall abstraction for checkpoint durability (test-injectable)
// ---------------------------------------------------------------------------
// The default implementation delegates to real POSIX/Windows syscalls.
// Tests can provide a custom instance that counts calls, injects EINTR,
// or simulates hard errors (EIO), etc., without affecting other stores.
// Thread-safe; per-store (NOT a mutable global).

struct syscall_interface {
    virtual ~syscall_interface() = default;

    // fsync a file descriptor; returns 0 on success, errno on failure
    virtual int do_fsync(int fd) = 0;

    // flock a file descriptor; returns 0 on success, errno on failure
    virtual int do_flock(int fd, int ops) = 0;

    // open a file; returns fd >= 0 on success, -1 on failure (sets errno)
    virtual int do_open(const char *path, int oflag, mode_t mode) = 0;

    // close a file descriptor; returns 0 on success, errno on failure
    virtual int do_close(int fd) = 0;

    // rename old_path -> new_path; returns 0 on success, errno on failure
    virtual int do_rename(const char *old_path, const char *new_path) = 0;
};

/// Default production implementation that calls real OS syscalls.
struct syscall_default : public syscall_interface {
    int do_fsync(int fd) override;
    int do_flock(int fd, int ops) override;
    int do_open(const char *path, int oflag, mode_t mode) override;
    int do_close(int fd) override;
    int do_rename(const char *old_path, const char *new_path) override;
};


// ---------------------------------------------------------------------------
// Checkpoint error codes
// ---------------------------------------------------------------------------

enum class ckpt_error : int {
    ok = 0,
    err_io,                    // filesystem I/O error
    err_invalid_path,          // path traversal or invalid characters
    err_schema_version,        // unsupported schema version
    err_fingerprint_mismatch,  // source fingerprint doesn't match
    err_hash_mismatch,         // shard hash verification failed
    err_corrupt_manifest,      // manifest is malformed/truncated
    err_missing_commit,        // COMMIT marker missing (incomplete save)
    err_partial_load,          // load rejected due to incomplete data
    err_concurrent_write,      // another writer detected
    err_cross_device,          // atomic rename not supported (cross-device)
    err_oversized,             // declaration exceeds configured limits
    err_unknown_feature,       // unknown mandatory feature flag
    err_path_traversal,        // detected path traversal attempt
    err_symlink,               // unexpected symlink (safety)
    err_no_space,              // disk space insufficient
    err_max_files,             // too many shard files
    err_overflow,              // checked arithmetic overflowed
    err_size_mismatch,         // buffer size doesn't match expected
};

const char * ckpt_error_name(ckpt_error e);

// ---------------------------------------------------------------------------
// Checkpoint configuration
// ---------------------------------------------------------------------------

struct checkpoint_config {
    // Maximum bytes per shard file (default 64MB)
    size_t max_shard_bytes = 64 * 1024 * 1024;

    // Maximum total checkpoint size (default 8GB)
    size_t max_total_bytes = 8ULL * 1024 * 1024 * 1024;

    // Maximum number of shard files (default 1024)
    uint32_t max_shard_count = 1024;

    // Enable fsync (default true for crash consistency)
    bool use_fsync = true;

    // Number of generations to retain (0 = unlimited)
    uint32_t retain_generations = 3;

    // Timeout in milliseconds for write lock acquisition (0 = infinite)
    uint64_t lock_timeout_ms = 30000;

    // Maximum manifest size bytes
    size_t max_manifest_bytes = 1 * 1024 * 1024;

    // Reject symlinks in checkpoint directory
    bool reject_symlinks = true;
};

// ---------------------------------------------------------------------------
// Manifest (JSON-based, human-readable)
// ---------------------------------------------------------------------------

/// Manifest schema version.
static constexpr uint32_t MANIFEST_SCHEMA_VERSION = 1;

struct manifest_entry {
    std::string shard_path;         // relative path within checkpoint dir
    uint64_t shard_size;           // file size in bytes
    std::array<uint8_t, 32> shard_hash; // SHA-256 of shard content
    uint32_t tensor_count;         // number of tensor records in this shard
};

struct checkpoint_manifest {
    uint32_t schema_version = MANIFEST_SCHEMA_VERSION;
    uint64_t generation;           // committed generation number
    source_fingerprint source_fp;  // SHA-256 of source GGUF model
    std::string source_path;       // original source model path (for reference)
    uint64_t tensor_count;         // total unique tensors
    uint64_t optimizer_step;       // global optimizer step at checkpoint
    session_cursors cursors;       // complete cursor/RNG state
    std::vector<manifest_entry> shards;
    std::array<uint8_t, 32> manifest_hash; // SHA-256 of manifest content (self-check)

    /// Compute and set the self-hash.
    void compute_self_hash();

    /// Verify the self-hash.
    bool verify_self_hash() const;
};

// ---------------------------------------------------------------------------
// Binary shard format
// ---------------------------------------------------------------------------

/// Magic bytes for checkpoint data shards.
static constexpr uint32_t SHARD_MAGIC = 0x54524E43; // "TRNC"

/// Shard record header (binary, little-endian).
struct shard_record_header {
    uint32_t magic;               // SHARD_MAGIC
    uint32_t version;             // schema version
    uint64_t tensor_id_hash;      // lower 64 bits of tensor SHA-256 ID
    uint32_t tensor_name_len;     // UTF-8 name length
    uint32_t param_type;          // canonical quantized type (stored as uint32)
    uint32_t rank;                // number of dimensions
    int64_t dims[4];              // up to 4 dimensions (0 if unused)
    uint64_t payload_size;        // quantized payload bytes
    uint64_t residual_size;       // F32 residual bytes
    uint64_t grad_size;           // gradient accumulator bytes
    uint64_t m_size;              // AdamW first moment bytes
    uint64_t v_size;              // AdamW second moment bytes
    uint32_t optimizer_kind;      // enum optimizer_kind
    uint64_t optimizer_step;      // per-tensor step
    float lr;                     // learning rate at checkpoint
    float weight_decay;           // weight decay at checkpoint
};

// ---------------------------------------------------------------------------
// Checkpoint store interface
// ---------------------------------------------------------------------------

class checkpoint_store {
public:
    explicit checkpoint_store(const checkpoint_config & cfg = checkpoint_config(),
                              std::shared_ptr<syscall_interface> syscalls = nullptr);
    ~checkpoint_store();

    checkpoint_store(checkpoint_store &&) noexcept;
    checkpoint_store & operator=(checkpoint_store &&) noexcept;
    checkpoint_store(const checkpoint_store &) = delete;
    checkpoint_store & operator=(const checkpoint_store &) = delete;

    /// Save the complete training state to the given directory path.
    /// Returns ckpt_error::ok on success.
    /// The save is atomic: previous checkpoint remains valid until commit.
    ckpt_error save(const std::filesystem::path & root,
                   const train_state & state);

    /// Load training state from the given directory path.
    /// Returns ckpt_error::ok on success with state populated.
    /// All-or-nothing: no partial load on any verification failure.
    ckpt_error load(const std::filesystem::path & root,
                   train_state & state);

    /// Get the last committed generation for a checkpoint directory.
    std::optional<uint64_t> get_last_generation(
        const std::filesystem::path & root) const;

    /// List available generations (sorted descending).
    std::vector<uint64_t> list_generations(
        const std::filesystem::path & root) const;

    /// Purge old generations beyond retain_generations.
    ckpt_error purge_old(const std::filesystem::path & root);

    /// Clean up stale staging directories.
    ckpt_error cleanup_staging(const std::filesystem::path & root);

    /// Compute source model fingerprint from GGUF file path.
    static source_fingerprint compute_source_fp(
        const std::filesystem::path & gguf_path);

    /// Get the current configuration.
    const checkpoint_config & config() const { return cfg_; }

private:
    checkpoint_config cfg_;
    // Optional test-injectable syscall interface; nullptr means use real syscalls.
    std::shared_ptr<syscall_interface> syscalls_;

    // Internal helpers
    ckpt_error write_manifest(const std::filesystem::path & dir,
                              const checkpoint_manifest & manifest);
    ckpt_error read_manifest(const std::filesystem::path & dir,
                             checkpoint_manifest & manifest);
    ckpt_error write_shard(const std::filesystem::path & path,
                          const train_state & state,
                          size_t start_index,
                          size_t max_bytes,
                          manifest_entry & entry_out);
    ckpt_error verify_and_load_shard(const std::filesystem::path & dir,
                                     const manifest_entry & entry,
                                     train_state & state);
    ckpt_error fsync_file(std::ofstream & f);

public:
    // File-level fsync (public for testing durability guarantees)
    ckpt_error fsync_file_path(const std::filesystem::path & p);

    // Directory fsync (public for testing durability guarantees)
    ckpt_error fsync_directory(const std::filesystem::path & dir);

    std::optional<std::filesystem::path> find_active_checkpoint(
        const std::filesystem::path & root) const;
    bool is_safe_path(const std::filesystem::path & p) const;

    // Cross-process write locking (RAII)
    // On POSIX, uses flock() on a dedicated .lock file beside the checkpoint root.
    // On Windows, falls back to create-exclusive-file semantics where possible.
    // Public for testing lock contention and symlink-security properties.
    ckpt_error acquire_write_lock(const std::filesystem::path & root,
                                  int & lock_fd);
    void release_write_lock(int & lock_fd);
};

} // namespace llama_train
