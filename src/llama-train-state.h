#pragma once

// P2: Independently typed external training state
//
// Provides canonical quantized parameter metadata, FP32 error-feedback
// residual, gradient accumulation, SGD momentum, and AdamW first/second
// moments - all keyed by stable tensor identity (not graph indices).
//
// Parameter dtype, gradient dtype, each optimizer-state dtype, and
// residual dtype are separately encoded and validated.
//
// This package stores/checkpoints typed state. The later ggml-opt package
// consumes it. No integration with llama-context or ggml-opt execution
// occurs here.

#include "llama-train-identity.h"

#include <ggml.h>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_train {

// ---------------------------------------------------------------------------
// Error codes for state operations
// ---------------------------------------------------------------------------

enum class state_error : int {
    ok = 0,
    err_invalid_id,              // tensor_id is all zeros
    err_duplicate_id,            // duplicate tensor identity
    err_bad_dtype,               // unsupported dtype for this role
    err_null_pointer,            // required buffer is null
    err_size_mismatch,           // buffer size doesn't match expected
    err_overflow,                // checked arithmetic overflow
    err_invalid_optimizer,       // unknown optimizer kind
    err_invalid_cursor,          // impossible cursor combination
    err_schema_version,          // incompatible schema version
    err_corrupt_state,           // deserialization failure
    err_allocation_failed,       // memory allocation failed
    err_incompatible_alias,      // tied alias has different shape/type
    err_missing_field,           // required field not present
    err_dtype_coercion,          // rejected silent dtype coercion
};

const char * state_error_name(state_error e);

// ---------------------------------------------------------------------------
// Dtype roles and validation
// ---------------------------------------------------------------------------

/// Each piece of optimizer state has an independently configurable dtype.
enum class dtype_role : int {
    parameter,       // canonical quantized parameter (Q4_K, Q8_0, etc.)
    gradient,        // gradient accumulator
    residual,        // FP32 error-feedback residual
    sgd_momentum,    // SGD momentum buffer
    adamw_m1,        // AdamW first moment (m)
    adamw_m2,        // AdamW second moment (v)
};

const char * dtype_role_name(dtype_role r);

/// Get the accepted dtypes for a given role.
std::vector<ggml_type> get_accepted_dtypes(dtype_role role);

/// Validate that a dtype is acceptable for a role.
bool validate_dtype(dtype_role role, ggml_type type);

// ---------------------------------------------------------------------------
// Optimizer kind
// ---------------------------------------------------------------------------

enum class optimizer_kind : int {
    none = 0,
    sgd,           // SGD with optional momentum
    adamw,         // AdamW
};

const char * optimizer_kind_name(optimizer_kind k);

// ---------------------------------------------------------------------------
// Per-tensor optimizer state
// ---------------------------------------------------------------------------

/// Independent dtype descriptors for each role in this tensor's optimizer state.
struct dtype_matrix {
    ggml_type param_type;        // canonical quantized type (e.g., GGML_TYPE_Q8_0)
    ggml_type grad_type;         // gradient accumulator type (default F32)
    ggml_type residual_type;     // error-feedback residual type (default F32)
    ggml_type optimizer_type;    // optimizer state type (F32 for AdamW/SGD)

    /// Validate all dtypes are acceptable for their roles.
    state_error validate() const;
};

/// Per-tensor optimizer state record.
struct tensor_state {
    tensor_id id_val;                    // stable identity
    tensor_descriptor desc;              // name, type, shape metadata
    optimizer_kind kind;                 // SGD, AdamW, or none
    uint64_t step;                      // last optimizer step for this tensor
    float lr;                           // learning rate at last step
    float weight_decay;                 // weight decay at last step

    // Dtype configuration
    dtype_matrix dtypes;

    // Buffers (owned, allocated on demand)
    std::unique_ptr<float[]> residual;   // FP32 error-feedback residual
    std::unique_ptr<float[]> grad;       // gradient accumulator
    std::unique_ptr<float[]> m;          // AdamW first moment / SGD momentum
    std::unique_ptr<float[]> v;          // AdamW second moment

    int64_t nelements;                  // total elements (for buffer sizing)

    tensor_state();
    ~tensor_state();
    tensor_state(tensor_state && other) noexcept;
    tensor_state & operator=(tensor_state && other) noexcept;
    tensor_state(const tensor_state &) = delete;
    tensor_state & operator=(const tensor_state &) = delete;

    /// Allocate buffers based on descriptor and dtype matrix.
    state_error allocate(int64_t nelements);

    /// Free all buffers.
    void free_buffers();

    /// Get pointer to residual buffer (may be null if not allocated).
    float * residual_ptr() { return residual.get(); }
    const float * residual_ptr() const { return residual.get(); }

    /// Get pointer to gradient buffer.
    float * grad_ptr() { return grad.get(); }
    const float * grad_ptr() const { return grad.get(); }

    /// Get pointer to AdamW first moment.
    float * m_ptr() { return m.get(); }
    const float * m_ptr() const { return m.get(); }

    /// Get pointer to AdamW second moment.
    float * v_ptr() { return v.get(); }
    const float * v_ptr() const { return v.get(); }
};

// ---------------------------------------------------------------------------
// Session cursors and RNG state
// ---------------------------------------------------------------------------

/// Complete resumable cursor/RNG/scheduler state.
struct session_cursors {
    uint64_t global_optimizer_step = 0;   // total optimizer steps completed
    uint64_t micro_step = 0;              // current micro-step within gradient accumulation
    uint64_t grad_accum_steps = 0;        // total gradient accumulation steps
    uint64_t epoch = 0;                   // current epoch
    uint64_t sample_index = 0;            // current sample/token index within epoch
    uint64_t data_cursor = 0;             // absolute data position
    uint64_t shard_cursor = 0;            // current shard file index

    // Learning rate schedule state
    uint64_t lr_warmup_steps = 0;         // warmup step count (0 = no warmup)
    float base_lr = 0.0f;                 // base learning rate
    float min_lr = 0.0f;                  // minimum learning rate

    // Loss scaling / overflow state
    float loss_scale = 1.0f;              // current dynamic loss scale (default 1.0)
    uint64_t overflow_count = 0;          // consecutive overflow count

    // RNG state for deterministic resume
    uint64_t sr_seed = 0;                // stochastic rounding seed
    uint64_t sr_counter = 0;             // stochastic rounding counter
    uint64_t shuffle_rng_state[8]{};     // data shuffle RNG (SplitMix64 state)

    // Committed state generation (for checkpoint integrity)
    uint64_t committed_generation = 0;    // last committed checkpoint generation

    /// Validate impossible combinations.
    state_error validate() const;

    /// Compute current learning rate from schedule state and global step.
    float compute_lr(uint64_t current_step) const;
};

// ---------------------------------------------------------------------------
// Source model fingerprint
// ---------------------------------------------------------------------------

/// 32-byte SHA-256 fingerprint of the source GGUF model.
using source_fingerprint = std::array<uint8_t, 32>;

std::string source_fingerprint_to_hex(const source_fingerprint & fp);

// ---------------------------------------------------------------------------
// Complete training state container
// ---------------------------------------------------------------------------

/// Version identifier for the serialization schema.
static constexpr uint32_t STATE_SCHEMA_VERSION = 1;

/// Complete external training state.
class train_state {
public:
    train_state();
    ~train_state();
    train_state(train_state && other) noexcept;
    train_state & operator=(train_state && other) noexcept;
    train_state(const train_state &) = delete;
    train_state & operator=(const train_state &) = delete;

    // Schema version
    uint32_t schema_version = STATE_SCHEMA_VERSION;

    // Source model fingerprint (SHA-256 of source GGUF)
    source_fingerprint source_fp{};

    // Session cursors and RNG state
    session_cursors cursors;

    // Per-tensor optimizer state, keyed by stable tensor_id
    std::unordered_map<std::string, tensor_state> entries;

    // Tied group information (for validation on load)
    std::vector<tied_group> tied_groups;

    /// Add or update a tensor's optimizer state.
    state_error add_tensor(tensor_descriptor desc,
                          optimizer_kind kind,
                          const dtype_matrix & dt,
                          int64_t nelements);

    /// Get mutable reference to tensor state by ID.
    std::optional<std::reference_wrapper<tensor_state>> get(tensor_id id);

    /// Get const reference to tensor state by ID.
    std::optional<std::reference_wrapper<const tensor_state>> get(tensor_id id) const;

    /// Remove a tensor's state.
    void remove(tensor_id id);

    /// Clear all state.
    void clear();

    /// Number of tracked tensors.
    size_t size() const { return entries.size(); }

    /// Check if empty.
    bool empty() const { return entries.empty(); }

    /// Enumerate unique tensor IDs (for audit).
    std::vector<tensor_id> unique_ids() const;

    /// Validate complete state consistency.
    state_error validate() const;
};

// ===========================================================================
// Inline implementations
// ===========================================================================

inline tensor_id tied_group::id() const {
    return base.id();
}

} // namespace llama_train
