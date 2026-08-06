#include "llama-train-state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace llama_train {

// ===========================================================================
// Error code names
// ===========================================================================

const char * state_error_name(state_error e) {
    switch (e) {
        case state_error::ok:                  return "ok";
        case state_error::err_invalid_id:      return "err_invalid_id";
        case state_error::err_duplicate_id:    return "err_duplicate_id";
        case state_error::err_bad_dtype:       return "err_bad_dtype";
        case state_error::err_null_pointer:    return "err_null_pointer";
        case state_error::err_size_mismatch:   return "err_size_mismatch";
        case state_error::err_overflow:        return "err_overflow";
        case state_error::err_invalid_optimizer: return "err_invalid_optimizer";
        case state_error::err_invalid_cursor:  return "err_invalid_cursor";
        case state_error::err_schema_version:  return "err_schema_version";
        case state_error::err_corrupt_state:   return "err_corrupt_state";
        case state_error::err_allocation_failed: return "err_allocation_failed";
        case state_error::err_incompatible_alias: return "err_incompatible_alias";
        case state_error::err_missing_field:   return "err_missing_field";
        case state_error::err_dtype_coercion:  return "err_dtype_coercion";
    }
    return "unknown_state_error";
}

// ===========================================================================
// Dtype role names and validation
// ===========================================================================

const char * dtype_role_name(dtype_role r) {
    switch (r) {
        case dtype_role::parameter:      return "parameter";
        case dtype_role::gradient:       return "gradient";
        case dtype_role::residual:       return "residual";
        case dtype_role::sgd_momentum:   return "sgd_momentum";
        case dtype_role::adamw_m1:       return "adamw_m1";
        case dtype_role::adamw_m2:       return "adamw_m2";
    }
    return "unknown_dtype_role";
}

std::vector<ggml_type> get_accepted_dtypes(dtype_role role) {
    switch (role) {
        case dtype_role::parameter:
            // Accept any quantized type plus F32 for identity training
            return {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                    GGML_TYPE_Q6_K, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K};
        case dtype_role::gradient:
            // FP32 only for gradients (precision requirement)
            return {GGML_TYPE_F32};
        case dtype_role::residual:
            // FP32 required for error feedback
            return {GGML_TYPE_F32};
        case dtype_role::sgd_momentum:
            // FP32 for momentum state
            return {GGML_TYPE_F32};
        case dtype_role::adamw_m1:
        case dtype_role::adamw_m2:
            // FP32 for AdamW moments
            return {GGML_TYPE_F32};
    }
    return {};
}

bool validate_dtype(dtype_role role, ggml_type type) {
    auto accepted = get_accepted_dtypes(role);
    return std::find(accepted.begin(), accepted.end(), type) != accepted.end();
}

// ===========================================================================
// Optimizer kind names
// ===========================================================================

const char * optimizer_kind_name(optimizer_kind k) {
    switch (k) {
        case optimizer_kind::none: return "none";
        case optimizer_kind::sgd:  return "sgd";
        case optimizer_kind::adamw: return "adamw";
    }
    return "unknown_optimizer";
}

// ===========================================================================
// Dtype matrix validation
// ===========================================================================

state_error dtype_matrix::validate() const {
    // Parameter type must be valid
    if (!validate_dtype(dtype_role::parameter, param_type))
        return state_error::err_bad_dtype;

    // Gradient must be F32
    if (grad_type != GGML_TYPE_F32)
        return state_error::err_bad_dtype;

    // Residual must be F32
    if (residual_type != GGML_TYPE_F32)
        return state_error::err_bad_dtype;

    // Optimizer state must be F32
    if (optimizer_type != GGML_TYPE_F32)
        return state_error::err_bad_dtype;

    return state_error::ok;
}

// ===========================================================================
// Tensor state
// ===========================================================================

tensor_state::tensor_state()
    : id_val{},
      desc{},
      kind(optimizer_kind::none),
      step(0),
      lr(0.0f),
      weight_decay(0.0f),
      dtypes{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32},
      nelements(0) {}

tensor_state::~tensor_state() {
    free_buffers();
}

tensor_state::tensor_state(tensor_state && other) noexcept
    : id_val(other.id_val),
      desc(std::move(other.desc)),
      kind(other.kind),
      step(other.step),
      lr(other.lr),
      weight_decay(other.weight_decay),
      dtypes(other.dtypes),
      residual(std::move(other.residual)),
      grad(std::move(other.grad)),
      m(std::move(other.m)),
      v(std::move(other.v)),
      nelements(other.nelements) {
    other.nelements = 0;
}

tensor_state & tensor_state::operator=(tensor_state && other) noexcept {
    if (this != &other) {
        free_buffers();
        id_val = other.id_val;
        desc = std::move(other.desc);
        kind = other.kind;
        step = other.step;
        lr = other.lr;
        weight_decay = other.weight_decay;
        dtypes = other.dtypes;
        residual = std::move(other.residual);
        grad = std::move(other.grad);
        m = std::move(other.m);
        v = std::move(other.v);
        nelements = other.nelements;
        other.nelements = 0;
    }
    return *this;
}

state_error tensor_state::allocate(int64_t elems) {
    if (elems <= 0) return state_error::err_size_mismatch;

    // Check overflow: elems * sizeof(float)
    auto max_safe = static_cast<int64_t>(SIZE_MAX / sizeof(float));
    if (elems > max_safe) return state_error::err_overflow;

    size_t bytes = static_cast<size_t>(elems) * sizeof(float);

    free_buffers();
    nelements = elems;

    try {
        // Always allocate residual and grad as F32
        residual.reset(new float[elems]());
        grad.reset(new float[elems]());

        // Allocate optimizer state based on kind
        if (kind == optimizer_kind::sgd) {
            m.reset(new float[elems]());
        } else if (kind == optimizer_kind::adamw) {
            m.reset(new float[elems]());
            v.reset(new float[elems]());
        }
    } catch (const std::bad_alloc &) {
        free_buffers();
        return state_error::err_allocation_failed;
    }

    return state_error::ok;
}

void tensor_state::free_buffers() {
    residual.reset();
    grad.reset();
    m.reset();
    v.reset();
    nelements = 0;
}

// ===========================================================================
// Session cursors validation
// ===========================================================================

state_error session_cursors::validate() const {
    // micro_step must be <= grad_accum_steps (or grad_accum_steps == 0 means no accumulation)
    if (grad_accum_steps > 0 && micro_step > grad_accum_steps)
        return state_error::err_invalid_cursor;

    // Learning rates must be non-negative and finite
    if (!std::isfinite(base_lr) || base_lr < 0.0f)
        return state_error::err_invalid_cursor;

    if (!std::isfinite(min_lr) || min_lr < 0.0f)
        return state_error::err_invalid_cursor;

    if (base_lr < min_lr)
        return state_error::err_invalid_cursor;

    // Loss scale must be positive and finite
    if (!std::isfinite(loss_scale) || loss_scale <= 0.0f)
        return state_error::err_invalid_cursor;

    return state_error::ok;
}

float session_cursors::compute_lr(uint64_t current_step) const {
    if (base_lr == 0.0f) return 0.0f;

    float lr = base_lr;

    // Warmup phase
    if (lr_warmup_steps > 0 && current_step < lr_warmup_steps) {
        lr *= static_cast<float>(current_step) / static_cast<float>(lr_warmup_steps);
    }

    return std::max(lr, min_lr);
}

// ===========================================================================
// Source fingerprint helpers
// ===========================================================================

std::string source_fingerprint_to_hex(const source_fingerprint & fp) {
    return tensor_id_to_hex(fp);
}

// ===========================================================================
// Train state
// ===========================================================================

train_state::train_state() = default;

train_state::~train_state() {
    clear();
}

train_state::train_state(train_state && other) noexcept
    : schema_version(other.schema_version),
      source_fp(other.source_fp),
      cursors(other.cursors),
      entries(std::move(other.entries)),
      tied_groups(std::move(other.tied_groups)) {}

train_state & train_state::operator=(train_state && other) noexcept {
    if (this != &other) {
        clear();
        schema_version = other.schema_version;
        source_fp = other.source_fp;
        cursors = other.cursors;
        entries = std::move(other.entries);
        tied_groups = std::move(other.tied_groups);
    }
    return *this;
}

state_error train_state::add_tensor(tensor_descriptor desc,
                                    optimizer_kind kind,
                                    const dtype_matrix & dt,
                                    int64_t nelements) {
    if (desc.name.empty()) return state_error::err_missing_field;
    if (desc.dims.empty()) return state_error::err_missing_field;
    if (nelements <= 0) return state_error::err_size_mismatch;

    auto id = desc.id();
    if (!is_valid(id)) return state_error::err_invalid_id;

    std::string key = tensor_id_to_hex(id);
    if (entries.count(key)) return state_error::err_duplicate_id;

    auto err = dt.validate();
    if (err != state_error::ok) return err;

    tensor_state ts;
    ts.id_val = id;
    ts.desc = std::move(desc);
    ts.kind = kind;
    ts.dtypes = dt;
    ts.lr = cursors.compute_lr(cursors.global_optimizer_step);
    ts.weight_decay = 0.01f; // default

    err = ts.allocate(nelements);
    if (err != state_error::ok) return err;

    entries[key] = std::move(ts);
    return state_error::ok;
}

std::optional<std::reference_wrapper<tensor_state>> train_state::get(tensor_id id) {
    std::string key = tensor_id_to_hex(id);
    auto it = entries.find(key);
    if (it != entries.end()) return it->second;
    return std::nullopt;
}

std::optional<std::reference_wrapper<const tensor_state>> train_state::get(tensor_id id) const {
    std::string key = tensor_id_to_hex(id);
    auto it = entries.find(key);
    if (it != entries.end()) return it->second;
    return std::nullopt;
}

void train_state::remove(tensor_id id) {
    std::string key = tensor_id_to_hex(id);
    entries.erase(key);
}

void train_state::clear() {
    entries.clear();
    tied_groups.clear();
}

std::vector<tensor_id> train_state::unique_ids() const {
    std::vector<tensor_id> ids;
    ids.reserve(entries.size());
    for (const auto & [key, ts] : entries) {
        ids.push_back(ts.id_val);
    }
    return ids;
}

state_error train_state::validate() const {
    if (schema_version != STATE_SCHEMA_VERSION)
        return state_error::err_schema_version;

    auto err = cursors.validate();
    if (err != state_error::ok) return err;

    for (const auto & [key, ts] : entries) {
        if (!is_valid(ts.id_val))
            return state_error::err_invalid_id;

        err = ts.dtypes.validate();
        if (err != state_error::ok) return err;

        // Verify key matches ID
        if (key != tensor_id_to_hex(ts.id_val))
            return state_error::err_corrupt_state;
    }

    return state_error::ok;
}

} // namespace llama_train
