#pragma once

// Quantized training primitives (P1 - reference/CPU layer)
//
// Same-type dequantize → FP32 update with error feedback → requantize.
// Numerically authoritative reference implementation; CUDA kernels (P4) must
// reproduce these results bit-for-bit under deterministic stochastic rounding.
//
// This module does NOT own tensors. It operates on externally-owned payload
// buffers and persistent FP32 residual state. Graph indices, optimizer moments,
// checkpointing, paging, and CLI are all out of scope (deferred to P2–P6).

#include <ggml.h>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace llama_train_quant {

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

enum class error_code : int {
    ok = 0,
    err_unsupported_type,       // type has no to_float or from_float
    err_bad_row_alignment,      // n_per_row not a multiple of blck_size
    err_zero_size,              // n_rows == 0 || n_per_row == 0
    err_null_pointer,           // required buffer is null
    err_overflow,               // checked size/stride arithmetic overflowed
    err_non_finite,             // input, update, or residual contains NaN/Inf
    err_malformed_buffer,       // quantized payload failed ggml_validate_row_data
    err_validation_failed,      // generic validation failure
};

inline const char * error_code_name(error_code ec) {
    switch (ec) {
        case error_code::ok:                    return "ok";
        case error_code::err_unsupported_type:  return "err_unsupported_type";
        case error_code::err_bad_row_alignment: return "err_bad_row_alignment";
        case error_code::err_zero_size:         return "err_zero_size";
        case error_code::err_null_pointer:      return "err_null_pointer";
        case error_code::err_overflow:          return "err_overflow";
        case error_code::err_non_finite:        return "err_non_finite";
        case error_code::err_malformed_buffer:  return "err_malformed_buffer";
        case error_code::err_validation_failed: return "err_validation_failed";
    }
    return "unknown_error_code";
}

// ---------------------------------------------------------------------------
// Result struct (carries diagnostics)
// ---------------------------------------------------------------------------

struct update_result {
    error_code ec;
    int64_t n_rows;            // actual rows processed (0 on failure)
    int64_t n_per_row;         // elements per row
    double quantization_rms_error;  // RMS error of last quantize step
    double max_abs_error;          // max absolute error of last quantize step
};

// ---------------------------------------------------------------------------
// Stochastic rounding configuration
// ---------------------------------------------------------------------------
//
// Deterministic counter-based stochastic rounding. No hidden global RNG state.
// The caller provides a seed (64-bit), a per-tensor key (64-bit), and a
// starting counter offset. Each element maps to a unique counter value:
//
//   counter[i] = counter_start + row * n_per_row + col
//
// The rounding decision for element i uses a hash of (seed, key, counter[i])
// to produce a uniform fraction in [0,1) and applies standard stochastic
// rounding: round down with probability (1 - frac), up with probability frac,
// where frac = value - floor(value) in the quantized domain.
//
// A separate "nearest" mode disables stochastic rounding entirely (deterministic
// round-to-nearest). This is required for forward-pass dequantize operations
// and any context where unbiasedness is not needed.

struct sr_config {
    bool stochastic;           // true = stochastic rounding, false = nearest
    uint64_t seed;             // global session seed
    uint64_t key;              // per-tensor key (stable identity)
    int64_t counter_start;     // starting counter for this call
};

// Default: nearest rounding (safe default for dequantize/read paths).
inline sr_config sr_nearest() {
    return {false, 0, 0, 0};
}

// Stochastic rounding config with explicit seed/key/counter.
inline sr_config sr_stochastic(uint64_t seed, uint64_t key, int64_t counter_start) {
    return {true, seed, key, counter_start};
}

// ---------------------------------------------------------------------------
// Type support queries
// ---------------------------------------------------------------------------

// Returns true if the type is supported for quantized training primitives.
// A type is supported when:
//   - ggml_get_type_traits(type)->to_float != NULL  (dequantize)
//   - ggml_get_type_traits_cpu(type)->from_float != NULL  (quantize)
//   - blck_size > 0
bool is_supported(ggml_type type);

// Returns the first supported type from a list, or GGML_TYPE_COUNT if none.
ggml_type first_supported(const std::vector<ggml_type> & types);

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

// Validate tensor geometry and type support before any operation.
error_code validate(ggml_type type, int64_t n_rows, int64_t n_per_row);

// ---------------------------------------------------------------------------
// Bounded row dequantization (quantized → FP32)
// ---------------------------------------------------------------------------
//
// Dequantize rows[row_start .. row_start + n_rows) from src into dst.
// dst_stride is the number of floats between consecutive output rows
// (dst_stride >= n_per_row). Use dst_stride == n_per_row for contiguous layout.
//
// Always uses deterministic nearest rounding (no stochastic rounding on read).

update_result dequantize_rows(
    ggml_type type,
    const void * src,
    int64_t row_start,
    int64_t n_rows,
    int64_t n_per_row,
    float * dst,
    int64_t dst_stride);

// ---------------------------------------------------------------------------
// Bounded same-type requantization (FP32 → quantized)
// ---------------------------------------------------------------------------
//
// Quantize rows from src (contiguous FP32, stride src_stride floats/row) into
// dst (quantized buffer). Returns quantization error statistics.
//
// Uses the provided sr_config for rounding policy.

update_result quantize_rows(
    ggml_type type,
    const float * src,
    int64_t src_stride,
    int64_t n_rows,
    int64_t n_per_row,
    void * dst,
    const sr_config & sr);

// ---------------------------------------------------------------------------
// Error-feedback quantized update (training core primitive)
// ---------------------------------------------------------------------------
//
// Atomic dequantize → FP32 update → requantize with persistent error feedback.
//
// For each element:
//   1. proposed[i] = dequant(Q)[i] + residual[i] + lr * update[i]
//   2. Q_new = quantize(proposed)   [uses sr_config for rounding]
//   3. residual[i] = proposed[i] - dequant(Q_new)[i]
//
// The residual buffer must be exactly n_rows * n_per_row floats and is
// updated in-place. On first call, initialize residual to zero.
//
// Invariant: repeated sub-quantum updates eventually change the quantized
// payload, because the residual accumulates the quantization error.
//
// Parameters:
//   src       - current quantized payload (read-write, updated in-place)
//   residual  - persistent FP32 error feedback (read-write, must be zero-init on first use)
//   update    - FP32 gradient/update delta (read-only)
//   lr        - learning rate multiplier applied to update
//   sr        - stochastic rounding configuration
//
// Returns update_result with error statistics and any error code.

update_result apply_update_with_error_feedback(
    ggml_type type,
    void * src,               // quantized payload (read-write)
    float * residual,         // persistent FP32 error (read-write)
    const float * update,     // FP32 update delta (read-only)
    float lr,
    int64_t n_rows,
    int64_t n_per_row,
    const sr_config & sr);

// ---------------------------------------------------------------------------
// Same-type round-trip (dequantize → requantize, no update)
// ---------------------------------------------------------------------------
//
// Quantize src into dst using the same type. Useful for testing payload
// stability and measuring quantization error without an update step.

update_result roundtrip_same_type(
    ggml_type type,
    const void * src,
    int64_t n_rows,
    int64_t n_per_row,
    void * dst,
    const sr_config & sr);

} // namespace llama_train_quant
