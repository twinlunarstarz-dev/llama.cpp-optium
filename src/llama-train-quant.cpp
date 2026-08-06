#include "llama-train-quant.h"

#include <ggml.h>
#include <ggml-cpu.h>

// Q8_0 block layout (from ggml-common.h, reproduced here to avoid internal-header dependency).
// Layout: ggml_half d + int8_t qs[32]. Total 34 bytes per block of 32 elements.
struct block_q8_0_staging {
    uint16_t d;       // ggml_half (fp16) delta
    int8_t  qs[32];   // quants
};
static_assert(sizeof(block_q8_0_staging) == sizeof(uint16_t) + 32, "wrong q8_0 block size");

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llama_train_quant {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Deterministic counter-based hash for stochastic rounding.
// Produces a value in [0, 1) from (seed, key, counter).
// Uses SplitMix64-style mixing for uniform distribution of high bits.
inline double sr_hash(uint64_t seed, uint64_t key, int64_t counter) {
    uint64_t x = static_cast<uint64_t>(counter) + seed + key * 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    // Extract top 52 bits for uniform double in [0, 1)
    return (x >> 11) * (1.0 / (1ULL << 52));
}

// Stochastic rounding: round a normalized value to [0, max_code] using
// unbiased counter-based stochastic projection.
// For normalized in [lo, hi] where lo = floor(normalized), hi = ceil(normalized):
//   frac = normalized - lo
//   select hi with probability frac, lo with probability (1 - frac)
// Endpoints: if frac == 0, always return lo. If frac == 1, always return hi.
inline int sr_round(double normalized, int max_code,
                    uint64_t seed, uint64_t key, int64_t counter) {
    double floored = std::floor(normalized);
    double frac = normalized - floored;

    // Exact endpoint handling
    if (frac <= 0.0) {
        int code = static_cast<int>(floored);
        return std::max(0, std::min(code, max_code));
    }
    if (frac >= 1.0) {
        int code = static_cast<int>(floored) + 1;
        return std::max(0, std::min(code, max_code));
    }

    // Unbiased stochastic projection: select ceil with probability frac
    double r = sr_hash(seed, key, counter);
    int code = static_cast<int>(floored) + (r < frac ? 1 : 0);
    return std::max(0, std::min(code, max_code));
}

// Checked multiplication that returns false on overflow.
inline bool checked_mul_int64(int64_t a, int64_t b, int64_t & out) {
    if (a == 0 || b == 0) { out = 0; return true; }
    auto max_int64 = static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL);
    if (a > 0 && b > 0 && a > max_int64 / b) return false;
    if (a < 0 && b < 0 && a < max_int64 / (-b)) return false;
    if (a > 0 && b < 0 && b < -max_int64 / a) return false;
    if (a < 0 && b > 0 && a < -max_int64 / b) return false;
    out = a * b;
    return true;
}

// Check for non-finite values in an FP32 buffer.
bool has_non_finite(const float * data, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) return true;
    }
    return false;
}

// Compute quantization error by dequantizing and comparing with source.
update_result compute_quant_error(
        ggml_type type, const void * quant_data,
        const float * src_f32, int64_t n_per_row) {
    const auto * traits = ggml_get_type_traits(type);
    std::vector<float> dequant(n_per_row);
    traits->to_float(quant_data, dequant.data(), n_per_row);

    double sum_sq = 0;
    double max_err = 0;
    for (int64_t i = 0; i < n_per_row; ++i) {
        double err = static_cast<double>(dequant[i]) - static_cast<double>(src_f32[i]);
        sum_sq += err * err;
        double abs_err = std::fabs(err);
        if (abs_err > max_err) max_err = abs_err;
    }

    return {error_code::ok, 1, n_per_row, std::sqrt(sum_sq / n_per_row), max_err};
}

// Quantize a single row using ggml from_float (nearest rounding).
update_result quantize_row_nearest(
        ggml_type type, const float * src, int64_t n_per_row, void * dst) {
    const auto * traits = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);

    if (!traits || !traits->to_float)
        return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};

    if (traits_cpu && traits_cpu->from_float) {
        traits_cpu->from_float(src, dst, n_per_row);
    } else if (traits->from_float_ref) {
        traits->from_float_ref(src, dst, n_per_row);
    } else {
        return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};
    }

    return compute_quant_error(type, dst, src, n_per_row);
}

// Stochastic quantization for Q8_0: elementwise unbiased projection.
// Q8_0 blocks have independent per-block scales computed from max|x|/127.
// The scale depends only on the data range, not on individual codes, so
// stochastic rounding per-element is mathematically valid.
// Actual Q8_0 layout: ggml_half d (2 bytes) + int8_t qs[32] = 34 bytes/block.
update_result quantize_row_q8_0_stochastic(
        const float * src, int64_t n_per_row, void * dst,
        uint64_t seed, uint64_t key, int64_t counter_start) {
    const int qk = 32;
    const int nb = static_cast<int>(n_per_row / qk);
    size_t block_size = ggml_row_size(GGML_TYPE_Q8_0, qk); // should be 34

    for (int b = 0; b < nb; ++b) {
        int64_t base = static_cast<int64_t>(b) * qk;

        // Step 1: Compute block scale (same as ggml from_float)
        float max_abs = 0.0f;
        for (int i = 0; i < qk; ++i) {
            float v = std::fabs(src[base + i]);
            if (v > max_abs) max_abs = v;
        }
        float d = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;

        // Step 2: Write to dst block-by-block
        char * block_dst = static_cast<char *>(dst) + b * block_size;

        // Write d as fp16 at offset 0
        uint16_t d_fp16 = ggml_fp32_to_fp16(d);
        std::memcpy(block_dst, &d_fp16, sizeof(uint16_t));

        // Write qs starting at offset 2
        int8_t * qs = reinterpret_cast<int8_t *>(block_dst + sizeof(uint16_t));

        // Step 3: Stochastic rounding per element
        for (int i = 0; i < qk; ++i) {
            double normalized = static_cast<double>(src[base + i]) * static_cast<double>(d);
            // Map from [-127, 127] to [0, 254] for stochastic rounding
            double shifted = normalized + 127.0;
            int code = sr_round(shifted, 254, seed, key, counter_start + base + i);
            qs[i] = static_cast<int8_t>(code - 127);
        }
    }

    return compute_quant_error(GGML_TYPE_Q8_0, dst, src, n_per_row);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool is_supported(ggml_type type) {
    if (type >= GGML_TYPE_COUNT) return false;
    if (type == GGML_TYPE_F32) return true;
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) return false;

    const auto * traits = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);
    if (!traits || !traits_cpu) return false;
    if (traits->blck_size <= 0) return false;
    if (!traits->to_float) return false;
    if (!traits_cpu->from_float && !traits->from_float_ref) return false;
    return true;
}

ggml_type first_supported(const std::vector<ggml_type> & types) {
    for (auto t : types) {
        if (is_supported(t)) return t;
    }
    return GGML_TYPE_COUNT;
}

error_code validate(ggml_type type, int64_t n_rows, int64_t n_per_row) {
    if (n_rows <= 0 || n_per_row <= 0) return error_code::err_zero_size;
    if (!is_supported(type)) return error_code::err_unsupported_type;

    int64_t blck = ggml_blck_size(type);
    if (blck <= 0 || n_per_row % blck != 0) return error_code::err_bad_row_alignment;

    int64_t total;
    if (!checked_mul_int64(n_rows, n_per_row, total)) return error_code::err_overflow;

    return error_code::ok;
}

update_result dequantize_rows(
        ggml_type type, const void * src, int64_t row_start,
        int64_t n_rows, int64_t n_per_row, float * dst, int64_t dst_stride) {
    if (!src || !dst) return {error_code::err_null_pointer, 0, n_per_row, 0, 0};
    if (dst_stride < n_per_row)
        return {error_code::err_bad_row_alignment, 0, n_per_row, 0, 0};

    auto ec = validate(type, n_rows, n_per_row);
    if (ec != error_code::ok) return {ec, 0, n_per_row, 0, 0};

    const auto * traits = ggml_get_type_traits(type);
    size_t row_size = ggml_row_size(type, n_per_row);

    int64_t total_dst;
    if (!checked_mul_int64(n_rows, dst_stride, total_dst))
        return {error_code::err_overflow, 0, n_per_row, 0, 0};

    for (int64_t r = 0; r < n_rows; ++r) {
        const char * row_src = static_cast<const char *>(src) + (row_start + r) * row_size;
        float * row_dst = dst + r * dst_stride;

        if (type == GGML_TYPE_F32) {
            std::memcpy(row_dst, row_src, n_per_row * sizeof(float));
        } else {
            traits->to_float(row_src, row_dst, n_per_row);
        }
    }

    return {error_code::ok, n_rows, n_per_row, 0, 0};
}

update_result quantize_rows(
        ggml_type type, const float * src, int64_t src_stride,
        int64_t n_rows, int64_t n_per_row, void * dst, const sr_config & sr) {
    if (!src || !dst) return {error_code::err_null_pointer, 0, n_per_row, 0, 0};
    if (src_stride < n_per_row)
        return {error_code::err_bad_row_alignment, 0, n_per_row, 0, 0};

    auto ec = validate(type, n_rows, n_per_row);
    if (ec != error_code::ok) return {ec, 0, n_per_row, 0, 0};

    int64_t total;
    if (!checked_mul_int64(n_rows, n_per_row, total))
        return {error_code::err_overflow, 0, n_per_row, 0, 0};

    // F32: identity quantization
    if (type == GGML_TYPE_F32) {
        for (int64_t r = 0; r < n_rows; ++r) {
            std::memcpy(
                static_cast<char *>(dst) + r * n_per_row * sizeof(float),
                src + r * src_stride,
                n_per_row * sizeof(float));
        }
        return {error_code::ok, n_rows, n_per_row, 0, 0};
    }

    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float)
        return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};

    // Nearest rounding: use ggml from_float directly (fast, verified)
    if (!sr.stochastic) {
        double total_sum_sq = 0;
        double total_max_err = 0;

        for (int64_t r = 0; r < n_rows; ++r) {
            size_t row_size = ggml_row_size(type, n_per_row);
            char * row_dst = static_cast<char *>(dst) + r * row_size;

            auto res = quantize_row_nearest(type, src + r * src_stride, n_per_row, row_dst);
            if (res.ec != error_code::ok) return {res.ec, r, n_per_row, 0, 0};

            total_sum_sq += res.quantization_rms_error * res.quantization_rms_error * n_per_row;
            if (res.max_abs_error > total_max_err) total_max_err = res.max_abs_error;
        }

        return {error_code::ok, n_rows, n_per_row,
                std::sqrt(total_sum_sq / total), total_max_err};
    }

    // Stochastic rounding path:
    // For Q8_0: elementwise unbiased stochastic projection (scale is block-local
    // but independent of individual code choices).
    // For other types (Q4_K, NVFP4): stochastic rounding requires block-coupled
    // scale/code co-optimization that cannot be correctly reduced to elementwise
    // projection. Return explicit error rather than silently falling back to
    // biased approximation.

    if (type == GGML_TYPE_Q8_0) {
        double total_sum_sq = 0;
        double total_max_err = 0;

        for (int64_t r = 0; r < n_rows; ++r) {
            size_t row_size = ggml_row_size(type, n_per_row);
            char * row_dst = static_cast<char *>(dst) + r * row_size;

            auto res = quantize_row_q8_0_stochastic(
                src + r * src_stride, n_per_row, row_dst,
                sr.seed, sr.key, sr.counter_start + r * n_per_row);
            if (res.ec != error_code::ok) return {res.ec, r, n_per_row, 0, 0};

            total_sum_sq += res.quantization_rms_error * res.quantization_rms_error * n_per_row;
            if (res.max_abs_error > total_max_err) total_max_err = res.max_abs_error;
        }

        return {error_code::ok, n_rows, n_per_row,
                std::sqrt(total_sum_sq / total), total_max_err};
    }

    // Q4_K, NVFP4, and other block-coupled types: stochastic rounding is not
    // supported because scales are computed jointly with codes. Changing a single
    // code stochastically would require recomputing the block scale, which changes
    // all elements in the block, making elementwise stochastic projection invalid.
    return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};
}

update_result apply_update_with_error_feedback(
        ggml_type type, void * src, float * residual,
        const float * update, float lr,
        int64_t n_rows, int64_t n_per_row, const sr_config & sr) {
    if (!src || !residual || !update)
        return {error_code::err_null_pointer, 0, n_per_row, 0, 0};

    auto ec = validate(type, n_rows, n_per_row);
    if (ec != error_code::ok) return {ec, 0, n_per_row, 0, 0};

    int64_t total;
    if (!checked_mul_int64(n_rows, n_per_row, total))
        return {error_code::err_overflow, 0, n_per_row, 0, 0};

    // Check for non-finite in update and residual BEFORE any mutation
    if (has_non_finite(update, total))
        return {error_code::err_non_finite, 0, n_per_row, 0, 0};
    if (has_non_finite(residual, total))
        return {error_code::err_non_finite, 0, n_per_row, 0, 0};

    // F32 fast path: direct update, no quantization needed
    if (type == GGML_TYPE_F32) {
        float * data = static_cast<float *>(src);
        for (int64_t i = 0; i < total; ++i) {
            data[i] += lr * update[i];
            residual[i] = 0.0f;
        }
        return {error_code::ok, n_rows, n_per_row, 0, 0};
    }

    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float)
        return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};

    // Allocate scratch buffers (bounded by O(n_per_row))
    std::vector<float> proposed(n_per_row);
    std::vector<float> dequant_new(n_per_row);
    size_t row_size = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> quant_scratch(row_size);

    double total_sum_sq = 0;
    double total_max_err = 0;
    int64_t rows_processed = 0;

    for (int64_t r = 0; r < n_rows; ++r) {
        // Step 1: Dequantize current payload
        const char * row_src = static_cast<const char *>(src) + r * row_size;
        std::vector<float> dequant_old(n_per_row);
        traits->to_float(row_src, dequant_old.data(), n_per_row);

        // Step 2: Compute proposed values
        // proposed[i] = dequant(Q)[i] + residual[i] + lr * update[i]
        int64_t row_offset = r * n_per_row;
        for (int64_t i = 0; i < n_per_row; ++i) {
            proposed[i] = dequant_old[i] + residual[row_offset + i] + lr * update[row_offset + i];
        }

        // Check for non-finite proposed values
        bool finite = true;
        for (int64_t i = 0; i < n_per_row; ++i) {
            if (!std::isfinite(proposed[i])) { finite = false; break; }
        }
        if (!finite)
            return {error_code::err_non_finite, rows_processed, n_per_row, 0, 0};

        // Step 3: Quantize proposed values
        auto qr = quantize_rows(type, proposed.data(), n_per_row, 1, n_per_row,
                                 quant_scratch.data(), sr);
        if (qr.ec != error_code::ok)
            return {qr.ec, rows_processed, n_per_row, 0, 0};

        // Validate quantized output
        if (!ggml_validate_row_data(type, quant_scratch.data(), row_size))
            return {error_code::err_malformed_buffer, rows_processed, n_per_row, 0, 0};

        // Step 4: Dequantize new payload for error computation
        traits->to_float(quant_scratch.data(), dequant_new.data(), n_per_row);

        // Step 5: Update residual
        // residual[i] = proposed[i] - dequant(Q_new)[i]
        for (int64_t i = 0; i < n_per_row; ++i) {
            residual[row_offset + i] = proposed[i] - dequant_new[i];

            double err = static_cast<double>(proposed[i]) -
                         static_cast<double>(dequant_new[i]);
            total_sum_sq += err * err;
            double abs_err = std::fabs(err);
            if (abs_err > total_max_err) total_max_err = abs_err;
        }

        // Step 6: Write new quantized payload back to src
        char * row_dst = static_cast<char *>(src) + r * row_size;
        std::memcpy(row_dst, quant_scratch.data(), row_size);

        ++rows_processed;
    }

    return {error_code::ok, rows_processed, n_per_row,
            std::sqrt(total_sum_sq / total), total_max_err};
}

update_result roundtrip_same_type(
        ggml_type type, const void * src, int64_t n_rows, int64_t n_per_row,
        void * dst, const sr_config & sr) {
    if (!src || !dst) return {error_code::err_null_pointer, 0, n_per_row, 0, 0};

    auto ec = validate(type, n_rows, n_per_row);
    if (ec != error_code::ok) return {ec, 0, n_per_row, 0, 0};

    // F32: identity
    if (type == GGML_TYPE_F32) {
        size_t row_size = ggml_row_size(GGML_TYPE_F32, n_per_row);
        for (int64_t r = 0; r < n_rows; ++r) {
            std::memcpy(
                static_cast<char *>(dst) + r * row_size,
                static_cast<const char *>(src) + r * row_size,
                row_size);
        }
        return {error_code::ok, n_rows, n_per_row, 0, 0};
    }

    const auto * traits = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);

    if (!traits || !traits->to_float)
        return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};

    std::vector<float> f32_row(n_per_row);
    size_t dst_row_size = ggml_row_size(type, n_per_row);
    double total_sum_sq = 0;
    double total_max_err = 0;

    for (int64_t r = 0; r < n_rows; ++r) {
        const char * row_src = static_cast<const char *>(src) +
                               r * ggml_row_size(type, n_per_row);
        traits->to_float(row_src, f32_row.data(), n_per_row);

        char * row_dst = static_cast<char *>(dst) + r * dst_row_size;

        if (sr.stochastic) {
            auto qr = quantize_rows(type, f32_row.data(), n_per_row, 1, n_per_row,
                                     row_dst, sr);
            // Stochastic mode may return err_unsupported_type for block-coupled formats
            if (qr.ec != error_code::ok) return qr;
        } else {
            if (!traits_cpu || !traits_cpu->from_float)
                return {error_code::err_unsupported_type, 0, n_per_row, 0, 0};
            traits_cpu->from_float(f32_row.data(), row_dst, n_per_row);
        }

        std::vector<float> dequant(n_per_row);
        traits->to_float(row_dst, dequant.data(), n_per_row);

        for (int64_t i = 0; i < n_per_row; ++i) {
            double err = static_cast<double>(dequant[i]) -
                         static_cast<double>(f32_row[i]);
            total_sum_sq += err * err;
            double abs_err = std::fabs(err);
            if (abs_err > total_max_err) total_max_err = abs_err;
        }
    }

    int64_t total = 0;
    checked_mul_int64(n_rows, n_per_row, total);
    return {error_code::ok, n_rows, n_per_row,
            std::sqrt(total_sum_sq / total), total_max_err};
}

} // namespace llama_train_quant
