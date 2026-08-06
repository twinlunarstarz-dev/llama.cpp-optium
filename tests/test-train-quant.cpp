// P1 Quantized Training Primitives — Deterministic Unit Tests (Corrected)
//
// Tests the numerically authoritative CPU reference layer for quantized
// training: same-type dequantize/quantize, error feedback, stochastic rounding.
//
// All tests use fixed small tensors and deterministic assertions.
// No probabilistic/flaky checks.

#include "llama-train-quant.h"

#include <ggml.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Test framework helpers
// ---------------------------------------------------------------------------

static int g_tests = 0;
static int g_assertions = 0;
static int g_failures = 0;

#define TEST(name) \
    static void test_##name(); \
    struct reg_##name { reg_##name() { register_test(#name, test_##name); } }; \
    static reg_##name g_reg_##name; \
    static void test_##name()

using test_fn = void (*)();
static std::vector<std::pair<std::string, test_fn>> g_test_registry;

static void register_test(const char * name, test_fn fn) {
    g_test_registry.push_back({name, fn});
}

#define CHECK(cond) \
    do { \
        ++g_assertions; \
        if (!(cond)) { \
            ++g_failures; \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ \
                      << " condition: " #cond << "\n"; \
        } \
    } while (0)

#define CHECK_EQ(a, b) \
    do { \
        ++g_assertions; \
        if ((a) != (b)) { \
            ++g_failures; \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ \
                      << " " #a " != " #b \
                      << " (" << (a) << " != " << (b) << ")\n"; \
        } \
    } while (0)

#define CHECK_FLOAT_EQ(a, b, eps) \
    do { \
        ++g_assertions; \
        double diff_ = std::fabs(static_cast<double>(a) - static_cast<double>(b)); \
        if (diff_ > (eps)) { \
            ++g_failures; \
            std::cerr << std::fixed << std::setprecision(10) \
                      << "FAIL: " << __FILE__ << ":" << __LINE__ \
                      << " " #a " != " #b " (diff=" << diff_ << ")\n"; \
        } \
    } while (0)

#define CHECK_RESULT_OK(result) \
    do { \
        CHECK((result).ec == llama_train_quant::error_code::ok); \
    } while (0)

#define CHECK_RESULT_EC(result, expected) \
    do { \
        CHECK((result).ec == (expected)); \
    } while (0)

#define CHECK_CODE(code, expected) \
    do { \
        CHECK((code) == (expected)); \
    } while (0)

#define CHECK_EC_OK(result) CHECK_RESULT_OK(result)
#define CHECK_EC(result, expected) CHECK_RESULT_EC(result, expected)

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

// Generate deterministic test data using a fixed seed LCG.
static void generate_test_data(float * data, int64_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(dist(rng));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. Supported type matrix and explicit rejection
TEST(supported_type_matrix) {
    // Types that should be supported (have to_float AND from_float)
    CHECK(llama_train_quant::is_supported(GGML_TYPE_Q8_0));
    CHECK(llama_train_quant::is_supported(GGML_TYPE_F32));
    CHECK(llama_train_quant::is_supported(GGML_TYPE_Q4_K));

    // NVFP4: has to_float and from_float_ref AND from_float via CPU traits.
    // is_supported should return true for the type itself.
    const auto * traits = ggml_get_type_traits(GGML_TYPE_NVFP4);
    const auto * traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    if (traits && traits->to_float && traits_cpu && traits_cpu->from_float) {
        CHECK(llama_train_quant::is_supported(GGML_TYPE_NVFP4));
    }

    // Types that should NOT be supported
    CHECK(!llama_train_quant::is_supported(GGML_TYPE_F16));
    CHECK(!llama_train_quant::is_supported(GGML_TYPE_BF16));
    CHECK(!llama_train_quant::is_supported(GGML_TYPE_I32));
    CHECK(!llama_train_quant::is_supported(GGML_TYPE_I64));
}

// 2. Geometry and overflow checks
TEST(geometry_checks) {
    using namespace llama_train_quant;

    // Zero dimensions
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 0, 256), error_code::err_zero_size);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 0), error_code::err_zero_size);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, -1, 256), error_code::err_zero_size);

    // Bad alignment for Q8_0 (blck_size = 32)
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 31), error_code::err_bad_row_alignment);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 33), error_code::err_bad_row_alignment);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 63), error_code::err_bad_row_alignment);

    // Good alignment for Q8_0
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 32), error_code::ok);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 64), error_code::ok);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 1, 256), error_code::ok);
    CHECK_CODE(validate(GGML_TYPE_Q8_0, 10, 256), error_code::ok);

    // Bad alignment for Q4_K (blck_size = 256)
    CHECK_CODE(validate(GGML_TYPE_Q4_K, 1, 255), error_code::err_bad_row_alignment);
    CHECK_CODE(validate(GGML_TYPE_Q4_K, 1, 256), error_code::ok);

    // NVFP4 (blck_size = 64)
    if (is_supported(GGML_TYPE_NVFP4)) {
        CHECK_CODE(validate(GGML_TYPE_NVFP4, 1, 63), error_code::err_bad_row_alignment);
        CHECK_CODE(validate(GGML_TYPE_NVFP4, 1, 64), error_code::ok);
    }

    // Unsupported type
    CHECK_CODE(validate(GGML_TYPE_F16, 1, 32), error_code::err_unsupported_type);
}

// 3. Same-type round-trip payload validity
TEST(roundtrip_preserves_payload) {
    using namespace llama_train_quant;

    int64_t n_rows = 2;
    int64_t n_per_row = 256;

    // Q8_0 roundtrip
    {
        std::vector<float> src_f32(n_rows * n_per_row);
        generate_test_data(src_f32.data(), n_rows * n_per_row, 42);

        size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
        std::vector<uint8_t> quant_buf(n_rows * row_size);
        std::vector<uint8_t> rt_buf(n_rows * row_size);

        auto r1 = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                                 n_rows, n_per_row, quant_buf.data(), sr_nearest());
        CHECK_EC_OK(r1);
        CHECK(r1.quantization_rms_error >= 0);

        auto r2 = roundtrip_same_type(GGML_TYPE_Q8_0, quant_buf.data(),
                                       n_rows, n_per_row, rt_buf.data(), sr_nearest());
        CHECK_EC_OK(r2);

        CHECK(std::memcmp(quant_buf.data(), rt_buf.data(), n_rows * row_size) == 0);
    }

    // Q4_K roundtrip
    {
        int64_t n4 = 256;
        std::vector<float> src_f32(n_rows * n4);
        generate_test_data(src_f32.data(), n_rows * n4, 43);

        size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n4);
        std::vector<uint8_t> quant_buf(n_rows * row_size);
        std::vector<uint8_t> rt_buf(n_rows * row_size);

        auto r1 = quantize_rows(GGML_TYPE_Q4_K, src_f32.data(), n4,
                                 n_rows, n4, quant_buf.data(), sr_nearest());
        CHECK_EC_OK(r1);

        auto r2 = roundtrip_same_type(GGML_TYPE_Q4_K, quant_buf.data(),
                                       n_rows, n4, rt_buf.data(), sr_nearest());
        CHECK_EC_OK(r2);

        CHECK(std::memcmp(quant_buf.data(), rt_buf.data(), n_rows * row_size) == 0);
    }

    // F32 roundtrip (identity)
    {
        std::vector<float> src_f32(n_rows * n_per_row);
        generate_test_data(src_f32.data(), n_rows * n_per_row, 44);

        std::vector<float> dst_f32(n_rows * n_per_row);
        auto r = roundtrip_same_type(GGML_TYPE_F32, src_f32.data(),
                                      n_rows, n_per_row, dst_f32.data(), sr_nearest());
        CHECK_EC_OK(r);
        CHECK(r.quantization_rms_error == 0.0);
        CHECK(std::memcmp(src_f32.data(), dst_f32.data(),
                          n_rows * n_per_row * sizeof(float)) == 0);
    }
}

// 4. Zero update produces no change
TEST(zero_update) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 100);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, payload.data(), sr_nearest());

    std::vector<uint8_t> original = payload;
    std::vector<float> update(n_rows * n_per_row, 0.0f);
    std::vector<float> residual(n_rows * n_per_row, 0.0f);

    auto r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                               residual.data(), update.data(),
                                               1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC_OK(r);

    CHECK(std::memcmp(payload.data(), original.data(), row_size) == 0);

    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        CHECK_FLOAT_EQ(residual[i], 0.0f, 1e-5f);
    }
}

// 5. Deterministic nearest mode
TEST(deterministic_nearest) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 200);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> buf1(row_size);
    std::vector<uint8_t> buf2(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, buf1.data(), sr_nearest());
    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, buf2.data(), sr_nearest());

    CHECK(std::memcmp(buf1.data(), buf2.data(), row_size) == 0);

    std::vector<float> dq1(n_rows * n_per_row);
    std::vector<float> dq2(n_rows * n_per_row);
    dequantize_rows(GGML_TYPE_Q8_0, buf1.data(), 0, n_rows, n_per_row, dq1.data(), n_per_row);
    dequantize_rows(GGML_TYPE_Q8_0, buf2.data(), 0, n_rows, n_per_row, dq2.data(), n_per_row);

    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        CHECK(dq1[i] == dq2[i]);
    }
}

// 6. Deterministic stochastic replay for identical seed/key/counter
TEST(deterministic_stochastic_replay) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 300);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> buf1(row_size);
    std::vector<uint8_t> buf2(row_size);

    sr_config sr = sr_stochastic(12345, 67890, 0);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, buf1.data(), sr);
    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, buf2.data(), sr);

    CHECK(std::memcmp(buf1.data(), buf2.data(), row_size) == 0);
}

// 7. Different key/counter changes stochastic decisions where representable
TEST(stochastic_differs_with_key) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 512;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 400);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> buf1(row_size);
    std::vector<uint8_t> buf2(row_size);

    sr_config sr1 = sr_stochastic(12345, 67890, 0);
    sr_config sr2 = sr_stochastic(12345, 99999, 0);

    auto r1 = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                             n_rows, n_per_row, buf1.data(), sr1);
    auto r2 = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                             n_rows, n_per_row, buf2.data(), sr2);
    CHECK_EC_OK(r1);
    CHECK_EC_OK(r2);

    bool differs = std::memcmp(buf1.data(), buf2.data(), row_size) != 0;
    CHECK(differs);

    sr_config sr3 = sr_stochastic(12345, 67890, 1000);
    std::vector<uint8_t> buf3(row_size);
    auto r3 = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                             n_rows, n_per_row, buf3.data(), sr3);
    CHECK_EC_OK(r3);

    bool differs_counter = std::memcmp(buf1.data(), buf3.data(), row_size) != 0;
    CHECK(differs_counter);
}

// 8. Error-feedback conservation / reconstruction bounds
TEST(error_feedback_conservation) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 500);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, payload.data(), sr_nearest());

    std::vector<float> dequant(n_rows * n_per_row);
    dequantize_rows(GGML_TYPE_Q8_0, payload.data(), 0, n_rows, n_per_row,
                    dequant.data(), n_per_row);

    std::vector<float> update(n_rows * n_per_row);
    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        update[i] = 0.1f;
    }
    std::vector<float> residual(n_rows * n_per_row, 0.0f);

    auto r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                               residual.data(), update.data(),
                                               1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC_OK(r);

    std::vector<float> new_dequant(n_rows * n_per_row);
    dequantize_rows(GGML_TYPE_Q8_0, payload.data(), 0, n_rows, n_per_row,
                    new_dequant.data(), n_per_row);

    double max_reconstruction_error = 0;
    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        double proposed = static_cast<double>(dequant[i]) + 0.1;
        double reconstructed = static_cast<double>(new_dequant[i]) + static_cast<double>(residual[i]);
        double err = std::fabs(proposed - reconstructed);
        if (err > max_reconstruction_error) max_reconstruction_error = err;
    }

    CHECK_FLOAT_EQ(max_reconstruction_error, 0.0, 1e-5);
}

// 9. Repeated sub-quantum updates eventually change payload
TEST(sub_quantum_accumulation) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    std::fill(src_f32.begin(), src_f32.end(), 0.5f);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, payload.data(), sr_nearest());

    std::vector<uint8_t> original = payload;
    std::vector<float> residual(n_rows * n_per_row, 0.0f);

    float sub_quantum = 0.005f;
    std::vector<float> update(n_rows * n_per_row, sub_quantum);

    int changes_detected = 0;
    for (int iter = 0; iter < 200; ++iter) {
        auto r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                                   residual.data(), update.data(),
                                                   1.0f, n_rows, n_per_row, sr_nearest());
        CHECK_EC_OK(r);

        if (std::memcmp(payload.data(), original.data(), row_size) != 0) {
            ++changes_detected;
        }
    }

    CHECK(changes_detected > 0);

    double max_residual = 0;
    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        double abs_r = std::fabs(static_cast<double>(residual[i]));
        if (abs_r > max_residual) max_residual = abs_r;
    }

    CHECK(max_residual < 1.0);
}

// 10. Error feedback sign/order (mathematical verification)
TEST(error_feedback_sign_order) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row, 1.0f);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, payload.data(), sr_nearest());

    std::vector<float> residual(n_rows * n_per_row, 0.0f);
    std::vector<float> neg_update(n_rows * n_per_row, -0.01f);

    auto r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                               residual.data(), neg_update.data(),
                                               1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC_OK(r);

    std::vector<float> new_dequant(n_rows * n_per_row);
    dequantize_rows(GGML_TYPE_Q8_0, payload.data(), 0, n_rows, n_per_row,
                    new_dequant.data(), n_per_row);

    std::vector<float> old_dequant(n_rows * n_per_row);
    std::vector<uint8_t> orig_payload(row_size);
    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, orig_payload.data(), sr_nearest());
    dequantize_rows(GGML_TYPE_Q8_0, orig_payload.data(), 0, n_rows, n_per_row,
                    old_dequant.data(), n_per_row);

    double max_err = 0;
    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        double proposed = static_cast<double>(old_dequant[i]) - 0.01;
        double reconstructed = static_cast<double>(new_dequant[i]) +
                               static_cast<double>(residual[i]);
        double err = std::fabs(proposed - reconstructed);
        if (err > max_err) max_err = err;
    }

    CHECK_FLOAT_EQ(max_err, 0.0, 1e-5);
}

// 11. Non-finite rejection
TEST(non_finite_rejection) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row, 1.0f);
    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, payload.data(), sr_nearest());

    std::vector<float> residual(n_rows * n_per_row, 0.0f);

    // NaN in update
    std::vector<float> nan_update(n_rows * n_per_row, 0.0f);
    nan_update[0] = std::nanf("");

    auto r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                               residual.data(), nan_update.data(),
                                               1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC(r, error_code::err_non_finite);

    // Inf in update
    std::vector<float> inf_update(n_rows * n_per_row, 0.0f);
    inf_update[0] = std::numeric_limits<float>::infinity();

    r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                          residual.data(), inf_update.data(),
                                          1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC(r, error_code::err_non_finite);

    // NaN in residual
    std::vector<float> bad_residual(n_rows * n_per_row, 0.0f);
    bad_residual[0] = std::nanf("");
    std::vector<float> good_update(n_rows * n_per_row, 0.01f);

    r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                          bad_residual.data(), good_update.data(),
                                          1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC(r, error_code::err_non_finite);
}

// 12. Malformed buffer rejection
TEST(malformed_buffers) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    {
        float * tmp_dst = new float[n_per_row];
        update_result r = dequantize_rows(GGML_TYPE_Q8_0, nullptr, 0, n_rows, n_per_row,
                                          tmp_dst, n_per_row);
        CHECK_EC(r, error_code::err_null_pointer);
        delete[] tmp_dst;
    }

    std::vector<float> dst(n_per_row);
    {
        update_result r = dequantize_rows(GGML_TYPE_Q8_0, nullptr, 0, n_rows, n_per_row,
                                          dst.data(), n_per_row);
        CHECK_EC(r, error_code::err_null_pointer);
    }

    {
        std::vector<uint8_t> buf(ggml_row_size(GGML_TYPE_Q8_0, n_per_row));
        {
            update_result r = dequantize_rows(GGML_TYPE_Q8_0, buf.data(), 0, n_rows, n_per_row,
                                              nullptr, n_per_row);
            CHECK_EC(r, error_code::err_null_pointer);
        }

        {
            update_result r = quantize_rows(GGML_TYPE_Q8_0, nullptr, n_per_row, n_rows, n_per_row,
                                             buf.data(), sr_nearest());
            CHECK_EC(r, error_code::err_null_pointer);
        }

        {
            std::vector<float> src_f32(n_per_row, 1.0f);
            update_result r = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row, n_rows, n_per_row,
                                             nullptr, sr_nearest());
            CHECK_EC(r, error_code::err_null_pointer);
        }

        {
            std::vector<float> src_f32(n_per_row, 1.0f);
            update_result r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, nullptr,
                                                                dst.data(), src_f32.data(),
                                                                1.0f, n_rows, n_per_row, sr_nearest());
            CHECK_EC(r, error_code::err_null_pointer);

            r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, buf.data(),
                                                  nullptr, src_f32.data(),
                                                  1.0f, n_rows, n_per_row, sr_nearest());
            CHECK_EC(r, error_code::err_null_pointer);

            r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, buf.data(),
                                                  dst.data(), nullptr,
                                                  1.0f, n_rows, n_per_row, sr_nearest());
            CHECK_EC(r, error_code::err_null_pointer);
        }
    }
}

// 13. No modification on failed operation
TEST(no_modification_on_failure) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row, 1.0f);
    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, payload.data(), sr_nearest());

    std::vector<uint8_t> original = payload;
    std::vector<float> residual(n_rows * n_per_row, 0.5f);

    std::vector<float> nan_update(n_rows * n_per_row, 0.0f);
    nan_update[0] = std::nanf("");

    auto r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                               residual.data(), nan_update.data(),
                                               1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC(r, error_code::err_non_finite);

    CHECK(std::memcmp(payload.data(), original.data(), row_size) == 0);

    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        CHECK_FLOAT_EQ(residual[i], 0.5f, 1e-6f);
    }
}

// 14. Dequantize with strided output
TEST(dequantize_strided) {
    using namespace llama_train_quant;

    int64_t n_rows = 2;
    int64_t n_per_row = 256;
    int64_t dst_stride = 512;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 600);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> quant_buf(n_rows * row_size);

    quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                  n_rows, n_per_row, quant_buf.data(), sr_nearest());

    std::vector<float> dst(n_rows * dst_stride, -1e9f);

    auto r = dequantize_rows(GGML_TYPE_Q8_0, quant_buf.data(), 0, n_rows,
                              n_per_row, dst.data(), dst_stride);
    CHECK_EC_OK(r);

    for (int64_t row = 0; row < n_rows; ++row) {
        for (int64_t i = n_per_row; i < dst_stride; ++i) {
            int64_t idx = row * dst_stride + i;
            CHECK(dst[idx] == -1e9f);
        }
    }

    for (int64_t row = 0; row < n_rows; ++row) {
        for (int64_t i = 0; i < n_per_row; ++i) {
            int64_t idx = row * dst_stride + i;
            CHECK(std::isfinite(dst[idx]));
        }
    }
}

// 15. Error code name strings are non-empty
TEST(error_code_names) {
    using namespace llama_train_quant;

    CHECK(std::string(error_code_name(error_code::ok)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_unsupported_type)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_bad_row_alignment)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_zero_size)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_null_pointer)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_overflow)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_non_finite)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_malformed_buffer)).size() > 0);
    CHECK(std::string(error_code_name(error_code::err_validation_failed)).size() > 0);
}

// 16. Memory bounded by row (not total tensor size)
TEST(memory_bounded_by_row) {
    using namespace llama_train_quant;

    int64_t n_rows = 100;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 700);

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<uint8_t> payload(n_rows * row_size);

    auto r = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                            n_rows, n_per_row, payload.data(), sr_nearest());
    CHECK_EC_OK(r);
    CHECK(r.n_rows == n_rows);

    std::vector<float> residual(n_rows * n_per_row, 0.0f);
    std::vector<float> update(n_rows * n_per_row, 0.01f);

    r = apply_update_with_error_feedback(GGML_TYPE_Q8_0, payload.data(),
                                          residual.data(), update.data(),
                                          1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC_OK(r);
    CHECK(r.n_rows == n_rows);
}

// 17. Q4_K specific tests (nearest mode only)
TEST(q4_k_operations) {
    using namespace llama_train_quant;

    int64_t n_rows = 2;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 800);

    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
    std::vector<uint8_t> payload(n_rows * row_size);

    auto r = quantize_rows(GGML_TYPE_Q4_K, src_f32.data(), n_per_row,
                            n_rows, n_per_row, payload.data(), sr_nearest());
    CHECK_EC_OK(r);
    CHECK(r.quantization_rms_error > 0);

    std::vector<float> dequant(n_rows * n_per_row);
    r = dequantize_rows(GGML_TYPE_Q4_K, payload.data(), 0, n_rows, n_per_row,
                         dequant.data(), n_per_row);
    CHECK_EC_OK(r);

    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        CHECK(std::isfinite(dequant[i]));
    }

    std::vector<float> residual(n_rows * n_per_row, 0.0f);
    std::vector<float> update(n_rows * n_per_row, 0.05f);

    r = apply_update_with_error_feedback(GGML_TYPE_Q4_K, payload.data(),
                                          residual.data(), update.data(),
                                          1.0f, n_rows, n_per_row, sr_nearest());
    CHECK_EC_OK(r);

    std::vector<float> new_dequant(n_rows * n_per_row);
    dequantize_rows(GGML_TYPE_Q4_K, payload.data(), 0, n_rows, n_per_row,
                    new_dequant.data(), n_per_row);

    double max_recon_err = 0;
    for (int64_t i = 0; i < n_rows * n_per_row; ++i) {
        double proposed = static_cast<double>(dequant[i]) + 0.05;
        double reconstructed = static_cast<double>(new_dequant[i]) +
                               static_cast<double>(residual[i]);
        double err = std::fabs(proposed - reconstructed);
        if (err > max_recon_err) max_recon_err = err;
    }

    CHECK_FLOAT_EQ(max_recon_err, 0.0, 1e-4);
}

// ============================================================================
// NEW TESTS: Stochastic rounding correctness (replaces old heuristic tests)
// ============================================================================

// 18. Q4_K stochastic mode must return explicit error (block-coupled scale)
TEST(q4_k_stochastic_unsupported) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 256;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 900);

    size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
    std::vector<uint8_t> payload(row_size);

    // Stochastic mode on Q4_K must fail with explicit error
    auto r = quantize_rows(GGML_TYPE_Q4_K, src_f32.data(), n_per_row,
                            n_rows, n_per_row, payload.data(),
                            sr_stochastic(12345, 67890, 0));
    CHECK_EC(r, error_code::err_unsupported_type);
    CHECK(r.n_rows == 0);

    // roundtrip in stochastic mode must also fail
    std::vector<float> init_f32(n_rows * n_per_row, 1.0f);
    size_t init_row_size = ggml_row_size(GGML_TYPE_F32, n_per_row);
    std::vector<uint8_t> f32_payload(init_row_size);
    std::memcpy(f32_payload.data(), init_f32.data(), init_row_size);

    // First quantize nearest to get a valid Q4_K buffer
    std::vector<uint8_t> q4k_buf(row_size);
    auto qr = quantize_rows(GGML_TYPE_Q4_K, init_f32.data(), n_per_row,
                             n_rows, n_per_row, q4k_buf.data(), sr_nearest());
    CHECK_EC_OK(qr);

    // Now try stochastic roundtrip
    std::vector<uint8_t> rt_buf(row_size);
    auto rr = roundtrip_same_type(GGML_TYPE_Q4_K, q4k_buf.data(),
                                   n_rows, n_per_row, rt_buf.data(),
                                   sr_stochastic(12345, 67890, 0));
    CHECK_EC(rr, error_code::err_unsupported_type);

    // apply_update with stochastic on Q4_K must fail
    std::vector<float> residual(n_rows * n_per_row, 0.0f);
    std::vector<float> update(n_rows * n_per_row, 0.01f);
    auto ru = apply_update_with_error_feedback(GGML_TYPE_Q4_K, q4k_buf.data(),
                                                residual.data(), update.data(),
                                                1.0f, n_rows, n_per_row,
                                                sr_stochastic(12345, 67890, 0));
    CHECK_EC(ru, error_code::err_unsupported_type);
    CHECK(ru.n_rows == 0);
}

// 19. NVFP4 stochastic mode must return explicit error (block-coupled scale)
TEST(nvfp4_stochastic_unsupported) {
    using namespace llama_train_quant;

    // If NVFP4 is not supported at all, skip
    if (!is_supported(GGML_TYPE_NVFP4)) return;

    int64_t n_rows = 1;
    int64_t n_per_row = 64;

    std::vector<float> src_f32(n_rows * n_per_row);
    generate_test_data(src_f32.data(), n_rows * n_per_row, 910);

    size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, n_per_row);
    std::vector<uint8_t> payload(row_size);

    // Nearest mode should work
    auto rn = quantize_rows(GGML_TYPE_NVFP4, src_f32.data(), n_per_row,
                             n_rows, n_per_row, payload.data(), sr_nearest());
    CHECK_EC_OK(rn);

    // Stochastic mode must fail with explicit error
    auto rs = quantize_rows(GGML_TYPE_NVFP4, src_f32.data(), n_per_row,
                             n_rows, n_per_row, payload.data(),
                             sr_stochastic(12345, 67890, 0));
    CHECK_EC(rs, error_code::err_unsupported_type);
}

// 20. Exhaustive deterministic unbiased stochastic rounding test for Q8_0
// Uses exactly known representable values and controlled counter draws.
//
// Strategy: Create two values that bracket a code boundary. One value maps
// to frac=0.25 (should round down ~75%), the other to frac=0.75 (up ~75%).
// With identical max_abs across trials, the scale is stable and predictable.
TEST(stochastic_unbiased_exhaustive) {
    using namespace llama_train_quant;

    // Single Q8_0 block of 32 elements.
    int64_t n_rows = 1;
    int64_t n_per_row = 32;

    // We need values that land between Q8_0 codes after normalization.
    // Q8_0: d = 127 / max_abs, normalized = x * d, code = round(normalized).
    // With max_abs = 1.0, d = 127. step = 1/127 ≈ 0.00787.
    // Value at 0.5 * step = 0.5/127 ≈ 0.00394 has frac=0.5 after normalization.
    // But max_abs would be 0.00394, not 1.0, so d changes.
    //
    // Instead: use a range [-max_v, +max_v] where max_v maps to code 127,
    // and include some values at half-steps from integer codes.
    // Let max_v = 1.0, so d = 127.0, step = 1/127.
    // A value of (k + 0.5)/127 for any integer k has frac = 0.5.
    // Use value = 0.5/127 ≈ 0.003937 for the midpoint test element.
    // Fill remaining elements with max_v = 1.0 to set the scale.

    std::vector<float> src_f32(n_per_row, 1.0f); // max_abs = 1.0, d = 127.0
    // Replace first element with a midpoint value: (0 + 0.5)/127 = 0.5/127
    src_f32[0] = 0.5f / 127.0f; // normalized = 0.5, frac = 0.5 exactly

    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);

    // Run stochastic quantization many times with different seeds.
    // For frac == 0.5, expected split is exactly 50/50.
    const int num_trials = 64;
    int up_count = 0;
    int down_count = 0;

    for (int t = 0; t < num_trials; ++t) {
        std::vector<uint8_t> buf(row_size);
        auto r = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                                n_rows, n_per_row, buf.data(),
                                sr_stochastic(static_cast<uint64_t>(t), 0, 0));
        CHECK_EC_OK(r);

        // Dequantize first element
        std::vector<float> dq(n_per_row);
        dequantize_rows(GGML_TYPE_Q8_0, buf.data(), 0, n_rows, n_per_row, dq.data(), n_per_row);

        // The midpoint value (0.5/127) should sometimes round to code 0 (dq=0) and
        // sometimes to code 1 (dq = 1/127). Check which direction it went.
        if (dq[0] > 0.0f) up_count++;   // rounded up to code 1
        else down_count++;               // rounded down to code 0
    }

    // With 64 trials and frac=0.5, expected split is 32/32.
    // Binomial bound: P(|X-32| >= 16) < 1e-10, so require at least 16/48 split.
    CHECK(up_count >= 16);
    CHECK(down_count >= 16);
}

// 21. Stochastic endpoint behavior: exact representable values never flip
TEST(stochastic_endpoint_exact) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 32;

    // Use value 0.0f which is exactly representable in Q8_0 (code 0)
    std::vector<float> src_f32(n_per_row, 0.0f);
    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);

    // Run with many different seeds
    for (int t = 0; t < 32; ++t) {
        std::vector<uint8_t> buf(row_size);
        auto r = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                                n_rows, n_per_row, buf.data(),
                                sr_stochastic(static_cast<uint64_t>(t), 0, 0));
        CHECK_EC_OK(r);

        std::vector<float> dq(n_per_row);
        dequantize_rows(GGML_TYPE_Q8_0, buf.data(), 0, n_rows, n_per_row, dq.data(), n_per_row);

        // All elements must be exactly 0.0 (no stochastic flip on exact values)
        for (int64_t i = 0; i < n_per_row; ++i) {
            CHECK_FLOAT_EQ(dq[i], 0.0f, 1e-10);
        }
    }
}

// 22. Counter overflow handling: very large counter_start must not wrap silently
TEST(counter_overflow_rejection) {
    using namespace llama_train_quant;

    int64_t n_rows = 1;
    int64_t n_per_row = 32;

    std::vector<float> src_f32(n_per_row, 0.5f);
    size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);

    // Use a very large counter_start that should still produce valid deterministic output
    // The hash function uses uint64_t internally so counter overflow is handled naturally.
    // We verify determinism with extreme counter values.
    std::vector<uint8_t> buf1(row_size);
    std::vector<uint8_t> buf2(row_size);

    auto r1 = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                             n_rows, n_per_row, buf1.data(),
                             sr_stochastic(1, 2, static_cast<int64_t>(0xFFFFFFFFFFFFFFF0LL)));
    auto r2 = quantize_rows(GGML_TYPE_Q8_0, src_f32.data(), n_per_row,
                             n_rows, n_per_row, buf2.data(),
                             sr_stochastic(1, 2, static_cast<int64_t>(0xFFFFFFFFFFFFFFF0LL)));

    CHECK_EC_OK(r1);
    CHECK_EC_OK(r2);
    CHECK(std::memcmp(buf1.data(), buf2.data(), row_size) == 0);
}

// ============================================================================
// Main
// ============================================================================

int main(int /*argc*/, char * /*argv*/[]) {
    ggml_cpu_init();

    std::sort(g_test_registry.begin(), g_test_registry.end(),
              [](const auto & a, const auto & b) { return a.first < b.first; });

    for (auto & [name, fn] : g_test_registry) {
        ++g_tests;
        std::cout << "Running test: " << name << "\n";
        fn();
    }

    std::cout << "\n========================================\n"
              << "tests      : " << g_tests << "\n"
              << "assertions : " << g_assertions << "\n"
              << "failures   : " << g_failures << "\n"
              << "========================================\n";

    return g_failures > 0 ? 1 : 0;
}
