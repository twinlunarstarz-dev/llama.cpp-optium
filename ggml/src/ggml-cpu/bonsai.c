#include "quants.h"
#include "ggml-quants.h"

#include <assert.h>

void quantize_row_pq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_pq2_0_ref(x, (block_pq2_0 *) y, k);
}

void quantize_row_ptq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_ptq1_0_ref(x, (block_ptq1_0 *) y, k);
}

static float bonsai_dot_q8_0(const float * x, const block_q8_0 * y, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const block_q8_0 * block = y + i / QK8_0;
        const int j = (int) (i % QK8_0);
        sum += x[i] * ((float) block->qs[j]) * ggml_fp16_to_fp32(block->d);
    }
    return sum;
}

void ggml_vec_dot_pq2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_PQ2_0 == 0);
    const block_pq2_0 * x = (const block_pq2_0 *) ((const char *) vx + bx);
    const block_q8_0 * y = (const block_q8_0 *) ((const char *) vy + by);
    float values[QK_PQ2_0];
    *s = 0.0f;
    for (int64_t block = 0; block < n / QK_PQ2_0; ++block) {
        dequantize_row_pq2_0(x + block, values, QK_PQ2_0);
        *s += bonsai_dot_q8_0(values, y + (block * QK_PQ2_0) / QK8_0, QK_PQ2_0);
    }
    (void) bs;
}

void ggml_vec_dot_ptq1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) nrc;
    assert(n % QK_PTQ1_0 == 0);
    const block_ptq1_0 * x = (const block_ptq1_0 *) ((const char *) vx + bx);
    const block_q8_0 * y = (const block_q8_0 *) ((const char *) vy + by);
    float values[QK_PTQ1_0];
    *s = 0.0f;
    for (int64_t block = 0; block < n / QK_PTQ1_0; ++block) {
        dequantize_row_ptq1_0(x + block, values, QK_PTQ1_0);
        *s += bonsai_dot_q8_0(values, y + (block * QK_PTQ1_0) / QK8_0, QK_PTQ1_0);
    }
    (void) bs;
}
