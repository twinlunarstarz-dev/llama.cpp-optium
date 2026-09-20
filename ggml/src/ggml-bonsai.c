#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>

void quantize_row_pq2_0_ref(const float * GGML_RESTRICT x, block_pq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ2_0 == 0);
    const int64_t nb = k / QK_PQ2_0;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PQ2_0; ++j) {
            amax = fmaxf(amax, fabsf(x[j]));
        }
        const float d = amax;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        y[i].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < QK_PQ2_0/4; ++j) {
            y[i].qs[j] = 0;
        }
        for (int j = 0; j < QK_PQ2_0; ++j) {
            int q = (int) lroundf(x[j] * id) + 1;
            q = q < 0 ? 0 : q > 3 ? 3 : q;
            y[i].qs[j/4] |= (uint8_t) q << ((j % 4) * 2);
        }
        x += QK_PQ2_0;
    }
}

void dequantize_row_pq2_0(const block_pq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ2_0 == 0);
    const int64_t nb = k / QK_PQ2_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK_PQ2_0; ++j) {
            const uint8_t q = (x[i].qs[j/4] >> ((j % 4) * 2)) & 3;
            y[j] = ((int) q - 1) * d;
        }
        y += QK_PQ2_0;
    }
}

static const size_t ptq1_0_stages[3] = { 32, 16, 8 };

void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PTQ1_0; ++j) {
            amax = fmaxf(amax, fabsf(x[j]));
        }
        const float d = amax;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        y[i].d = ggml_fp32_to_fp16(d);
        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) {
                        int xi = (int) lroundf(x[m + n*c] * id) + 1;
                        xi = xi < 0 ? 0 : xi > 2 ? 2 : xi;
                        q = (uint8_t) (q * 3 + xi);
                    }
                    y[i].qs[j + m] = (uint8_t) (((uint16_t) q * 256 + 242) / 243);
                }
                x += 5*c;
            }
        }
        for (size_t h = 0; h < sizeof(y->qh); ++h) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = (int) lroundf(x[h + m*sizeof(y->qh)] * id) + 1;
                xi = xi < 0 ? 0 : xi > 2 ? 2 : xi;
                q = (uint8_t) (q * 3 + xi);
            }
            q = (uint8_t) (q * 3);
            y[i].qh[h] = (uint8_t) (((uint16_t) q * 256 + 242) / 243);
        }
        x += 4*sizeof(y->qh);
    }
}

void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;
    const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };
    for (int64_t i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t n = 0; n < 5; ++n) {
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t q = (uint8_t) (x[i].qs[j + m] * pow3[n]);
                        const int xi = ((uint16_t) q * 3) >> 8;
                        *y++ = (float) (xi - 1) * d;
                    }
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t q = (uint8_t) (x[i].qh[h] * pow3[n]);
                const int xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

size_t quantize_pq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_PQ2_0, n_per_row);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_pq2_0_ref(src, (block_pq2_0 *) qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_PTQ1_0, n_per_row);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_ptq1_0_ref(src, (block_ptq1_0 *) qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}
