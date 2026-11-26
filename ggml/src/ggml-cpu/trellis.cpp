#include "trellis.h"
#include "ops.h"

#include <algorithm>
#include <cstring>

// trellis fork: rows of a coded embedding (ggml_trellis_get_rows); each value has its own 16-bit window, so a row decodes without its 16-row group
static inline int ggml_trellis_value(const uint32_t * tile, int k, int p) {
    const int total = 256 * k;
    const int start = ((p + 1) * k - 16 + total) % total;
    const int w = start >> 5, o = start & 31;
    const uint64_t pair = ((uint64_t) tile[w] << 32) | tile[(w + 1) % (8 * k)];
    const uint32_t state = (uint32_t) (pair >> (48 - o)) & 0xFFFFu;
    const uint32_t x = state * 0x83DCD12Du;
    return (int) (((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126;
}

// per thread: the 2 L floats of an inverse whole-slice Hadamard
static int64_t ggml_trellis_get_rows_scratch(const int32_t * desc, int64_t K) {
    return (2 * K / desc[5] + 15) / 16 * 16 + (int64_t) CACHE_LINE_SIZE_F32;
}

size_t ggml_compute_forward_trellis_get_rows_work_size(const ggml_tensor * dst, int n_tasks) {
    const int32_t * desc = (const int32_t *) dst->op_params;
    if (desc[3] != 2 || desc[5] < 1) {
        return 0;
    }
    return (size_t) n_tasks * ggml_trellis_get_rows_scratch(desc, dst->src[0]->ne[0]) * sizeof(float);
}

void ggml_compute_forward_trellis_get_rows(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * ids  = dst->src[1];
    const ggml_tensor * s16  = dst->src[2];
    const ggml_tensor * pre  = dst->src[3];
    const ggml_tensor * post = dst->src[4];
    const ggml_tensor * gather = dst->src[5];  // a band of a banded embedding (ggml_trellis_get_rows_band)
    const int32_t * desc = (const int32_t *) dst->op_params;
    const int32_t row0 = gather ? desc[GGML_TRELLIS_BAND_ROW0] : 0;
    GGML_ASSERT(desc[0] == 0 && "trellis_get_rows: the embedding's rows must be unrotated");

    const int k = w->type - GGML_TYPE_TRL1 + 1;
    const int K = (int) w->ne[0], N = (int) w->ne[1], kt = K / 16, nw = 8 * k;
    GGML_ASSERT(ggml_trellis_desc_supported(K, N, desc, 7 + desc[6]));
    const uint32_t * codes = (const uint32_t *) w->data;
    const ggml_fp16_t * sc = (const ggml_fp16_t *) s16->data;
    const float * a = (const float *) pre->data;
    const float * b = (const float *) post->data;
    const int n = (int) ids->ne[0];
    GGML_ASSERT(params->wsize >= ggml_compute_forward_trellis_get_rows_work_size(dst, params->nth));
    float * tmp = desc[3] == 2 ? (float *) params->wdata + params->ith * ggml_trellis_get_rows_scratch(desc, K) : nullptr;

    for (int r0 = params->ith; r0 < n; r0 += params->nth) {
        int t = ((const int32_t *) ids->data)[r0];
        if (gather) {  // the band's own row, or a row of another band (left as it is)
            GGML_ASSERT(t >= 0 && t < gather->ne[0]);
            t = ((const int32_t *) gather->data)[t] - row0;
            if (t < 0 || t >= N) {
                continue;
            }
        }
        GGML_ASSERT(t >= 0 && t < N);
        float * out = (float *) ((char *) dst->data + (size_t) r0 * dst->nb[1]);
        const int J = t / 16, r = t % 16;
        for (int i = 0; i < kt; ++i) {
            const uint32_t * tile = codes + ((size_t) J * kt + i) * nw;
            const float s = GGML_CPU_FP16_TO_FP32(sc[i]);
            for (int c = 0; c < 16; ++c) {
                const int p = 8 * (4 * (r % 8) + (c % 8) / 2) + 4 * (r / 8) + 2 * (c / 8) + c % 2;
                out[16 * i + c] = (float) ggml_trellis_value(tile, k, p) * s;
            }
        }
        if (desc[3] == 1) {
            ggml_trellis_rht(out, K, desc[4]);
        } else if (desc[3] == 2) {  // T_cols^T: the inverse of the input side's whole-slice Hadamard
            for (int sl = 0; sl < desc[5]; ++sl) {
                const bool ok = ggml_trellis_hadfull_slice(out, K, nullptr, 0, desc[5], sl, true, nullptr, nullptr, tmp);
                GGML_ASSERT(ok);
            }
        }
        for (int j = 0; j < K; ++j) {
            out[j] *= a[j] * b[t];
        }
    }
}

// trellis fork: the coded linear (ggml_trellis_mul_mat) in fp32, in four phases with a barrier after each.
// The phases: rotate the inputs (s16 folded in), multiply each decoded tile row of V with all columns, apply the output side and post, gather the outputs.

#if !defined(__AVX2__)
// the window of value q = 16 c + r of a tile (input c, output row r): the first word of its 16-bit window, and the shift of that word pair
struct ggml_trellis_windows {
    uint8_t word[6][256];
    uint8_t shift[6][256];
};

static const ggml_trellis_windows & ggml_trellis_window_table() {
    static const ggml_trellis_windows tab = [] {
        ggml_trellis_windows t{};
        for (int k = 1; k <= 6; ++k) {
            const int total = 256 * k;
            for (int c = 0; c < 16; ++c) {
                for (int r = 0; r < 16; ++r) {
                    const int p = 8 * (4 * (r % 8) + (c % 8) / 2) + 4 * (r / 8) + 2 * (c / 8) + c % 2;
                    const int start = ((p + 1) * k - 16 + total) % total;
                    t.word[k - 1][16 * c + r] = (uint8_t) (start >> 5);
                    t.shift[k - 1][16 * c + r] = (uint8_t) (48 - (start & 31));
                }
            }
        }
        return t;
    }();
    return tab;
}
#endif

#if defined(__AVX2__)
// SIMD decode: the 8 windows of a group (input c, outputs 8 h .. 8 h + 7) start k words apart at the same bit offset.
// With the tile's words as 8 x k columns in registers, a group is two permutes and one funnel shift: state = ((H << (32 - c)) | (L >> c)) & 0xFFFF.
struct ggml_trellis_groups {
    uint8_t hc[6][32], hr[6][32], lc[6][32], lr[6][32]; // per group: H's and L's column and rotation
    int64_t cr[6][32], cl[6][32];                       // per group: c and 32 - c
#if defined(__AVX512F__) && defined(__AVX512BW__)
    // per input c, both of its groups in one 16-lane register: lanes 8 h + r' of group 2 c + h
    int32_t hidx[6][16][16], lidx[6][16][16], rcnt[6][16][16], lcnt[6][16][16];
#endif
};

static const ggml_trellis_groups & ggml_trellis_group_table() {
    static const ggml_trellis_groups tab = [] {
        ggml_trellis_groups t{};
        for (int k = 1; k <= 6; ++k) {
            const int total = 256 * k, q = k - 1;
            for (int g = 0; g < 32; ++g) {
                const int c = g / 2, h = g % 2;
                const int p0 = 8 * ((c % 8) / 2) + 4 * h + 2 * (c / 8) + c % 2;
                const int start = ((p0 + 1) * k - 16 + total) % total;
                const int w0 = start >> 5, s = 48 - (start & 31);
                const int a0 = w0 / k, b0 = w0 % k;
                t.hc[q][g] = (uint8_t) b0;
                t.hr[q][g] = (uint8_t) a0;
                if (s < 32) {
                    t.lc[q][g] = (uint8_t) (b0 + 1 < k ? b0 + 1 : 0);
                    t.lr[q][g] = (uint8_t) (b0 + 1 < k ? a0 : (a0 + 1) % 8);
                    t.cr[q][g] = s;
                } else {
                    t.lc[q][g] = (uint8_t) b0;
                    t.lr[q][g] = (uint8_t) a0;
                    t.cr[q][g] = s - 32;
                }
                t.cl[q][g] = 32 - t.cr[q][g];  // 32 (c = 0) clears the lanes
            }
#if defined(__AVX512F__) && defined(__AVX512BW__)
            for (int c = 0; c < 16; ++c) {
                for (int l = 0; l < 16; ++l) {
                    const int h = l / 8, r = l % 8, g = 2 * c + h;
                    t.hidx[q][c][l] = 16 * h + (t.hr[q][g] + r) % 8;  // h = 1: from the second source
                    t.lidx[q][c][l] = 16 * h + (t.lr[q][g] + r) % 8;
                    t.rcnt[q][c][l] = (int32_t) t.cr[q][g];
                    t.lcnt[q][c][l] = (int32_t) t.cl[q][g];
                }
            }
#endif
        }
        return t;
    }();
    return tab;
}
#endif

// V of one tile row (kt tiles of 8 k words) into wt: wt[16 j + r] = V[16 J + r][j], j = 16 i + c
static void ggml_trellis_decode_tile_row(const uint32_t * words, int k, int64_t kt, float * wt) {
#if defined(__AVX2__)
    const ggml_trellis_groups & gt = ggml_trellis_group_table();
    const int q = k - 1, nw = 8 * k;
    const __m256i stride = _mm256_setr_epi32(0, k, 2 * k, 3 * k, 4 * k, 5 * k, 6 * k, 7 * k);
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i mult = _mm512_set1_epi32((int) 0x83DCD12Du), m16 = _mm512_set1_epi32(0xFFFF);
    const __m512i m3f = _mm512_set1_epi32(0x3F3F3F3F), ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    const __m512i off = _mm512_set1_epi32(126);
    for (int64_t i = 0; i < kt; ++i) {
        const int * tile = (const int *) (words + i * nw);
        __m512i col[6];
        for (int b = 0; b < k; ++b) {  // words b, k + b, .., 7 k + b (the upper lanes are never selected)
            col[b] = _mm512_castsi256_si512(_mm256_i32gather_epi32(tile + b, stride, 4));
        }
        float * out = wt + i * 256;
        for (int c = 0; c < 16; ++c) {
            const int g0 = 2 * c, g1 = 2 * c + 1;
            const __m512i H = _mm512_permutex2var_epi32(col[gt.hc[q][g0]], _mm512_loadu_si512(gt.hidx[q][c]),
                                                        col[gt.hc[q][g1]]);
            const __m512i L = _mm512_permutex2var_epi32(col[gt.lc[q][g0]], _mm512_loadu_si512(gt.lidx[q][c]),
                                                        col[gt.lc[q][g1]]);
            const __m512i st = _mm512_and_si512(_mm512_or_si512(_mm512_sllv_epi32(H, _mm512_loadu_si512(gt.lcnt[q][c])),
                                                                _mm512_srlv_epi32(L, _mm512_loadu_si512(gt.rcnt[q][c]))),
                                                m16);
            const __m512i d = _mm512_and_si512(_mm512_srli_epi32(_mm512_mullo_epi32(st, mult), 2), m3f);
            const __m512i v = _mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(d, ones8), ones16), off);
            _mm512_storeu_ps(out + 16 * c, _mm512_cvtepi32_ps(v));
        }
    }
#else
    __m256i rot[8];
    for (int a = 0; a < 8; ++a) {
        rot[a] = _mm256_setr_epi32(a % 8, (a + 1) % 8, (a + 2) % 8, (a + 3) % 8, (a + 4) % 8, (a + 5) % 8, (a + 6) % 8,
                                   (a + 7) % 8);
    }
    const __m256i mult = _mm256_set1_epi32((int) 0x83DCD12Du), m16 = _mm256_set1_epi32(0xFFFF);
    const __m256i m3f = _mm256_set1_epi32(0x3F3F3F3F), ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    const __m256i off = _mm256_set1_epi32(126);
    const uint8_t * hc = gt.hc[q], * hr = gt.hr[q], * lc = gt.lc[q], * lr = gt.lr[q];
    const int64_t * cr = gt.cr[q], * cl = gt.cl[q];
    for (int64_t i = 0; i < kt; ++i) {
        const int * tile = (const int *) (words + i * nw);
        __m256i col[6];
        for (int b = 0; b < k; ++b) {
            col[b] = _mm256_i32gather_epi32(tile + b, stride, 4);  // words b, k + b, .., 7 k + b
        }
        float * out = wt + i * 256;
        for (int g = 0; g < 32; ++g) {
            const __m256i H = _mm256_permutevar8x32_epi32(col[hc[g]], rot[hr[g]]);
            const __m256i L = _mm256_permutevar8x32_epi32(col[lc[g]], rot[lr[g]]);
            const __m256i st = _mm256_and_si256(_mm256_or_si256(_mm256_sll_epi32(H, _mm_set_epi64x(0, cl[g])),
                                                                _mm256_srl_epi32(L, _mm_set_epi64x(0, cr[g]))),
                                                m16);
            const __m256i d = _mm256_and_si256(_mm256_srli_epi32(_mm256_mullo_epi32(st, mult), 2), m3f);
            const __m256i v = _mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(d, ones8), ones16), off);
            _mm256_storeu_ps(out + 8 * g, _mm256_cvtepi32_ps(v));
        }
    }
#endif
#else
    const ggml_trellis_windows & tab = ggml_trellis_window_table();
    const uint8_t * tw = tab.word[k - 1];
    const uint8_t * ts = tab.shift[k - 1];
    const int nw = 8 * k;
    uint32_t buf[8 * 6 + 1];  // the tile's words and the first again: the stream is circular
    for (int64_t i = 0; i < kt; ++i) {
        memcpy(buf, words + i * nw, nw * sizeof(uint32_t));
        buf[nw] = buf[0];
        float * out = wt + i * 256;
        for (int q = 0; q < 256; ++q) {
            const uint64_t pair = ((uint64_t) buf[tw[q]] << 32) | buf[tw[q] + 1];
            const uint32_t x = ((uint32_t) (pair >> ts[q]) & 0xFFFFu) * 0x83DCD12Du;  // the INT8 grid
            out[q] = (float) ((int) (((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126);
        }
    }
#endif
}

#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v) && defined(GGML_F32_EPR) && \
    (GGML_F32_EPR == 4 || GGML_F32_EPR == 8 || GGML_F32_EPR == 16)
#define GGML_TRELLIS_SIMD
#define GGML_TRELLIS_NV (16 / GGML_F32_EPR)          // registers per 16 output rows
#define GGML_TRELLIS_CT (GGML_F32_EPR == 16 ? 8 : 4) // columns per pass
#else
#define GGML_TRELLIS_CT 4
#endif

// ys[c][0 .. 15] = the tile row's 16 outputs for column xs[c], c < CT (wt as ggml_trellis_decode_tile_row)
template <int CT>
static void ggml_trellis_tile_mm(const float * wt, int64_t K, const float * const * xs, float * const * ys) {
#ifdef GGML_TRELLIS_SIMD
    GGML_F32_VEC acc[CT][GGML_TRELLIS_NV];
    for (int c = 0; c < CT; ++c) {
        for (int v = 0; v < GGML_TRELLIS_NV; ++v) {
            acc[c][v] = GGML_F32_VEC_ZERO;
        }
    }
    for (int64_t j = 0; j < K; ++j) {
        GGML_F32_VEC wv[GGML_TRELLIS_NV];
        for (int v = 0; v < GGML_TRELLIS_NV; ++v) {
            wv[v] = GGML_F32_VEC_LOAD(wt + 16 * j + v * GGML_F32_EPR);
        }
        for (int c = 0; c < CT; ++c) {
            const GGML_F32_VEC xv = GGML_F32_VEC_SET1(xs[c][j]);
            for (int v = 0; v < GGML_TRELLIS_NV; ++v) {
                acc[c][v] = GGML_F32_VEC_FMA(acc[c][v], wv[v], xv);
            }
        }
    }
    for (int c = 0; c < CT; ++c) {
        for (int v = 0; v < GGML_TRELLIS_NV; ++v) {
            GGML_F32_VEC_STORE(ys[c] + v * GGML_F32_EPR, acc[c][v]);
        }
    }
#else
    float acc[CT][16] = {};
    for (int64_t j = 0; j < K; ++j) {
        const float * wj = wt + 16 * j;
        for (int c = 0; c < CT; ++c) {
            const float xv = xs[c][j];
            for (int r = 0; r < 16; ++r) {
                acc[c][r] += wj[r] * xv;
            }
        }
    }
    for (int c = 0; c < CT; ++c) {
        for (int r = 0; r < 16; ++r) {
            ys[c][r] = acc[c][r];
        }
    }
#endif
}

static int64_t ggml_trellis_pad(int64_t n) {  // floats, to 64 bytes
    return (n + 15) / 16 * 16;
}

// per thread: the decoded tile row, or the whole-slice transforms' 2 L floats
static int64_t ggml_trellis_scratch(const int32_t * d, int64_t K, int64_t N) {
    int64_t s = 16 * K;
    if (d[3] == 2) {
        s = std::max<int64_t>(s, 2 * K / d[5]);
    }
    if (d[0] == 2) {
        s = std::max<int64_t>(s, 2 * N / d[2]);
    }
    return ggml_trellis_pad(s) + (int64_t) CACHE_LINE_SIZE_F32;
}

static int64_t ggml_trellis_work_floats(const ggml_tensor * dst, int n_tasks) {
    const int64_t K = dst->src[0]->ne[0], N = dst->src[0]->ne[1], T = ggml_nrows(dst->src[1]);
    return ggml_trellis_pad(K * T) + (dst->src[6] != nullptr ? ggml_trellis_pad(N * T) : 0) +
           (int64_t) n_tasks * ggml_trellis_scratch((const int32_t *) dst->op_params, K, N);
}

size_t ggml_compute_forward_trellis_work_size(const ggml_tensor * dst, int n_tasks) {
    return (size_t) ggml_trellis_work_floats(dst, n_tasks) * sizeof(float);
}

void ggml_compute_forward_trellis(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * x    = dst->src[1];
    const ggml_tensor * s16  = dst->src[2];
    const ggml_tensor * pre  = dst->src[3];
    const ggml_tensor * post = dst->src[4];
    const ggml_tensor * pin  = dst->src[5];
    const ggml_tensor * pout = dst->src[6];
    const int32_t * d = (const int32_t *) dst->op_params;
    const int rows_kind = d[0], rows_block = d[1], rows_shards = d[2];
    const int cols_kind = d[3], cols_block = d[4], cols_shards = d[5], nparts = d[6];
    const int32_t * parts = nparts > 0 ? d + 7 : nullptr;

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int ith = params->ith, nth = params->nth;
    const int k = (int) w->type - (int) GGML_TYPE_TRL1 + 1;
    const int64_t K = w->ne[0], N = w->ne[1], kt = K / 16, NJ = N / 16;
    const int64_t T = ggml_nrows(x);
    const int64_t ne11 = x->ne[1], ne12 = x->ne[2];
    GGML_ASSERT(ggml_trellis_desc_supported(K, N, d, 7 + nparts));

    GGML_ASSERT(params->wsize >= (size_t) ggml_trellis_work_floats(dst, nth) * sizeof(float));
    const int64_t scratch = ggml_trellis_scratch(d, K, N);
    float * xr = (float *) params->wdata;                                   // x_rot, K x T
    float * yb = pout ? xr + ggml_trellis_pad(K * T) : (float *) dst->data; // the product, N x T
    float * sc = xr + ggml_trellis_pad(K * T) + (pout ? ggml_trellis_pad(N * T) : 0) + ith * scratch;

    const uint32_t * codes = (const uint32_t *) w->data;
    const ggml_fp16_t * s = (const ggml_fp16_t *) s16->data;
    const float * a = (const float *) pre->data;
    const float * b = (const float *) post->data;
    const int32_t * gin = pin ? (const int32_t *) pin->data : nullptr;
    const int32_t * gout = pout ? (const int32_t *) pout->data : nullptr;

    // 1. the inputs
    {
        const int64_t seg = cols_kind == 2 ? K / cols_shards : cols_kind == 1 ? cols_block : K;
        const int64_t nseg = K / seg;
        for (int64_t it = ith; it < T * nseg; it += nth) {
            const int64_t t = it / nseg, j0 = (it % nseg) * seg;
            const int64_t i3 = t / (ne11 * ne12), i2 = (t / ne11) % ne12, i1 = t % ne11;
            const float * xc = (const float *) ((const char *) x->data + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);
            float * v = xr + t * K;
            for (int64_t j = j0; j < j0 + seg; ++j) {
                v[j] = (gin ? xc[gin[j]] : xc[j]) * a[j];
            }
            if (cols_kind == 1) {
                ggml_trellis_rht(v + j0, seg, cols_block);
            } else if (cols_kind == 2) {
                const bool ok = ggml_trellis_hadfull_slice(v, K, nullptr, 0, cols_shards, (int) (j0 / seg), false,
                                                           nullptr, nullptr, sc);
                GGML_ASSERT(ok);
            }
            for (int64_t j = j0; j < j0 + seg; ++j) {
                v[j] *= GGML_CPU_FP16_TO_FP32(s[j / 16]);
            }
        }
    }
    ggml_barrier(params->threadpool);

    // 2. the product, tile row by tile row
    {
        const int64_t J0 = NJ * ith / nth, J1 = NJ * (ith + 1) / nth;
        for (int64_t J = J0; J < J1; ++J) {
            ggml_trellis_decode_tile_row(codes + J * kt * 8 * k, k, kt, sc);
            for (int64_t t0 = 0; t0 < T; t0 += GGML_TRELLIS_CT) {
                const int nc = (int) std::min<int64_t>(GGML_TRELLIS_CT, T - t0);
                const float * xs[GGML_TRELLIS_CT];
                float * ys[GGML_TRELLIS_CT];
                for (int c = 0; c < nc; ++c) {
                    xs[c] = xr + (t0 + c) * K;
                    ys[c] = yb + (t0 + c) * N + 16 * J;
                }
                switch (nc) {
                    case 1: ggml_trellis_tile_mm<1>(sc, K, xs, ys); break;
                    case 2: ggml_trellis_tile_mm<2>(sc, K, xs, ys); break;
                    case 3: ggml_trellis_tile_mm<3>(sc, K, xs, ys); break;
#if GGML_TRELLIS_CT > 4
                    case 4: ggml_trellis_tile_mm<4>(sc, K, xs, ys); break;
                    case 5: ggml_trellis_tile_mm<5>(sc, K, xs, ys); break;
                    case 6: ggml_trellis_tile_mm<6>(sc, K, xs, ys); break;
                    case 7: ggml_trellis_tile_mm<7>(sc, K, xs, ys); break;
#endif
                    default: ggml_trellis_tile_mm<GGML_TRELLIS_CT>(sc, K, xs, ys); break;
                }
            }
        }
    }
    ggml_barrier(params->threadpool);

    // 3. the output side and post
    {
        const int64_t seg = rows_kind == 2 ? N / rows_shards : rows_kind == 1 ? rows_block : N;
        const int64_t nseg = N / seg;
        for (int64_t it = ith; it < T * nseg; it += nth) {
            const int64_t t = it / nseg, i0 = (it % nseg) * seg;
            float * v = yb + t * N;
            if (rows_kind == 2) {  // a slice gathers its entries from the fused parts: post on its own entries
                const bool ok = ggml_trellis_hadfull_slice(v, N, parts, nparts, rows_shards, (int) (i0 / seg), true,
                                                           nullptr, b, sc);
                GGML_ASSERT(ok);
                continue;
            }
            if (rows_kind == 1) {
                ggml_trellis_rht(v + i0, seg, rows_block);
            }
            for (int64_t i = i0; i < i0 + seg; ++i) {
                v[i] *= b[i];
            }
        }
    }

    // 4. the output gather
    if (gout) {
        ggml_barrier(params->threadpool);
        float * y = (float *) dst->data;
        for (int64_t t = ith; t < T; t += nth) {
            const float * v = yb + t * N;
            float * o = y + t * N;
            for (int64_t i = 0; i < N; ++i) {
                o[i] = v[gout[i]];
            }
        }
    }
}

#undef GGML_TRELLIS_CT
#ifdef GGML_TRELLIS_SIMD
#undef GGML_TRELLIS_NV
#undef GGML_TRELLIS_SIMD
#endif
