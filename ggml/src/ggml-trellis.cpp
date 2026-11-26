// Host helpers for the trellis types (GGML_TYPE_TRL1..6, nariq/FORMAT.md): side transforms as in libtrellis, descriptor checks, decode and dequantization.
// Plain C++ without SIMD: the CPU backend uses the transforms here but has its own decode and product loops.
#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Paley matrices H_m (m = 12, 20, 68) built at compile time as in libtrellis rotate.cu: H = I + S with S[0][j] = 1, S[j][0] = -1 (j > 0) and S[i][j] = chi(j - i), chi the quadratic character mod m - 1
struct trl_paley {
    signed char h[68 * 68];
};

constexpr int trl_chi(int d, int q) {
    d = ((d % q) + q) % q;
    if (d == 0) {
        return 0;
    }
    for (int i = 1; i < q; ++i) {
        if ((i * i) % q == d) {
            return 1;
        }
    }
    return -1;
}

constexpr trl_paley trl_make_paley(int m) {
    trl_paley p{};
    const int q = m - 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            const int s = i == 0 ? (j > 0 ? 1 : 0) : (j == 0 ? -1 : trl_chi(j - i, q));
            p.h[i * m + j] = static_cast<signed char>(s + (i == j ? 1 : 0));
        }
    }
    return p;
}

constexpr trl_paley trl_h12 = trl_make_paley(12);
constexpr trl_paley trl_h20 = trl_make_paley(20);
constexpr trl_paley trl_h68 = trl_make_paley(68);

const signed char * trl_hm(int m) {
    return m == 12 ? trl_h12.h : m == 20 ? trl_h20.h : trl_h68.h;
}

// 1/sqrt(n) in double, rounded once to fp32 to match libtrellis
float trl_inv_sqrt(int64_t n) {
    return static_cast<float>(std::pow(static_cast<double>(n), -0.5));
}

// normalised Sylvester butterflies on `block` entries in fp32, then one multiply by c; this order is bit-identical to libtrellis tq_rht
void trl_fwht(float * u, int64_t block, float c) {
    for (int64_t h = 1; h < block; h *= 2) {
        for (int64_t i = 0; i < block; i += 2 * h) {
            for (int64_t j = i; j < i + h; ++j) {
                const float a = u[j], b = u[j + h];
                u[j] = a + b;
                u[j + h] = a - b;
            }
        }
    }
    for (int64_t i = 0; i < block; ++i) {
        u[i] *= c;
    }
}

bool trl_pow2(int64_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

// the fused parts of a whole-slice side (libtrellis's Parts): slice r is shard r of every part, side by side
struct trl_parts {
    int n;
    int64_t start[GGML_TRELLIS_MAX_PARTS];  // stored offset of each part
    int64_t shard[GGML_TRELLIS_MAX_PARTS];  // its shard length (part size / shards)
    int64_t cum[GGML_TRELLIS_MAX_PARTS + 1]; // slice offset where each part's shard begins
};

bool trl_make_parts(trl_parts & p, int64_t n, const int32_t * parts, int nparts, int shards) {
    if (shards < 1 || (parts != nullptr && (nparts < 1 || nparts > GGML_TRELLIS_MAX_PARTS))) {
        return false;
    }
    p.n = parts != nullptr ? nparts : 1;
    int64_t start = 0, cum = 0;
    for (int i = 0; i < p.n; ++i) {
        const int64_t size = parts != nullptr ? parts[i] : n;
        if (size <= 0 || size % shards) {
            return false;
        }
        p.start[i] = start;
        p.shard[i] = size / shards;
        p.cum[i] = cum;
        start += size;
        cum += size / shards;
    }
    p.cum[p.n] = cum;
    return start == n;
}

// the stored index of slice r's local entry l
inline int64_t trl_stored(const trl_parts & p, int r, int64_t l) {
    int i = 0;
    while (i + 1 < p.n && l >= p.cum[i + 1]) {
        ++i;
    }
    return p.start[i] + (int64_t) r * p.shard[i] + (l - p.cum[i]);
}

// out[a run + j] = c sum_b H[a][b] in[b run + j] (transpose: H[b][a]), summed over b in order as in tq_hadfull
void trl_cross(const float * in, float * out, int m, int64_t run, bool transpose, float c) {
    const signed char * h = trl_hm(m);
    for (int a = 0; a < m; ++a) {
        float * o = out + a * run;
        for (int64_t j = 0; j < run; ++j) {
            o[j] = 0.0f;
        }
        for (int b = 0; b < m; ++b) {
            const int s = transpose ? h[b * m + a] : h[a * m + b];
            const float * x = in + b * run;
            if (s > 0) {
                for (int64_t j = 0; j < run; ++j) {
                    o[j] += x[j];
                }
            } else {
                for (int64_t j = 0; j < run; ++j) {
                    o[j] += -x[j];
                }
            }
        }
        for (int64_t j = 0; j < run; ++j) {
            o[j] *= c;
        }
    }
}

// one side of a coded linear on n entries, T or T^T (inverse); kinds 0 and 1 are symmetric and ignore inverse. tmp: 2 n floats
bool trl_side(float * v, int64_t n, int kind, int block, int shards, const int32_t * parts, int nparts, bool inverse,
              float * tmp) {
    if (kind == 0) {
        return true;
    }
    if (kind == 1) {
        ggml_trellis_rht(v, n, block);
        return true;
    }
    for (int r = 0; r < shards; ++r) {
        if (!ggml_trellis_hadfull_slice(v, n, parts, nparts, shards, r, inverse, nullptr, nullptr, tmp)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int ggml_trellis_hadfull_m(int64_t len) {
    for (const int b : { 1, 12, 20, 68 }) {
        if (len >= b && len % b == 0 && trl_pow2(len / b)) {
            return b;
        }
    }
    return 0;
}

void ggml_trellis_rht(float * v, int64_t n, int block) {
    GGML_ASSERT(trl_pow2(block) && n % block == 0);
    const float c = trl_inv_sqrt(block);
    for (int64_t b0 = 0; b0 < n; b0 += block) {
        trl_fwht(v + b0, block, c);
    }
}

bool ggml_trellis_hadfull_slice(float * v, int64_t n, const int32_t * parts, int nparts, int shards, int r,
                                bool inverse, const float * pre, const float * post, float * tmp) {
    trl_parts p;
    if (!trl_make_parts(p, n, parts, nparts, shards) || r < 0 || r >= shards) {
        return false;
    }
    const int64_t len = n / shards;
    const int m = ggml_trellis_hadfull_m(len);
    if (m == 0) {
        return false;
    }
    const int64_t run = len / m;
    const float c_run = trl_inv_sqrt(run), c_m = trl_inv_sqrt(m);
    float * a = tmp;
    float * b = tmp + len;
    for (int64_t l = 0; l < len; ++l) {
        const int64_t i = trl_stored(p, r, l);
        a[l] = pre != nullptr ? v[i] * pre[i] : v[i];
    }
    float * res = a;
    if (m == 1) {
        trl_fwht(a, len, c_run);
    } else if (!inverse) {  // the runs' butterflies, then H_m across the runs
        for (int i = 0; i < m; ++i) {
            trl_fwht(a + i * run, run, c_run);
        }
        trl_cross(a, b, m, run, false, c_m);
        res = b;
    } else {  // H_m^T across the runs, then the runs' butterflies
        trl_cross(a, b, m, run, true, c_m);
        for (int i = 0; i < m; ++i) {
            trl_fwht(b + i * run, run, c_run);
        }
        res = b;
    }
    for (int64_t l = 0; l < len; ++l) {
        const int64_t i = trl_stored(p, r, l);
        v[i] = post != nullptr ? res[l] * post[i] : res[l];
    }
    return true;
}

bool ggml_trellis_desc_supported(int64_t K, int64_t N, const int32_t * desc, int n_desc) {
    if (desc == nullptr || n_desc < 7 || desc[6] < 0 || desc[6] > GGML_TRELLIS_MAX_PARTS || n_desc != 7 + desc[6] ||
        K % 16 || N % 16) {
        return false;
    }
    auto side_ok = [](int64_t n, int kind, int block, int shards, const int32_t * parts, int nparts) {
        switch (kind) {
            case 0:
                return true;
            case 1:
                return trl_pow2(block) && block >= 2 && n % block == 0;
            case 2:
                {
                    trl_parts p;
                    return trl_make_parts(p, n, nparts > 0 ? parts : nullptr, nparts, shards) &&
                           ggml_trellis_hadfull_m(n / shards) != 0;
                }
            default:
                return false;
        }
    };
    // the output rows' side carries the fused parts; an input-side whole-slice Hadamard takes one part [K]
    return side_ok(N, desc[0], desc[1], desc[2], desc + 7, desc[6]) &&
           side_ok(K, desc[3], desc[4], desc[5], nullptr, 0);
}

void ggml_trellis_decode_rows(enum ggml_type type, const void * data, int64_t K, int64_t n0, int64_t n_rows,
                              const ggml_fp16_t * s16, float * out) {
    GGML_ASSERT(ggml_is_trellis_type(type));
    GGML_ASSERT(K % 16 == 0 && n0 % 16 == 0 && n_rows % 16 == 0);
    const int k = (int) type - (int) GGML_TYPE_TRL1 + 1;
    const int nw = 8 * k, total = 256 * k;
    const int64_t kt = K / 16;
    const uint32_t * words = (const uint32_t *) data;
    for (int64_t J = n0 / 16; J < (n0 + n_rows) / 16; ++J) {
        for (int64_t i = 0; i < kt; ++i) {
            const uint32_t * t = words + (J * kt + i) * nw;  // output-tile-major: tile (i, J)
            const float s = s16 != nullptr ? ggml_fp16_to_fp32(s16[i]) : 1.0f;
            for (int p = 0; p < 256; ++p) {
                // the 16-bit window of the tile's circular stream ending at bit (p + 1) k; stream bit q is bit 31 - q % 32 of word q / 32
                const int start = ((p + 1) * k - 16 + total) % total;
                const int w = start >> 5, o = start & 31;
                const uint64_t pair = ((uint64_t) t[w] << 32) | t[(w + 1) % nw];
                const uint32_t state = (uint32_t) (pair >> (48 - o)) & 0xFFFFu;
                const uint32_t x = state * 0x83DCD12Du;  // the INT8 grid
                const int v = (int) (((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126;
                // ring position p = 8 a + j -> (row, column) within the tile (exllamav3's tensor-core order)
                const int a = p >> 3, jj = p & 7;
                const int row = (a >> 2) + 8 * ((jj >> 2) & 1);
                const int col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
                out[(16 * J + row - n0) * K + 16 * i + col] = (float) v * s;
            }
        }
    }
}

bool ggml_trellis_dequantize(enum ggml_type type, const void * data, int64_t K, int64_t N, const ggml_fp16_t * s16,
                             const float * pre, const float * post, const int32_t * pin, const int32_t * pout,
                             const int32_t * desc, int n_desc, float * out) {
    if (!ggml_is_trellis_type(type) || !ggml_trellis_desc_supported(K, N, desc, n_desc) || pre == nullptr ||
        post == nullptr) {
        return false;
    }
    // the gathers come from file data: reject them instead of aborting
    if (pin != nullptr && !std::all_of(pin, pin + K, [K](int32_t j) { return j >= 0 && j < K; })) {
        return false;
    }
    if (pout != nullptr && !std::all_of(pout, pout + N, [N](int32_t i) { return i >= 0 && i < N; })) {
        return false;
    }
    // W = P_out diag(post) T_rows^T W^ T_cols diag(pre) P_in (y = W x for the op, nariq/FORMAT.md)
    ggml_trellis_decode_rows(type, data, K, 0, N, s16, out);
    const int64_t nmax = K > N ? K : N;
    std::vector<float> tmp(2 * nmax), row(K);
    for (int64_t n = 0; n < N; ++n) {  // each row w -> (w T_cols) diag(pre) P_in: w T = (T^T w^T)^T
        float * w = out + n * K;
        if (!trl_side(w, K, desc[3], desc[4], desc[5], nullptr, 0, true, tmp.data())) {
            return false;
        }
        for (int64_t j = 0; j < K; ++j) {
            w[j] *= pre[j];
        }
        if (pin != nullptr) {  // x_used[j] = x[pin[j]]: column pin[j] of W collects column j of W^ T diag(pre)
            std::fill(row.begin(), row.end(), 0.0f);
            for (int64_t j = 0; j < K; ++j) {
                row[pin[j]] += w[j];
            }
            std::memcpy(w, row.data(), K * sizeof(float));
        }
    }
    // each column -> post diag T_rows^T column, 16 columns at a time
    std::vector<float> cols(16 * N);
    for (int64_t c0 = 0; c0 < K; c0 += 16) {
        for (int64_t n = 0; n < N; ++n) {
            for (int c = 0; c < 16; ++c) {
                cols[c * N + n] = out[n * K + c0 + c];
            }
        }
        for (int c = 0; c < 16; ++c) {
            float * col = cols.data() + c * N;
            if (!trl_side(col, N, desc[0], desc[1], desc[2], desc[6] > 0 ? desc + 7 : nullptr, desc[6], true,
                          tmp.data())) {
                return false;
            }
        }
        for (int64_t n = 0; n < N; ++n) {
            for (int c = 0; c < 16; ++c) {
                out[n * K + c0 + c] = cols[c * N + n] * post[n];
            }
        }
    }
    if (pout != nullptr) {  // y[i] = y_coded[pout[i]]: row i of W is row pout[i]
        std::vector<float> rows((size_t) N * K);
        for (int64_t i = 0; i < N; ++i) {
            std::memcpy(rows.data() + i * K, out + (int64_t) pout[i] * K, K * sizeof(float));
        }
        std::memcpy(out, rows.data(), rows.size() * sizeof(float));
    }
    return true;
}
