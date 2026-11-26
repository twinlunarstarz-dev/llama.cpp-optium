// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis: the host (CPU) decode, tq_decode_host: plain C++, no CUDA. The format's definition in loop form, for a
// CPU fallback (e.g. ggml's CPU backend on whole 16-row groups) and for converters.
#include <cstdint>
#include <cstring>

#include "libtrellis.h"

namespace {

// ring position p = 8 a + j of a tile -> (output row, input column) within the tile
inline void position(int p, int& row, int& col) {
    const int a = p >> 3, j = p & 7;
    row = (a >> 2) + 8 * ((j >> 2) & 1);
    col = 2 * (a & 3) + (j & 1) + 8 * ((j >> 1) & 1);
}

inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) man <<= 1, --exp;
            bits = sign | (exp << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

extern "C" int tq_decode_host(const void* codes, int bits, int K, int N, int64_t tile_stride_k, int64_t tile_stride_n,
                              const void* block_scale, uint32_t mult, int what, int n0, int n_rows, float* out,
                              int64_t ld) {
    if (codes == nullptr || out == nullptr || bits < 1 || bits > 6 || K <= 0 || N <= 0 || K % 16 || N % 16 ||
        n0 < 0 || n_rows <= 0 || n0 % 16 || n_rows % 16 || n0 + n_rows > N || ld < K || (what != TQ_W && what != TQ_V))
        return TQ_E_ARG;
    const uint32_t* words = static_cast<const uint32_t*>(codes);
    const uint16_t* scales = static_cast<const uint16_t*>(block_scale);
    const int nw = 8 * bits, total = 256 * bits;
    for (int j = n0 / 16; j < (n0 + n_rows) / 16; ++j) {
        for (int i = 0; i < K / 16; ++i) {
            const uint32_t* t = words + (i * tile_stride_k + j * tile_stride_n) * nw;
            const float s = what == TQ_W && scales != nullptr ? half_to_float(scales[i]) : 1.f;
            for (int p = 0; p < 256; ++p) {
                // the 16-bit window of the circular stream ending at bit (p + 1) k; stream bit q is bit 31 - q % 32
                // of word q / 32
                uint32_t state = 0;
                const int start = ((p + 1) * bits - 16 + total) % total;
                for (int b = 0; b < 16; ++b) {
                    const int q = (start + b) % total;
                    state = (state << 1) | ((t[q >> 5] >> (31 - (q & 31))) & 1u);
                }
                const uint32_t x = state * mult;
                const int v = (int)(((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126;
                int r, c;
                position(p, r, c);
                out[(int64_t)(16 * j + r - n0) * ld + 16 * i + c] = (float)v * s;  // exact
            }
        }
    }
    return 0;
}
