// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// Shared helpers of libtrellis: errors, the trellis decode chain, small device utilities.
//
// The decode chain uses rolling windows and the INT8 grid represented as fp16 bit patterns.
// See nariq/FORMAT.md for the tile layout.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "libtrellis.h"

namespace tq {

int fail(int code, const char* what);     // sets tq_last_error(), returns code
int check(cudaError_t e, const char* what);  // 0, or the CUDA error (message set)
int device_sms();                         // SMs of the current device (cached per device)
int device_cc();                          // compute capability of the current device, e.g. 120 (cached)
int check_matrix(const tq_matrix* m, int k_align, const char* fn);  // 0, or TQ_E_ARG with the message set

// ---- the trellis: a lane's 8 positions of a tile, their 16-bit states ---------------------------------------------
//
// Ring position p = 8 a + j (a = 0..31, j = 0..7) holds the tile element at output row a // 4 + 8 ((j >> 2) & 1) and
// input column 2 (a % 4) + (j & 1) + 8 ((j >> 1) & 1): exactly the m16n8k16 A fragment of lane a (rows = outputs g,
// g + 8; columns = inputs 2q, 2q + 1, 2q + 8, 2q + 9 with g = a >> 2, q = a & 3). So lane a decodes positions
// 8 a .. 8 a + 7, one contiguous span of the stream plus the 16 bits before it.

// The code words a lane needs: each "anchor" 32-bit value (two words funnel-shifted) holds SPAN consecutive windows.
template <int K>
struct Plan {
    static constexpr int NW = 8 * K, TOTAL = 256 * K;
    static constexpr int SPAN = 16 / K + 1;
    static constexpr int ANCH = (8 + SPAN - 1) / SPAN;
    int hi[ANCH], lo[ANCH], sh[ANCH];
    __device__ __forceinline__ explicit Plan(int lane) {
#pragma unroll
        for (int a = 0; a < ANCH; ++a) {
            const int e = ((8 * lane + 7 - a * SPAN + 1) * K) % TOTAL;  // the anchor window's end (exclusive)
            const int w0 = ((e - 32 + TOTAL) >> 5) % NW;               // the word holding bit e - 32
            hi[a] = w0;
            lo[a] = (w0 + 1) % NW;
            sh[a] = 32 - (e & 31);  // 32 when e is word-aligned: funnelshift clamps, giving word w0
        }
    }
};

template <int C>
__device__ __forceinline__ uint32_t low16(uint32_t r) {
    if constexpr (C == 0) return r & 0xFFFFu;
    else if constexpr (C == 16) return r >> 16;
    else if constexpr (C % 8 == 0) return __byte_perm(r, 0u, 0x4400u | ((C / 8 + 1) << 4) | (C / 8));
    else return (r >> C) & 0xFFFFu;
}

// the state of the lane's position 8 lane + I from its anchors
template <int K, int I>
__device__ __forceinline__ uint32_t state(const uint32_t (&r)[Plan<K>::ANCH]) {
    constexpr int A = (7 - I) / Plan<K>::SPAN, ANCHOR = 7 - A * Plan<K>::SPAN;
    return low16<(ANCHOR - I) * K>(r[A]);
}

template <int K>
__device__ __forceinline__ void anchors(const uint32_t (&w)[2 * Plan<K>::ANCH], const Plan<K>& plan,
                                        uint32_t (&r)[Plan<K>::ANCH]) {
#pragma unroll
    for (int x = 0; x < Plan<K>::ANCH; ++x) r[x] = __funnelshift_rc(w[2 * x], w[2 * x + 1], plan.sh[x]);
}

// The INT8 grid as an integer: x = state * mult; v = sum of the four 6-bit digits (x >> (8 i + 2)) & 63, minus 126.
// (x & 0xFCFCFCFC) holds each digit times 4 in its byte: dp4a sums them, >> 2 removes the 4.
__device__ __forceinline__ int v_int(uint32_t st, uint32_t mult) {
    return static_cast<int>(__dp4a((st * mult) & 0xFCFCFCFCu, 0x01010101u, 0u) >> 2) - 126;
}

// The 8 integers of a lane's positions 8 lane + I (I = 0..7)
template <int K>
__device__ __forceinline__ void decode_ints(const uint32_t (&w)[2 * Plan<K>::ANCH], const Plan<K>& plan, uint32_t mult,
                                            int (&v)[8]) {
    uint32_t r[Plan<K>::ANCH];
    anchors<K>(w, plan, r);
    v[0] = v_int(state<K, 0>(r), mult), v[1] = v_int(state<K, 1>(r), mult);
    v[2] = v_int(state<K, 2>(r), mult), v[3] = v_int(state<K, 3>(r), mult);
    v[4] = v_int(state<K, 4>(r), mult), v[5] = v_int(state<K, 5>(r), mult);
    v[6] = v_int(state<K, 6>(r), mult), v[7] = v_int(state<K, 7>(r), mult);
}

// fp16 bit pattern of 1024 + 4 s in the low half (s = the digit sum, <= 252, so 4 s <= 1008 stays in the mantissa):
// 4 v + 1528 with 1528 = 1024 + 4 x 126, exact in fp16
__device__ __forceinline__ uint32_t h_grid(uint32_t st, uint32_t mult) {
    return __dp4a((st * mult) & 0xFCFCFCFCu, 0x01010101u, 0x6400u);
}

// The offset decode: the A fragment of 4 v + 1528 (no subtraction; the GEMV removes 1528 x the activations' sums).
template <int K>
__device__ __forceinline__ void decode_raw(const uint32_t (&w)[2 * Plan<K>::ANCH], const Plan<K>& plan, uint32_t mult,
                                           uint32_t (&a)[4]) {
    uint32_t r[Plan<K>::ANCH];
    anchors<K>(w, plan, r);
    const uint32_t f0 = h_grid(state<K, 0>(r), mult), f1 = h_grid(state<K, 1>(r), mult);
    const uint32_t f2 = h_grid(state<K, 2>(r), mult), f3 = h_grid(state<K, 3>(r), mult);
    const uint32_t f4 = h_grid(state<K, 4>(r), mult), f5 = h_grid(state<K, 5>(r), mult);
    const uint32_t f6 = h_grid(state<K, 6>(r), mult), f7 = h_grid(state<K, 7>(r), mult);
    a[0] = __byte_perm(f0, f1, 0x5410);  // output g, inputs 2q, 2q + 1
    a[1] = __byte_perm(f4, f5, 0x5410);  // output g + 8
    a[2] = __byte_perm(f2, f3, 0x5410);  // output g, inputs 2q + 8, 2q + 9
    a[3] = __byte_perm(f6, f7, 0x5410);
}

// The FMA decode: the same fragments as the offset decode with fewer ALU-pipe instructions.
// 2 bits: every window is one byte permute or shift of r, r << 2, r << 4 or r << 6.
// Other widths: the pairs packed by a multiply-add (the FMA pipe); the multiplier 2^16 comes from constant memory so that ptxas cannot turn it back into a shift.
static __constant__ uint32_t k_pack = 0x10000u;  // one copy per translation unit (no relocatable device code)

__device__ __forceinline__ void decode2_raw(const uint32_t (&w)[2], const Plan<2>& plan, uint32_t mult,
                                            uint32_t (&a)[4]) {
    const uint32_t r = __funnelshift_rc(w[0], w[1], plan.sh[0]);
    const uint32_t r2 = r << 2, r4 = r << 4, r6 = r << 6;
    uint32_t s[8];
    s[0] = r2 >> 16, s[1] = r4 >> 16, s[2] = r6 >> 16;
    s[3] = __byte_perm(r, 0u, 0x4421), s[4] = __byte_perm(r2, 0u, 0x4421);
    s[5] = __byte_perm(r4, 0u, 0x4421), s[6] = __byte_perm(r6, 0u, 0x4421);
    s[7] = r & 0xFFFFu;
    uint32_t f[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = h_grid(s[i], mult);
    a[0] = __byte_perm(f[0], f[1], 0x5410);
    a[1] = __byte_perm(f[4], f[5], 0x5410);
    a[2] = __byte_perm(f[2], f[3], 0x5410);
    a[3] = __byte_perm(f[6], f[7], 0x5410);
}

template <int K, int I>
__device__ __forceinline__ uint32_t masked(const uint32_t (&r)[Plan<K>::ANCH], uint32_t mult) {
    return (state<K, I>(r) * mult) & 0xFCFCFCFCu;
}

// a half2 of 4 v + 1528 from two masked products (lo, hi): 4 s_hi 2^16 + 0x64006400 by a multiply-add, + 4 s_lo
__device__ __forceinline__ uint32_t pack_fma(uint32_t lo, uint32_t hi) {
    return __dp4a(lo, 0x01010101u, __dp4a(hi, 0x01010101u, 0u) * k_pack + 0x64006400u);
}

template <int K>
__device__ __forceinline__ void decode_raw_fma(const uint32_t (&w)[2 * Plan<K>::ANCH], const Plan<K>& plan,
                                               uint32_t mult, uint32_t (&a)[4]) {
    uint32_t r[Plan<K>::ANCH];
    anchors<K>(w, plan, r);
    a[0] = pack_fma(masked<K, 0>(r, mult), masked<K, 1>(r, mult));
    a[1] = pack_fma(masked<K, 4>(r, mult), masked<K, 5>(r, mult));
    a[2] = pack_fma(masked<K, 2>(r, mult), masked<K, 3>(r, mult));
    a[3] = pack_fma(masked<K, 6>(r, mult), masked<K, 7>(r, mult));
}

// D = 0: the offset decode; D = 1: the FMA decode (the same bits)
template <int K, int D>
__device__ __forceinline__ void decode_frag(const uint32_t (&w)[2 * Plan<K>::ANCH], const Plan<K>& plan, uint32_t mult,
                                            uint32_t (&a)[4]) {
    if constexpr (D == 0) decode_raw<K>(w, plan, mult, a);
    else if constexpr (K == 2) decode2_raw(w, plan, mult, a);
    else decode_raw_fma<K>(w, plan, mult, a);
}

// The A fragment of 4 v itself (exact in fp16: 4 v + 1528 and 1528 are exact, |4 v| <= 504): decode_frag, then one half2 subtract per pair.
// The prefill GEMM uses it, so it needs no activation sums. D as decode_frag.
template <int K, int D>
__device__ __forceinline__ void decode_exact(const uint32_t (&w)[2 * Plan<K>::ANCH], const Plan<K>& plan,
                                             uint32_t mult, uint32_t (&a)[4]) {
    decode_frag<K, D>(w, plan, mult, a);
    const __half2 off = __half2half2(__ushort_as_half(0x65F8u));  // 1528
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&a[i]), off);
        a[i] = *reinterpret_cast<uint32_t*>(&h);
    }
}

// Shared memory to plan with, per build: the whole library is compiled for one architecture list, so the budget is the
// smallest member's. sm_75 (Turing) has 64 KB per SM and per CTA, sm_80+ has 100 KB+.
#ifndef TQ_SMEM_SM
#define TQ_SMEM_SM 102400
#endif
#ifndef TQ_SMEM_CTA
#define TQ_SMEM_CTA 101312
#endif

// ---- small device utilities ---------------------------------------------------------------------------------------

// sm_75 (Turing) has no cp.async: the copies complete synchronously there and the group waits are no-ops, which
// keeps the stage-and-consume code identical, just without the overlap.
// The synchronous Turing copy is out of line on purpose: it is a real store the scheduler has to keep ordered, and
// inlining it into the fully unrolled stage loops made nvcc take an hour per translation unit.
static __device__ __noinline__ void cp16s_sync(uint32_t smem, const void* gmem) {
    *reinterpret_cast<uint4*>(__cvta_shared_to_generic(smem)) = *reinterpret_cast<const uint4*>(gmem);
}
__device__ __forceinline__ void cp16s(uint32_t smem, const void* gmem) {  // 16-byte global -> shared, L2 only
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
#else
    cp16s_sync(smem, gmem);
#endif
}
__device__ __forceinline__ void cp16s_l2_256(uint32_t smem, const void* gmem) {  // cp16s, prefetching 256 bytes into L2
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
#else
    cp16s(smem, gmem);
#endif
}
__device__ __forceinline__ void cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}

// Lets the next kernel on the stream, if it was launched with programmatic stream serialization, start early: it still
// waits for this grid's completion (griddepcontrol.wait) before it reads what this grid writes. A no-op before sm_90.
__device__ __forceinline__ void launch_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ float warp_max(float m) {
#pragma unroll
    for (int o = 16; o; o /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    return m;
}

// The unnormalized 128-point Walsh-Hadamard of one 128-block held by a warp: lane l holds entries l, l + 32, l + 64,
// l + 96 in v[0..3]. Bits 0-4 by shuffle, bits 5-6 in registers.
__device__ __forceinline__ void hadamard128(float (&v)[4], int lane) {
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
        const bool hi = lane & h;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float o = __shfl_xor_sync(0xffffffffu, v[m], h);
            v[m] = hi ? o - v[m] : v[m] + o;
        }
    }
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];  // h = 32
    v[0] = a + c, v[1] = b + d, v[2] = a - c, v[3] = b - d;                          // h = 64
}

// The butterflies of one row's block of B entries (256-4096) by a CTA of THREADS threads, each holding B / THREADS
// entries: stages h < 32 by shuffles, the next ones up to h = 512 by shuffles again after a transpose in shared memory
// (rows padded against bank conflicts), the rest in registers; each thread rewrites only the cells it read, so the
// transpose back needs no barrier before it. Reads pre * x[base + k] (x[idx[base + k]] with idx) and leaves the
// unnormalised H_B of it in tile: entry k at tile[k / 32][k % 32], every thread past the final barrier.
template <int B, int THREADS>
__device__ __forceinline__ void rht_row_stages(const float* x, const int32_t* idx, const float* pre, int base,
                                               float (&tile)[B / 32][33]) {
    static_assert(B % THREADS == 0 && THREADS % 32 == 0 && THREADS <= 1024, "a CTA of whole warps covers the block");
    constexpr int VALUES = B / THREADS;
    constexpr int COL = B < 1024 ? B / 32 : 32;
    const int tid = threadIdx.x, lane = tid % 32;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int k = tid + i * THREADS;
        float v = x[idx != nullptr ? idx[base + k] : base + k];
        if (pre) v *= pre[base + k];
#pragma unroll
        for (int h = 1; h < 32; h *= 2) {
            const float other = __shfl_xor_sync(0xffffffff, v, h);
            v = (lane & h) ? other - v : v + other;
        }
        tile[k / 32][lane] = v;
    }
    __syncthreads();
    float v[VALUES];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        v[i] = tile[t / (32 * COL) * COL + t % COL][t / COL % 32];
#pragma unroll
        for (int h = 1; h < COL; h *= 2) {
            const float other = __shfl_xor_sync(0xffffffff, v[i], h);
            v[i] = (lane & h) ? other - v[i] : v[i] + other;
        }
    }
#pragma unroll
    for (int step = 1024 / THREADS; step < VALUES; step *= 2) {
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            if (!(i & step)) {
                const float a = v[i], b = v[i + step];
                v[i] = a + b;
                v[i + step] = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int t = tid + i * THREADS;
        tile[t / (32 * COL) * COL + t % COL][t / COL % 32] = v[i];
    }
    __syncthreads();
}

// The position of input k (0-15) of a 16-input tile within a token's 16 halves in the m16n8k16 B fragment order:
// words (2q, 2q + 1) of token g hold inputs (2q, 2q + 1) and (2q + 8, 2q + 9), so that lane 4 g + q loads b0, b1 as
// one 8-byte pair.
__device__ __forceinline__ int frag_pos(int k) { return 4 * ((k & 7) >> 1) + 2 * (k >> 3) + (k & 1); }

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

}  // namespace tq
