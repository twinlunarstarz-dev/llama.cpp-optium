// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis: the fused decode GEMV for 1-64 token rows (tq_gemv), sm_80+.
//
// Supports tile strides and activations prepared from fp32 / BF16 / fp16 inputs and block scales.
// The 128-point input Hadamard (TQ_HAD128_IN) is fused into the prepare step; larger input rotations are separate (tq_rht, tq_hadfull).
// The last CTA to arrive does the split reduction (no reliance on the order CTAs are dispatched in).
//
// 1. Prepare (one warp per (row, 128 inputs)): u = x * s16, then fp16 in the m16n8k16 B-fragment order
//    ([128-block c][row group r][k-tile][token g][16 halves], frag_pos) with one power-of-two scale per (row, block):
//    2^e with max |u| 2^e in [2^13, 2^14), and the block's sum of its fp16 values.
// 2. GEMV: a CTA owns 8 output tiles (128 outputs) over one split of the inputs. Per chunk of 8 k-tiles (128 inputs,
//    one activation scale), cp.async stages the warps' code words and the chunk's activations of every row group into
//    shared memory, S chunks ahead. Each warp decodes its tile once into an A fragment holding 4 v + 1528 (exact in
//    fp16) and multiplies it with each row group's B fragment; per chunk, acc += (product - 1528 sum) 2^-e.
// 3. Reduction: each split writes fp32 partials to the workspace; the last CTA of a column block to arrive sums them in
//    split order (deterministic), applies 1/4, the optional output Hadamard and post, and stores out.
#include <cuda_bf16.h>

#include <atomic>
#include <cmath>
#include <cstdio>

#include "common.cuh"

namespace {

constexpr int W4 = 8;   // warps = output tiles per CTA
constexpr int C4 = 8;   // k-tiles per chunk (128 inputs)
constexpr int MAX_ROWS = 64;

template <int K, int RM, bool SINGLE>
__host__ __device__ constexpr int stage_words() {
    // codes [warp][tile][8K]; acts [group][tile][64] (one row: [tile][8]); scales and sums
    return W4 * C4 * 8 * K + RM * (C4 * (SINGLE ? 8 : 64) + 16);
}

// Occupancy plan: the most CTAs per SM (4 for 1-2 row groups, whose registers fit 64 per thread; else 2) whose shared memory fits 100 KB per SM with 2-4 chunk stages, and the most stages at that count.
// 100 KB is sm_86/89/120's per-SM shared memory; sm_80/90 have more. A single row uses two stages.
template <int K, int RM, bool SINGLE>
struct Occ {
    static constexpr int STAGE_BYTES = stage_words<K, RM, SINGLE>() * 4;
    static constexpr int bytes(int s) { return (s * STAGE_BYTES + 16 + 255) / 256 * 256 + 1024; }
    // Two stages at least: the consume loop waits with cp_wait<S - 2> and issues chunk c + S - 1 before reading chunk c,
    // so with one stage it would overwrite the stage it is about to read.
    static constexpr int fit(int ctas) {
        for (int s = 4; s >= 2; --s)
            if (ctas * bytes(s) <= TQ_SMEM_SM) return s;
        return 2;
    }
    static constexpr int CTAS = RM >= 4 ? 2 : (fit(4) ? 4 : (fit(3) ? 3 : 2));
    static constexpr int S = SINGLE ? 2 : (fit(CTAS) > 2 ? fit(CTAS) : 2);
    static constexpr int SMEM = S * STAGE_BYTES;
    static_assert(SMEM <= TQ_SMEM_CTA, "the stage pair does not fit the shared-memory budget");
};

struct Args {
    const uint32_t* codes;
    int kt, nt;
    int64_t sk, sn;   // tile strides
    const uint32_t* xf;
    const float* xs;
    int rows, kb, splits;
    float* ws;        // [splits][rows][N] partials
    int* counters;    // [column blocks]; zeroed by the prepare launch, so a prepare must precede each GEMV launch
    void* out;
    int out_type;
    int64_t out_stride;
    const float* post;
    int had;
    uint32_t mult;
};

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    // sm_75 (Turing) has no m16n8k16: the same 16x8 tile accumulates as two m16n8k8 steps. a0, a1 are rows 0-7 and
    // 8-15 at k 0-7 and a2, a3 are rows 0-7 and 8-15 at k 8-15, so the k 0-7 step takes (a0, a1) with b0 and the k 8-15
    // step takes (a2, a3) with b1; the C fragment (16x8 f32, four registers) is identical in both shapes.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
                 "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%7,%8}, {%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b0), "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

__device__ __forceinline__ int arrive(int* counter) {  // gpu-scope acquire-release add: the previous count
    int prev;
    asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], 1;" : "=r"(prev) : "l"(counter) : "memory");
    return prev;
}

__device__ __forceinline__ void store_out(void* out, int type, int64_t idx, float v) {
    if (type == TQ_F32) static_cast<float*>(out)[idx] = v;
    else if (type == TQ_BF16) static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
    else static_cast<__half*>(out)[idx] = __float2half_rn(v);
}

// rows x 128 sums in red (shared): per row (one warp each) 1/4, the optional output Hadamard, post; the stores
__device__ __forceinline__ void finish(const Args& a, const float* red, int o0, int nv) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    constexpr float H128 = 0.08838834764831845f;  // 128^-1/2
    float scale[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int o = lane + 32 * m;
        scale[m] = (a.had ? 0.25f * H128 : 0.25f) * (a.post != nullptr && o < nv ? __ldg(a.post + o0 + o) : 1.f);
    }
    for (int t = warp; t < a.rows; t += W4) {
        float v[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) v[m] = lane + 32 * m < nv ? red[t * 128 + lane + 32 * m] : 0.f;
        if (a.had) tq::hadamard128(v, lane);
#pragma unroll
        for (int m = 0; m < 4; ++m)
            if (lane + 32 * m < nv) store_out(a.out, a.out_type, t * a.out_stride + o0 + lane + 32 * m, v[m] * scale[m]);
    }
}

// one CTA's work: column block cb (8 output tiles) over split `split` of the inputs
// SINGLE: one live row (rows == 1); only its activations are staged
template <int K, int RM, int D, bool SINGLE>
__device__ __forceinline__ void gemv_body(const Args& a, int cb, int split) {
    constexpr int WORDS = 8 * K, CODE_W = W4 * C4 * WORDS;
    constexpr int STAGE = stage_words<K, RM, SINGLE>(), S = Occ<K, RM, SINGLE>::S;
    constexpr int ACT_W = C4 * (SINGLE ? 8 : 64);  // activation words per row group and chunk in shared memory
    extern __shared__ __align__(16) uint32_t sm[];
    __shared__ int last;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, q = lane & 3;
    const int R = (a.rows + 7) / 8;
    const int i0 = split * a.kb, nk = min(a.kt - i0, a.kb), nch = nk / C4;
    const int j0 = cb * W4, j = j0 + warp;
    const bool active = j < a.nt;
    // this thread's 16-byte copies per chunk: code units (each warp's 8 tiles are 2 WORDS units), activation units
    // (128 per row group), scale units (4 per row group)
    constexpr int UC = W4 * 2 * WORDS, NC = (UC + 32 * W4 - 1) / (32 * W4), NA = (RM * 128 + 32 * W4 - 1) / (32 * W4);
    const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(sm));
    const uint32_t* cnext[NC];
    uint32_t cs[NC];
    bool cok[NC];
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int u = threadIdx.x + 32 * W4 * i, w = u / (2 * WORDS), uu = u % (2 * WORDS);
        const int d = 4 * uu / WORDS, word = 4 * uu % WORDS;
        cok[i] = u < UC && j0 + w < a.nt;
        cnext[i] = a.codes + ((int64_t)(i0 + d) * a.sk + (int64_t)(j0 + w) * a.sn) * WORDS + word;
        cs[i] = base + 4 * (w * C4 * WORDS + 4 * uu);
    }
    const int64_t chunk_codes = (int64_t)C4 * a.sk * WORDS;
    const uint32_t* anext = a.xf + (size_t)(i0 / C4) * R * (C4 * 64) + 4 * threadIdx.x;
    const float* snext = a.xs + (size_t)(i0 / C4) * 16 * R + 4 * threadIdx.x;
    const uint32_t as0 = base + 4 * (CODE_W + 4 * threadIdx.x), ss = base + 4 * (CODE_W + RM * ACT_W + 4 * threadIdx.x);
    auto issue = [&](uint32_t off) {  // the next chunk into the stage at byte offset off
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            if (cok[i]) {
                if constexpr (SINGLE) tq::cp16s_l2_256(cs[i] + off, cnext[i]);
                else tq::cp16s(cs[i] + off, cnext[i]);
            }
            cnext[i] += chunk_codes;
        }
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            if constexpr (SINGLE) {
                // the row's 32 bytes of each k-tile, packed; the seven padding rows are not copied
                if (threadIdx.x < 2 * C4)
                    tq::cp16s(as0 + off, anext + 56 * (threadIdx.x / 2));
            } else if (threadIdx.x + 32 * W4 * i < R * 128) {
                tq::cp16s(as0 + off + 4 * 128 * W4 * i, anext + 128 * W4 * i);
            }
        }
        if (threadIdx.x < 4 * R) tq::cp16s(ss + off, snext);
        anext += R * (C4 * 64);
        snext += 16 * R;
    };
#pragma unroll 1
    for (int s = 0; s < S - 1; ++s) {
        if (s < nch) issue(4 * STAGE * s);
        tq::cp_commit();
    }
    const tq::Plan<K> plan(lane);
    float acc[RM][4];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][e] = 0.f;
#pragma unroll 1
    for (int c = 0, cur = 0, next = S - 1; c < nch; ++c) {
        tq::cp_wait<S - 2>();
        __syncthreads();  // chunk c landed for every thread; chunk c - 1's stage is free
        if (c + S - 1 < nch) issue(4 * STAGE * next);
        tq::cp_commit();
        const uint32_t* st = sm + cur * STAGE;
        cur = cur + 1 == S ? 0 : cur + 1;
        next = next + 1 == S ? 0 : next + 1;
        if (active) {
            float g[RM][4];
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int e = 0; e < 4; ++e) g[r][e] = 0.f;
            const uint32_t* tw = st + warp * C4 * WORDS;
            // one row: every lane reads row 0's fragment, the other columns of the product are unused
            const uint2* xb = reinterpret_cast<const uint2*>(st + CODE_W) + (SINGLE ? q : lane);
#pragma unroll
            for (int d = 0; d < C4; ++d) {
                uint32_t w[2 * tq::Plan<K>::ANCH], af[4];
#pragma unroll
                for (int x = 0; x < tq::Plan<K>::ANCH; ++x) {
                    w[2 * x] = tw[d * WORDS + plan.lo[x]];
                    w[2 * x + 1] = tw[d * WORDS + plan.hi[x]];
                }
                tq::decode_frag<K, D>(w, plan, a.mult, af);
#pragma unroll
                for (int r = 0; r < RM; ++r)
                    if (r == 0 || r < R) {
                        const uint2 bb = xb[(r * C4 + d) * (ACT_W / (2 * C4))];
                        mma16816(g[r], af, bb.x, bb.y);
                    }
            }
            // the chunk's scales 2^-e ([group][8]) then its sums ([group][8]): acc += (g - 1528 sum) 2^-e
            const float* sc = reinterpret_cast<const float*>(st + CODE_W + RM * ACT_W);
#pragma unroll
            for (int r = 0; r < RM; ++r)
                if (r == 0 || r < R) {
                    const float2 s2 = *reinterpret_cast<const float2*>(sc + 8 * r + 2 * q);
                    const float2 x2 = *reinterpret_cast<const float2*>(sc + 8 * R + 8 * r + 2 * q);
                    acc[r][0] = fmaf(fmaf(-1528.f, x2.x, g[r][0]), s2.x, acc[r][0]);
                    acc[r][1] = fmaf(fmaf(-1528.f, x2.y, g[r][1]), s2.y, acc[r][1]);
                    acc[r][2] = fmaf(fmaf(-1528.f, x2.x, g[r][2]), s2.x, acc[r][2]);
                    acc[r][3] = fmaf(fmaf(-1528.f, x2.y, g[r][3]), s2.y, acc[r][3]);
                }
        }
    }
    tq::cp_wait<0>();
    // d0, d1: output 16 j + g, tokens 2q, 2q + 1 of the group; d2, d3: output 16 j + g + 8
    const int out_dim = a.nt * 16, o0 = cb * 128, nv = min(128, out_dim - o0);
    float* red = reinterpret_cast<float*>(sm);  // rows x 128 sums (the stages are free once every warp is done)
    const bool one = a.splits == 1;
    if (one) __syncthreads();  // every warp has finished reading the stages
    if (active) {
        float* part = one ? red - o0 : a.ws + (size_t)split * a.rows * out_dim;
        const int stride = one ? 128 : out_dim;
        const int o = 16 * j + (lane >> 2);
#pragma unroll
        for (int r = 0; r < RM; ++r)
            if (r < R) {
                const int t0 = 8 * r + 2 * q;
                if (t0 < a.rows) {
                    part[(size_t)t0 * stride + o] = acc[r][0];
                    part[(size_t)t0 * stride + o + 8] = acc[r][2];
                }
                if (t0 + 1 < a.rows) {
                    part[(size_t)(t0 + 1) * stride + o] = acc[r][1];
                    part[(size_t)(t0 + 1) * stride + o + 8] = acc[r][3];
                }
            }
    }
    if (!one) {
        // the barrier orders the CTA's partial stores before thread 0's gpu-scope release in arrive
        __syncthreads();
        if (threadIdx.x == 0) last = arrive(a.counters + cb) == a.splits - 1;
        __syncthreads();
        if (!last) return;
        // the splits' sums per (row, output) in split order into shared memory
        const size_t step = (size_t)a.rows * out_dim;
        const float* src = a.ws + o0;
        for (int i = threadIdx.x; i < a.rows * 128; i += 32 * W4) {
            const int t = i / 128, o = i % 128;
            float sum = 0.f;
            if (o < nv) {
                const float* p = src + (size_t)t * out_dim + o;
                int s = 0;
                for (; s + 4 <= a.splits; s += 4) {
                    const float v0 = __ldcg(p + s * step), v1 = __ldcg(p + (s + 1) * step);
                    const float v2 = __ldcg(p + (s + 2) * step), v3 = __ldcg(p + (s + 3) * step);
                    sum += v0, sum += v1, sum += v2, sum += v3;
                }
                for (; s < a.splits; ++s) sum += __ldcg(p + s * step);
            }
            red[i] = sum;
        }
    }
    __syncthreads();
    finish(a, red, o0, nv);
}

template <int K, int RM, int D, bool SINGLE>
__global__ void __launch_bounds__(32 * W4, (Occ<K, RM, SINGLE>::CTAS)) gemv_k(Args a) {
    tq::launch_dependents();
    if constexpr (SINGLE) a.rows = 1;  // a constant row count folds the row-group loops
    gemv_body<K, RM, D, SINGLE>(a, blockIdx.x, blockIdx.y);
}

// ---- several matrices of one width in one launch (tq_gemv_group) ----------------------------------------------------
constexpr int MAX_GROUP = 8;
struct Multi {
    int n;
    int cb0[MAX_GROUP + 1];  // each member's first column block in the grid (cb0[n]: the grid's width)
    Args a[MAX_GROUP];
};

// grid (sum of the members' column blocks, the most splits); a CTA past its member's splits has nothing to do.
// __grid_constant__: the members' arguments are read in place from parameter space, whatever the member index.
template <int K, int RM, int D, bool SINGLE>
__global__ void __launch_bounds__(32 * W4, (Occ<K, RM, SINGLE>::CTAS)) gemv_multi_k(const __grid_constant__ Multi g) {
    tq::launch_dependents();
    int m = 0;
#pragma unroll
    for (int i = 1; i < MAX_GROUP; ++i)
        if (i < g.n && (int)blockIdx.x >= g.cb0[i]) m = i;
    const Args& a = g.a[m];
    if ((int)blockIdx.y >= a.splits) return;
    if constexpr (SINGLE) {
        Args single = a;
        single.rows = 1;
        gemv_body<K, RM, D, SINGLE>(single, blockIdx.x - g.cb0[m], blockIdx.y);
    } else {
        gemv_body<K, RM, D, SINGLE>(a, blockIdx.x - g.cb0[m], blockIdx.y);
    }
}

// ---- one row, members of several widths in one launch (tq_gemv_group) -----------------------------------------------
// grid: each member's (split, column block) CTAs in turn, from first[member]; the body is the single-row gemv_k's
struct MixedGroup {
    Args members[MAX_GROUP];
    int n, first[MAX_GROUP + 1], widths[MAX_GROUP];
};

__global__ void __launch_bounds__(32 * W4, 4) gemv_mixed_k(const __grid_constant__ MixedGroup g) {
    tq::launch_dependents();
    int member = 0;
    for (int i = 1; i < g.n; ++i) {
        if ((int)blockIdx.x >= g.first[i]) member = i;
    }
    Args a = g.members[member];
    a.rows = 1;
    const int local = blockIdx.x - g.first[member];
    const int blocks = (a.nt + W4 - 1) / W4;
    const int cb = local % blocks, split = local / blocks;
    switch (g.widths[member]) {
        case 1: gemv_body<1, 1, 1, true>(a, cb, split); break;
        case 2: gemv_body<2, 1, 1, true>(a, cb, split); break;
        case 3: gemv_body<3, 1, 1, true>(a, cb, split); break;
        case 4: gemv_body<4, 1, 1, true>(a, cb, split); break;
        case 5: gemv_body<5, 1, 1, true>(a, cb, split); break;
        case 6: gemv_body<6, 1, 1, true>(a, cb, split); break;
    }
}

// ---- prepare: one warp per (row, 128-block) --------------------------------------------------------------------

// one warp's (row t, 128-block c) of one matrix's activations from x's values xv (lane l, i: input 128 c + l + 32 i)
__device__ __forceinline__ void prep_block(const float (&xv)[4], int t, int c, int groups, const float* __restrict__ pre,
                                           int had, const __half* __restrict__ s16, __half* __restrict__ xf,
                                           float* __restrict__ xs) {
    const int lane = threadIdx.x % 32, k0 = 128 * c;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        v[i] = xv[i];
        if (pre != nullptr) v[i] *= pre[k0 + k];
    }
    if (had) {
        tq::hadamard128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;
    }
    if (s16 != nullptr)
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= __half2float(s16[(k0 + lane + 32 * i) / 16]);
    const float m = tq::warp_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    int x2 = 0;
    frexpf(m > 0.f && m <= 3.0e38f ? m : 1.f, &x2);
    const int e = 14 - x2;
    __half* out = xf + ((size_t)(c * groups + t / 8) * 64 + t % 8) * 16;  // [c][group][k-tile][token][16]
    float sum[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        const __half h = __float2half_rn(ldexpf(v[i], e));
        out[(k / 16) * 128 + tq::frag_pos(k % 16)] = h;
        sum[i] = __half2float(h);
    }
    float total = (sum[0] + sum[1]) + (sum[2] + sum[3]);
#pragma unroll
    for (int o = 1; o < 32; o *= 2) total += __shfl_xor_sync(0xffffffffu, total, o);
    if (lane == 0) {
        xs[(size_t)c * 16 * groups + t] = ldexpf(1.f, -e);
        xs[(size_t)c * 16 * groups + 8 * groups + t] = total;
    }
}

// the split counters of the GEMV that follows on the stream, zeroed by the launch that prepares its activations
__device__ __forceinline__ void clear_counters(int* counters, int count) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += (size_t)gridDim.x * blockDim.x)
        counters[i] = 0;
}

template <typename T>
__global__ void __launch_bounds__(256) prepare_k(const T* __restrict__ x, int64_t x_stride, int rows, int K,
                                                  const float* __restrict__ pre, int had,
                                                  const __half* __restrict__ s16, __half* __restrict__ xf,
                                                  float* __restrict__ xs, int* counters, int count) {
    clear_counters(counters, count);
    const int lane = threadIdx.x % 32, nb = K / 128, item = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (item >= rows * nb) return;
    const int t = item / nb, c = item % nb;
    const T* xr = x + (int64_t)t * x_stride + 128 * c;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = tq::to_f32<T>(xr[lane + 32 * i]);
    prep_block(v, t, c, (rows + 7) / 8, pre, had, s16, xf, xs);
}

// every member's activations from one read of x (tq_gemv_group)
struct PrepMulti {
    int n;
    const float* pre[MAX_GROUP];
    const __half* s16[MAX_GROUP];
    __half* xf[MAX_GROUP];
    float* xs[MAX_GROUP];
    int had[MAX_GROUP];
};

template <typename T>
__global__ void __launch_bounds__(256) prepare_multi_k(const T* __restrict__ x, int64_t x_stride, int rows, int K,
                                                        const __grid_constant__ PrepMulti p, int* counters, int count) {
    if (blockIdx.y == 0) clear_counters(counters, count);
    const int lane = threadIdx.x % 32, nb = K / 128, item = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (item >= rows * nb) return;
    const int t = item / nb, c = item % nb;
    const T* xr = x + (int64_t)t * x_stride + 128 * c;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = tq::to_f32<T>(xr[lane + 32 * i]);
    for (int m = blockIdx.y; m < p.n; m += gridDim.y)
        prep_block(v, t, c, (rows + 7) / 8, p.pre[m], p.had[m], p.s16[m], p.xf[m], p.xs[m]);
}

// one row's input side x_rot = H_B(pre * x[idx]) / sqrt(B) (one CTA per block of B entries, tq_rht_gather's stages)
// and its activations as prepare_k makes them from x_rot, without x_rot in memory: the same bits
template <int B, int THREADS>
__global__ void __launch_bounds__(THREADS) prepare_rht_k(const float* __restrict__ x, const int32_t* __restrict__ idx,
                                                          const float* __restrict__ pre, float c,
                                                          const __half* __restrict__ s16, __half* __restrict__ xf,
                                                          float* __restrict__ xs, int* counters, int count) {
    clear_counters(counters, count);
    __shared__ float tile[B / 32][33];
    tq::rht_row_stages<B, THREADS>(x, idx, pre, blockIdx.x * B, tile);
    const int lane = threadIdx.x % 32;
    for (int w = threadIdx.x / 32; w < B / 128; w += THREADS / 32) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = tile[4 * w + i][lane] * c;
        prep_block(v, 0, (blockIdx.x * B) / 128 + w, 1, nullptr, 0, s16, xf, xs);
    }
}

// ---- host side -----------------------------------------------------------------------------------------------------

size_t align256(size_t b) { return (b + 255) / 256 * 256; }

int max_splits(int rows) {
    const int groups = (rows + 7) / 8;
    return 16 / groups > 2 ? 16 / groups : 2;
}

// split length in k-tiles (a multiple of 8): at most 128; shorter while the grid has fewer than two CTAs per SM, but at most max_splits(rows) splits
int split_kb(int kt, int nt, int rows) {
    const int cbs = (nt + W4 - 1) / W4, target = 2 * tq::device_sms();
    int kb = kt < 128 ? kt : 128;
    while (kb > C4) {
        const int next = kb - C4, splits = (kt + next - 1) / next;
        if (cbs * ((kt + kb - 1) / kb) >= target || splits > max_splits(rows)) break;
        kb = next;
    }
    return kb;
}

struct Layout {
    size_t counters, xf, xs, part, total;
};

Layout layout(int K, int N, int rows) {
    const int kt = K / 16, groups = (rows + 7) / 8, cbs = (N / 16 + W4 - 1) / W4;
    const int bound = (kt + 127) / 128 > max_splits(rows) ? (kt + 127) / 128 : max_splits(rows);  // splits <= this
    (void)cbs;
    Layout l;
    l.counters = 0;  // a fixed region, whatever the shape; prepare zeroes the ones a call uses
    l.xf = TQ_GEMV_COUNTER_BYTES;
    l.xs = l.xf + align256((size_t)(K / 128) * groups * 2048);
    l.part = l.xs + align256((size_t)(K / 128) * 16 * groups * sizeof(float));
    l.total = l.part + align256((size_t)bound * rows * N * sizeof(float));
    return l;
}

template <int K, int RM, int D, bool SINGLE = false>
cudaError_t launch_v(const Args& a, int cbs, cudaStream_t s) {
    using O = Occ<K, RM, SINGLE>;
    static std::atomic<bool> configured[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !configured[dev].load(std::memory_order_acquire)) {
        cudaError_t e =
            cudaFuncSetAttribute(gemv_k<K, RM, D, SINGLE>, cudaFuncAttributeMaxDynamicSharedMemorySize, O::SMEM);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(gemv_k<K, RM, D, SINGLE>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        configured[dev].store(true, std::memory_order_release);
    }
    gemv_k<K, RM, D, SINGLE><<<dim3(cbs, a.splits), dim3(32 * W4), O::SMEM, s>>>(a);
    return cudaGetLastError();
}

// the decode variant: the FMA decode (D = 1), except at 2 bits and 33-64 rows (8 row groups), where the offset decode (D = 0) is faster
template <int K, int RM>
constexpr int decode_variant() {
    return K == 2 && RM == 8 ? 0 : 1;
}

template <int K>
cudaError_t launch_k(const Args& a, int cbs, cudaStream_t s) {
    const int groups = (a.rows + 7) / 8;
    if (a.rows == 1) return launch_v<K, 1, decode_variant<K, 1>(), true>(a, cbs, s);
    if (groups <= 1) return launch_v<K, 1, decode_variant<K, 1>()>(a, cbs, s);
    if (groups <= 2) return launch_v<K, 2, decode_variant<K, 2>()>(a, cbs, s);
    if (groups <= 4) return launch_v<K, 4, decode_variant<K, 4>()>(a, cbs, s);
    return launch_v<K, 8, decode_variant<K, 8>()>(a, cbs, s);
}

template <int K, int RM, int D, bool SINGLE = false>
cudaError_t launch_multi_v(const Multi& g, int splits, cudaStream_t s) {
    using O = Occ<K, RM, SINGLE>;
    static std::atomic<bool> configured[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !configured[dev].load(std::memory_order_acquire)) {
        const auto kernel = gemv_multi_k<K, RM, D, SINGLE>;
        cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, O::SMEM);
        if (e == cudaSuccess) e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (e != cudaSuccess) return e;
        configured[dev].store(true, std::memory_order_release);
    }
    gemv_multi_k<K, RM, D, SINGLE><<<dim3(g.cb0[g.n], splits), dim3(32 * W4), O::SMEM, s>>>(g);
    return cudaGetLastError();
}

template <int K>
cudaError_t launch_multi_k(const Multi& g, int rows, int splits, cudaStream_t s) {
    const int groups = (rows + 7) / 8;
    if (rows == 1) return launch_multi_v<K, 1, decode_variant<K, 1>(), true>(g, splits, s);
    if (groups <= 1) return launch_multi_v<K, 1, decode_variant<K, 1>()>(g, splits, s);
    if (groups <= 2) return launch_multi_v<K, 2, decode_variant<K, 2>()>(g, splits, s);
    if (groups <= 4) return launch_multi_v<K, 4, decode_variant<K, 4>()>(g, splits, s);
    return launch_multi_v<K, 8, decode_variant<K, 8>()>(g, splits, s);
}

// a group's workspace: the counters, then each member's activations, scales and partials (tq_gemv's layout)
size_t group_layout(int n, const tq_matrix* ms, int rows, size_t* xf, size_t* xs, size_t* part) {
    size_t off = TQ_GEMV_COUNTER_BYTES;
    for (int i = 0; i < n; ++i) {
        const Layout l = layout(ms[i].K, ms[i].N, rows);
        xf[i] = off, off += l.xs - l.xf;
        xs[i] = off, off += l.part - l.xs;
        part[i] = off, off += l.total - l.part;
    }
    return off;
}

// the GEMV of activations a prepare launch has put in the workspace (whose counters it zeroed): one launch for the width
int gemv_prepared(const tq_matrix* m, int rows, char* w, const Layout& l, void* out, tq_dtype out_type,
                  int64_t out_stride, const float* post, int flags, cudaStream_t s) {
    const int kt = m->K / 16, nt = m->N / 16, kb = split_kb(kt, nt, rows), cbs = (nt + W4 - 1) / W4;
    Args a;
    a.codes = static_cast<const uint32_t*>(m->codes);
    a.kt = kt, a.nt = nt, a.sk = m->tile_stride_k, a.sn = m->tile_stride_n;
    a.xf = reinterpret_cast<const uint32_t*>(w + l.xf);
    a.xs = reinterpret_cast<const float*>(w + l.xs);
    a.rows = rows, a.kb = kb, a.splits = (kt + kb - 1) / kb;
    a.ws = reinterpret_cast<float*>(w + l.part);
    a.counters = reinterpret_cast<int*>(w + l.counters);
    a.out = out, a.out_type = out_type, a.out_stride = out_stride;
    a.post = post, a.had = (flags & TQ_HAD128_OUT) ? 1 : 0, a.mult = m->mult;
    cudaError_t e;
    switch (m->bits) {
        case 1: e = launch_k<1>(a, cbs, s); break;
        case 2: e = launch_k<2>(a, cbs, s); break;
        case 3: e = launch_k<3>(a, cbs, s); break;
        case 4: e = launch_k<4>(a, cbs, s); break;
        case 5: e = launch_k<5>(a, cbs, s); break;
        default: e = launch_k<6>(a, cbs, s); break;
    }
    return tq::check(e, "tq_gemv launch");
}
// prepare the activations of a GEMV into the workspace, zeroing its count split counters
cudaError_t prepare(const void* x, tq_dtype xt, int64_t x_stride, int rows, int K, const float* pre, int had,
                    const void* s16, void* xf, void* xs, cudaStream_t s, int* counters, int count) {
    const int warps = rows == 1 ? 1 : 8;  // a single row: more CTAs for the same work
    const unsigned blocks = (unsigned)((rows * (K / 128) + warps - 1) / warps);
    const __half* sc = static_cast<const __half*>(s16);
    __half* f = static_cast<__half*>(xf);
    float* sx = static_cast<float*>(xs);
    switch (xt) {
        case TQ_F32:
            prepare_k<float><<<blocks, 32 * warps, 0, s>>>(static_cast<const float*>(x), x_stride, rows, K, pre, had,
                                                           sc, f, sx, counters, count);
            break;
        case TQ_BF16:
            prepare_k<__nv_bfloat16><<<blocks, 32 * warps, 0, s>>>(static_cast<const __nv_bfloat16*>(x), x_stride,
                                                                   rows, K, pre, had, sc, f, sx, counters, count);
            break;
        default:
            prepare_k<__half><<<blocks, 32 * warps, 0, s>>>(static_cast<const __half*>(x), x_stride, rows, K, pre, had,
                                                            sc, f, sx, counters, count);
    }
    return cudaGetLastError();
}
}  // namespace

extern "C" {
size_t tq_gemv_workspace_bytes(int K, int N, int rows) {
    if (K <= 0 || N <= 0 || K % 128 || N % 16 || rows < 1 || rows > MAX_ROWS) return 0;
    return layout(K, N, rows).total;
}

int tq_gemv(const tq_matrix* m, const void* x, tq_dtype x_type, int64_t x_stride, int rows, void* out,
            tq_dtype out_type, int64_t out_stride, const float* pre, const float* post, int flags, void* ws,
            size_t ws_bytes, void* stream) {
    if (int e = tq::check_matrix(m, 128, "tq_gemv")) return e;
    if (rows < 1 || rows > MAX_ROWS || x == nullptr || out == nullptr || x_stride < m->K || out_stride < m->N ||
        x_type < TQ_F32 || x_type > TQ_F16 || out_type < TQ_F32 || out_type > TQ_F16)
        return tq::fail(TQ_E_ARG, "tq_gemv: 1-64 rows, x and out with strides >= K and N");
    if ((flags & TQ_HAD128_OUT) && m->N % 128) return tq::fail(TQ_E_ARG, "tq_gemv: TQ_HAD128_OUT needs N % 128 == 0");
    if (tq::device_cc() < 75) return tq::fail(TQ_E_ARCH, "tq_gemv: needs sm_75 or newer (mma.sync m16n8k16)");
    if ((m->N / 16 + W4 - 1) / W4 > TQ_GEMV_COUNTER_BYTES / (int)sizeof(int))
        return tq::fail(TQ_E_ARG, "tq_gemv: N above the counters' capacity (2,097,152 outputs)");
    const Layout l = layout(m->K, m->N, rows);
    if (ws == nullptr || ws_bytes < l.total || reinterpret_cast<uintptr_t>(ws) % 256)
        return tq::fail(TQ_E_WORKSPACE, "tq_gemv: workspace smaller than tq_gemv_workspace_bytes (or not 256-aligned)");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    char* w = static_cast<char*>(ws);
    const int cbs = (m->N / 16 + W4 - 1) / W4;
    int* counters = reinterpret_cast<int*>(w + l.counters);
    if (cudaError_t e = prepare(x, x_type, x_stride, rows, m->K, pre, (flags & TQ_HAD128_IN) ? 1 : 0,
                                     m->block_scale, w + l.xf, w + l.xs, s, counters, cbs))
        return tq::check(e, "tq_gemv prepare launch");
    return gemv_prepared(m, rows, w, l, out, out_type, out_stride, post, flags, s);
}

int tq_gemv_rht(const tq_matrix* m, const float* x, const int32_t* idx, int block, const float* pre, void* out,
                tq_dtype out_type, const float* post, int flags, void* ws, size_t ws_bytes, void* stream) {
    if (int e = tq::check_matrix(m, 128, "tq_gemv_rht")) return e;
    if (x == nullptr || out == nullptr || out_type < TQ_F32 || out_type > TQ_F16 || block < 256 || block > 4096 ||
        (block & (block - 1)) || m->K % block || (flags & TQ_HAD128_IN))
        return tq::fail(TQ_E_ARG, "tq_gemv_rht: block a power of two in 256-4096 dividing K, no TQ_HAD128_IN");
    if ((flags & TQ_HAD128_OUT) && m->N % 128)
        return tq::fail(TQ_E_ARG, "tq_gemv_rht: TQ_HAD128_OUT needs N % 128 == 0");
    if (tq::device_cc() < 75) return tq::fail(TQ_E_ARCH, "tq_gemv_rht: needs sm_75 or newer (mma.sync m16n8k16)");
    if ((m->N / 16 + W4 - 1) / W4 > TQ_GEMV_COUNTER_BYTES / (int)sizeof(int))
        return tq::fail(TQ_E_ARG, "tq_gemv_rht: N above the counters' capacity (2,097,152 outputs)");
    const Layout l = layout(m->K, m->N, 1);
    if (ws == nullptr || ws_bytes < l.total || reinterpret_cast<uintptr_t>(ws) % 256)
        return tq::fail(TQ_E_WORKSPACE, "tq_gemv_rht: workspace smaller than tq_gemv_workspace_bytes (or not 256-aligned)");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    char* w = static_cast<char*>(ws);
    const int cbs = (m->N / 16 + W4 - 1) / W4;
    int* counters = reinterpret_cast<int*>(w + l.counters);
    const float c = static_cast<float>(std::pow(static_cast<double>(block), -0.5));  // as tq_rht_gather
    const __half* sc = static_cast<const __half*>(m->block_scale);
    __half* xf = reinterpret_cast<__half*>(w + l.xf);
    float* xs = reinterpret_cast<float*>(w + l.xs);
    const unsigned blocks = m->K / block;
    switch (block) {
        case 256: prepare_rht_k<256, 256><<<blocks, 256, 0, s>>>(x, idx, pre, c, sc, xf, xs, counters, cbs); break;
        case 512: prepare_rht_k<512, 512><<<blocks, 512, 0, s>>>(x, idx, pre, c, sc, xf, xs, counters, cbs); break;
        case 1024: prepare_rht_k<1024, 1024><<<blocks, 1024, 0, s>>>(x, idx, pre, c, sc, xf, xs, counters, cbs); break;
        case 2048: prepare_rht_k<2048, 1024><<<blocks, 1024, 0, s>>>(x, idx, pre, c, sc, xf, xs, counters, cbs); break;
        default: prepare_rht_k<4096, 1024><<<blocks, 1024, 0, s>>>(x, idx, pre, c, sc, xf, xs, counters, cbs); break;
    }
    if (cudaError_t e = cudaGetLastError()) return tq::check(e, "tq_gemv_rht prepare launch");
    return gemv_prepared(m, 1, w, l, out, out_type, m->N, post, flags, s);
}

size_t tq_gemv_group_workspace_bytes(int n, const tq_matrix* ms, int rows) {
    if (n < 1 || n > MAX_GROUP || ms == nullptr || rows < 1 || rows > MAX_ROWS) return 0;
    for (int i = 0; i < n; ++i)
        if (ms[i].K <= 0 || ms[i].N <= 0 || ms[i].K % 128 || ms[i].N % 16 || ms[i].K != ms[0].K) return 0;
    size_t xf[MAX_GROUP], xs[MAX_GROUP], part[MAX_GROUP];
    return group_layout(n, ms, rows, xf, xs, part);
}

int tq_gemv_group(int n, const tq_matrix* ms, const void* x, tq_dtype x_type, int64_t x_stride, int rows,
                  void* const* outs, tq_dtype out_type, const int64_t* out_strides, const float* const* pre,
                  const float* const* post, const int* flags, void* ws, size_t ws_bytes, void* stream) {
    if (n < 1 || n > MAX_GROUP || ms == nullptr || outs == nullptr)
        return tq::fail(TQ_E_ARG, "tq_gemv_group: 1-8 matrices, and an output for each");
    int total_cbs = 0;
    for (int i = 0; i < n; ++i) {
        if (int e = tq::check_matrix(&ms[i], 128, "tq_gemv_group")) return e;
        const int64_t os = out_strides != nullptr ? out_strides[i] : ms[i].N;
        if (ms[i].K != ms[0].K || outs[i] == nullptr || os < ms[i].N)
            return tq::fail(TQ_E_ARG, "tq_gemv_group: one K for every matrix, outputs with strides >= N");
        if (flags != nullptr && (flags[i] & TQ_HAD128_OUT) && ms[i].N % 128)
            return tq::fail(TQ_E_ARG, "tq_gemv_group: TQ_HAD128_OUT needs N % 128 == 0");
        total_cbs += (ms[i].N / 16 + W4 - 1) / W4;
    }
    const int K = ms[0].K;
    if (rows < 1 || rows > MAX_ROWS || x == nullptr || x_stride < K || x_type < TQ_F32 || x_type > TQ_F16 ||
        out_type < TQ_F32 || out_type > TQ_F16)
        return tq::fail(TQ_E_ARG, "tq_gemv_group: 1-64 rows, x with stride >= K");
    if (tq::device_cc() < 75) return tq::fail(TQ_E_ARCH, "tq_gemv_group: needs sm_75 or newer (mma.sync m16n8k16)");
    if (total_cbs > TQ_GEMV_COUNTER_BYTES / (int)sizeof(int))
        return tq::fail(TQ_E_ARG, "tq_gemv_group: outputs above the counters' capacity (2,097,152 in all)");
    size_t xf[MAX_GROUP], xs[MAX_GROUP], part[MAX_GROUP];
    const size_t need = group_layout(n, ms, rows, xf, xs, part);
    if (ws == nullptr || ws_bytes < need || reinterpret_cast<uintptr_t>(ws) % 256)
        return tq::fail(TQ_E_WORKSPACE,
                        "tq_gemv_group: workspace smaller than tq_gemv_group_workspace_bytes (or not 256-aligned)");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    char* w = static_cast<char*>(ws);
    PrepMulti p{};
    p.n = n;
    for (int i = 0; i < n; ++i) {
        p.pre[i] = pre != nullptr ? pre[i] : nullptr;
        p.s16[i] = static_cast<const __half*>(ms[i].block_scale);
        p.xf[i] = reinterpret_cast<__half*>(w + xf[i]);
        p.xs[i] = reinterpret_cast<float*>(w + xs[i]);
        p.had[i] = flags != nullptr && (flags[i] & TQ_HAD128_IN) ? 1 : 0;
    }
    // a single row: one warp per CTA and the members spread over grid y, for more CTAs
    int* counters = reinterpret_cast<int*>(w);
    const int warps = rows == 1 ? 1 : 8;
    const dim3 blocks((unsigned)((rows * (K / 128) + warps - 1) / warps), rows == 1 ? n : 1);
    switch (x_type) {
        case TQ_F32:
            prepare_multi_k<float><<<blocks, 32 * warps, 0, s>>>(static_cast<const float*>(x), x_stride, rows, K, p,
                                                                 counters, total_cbs);
            break;
        case TQ_BF16:
            prepare_multi_k<__nv_bfloat16><<<blocks, 32 * warps, 0, s>>>(static_cast<const __nv_bfloat16*>(x),
                                                                         x_stride, rows, K, p, counters, total_cbs);
            break;
        default:
            prepare_multi_k<__half><<<blocks, 32 * warps, 0, s>>>(static_cast<const __half*>(x), x_stride, rows, K, p,
                                                                  counters, total_cbs);
    }
    if (cudaError_t e = cudaGetLastError()) return tq::check(e, "tq_gemv_group prepare launch");
    // one GEMV launch per width, the members in their order; counters at each member's offset
    Args a[MAX_GROUP];
    int gcb[MAX_GROUP];
    for (int i = 0, cum = 0; i < n; ++i) {
        const tq_matrix* m = &ms[i];
        const int kt = K / 16, nt = m->N / 16, kb = split_kb(kt, nt, rows);
        Args& ai = a[i];
        ai.codes = static_cast<const uint32_t*>(m->codes);
        ai.kt = kt, ai.nt = nt, ai.sk = m->tile_stride_k, ai.sn = m->tile_stride_n;
        ai.xf = reinterpret_cast<const uint32_t*>(w + xf[i]);
        ai.xs = reinterpret_cast<const float*>(w + xs[i]);
        ai.rows = rows, ai.kb = kb, ai.splits = (kt + kb - 1) / kb;
        ai.ws = reinterpret_cast<float*>(w + part[i]);
        ai.counters = reinterpret_cast<int*>(w) + cum;
        ai.out = outs[i], ai.out_type = out_type, ai.out_stride = out_strides != nullptr ? out_strides[i] : m->N;
        ai.post = post != nullptr ? post[i] : nullptr;
        ai.had = flags != nullptr && (flags[i] & TQ_HAD128_OUT) ? 1 : 0, ai.mult = m->mult;
        gcb[i] = (nt + W4 - 1) / W4;
        cum += gcb[i];
    }
    // a single row with several widths: one launch for every member
    bool mixed = false;
    for (int i = 0; i < n; ++i) mixed |= ms[i].bits != ms[0].bits;
    if (rows == 1 && mixed) {
        constexpr int smem_bytes[] = {0, Occ<1, 1, true>::SMEM, Occ<2, 1, true>::SMEM, Occ<3, 1, true>::SMEM,
                                      Occ<4, 1, true>::SMEM, Occ<5, 1, true>::SMEM, Occ<6, 1, true>::SMEM};
        static_assert(Occ<6, 1, true>::SMEM <= 49152, "no shared memory opt-in for the mixed launch");
        MixedGroup g{};
        g.n = n;
        int smem = 0;
        for (int i = 0; i < n; ++i) {
            g.members[i] = a[i];
            g.first[i + 1] = g.first[i] + gcb[i] * a[i].splits;
            g.widths[i] = ms[i].bits;
            smem = smem_bytes[ms[i].bits] > smem ? smem_bytes[ms[i].bits] : smem;
        }
        gemv_mixed_k<<<g.first[n], 32 * W4, smem, s>>>(g);
        return tq::check(cudaGetLastError(), "tq_gemv_group launch");
    }
    bool done[MAX_GROUP] = {false};
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        Multi g{};
        int splits = 0;
        for (int j = i; j < n; ++j) {
            if (done[j] || ms[j].bits != ms[i].bits) continue;
            done[j] = true;
            g.a[g.n++] = a[j];
        }
        // the column-block offsets of this launch's members
        for (int k = 0, at = 0; k < g.n; ++k) {
            g.cb0[k] = at;
            at += (g.a[k].nt + W4 - 1) / W4;
            g.cb0[k + 1] = at;
            splits = g.a[k].splits > splits ? g.a[k].splits : splits;
        }
        cudaError_t e;
        switch (ms[i].bits) {
            case 1: e = launch_multi_k<1>(g, rows, splits, s); break;
            case 2: e = launch_multi_k<2>(g, rows, splits, s); break;
            case 3: e = launch_multi_k<3>(g, rows, splits, s); break;
            case 4: e = launch_multi_k<4>(g, rows, splits, s); break;
            case 5: e = launch_multi_k<5>(g, rows, splits, s); break;
            default: e = launch_multi_k<6>(g, rows, splits, s); break;
        }
        if (e != cudaSuccess) return tq::check(e, "tq_gemv_group launch");
    }
    return 0;
}
}
