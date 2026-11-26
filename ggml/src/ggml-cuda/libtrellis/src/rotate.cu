// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis: the rotations (tq_rht, tq_hadfull), fp32.
//
// Both run the Sylvester butterflies in a fixed order (stages h = 1, 2, 4, ... in turn, each (a + b, a - b) in fp32, then one multiply by fp32(p^-1/2)), so the result does not depend on the launch shape.
// A warp holds a chunk of 32 E entries (lane l, register e: entry l + 32 e) and runs the stages h < 32 by shuffles and h = 32 .. 16 E in registers.
// Blocks (and runs) of up to 128 entries: one warp per block. From 256: one CTA per block, its block / (32 E) warps continuing in shared memory (h = 32 E, 64 E, ...).
// E is 4, or at least block / 1024 (at most 32 warps).
// The H_m step of tq_hadfull computes c_m * sum_b (+-y_b) per output, in fp32 in a fixed order.
#include <atomic>
#include <cmath>
#include <cstdio>

#include "common.cuh"

namespace {

constexpr int RHT_MAX = 16384;
constexpr int HADFULL_MAX = 24576;

// ---- Paley matrices H_m (m = 12, 20, 68), built at compile time -----------------------------------------------------
// H = I + S, S[0][j] = 1 and S[j][0] = -1 for j > 0, S[i][j] = chi(j - i) for i, j > 0, chi the quadratic character
// mod q = m - 1 (nariq/FORMAT.md).
struct Paley {
    signed char h[68 * 68];
};

constexpr int chi(int d, int q) {
    d = ((d % q) + q) % q;
    if (d == 0) return 0;
    for (int i = 1; i < q; ++i)
        if ((i * i) % q == d) return 1;
    return -1;
}

constexpr Paley paley(int m) {
    Paley p{};
    const int q = m - 1;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            const int s = i == 0 ? (j > 0 ? 1 : 0) : (j == 0 ? -1 : chi(j - i, q));
            p.h[i * m + j] = static_cast<signed char>(s + (i == j ? 1 : 0));
        }
    return p;
}

__constant__ Paley k_h12 = paley(12);
__constant__ Paley k_h20 = paley(20);
__constant__ Paley k_h68 = paley(68);

__device__ __forceinline__ int base(int m, int a, int b) {
    return m == 12 ? k_h12.h[a * 12 + b] : m == 20 ? k_h20.h[a * 20 + b] : k_h68.h[a * 68 + b];
}

// The butterfly stages h = 1 .. min(p, 1024) / 2 on a warp's chunk v (lane l, register e: entry l + 32 e), in order.
template <int E>
__device__ __forceinline__ void warp_stages(float (&v)[E], int p, int lane) {
    const int hmax = p < 32 * E ? p : 32 * E;
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
        if (h >= hmax) break;
        const bool hi = lane & h;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], h);
            v[e] = hi ? o - v[e] : v[e] + o;  // (a + b, a - b) of the pair (low a, high b)
        }
    }
#pragma unroll
    for (int g = 1; g < E; g *= 2) {  // h = 32 g: registers e and e + g
        if (32 * g >= hmax) break;
#pragma unroll
        for (int e = 0; e < E; ++e)
            if (!(e & g)) {
                const float a = v[e], b = v[e + g];
                v[e] = a + b, v[e + g] = a - b;
            }
    }
}

// The stages h = h0 .. p / 2 on runs of p entries of shared memory s[0, len), in order.
__device__ __forceinline__ void smem_stages(float* s, int len, int p, int h0) {
    for (int h = h0; h < p; h *= 2) {
        __syncthreads();
        const int sh = __ffs(h) - 1;
        for (int i = threadIdx.x; i < len / 2; i += blockDim.x) {
            const int lo = ((i >> sh) << (sh + 1)) | (i & (h - 1)), hi = lo + h;
            const float a = s[lo], b = s[hi];
            s[lo] = a + b, s[hi] = a - b;
        }
    }
    __syncthreads();
}

// ---- tq_rht, blocks up to 128: one warp per chunk of S = 32 E entries, straight from global memory ----------------
template <int E>
__global__ void __launch_bounds__(256) rht_warp_k(const float* x, int64_t xs, float* y,
                                                   int64_t ys, int rows, int n, int block, float c,
                                                   const float* __restrict__ pre, const float* __restrict__ post) {
    constexpr int S = 32 * E;
    const int lane = threadIdx.x % 32, chunks = (n + S - 1) / S;
    const int64_t item = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (item >= (int64_t)rows * chunks) return;
    const int64_t row = item / chunks;
    const int k0 = static_cast<int>(item % chunks) * S;
    const float* xr = x + row * xs;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = k0 + lane + 32 * e;
        v[e] = k < n ? (pre != nullptr ? xr[k] * pre[k] : xr[k]) : 0.f;
    }
    warp_stages<E>(v, block, lane);
    float* yr = y + row * ys;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = k0 + lane + 32 * e;
        if (k < n) yr[k] = post != nullptr ? v[e] * c * post[k] : v[e] * c;
    }
}

// ---- tq_rht, blocks 256-16384: one CTA of block / (32 E) warps per (row, block) -----------------------------------
// Every entry is read before the first barrier and written after the last, by this CTA only: in place is safe.
template <int E>
__global__ void __launch_bounds__(1024) rht_cta_k(const float* x, int64_t xs, float* y,
                                                   int64_t ys, int block, float c, const float* __restrict__ pre,
                                                   const float* __restrict__ post) {
    extern __shared__ float s[];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int64_t row = blockIdx.y;
    const int k0 = blockIdx.x * block;
    const float* xr = x + row * xs + k0;
    float* yr = y + row * ys + k0;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int k = 32 * E * warp + lane + 32 * e;
        v[e] = pre != nullptr ? xr[k] * pre[k0 + k] : xr[k];
    }
    warp_stages<E>(v, block, lane);
    if (block <= 32 * E) {  // one warp: done in registers
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int k = lane + 32 * e;
            yr[k] = post != nullptr ? v[e] * c * post[k0 + k] : v[e] * c;
        }
        return;
    }
#pragma unroll
    for (int e = 0; e < E; ++e) s[32 * E * warp + lane + 32 * e] = v[e];
    smem_stages(s, block, block, 32 * E);
    for (int i = threadIdx.x; i < block; i += blockDim.x) yr[i] = post != nullptr ? s[i] * c * post[k0 + i] : s[i] * c;
}

// One row, one CTA per block of B entries (256-4096): tq::rht_row_stages, then y = post * c * H_B(...). GATHER: x[idx[k]]
// in place of x[k] (and no post).
template <int B, int THREADS, bool GATHER>
__device__ __forceinline__ void rht_row(const float* x, const int32_t* idx, float* y, float c, const float* pre,
                                        const float* post, int base) {
    __shared__ float tile[B / 32][33];
    tq::rht_row_stages<B, THREADS>(x, GATHER ? idx : nullptr, pre, base, tile);
    const int tid = threadIdx.x, lane = tid % 32;
#pragma unroll
    for (int i = 0; i < B / THREADS; ++i) {
        const int k = tid + i * THREADS;
        const float z = tile[k / 32][lane] * c;
        y[base + k] = !GATHER && post ? z * post[base + k] : z;
    }
}

template <int B, int THREADS, bool GATHER>
__global__ void rht_row_k(const float* x, const int32_t* idx, float* y, float c, const float* pre, const float* post) {
    rht_row<B, THREADS, GATHER>(x, idx, y, c, pre, post, blockIdx.x * B);
}

// tq_rht_rows: the rows of several vectors with one block size, in place, one launch; grid: each vector's blocks in turn
constexpr int MAX_ROWS_SET = 8;
struct RowsSet {
    float* y[MAX_ROWS_SET];
    const float* post[MAX_ROWS_SET];
    int cta0[MAX_ROWS_SET];  // each vector's first CTA
    int n;
    float c;
};

template <int B, int THREADS>
__global__ void rht_rows_k(const __grid_constant__ RowsSet set) {
    tq::launch_dependents();
    int m = 0;
#pragma unroll
    for (int i = 1; i < MAX_ROWS_SET; ++i)
        if (i < set.n && (int)blockIdx.x >= set.cta0[i]) m = i;
    rht_row<B, THREADS, false>(set.y[m], nullptr, set.y[m], set.c, nullptr, set.post[m], (blockIdx.x - set.cta0[m]) * B);
}

// ---- tq_hadfull: two kernels, each fully parallel ------------------------------------------------------------
// runs: the butterflies on every run of `run` entries of every (row, slice) (one warp per run chunk; runs above 1024:
// one CTA per run through shared memory); cross: H_m across the m runs, one CTA per (row, slice, 32 columns), the
// columns staged in shared memory, in place. Forward: runs (pre on load), then cross (post on store); inverse: cross
// (pre on load, H_m^T), then runs (post on store); m = 1: runs alone, with both.
constexpr int MAX_PARTS = 8;
struct Parts {
    int n;                      // parts
    int start[MAX_PARTS];       // stored offset of each part
    int shard[MAX_PARTS];       // its shard length (part size / shards)
    int cum[MAX_PARTS + 1];     // slice offset where each part's shard begins
};

// the stored index of slice r's local entry l (static indices only: Parts stays in parameter space)
__device__ __forceinline__ int64_t stored(const Parts& p, int r, int l) {
    int start = p.start[0], shard = p.shard[0], cum = p.cum[0];
#pragma unroll
    for (int i = 1; i < MAX_PARTS; ++i)
        if (i < p.n && l >= p.cum[i]) start = p.start[i], shard = p.shard[i], cum = p.cum[i];
    return start + (int64_t)r * shard + (l - cum);
}

template <int E>
__global__ void __launch_bounds__(256) hf_runs_k(const float* x, int64_t xs, float* y,
                                                  int64_t ys, Parts parts, int rows, int shards, int len, int run,
                                                  float c, const float* __restrict__ pre,
                                                  const float* __restrict__ post) {
    constexpr int S = 32 * E;
    const int lane = threadIdx.x % 32, cps = (len + S - 1) / S;
    const int64_t item = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (item >= (int64_t)rows * shards * cps) return;
    const int64_t row = item / ((int64_t)shards * cps);
    const int r = static_cast<int>((item / cps) % shards), k0 = static_cast<int>(item % cps) * S;
    const float* xr = x + row * xs;
    float v[E];
    int64_t at[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int l = k0 + lane + 32 * e;
        at[e] = l < len ? stored(parts, r, l) : -1;
        v[e] = at[e] >= 0 ? (pre != nullptr ? xr[at[e]] * pre[at[e]] : xr[at[e]]) : 0.f;
    }
    warp_stages<E>(v, run, lane);
    float* yr = y + row * ys;
#pragma unroll
    for (int e = 0; e < E; ++e)
        if (at[e] >= 0) yr[at[e]] = post != nullptr ? v[e] * c * post[at[e]] : v[e] * c;
}

// runs of 256 entries and more: one CTA of run / (32 E) warps per (row, slice, run), as rht_cta_k (in place is safe).
// E >= 16: at most 16 warps (128 registers: the part lookup of `stored` spills at 64)
template <int E>
__global__ void __launch_bounds__(E >= 16 ? 512 : 1024) hf_runs_cta_k(const float* x, int64_t xs, float* y,
                                                       int64_t ys, Parts parts, int run, float c,
                                                       const float* __restrict__ pre,
                                                       const float* __restrict__ post) {
    extern __shared__ float s[];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int r = blockIdx.y, k0 = blockIdx.x * run;
    const int64_t row = blockIdx.z;
    const float* xr = x + row * xs;
    float* yr = y + row * ys;
    float v[E];  // (the stored indices are recomputed at the store: no int64 array held across the stages)
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int64_t i = stored(parts, r, k0 + 32 * E * warp + lane + 32 * e);
        v[e] = pre != nullptr ? xr[i] * pre[i] : xr[i];
    }
    warp_stages<E>(v, run, lane);
    if (run <= 32 * E) {  // one warp: done in registers
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int64_t i = stored(parts, r, k0 + lane + 32 * e);
            yr[i] = post != nullptr ? v[e] * c * post[i] : v[e] * c;
        }
        return;
    }
#pragma unroll
    for (int e = 0; e < E; ++e) s[32 * E * warp + lane + 32 * e] = v[e];
    smem_stages(s, run, run, 32 * E);
    for (int l = threadIdx.x; l < run; l += blockDim.x) {
        const int64_t i = stored(parts, r, k0 + l);
        yr[i] = post != nullptr ? s[l] * c * post[i] : s[l] * c;
    }
}

// out[a run + j] = c sum_b H[a][b] in[b run + j] (transpose: H[b][a]) for the CTA's g columns j and every a: grid
// (column groups, slices, rows). The CTA stages all m inputs of its columns in shared memory before it writes, and no
// other CTA touches those columns, so in place (x == y) is safe. g = 8 / 16 / 32 for m = 68 / 20 / 12 keeps enough
// CTAs in flight at one row; one thread per output.
__global__ void __launch_bounds__(1024) hf_cross_k(const float* x, int64_t xs, float* y,
                                                   int64_t ys, Parts parts, int m, int run, int g, float c,
                                                   int transpose, const float* __restrict__ pre,
                                                   const float* __restrict__ post) {
    __shared__ float t[68 * 33];
    const int ncol = run < g ? run : g, j0 = blockIdx.x * g, r = blockIdx.y;
    const int64_t row = blockIdx.z;
    const float* xr = x + row * xs;
    for (int idx = threadIdx.x; idx < m * ncol; idx += blockDim.x) {
        const int b = idx / ncol, jj = idx % ncol;
        const int64_t i = stored(parts, r, b * run + j0 + jj);
        t[b * 33 + jj] = pre != nullptr ? xr[i] * pre[i] : xr[i];
    }
    __syncthreads();
    float* yr = y + row * ys;
    for (int idx = threadIdx.x; idx < m * ncol; idx += blockDim.x) {
        const int a = idx / ncol, jj = idx % ncol;
        float acc = 0.f;
        for (int b = 0; b < m; ++b) {
            const int h = transpose ? base(m, b, a) : base(m, a, b);
            const float v = t[b * 33 + jj];
            acc += h > 0 ? v : -v;
        }
        const int64_t i = stored(parts, r, a * run + j0 + jj);
        yr[i] = post != nullptr ? acc * c * post[i] : acc * c;
    }
}

template <int E>
cudaError_t launch_runs(const float* x, int64_t xs, float* y, int64_t ys, const Parts& p, int rows, int shards,
                        int len, int run, float c, const float* pre, const float* post, cudaStream_t s) {
    const int64_t items = (int64_t)rows * shards * ((len + 32 * E - 1) / (32 * E));
    hf_runs_k<E><<<(unsigned)((items + 7) / 8), 256, 0, s>>>(x, xs, y, ys, p, rows, shards, len, run, c, pre, post);
    return cudaGetLastError();
}

template <auto Kernel>  // one cache per kernel and device: the attribute is per function and device
cudaError_t set_smem(int bytes) {
    static std::atomic<bool> done[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (bytes <= 48 * 1024 || (dev >= 0 && dev < 64 && done[dev].load(std::memory_order_acquire))) return cudaSuccess;
    if (bytes > TQ_SMEM_CTA) return cudaErrorInvalidValue;  // sm_75 caps a CTA at 64 KB: the block is too large
    const cudaError_t e = cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (e == cudaSuccess && dev >= 0 && dev < 64) done[dev].store(true, std::memory_order_release);
    return e;
}

// entries per lane for a block (or run) of 256 or more: 4, at most 32 warps (16 for hadfull runs at E >= 16), at
// most block / 32
int rot_e(int block, bool runs = false) {
    int e = 4;
    while (e < 32 && block / (32 * e) > (runs && e >= 16 ? 16 : 32)) e *= 2;
    while (32 * e > block) e /= 2;
    return e;
}

template <int E>
cudaError_t launch_rht_cta(const float* x, int64_t xs, float* y, int64_t ys, int rows, int n, int block, float c,
                           const float* pre, const float* post, cudaStream_t s) {
    const int smem = block > 32 * E ? block * 4 : 0;
    if (cudaError_t e = set_smem<rht_cta_k<E>>(smem)) return e;
    rht_cta_k<E><<<dim3(n / block, rows), block / E, smem, s>>>(x, xs, y, ys, block, c, pre, post);
    return cudaGetLastError();
}

template <int E>
cudaError_t launch_runs_cta(const float* x, int64_t xs, float* y, int64_t ys, const Parts& p, int rows, int shards,
                            int len, int run, float c, const float* pre, const float* post, cudaStream_t s) {
    const int smem = run > 32 * E ? run * 4 : 0;
    if (cudaError_t e = set_smem<hf_runs_cta_k<E>>(smem)) return e;
    hf_runs_cta_k<E><<<dim3(len / run, shards, rows), run / E, smem, s>>>(x, xs, y, ys, p, run, c, pre, post);
    return cudaGetLastError();
}

template <int E>
cudaError_t launch_rht_warp(const float* x, int64_t xs, float* y, int64_t ys, int rows, int n, int block, float c,
                            const float* pre, const float* post, cudaStream_t s) {
    const int64_t chunks = (int64_t)rows * ((n + 32 * E - 1) / (32 * E));
    rht_warp_k<E><<<(unsigned)((chunks + 7) / 8), 256, 0, s>>>(x, xs, y, ys, rows, n, block, c, pre, post);
    return cudaGetLastError();
}

}  // namespace

extern "C" {
int tq_rht(const float* x, int64_t x_stride, float* y, int64_t y_stride, int rows, int n, int block, const float* pre,
           const float* post, void* stream) {
    if (x == nullptr || y == nullptr || rows < 1 || n < 1 || block < 16 || block > RHT_MAX || (block & (block - 1)) ||
        n % block || x_stride < n || y_stride < n || rows > 65535)
        return tq::fail(TQ_E_ARG, "tq_rht: block a power of two in 16-16384 dividing n, strides >= n, rows <= 65535");
    const float c = static_cast<float>(std::pow(static_cast<double>(block), -0.5));  // block^-0.5 in double, rounded once
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaError_t e;
    if (rows == 1 && block >= 256 && block <= 4096) {
        switch (block) {
            case 256: rht_row_k<256, 256, false><<<n / block, 256, 0, s>>>(x, nullptr, y, c, pre, post); break;
            case 512: rht_row_k<512, 256, false><<<n / block, 256, 0, s>>>(x, nullptr, y, c, pre, post); break;
            case 1024: rht_row_k<1024, 256, false><<<n / block, 256, 0, s>>>(x, nullptr, y, c, pre, post); break;
            case 2048: rht_row_k<2048, 1024, false><<<n / block, 1024, 0, s>>>(x, nullptr, y, c, pre, post); break;
            default: rht_row_k<4096, 1024, false><<<n / block, 1024, 0, s>>>(x, nullptr, y, c, pre, post); break;
        }
        return tq::check(cudaGetLastError(), "tq_rht launch");
    }
    switch (block) {
        case 16: case 32: e = launch_rht_warp<1>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
        case 64: e = launch_rht_warp<2>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
        case 128: e = launch_rht_warp<4>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
        default:
            switch (rot_e(block)) {
                case 4: e = launch_rht_cta<4>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
                case 8: e = launch_rht_cta<8>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
                case 16: e = launch_rht_cta<16>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
                default: e = launch_rht_cta<32>(x, x_stride, y, y_stride, rows, n, block, c, pre, post, s); break;
            }
    }
    return tq::check(e, "tq_rht launch");
}

int tq_rht_gather(const float* x, const int32_t* idx, float* y, int n, int block, const float* pre, void* stream) {
    if (x == nullptr || idx == nullptr || y == nullptr || n < 1 || block < 256 || block > 4096 ||
        (block & (block - 1)) || n % block)
        return tq::fail(TQ_E_ARG, "tq_rht_gather: block a power of two in 256-4096 dividing n");
    const float c = static_cast<float>(std::pow(static_cast<double>(block), -0.5));
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (block) {
        case 256: rht_row_k<256, 256, true><<<n / block, 256, 0, s>>>(x, idx, y, c, pre, nullptr); break;
        case 512: rht_row_k<512, 512, true><<<n / block, 512, 0, s>>>(x, idx, y, c, pre, nullptr); break;
        case 1024: rht_row_k<1024, 1024, true><<<n / block, 1024, 0, s>>>(x, idx, y, c, pre, nullptr); break;
        case 2048: rht_row_k<2048, 1024, true><<<n / block, 1024, 0, s>>>(x, idx, y, c, pre, nullptr); break;
        default: rht_row_k<4096, 1024, true><<<n / block, 1024, 0, s>>>(x, idx, y, c, pre, nullptr); break;
    }
    return tq::check(cudaGetLastError(), "tq_rht_gather launch");
}

int tq_rht_rows(int n, float* const* y, const int* len, const float* const* post, int block, void* stream) {
    if (n < 1 || n > MAX_ROWS_SET || y == nullptr || len == nullptr || block < 256 || block > 4096 ||
        (block & (block - 1)))
        return tq::fail(TQ_E_ARG, "tq_rht_rows: 1-8 vectors, block a power of two in 256-4096");
    RowsSet set{};
    set.n = n;
    set.c = static_cast<float>(std::pow(static_cast<double>(block), -0.5));
    int ctas = 0;
    for (int i = 0; i < n; ++i) {
        if (y[i] == nullptr || len[i] < block || len[i] % block)
            return tq::fail(TQ_E_ARG, "tq_rht_rows: every vector a multiple of block");
        set.y[i] = y[i];
        set.post[i] = post != nullptr ? post[i] : nullptr;
        set.cta0[i] = ctas;
        ctas += len[i] / block;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (block) {
        case 256: rht_rows_k<256, 256><<<ctas, 256, 0, s>>>(set); break;
        case 512: rht_rows_k<512, 256><<<ctas, 256, 0, s>>>(set); break;
        case 1024: rht_rows_k<1024, 256><<<ctas, 256, 0, s>>>(set); break;
        case 2048: rht_rows_k<2048, 1024><<<ctas, 1024, 0, s>>>(set); break;
        default: rht_rows_k<4096, 1024><<<ctas, 1024, 0, s>>>(set); break;
    }
    return tq::check(cudaGetLastError(), "tq_rht_rows launch");
}

int tq_hadfull(const float* x, int64_t x_stride, float* y, int64_t y_stride, int rows, int n, const int* parts,
               int nparts, int shards, int inverse, const float* pre, const float* post, void* stream) {
    if (x == nullptr || y == nullptr || rows < 1 || rows > 65535 || n < 1 || shards < 1 || x_stride < n ||
        y_stride < n || (parts != nullptr && (nparts < 1 || nparts > MAX_PARTS)))
        return tq::fail(TQ_E_ARG, "tq_hadfull: rows 1-65535, 1-8 parts, strides >= n");
    Parts p{};
    p.n = parts != nullptr ? nparts : 1;
    int start = 0, cum = 0;
    for (int i = 0; i < p.n; ++i) {
        const int size = parts != nullptr ? parts[i] : n;
        if (size <= 0 || size % shards) return tq::fail(TQ_E_ARG, "tq_hadfull: every part must split into the shards");
        p.start[i] = start, p.shard[i] = size / shards, p.cum[i] = cum;
        start += size, cum += size / shards;
    }
    p.cum[p.n] = cum;
    if (start != n) return tq::fail(TQ_E_ARG, "tq_hadfull: the parts must add up to n");
    const int len = n / shards;
    int m = 0;
    for (int b : {1, 12, 20, 68})
        if (len % b == 0 && ((len / b) & (len / b - 1)) == 0) {
            m = b;
            break;
        }
    if (m == 0 || len > HADFULL_MAX || len / m < 16) {
        static thread_local char msg[160];
        snprintf(msg, sizeof(msg), "tq_hadfull: a slice of %d is not m 2^k with m in 1, 12, 20, 68, 2^k >= 16 (at most %d)",
                 len, HADFULL_MAX);
        return tq::fail(TQ_E_ARG, msg);
    }
    const int run = len / m;
    const float c_run = static_cast<float>(std::pow(static_cast<double>(run), -0.5));
    const float c_m = static_cast<float>(std::pow(static_cast<double>(m), -0.5));
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    // the runs' butterflies: x -> y (forward, or m = 1) or in place on y (inverse, after the cross step)
    auto do_runs = [&](const float* src, int64_t ss, const float* pr, const float* po) -> cudaError_t {
        switch (run) {
            case 16: case 32: return launch_runs<1>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
            case 64: return launch_runs<2>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
            case 128: return launch_runs<4>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
            default:
                switch (rot_e(run, true)) {
                    case 4: return launch_runs_cta<4>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
                    case 8: return launch_runs_cta<8>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
                    case 16:
                        return launch_runs_cta<16>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
                    default:
                        return launch_runs_cta<32>(src, ss, y, y_stride, p, rows, shards, len, run, c_run, pr, po, s);
                }
        }
    };
    auto do_cross = [&](const float* src, int64_t ss, const float* pr, const float* po, int transpose) {
        const int g = m >= 68 ? 8 : m >= 20 ? 16 : 32, ncol = run < g ? run : g;
        const int threads = (m * ncol + 31) / 32 * 32 < 1024 ? (m * ncol + 31) / 32 * 32 : 1024;
        hf_cross_k<<<dim3((run + g - 1) / g, shards, rows), threads, 0, s>>>(src, ss, y, y_stride, p, m, run, g, c_m,
                                                                            transpose, pr, po);
        return cudaGetLastError();
    };
    cudaError_t e;
    if (m == 1) {
        e = do_runs(x, x_stride, pre, post);
    } else if (!inverse) {
        e = do_runs(x, x_stride, pre, nullptr);
        if (e == cudaSuccess) e = do_cross(y, y_stride, nullptr, post, 0);
    } else {
        e = do_cross(x, x_stride, pre, nullptr, 1);
        if (e == cudaSuccess) e = do_runs(y, y_stride, nullptr, post);
    }
    return tq::check(e, "tq_hadfull launch");
}
}
