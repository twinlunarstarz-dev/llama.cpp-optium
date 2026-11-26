// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis: decode to dense (tq_decode) and tile repacking (tq_repack).
#include <cuda_bf16.h>

#include <cstdio>

#include "common.cuh"

namespace {

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half_rn(v); }

constexpr int DW = 8;         // warps per CTA: 8 k-tiles of one output tile row (16 x 128 outputs)
constexpr int DSTRIDE = 132;  // floats per staged row (128 + 4)

// One CTA: output tile row n_tile (16 rows) x input tiles 8 blockIdx.x .. + 7. Warp w decodes tile (8 blockIdx.x + w,
// n_tile) into shared memory (fp32: v, or the exact product v * s16), then the CTA stores the 16 x 128 block with
// coalesced 16-byte stores.
template <int K, typename T>
__global__ void __launch_bounds__(32 * DW) decode_k(const uint32_t* __restrict__ codes, int kt, int64_t sk, int64_t sn,
                                                     const __half* __restrict__ s16, uint32_t mult, int what, int nt0,
                                                     T* __restrict__ out, int64_t ld, int in_dim, int vec) {
    constexpr int WORDS = 8 * K;
    __shared__ float tile[16 * DSTRIDE];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3;
    const int i = blockIdx.x * DW + warp, j = nt0 + blockIdx.y;
    if (i < kt) {
        const uint32_t* p = codes + (i * sk + j * sn) * WORDS;
        const tq::Plan<K> plan(lane);
        uint32_t w[2 * tq::Plan<K>::ANCH];
#pragma unroll
        for (int x = 0; x < tq::Plan<K>::ANCH; ++x) {
            w[2 * x] = __ldg(p + plan.lo[x]);
            w[2 * x + 1] = __ldg(p + plan.hi[x]);
        }
        int v[8];
        tq::decode_ints<K>(w, plan, mult, v);
        const float s = what == TQ_W && s16 != nullptr ? __half2float(s16[i]) : 1.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = g + 8 * ((e >> 2) & 1), c = 16 * warp + 2 * q + (e & 1) + 8 * ((e >> 1) & 1);
            tile[r * DSTRIDE + c] = static_cast<float>(v[e]) * s;  // exact: |v| <= 126, s an fp16 value
        }
    }
    __syncthreads();
    // thread t stores 8 consecutive outputs of row t / 16
    const int r = threadIdx.x / 16, c0 = (threadIdx.x % 16) * 8, k0 = blockIdx.x * 128 + c0;
    if (k0 >= in_dim) return;
    T* dst = out + (int64_t)(blockIdx.y * 16 + r) * ld + k0;
    const float* src = tile + r * DSTRIDE + c0;
    if (vec) {  // 16-byte aligned rows: whole vectors (k0 + 8 <= in_dim, since in_dim is a multiple of 16)
        if constexpr (sizeof(T) == 4) {
            reinterpret_cast<float4*>(dst)[0] = make_float4(src[0], src[1], src[2], src[3]);
            reinterpret_cast<float4*>(dst)[1] = make_float4(src[4], src[5], src[6], src[7]);
        } else {
            T h[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) h[e] = from_f32<T>(src[e]);
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(h);
        }
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[e] = from_f32<T>(src[e]);
    }
}

template <int K>
cudaError_t launch_decode(const tq_matrix* m, int what, int n0, int n_rows, void* out, tq_dtype ot, int64_t ld,
                          cudaStream_t s) {
    const int kt = m->K / 16;
    const dim3 grid((kt + DW - 1) / DW, n_rows / 16), block(32 * DW);
    const uint32_t* codes = static_cast<const uint32_t*>(m->codes);
    const __half* s16 = static_cast<const __half*>(m->block_scale);
    const size_t es = ot == TQ_F32 ? 4 : 2;
    const int vec = (reinterpret_cast<uintptr_t>(out) % 16 == 0) && ((ld * es) % 16 == 0);
    switch (ot) {
        case TQ_F32:
            decode_k<K, float><<<grid, block, 0, s>>>(codes, kt, m->tile_stride_k, m->tile_stride_n, s16, m->mult,
                                                      what, n0 / 16, static_cast<float*>(out), ld, m->K, vec);
            break;
        case TQ_BF16:
            decode_k<K, __nv_bfloat16><<<grid, block, 0, s>>>(codes, kt, m->tile_stride_k, m->tile_stride_n, s16,
                                                              m->mult, what, n0 / 16,
                                                              static_cast<__nv_bfloat16*>(out), ld, m->K, vec);
            break;
        default:
            decode_k<K, __half><<<grid, block, 0, s>>>(codes, kt, m->tile_stride_k, m->tile_stride_n, s16, m->mult,
                                                       what, n0 / 16, static_cast<__half*>(out), ld, m->K, vec);
    }
    return cudaGetLastError();
}

// one thread per 16-byte unit (2 k units per tile of 8 k words)
__global__ void repack_k(const uint4* __restrict__ src, int64_t ssk, int64_t ssn, uint4* __restrict__ dst, int64_t dsk,
                         int64_t dsn, int units, int kt, int64_t total) {
    const int64_t u = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    const int64_t tile = u / units;
    const int part = static_cast<int>(u % units);
    const int64_t i = tile % kt, j = tile / kt;
    dst[(i * dsk + j * dsn) * units + part] = __ldg(src + (i * ssk + j * ssn) * units + part);
}

}  // namespace

namespace tq {
int check_matrix(const tq_matrix* m, int k_align, const char* fn) {
    static thread_local char msg[256];
    if (m == nullptr || m->codes == nullptr) {
        snprintf(msg, sizeof(msg), "%s: null matrix or codes", fn);
        return fail(TQ_E_ARG, msg);
    }
    if (m->bits < 1 || m->bits > 6 || m->K <= 0 || m->N <= 0 || m->K % k_align || m->N % 16) {
        snprintf(msg, sizeof(msg), "%s: bits must be 1-6, K a positive multiple of %d, N of 16 (bits %d, K %d, N %d)",
                 fn, k_align, m->bits, m->K, m->N);
        return fail(TQ_E_ARG, msg);
    }
    if (reinterpret_cast<uintptr_t>(m->codes) % 16) {
        snprintf(msg, sizeof(msg), "%s: codes must be 16-byte aligned", fn);
        return fail(TQ_E_ARG, msg);
    }
    return 0;
}
}  // namespace tq

extern "C" {
int tq_decode(const tq_matrix* m, int what, int n0, int n_rows, void* out, tq_dtype out_type, int64_t ld,
              void* stream) {
    if (int e = tq::check_matrix(m, 16, "tq_decode")) return e;
    if ((what != TQ_W && what != TQ_V) || n0 < 0 || n_rows <= 0 || n0 % 16 || n_rows % 16 || n0 + n_rows > m->N ||
        ld < m->K || out == nullptr || out_type < TQ_F32 || out_type > TQ_F16)
        return tq::fail(TQ_E_ARG, "tq_decode: what TQ_W or TQ_V, rows a 16-aligned range of [0, N), ld >= K");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaError_t e;
    switch (m->bits) {
        case 1: e = launch_decode<1>(m, what, n0, n_rows, out, out_type, ld, s); break;
        case 2: e = launch_decode<2>(m, what, n0, n_rows, out, out_type, ld, s); break;
        case 3: e = launch_decode<3>(m, what, n0, n_rows, out, out_type, ld, s); break;
        case 4: e = launch_decode<4>(m, what, n0, n_rows, out, out_type, ld, s); break;
        case 5: e = launch_decode<5>(m, what, n0, n_rows, out, out_type, ld, s); break;
        default: e = launch_decode<6>(m, what, n0, n_rows, out, out_type, ld, s); break;
    }
    return tq::check(e, "tq_decode launch");
}

int tq_repack(const void* src, int64_t src_stride_k, int64_t src_stride_n, void* dst, int64_t dst_stride_k,
              int64_t dst_stride_n, int bits, int K, int N, void* stream) {
    if (src == nullptr || dst == nullptr || bits < 1 || bits > 6 || K <= 0 || N <= 0 || K % 16 || N % 16 ||
        reinterpret_cast<uintptr_t>(src) % 16 || reinterpret_cast<uintptr_t>(dst) % 16)
        return tq::fail(TQ_E_ARG, "tq_repack: bits 1-6, K and N multiples of 16, 16-byte aligned buffers");
    const int units = 2 * bits, kt = K / 16;
    const int64_t total = (int64_t)kt * (N / 16) * units;
    repack_k<<<(unsigned)((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint4*>(src), src_stride_k, src_stride_n, static_cast<uint4*>(dst), dst_stride_k,
        dst_stride_n, units, kt, total);
    return tq::check(cudaGetLastError(), "tq_repack launch");
}
}
