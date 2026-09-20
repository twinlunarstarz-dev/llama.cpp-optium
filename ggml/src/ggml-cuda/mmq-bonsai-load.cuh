#pragma once

// CUDA tile loaders for Prism Bonsai PQ2_0/PTQ1_0 blocks.
// These are kept separate from the upstream loader to minimize fork drift.
#if !defined(GGML_USE_HIP)

// PQ2_0 stores 128 two-bit values per block. The MMQ tile layout is the
// same as Q2_0, but the block and scale geometry is 128 values per block.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_pq2_0(
        const char * __restrict__ x, int * __restrict__ x_tile,
        const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    constexpr int blocks_per_iter       = MMQ_ITER_K / QK_PQ2_0;
    constexpr int threads_per_row       = blocks_per_iter * QI_PQ2_0;
    constexpr int nrows                 = warp_size / threads_per_row;
    constexpr int scale_entries_per_blk = QK_PQ2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_blk;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI_PQ2_0;
    const int kqsx = txi % QI_PQ2_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;
        if (fallback) {
            i = min(i, i_max);
        }

        const block_pq2_0 * bxi = (const block_pq2_0 *) x + kbx0 + i*stride + kbx;
        const int16_t * qxi = (const int16_t *) bxi->qs + kqsx * 4;
        const int dst_offset = kbx*(scale_entries_per_blk*QI8_0) + kqsx*QI8_0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qxi[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + dst_offset + j*2+0] = qx;
            x_qs[i*sram_stride + dst_offset + j*2+1] = qy;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*2+0] = qx;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*2+1] = qy;
#endif
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_blk;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_pq2_0 * bxi = (const block_pq2_0 *) x + kbx0 + i*stride + scale_block;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif
    }
}

static __device__ __forceinline__ void ggml_cuda_mmq_decode_ptq1_0_qs4(
        uint32_t packed, int * __restrict__ dst, int stride) {
    uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
    uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const uint32_t w_lo = v_lo * 3;
        const uint32_t w_hi = v_hi * 3;
        v_lo = w_lo & 0x00FF00FF;
        v_hi = w_hi & 0x00FF00FF;
        dst[t * stride] = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
    }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_ptq1_0(
        const char * __restrict__ x, int * __restrict__ x_tile,
        const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(TURING_MMA_AVAILABLE)
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    constexpr int blocks_per_iter   = MMQ_ITER_K / QK_PTQ1_0;
    constexpr int threads_per_block = 8;
    constexpr int threads_per_row   = blocks_per_iter * threads_per_block;
    constexpr int nrows             = warp_size / threads_per_row;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / threads_per_block;
    const int lane = txi % threads_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_ptq1_0 * bxi = (const block_ptq1_0 *) x + kbx0 + i*stride + kbx;
#if defined(TURING_MMA_AVAILABLE)
        int * row = x_qs + i*sram_stride + kbx*(QK_PTQ1_0/4);
#else
        int * row = x_qs + i*(2*MMQ_TILE_NE_K + 1) + kbx*(QK_PTQ1_0/4);
#endif
        if (lane < 4) {
            ggml_cuda_mmq_decode_ptq1_0_qs4(get_int_b4(bxi->qs, lane), row + lane, 4);
        } else if (lane < 6) {
            const int g = lane - 4;
            ggml_cuda_mmq_decode_ptq1_0_qs4(get_int_b4(bxi->qs + 16, g), row + 20 + g, 2);
        } else if (lane == 6) {
            uint32_t v = (uint32_t) bxi->qh[0] | ((uint32_t) bxi->qh[1] << 16);
#pragma unroll
            for (int t = 0; t < 4; t += 2) {
                const uint32_t w0 = v * 3;
                v = w0 & 0x00FF00FF;
                const uint32_t w1 = v * 3;
                v = w1 & 0x00FF00FF;
                row[30 + t/2] = __vsub4(__byte_perm(w0, w1, 0x7531), 0x01010101);
            }
        }
    }

    constexpr int scale_entries_per_blk = QK_PTQ1_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_blk;
    constexpr int rows_per_warp = warp_size / scale_entries_per_row;
    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_blk;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/scale_entries_per_row;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_ptq1_0 * bxi = (const block_ptq1_0 *) x + kbx0 + i*stride + scale_block;
#if defined(TURING_MMA_AVAILABLE)
        x_df[i*sram_stride + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif
    }
}

#endif // !GGML_USE_HIP
