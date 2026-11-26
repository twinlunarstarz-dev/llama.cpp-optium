// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis: CUDA operations for trellis-coded weights. See nariq/FORMAT.md.
//
// A coded linear has N outputs and K inputs (W is N x K, y = W x, PyTorch's and ggml's orientation). Its weight is
//
//     W = diag(row_scale) U^T W^ V,    W^[n][k] = s16[k / 16] * v(state[n][k])
//
// where v is the INT8-grid codebook (an integer in [-126, 126]), s16 one fp16 scale per 16 inputs, and U, V the
// output and input rotations. This library provides:
//   - tq_decode:  W^ (or the integer matrix V) to fp32 / BF16 / fp16, bit-exact against tq_decode_host;
//   - tq_gemv:    the fused decode GEMV for 1-64 token rows (tensor cores, sm_80+): y = W^ x for a rotated x;
//   - tq_gemm:    the fused prefill GEMM for any rows (codes decoded in registers onto tensor cores, sm_80+);
//   - tq_rht, tq_hadfull: the rotations, as separate ops (blockwise random-sign Hadamards of 16-16384 points, and the
//     whole-slice Hadamard H_m (x) H_2^k with m in 1, 12, 20, 68, optionally per GPU slice of fused parts).
// For a module: x_rot = V x (tq_rht / tq_hadfull, forward, pre = the input signs), y' = W^ x_rot (tq_gemv), then
// y = row_scale * U^T y' (tq_rht / tq_hadfull, inverse, post = the output signs times row_scale).
//
// Every function returns 0 on success, else a negative library error or a positive CUDA error code; tq_last_error()
// gives the message (thread-local). Pointers are device pointers unless stated otherwise. Streams are cudaStream_t
// passed as void* so that this header needs no CUDA headers.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_VERSION 1
#define TQ_INT8_MULT 0x83DCD12Du  // the INT8 grid's multiplier (exllamav3's MUL1 multiplier), digit shift 2

#define TQ_E_ARG (-1)       // an argument outside what the function supports
#define TQ_E_ARCH (-2)      // the device lacks a feature the call needs (tq_gemv: tensor cores, sm_80+)
#define TQ_E_WORKSPACE (-3)  // the workspace is smaller than the function's *_workspace_bytes

const char* tq_last_error(void);
int tq_version(void);

typedef enum { TQ_F32 = 0, TQ_BF16 = 1, TQ_F16 = 2 } tq_dtype;

// A coded matrix. The codes are tiles of 16 inputs x 16 outputs, each one circular bitstream of 256 k-bit symbols,
// stored as 16 k int16 (8 k little-endian 32-bit words) in exllamav3's packing and in-tile order. Tile (i, j) covers
// inputs 16 i.. and outputs 16 j.. and starts at int16 offset (i * tile_stride_k + j * tile_stride_n) * 16 k.
//   - the checkpoints' layout (exllamav3's `trellis` tensor, (K/16, N/16, 16 k)): tile_stride_k = N/16, tile_stride_n = 1;
//   - output-tile-major ((N/16, K/16, 16 k), every 16 output rows contiguous: the GGUF layout of nariq/FORMAT.md):
//     tile_stride_k = 1, tile_stride_n = K/16.
// tq_matrix_exl3 / tq_matrix_rows fill the struct for either.
typedef struct {
    const void* codes;          // device, 16-byte aligned
    int bits;                   // k, 1-6
    int K, N;                   // multiples of 16 (tq_gemv, tq_gemm: K a multiple of 128)
    int64_t tile_stride_k;      // in tiles
    int64_t tile_stride_n;      // in tiles
    const void* block_scale;    // device fp16 [K/16]: s16 (null: ones, i.e. W^ = V)
    uint32_t mult;              // the codebook multiplier: TQ_INT8_MULT
} tq_matrix;

tq_matrix tq_matrix_exl3(const void* codes, int bits, int K, int N, const void* block_scale);
tq_matrix tq_matrix_rows(const void* codes, int bits, int K, int N, const void* block_scale);

// ---- decode -------------------------------------------------------------------------------------------------------

// Output rows [n0, n0 + n_rows) of W^ (what = TQ_W) or of V (what = TQ_V, the integers, exact in every dtype) to
// out[(n - n0) * ld + k] in fp32 / BF16 / fp16. W^ is computed as the exact fp32 product v * s16 and rounded once
// (round to nearest even) to BF16 / fp16. n0 and n_rows multiples of 16, ld >= K.
#define TQ_W 0
#define TQ_V 1
int tq_decode(const tq_matrix* m, int what, int n0, int n_rows, void* out, tq_dtype out_type, int64_t ld,
              void* stream);

// Copy codes between tile layouts (each tile's 16 k int16 unchanged), e.g. exllamav3's to output-tile-major.
int tq_repack(const void* src, int64_t src_stride_k, int64_t src_stride_n, void* dst, int64_t dst_stride_k,
              int64_t dst_stride_n, int bits, int K, int N, void* stream);

// ---- fused decode GEMV (1-64 token rows) -------------------------------------------------------------------------

// out[t][n] = post[n] * T_out(y)[n] for t < rows, n < N, where y[t] = W^ x_rot[t], x_rot[t] = T_in(pre * x[t]):
//   - x: rows x K at x_stride elements, fp32 / BF16 / fp16; pre: fp32 [K] or null (ones);
//   - T_in: with flags TQ_HAD128_IN, the normalised 128-point Hadamard over each 128 inputs (an rht128 input side fused
//     into the activations: pre = its signs times the tuned column scales); else the identity (x already rotated);
//   - T_out: with TQ_HAD128_OUT, the normalised 128-point Hadamard over each 128 outputs (an rht128 output side: post =
//     its signs times the row scales); else the identity; post: fp32 [N] or null (ones).
// out: fp32 / BF16 / fp16 at out_stride elements per row.
// Activations are rounded to fp16 with one power-of-two scale per (row, 128 inputs) (about 2^-12 relative error per
// element); products accumulate in fp32 on tensor cores; split-K partials are summed in a fixed order, so the result
// is deterministic for a given rows and device. Needs sm_80+ (mma.sync m16n8k16).
// ws: tq_gemv_workspace_bytes(K, N, rows) bytes (256-byte aligned), uninitialized: each call zeroes the split
// counters it uses (its first TQ_GEMV_COUNTER_BYTES) before its GEMV. One workspace of the largest size serves every
// module, for calls ordered on one stream; concurrent calls need their own. N <= 2,097,152.
#define TQ_HAD128_OUT 1
#define TQ_HAD128_IN 2
#define TQ_GEMV_COUNTER_BYTES 65536
size_t tq_gemv_workspace_bytes(int K, int N, int rows);
int tq_gemv(const tq_matrix* m, const void* x, tq_dtype x_type, int64_t x_stride, int rows, void* out,
            tq_dtype out_type, int64_t out_stride, const float* pre, const float* post, int flags, void* ws,
            size_t ws_bytes, void* stream);

// tq_gemv for n <= 8 matrices that read the same x (an input group: attention's q/k/v, the MLP's gate/up, a DeltaNet
// layer's in_proj_qkv/z/a/b): one launch prepares every member's activations from one read of x (each with its own
// pre[i], flags[i] & TQ_HAD128_IN and block scales), then one GEMV launch per code width runs the members of that
// width (one launch for all of them at one row). Member i's output is byte-identical to tq_gemv(&ms[i], x, ...,
// outs[i], ..., pre[i], post[i], flags[i]).
// ms: n matrices with the same K. out_strides, pre, post, flags: arrays of n, or null (N, none, none, 0). ws:
// tq_gemv_group_workspace_bytes(n, ms, rows) bytes, as for tq_gemv (one workspace serves tq_gemv and tq_gemv_group
// calls alike); at most 2,097,152 outputs in all.
// tq_gemv of one row whose input side is blockwise and needs no 128-point fusion: x_rot = H_b(pre * x[idx]) / sqrt(b)
// on each run of `block` entries (256-4096; idx: K int32 indices into x, or null for x itself), then out = W^ x_rot
// as tq_gemv with flags & TQ_HAD128_OUT and post. The same bits as tq_rht_gather (or tq_rht) followed by tq_gemv, in
// one launch fewer and without x_rot in memory. x: fp32; out: N entries; ws: tq_gemv_workspace_bytes(K, N, 1).
int tq_gemv_rht(const tq_matrix* m, const float* x, const int32_t* idx, int block, const float* pre, void* out,
                tq_dtype out_type, const float* post, int flags, void* ws, size_t ws_bytes, void* stream);

size_t tq_gemv_group_workspace_bytes(int n, const tq_matrix* ms, int rows);
int tq_gemv_group(int n, const tq_matrix* ms, const void* x, tq_dtype x_type, int64_t x_stride, int rows,
                  void* const* outs, tq_dtype out_type, const int64_t* out_strides, const float* const* pre,
                  const float* const* post, const int* flags, void* ws, size_t ws_bytes, void* stream);

// ---- prefill GEMM (any number of rows) ----------------------------------------------------------------------------

// The same product as tq_gemv (same arguments and flags: pre, post, TQ_HAD128_IN, TQ_HAD128_OUT) for any rows, both tile layouts.
// A prologue rounds each row of x_rot to fp16 with one power-of-two scale per row; one GEMM kernel decodes the codes in registers onto tensor cores (fp32 accumulation) and applies the output Hadamard, the row scale and post in its epilogue.
// No decoded weights in memory. The result does not depend on the tiling the call picks, and two calls give the same bytes. Needs sm_80+.
// ws: tq_gemm_workspace_bytes(K, N, rows) bytes (256-byte aligned), allocated by the caller; nothing in it needs zeroing.
// It holds the activations (about 2 K bytes per row) and, for modules with few outputs (where the call splits K), fp32 partials: for one workspace that serves every module, take the largest size over the modules (k_proj's can exceed gate_proj's at the same rows).
// The size depends on (K, N, rows) and the device's SM count.
// No host synchronisation; capturable in CUDA graphs (as for tq_gemv, make a warm-up call before capturing: each kernel sets its shared-memory attribute on its first call).
size_t tq_gemm_workspace_bytes(int K, int N, int rows);
int tq_gemm(const tq_matrix* m, const void* x, tq_dtype x_type, int64_t x_stride, int rows, void* out,
            tq_dtype out_type, int64_t out_stride, const float* pre, const float* post, int flags, void* ws,
            size_t ws_bytes, void* stream);

// The host (CPU) decode, plain C++: rows [n0, n0 + n_rows) of W^ (TQ_W) or V (TQ_V) as fp32, out[(n - n0) * ld + k];
// codes and block_scale (fp16 bits, or null) in host memory. The format's definition in loop form (slow).
int tq_decode_host(const void* codes, int bits, int K, int N, int64_t tile_stride_k, int64_t tile_stride_n,
                   const void* block_scale, uint32_t mult, int what, int n0, int n_rows, float* out, int64_t ld);

// ---- rotations (fp32, in place allowed: y == x with equal strides) ------------------------------------------------

// y[r] = post * H_b (pre * x[r]) on each run of `block` entries (b a power of two, 16-16384), H_b the normalised Sylvester Hadamard (symmetric, so this is both the input side's T x and the output side's T^T x).
// pre, post: fp32 vectors of n entries, or null.
// The butterfly stages h = 1, 2, 4, ... run in order, each (a + b, a - b) in fp32, then one multiply by fp32(block^-0.5).
int tq_rht(const float* x, int64_t x_stride, float* y, int64_t y_stride, int rows, int n, int block, const float* pre,
           const float* post, void* stream);

// tq_rht of one gathered row: y = H_b (pre * x[idx]) on each run of `block` entries (256-4096), the same bits as a
// gather followed by tq_rht. idx: n int32 indices into x. y must not overlap x.
int tq_rht_gather(const float* x, const int32_t* idx, float* y, int n, int block, const float* pre, void* stream);

// tq_rht of n <= 8 single rows (vectors y[i] of len[i] entries) that share one block (256-4096), in place, in one
// launch: y[i] = post[i] * H_b(y[i]) on each run of `block` entries, the same bits as one tq_rht per vector. post: n
// fp32 vectors, or null (none).
int tq_rht_rows(int n, float* const* y, const int* len, const float* const* post, int block, void* stream);

// A whole-slice side (kind 2 in nariq/FORMAT.md).
// The n entries are `nparts` fused parts in stored order, each cut in `shards` equal contiguous shards; slice r is shard r of every part, side by side (one slice of all n when shards = 1).
// Each slice of length L = n / shards = m 2^k (m the smallest of 1, 12, 20, 68 that leaves a power of two) is multiplied by H_L = (H_m (x) H_2^k) / sqrt(L), H_m the Paley matrices of nariq/FORMAT.md (the H_m index is the slow one).
// Forward (inverse = 0): y = post * H_L (pre * x) per slice; the input side's T x is pre = its signs, post null.
// Inverse: y = post * H_L^T (pre * x) per slice; the output side's T^T x is pre null, post = its signs (times row scales).
// pre / post: fp32 vectors of n entries in stored order, or null. L <= 24576 and 2^k >= 16. parts: host array of nparts sizes (null: one part).
int tq_hadfull(const float* x, int64_t x_stride, float* y, int64_t y_stride, int rows, int n, const int* parts,
               int nparts, int shards, int inverse, const float* pre, const float* post, void* stream);

#ifdef __cplusplus
}
#endif
