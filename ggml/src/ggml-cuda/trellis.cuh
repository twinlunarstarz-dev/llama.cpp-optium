#pragma once

#include "common.cuh"

// trellis fork: GGML_OP_TRELLIS (ggml_trellis_mul_mat) through libtrellis
void ggml_cuda_op_trellis(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_trellis_supported(const ggml_tensor * op);

static const int TRELLIS_GROUP_MAX = 8; // tq_gemv_group's members

// An input group: nodes[i], nodes[i + 1], ... are coded linears of the same x at up to 64 columns (attention's q/k/v,
// the MLP's up/gate, a DeltaNet layer's in-projections, the lm_head's bands), each with no input gather and an input
// side the GEMV fuses. Returns how many (1: no group; at most 8).
int ggml_cuda_trellis_group_size(const ggml_cgraph * cgraph, int i);

// The members of an input group as one tq_gemv_group call (one activation pass, one GEMV launch per code width),
// then each member's output side and gather: the same bytes as ggml_cuda_op_trellis on each.
void ggml_cuda_op_trellis_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n);

// The MLP chain: nodes[i], nodes[i + 1] an input group of up and gate, nodes[i + 2] swiglu(gate, up), nodes[i + 3]
// down reading it, with up's and gate's output sides and down's input side rht<b> of one b (256-4096), the three
// intermediate tensors read nowhere else. Returns 4 when they are (0: no chain).
int ggml_cuda_trellis_mlp_chain(const ggml_cgraph * cgraph, int i);

// The chain: the group's raw products, then the two output rotations, the swiglu and down's input rotation as one
// kernel, then down's product. The same bytes as the four nodes one by one; the swiglu node's dst is not written.
void ggml_cuda_op_trellis_mlp_chain(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes);
