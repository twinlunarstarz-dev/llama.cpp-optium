#pragma once

#include "common.h"

// trellis fork: the CPU trellis ops (GGML_OP_TRELLIS, GGML_OP_TRELLIS_GET_ROWS)

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_trellis(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_trellis_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
size_t ggml_compute_forward_trellis_work_size(const struct ggml_tensor * dst, int n_tasks);
size_t ggml_compute_forward_trellis_get_rows_work_size(const struct ggml_tensor * dst, int n_tasks);

#ifdef __cplusplus
}
#endif
