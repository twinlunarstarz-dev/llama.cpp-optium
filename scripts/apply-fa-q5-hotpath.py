#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(__file__).resolve().parents[1] / "ggml/src/ggml-cuda/fattn.cu"
s = p.read_text()

def rep(old, new):
    global s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"expected one match, found {n}: {old[:120]!r}")
    s = s.replace(old, new, 1)

# Production KV combinations used by the server presets should not require the
# enormous FA_ALL_QUANTS compile-time matrix.  Keep the common symmetric paths
# plus q8 K / q5 V (and its reverse/symmetric q5 variants) in the default build.
rep(
'''#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS
''',
'''#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS
''')

rep(
'''        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
''',
'''        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_0:
''')

rep(
'''#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS
''',
'''#ifndef GGML_CUDA_FA_ALL_QUANTS
    const bool production_mixed_q5 =
        (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q5_0) ||
        (K->type == GGML_TYPE_Q5_0 && V->type == GGML_TYPE_Q8_0);
    if (K->type != V->type && !production_mixed_q5) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS
''')

p.write_text(s)
print("enabled default CUDA FlashAttention coverage for q5_0 and q8_0/q5_0 KV")
