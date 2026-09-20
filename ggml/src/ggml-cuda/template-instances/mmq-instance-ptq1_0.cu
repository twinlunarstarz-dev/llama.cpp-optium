// Bonsai MMQ template instance.
#include "../mmq.cuh"

#if !defined(GGML_USE_HIP)
DECL_MMQ_CASE(GGML_TYPE_PTQ1_0);
#endif
