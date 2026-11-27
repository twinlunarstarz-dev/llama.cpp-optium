#include "scale.cuh"
#include "convert.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

template <typename T>
static __global__ void scale_f32(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(x[i]) + bias);
    }
}

template <typename T>
static void scale_f32_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32<T>, launch_params, x, dst, scale, bias, nelements);
}

template <typename T>
static __global__ void scale_f32_strided(const T * x, T * dst, const float scale, const float bias, const int64_t nelements,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const size_t nb0, const size_t nb1, const size_t nb2, const size_t nb3) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        int64_t rem = i;
        const int64_t i0 = rem % ne0; rem /= ne0;
        const int64_t i1 = rem % ne1; rem /= ne1;
        const int64_t i2 = rem % ne2;
        const int64_t i3 = rem / ne2;
        const size_t src_off = (size_t)(i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
        const T * src_ptr = (const T *)((const char *)x + src_off);
        dst[i] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(*src_ptr) + bias);
    }
}

template <typename T>
static void scale_f32_strided_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const size_t nb0, const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32_strided<T>, launch_params, x, dst, scale, bias, nelements, ne0, ne1, ne2, nb0, nb1, nb2, nb3);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    const bool strided = !ggml_is_contiguous(src0);
    if (src0->type == GGML_TYPE_BF16) {
        if (strided) {
            scale_f32_strided_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, ggml_nelements(src0),
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else {
            scale_f32_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, ggml_nelements(src0), stream);
        }
    } else {
        if (strided) {
            scale_f32_strided_cuda((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0),
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else {
            scale_f32_cuda((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
        }
    }
}
