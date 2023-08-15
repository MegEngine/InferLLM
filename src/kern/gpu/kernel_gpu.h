#pragma once

#if ENABLE_GPU
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <time.h>
#include <algorithm>
#include <iostream>

using namespace std;
namespace inferllm {
struct cudaHandle {
    cudaStream_t stream{nullptr};
    cublasHandle_t cublas_handle{nullptr};
};

namespace gpu {

void llm_embedding_get_int4_float(
        const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd, cudaHandle* handle);
void llm_embedding_get_float_float(
        const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd, cudaHandle* handle);

void llm_elemwise_compute_float(
        InData<float> srcs, float* dst, size_t len, ElemMode mode, cudaHandle* handle);

void llm_elemwise_compute_float_scale(
        float* src, float* dst, size_t len, float scale, cudaHandle* handle);

void llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0, uint32_t len1,
        ElemMode mode, cudaHandle* handle);

void llm_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps,
        cudaHandle* handle);

void llm_rms_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps,
        cudaHandle* handle);

void llm_softmax_compute_float(
        const float* src, float* dst, uint32_t len_row, uint32_t col,
        cudaHandle* handle);

// compute the softmax of the last dim of src, and store the result in dst
void llm_matmul_compute_int4_float(
        float* dst, const void* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size, cudaHandle* handle);

void llm_matmul_compute_int4_float_packed(
        float* dst, const void* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size, cudaHandle* handle);

void llm_matmul_compute_float_float(
        float* dst, const float* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size, cudaHandle* handle);

size_t llm_matmul_get_workspace_float(uint32_t M, uint32_t N, uint32_t K);

size_t llm_matmul_get_workspace_float_float(
        uint32_t M, uint32_t N, uint32_t K, cudaHandle* handle);

void llm_rope_compute_float(
        float* dst, const float* src0, uint32_t n_past, uint32_t n_rot, RotMode m,
        uint32_t N, uint32_t head, uint32_t embd, cudaHandle* handle);

void llm_glm_rope_compute_float(
        float* dst, const float* src0, uint32_t n_past, uint32_t gmask_positon,
        uint32_t seqlen, uint32_t head, uint32_t embd, cudaHandle* handle);

void llm_diag_mask_inf_float(
        float* dst, const float* src0, uint32_t n_past, uint32_t N, uint32_t head,
        cudaHandle* handle);

void llm_glm_gmask_inf_float(
        float* dst, uint32_t n_past, uint32_t seqlen, uint32_t head,
        cudaHandle* handle);

void llm_scale_diag_mask_inf_float(
        float* dst, const float* src0, float scale, uint32_t n_past, uint32_t seqlen,
        uint32_t head, cudaHandle* handle);

void llm_permute_compute_float(
        float* dst, const float* src0, uint32_t dim0, uint32_t dim1, uint32_t dim2,
        std::vector<uint32_t> param, cudaHandle* handle);

void llm_matmul_compute_with_head_strideq_broadcastk_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t query_group_num, uint32_t nr_past,
        cudaHandle* handle);

void llm_matmul_compute_with_head_stride_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past, cudaHandle* handle);

void llm_head_batched_matmul_broadcastv_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t query_group_num, uint32_t nr_past, cudaHandle* handle);

void llm_head_batched_matmul_compute_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t nr_past, cudaHandle* handle);

template <KernelID Id, typename... Args>
struct Comp {
    static void exec(Args... args, cudaHandle* handle);
};

template <KernelID Id, typename... Args>
struct Space {
    static size_t get(Args... args);
};

}  // namespace gpu

}  // namespace inferllm
#ifdef PartialImplementKernel
#undef PartialImplementKernel
#endif
#ifdef PartialImplementSpace
#undef PartialImplementSpace
#endif

#define PartialImplementKernel(kernel_id, fun)               \
    template <typename... Args>                              \
    struct Comp<KernelID::kernel_id, Args...> {              \
        static void exec(Args... args, cudaHandle* handle) { \
            return fun(std::forward<Args>(args)..., handle); \
        }                                                    \
    };

#define PartialImplementSpace(kernel_id, fun)        \
    template <typename... Args>                      \
    struct Space<KernelID::kernel_id, Args...> {     \
        static size_t get(Args... args) {            \
            return fun(std::forward<Args>(args)...); \
        }                                            \
    };

#define NOImplementKernel(kernel_id)                         \
    template <typename... Args>                              \
    struct Comp<KernelID::kernel_id, Args...> {              \
        static void exec(Args... args, cudaHandle* handle) { \
            INFER_ASSERT(0, "kernel not implement");         \
        }                                                    \
    };

namespace inferllm {
namespace gpu {
PartialImplementKernel(ElemwiseFloat, llm_elemwise_compute_float);
PartialImplementKernel(ElemwiseFloatScale, llm_elemwise_compute_float_scale);
PartialImplementKernel(
        ElemwiseBroadcastDim0Src1Float, llm_elemwise_broadcast_dim0_src1_compute_float);
PartialImplementKernel(NormFloat, llm_norm_compute_float);
PartialImplementKernel(RmsNormFloat, llm_rms_norm_compute_float);
PartialImplementKernel(EmbeddingGetInt4Float, llm_embedding_get_int4_float);
PartialImplementKernel(EmbeddingGetFloatFloat, llm_embedding_get_float_float);
PartialImplementKernel(SoftmaxFloat, llm_softmax_compute_float);
PartialImplementKernel(MatmulInt4Float, llm_matmul_compute_int4_float);
PartialImplementKernel(MatmulFloatFloat, llm_matmul_compute_float_float);
PartialImplementKernel(
        MatmulWithHeadStrideFloat, llm_matmul_compute_with_head_stride_float);
PartialImplementKernel(HeadBatchedMatmulFloat, llm_head_batched_matmul_compute_float);
PartialImplementKernel(DiagMaskFloat, llm_diag_mask_inf_float);
PartialImplementKernel(RopeFloat, llm_rope_compute_float);
PartialImplementKernel(GlmRopeFloat, llm_glm_rope_compute_float);
PartialImplementKernel(ScaleDiagMaskFloat, llm_scale_diag_mask_inf_float);
PartialImplementKernel(GlmGmask, llm_glm_gmask_inf_float);
PartialImplementKernel(PermuteFloat, llm_permute_compute_float);

//! multi query attention
PartialImplementKernel(
        MatmulWithHeadStrideQBroadCastKFloat,
        llm_matmul_compute_with_head_strideq_broadcastk_float);
PartialImplementKernel(
        HeadBatchedMatmulBroadCastVFloat, llm_head_batched_matmul_broadcastv_float);

PartialImplementSpace(MatmulInt4Float, llm_matmul_get_workspace_float);
PartialImplementSpace(MatmulFloatFloat, llm_matmul_get_workspace_float_float);

NOImplementKernel(MatmulInt4FloatPacked);
NOImplementKernel(MatmulInt4WeightReorder);
NOImplementKernel(MatmulInt8Float);
NOImplementKernel(EmbeddingGetInt8Float);

#undef PartialImplementKernel
#undef PartialImplementSpace
#undef NOImplementKernel

}  // namespace gpu
}  // namespace inferllm

#endif