#pragma once

#include "kern/naive/naive.h"
#include "math.h"
#include "string.h"

namespace inferllm {
namespace opt {

void init();

TaskSet llm_embedding_get_int4_float(
        const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd);

TaskSet llm_elemwise_compute_float(
        InData<float> srcs, float* dst, size_t len, ElemMode mode);

TaskSet llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0, uint32_t len1,
        ElemMode mode);

TaskSet llm_rms_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps);

TaskSet llm_softmax_compute_float(
        const float* src, float* dst, uint32_t len_row, uint32_t col);

// compute the softmax of the last dim of src, and store the result in dst
TaskSet llm_matmul_compute_int4_float(
        float* dst, const void* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size);

size_t llm_matmul_get_workspace_float(
        uint32_t nr_thread, uint32_t M, uint32_t N, uint32_t K);

TaskSet llm_matmul_compute_with_head_stride_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past);

TaskSet llm_head_batched_matmul_compute_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t nr_past);

PartialImplementKernel(ElemwiseFloat, llm_elemwise_compute_float);
PartialImplementKernel(
        ElemwiseBroadcastDim0Src1Float, llm_elemwise_broadcast_dim0_src1_compute_float);
PartialImplementKernel(RmsNormFloat, llm_rms_norm_compute_float);
PartialImplementKernel(EmbeddingGetInt4Float, llm_embedding_get_int4_float);
PartialImplementKernel(MatmulInt4Float, llm_matmul_compute_int4_float);
PartialImplementKernel(
        MatmulWithHeadStrideFloat, llm_matmul_compute_with_head_stride_float);
PartialImplementKernel(HeadBatchedMatmulFloat, llm_head_batched_matmul_compute_float);

PartialImplementSpace(MatmulInt4Float, llm_matmul_get_workspace_float);

}  // namespace opt
}  // namespace inferllm
