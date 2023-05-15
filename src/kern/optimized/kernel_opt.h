#pragma once
#include "math.h"
#include "string.h"
#include "kern/naive/naive.h"

namespace inferllm {
namespace opt {

TaskSet llm_embedding_get_int4_float(const void* weights, const uint32_t* index,
                                  float* dst, uint32_t len_seq, uint32_t embd);

TaskSet llm_elemwise_compute_float(InData<float> srcs, float* dst, size_t len,
                                ElemMode mode);

TaskSet llm_elemwise_broadcast_dim0_src1_compute_float(const float* src0,
                                                    const float* src1,
                                                    float* dst, uint32_t len0,
                                                    uint32_t len1,
                                                    ElemMode mode);

TaskSet llm_rms_norm_compute_float(const float* src, float* dst, uint32_t seq_len,
                                uint32_t embd);

TaskSet llm_softmax_compute_float(const float* src, float* dst, uint32_t len_row,
                               uint32_t col);

// compute the softmax of the last dim of src, and store the result in dst
TaskSet llm_matmul_compute_int4_float(float* dst, const void* src0,
                                      const float* bias, const float* src1,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      void* workspace, uint32_t size);

size_t llm_matmul_get_workspace_float(uint32_t nr_thread, uint32_t M,
                                      uint32_t N, uint32_t K);

TaskSet llm_matmul_compute_with_head_stride_float(float* dst, const float* srck,
                                               const float* srcq,
                                               uint32_t seqlen, uint32_t embd,
                                               uint32_t head, uint32_t nr_past);

TaskSet llm_head_batched_matmul_compute_float(float* dst, const float* v,
                                           const float* qk, uint32_t seqlen,
                                           uint32_t embd, uint32_t head,
                                           uint32_t nr_past);

template <KernelID Id, typename... Args>
struct Comp {
    static TaskSet get_all_task(Args... args) {
        //! if arm not implement, fallback to naive
        return naive::Comp<Id, Args...>::get_all_task(
                std::forward<Args>(args)...);
    }
};

template <KernelID Id, typename... Args>
struct Space {
    //! if arm not implement, fallback to naive
    static size_t get(Args... args) {
        return naive::Space<Id, Args...>::get(std::forward<Args>(args)...);
    }
};

#ifdef PartialImplementKernel
#undef PartialImplementKernel
#endif
#ifdef PartialImplementSpace
#undef PartialImplementSpace
#endif

#define PartialImplementKernel(kernel_id, fun)       \
    template <typename... Args>                      \
    struct Comp<KernelID::kernel_id, Args...> {      \
        static TaskSet get_all_task(Args... args) {  \
            return fun(std::forward<Args>(args)...); \
        }                                            \
    };

#define PartialImplementSpace(kernel_id, fun)        \
    template <typename... Args>                      \
    struct Space<KernelID::kernel_id, Args...> {     \
        static size_t get(Args... args) {            \
            return fun(std::forward<Args>(args)...); \
        }                                            \
    };

PartialImplementKernel(ElemwiseFloat, llm_elemwise_compute_float);
PartialImplementKernel(ElemwiseBroadcastDim0Src1Float,
                       llm_elemwise_broadcast_dim0_src1_compute_float);
PartialImplementKernel(RmsNormFloat, llm_rms_norm_compute_float);
PartialImplementKernel(EmbeddingGetInt4Float, llm_embedding_get_int4_float);
// PartialImplementKernel(SoftmaxFloat, llm_softmax_compute_float);
PartialImplementKernel(MatmulInt4Float, llm_matmul_compute_int4_float);
PartialImplementKernel(MatmulWithHeadStrideFloat,
                       llm_matmul_compute_with_head_stride_float);
PartialImplementKernel(HeadBatchedMatmulFloat,
                       llm_head_batched_matmul_compute_float);

PartialImplementSpace(MatmulInt4Float, llm_matmul_get_workspace_float);

#undef PartialImplementKernel
#undef PartialImplementSpace

}  // namespace opt
}  // namespace inferllm
