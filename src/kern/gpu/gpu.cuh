#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;

#define N  1024
// elementwise implementation copyed from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/elementwise.cuh
namespace inferllm {
namespace gpu {

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                     sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

template <typename Function, typename... Args>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(Function functor, int64_t n, float *ret, Args... args)
{

    const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
    for (int64_t i = global_tid; i < n; i += blockDim.x * gridDim.x)
    {
        ret[i] = functor(i, args...);
    }
}

template <typename Function, typename... Args>
cudaError_t LaunchKernel(Function fun, int64_t n, float *ret, Args... args)
{
    int num_blocks;
    {
        cudaError_t err = GetNumBlocks(n,&num_blocks);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    ApplyGeneric<<<num_blocks, kBlockSize>>>(fun, n, ret, args...);
    return cudaPeekAtLastError();
}

struct MultiplyFunctor
{
    __device__ float operator()(uint32_t i, float *input1, float *input2) const
    {
        return input1[i] + input2[i];
    }
};











// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int KUIPER_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + KUIPER_CUDA_NUM_THREADS - 1) / KUIPER_CUDA_NUM_THREADS;
}


void  llm_embedding_get_int4_float(const void* weights, const uint32_t* index,
                                     float* dst, uint32_t len_seq,
                                     uint32_t embd);
void llm_embedding_get_float_float(const float* weights,
                                      const uint32_t* index, float* dst,
                                      uint32_t len_seq, uint32_t embd);

void llm_elemwise_compute_float(InData<float> srcs, float* dst, size_t len,
                                   ElemMode mode);

void llm_elemwise_compute_float_scale(float* src, float* dst, size_t len,
                                   float scale);

void llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0,
        uint32_t len1, ElemMode mode);

void llm_norm_compute_float(const float* src, float* dst, uint32_t seq_len,
                               uint32_t embd);

void llm_rms_norm_compute_float(const float* src, float* dst,
                                   uint32_t seq_len, uint32_t embd);

void llm_softmax_compute_float(const float* src, float* dst,
                                  uint32_t len_row, uint32_t col);

// compute the softmax of the last dim of src, and store the result in dst
void llm_matmul_compute_int4_float(float* dst, const void* src0,
                                      const float* bias, const float* src1,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      void* workspace, uint32_t size);
void llm_matmul_compute_float_float(float* dst, const float* src0,
                                      const float* bias, const float* src1,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      void* workspace, uint32_t size);

size_t llm_matmul_get_workspace_float(uint32_t nr_thread, uint32_t M,
                                      uint32_t N, uint32_t K);

size_t llm_matmul_get_workspace_float_float(uint32_t nr_thread, uint32_t M,
                                            uint32_t N, uint32_t K);

void llm_rope_compute_float(float* dst, const float* src0, uint32_t n_past,
                               uint32_t n_rot, RotMode m, uint32_t N,
                               uint32_t head, uint32_t embd);

void llm_glm_rope_compute_float(float* dst, const float* src0,
                                   uint32_t n_past, uint32_t gmask_positon,
                                   uint32_t seqlen, uint32_t head,
                                   uint32_t embd);

void llm_diag_mask_inf_float(float* dst, const float* src0, uint32_t n_past,
                                uint32_t N, uint32_t head);

void llm_glm_gmask_inf_float(float* dst, uint32_t n_past, uint32_t seqlen,
                                uint32_t head);

void llm_scale_diag_mask_inf_float(float* dst, const float* src0,
                                      float scale, uint32_t n_past,
                                      uint32_t seqlen, uint32_t head);

void llm_permute_compute_float(float* dst, const float* src0, uint32_t dim0,
                                  uint32_t dim1, uint32_t dim2,
                                  std::vector<uint32_t> param);

void llm_matmul_compute_with_head_stride_float(float* dst, const float* srck,
                                                  const float* srcq,
                                                  uint32_t seqlen,
                                                  uint32_t embd, uint32_t head,
                                                  uint32_t nr_past);

void llm_head_batched_matmul_compute_float(float* dst, const float* v,
                                              const float* qk, uint32_t seqlen,
                                              uint32_t embd, uint32_t head,
                                              uint32_t nr_past);
template <KernelID Id, typename... Args>
struct Comp {
    static void get_all_task(Args... args);
};

template <KernelID Id, typename... Args>
struct Space {
    static size_t get(Args... args);
};

}  // namespace naive

}  // namespace inferllm
#ifdef PartialImplementKernel
#undef PartialImplementKernel
#endif
#ifdef PartialImplementSpace
#undef PartialImplementSpace
#endif

#define PartialImplementKernel(kernel_id, fun)       \
    template <typename... Args>                      \
    struct Comp<KernelID::kernel_id, Args...> {      \
        static void get_all_task(Args... args) {  \
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

namespace inferllm {
namespace gpu {
PartialImplementKernel(ElemwiseFloat, llm_elemwise_compute_float);
PartialImplementKernel(ElemwiseFloatScale, llm_elemwise_compute_float_scale);
// PartialImplementKernel(ElemwiseBroadcastDim0Src1Float,
//                        llm_elemwise_broadcast_dim0_src1_compute_float);
// PartialImplementKernel(NormFloat, llm_norm_compute_float);
// PartialImplementKernel(RmsNormFloat, llm_rms_norm_compute_float);
// PartialImplementKernel(EmbeddingGetInt4Float, llm_embedding_get_int4_float);
// PartialImplementKernel(EmbeddingGetFloatFloat, llm_embedding_get_float_float);
// PartialImplementKernel(SoftmaxFloat, llm_softmax_compute_float);
// PartialImplementKernel(MatmulInt4Float, llm_matmul_compute_int4_float);
// PartialImplementKernel(MatmulFloatFloat, llm_matmul_compute_float_float);
// PartialImplementKernel(MatmulWithHeadStrideFloat,
//                        llm_matmul_compute_with_head_stride_float);
// PartialImplementKernel(HeadBatchedMatmulFloat,
//                        llm_head_batched_matmul_compute_float);
// PartialImplementKernel(DiagMaskFloat, llm_diag_mask_inf_float);
// PartialImplementKernel(RopeFloat, llm_rope_compute_float);
// PartialImplementKernel(GlmRopeFloat, llm_glm_rope_compute_float);
// PartialImplementKernel(ScaleDiagMaskFloat, llm_scale_diag_mask_inf_float);
// PartialImplementKernel(GlmGmask, llm_glm_gmask_inf_float);
// PartialImplementKernel(PermuteFloat, llm_permute_compute_float);

// PartialImplementSpace(MatmulInt4Float, llm_matmul_get_workspace_float);
// PartialImplementSpace(MatmulFloatFloat, llm_matmul_get_workspace_float_float);

#undef PartialImplementKernel
#undef PartialImplementSpace

}  // namespace naive
}  // namespace inferllm
