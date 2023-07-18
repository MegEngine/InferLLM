#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

#define ENABLE_GPU 1

#include "core/tensor.h"
#include "kern/kernel.h"
#include "kernel_gpu.h"
#include "math.h"
#include "string.h"
#include "utils.h"

namespace inferllm {
namespace gpu {

#define CUDA_KERNEL_LOOP(i, n)                                   \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
constexpr int DequantizedBlockSize = 256;
const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void llm_elemwise_broadcast_dim0_src1_compute_float_add_gpu(
        const float* src0, const float* src1, float* dst, uint32_t len0,
        uint32_t len1) {
    int row = blockIdx.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < len1) {
        int index = row * len1 + col;
        dst[index] = src0[index] + src1[col];
    }
}

__global__ void llm_elemwise_broadcast_dim0_src1_compute_float_mul_gpu(
        const float* src0, const float* src1, float* dst, uint32_t rows,
        uint32_t ncols) {
    int row = blockIdx.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < ncols) {
        int index = row * ncols + col;
        dst[index] = src0[index] * src1[col];
    }
}

void llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t rows, uint32_t ncols,
        ElemMode mode, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    const dim3 block_dims(512, 1, 1);
    const dim3 block_nums((ncols + 511) / 512, rows, 1);
    switch (mode) {
        case ElemMode::Add: {
            llm_elemwise_broadcast_dim0_src1_compute_float_add_gpu<<<
                    block_nums, block_dims, 0, stream>>>(src0, src1, dst, rows, ncols);
            break;
        }
        case ElemMode::Mul: {
            llm_elemwise_broadcast_dim0_src1_compute_float_mul_gpu<<<
                    block_nums, block_dims, 0, stream>>>(src0, src1, dst, rows, ncols);
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
}


__global__ void softmax_f32_cuda(const float* x, float* dst, const int cols) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;
    const float* src = x + row * cols;
    dst = dst + row * cols;

    float max = -INFINITY;
    for (int col = tid; col < cols; col += block_size) {
        const float val = src[col];
        max = val > max ? val : max;
    }

    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float temp = __shfl_xor_sync(0xffffffff, max, mask);
        max = max > temp ? max : temp;
    }

    float sum = 0.0;
    for (int col = tid; col < cols; col += block_size) {
        const float val = expf(src[col] - max);
        sum += val;
        dst[col] = val;
    }

    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    for (int col = tid; col < cols; col += block_size) {
        dst[col] /= sum;
    }
}

void llm_softmax_compute_float(
        const float* src, float* dst, uint32_t len_row, uint32_t col,
        cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    const dim3 block_dims(kNumWaves, 1, 1);
    const dim3 block_nums(1, len_row, 1);
    softmax_f32_cuda<<<block_nums, block_dims, 0, stream>>>(src, dst, col);
}

__global__ void embeding_float_cuda(
        const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd) {
    int seq_id = blockIdx.y;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < embd) {
        uint32_t row = index[seq_id];
        dst = dst + seq_id * embd;
        weights = weights + row * embd;
        dst[thread_id] = weights[thread_id];
    }
}

void llm_embedding_get_float_float(
        const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    const dim3 block_dims(512, 1, 1);
    const dim3 block_nums((embd + 512) / 512, len_seq, 1);
    embeding_float_cuda<<<block_nums, block_dims, 0, stream>>>(
            weights, index, dst, len_seq, embd);
}

__global__ void llm_embedding_get_int4_float_gpu(
        const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd, const int weight_stride) {
    int seq_id = blockIdx.y;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < embd / 2) {
        uint32_t row = index[seq_id];
        dst = dst + seq_id * embd;
        const void* src = (static_cast<const char*>(weights) + row * weight_stride);
        int q40_block_id = thread_id * 2 / QK40;
        int block_offset = thread_id % (QK40 / 2);
        BlockQ40* q40_block = (BlockQ40*)src + q40_block_id;
        float scale = q40_block->d;
        uint8_t value = q40_block->qs[block_offset];
        const int8_t v1 = value & 0xf;
        const int8_t v2 = value >> 4;
        dst[thread_id * 2] = (v1 - 8) * scale;
        dst[thread_id * 2 + 1] = (v2 - 8) * scale;
    }
}

void llm_embedding_get_int4_float(
        const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd, cudaHandle* handle) {
    const int weight_stride = embd * sizeof(BlockQ40) / QK40;
    // one thread compute two data
    int grid_1 = (embd / 2 + DequantizedBlockSize - 1) / DequantizedBlockSize;
    dim3 grid(grid_1, len_seq);
    cudaStream_t stream = handle->stream;
    llm_embedding_get_int4_float_gpu<<<grid, DequantizedBlockSize, 0, stream>>>(
            weights, index, dst, len_seq, embd, weight_stride);
}

struct SiluFunctor {
    __device__ float operator()(uint32_t i, const float* input) const {
        float src = input[i];
        return src / (1.0 + exp(-src));
    }
};

struct GeluFunctor {
    __device__ float operator()(uint32_t i, const float* input) const {
        float src = input[i];
        return 0.5 * src * (1 + tanh(sqrt(2.0 / PI) * (src + PGELU * src * src * src)));
    }
};

struct AddFunctor {
    __device__ float operator()(
            uint32_t i, const float* input1, const float* input2) const {
        return input1[i] + input2[i];
    }
};

struct MulFunctor {
    __device__ float operator()(
            uint32_t i, const float* input1, const float* input2) const {
        return input1[i] * input2[i];
    }
};

template <typename Function, typename... Args>
__global__ void ApplyFunction(Function functor, int64_t n, float* ret, Args... args) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        ret[tid] = functor(tid, args...);
    }
}

template <typename Function, typename... Args>
cudaError_t LaunchKernel(
        Function fun, cudaStream_t stream, int64_t n, float* ret, Args... args) {
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    ApplyFunction<Function, Args...>
            <<<num_blocks, kBlockSize, 0, stream>>>(fun, n, ret, args...);
    return cudaPeekAtLastError();
}

void llm_elemwise_compute_float(
        InData<float> srcs, float* dst, size_t len, ElemMode mode, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    switch (mode) {
        case ElemMode::Add: {
            const float* src0 = srcs[0];
            const float* src1 = srcs[1];
            LaunchKernel(AddFunctor(), stream, len, dst, src0, src1);
            break;
        }
        case ElemMode::Mul: {
            const float* src0 = srcs[0];
            const float* src1 = srcs[1];

            LaunchKernel(MulFunctor(), stream, len, dst, src0, src1);
            break;
        }
        case ElemMode::Silu: {
            const float* src0 = srcs[0];
            LaunchKernel(SiluFunctor(), stream, len, dst, src0);
            break;
        }
        case ElemMode::Gelu: {
            const float* src0 = srcs[0];
            LaunchKernel(GeluFunctor(), stream, len, dst, src0);
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
}

__global__ void rms_norm_f32(const float* x, float* dst, const int ncols, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int WARP_SIZE = blockDim.x;

    float tmp = 0.0f;  // partial sum for thread in warp
    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        const float xi = x[row * ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float mean = tmp / ncols;
    const float scale = 1.0f / sqrtf(mean + eps);

    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        dst[row * ncols + col] = scale * x[row * ncols + col];
    }
}

void llm_rms_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps,
        cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    rms_norm_f32<<<seq_len, kNumWaves, 0, stream>>>(
            src, dst,  embd, eps);
}

__global__ void norm_f32(const float* x, float* dst, const int ncols, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int WARP_SIZE = blockDim.x;

    float mean = 0.0f;  // partial sum for thread in warp
    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        const float xi = x[row * ncols + col];
        mean += xi;
    }
    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        mean += __shfl_xor_sync(0xffffffff, mean, mask, 32);
    }
    mean = mean / ncols;

    float sum = 0.0f;  // partial sum for thread in warp
    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        const float xi = x[row * ncols + col] - mean;
        sum += xi * xi;
        dst[row * ncols + col] = xi;
    }
    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    const float scale = 1.0f / sqrtf(sum / ncols + eps);

    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        dst[row * ncols + col] = scale * x[row * ncols + col];
    }
}

void llm_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps,
        cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    norm_f32<<<seq_len, kNumWaves, 0, stream>>>(src, dst, embd, eps);
}

template <bool halfmode>
__global__ void rope_compute_float(
        float* dst, const float* src, float theta_scale, uint32_t position_offset,
        uint32_t n_rot, uint32_t seqlen, uint32_t n_head, uint32_t head_embd) {
    const int rot = threadIdx.x;
    const int head = blockIdx.x;
    const int seq = blockIdx.y;

    if (rot >= n_rot / 2 || head >= n_head || seq >= seqlen) {
        return;
    }

    const float theta = (position_offset + seq) * powf(theta_scale, rot);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const int offset = seq * n_head * head_embd + head * head_embd;
    if (halfmode) {
        const int half_embd = head_embd / 2;
        const float x0 = src[offset + rot];
        const float x1 = src[offset + rot + half_embd];
        dst[offset + rot] = x0 * cos_theta - x1 * sin_theta;
        dst[offset + rot + half_embd] = x0 * sin_theta + x1 * cos_theta;
    } else {
        const float x0 = src[offset + 2 * rot];
        const float x1 = src[offset + 2 * rot + 1];
        dst[offset + 2 * rot] = x0 * cos_theta - x1 * sin_theta;
        dst[offset + 2 * rot + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

void llm_rope_compute_float(
        float* dst, const float* src, uint32_t n_past, uint32_t n_rot, RotMode m,
        uint32_t seqlen, uint32_t head, uint32_t head_embd, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;

    const float theta_scale = powf(10000.0, -2.0f / n_rot);
    const float position_offset = n_past;

    INFER_ASSERT(n_rot <= 2048, "n_rot is two large.");
    INFER_ASSERT(n_rot % 2 == 0, "n_rot must be even.");
    const dim3 block_dims(n_rot / 2, 1, 1);
    const dim3 block_nums(head, seqlen, 1);
    //! offset to nr_past
    if (m == RotMode::Mode1) {
        src = src + n_past * head_embd * head;
        dst = dst + n_past * head_embd * head;
    }
    if (m == RotMode::ModelRotHalf) {
        rope_compute_float<true><<<block_nums, block_dims, 0, stream>>>(
                dst, src, theta_scale, position_offset, n_rot, seqlen, head, head_embd);
    } else {
        rope_compute_float<false><<<block_nums, block_dims, 0, stream>>>(
                dst, src, theta_scale, position_offset, n_rot, seqlen, head, head_embd);
    }
}

__global__ void llm_elemwise_compute_float_scale_gpu(
        float* src, float* dst, size_t len, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        dst[index] = src[index] * scale;
    }
}

void llm_elemwise_compute_float_scale(
        float* src, float* dst, size_t len, float scale, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    const dim3 block_dims(CUDA_NUM_THREADS, 1, 1);
    const dim3 block_nums((len + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, 1, 1);
    llm_elemwise_compute_float_scale_gpu<<<block_nums, block_dims, 0, stream>>>(
            src, dst, len, scale);
}

void llm_matmul_compute_float_float(
        float* dst, const float* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    cublasHandle_t cublas_handle = handle->cublas_handle;
    float alpha = 1.f;
    float beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUBLAS_CHECK(cublasSgemm(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &alpha, src0, K, src1, K, &beta, dst, N));
    if (bias != nullptr) {
        llm_elemwise_broadcast_dim0_src1_compute_float(
                dst, bias, dst, M, N, ElemMode::Add, handle);
    }
}

__global__ void dequantize_mul_mat_vec(
        const void* dx, const float* y, const float* bias, float* dst, const int M,
        const int N, const int K) {
        const int m_id = blockIdx.y;
    const int n_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    if (m_id >= M || n_id >= N) {
        return;
    }

    const int iter_stride = 2 * 32;
    const int vals_per_iter =
            iter_stride / 32;  // num quantized vals per thread and i iter

    // partial sum for each thread
    float tmp = 0.0f;
    const float* srcy = y + m_id * K;
    dst = dst + m_id * N;
    float bias_val = bias ? bias[n_id] : 0.0f;

    for (int i = 0; i < K; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        const int ib = (n_id * K + col) / QK40;  // x block index
        const int iqs = (col % QK40) / 2;        // x quant index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights
            // per data val
            float2 v;
            const BlockQ40* x = (const BlockQ40*)dx + ib;
            const float d = x->d;
            const int vui = x->qs[iqs];

            v.x = vui & 0xF;
            v.y = vui >> 4;

            v.x = (v.x - 8.0f) * d;
            v.y = (v.y - 8.0f) * d;

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of
            // y_offset = qk/2
            tmp += v.x * srcy[col];
            tmp += v.y * srcy[col + 1];
        }
    }

    // sum up partial sums and write back result
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[n_id] = tmp + bias_val;
    }
}

void llm_matmul_compute_int4_float(
        float* dst, const void* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size, cudaHandle* handle) {
    INFER_ASSERT(K % QK40 == 0, "embd is not the time of QK40.");
    cudaStream_t stream = handle->stream;
    const dim3 block_nums(N + 15 / 16, M, 1);
    const dim3 block_dims(32, 16, 1);
    dequantize_mul_mat_vec<<<block_nums, block_dims, 0, stream>>>(
            src0, src1, bias, dst, M, N, K);
}

__global__ void llm_scale_diag_mask_inf_float_gpu(
        const float* src, float* dst, const int past, const int len, const int head_dim,
        float scale) {
    const int head = blockIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= past + len || row >= len || head >= head_dim)
        return;

    const int row_stride = len + past;
    const int head_stride = len * (len + past);

    src = src + head * head_stride + row * row_stride;
    dst = dst + head * head_stride + row * row_stride;

    dst[col] = (col > past + row) ? -INFINITY : src[col] * scale;
}

void llm_scale_diag_mask_inf_float(
        float* dst, const float* src, float scale, uint32_t past, uint32_t seqlen,
        uint32_t head, cudaHandle* handle) {
    cudaStream_t stream = handle->stream;

    constexpr int kBlockSize = 32;
    const int block_y = (seqlen + kBlockSize - 1) / kBlockSize;
    const int block_x = (past + seqlen + kBlockSize - 1) / kBlockSize;
    const dim3 block_dims(kBlockSize, kBlockSize, 1);
    const dim3 block_nums(block_x, block_y, head);

    llm_scale_diag_mask_inf_float_gpu<<<block_nums, block_dims, 0, stream>>>(
            src, dst, past, seqlen, head, scale);
}

__global__ void diag_mask_inf_f32(
        const float* src, float* dst, const int past, const int len,
        const int head_dim) {
    const int head = blockIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= len || row >= len || head >= head_dim)
        return;

    const int row_stride = len + past;
    const int head_stride = len * (len + past);

    src = src + head * head_stride + row * row_stride + past;
    dst = dst + head * head_stride + row * row_stride + past;
    dst[col] = (col > row) ? -INFINITY : src[col];
}

void llm_diag_mask_inf_float(
        float* dst, const float* src, uint32_t n_past, uint32_t N, uint32_t head,
        cudaHandle* handle) {
    cudaStream_t stream = handle->stream;
    constexpr int kBlockSize = 32;
    const int block_n = (N + kBlockSize - 1) / kBlockSize;
    const dim3 block_dims(kBlockSize, kBlockSize, 1);
    const dim3 block_nums(block_n, block_n, head);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(src, dst, n_past, N, head);
}

void llm_permute_compute_float(
        float* dst, const float* src0, uint32_t dim0, uint32_t dim1, uint32_t dim2,
        std::vector<uint32_t> param, cudaHandle* handle) {
    return;
}
/**
 * dst :head *seqlen *(seql)
 */

void llm_matmul_compute_with_head_stride_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past, cudaHandle* handle) {
    uint32_t head_embd = embd / head;
    uint32_t M = seqlen;
    uint32_t N = seqlen + nr_past;
    uint32_t K = head_embd;
    cudaStream_t stream = handle->stream;
    cublasHandle_t cublas_handle = handle->cublas_handle;
    float alpha = 1.f;
    float beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, srck, embd,
            head_embd, srcq, embd, head_embd, &beta, dst, N, M * N, head));
}

void llm_head_batched_matmul_compute_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t nr_past, cudaHandle* handle) {
    uint32_t head_embd = embd / head;
    uint32_t M = head_embd;
    uint32_t K = seqlen + nr_past;
    uint32_t N = seqlen;
    cudaStream_t stream = handle->stream;
    cublasHandle_t cublas_handle = handle->cublas_handle;
    float alpha = 1.f;
    float beta = 0.f;

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, v, embd,
            head_embd, qk, K, K * N, &beta, dst, embd, head_embd, head));
}

void llm_glm_gmask_inf_float(
        float* dst, uint32_t n_past, uint32_t seqlen, uint32_t head,
        cudaHandle* handle) {
    //! set every head the last number of data to -inf of every row expect
    //! the
    //! last row
    // const int nc = n_past + seqlen;
    // auto task = [=](const TaskId& id) {
    //     for (int k = id.start; k < id.end; k++) {
    //         for (int j = 0; j < seqlen - 1; j++) {
    //             dst[k * nc * seqlen + j * nc + nc - 1] = -INFINITY;
    //         }
    //     }
    // };
}
void llm_glm_rope_compute_float(
        float* dst, const float* src0, uint32_t n_past, uint32_t gmask_positon,
        uint32_t seqlen, uint32_t head, uint32_t embd, cudaHandle* handle) {
    // bool prefill = false;
    // if (n_past == 0) {
    //     prefill = true;
    // }
    // int quart_embd = embd / 4;
    // int half_embd = embd / 2;
    // auto task = [=](const TaskId& id) {
    //     for (int h = id.start; h < id.end; h++) {
    //         for (int seq = 0; seq < seqlen; seq++) {
    //             int position_id = std::min(seq + n_past, gmask_positon);
    //             int block_position_id =
    //                     std::max((int)(n_past + seq) - (int)gmask_positon, 0);
    //             for (int p = 0; p < quart_embd; p++) {
    //                 const double theta = pow(10000.0, ((double)-2 * p) /
    //                 (half_embd)); const double cos_theta = cos(position_id * theta);
    //                 const double sin_theta = sin(position_id * theta);

    //                 const double cos_theta_b = cos(block_position_id * theta);
    //                 const double sin_theta_b = sin(block_position_id * theta);

    //                 //! first half
    //                 {
    //                     const float* const src =
    //                             src0 + seq * head * embd + h * embd + p;
    //                     float* dst_data = dst + seq * head * embd + h * embd + p;
    //                     double x0 = src[0];
    //                     double x32 = src[quart_embd];
    //                     dst_data[0] = x0 * cos_theta - x32 * sin_theta;
    //                     dst_data[quart_embd] = x32 * cos_theta + x0 * sin_theta;
    //                 }
    //                 //! second half
    //                 {
    //                     const float* const src =
    //                             src0 + seq * head * embd + h * embd + half_embd + p;
    //                     float* dst_data =
    //                             dst + seq * head * embd + h * embd + half_embd + p;
    //                     double x0 = src[0];
    //                     double x32 = src[quart_embd];
    //                     dst_data[0] = x0 * cos_theta_b - x32 * sin_theta_b;
    //                     dst_data[quart_embd] = x32 * cos_theta_b + x0 * sin_theta_b;
    //                 }
    //             }
    //         }
    //     }
    // };
}

void llm_matmul_compute_with_head_strideq_broadcastk_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t query_group_num, uint32_t nr_past,
        cudaHandle* handle) {}

void llm_head_batched_matmul_broadcastv_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t query_group_num, uint32_t nr_past, cudaHandle* handle) {
}

size_t llm_matmul_get_workspace_float(uint32_t M, uint32_t N, uint32_t K) {
    return 0;
}

size_t llm_matmul_get_workspace_float_float(uint32_t M, uint32_t N, uint32_t K) {
    return 0;
}
}  // namespace gpu
}  // namespace inferllm