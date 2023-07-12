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
        uint32_t n, const float* src0, const float* src1, float* dst, uint32_t len0,
        uint32_t len1) {
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t embed_loc = index % len1;
        dst[index] = src0[index] + src1[embed_loc];
    }
}

__global__ void llm_elemwise_broadcast_dim0_src1_compute_float_mul_gpu(
        uint32_t n, const float* src0, const float* src1, float* dst, uint32_t len0,
        uint32_t len1) {
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t embed_loc = index % len1;
        dst[index] = src0[index] * src1[embed_loc];
    }
}

void llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0, uint32_t len1,
        ElemMode mode, cudaHandle* handle) {
    uint32_t count = len0 * len1;
    switch (mode) {
        case ElemMode::Add: {
            llm_elemwise_broadcast_dim0_src1_compute_float_add_gpu<<<
                    GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                    count, src0, src1, dst, len0, len1);
            break;
        }
        case ElemMode::Mul: {
            llm_elemwise_broadcast_dim0_src1_compute_float_mul_gpu<<<
                    GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
                    count, src0, src1, dst, len0, len1);
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
}

template <typename Function, typename... Args>
__global__ void ApplyFunction(Function functor, int64_t n, float* ret, Args... args) {
    const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
    for (int64_t i = global_tid; i < n; i += blockDim.x * gridDim.x) {
        ret[i] = functor(i, args...);
    }
}

template <typename Function, typename... Args>
cudaError_t LaunchKernel(Function fun, int64_t n, float* ret, Args... args) {
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    ApplyFunction<Function, Args...><<<num_blocks, kBlockSize>>>(fun, n, ret, args...);
    return cudaPeekAtLastError();
}

__global__ void llm_softmax_compute_float_gpu(
        uint32_t n, const float* src, float* dst, uint32_t len_row, uint32_t col) {
    CUDA_KERNEL_LOOP(index, n) {
        const float* psrc = src + index * col;
        float* pdst = dst + index * col;

        float max = -INFINITY;
        for (uint32_t i = 0; i < col; ++i) {
            if (max < psrc[i])
                max = psrc[i];
        }

        float sum = 0.0f;

        for (uint32_t i = 0; i < col; i++) {
            if (psrc[i] == -INFINITY) {
                pdst[i] = 0.0f;
            } else {
                float val = exp(psrc[i] - max);
                sum += val;
                pdst[i] = val;
            }
        }

        sum = 1.0 / sum;
        for (uint32_t j = 0; j < col; j++) {
            pdst[j] = pdst[j] * sum;
        }
    }
}

void llm_softmax_compute_float(
        const float* src, float* dst, uint32_t len_row, uint32_t col,
        cudaHandle* handle) {
    llm_softmax_compute_float_gpu<<<GET_BLOCKS(len_row), CUDA_NUM_THREADS>>>(
            len_row, src, dst, len_row, col);
}

void llm_embedding_get_float_float(
        const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd, cudaHandle* handle) {
    std::vector<uint32_t> cpu_index(len_seq);
    cudaMemcpy(
            cpu_index.data(), index, len_seq * sizeof(uint32_t),
            cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < len_seq; ++i) {
        const int row = cpu_index[i];
        const int weight_stride = embd;
        cudaMemcpy(
                dst + i * embd, weights + row * weight_stride, embd * sizeof(float),
                cudaMemcpyDeviceToDevice);
    }
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

void llm_elemwise_compute_float(
        InData<float> srcs, float* dst, size_t len, ElemMode mode, cudaHandle* handle) {
    switch (mode) {
        case ElemMode::Add: {
            const float* src0 = srcs[0];
            const float* src1 = srcs[1];
            LaunchKernel(AddFunctor(), len, dst, src0, src1);
            break;
        }
        case ElemMode::Mul: {
            const float* src0 = srcs[0];
            const float* src1 = srcs[1];

            LaunchKernel(MulFunctor(), len, dst, src0, src1);
            break;
        }
        case ElemMode::Silu: {
            const float* src0 = srcs[0];
            LaunchKernel(SiluFunctor(), len, dst, src0);
            break;
        }
        case ElemMode::Gelu: {
            const float* src0 = srcs[0];
            LaunchKernel(GeluFunctor(), len, dst, src0);
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
}

__global__ void llm_rms_norm_compute_float_gpu(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps) {
    CUDA_KERNEL_LOOP(i, seq_len) {
        const float* row = src + i * embd;
        float* out = dst + i * embd;

        float mean = 0.0;
        for (uint32_t j = 0; j < embd; j++) {
            mean += row[j] * row[j];
        }
        mean /= embd;

        const float scale = 1.0 / sqrt(mean + eps);

        for (uint32_t j = 0; j < embd; j++) {
            out[j] = row[j] * scale;
        }
    }
}

void llm_rms_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps,
        cudaHandle* handle) {
    llm_rms_norm_compute_float_gpu<<<GET_BLOCKS(seq_len), CUDA_NUM_THREADS>>>(
            src, dst, seq_len, embd, eps);
}

__global__ void llm_norm_compute_float_gpu(
        uint32_t n, const float* src, float* dst, uint32_t seq_len, uint32_t embd,
        float eps) {
    CUDA_KERNEL_LOOP(i, n) {
        const float* row = src + i * embd;
        float* out = dst + i * embd;

        float mean = 0.0;
        for (uint32_t j = 0; j < embd; j++) {
            mean += row[j];
        }
        mean /= embd;

        float sum2 = 0.0;
        for (uint32_t j = 0; j < embd; j++) {
            float v = row[j] - mean;
            out[j] = v;
            sum2 += v * v;
        }

        const float scale = 1.0 / sqrt(sum2 / embd + eps);

        for (uint32_t j = 0; j < embd; j++) {
            out[j] = out[j] * scale;
        }
    }
}

void llm_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps,
        cudaHandle* handle) {
    llm_norm_compute_float_gpu<<<GET_BLOCKS(seq_len), CUDA_NUM_THREADS>>>(
            seq_len, src, dst, seq_len, embd, eps);
}

__global__ void llm_rope_compute_float_gpu(
        uint32_t n, float* dst, const float* src0, uint32_t n_past, uint32_t n_rot,
        RotMode m, uint32_t seqlen, uint32_t head, uint32_t embd) {
    int mode = static_cast<int>(m);
    int n_dims = n_rot;
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t half_rot = n_rot / 2;

        uint32_t rot_loc = index % half_rot;
        uint32_t head_loc = (index / half_rot) % head;
        uint32_t seq_loc = index / (head * half_rot);

        const int p = (mode == 0 ? n_past + seq_loc : seq_loc);
        const double theta = pow(10000.0, ((double)-rot_loc * 2) / n_dims);

        const double cos_theta = cos(p * theta);
        const double sin_theta = sin(p * theta);

        const float* const src =
                src0 + seq_loc * head * embd + head_loc * embd + rot_loc * 2;
        float* dst_data = dst + seq_loc * head * embd + head_loc * embd + rot_loc * 2;

        double x0 = src[0];
        double x1 = src[1];
        if (mode == 0) {
            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[1] = x0 * sin_theta + x1 * cos_theta;

        } else {
            if (seq_loc >= n_past) {
                dst_data[0] = x0 * cos_theta - x1 * sin_theta;
                dst_data[1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

void llm_rope_compute_float_cpu(
        float* dst, const float* src0, uint32_t n_past, uint32_t n_rot, RotMode m,
        uint32_t seqlen, uint32_t head, uint32_t embd, cudaHandle* handle) {
    int ne2 = seqlen;
    // int ne1 = head;
    // int ne0 = embd;
    int mode = static_cast<int>(m);
    int n_dims = n_rot;

    for (int i1 = 0; i1 < head; i1++) {
        for (int i2 = (mode == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = (mode == 0 ? n_past + i2 : i2);
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const double theta = pow(10000.0, ((double)-i0) / n_dims);

                const double cos_theta = cos(p * theta);
                const double sin_theta = sin(p * theta);

                const float* const src = src0 + i2 * head * embd + i1 * embd + i0;
                float* dst_data = dst + i2 * head * embd + i1 * embd + i0;

                double x0 = src[0];
                double x1 = src[1];

                dst_data[0] = x0 * cos_theta - x1 * sin_theta;
                dst_data[1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}
void llm_rope_compute_float(
        float* dst, const float* src0, uint32_t n_past, uint32_t n_rot, RotMode m,
        uint32_t seqlen, uint32_t head, uint32_t embd, cudaHandle* handle) {
    uint32_t count = seqlen * head * (n_rot / 2);

    llm_rope_compute_float_gpu<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, dst, src0, n_past, n_rot, m, seqlen, head, embd);
}

__global__ void llm_elemwise_compute_float_scale_gpu(
        float* src, float* dst, size_t len, float scale) {
    CUDA_KERNEL_LOOP(i, len) {
        dst[i] = src[i] * scale;
    }
}

void llm_elemwise_compute_float_scale(
        float* src, float* dst, size_t len, float scale, cudaHandle* handle) {
    llm_elemwise_compute_float_scale_gpu<<<GET_BLOCKS(len), CUDA_NUM_THREADS>>>(
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
        uint32_t n, float* dst, const float* src0, float scale, uint32_t n_past,
        uint32_t seqlen, uint32_t head) {
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t seq_loc = (index / (seqlen + n_past)) % seqlen;
        uint32_t len_loc = index % (seqlen + n_past);

        if (len_loc > n_past + seq_loc) {
            dst[index] = -INFINITY;
        } else {
            dst[index] = src0[index] * scale;
        }
    }
}
/**
 * dst :head *seq * (seq +nr_past)
 */

void llm_scale_diag_mask_inf_float(
        float* dst, const float* src0, float scale, uint32_t n_past, uint32_t seqlen,
        uint32_t head, cudaHandle* handle) {
    uint32_t count = head * seqlen * (n_past + seqlen);

    llm_scale_diag_mask_inf_float_gpu<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, dst, src0, scale, n_past, seqlen, head);
}

__global__ void llm_diag_mask_inf_float_gpu(
        uint32_t n, float* dst, const float* src0, uint32_t n_past, uint32_t N,
        uint32_t head) {
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t dim_loc = index % (n_past + N);
        uint32_t seq_loc = (index / (n_past + N)) % N;
        if (dim_loc > n_past + seq_loc) {
            dst[index] = -INFINITY;
        }
    }
}

void llm_diag_mask_inf_float(
        float* dst, const float* src0, uint32_t n_past, uint32_t N, uint32_t head,
        cudaHandle* handle) {
    uint32_t count = head * N * (N + n_past);

    llm_diag_mask_inf_float_gpu<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, dst, src0, n_past, N, head);
}

void llm_permute_compute_float(
        float* dst, const float* src0, uint32_t dim0, uint32_t dim1, uint32_t dim2,
        std::vector<uint32_t> param, cudaHandle* handle) {
    return;
}
/**
 * dst :head *seqlen *(seql)
 */
__global__ void llm_matmul_compute_with_head_stride_float_gpu(
        uint32_t n, float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past) {
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t sub_embd = embd / head;
        uint32_t line_stride = embd;
        uint32_t head_loc = index / (seqlen * (seqlen + nr_past));
        uint32_t seq_loc = (index / (seqlen + nr_past)) % seqlen;
        uint32_t line_loc = index % (seqlen + nr_past);

        auto p_srcq = srcq + head_loc * sub_embd + seq_loc * line_stride;
        auto p_srck = srck + head_loc * sub_embd + line_loc * line_stride;
        float sum = 0.0;

        for (uint32_t k = 0; k < sub_embd; k++) {
            sum += p_srck[k] * p_srcq[k];
        }
        dst[index] = sum;
    }
}

void llm_matmul_compute_with_head_stride_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past, cudaHandle* handle) {
    // 用于计算query和key的点积
    uint32_t count = seqlen * head * (seqlen + nr_past);
    llm_matmul_compute_with_head_stride_float_gpu<<<
            GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, dst, srck, srcq, seqlen, embd, head, nr_past);
}

size_t llm_matmul_get_workspace_float(uint32_t M, uint32_t N, uint32_t K) {
    return M * K * dtype_in_byte(DType::Int8) / dtype_block_size(DType::Int8);
}

size_t llm_matmul_get_workspace_float_float(uint32_t M, uint32_t N, uint32_t K) {
    return 0;
}

/**
 *  dst :seq * head * subdim
 *
 */
__global__ void llm_head_batched_matmul_compute_float_gpu(
        uint32_t n, float* dst, const float* v, const float* qk, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past) {
    uint32_t sub_embd = embd / head;
    uint32_t length = nr_past + seqlen;
    CUDA_KERNEL_LOOP(index, n) {
        uint32_t seq_loc = (index / embd);

        uint32_t head_loc = (index / sub_embd) % head;

        uint32_t sub_embed_loc = index % sub_embd;

        auto p_qk = qk + head_loc * seqlen * length + seq_loc * length;
        auto p_v = v + head_loc * sub_embd + sub_embed_loc;
        float sum = 0.0;

        for (uint32_t k = 0; k < length; k++) {
            sum += p_v[k * embd] * p_qk[k];
        }
        dst[index] = sum;
    }
}

void llm_head_batched_matmul_compute_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t nr_past, cudaHandle* handle) {
    uint32_t count = seqlen * embd;
    llm_head_batched_matmul_compute_float_gpu<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, dst, v, qk, seqlen, embd, head, nr_past);
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
}  // namespace gpu
}  // namespace inferllm