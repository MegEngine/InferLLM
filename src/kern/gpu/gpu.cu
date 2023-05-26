#include <assert.h>
#include "core/tensor.h"
#include "kern/kernel.h"
#include "math.h"
#include "string.h"
#include "utils.h"

namespace inferllm {
namespace gpu {

__global__ void llm_softmax_compute_float_gpu(
        const float* src, float* dst, uint32_t len_row, uint32_t col) {
    CUDA_KERNEL_LOOP(index, len_row) {
        const float* psrc = src + index * col;
        float* pdst = dst + index * col;

        float max = -INFINITY;
        for (uint32_t i = 0; i < col; ++i) {
            max = std::max(max, psrc[i]);
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
        const float* src, float* dst, uint32_t len_row, uint32_t col) {
    llm_softmax_compute_float_gpu<<<GET_BLOCKS(len_row), CUDA_NUM_THREADS>>>(
            src, dst, len_row, col);
}

__global__ llm_norm_compute_float_gpu(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd) {
    CUDA_KERNEL_LOOP(i, seq_len) {
        const float eps = 1e-5f;
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

void llm_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd) {
    llm_norm_compute_float_gpu<<<GET_BLOCKS(len_row), CUDA_NUM_THREADS>>>(
            src, dst, seq_len, embd);
}

// TaskSet llm_embedding_get_int4_float(
//         const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
//         uint32_t embd) {
//     auto task = [=](const TaskId& id) {
//         for (uint32_t i = id.start; i < id.end; ++i) {
//             const int row = index[i];
//             const int weight_stride =
//                     embd * dtype_in_byte(DType::Int4) / dtype_block_size(DType::Int4);
//             dequantize_row_q4_0_reference(
//                     (static_cast<const char*>(weights) + row * weight_stride),
//                     dst + i * embd, embd);
//         }
//     };
//     return TaskSet{{task, len_seq}};
// }

// __global__ void llm_embedding_get_float_float_gpu(
//         const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
//         uint32_t embd) {
//     // const int row = index[i];
//     // const int weight_stride = embd;

//     // for ()

//     // memcpy(dst + i * embd, weights + row * weight_stride, embd * sizeof(float));
// }

// //
// void llm_embedding_get_float_float(
//         const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
//         uint32_t embd) {
//     auto task = [=](const TaskId& id) {
//         for (uint32_t i = id.start; i < id.end; ++i) {
//             const int row = index[i];
//             const int weight_stride = embd;
//             memcpy(dst + i * embd, weights + row * weight_stride, embd *
//             sizeof(float));
//         }
//     };
//     return TaskSet{{task, len_seq}};
// }

struct SiluFunctor {
    __device__ float operator()(uint32_t i, float* input) const {
        float src = input[i];
        return src / (1.0 + exp(-src));
    }
};

struct GeluFunctor {
    __device__ float operator()(uint32_t i, float* input) const {
        float src = input[i];
        return 0.5 * src * (1 + tanh(sqrt(2.0 / PI) * (src + PGELU * src * src * src)));
    }
}

struct AddFunctor {
    __device__ float operator()(uint32_t i, float* input1, float* input2) const {
        return input1[i] + input2[i];
    }
};

struct MultiplyFunctor {
    __device__ float operator()(uint32_t i, float* input1, float* input2) const {
        return input1[i] + input2[i];
    }
};

void llm_elemwise_compute_float(
        InData<float> srcs, float* dst, size_t len, ElemMode mode) {
    MultiThreadingTask task;
    switch (mode) {
        case ElemMode::Add: {
            const float* src0 = srcs[0];
            const float* src1 = srcs[1];
            LaunchKernel(AddFunctor(), N, dst, src0, src1);
            break;
        }
        case ElemMode::Mul: {
            const float* src0 = srcs[0];
            LaunchKernel(SiluFunctor(), N, dst, src0);
            break;
        }
        case ElemMode::Silu: {
            const float* src0 = srcs[0];
            LaunchKernel(SiluFunctor(), N, dst, src0);
            break;
        }
        case ElemMode::Gelu: {
            {
                const float* src0 = srcs[0];
                LaunchKernel(GeluFunctor(), N, dst, src0);
                break;
            }
            default:
                INFER_ASSERT(0, "Not supported.");
        }
            return TaskSet{{task, len}};
    }

}  // namespace gpu

__global__ llm_rms_norm_compute_float_gpu(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd) {
    CUDA_KERNEL_LOOP(i, seq_len) {
        for (uint32_t i = id.start; i < id.end; i++) {
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
}



void llm_rms_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd) {
    llm_rms_norm_compute_float_gpu<<<GET_BLOCKS(len_row), CUDA_NUM_THREADS>>>(
            src, dst, seq_len, embd);
}

}  // namespace gpu
