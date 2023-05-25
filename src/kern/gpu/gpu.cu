#include <assert.h>
#include "core/tensor.h"
#include "kern/kernel.h"
#include "math.h"
#include "string.h"
#include "utils.h"

namespace inferllm {
namespace gpu {




__global void llm_softmax_compute_float_gpu(const float* src, float* dst,
                                       uint32_t len_row, uint32_t col) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < len_row) {
        const float* psrc = src + row * col;
        float* pdst = dst + row * col;

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

void  llm_softmax_compute_float(const float* src, float* dst,
                                  uint32_t len_row, uint32_t col) {
    auto task = [=](const TaskId& id) {
        for (uint32_t row = id.start; row < id.end; row++) {
            const float* psrc = src + row * col;
            float* pdst = dst + row * col;

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
    };
}



TaskSet llm_embedding_get_int4_float(
        const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd) {
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; ++i) {
            const int row = index[i];
            const int weight_stride =
                    embd * dtype_in_byte(DType::Int4) / dtype_block_size(DType::Int4);
            dequantize_row_q4_0_reference(
                    (static_cast<const char*>(weights) + row * weight_stride),
                    dst + i * embd, embd);
        }
    };
    return TaskSet{{task, len_seq}};
}

__global__ void llm_embedding_get_float_float_gpu(
        const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd) {
    const int row = index[i];
    const int weight_stride = embd;
    memcpy(dst + i * embd, weights + row * weight_stride, embd * sizeof(float));
}
__global void llm_embefing




void llm_embedding_get_float_float(
        const float* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd) {
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; ++i) {
            const int row = index[i];
            const int weight_stride = embd;
            memcpy(dst + i * embd, weights + row * weight_stride, embd * sizeof(float));
        }
    };
    return TaskSet{{task, len_seq}};
}



__global__ void llm_elem



// TaskSet llm_elemwise_compute_float(
//         InData<float> srcs, float* dst, size_t len, ElemMode mode) {
//     MultiThreadingTask task;
//     switch (mode) {
//         case ElemMode::Add: {
//             task = [=](const TaskId& id) {
//                 const float* src0 = srcs[0];
//                 const float* src1 = srcs[1];
//                 for (size_t i = id.start; i < id.end; i++) {
//                     dst[i] = src0[i] + src1[i];
//                 }
//             };
//             break;
//         }
//         case ElemMode::Mul: {
//             task = [=](const TaskId& id) {
//                 const float* src0 = srcs[0];
//                 const float* src1 = srcs[1];
//                 for (size_t i = id.start; i < id.end; i++) {
//                     dst[i] = src0[i] * src1[i];
//                 }
//             };
//             break;
//         }
//         case ElemMode::Silu: {
//             task = [=](const TaskId& id) {
//                 const float* src0 = srcs[0];
//                 for (size_t i = id.start; i < id.end; i++) {
//                     float src = src0[i];
//                     dst[i] = src / (1.0 + exp(-src));
//                 }
//                 return;
//             };
//             break;
//         }
//         case ElemMode::Gelu: {
//             task = [=](const TaskId& id) {
//                 const float* src0 = srcs[0];
//                 for (size_t i = id.start; i < id.end; i++) {
//                     float src = src0[i];
//                     dst[i] = 0.5 * src *
//                              (1 +
//                               tanh(sqrt(2.0 / PI) * (src + PGELU * src * src * src)));
//                 }
//                 return;
//             };
//             break;
//         }
//         default:
//             INFER_ASSERT(0, "Not supported.");
//     }
//     return TaskSet{{task, len}};
// }

}  // namespace gpu
}  // namespace inferllm
