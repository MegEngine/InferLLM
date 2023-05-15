#include "math.h"
#include "string.h"
#include "utils.h"
#include <assert.h>

#include "core/tensor.h"
#include "kern/optimized/kernel_opt.h"

#if INFER_X86
#include "kern/optimized/x86/optimized_x86.h"
#include "kern/optimized/x86/quantize.h"
#endif

#if INFER_ARM
#include "kern/optimized/arm/optimized_arm.h"
#include "kern/optimized/arm/quantize.h"
#endif

using namespace inferllm;

namespace inferllm {
namespace opt {
TaskSet llm_embedding_get_int4_float(const void* weights, const uint32_t* index,
                                     float* dst, uint32_t len_seq,
                                     uint32_t embd) {
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; ++i) {
            const int row = index[i];
            const int weight_stride = embd * dtype_in_byte(DType::Int4) /
                                      dtype_block_size(DType::Int4);
            dequantize_row_q4_0(
                    (static_cast<const char*>(weights) + row * weight_stride),
                    dst + i * embd, embd);
        }
    };
    return TaskSet{{task, len_seq}};
}

TaskSet llm_elemwise_compute_float(InData<float> srcs, float* dst,
                                   size_t length, ElemMode mode) {
    MultiThreadingTask task;
    switch (mode) {
        case ElemMode::Add: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                elemwise_vector_add(len, srcs[0] + offset, srcs[1] + offset,
                                    dst + offset);
            };
            break;
        }
        case ElemMode::Mul: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                elemwise_vector_mul(len, srcs[0] + offset, srcs[1] + offset,
                                    dst + offset);
            };
            break;
        }
        case ElemMode::Silu: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                return elemwise_vector_silu(len, srcs[0] + offset,
                                            dst + offset);
            };
            break;
        }
        case ElemMode::Gelu: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                return elemwise_vector_gelu(len, srcs[0] + offset,
                                            dst + offset);
            };
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
    return TaskSet{{task, length}};
}

TaskSet llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0,
        uint32_t len1, ElemMode mode) {
    MultiThreadingTask task;
    switch (mode) {
        case ElemMode::Add: {
            task = [=](const TaskId& id) {
                for (size_t i = id.start; i < id.end; i++) {
                    const float* p_src = src0 + i * len1;
                    float* p_dst = dst + i * len1;
                    elemwise_vector_add(len1, p_src, src1, p_dst);
                }
            };
            break;
        }
        case ElemMode::Mul: {
            task = [=](const TaskId& id) {
                for (size_t i = id.start; i < id.end; i++) {
                    auto p_src = src0 + i * len1;
                    auto p_dst = dst + i * len1;
                    elemwise_vector_mul(len1, p_src, src1, p_dst);
                }
            };
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
    return TaskSet{{task, len0}};
}

TaskSet llm_rms_norm_compute_float(const float* src, float* dst,
                                   uint32_t seq_len, uint32_t embd) {
    const float eps = 1e-5f;
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; i++) {
            const float* row = src + i * embd;
            float* out = dst + i * embd;
            float mean = reduce_square_sum(embd, row) / embd;
            const float scale = 1.0 / sqrt(mean + eps);
            elemwise_vec_scale(embd, row, scale, out);
        }
    };
    return TaskSet{{task, seq_len}};
}

TaskSet llm_softmax_compute_float(const float* src, float* dst,
                                  uint32_t len_row, uint32_t col) {
    auto task = [=](const TaskId& id) {
        for (uint32_t row = id.start; row < id.end; row++) {
            const float* psrc = src + row * col;
            float* pdst = dst + row * col;

            float max = reduce_max(col, psrc);
            float sum = select_sub_max_and_reduce_sum(col, psrc, pdst, max);
            sum = 1.0 / sum;
            elemwise_vec_scale(col, pdst, sum, pdst);
        }
    };
    return TaskSet{{task, len_row}};
}

// compute the softmax of the last dim of src, and store the result in dst
TaskSet llm_matmul_compute_int4_float(float* dst, const void* src0,
                                      const float* bias, const float* src1,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      void* workspace, uint32_t size) {
    //! src0 is quantized weights, weights store in 32 data as block and a block
    //! share the same scale, src1 is featureMap. src0 layout is {N,
    //! K}, src1 layout is {M, K}, the dst is {M, N}
    INFER_ASSERT(sizeof(float) * K <= size, "workspace is not enough.");
    uint32_t weight_q40_stride =
            K * dtype_in_byte(DType::Int4) / dtype_block_size(DType::Int4);
    uint32_t weight_q80_stride =
            K * dtype_in_byte(DType::Int8) / dtype_block_size(DType::Int8);
    //! dequantize input, and store in workspace
    //! becuase the input is small than the weights, quantized the input will
    //! reduce the memory traffic
    auto task1 = [=](const TaskId& id) {
        for (uint32_t m = id.start; m < id.end; m++) {
            BlockQ80* q_src1 = (BlockQ80*)(static_cast<uint8_t*>(workspace) +
                                           m * weight_q80_stride);
            quantize_row_q8_0(src1 + m * K, q_src1, K);
        }
    };
    int8_t* q_src = static_cast<int8_t*>(workspace);
    auto task2 = [=](const TaskId& id) {
        uint32_t N_len = id.end - id.start;
        uint32_t n_block_4 = N_len / 4;
        uint32_t n_block_4_left = N_len - n_block_4 * 4;
        for (uint32_t block4 = 0; block4 < n_block_4; block4++) {
            uint32_t n = block4 * 4 + id.start;
            float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.0f;
            if (bias) {
                b0 = bias[n];
                b1 = bias[n + 1];
                b2 = bias[n + 2];
                b3 = bias[n + 3];
            }
            const void* q_weight0 =
                    static_cast<const uint8_t*>(src0) + n * weight_q40_stride;
            const void* q_weight1 =
                    static_cast<const uint8_t*>(src0) + (n + 1) * weight_q40_stride;
            const void* q_weight2 =
                    static_cast<const uint8_t*>(src0) + (n + 2) * weight_q40_stride;
            const void* q_weight3 =
                    static_cast<const uint8_t*>(src0) + (n + 3) * weight_q40_stride;
            for (uint32_t m = 0; m < M; m++) {
                int8_t* src = q_src + m * weight_q80_stride;
                dst[m * N + n] = vec_vec_dot_q40_with_q80(K, q_weight0, src) + b0;
                dst[m * N + n + 1] = vec_vec_dot_q40_with_q80(K, q_weight1, src) + b1;
                dst[m * N + n + 2] = vec_vec_dot_q40_with_q80(K, q_weight2, src) + b2;
                dst[m * N + n + 3] = vec_vec_dot_q40_with_q80(K, q_weight3, src) + b3;
            }
        }

        for (uint32_t left = 0; left < n_block_4_left; left++) {
            uint32_t n = n_block_4 * 4 + left + id.start;
            float b0 = 0.f;
            if (bias) {
                b0 = bias[n];
            }
            const void* q_weight =
                    static_cast<const uint8_t*>(src0) + n * weight_q40_stride;
            for (uint32_t m = 0; m < M; m++) {
                int8_t* src = q_src + m * weight_q80_stride;
                dst[m * N + n] = vec_vec_dot_q40_with_q80(K, q_weight, src) + b0;
            }
        }
    };
    return TaskSet{{task1, M}, {task2, N}};
}

size_t llm_matmul_get_workspace_float(uint32_t, uint32_t M, uint32_t N,
                                      uint32_t K) {
    return sizeof(float) * K * M;
}

TaskSet llm_matmul_compute_with_head_stride_float(float* dst, const float* srck,
                                                  const float* srcq,
                                                  uint32_t seqlen,
                                                  uint32_t embd, uint32_t head,
                                                  uint32_t nr_past) {
    uint32_t sub_embd = embd / head;
    uint32_t length = nr_past + seqlen;
    uint32_t line_stride = embd;

    auto task = [=](const TaskId& id) {
        for (uint32_t h = id.start; h < id.end; h++) {
            auto dst_head = dst + h * seqlen * (nr_past + seqlen);
            auto srck_head = srck + h * sub_embd;
            auto srcq_head = srcq + h * sub_embd;
            compute_src_offset_embd_matmul(srcq_head, embd, srck_head, embd,
                                           dst_head, seqlen, length, sub_embd);
        }
    };
    return TaskSet{{task, head}};
}

TaskSet llm_head_batched_matmul_compute_float(float* dst, const float* v,
                                              const float* qk, uint32_t seqlen,
                                              uint32_t embd, uint32_t head,
                                              uint32_t nr_past) {
    uint32_t sub_embd = embd / head;
    uint32_t length = nr_past + seqlen;
    uint32_t line_stride = embd;

    auto task = [=](const TaskId& id) {
        for (uint32_t h = id.start; h < id.end; h++) {
            float* dst_head = dst + h * sub_embd;
            const float* v_head = v + h * sub_embd;
            const float* qk_head = qk + h * seqlen * length;
            comput_matmul_with_dst_uncontinue(dst_head, embd, v_head, embd,
                                              qk_head, seqlen, length,
                                              sub_embd);
        }
    };
    return TaskSet{{task, head}};
}
}  // namespace opt
}  // namespace inferllm
