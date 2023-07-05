#include <assert.h>
#include "math.h"
#include "string.h"
#include "utils.h"

#include "core/tensor.h"
#include "kernel.h"
#include "optimized.h"
#include "quantize.h"

using namespace inferllm;

namespace inferllm {
namespace opt {

TaskSet llm_embedding_get_int4_float(
        const void* weights, const uint32_t* index, float* dst, uint32_t len_seq,
        uint32_t embd) {
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; ++i) {
            const int row = index[i];
            const int weight_stride =
                    embd * dtype_in_byte(DType::Int4) / dtype_block_size(DType::Int4);
            dequantize_row_q4_0(
                    (static_cast<const char*>(weights) + row * weight_stride),
                    dst + i * embd, embd);
        }
    };
    return TaskSet{{task, len_seq}};
}

TaskSet llm_elemwise_compute_float(
        InData<float> srcs, float* dst, size_t length, ElemMode mode) {
    MultiThreadingTask task;
    switch (mode) {
        case ElemMode::Add: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                elemwise_vector_add(
                        len, srcs[0] + offset, srcs[1] + offset, dst + offset);
            };
            break;
        }
        case ElemMode::Mul: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                elemwise_vector_mul(
                        len, srcs[0] + offset, srcs[1] + offset, dst + offset);
            };
            break;
        }
        case ElemMode::Silu: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                return elemwise_vector_silu(len, srcs[0] + offset, dst + offset);
            };
            break;
        }
        case ElemMode::Gelu: {
            task = [=](const TaskId& id) {
                uint32_t offset = id.start;
                uint32_t len = id.end - id.start;
                return elemwise_vector_gelu(len, srcs[0] + offset, dst + offset);
            };
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
    return TaskSet{{task, length}};
}

TaskSet llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0, uint32_t len1,
        ElemMode mode) {
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

TaskSet llm_rms_norm_compute_float(
        const float* src, float* dst, uint32_t seq_len, uint32_t embd, float eps) {
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

TaskSet llm_softmax_compute_float(
        const float* src, float* dst, uint32_t len_row, uint32_t col) {
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
TaskSet llm_matmul_compute_int4_float(
        float* dst, const void* src0, const float* bias, const float* src1, uint32_t M,
        uint32_t N, uint32_t K, void* workspace, uint32_t size) {
    //! src0 is quantized weights, weights store in 32 data as block and a block
    //! share the same scale, src1 is featureMap. src0 layout is {N,
    //! K}, src1 layout is {M, K}, the dst is {M, N}
    INFER_ASSERT(sizeof(float) * K <= size, "workspace is not enough.");
    uint32_t q4off = K / dtype_block_size(DType::Int4);
    uint32_t q8off = K / dtype_block_size(DType::Int8);
    const BlockQ40* x = static_cast<const BlockQ40*>(src0);
    BlockQ80* y = static_cast<BlockQ80*>(workspace);

    //! dequantize input, and store in workspace
    //! becuase the input is small than the weights, quantized the input will
    //! reduce the memory traffic
    auto task1 = [=](const TaskId& id) {
        for (uint32_t m = id.start; m < id.end; m++)
            quantize_row_q8_0(&src1[m * K], &y[m * q8off], K);
    };
    auto task2 = [=](const TaskId& id) {
        size_t lmul = mk_lmul(E16, QK80);
        size_t vt16 = mk_vtype(E16, lmul), vt32 = mk_vtype(E32, lmul);
        for (uint32_t n = id.start; n < id.end; n++) {
            float b0 = bias ? bias[n] : 0.f;
            for (uint32_t m = 0; m < M; m++) {
                const int nb = K / QK80;
                float sumf = b0;
                for (int i = 0; i < nb; i++) {
                    auto& px = x[n * q4off + i];
                    auto& py = y[m * q8off + i];

                    VSET1(e32, m1);
                    asm volatile("vmv.s.x v0, x0\n");
                    for (int sz = 0; sz < QK80;) {
                        int rsz = QK80 - sz;
                        int vl;
                        asm volatile(
                                "vsetvl        x0, %[sz4], %[vt32]\n"  // load q4
                                "vlbu.v        v16,  (%[x])\n"
                                "vand.vi       v24, v16, 0b1111\n"     // save low part
                                "vsrl.vi       v16, v16, 4\n"
                                "vsll.vi       v16, v16, 16\n"         // make high part
                                "vor.vv        v16, v16, v24\n"

                                "vsetvl        %[vl], %[sz8], %[vt16]\n"  // load q8
                                "vadd.vi       v16, v16, -8\n"
                                "vlb.v         v24,  (%[y])\n"

                                "vmul.vv       v16, v16, v24\n"  // mul and sum
                                "vwredsum.vs   v0, v16, v0\n"
                                : [vl] "=r"(vl)
                                : [sz4] "r"(rsz / 2), [vt32] "r"(vt32), [sz8] "r"(rsz),
                                  [vt16] "r"(vt16), [x] "r"(&px.qs[sz / 2]),
                                  [y] "r"(&py.qs[sz]), [ratio] "f"(px.d * py.d));
                        sz += vl;
                    }
                    int ret;
                    VSET1(e32, m1);
                    asm volatile("vmv.x.s %[init], v0\n" : [init] "=r"(ret));
                    sumf += px.d * py.d * ret;
                }
                dst[m * N + n] = sumf;
            }
        }
    };
    return TaskSet{{task1, M}, {task2, N}};
}

size_t llm_matmul_get_workspace_float(uint32_t, uint32_t M, uint32_t N, uint32_t K) {
    return sizeof(float) * K * M;
}

TaskSet llm_matmul_compute_with_head_stride_float(
        float* dst, const float* srck, const float* srcq, uint32_t seqlen,
        uint32_t embd, uint32_t head, uint32_t nr_past) {
    uint32_t sub_embd = embd / head;
    uint32_t length = nr_past + seqlen;
    uint32_t line_stride = embd;

    auto task = [=](const TaskId& id) {
        for (uint32_t h = id.start; h < id.end; h++) {
            auto dst_head = dst + h * seqlen * (nr_past + seqlen);
            auto srck_head = srck + h * sub_embd;
            auto srcq_head = srcq + h * sub_embd;
            for (uint32_t row = 0; row < seqlen; row++) {
                auto p_srcq = srcq_head + row * embd;
                uint32_t len = 0;
                for (; len < length; len++) {
                    auto p_dst = dst_head + row * length + len;
                    auto p_srck = srck_head + len * embd;
                    *p_dst = vmulsum(p_srck, p_srcq, sub_embd, 0);
                }
            }
        }
    };
    return TaskSet{{task, head}};
}

TaskSet llm_head_batched_matmul_compute_float(
        float* dst, const float* v, const float* qk, uint32_t seqlen, uint32_t embd,
        uint32_t head, uint32_t nr_past) {
    uint32_t sub_embd = embd / head;
    uint32_t length = nr_past + seqlen;
    uint32_t line_stride = embd;

    auto task = [=](const TaskId& id) {
        for (uint32_t h = id.start; h < id.end; h++) {
            float* dst_head = dst + h * sub_embd;
            const float* v_head = v + h * sub_embd;
            const float* qk_head = qk + h * seqlen * length;
            comput_matmul_with_dst_uncontinue(
                    dst_head, embd, v_head, embd, qk_head, seqlen, length, sub_embd);
        }
    };
    return TaskSet{{task, head}};
}

}  // namespace opt
}  // namespace inferllm
