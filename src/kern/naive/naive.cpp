#include "core/tensor.h"
#include "kern/kernel.h"
#include "math.h"
#include "string.h"
#include "utils.h"
#include <assert.h>

namespace inferllm {
namespace naive {

TaskSet llm_embedding_get_int4_float(const void* weights, const uint32_t* index,
                                     float* dst, uint32_t len_seq,
                                     uint32_t embd) {
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; ++i) {
            const int row = index[i];
            const int weight_stride = embd * dtype_in_byte(DType::Int4) /
                                      dtype_block_size(DType::Int4);
            dequantize_row_q4_0_reference(
                    (static_cast<const char*>(weights) + row * weight_stride),
                    dst + i * embd, embd);
        }
    };
    return TaskSet{{task, len_seq}};
}

TaskSet llm_embedding_get_float_float(const float* weights,
                                      const uint32_t* index, float* dst,
                                      uint32_t len_seq, uint32_t embd) {
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; ++i) {
            const int row = index[i];
            const int weight_stride = embd;
            memcpy(dst + i * embd, weights + row * weight_stride,
                   embd * sizeof(float));
        }
    };
    return TaskSet{{task, len_seq}};
}

TaskSet llm_elemwise_compute_float(InData<float> srcs, float* dst, size_t len,
                                ElemMode mode) {
    MultiThreadingTask task;
    switch (mode) {
        case ElemMode::Add: {
            task = [=](const TaskId& id) {
                const float* src0 = srcs[0];
                const float* src1 = srcs[1];
                for (size_t i = id.start; i < id.end; i++) {
                    dst[i] = src0[i] + src1[i];
                }
            };
            break;
        }
        case ElemMode::Mul: {
            task = [=](const TaskId& id) {
                const float* src0 = srcs[0];
                const float* src1 = srcs[1];
                for (size_t i = id.start; i < id.end; i++) {
                    dst[i] = src0[i] * src1[i];
                }
            };
            break;
        }
        case ElemMode::Silu: {
            task = [=](const TaskId& id) {
                const float* src0 = srcs[0];
                for (size_t i = id.start; i < id.end; i++) {
                    float src = src0[i];
                    dst[i] = src / (1.0 + exp(-src));
                }
                return;
            };
            break;
        }
        case ElemMode::Gelu: {
            task = [=](const TaskId& id) {
                const float* src0 = srcs[0];
                for (size_t i = id.start; i < id.end; i++) {
                    float src = src0[i];
                    dst[i] = 0.5 * src *
                             (1 + tanh(sqrt(2.0 / PI) *
                                       (src + PGELU * src * src * src)));
                }
                return;
            };
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
    return TaskSet{{task, len}};
}

TaskSet llm_elemwise_compute_float_scale(float* src, float* dst, size_t len,
                                         float scale) {
    MultiThreadingTask task;
    task = [=](const TaskId& id) {
        for (size_t i = id.start; i < id.end; i++) {
            dst[i] = src[i] * scale;
        }
    };
    return TaskSet{{task, len}};
}

TaskSet llm_elemwise_broadcast_dim0_src1_compute_float(
        const float* src0, const float* src1, float* dst, uint32_t len0,
        uint32_t len1, ElemMode mode) {
    MultiThreadingTask task;
    switch (mode) {
        case ElemMode::Add: {
            task = [=](const TaskId& id) {
                for (size_t i = id.start; i < id.end; i++) {
                    auto p_src = src0 + i * len1;
                    auto p_dst = dst + i * len1;
                    for (size_t j = 0; j < len1; j++) {
                        p_dst[j] = p_src[j] + src1[j];
                    }
                }
            };
            break;
        }
        case ElemMode::Mul: {
            task = [=](const TaskId& id) {
                for (size_t i = id.start; i < id.end; i++) {
                    auto p_src = src0 + i * len1;
                    auto p_dst = dst + i * len1;
                    for (size_t j = 0; j < len1; j++) {
                        p_dst[j] = p_src[j] * src1[j];
                    }
                }
            };
            break;
        }
        default:
            INFER_ASSERT(0, "Not supported.");
    }
    return TaskSet{{task, len0}};
}

TaskSet llm_norm_compute_float(const float* src, float* dst, uint32_t seq_len,
                               uint32_t embd) {
    const float eps = 1e-5f;
    auto task = [=](const TaskId& id) {
        for (uint32_t i = id.start; i < id.end; i++) {
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
    };
    return TaskSet{{task, seq_len}};
}

TaskSet llm_rms_norm_compute_float(const float* src, float* dst,
                                   uint32_t seq_len, uint32_t embd) {
    const float eps = 1e-5f;
    auto task = [=](const TaskId& id) {
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
    };
    return TaskSet{{task, seq_len}};
}

TaskSet llm_softmax_compute_float(const float* src, float* dst,
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
            quantize_row_q8_0_reference(src1 + m * K, q_src1, K);
        }
    };
    int8_t* q_src = static_cast<int8_t*>(workspace);
    auto task2 = [=](const TaskId& id) {
        for (uint32_t n = id.start; n < id.end; n++) {
            const void* q_weight =
                    static_cast<const uint8_t*>(src0) + n * weight_q40_stride;
            float b = bias ? bias[n] : 0.0f;
            for (uint32_t m = 0; m < M; m++) {
                int8_t* src = q_src + m * weight_q80_stride;
                dst[m * N + n] =
                        vec_vec_dot_q40_with_q80_reference(K, q_weight, src) +
                        b;
            }
        }
    };
    return TaskSet{{task1, M}, {task2, N}};
}

size_t llm_matmul_get_workspace_float(uint32_t nr_thread, uint32_t M,
                                      uint32_t N, uint32_t K) {
    return M * K * dtype_in_byte(DType::Int8) / dtype_block_size(DType::Int8);
}

size_t llm_matmul_get_workspace_float_float(uint32_t nr_thread, uint32_t M,
                                            uint32_t N, uint32_t K) {
    return 0;
}

// compute the softmax of the last dim of src, and store the result in dst
TaskSet llm_matmul_compute_float_float(float* dst, const float* src0,
                                       const float* bias, const float* src1,
                                       uint32_t M, uint32_t N, uint32_t K,
                                       void* workspace, uint32_t size) {
    const float* src = src1;
    auto task = [=](const TaskId& id) {
        for (uint32_t n = id.start; n < id.end; n++) {
            const float* weight = src0 + n * K;
            float b = bias ? bias[n] : 0.0f;
            for (uint32_t m = 0; m < M; m++) {
                const float* p_src = src + m * K;
                dst[m * N + n] = vec_vec_dot_float_with_float_reference(
                                         K, weight, p_src) +
                                 b;
            }
        }
    };
    return TaskSet{{task, N}};
}

TaskSet llm_rope_compute_float(float* dst, const float* src0, uint32_t n_past,
                               uint32_t n_rot, RotMode m, uint32_t N,
                               uint32_t head, uint32_t embd) {
    int ne2 = N;
    int ne1 = head;
    int ne0 = embd;
    int mode = static_cast<int>(m);
    int n_dims = n_rot;

    auto task = [=](const TaskId& id) {
        for (int i1 = id.start; i1 < id.end; i1++) {
            for (int i2 = (mode == 0 ? 0 : n_past); i2 < ne2; i2++) {
                const int p = (mode == 0 ? n_past + i2 : i2);
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const double theta = pow(10000.0, ((double)-i0) / n_dims);

                    const double cos_theta = cos(p * theta);
                    const double sin_theta = sin(p * theta);

                    const float* const src =
                            src0 + i2 * head * embd + i1 * embd + i0;
                    float* dst_data = dst + i2 * head * embd + i1 * embd + i0;

                    double x0 = src[0];
                    double x1 = src[1];

                    dst_data[0] = x0 * cos_theta - x1 * sin_theta;
                    dst_data[1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    };
    return TaskSet{{task, ne1}};
}

TaskSet llm_glm_rope_compute_float(float* dst, const float* src0,
                                   uint32_t n_past, uint32_t gmask_positon,
                                   uint32_t seqlen, uint32_t head,
                                   uint32_t embd) {
    bool prefill = false;
    if (n_past == 0) {
        prefill = true;
    }
    int quart_embd = embd / 4;
    int half_embd = embd / 2;
    auto task = [=](const TaskId& id) {
        for (int h = id.start; h < id.end; h++) {
            for (int seq = 0; seq < seqlen; seq++) {
                int position_id = std::min(seq + n_past, gmask_positon);
                int block_position_id =
                        std::max((int)(n_past + seq) - (int)gmask_positon, 0);
                for (int p = 0; p < quart_embd; p++) {
                    const double theta =
                            pow(10000.0, ((double)-2 * p) / (half_embd));
                    const double cos_theta = cos(position_id * theta);
                    const double sin_theta = sin(position_id * theta);

                    const double cos_theta_b = cos(block_position_id * theta);
                    const double sin_theta_b = sin(block_position_id * theta);

                    //! first half
                    {
                        const float* const src =
                                src0 + seq * head * embd + h * embd + p;
                        float* dst_data =
                                dst + seq * head * embd + h * embd + p;
                        double x0 = src[0];
                        double x32 = src[quart_embd];
                        dst_data[0] = x0 * cos_theta - x32 * sin_theta;
                        dst_data[quart_embd] = x32 * cos_theta + x0 * sin_theta;
                    }
                    //! second half
                    {
                        const float* const src = src0 + seq * head * embd +
                                                 h * embd + half_embd + p;
                        float* dst_data = dst + seq * head * embd + h * embd +
                                          half_embd + p;
                        double x0 = src[0];
                        double x32 = src[quart_embd];
                        dst_data[0] = x0 * cos_theta_b - x32 * sin_theta_b;
                        dst_data[quart_embd] = x32 * cos_theta_b + x0 * sin_theta_b;
                    }
                }
            }
        }
    };
    return TaskSet{{task, head}};
}

TaskSet llm_diag_mask_inf_float(float* dst, const float* src0, uint32_t n_past,
                                uint32_t N, uint32_t head) {
    const int nc = n_past + N;
    const int nr = N;
    const int nz = head;

    auto task = [=](const TaskId& id) {
        for (int k = id.start; k < id.end; k++) {
            for (int j = 0; j < nr; j++) {
                for (uint32_t i = n_past; i < nc; i++) {
                    if (i > n_past + j) {
                        dst[k * nc * nr + j * nc + i] = -INFINITY;
                    }
                }
            }
        }
    };
    return TaskSet{{task, nz}};
}

TaskSet llm_glm_gmask_inf_float(float* dst, uint32_t n_past, uint32_t seqlen,
                                uint32_t head) {
    //! set every head the last number of data to -inf of every row expect
    //! the
    //! last row
    const int nc = n_past + seqlen;
    auto task = [=](const TaskId& id) {
        for (int k = id.start; k < id.end; k++) {
            for (int j = 0; j < seqlen - 1; j++) {
                dst[k * nc * seqlen + j * nc + nc - 1] = -INFINITY;
            }
        }
    };
    return TaskSet{{task, head}};
}

TaskSet llm_scale_diag_mask_inf_float(float* dst, const float* src0,
                                      float scale, uint32_t n_past,
                                      uint32_t seqlen, uint32_t head) {
    const int nc = n_past + seqlen;
    const int nr = seqlen;
    const int nz = head;

    auto task = [=](const TaskId& id) {
        for (int k = id.start; k < id.end; k++) {
            for (int j = 0; j < seqlen; j++) {
                for (uint32_t i = 0; i < nc; i++) {
                    uint32_t index = k * nc * nr + j * nc + i;
                    if (i > n_past + j) {
                        dst[index] = -INFINITY;
                    } else {
                        dst[index] = src0[index] * scale;
                    }
                }
            }
        }
    };
    return TaskSet{{task, nz}};
}

TaskSet llm_permute_compute_float(float* dst, const float* src0, uint32_t dim0,
                                  uint32_t dim1, uint32_t dim2,
                                  std::vector<uint32_t> param) {
    uint32_t axis0 = param[0];
    uint32_t axis1 = param[1];
    uint32_t axis2 = param[2];

    auto task = [=](const TaskId& id) {
        if (axis0 == 1 && axis1 == 0 && axis2 == 2) {
            for (int i0 = 0; i0 < dim0; i0++) {
                for (int i1 = 0; i1 < dim1; i1++) {
                    const float* p_src = src0 + (i0 * dim1 + i1) * dim2;
                    float* p_dst = dst + (i1 * dim0 + i0) * dim2;
                    memcpy(p_dst, p_src, dim2 * sizeof(float));
                }
            }
        }
    };
    return TaskSet{{task, 1}};
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
            for (uint32_t row = 0; row < seqlen; row++) {
                auto p_srcq = srcq_head + row * line_stride;
                for (uint32_t len = 0; len < length; len++) {
                    auto p_dst = dst_head + row * length + len;
                    auto p_srck = srck_head + len * line_stride;
                    float sum = 0;
                    for (uint32_t k = 0; k < sub_embd; k++) {
                        sum += p_srck[k] * p_srcq[k];
                    }
                    *p_dst = sum;
                }
            }
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
            for (uint32_t row = 0; row < seqlen; row++) {
                auto p_qk = qk_head + row * length;
                for (uint32_t len = 0; len < sub_embd; len++) {
                    auto p_dst = dst_head + row * line_stride + len;
                    auto p_v = v_head + len;
                    float sum = 0;
                    for (uint32_t k = 0; k < length; k++) {
                        sum += p_v[k * embd] * p_qk[k];
                    }
                    *p_dst = sum;
                }
            }
        }
    };
    return TaskSet{{task, head}};
}
}  // namespace naive
}  // namespace inferllm
