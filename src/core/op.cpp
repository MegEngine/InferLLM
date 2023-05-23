#include "op.h"
#include "kern/kernel.h"
#include "kern/naive/naive.h"

using namespace inferllm;

void LayerNorm::execute(WorkSpace* workspace, uint32_t nr_past) {
    std::shared_ptr<Tensor> weight = nullptr, bias = nullptr;
    if (m_mul) {
        weight = weights()[0];
        DType weight_type = weight->dtype();
        INFER_ASSERT(weight_type == DType::Float32,
                     "layer norm weights must be float32.");
    }
    if (m_bias) {
        bias = weights()[1];
    }
    auto input = inputs()[0];
    auto output = outputs()[0];
    uint32_t seq_len = input->shape()[0];
    uint32_t embd = input->shape()[1];
    auto kernel = get_kernel();
    if (input->dtype() == DType::Float32) {
        const float* src = input->ptr<float>();
        float* dst = output->ptr<float>();
        float *weight_ptr = nullptr, *bias_ptr = nullptr;
        if (m_mul) {
            weight_ptr = weight->ptr<float>();
        }
        if (m_bias) {
            bias_ptr = bias->ptr<float>();
        }
        if (m_rms) {
            kernel->operator()<KernelID::RmsNormFloat>(src, dst, seq_len, embd);
        } else {
            kernel->operator()<KernelID::NormFloat>(src, dst, seq_len, embd);
        }
        if (weight_ptr) {
            kernel->operator()<KernelID::ElemwiseBroadcastDim0Src1Float>(
                    dst, weight_ptr, dst, seq_len, embd, ElemMode::Mul);
        }
        if (bias_ptr) {
            kernel->operator()<KernelID::ElemwiseBroadcastDim0Src1Float>(
                    dst, bias_ptr, dst, seq_len, embd, ElemMode::Add);
        }
    }
}

void Embedding::execute(WorkSpace*, uint32_t) {
    auto weight = weights()[0];
    auto input = inputs()[0];
    auto output = outputs()[0];
    DType weight_type = weight->dtype();
    auto len = input->shape()[0];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        if (weight_type == DType::Int4) {
            kernel->operator()<KernelID::EmbeddingGetInt4Float>(
                    weight->ptr(), input->ptr<uint32_t>(), output->ptr<float>(),
                    len, m_embd);
        } else if (weight_type == DType::Float32) {
            kernel->operator()<KernelID::EmbeddingGetFloatFloat>(
                    weight->ptr<float>(), input->ptr<uint32_t>(), output->ptr<float>(),
                    len, m_embd);
        }
    } else {
        //! fp16
    }
}

void SoftMax::execute(WorkSpace*, uint32_t) {
    auto input = inputs()[0];
    auto output = outputs()[0];
    uint32_t seq_len = input->shape()[0];
    uint32_t embd = input->shape()[1];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        float* src = input->ptr<float>();
        float* dst = output->ptr<float>();
        kernel->operator()<KernelID::SoftmaxFloat>(src, dst, seq_len, embd);
    } else {
        //! fp16
    }
}

void Elemwise::execute(WorkSpace*, uint32_t) {
    auto output = outputs()[0];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        if (m_scale == -INFINITY) {
            InData<float> in_datas;
            for (auto input : inputs()) {
                in_datas.push_back(input->ptr<float>());
            }
            float* dst = output->ptr<float>();
            size_t len = output->length();
            kernel->operator()<KernelID::ElemwiseFloat>(in_datas, dst, len,
                                                        m_mode);
        } else {
            float* dst = output->ptr<float>();
            size_t len = output->length();
            kernel->operator()<KernelID::ElemwiseFloatScale>(
                    inputs()[0]->ptr<float>(), dst, len, m_scale);

            InData<float> in_datas;
            for (auto input : inputs()) {
                in_datas.push_back(input->ptr<float>());
            }
            in_datas[0] = dst;
            kernel->operator()<KernelID::ElemwiseFloat>(in_datas, dst, len,
                                                        m_mode);
        }
    } else {
        //! fp16
    }
}

void MatMul::execute(WorkSpace* workspace, uint32_t) {
    auto N = weights()[0]->shape()[0];
    auto K = weights()[0]->shape()[1];
    auto M = inputs()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto weight_dtype = weights()[0]->dtype();
    void* p_workspace = workspace->ptr();
    uint32_t p_workspace_size = workspace->length();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32 && weight_dtype == DType::Int4) {
        float* dst = outputs()[0]->ptr<float>();
        const void* weight = weights()[0]->ptr();
        const float* bias = nullptr;
        if (m_bias) {
            bias = weights()[1]->ptr<float>();
        }
        const float* src = inputs()[0]->ptr<float>();
        kernel->operator()<KernelID::MatmulInt4Float>(
                dst, weight, bias, src, M, N, K, p_workspace, p_workspace_size);
    }
    if (src_dtype == DType::Float32 && weight_dtype == DType::Float32) {
        float* dst = outputs()[0]->ptr<float>();
        const float* weight = weights()[0]->ptr<float>();
        const float* bias = nullptr;
        if (m_bias) {
            bias = weights()[1]->ptr<float>();
        }
        const float* src = inputs()[0]->ptr<float>();
        kernel->operator()<KernelID::MatmulFloatFloat>(
                dst, weight, bias, src, M, N, K, p_workspace, p_workspace_size);
    }
}

size_t MatMul::get_workspace_in_byte() {
    uint32_t M = inputs()[0]->shape()[0];
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32) {
        return kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), M, N, K);
    }
    return 0;
}

void MatMulLast::execute(WorkSpace* workspace, uint32_t) {
    auto N = weights()[0]->shape()[0];
    auto K = weights()[0]->shape()[1];
    //! only compute the last token
    auto M = 1;
    auto row = inputs()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto weight_dtype = weights()[0]->dtype();
    void* p_workspace = workspace->ptr();
    uint32_t p_workspace_size = workspace->length();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32 && weight_dtype == DType::Int4) {
        float* dst = outputs()[0]->ptr<float>();
        const void* weight = weights()[0]->ptr();
        const float* bias = nullptr;
        if (m_bias) {
            bias = weights()[1]->ptr<float>();
        }
        const float* src = inputs()[0]->ptr<float>() + (row - 1) * K;
        kernel->operator()<KernelID::MatmulInt4Float>(
                dst, weight, bias, src, M, N, K, p_workspace, p_workspace_size);
    }
    if (src_dtype == DType::Float32 && weight_dtype == DType::Float32) {
        float* dst = outputs()[0]->ptr<float>();
        const float* weight = weights()[0]->ptr<float>();
        const float* bias = nullptr;
        if (m_bias) {
            bias = weights()[1]->ptr<float>();
        }
        const float* src = inputs()[0]->ptr<float>();
        kernel->operator()<KernelID::MatmulFloatFloat>(
                dst, weight, bias, src, M, N, K, p_workspace, p_workspace_size);
    }
}

size_t MatMulLast::get_workspace_in_byte() {
    uint32_t M = 1;
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto src_dtype = inputs()[0]->dtype();
    auto kernel = get_kernel();
    if (src_dtype == DType::Float32) {
        return kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), M, N, K);
    }
    return 0;
}

void LlamaAttention::execute(WorkSpace* workspace, uint32_t nr_past) {
    INFER_ASSERT(nr_past == m_kstorage->current_index(),
                 "The index in kv storage is not the same as input\n");
    auto w_dtype = weights()[0]->dtype();
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto in_dtype = input->dtype();
    uint32_t seqlen = input->shape()[0];
    uint32_t embd = input->shape()[1];
    uint32_t head = m_head;
    auto kernel = get_kernel();

    void *p_wq = nullptr, *p_wk = nullptr, *p_wv = nullptr;
    float *p_bq = nullptr, *p_bk = nullptr, *p_bv = nullptr;
    if (m_fused_weights) {
        size_t offset = embd * embd * dtype_in_byte(w_dtype) /
                        dtype_block_size(w_dtype);
        p_wq = weights()[0]->ptr();
        p_wk = static_cast<int8_t*>(p_wq) + offset;
        p_wv = static_cast<int8_t*>(p_wk) + offset;
        if (m_bias) {
            p_bq = weights()[1]->ptr<float>();
            p_bk = p_bq + embd;
            p_bv = p_bk + embd;
        }
    } else {
        p_wq = weights()[0]->ptr();
        p_wk = weights()[1]->ptr();
        p_wv = weights()[2]->ptr();
        if (m_bias) {
            p_bq = weights()[3]->ptr<float>();
            p_bk = weights()[4]->ptr<float>();
            p_bv = weights()[5]->ptr<float>();
        }
    }

    void* p_work = workspace->ptr();
    uint32_t matmul_size = kernel->get_workspace<KernelID::MatmulInt4Float>(
            kernel->nr_thread(), seqlen, embd, embd);
    uint32_t size = workspace->length();

    void* q_out = static_cast<void*>(static_cast<char*>(p_work) + matmul_size);
    void* qk_out = static_cast<void*>(static_cast<char*>(q_out) +
                                      seqlen * m_embd * sizeof(float));

    if (in_dtype == DType::Float32) {
        //! compute k, q, v
        const float* pdata = input->ptr<float>();
        float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
        float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
        float* p_outq = static_cast<float*>(q_out);
        if (w_dtype == DType::Int4) {
            kernel->operator()<KernelID::MatmulInt4Float>(p_outq, p_wq, p_bq,
                                                          pdata, seqlen, embd,
                                                          embd, p_work, size);
            kernel->operator()<KernelID::MatmulInt4Float>(p_outk, p_wk, p_bk,
                                                          pdata, seqlen, embd,
                                                          embd, p_work, size);
            kernel->operator()<KernelID::MatmulInt4Float>(p_outv, p_wv, p_bv,
                                                          pdata, seqlen, embd,
                                                          embd, p_work, size);
        }
        //! rope Q
        kernel->operator()<KernelID::RopeFloat>(p_outq, p_outq, nr_past, m_rot,
                                                RotMode::Mode0, seqlen, head,
                                                embd / head);
        //! rope K
        float* p_totalk = static_cast<float*>(m_kstorage->ptr());
        kernel->operator()<KernelID::RopeFloat>(
                p_totalk, p_totalk, nr_past, m_rot, RotMode::Mode1,
                seqlen + nr_past, head, embd / head);
        //! Q*k with transpose
        kernel->operator()<KernelID::MatmulWithHeadStrideFloat>(
                (float*)qk_out, p_totalk, p_outq, seqlen, embd, head, nr_past);
        //! scale and diag
        float scale = 1.0f / sqrt(float(embd) / head);
        kernel->operator()<KernelID::ScaleDiagMaskFloat>(
                (float*)qk_out, (float*)qk_out, scale, nr_past, seqlen, head);
        //! softmax
        kernel->operator()<KernelID::SoftmaxFloat>(
                (float*)qk_out, (float*)qk_out, head * seqlen,
                nr_past + seqlen);
        //! compute v_out
        float* out = outputs()[0]->ptr<float>();
        float* p_totalv = static_cast<float*>(m_vstorage->ptr());
        kernel->operator()<KernelID::HeadBatchedMatmulFloat>(
                out, p_totalv, (float*)qk_out, seqlen, embd, head, nr_past);
    }
}

size_t LlamaAttention::get_workspace_in_byte() {
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto src_dtype = input->dtype();

    uint32_t M = inputs()[0]->shape()[0];
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto kernel = get_kernel();

    uint32_t seqlen = input->shape()[0];

    size_t total = 0;
    if (src_dtype == DType::Float32) {
        //! matmul tmp
        total += kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), M, N, K);
        //! out q
        total += seqlen * m_embd * sizeof(float);
        //! kq out
        total += m_head * seqlen * m_ctx * sizeof(float);
    }
    return total;
}

void GlmAttention::execute(WorkSpace* workspace, uint32_t nr_past) {

    INFER_ASSERT(nr_past == m_kstorage->current_index(),
                 "The index in kv storage is not the same as input\n");
    auto w_dtype = weights()[0]->dtype();
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto in_dtype = input->dtype();
    uint32_t seqlen = input->shape()[0];
    uint32_t embd = input->shape()[1];
    uint32_t head = m_head;
    auto kernel = get_kernel();
    if (nr_past == 0) {
        INFER_ASSERT(seqlen > 2,
                     "seqlen is too short, must end with gmask and end token");
        m_gmask_position = seqlen - 2;
    }

    void *p_wq = nullptr, *p_wk = nullptr, *p_wv = nullptr;
    float *p_bq = nullptr, *p_bk = nullptr, *p_bv = nullptr;
    if (m_fused_weights) {
        size_t offset = embd * embd * dtype_in_byte(w_dtype) /
                        dtype_block_size(w_dtype);
        p_wq = weights()[0]->ptr();
        p_wk = static_cast<int8_t*>(p_wq) + offset;
        p_wv = static_cast<int8_t*>(p_wk) + offset;
        if (m_bias) {
            p_bq = weights()[1]->ptr<float>();
            p_bk = p_bq + embd;
            p_bv = p_bk + embd;
        }
    } else {
        p_wq = weights()[0]->ptr();
        p_wk = weights()[1]->ptr();
        p_wv = weights()[2]->ptr();
        if (m_bias) {
            p_bq = weights()[3]->ptr<float>();
            p_bk = weights()[4]->ptr<float>();
            p_bv = weights()[5]->ptr<float>();
        }
    }

    auto weight_type = weights()[0]->dtype();

    void* p_work = workspace->ptr();
    size_t matmul_size = 0;
    if (weight_type == DType::Int4) {
        matmul_size = kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), seqlen, embd, embd);
    } else if (weight_type == DType::Float32) {
        matmul_size = kernel->get_workspace<KernelID::MatmulFloatFloat>(
                kernel->nr_thread(), seqlen, embd, embd);
    }
    uint32_t size = workspace->length();

    void* q_out = static_cast<void*>(static_cast<char*>(p_work) + matmul_size);
    void* qk_out = static_cast<void*>(static_cast<char*>(q_out) +
                                      seqlen * m_embd * sizeof(float));

    if (in_dtype == DType::Float32) {
        //! compute k, q, v
        const float* pdata = input->ptr<float>();
        float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
        float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
        float* p_outq = static_cast<float*>(q_out);
        if (w_dtype == DType::Int4) {
            kernel->operator()<KernelID::MatmulInt4Float>(p_outq, p_wq, p_bq,
                                                          pdata, seqlen, embd,
                                                          embd, p_work, size);
            kernel->operator()<KernelID::MatmulInt4Float>(p_outk, p_wk, p_bk,
                                                          pdata, seqlen, embd,
                                                          embd, p_work, size);
            kernel->operator()<KernelID::MatmulInt4Float>(p_outv, p_wv, p_bv,
                                                          pdata, seqlen, embd,
                                                          embd, p_work, size);
        } else if (w_dtype == DType::Float32) {
            kernel->operator()<KernelID::MatmulFloatFloat>(
                    p_outq, (float*)p_wq, p_bq, pdata, seqlen, embd, embd,
                    p_work, size);
            kernel->operator()<KernelID::MatmulFloatFloat>(
                    p_outk, (float*)p_wk, p_bk, pdata, seqlen, embd, embd,
                    p_work, size);
            kernel->operator()<KernelID::MatmulFloatFloat>(
                    p_outv, (float*)p_wv, p_bv, pdata, seqlen, embd, embd,
                    p_work, size);
        }
        //! rope Q
        kernel->operator()<KernelID::GlmRopeFloat>(p_outq, p_outq, nr_past,
                                                   m_gmask_position, seqlen,
                                                   head, embd / head);
        //! scale Q
        float scale_q = 1 / ((m_layer_id + 1) * sqrt(embd / head));
        kernel->operator()<KernelID::ElemwiseFloatScale>(
                p_outq, p_outq, seqlen * embd, scale_q);
        //! rope K
        float* p_totalk = static_cast<float*>(m_kstorage->ptr());
        kernel->operator()<KernelID::GlmRopeFloat>(p_outk, p_outk, nr_past,
                                                   m_gmask_position, seqlen,
                                                   head, embd / head);
        //! Q*k with transpose
        kernel->operator()<KernelID::MatmulWithHeadStrideFloat>(
                (float*)qk_out, p_totalk, p_outq, seqlen, embd, head, nr_past);
        //! qk * (layer + 1)
        kernel->operator()<KernelID::ElemwiseFloatScale>(
                (float*)qk_out, (float*)qk_out,
                head * seqlen * (nr_past + seqlen), (m_layer_id + 1));
        if (seqlen > 1) {
            //! configure the gMask
            kernel->operator()<KernelID::GlmGmask>((float*)qk_out, nr_past,
                                                   seqlen, head);
        }
        //! softmax
        kernel->operator()<KernelID::SoftmaxFloat>(
                (float*)qk_out, (float*)qk_out, head * seqlen,
                nr_past + seqlen);
        //! compute v_out
        float* out = outputs()[0]->ptr<float>();
        float* p_totalv = static_cast<float*>(m_vstorage->ptr());
        kernel->operator()<KernelID::HeadBatchedMatmulFloat>(
                out, p_totalv, (float*)qk_out, seqlen, embd, head, nr_past);
    }
}

size_t GlmAttention::get_workspace_in_byte() {
    auto out = outputs()[0];
    auto input = inputs()[0];
    auto src_dtype = input->dtype();
    auto w_dtype = weights()[0]->dtype();

    uint32_t M = inputs()[0]->shape()[0];
    uint32_t K = inputs()[0]->shape()[1];
    uint32_t N = weights()[0]->shape()[0];
    auto kernel = get_kernel();

    uint32_t seqlen = input->shape()[0];

    size_t total = 0;
    if (src_dtype == DType::Float32 && w_dtype == DType::Int4) {
        //! matmul tmp
        total += kernel->get_workspace<KernelID::MatmulInt4Float>(
                kernel->nr_thread(), M, N, K);
        //! out q
        total += seqlen * m_embd * sizeof(float);
        //! kv out
        total += m_head * seqlen * m_ctx * sizeof(float);
    }
    if (src_dtype == DType::Float32 && w_dtype == DType::Float32) {
        //! matmul tmp
        total += kernel->get_workspace<KernelID::MatmulFloatFloat>(
                kernel->nr_thread(), M, N, K);
        //! out q
        total += seqlen * m_embd * sizeof(float);
        //! kv out
        total += m_head * seqlen * m_ctx * sizeof(float);
    }
    return total;
}

void DiagMask::execute(WorkSpace*, uint32_t n_past) {
    auto output = outputs()[0];
    uint32_t head = output->shape()[0];
    uint32_t N = output->shape()[1];
    auto kernel = get_kernel();
    if (output->dtype() == DType::Float32) {
        const float* in_data = inputs()[0]->ptr<float>();
        float* dst = output->ptr<float>();
        kernel->operator()<KernelID::DiagMaskFloat>(dst, in_data, n_past, N,
                                                    head);
    } else {
        //! fp16
    }
}
