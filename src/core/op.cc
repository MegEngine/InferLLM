/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
 
#include "op.h"

#include "kern/kernel.h"
#include "kern/naive/naive.h"

using namespace llm_learning;

void LayerNorm::execute(WorkSpace* workspace, uint32_t nr_past) {
  auto weight = weights()[0];
  auto input = inputs()[0];
  auto output = outputs()[0];
  uint32_t seq_len = input->shape()[0];
  uint32_t embd = input->shape()[1];
  DType weight_type = weight->dtype();
  INFER_ASSERT(weight_type == DType::Float32, "layer norm weights must be float32.");
  auto kernel = get_kernel();
  if (input->dtype() == DType::Float32) {
    const float* src = input->ptr<float>();
    float* dst = output->ptr<float>();
    float* weight_ptr = weight->ptr<float>();
    kernel->operator()<KernelID::RmsNormFloat>(src, dst, seq_len, embd);
    kernel->operator()<KernelID::ElemwiseBroadcastDim0Src1Float>(dst, weight_ptr, dst, seq_len,
                                                                 embd, ElemMode::Mul);
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
      kernel->operator()<KernelID::EmbeddingGetInt4Float>(weight->ptr(), input->ptr<uint32_t>(),
                                                          output->ptr<float>(), len, m_embd);
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
    InData<float> in_datas;
    for (auto input : inputs()) {
      in_datas.push_back(input->ptr<float>());
    }
    float* dst = output->ptr<float>();
    size_t len = output->length();
    kernel->operator()<KernelID::ElemwiseFloat>(in_datas, dst, len, m_mode);
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
    const float* src = inputs()[0]->ptr<float>();
    kernel->operator()<KernelID::MatmulInt4Float>(dst, weight, src, M, N, K, p_workspace,
                                                  p_workspace_size);
  }
}

size_t MatMul::get_workspace_in_byte() {
  uint32_t M = inputs()[0]->shape()[0];
  uint32_t K = inputs()[0]->shape()[1];
  uint32_t N = weights()[0]->shape()[0];
  auto src_dtype = inputs()[0]->dtype();
  auto kernel = get_kernel();
  if (src_dtype == DType::Float32) {
    return kernel->get_workspace<KernelID::MatmulInt4Float>(kernel->nr_thread(), M, N, K);
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
    const float* src = inputs()[0]->ptr<float>() + (row - 1) * K;
    kernel->operator()<KernelID::MatmulInt4Float>(dst, weight, src, M, N, K, p_workspace,
                                                  p_workspace_size);
  }
}

size_t MatMulLast::get_workspace_in_byte() {
  uint32_t M = 1;
  uint32_t K = inputs()[0]->shape()[1];
  uint32_t N = weights()[0]->shape()[0];
  auto src_dtype = inputs()[0]->dtype();
  auto kernel = get_kernel();
  if (src_dtype == DType::Float32) {
    return kernel->get_workspace<KernelID::MatmulInt4Float>(kernel->nr_thread(), M, N, K);
  }
  return 0;
}

void MatMulNoWeight::execute(WorkSpace* workspace, uint32_t) {}

size_t MatMulNoWeight::get_workspace_in_byte() { return 0; }

void MatMulCacheKv::execute(WorkSpace* workspace, uint32_t nr_past) {
  auto weight_q = weights()[0];
  auto weight_k = weights()[1];
  auto weight_v = weights()[2];
  INFER_ASSERT(nr_past == m_kstorage->current_index(),
               "The index in kv storage is not the same as input\n");

  void* p_wq = weight_q->ptr();
  void* p_wk = weight_k->ptr();
  void* p_wv = weight_v->ptr();

  auto w_dtype = weight_q->dtype();
  auto out_q = outputs()[1];
  auto input = inputs()[0];
  auto in_dtype = input->dtype();

  uint32_t M = input->shape()[0];
  uint32_t K = input->shape()[1];
  uint32_t N = weight_q->shape()[0];

  void* p_work = workspace->ptr();
  uint32_t size = workspace->length();
  auto kernel = get_kernel();

  //! compute k, q, v
  if (w_dtype == DType::Int4 && in_dtype == DType::Float32) {
    const float* pdata = input->ptr<float>();
    float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
    float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
    float* p_outq = static_cast<float*>(out_q->ptr());
    kernel->operator()<KernelID::MatmulInt4Float>(p_outq, p_wq, pdata, M, N, K, p_work, size);
    kernel->operator()<KernelID::MatmulInt4Float>(p_outk, p_wk, pdata, M, N, K, p_work, size);
    kernel->operator()<KernelID::MatmulInt4Float>(p_outv, p_wv, pdata, M, N, K, p_work, size);
  }
}

size_t MatMulCacheKv::get_workspace_in_byte() {
  uint32_t M = inputs()[0]->shape()[0];
  uint32_t K = inputs()[0]->shape()[1];
  uint32_t N = weights()[0]->shape()[0];
  auto src_dtype = inputs()[1]->dtype();
  auto kernel = get_kernel();
  if (src_dtype == DType::Float32) {
    return kernel->get_workspace<KernelID::MatmulInt4Float>(kernel->nr_thread(), M, N, K);
  }
  return 0;
}

void Attention::execute(WorkSpace* workspace, uint32_t nr_past) {
  auto weight_q = weights()[0];
  auto weight_k = weights()[1];
  auto weight_v = weights()[2];
  INFER_ASSERT(nr_past == m_kstorage->current_index(),
               "The index in kv storage is not the same as input\n");

  auto kernel = get_kernel();
  void* p_wq = weight_q->ptr();
  void* p_wk = weight_k->ptr();
  void* p_wv = weight_v->ptr();

  auto w_dtype = weight_q->dtype();
  auto out = outputs()[0];
  auto input = inputs()[0];
  auto in_dtype = input->dtype();

  uint32_t seqlen = input->shape()[0];
  uint32_t embd = input->shape()[1];
  uint32_t head = m_head;

  void* p_work = workspace->ptr();
  uint32_t matmul_size = kernel->get_workspace<KernelID::MatmulInt4Float>(
      kernel->nr_thread(), seqlen, embd, static_cast<uint32_t>(weight_q->shape()[0]));
  uint32_t size = workspace->length();

  void* q_out = static_cast<void*>(static_cast<char*>(p_work) + matmul_size);
  void* qk_out = static_cast<void*>(static_cast<char*>(q_out) + seqlen * m_embd * sizeof(float));

  if (in_dtype == DType::Float32) {
    //! compute k, q, v
    const float* pdata = input->ptr<float>();
    float* p_outk = static_cast<float*>(m_kstorage->get_current_data());
    float* p_outv = static_cast<float*>(m_vstorage->get_current_data());
    float* p_outq = static_cast<float*>(q_out);
    if (w_dtype == DType::Int4) {
      kernel->operator()<KernelID::MatmulInt4Float>(p_outq, p_wq, pdata, seqlen, embd, embd, p_work,
                                                    size);
      kernel->operator()<KernelID::MatmulInt4Float>(p_outk, p_wk, pdata, seqlen, embd, embd, p_work,
                                                    size);
      kernel->operator()<KernelID::MatmulInt4Float>(p_outv, p_wv, pdata, seqlen, embd, embd, p_work,
                                                    size);
    }
    //! rope Q
    kernel->operator()<KernelID::RopeFloat>(p_outq, p_outq, nr_past, m_rot, RotMode::Mode0, seqlen,
                                            head, embd / head);
    //! rope K
    float* p_totalk = static_cast<float*>(m_kstorage->ptr());
    kernel->operator()<KernelID::RopeFloat>(p_totalk, p_totalk, nr_past, m_rot, RotMode::Mode1,
                                            seqlen + nr_past, head, embd / head);
    //! Q*k with transpose
    kernel->operator()<KernelID::MatmulWithHeadStrideFloat>((float*)qk_out, p_totalk, p_outq,
                                                            seqlen, embd, head, nr_past);
    //! scale and diag
    float scale = 1.0f / sqrt(float(embd) / head);
    kernel->operator()<KernelID::ScaleDiagMaskFloat>((float*)qk_out, (float*)qk_out, scale, nr_past,
                                                     seqlen, head);
    //! softmax
    kernel->operator()<KernelID::SoftmaxFloat>((float*)qk_out, (float*)qk_out, head * seqlen,
                                               nr_past + seqlen);
    //! compute v_out
    float* out = outputs()[0]->ptr<float>();
    float* p_totalv = static_cast<float*>(m_vstorage->ptr());
    kernel->operator()<KernelID::HeadBatchedMatmulFloat>(out, p_totalv, (float*)qk_out, seqlen,
                                                         embd, head, nr_past);
  }
}

size_t Attention::get_workspace_in_byte() {
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
    total += kernel->get_workspace<KernelID::MatmulInt4Float>(kernel->nr_thread(), M, N, K);
    //! out q
    total += seqlen * m_embd * sizeof(float);
    //! kq out
    total += m_head * seqlen * m_ctx * sizeof(float);
  }
  return total;
}

void Rope::execute(WorkSpace* workspace, uint32_t nr_past) {
  auto output = outputs()[0];
  uint32_t N = output->shape()[0];
  uint32_t head = output->shape()[1];
  uint32_t embd = output->shape()[2];
  auto kernel = get_kernel();
  if (output->dtype() == DType::Float32) {
    const float* in_data = inputs()[0]->ptr<float>();
    float* dst = output->ptr<float>();
    kernel->operator()<KernelID::RopeFloat>(dst, in_data, nr_past, m_rot, m_mode, N, head, embd);
  } else {
    //! fp16
  }
}

void DiagMask::execute(WorkSpace*, uint32_t n_past) {
  auto output = outputs()[0];
  uint32_t head = output->shape()[0];
  uint32_t N = output->shape()[1];
  auto kernel = get_kernel();
  if (output->dtype() == DType::Float32) {
    const float* in_data = inputs()[0]->ptr<float>();
    float* dst = output->ptr<float>();
    kernel->operator()<KernelID::DiagMaskFloat>(dst, in_data, n_past, N, head);
  } else {
    //! fp16
  }
}

void Permute::execute(WorkSpace* workspace, uint32_t nr_past) {
  auto output = outputs()[0];
  auto input = inputs()[0];
  auto kernel = get_kernel();
  uint32_t dim0 = input->shape()[0];
  uint32_t dim1 = input->shape()[1];
  uint32_t dim2 = input->shape()[2];
  if (output->dtype() == DType::Float32) {
    const float* in_data = inputs()[0]->ptr<float>();
    float* dst = output->ptr<float>();
    kernel->operator()<KernelID::PermuteFloat>(dst, in_data, dim0, dim1, dim2, m_param);
  } else {
    //! fp16
  }
}
