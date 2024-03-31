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
#ifndef SRC_KERN_NAIVE_H_
#define SRC_KERN_NAIVE_H_

#include "kern/naive/quantize.h"
#include "math.h"
#include "string.h"

namespace llm_learning {
namespace naive {

TaskSet llm_embedding_get_int4_float(const void* weights, const uint32_t* index, float* dst,
                                     uint32_t len_seq, uint32_t embd);

TaskSet llm_elemwise_compute_float(InData<float> srcs, float* dst, size_t len, ElemMode mode);

TaskSet llm_elemwise_broadcast_dim0_src1_compute_float(const float* src0, const float* src1,
                                                       float* dst, uint32_t len0, uint32_t len1,
                                                       ElemMode mode);

TaskSet llm_norm_compute_float(const float* src, float* dst, uint32_t seq_len, uint32_t embd);

TaskSet llm_rms_norm_compute_float(const float* src, float* dst, uint32_t seq_len, uint32_t embd);

TaskSet llm_softmax_compute_float(const float* src, float* dst, uint32_t len_row, uint32_t col);

// compute the softmax of the last dim of src, and store the result in dst
TaskSet llm_matmul_compute_int4_float(float* dst, const void* src0, const float* src1, uint32_t M,
                                      uint32_t N, uint32_t K, void* workspace, uint32_t size);

size_t llm_matmul_get_workspace_float(uint32_t nr_thread, uint32_t M, uint32_t N, uint32_t K);

TaskSet llm_rope_compute_float(float* dst, const float* src0, uint32_t n_past, uint32_t n_rot,
                               RotMode m, uint32_t N, uint32_t head, uint32_t embd);

TaskSet llm_diag_mask_inf_float(float* dst, const float* src0, uint32_t n_past, uint32_t N,
                                uint32_t head);

TaskSet llm_scale_diag_mask_inf_float(float* dst, const float* src0, float scale, uint32_t n_past,
                                      uint32_t seqlen, uint32_t head);

TaskSet llm_permute_compute_float(float* dst, const float* src0, uint32_t dim0, uint32_t dim1,
                                  uint32_t dim2, std::vector<uint32_t> param);

TaskSet llm_matmul_compute_with_head_stride_float(float* dst, const float* srck, const float* srcq,
                                                  uint32_t seqlen, uint32_t embd, uint32_t head,
                                                  uint32_t nr_past);

TaskSet llm_head_batched_matmul_compute_float(float* dst, const float* v, const float* qk,
                                              uint32_t seqlen, uint32_t embd, uint32_t head,
                                              uint32_t nr_past);
template <KernelID Id, typename... Args>
struct Comp {
  static TaskSet get_all_task(Args... args);
};

template <KernelID Id, typename... Args>
struct Space {
  static size_t get(Args... args);
};

}  // namespace naive
}  // namespace llm_learning

#ifdef PartialImplementKernel
#undef PartialImplementKernel
#endif
#ifdef PartialImplementSpace
#undef PartialImplementSpace
#endif

#define PartialImplementKernel(kernel_id, fun)                                             \
  template <typename... Args>                                                              \
  struct Comp<KernelID::kernel_id, Args...> {                                              \
    static TaskSet get_all_task(Args... args) { return fun(std::forward<Args>(args)...); } \
  };

#define PartialImplementSpace(kernel_id, fun)                                    \
  template <typename... Args>                                                    \
  struct Space<KernelID::kernel_id, Args...> {                                   \
    static size_t get(Args... args) { return fun(std::forward<Args>(args)...); } \
  };

namespace llm_learning {
namespace naive {
PartialImplementKernel(ElemwiseFloat, llm_elemwise_compute_float);
PartialImplementKernel(ElemwiseBroadcastDim0Src1Float,
                       llm_elemwise_broadcast_dim0_src1_compute_float);
PartialImplementKernel(NormFloat, llm_norm_compute_float);
PartialImplementKernel(RmsNormFloat, llm_rms_norm_compute_float);
PartialImplementKernel(EmbeddingGetInt4Float, llm_embedding_get_int4_float);
PartialImplementKernel(SoftmaxFloat, llm_softmax_compute_float);
PartialImplementKernel(MatmulInt4Float, llm_matmul_compute_int4_float);
PartialImplementKernel(MatmulWithHeadStrideFloat, llm_matmul_compute_with_head_stride_float);
PartialImplementKernel(HeadBatchedMatmulFloat, llm_head_batched_matmul_compute_float);
PartialImplementKernel(DiagMaskFloat, llm_diag_mask_inf_float);
PartialImplementKernel(RopeFloat, llm_rope_compute_float);
PartialImplementKernel(ScaleDiagMaskFloat, llm_scale_diag_mask_inf_float);
PartialImplementKernel(PermuteFloat, llm_permute_compute_float);

PartialImplementSpace(MatmulInt4Float, llm_matmul_get_workspace_float);

#undef PartialImplementKernel
#undef PartialImplementSpace

}  // namespace naive
}  // namespace llm_learning

#endif //SRC_KERN_NAIVE_H_