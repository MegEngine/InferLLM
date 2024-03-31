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

#ifndef SRC_KERN_KERNEL_DEFINE_H_
#define SRC_KERN_KERNEL_DEFINE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <vector>

#define INFER_ATTRIBUTE_TARGET(simd) __attribute__((target(simd)))
namespace llm_learning {

template <class Dtype>
using InData = std::vector<const Dtype*>;

enum class KernelID {
  EmbeddingGetInt4Float = 0,
  ElemwiseFloat,
  ElemwiseBroadcastDim0Src1Float,
  NormFloat,
  RmsNormFloat,
  SoftmaxFloat,
  MatmulInt4Float,
  MatmulWithHeadStrideFloat,
  HeadBatchedMatmulFloat,
  RopeFloat,
  ScaleDiagMaskFloat,
  DiagMaskFloat,
  PermuteFloat,
};

enum class ElemMode {
  Add = 0,
  Mul,
  Silu,
};

enum class RotMode {
  Mode0 = 0,
  Mode1,
};

enum class KernelType {
  Naive = 0,
  Arm = 1,
  X86 = 2,
};

struct TaskId {
  uint32_t start;
  uint32_t end;
  uint32_t thread_id;
};

//! the task, the first parameter is the task start id, the second parameter is
//! the task end if, the third parameter is the thread id
using MultiThreadingTask = std::function<void(TaskId)>;

//! the task pair, the first parameter is the task, the second parameter is the
//! number of sub task, some kernel may need to split the task into several
using TaskSet = std::vector<std::pair<MultiThreadingTask, uint32_t>>;

#define QK40 32
struct BlockQ40 {
  float d;               // delta
  uint8_t qs[QK40 / 2];  // nibbles / quants
};

#define QK80 32
struct BlockQ80 {
  float d;          // delta
  int8_t qs[QK80];  // nibbles
};
}  // namespace llm_learning

#endif  // SRC_KERN_KERNEL_DEFINE_H_