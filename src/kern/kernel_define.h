#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#define PI (3.1415)
#define PGELU (0.044715)
#define INFER_ATTRIBUTE_TARGET(simd)   __attribute__((target(simd)))
namespace inferllm {

template <class Dtype>
using InData = std::vector<const Dtype*>;

enum class KernelID {
    EmbeddingGetInt4Float = 0,
    EmbeddingGetFloatFloat = 1,
    ElemwiseFloat,
    ElemwiseFloatScale,
    ElemwiseBroadcastDim0Src1Float,
    NormFloat,
    RmsNormFloat,
    SoftmaxFloat,
    MatmulInt4Float,
    MatmulFloatFloat,
    MatmulWithHeadStrideFloat,
    HeadBatchedMatmulFloat,
    RopeFloat,
    GlmRopeFloat,
    ScaleDiagMaskFloat,
    DiagMaskFloat,
    GlmGmask,
    PermuteFloat,
};

enum class ElemMode {
    Add = 0,
    Mul,
    Silu,
    Gelu,
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
    float d;           // delta
    int8_t qs[QK80];  // nibbles
};
}  // namespace inferllm