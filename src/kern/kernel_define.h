#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <functional>
#include <vector>

#define PI                           (3.1415)
#define PGELU                        (0.044715)
#define INFER_ATTRIBUTE_TARGET(simd) __attribute__((target(simd)))
namespace inferllm {

template <class Dtype>
using InData = std::vector<const Dtype*>;

enum class KernelID {
    EmbeddingGetInt4Float = 0,
    EmbeddingGetInt8Float,
    EmbeddingGetFloatFloat,
    ElemwiseFloat,
    ElemwiseFloatScale,
    ElemwiseBroadcastDim0Src1Float,
    NormFloat,
    RmsNormFloat,
    SoftmaxFloat,
    MatmulInt4Float,
    MatmulInt4FloatPacked, // 10
    MatmulInt8Float,
    MatmulFloatFloat,
    MatmulWithHeadStrideFloat,
    //! multi query attention, q*kT
    MatmulWithHeadStrideQBroadCastKFloat,
    HeadBatchedMatmulFloat,
    //! multi query attention
    HeadBatchedMatmulBroadCastVFloat,
    RopeFloat,
    GlmRopeFloat,
    ScaleDiagMaskFloat,
    DiagMaskFloat,
    GlmGmask,
    PermuteFloat,
    MatmulInt4WeightReorder,
};

enum class KernelOptMethod {
    MatmulInt4Reorder = 0,
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
    ModelRotHalf,
};

enum class KernelType { Naive = 0, Arm = 1, X86 = 2, GPU = 3 };

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
static_assert(sizeof(BlockQ40) == 20, "BlockQ40 size error");

struct BlockQ40X8 {
    uint8_t qs[QK40 / 2 * 8];     // nibbles / quants
    float scale[8];               // delta
};
static_assert(sizeof(BlockQ40X8) == 160, "BlockQ40X8 size error");

#define QK80 32
struct BlockQ80 {
    float d;          // delta
    int8_t qs[QK80];  // nibbles
};
static_assert(sizeof(BlockQ80) == 36, "BlockQ80 size error");
}  // namespace inferllm

#define PartialImplementKernel(kernel_id, fun)       \
    template <typename... Args>                      \
    struct Comp<KernelID::kernel_id, Args...> {      \
        static TaskSet get_all_task(Args... args) {  \
            return fun(std::forward<Args>(args)...); \
        }                                            \
    };

#define PartialImplementSpace(kernel_id, fun)        \
    template <typename... Args>                      \
    struct Space<KernelID::kernel_id, Args...> {     \
        static size_t get(Args... args) {            \
            return fun(std::forward<Args>(args)...); \
        }                                            \
    };
