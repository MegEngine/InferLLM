#pragma once

#include <assert.h>
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include "kern/kernel_define.h"
#include "kern/naive/quantize.h"

#include "common.h"

namespace inferllm {
namespace opt {

inline void elemwise_vector_add(
        const int n, const float* __restrict x, const float* __restrict y,
        float* __restrict z) {
    vadd(n, x, y, z);
}

inline void elemwise_vector_mul(
        const int n, const float* __restrict x, const float* __restrict y,
        float* __restrict z) {
    vmul(n, x, y, z);
}

inline void elemwise_vector_silu(
        const int n, const float* __restrict x, float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] / (1 + exp(-x[i]));
    }
}

inline void elemwise_vector_gelu(
        const int n, const float* __restrict x, float* __restrict z) {
    for (int i = 0; i < n; i++) {
        float src = x[i];
        z[i] = 0.5 * src * (1 + tanh(sqrt(2.0 / PI) * (src + PGELU * src * src * src)));
    }
}

inline void elemwise_vec_scale(
        const int n, const float* __restrict x, float scale, float* __restrict z) {
    vscal(n, x, z, scale);
}

inline float reduce_square_sum(const int n, const float* __restrict x) {
    return vsqrsum(x, n, 0.0);
}

inline float reduce_max(const int n, const float* __restrict x) {
    return vmax(n, x, -INFINITY);
}

inline float select_sub_max_and_reduce_sum(
        const int n, const float* __restrict x, float* __restrict y, const float max) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        if (x[i] == -INFINITY) {
            y[i] = 0.0f;
        } else {
            float val = exp(x[i] - max);
            sum += val;
            y[i] = val;
        }
    }
    return sum;
}

inline void comput_matmul_with_dst_uncontinue(
        float* __restrict dst, int offset_dst, const float* __restrict srcv,
        int offset_v, const float* __restrict srcqk, int seqlen, int length, int K) {
    for (uint32_t row = 0; row < seqlen; row++) {
        auto p_qk = srcqk + row * length;
        for (uint32_t len = 0; len < K; len++) {
            auto p_dst = dst + row * offset_dst + len;
            auto p_v = srcv + len;
            float sum = 0;
            for (uint32_t k = 0; k < length; k++) {
                sum += p_v[k * offset_v] * p_qk[k];
            }
            *p_dst = sum;
        }
    }
    return;
}

}  // namespace opt
}  // namespace inferllm
