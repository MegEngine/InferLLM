#pragma once
#if INFER_ARM
#include <assert.h>
#include "arm_neon.h"
#include "kern/kernel_define.h"

namespace inferllm {
namespace opt {

inline void elemwise_vector_add(const int n, const float* __restrict x,
                                const float* __restrict y,
                                float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}


inline void elemwise_vector_mul(const int n, const float* __restrict x,
                                const float* __restrict y,
                                float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}


inline void elemwise_vector_silu(const int n, const float* __restrict x,
                                 float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] / (1 + exp(-x[i]));
    }
}

inline void elemwise_vector_gelu(const int n, const float* __restrict x,
                                 float* __restrict z) {
    for (int i = 0; i < n; i++) {
        float src = x[i];
        z[i] = 0.5 * src *
               (1 + tanh(sqrt(2.0 / PI) * (src + PGELU * src * src * src)));
    }
}

inline void elemwise_vec_scale(const int n, const float* __restrict x,
                               float scale, float* __restrict z) {
    int i = 0;
    for (; i < n; i++) {
        z[i] = x[i] * scale;
    }
}


inline float reduce_square_sum(const int n, const float* __restrict x) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}


inline float reduce_max(const int n, const float* __restrict x) {
    float max = -INFINITY;
    for (int i = 0; i < n; i++) {
        max = std::max(max, x[i]);
    }
    return max;
}



inline float select_sub_max_and_reduce_sum(const int n,
                                           const float* __restrict x,
                                           float* __restrict y,
                                           const float max) {
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

inline void compute_src_offset_embd_matmul(const float* __restrict srcq_head,
                                           int offsetq,
                                           const float* __restrict srck_head,
                                           int offsetk, float* dst_head,
                                           int seqlen, int length,
                                           int sub_embd) {
    for (uint32_t row = 0; row < seqlen; row++) {
        auto p_srcq = srcq_head + row * offsetq;
        uint32_t len = 0;
        for (; len + 3 < length; len += 4) {
            auto p_dst = dst_head + row * length + len;
            auto p_srck0 = srck_head + len * offsetk;
            auto p_srck1 = srck_head + (len + 1) * offsetk;
            auto p_srck2 = srck_head + (len + 2) * offsetk;
            auto p_srck3 = srck_head + (len + 3) * offsetk;
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (uint32_t k = 0; k < sub_embd; k++) {
                sum0 += p_srck0[k] * p_srcq[k];
                sum1 += p_srck1[k] * p_srcq[k];
                sum2 += p_srck2[k] * p_srcq[k];
                sum3 += p_srck3[k] * p_srcq[k];
            }
            p_dst[0] = sum0;
            p_dst[1] = sum1;
            p_dst[2] = sum2;
            p_dst[3] = sum3;
        }
        for (; len < length; len++) {
            auto p_dst = dst_head + row * length + len;
            auto p_srck = srck_head + len * offsetk;
            float sum = 0;
            for (uint32_t k = 0; k < sub_embd; k++) {
                sum += p_srck[k] * p_srcq[k];
            }
            *p_dst = sum;
        }
    }
}

inline void comput_matmul_with_dst_uncontinue(float* __restrict dst,
                                              int offset_dst,
                                              const float* __restrict srcv,
                                              int offset_v,
                                              const float* __restrict srcqk,
                                              int seqlen, int length, int K) {
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
}

inline float vec_vec_dot_q40_with_q80(const int n, const void* __restrict vx,
                                      const void* __restrict vy) {
    const int nb = n / QK80;

    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ40* __restrict x = (BlockQ40*)vx;
    const BlockQ80* __restrict y = (BlockQ80*)vy;

    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const BlockQ40* __restrict x0 = &x[i + 0];
        const BlockQ40* __restrict x1 = &x[i + 1];
        const BlockQ80* __restrict y0 = &y[i + 0];
        const BlockQ80* __restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // interleave
        const int8x16_t v1_0ls = vuzp1q_s8(v1_0l, v1_0h);
        const int8x16_t v1_0hs = vuzp2q_s8(v1_0l, v1_0h);
        const int8x16_t v1_1ls = vuzp1q_s8(v1_1l, v1_1h);
        const int8x16_t v1_1hs = vuzp2q_s8(v1_1l, v1_1h);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls), v0_0hs, v1_0hs);
        const int32x4_t p_1 = vdotq_s32(
                vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1ls), v0_1hs, v1_1hs);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), x0->d * y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), x1->d * y1->d);
#else
        const int16x8_t pl0l =
                vmull_s8(vget_low_s8(v0_0ls), vget_low_s8(v1_0ls));
        const int16x8_t pl0h =
                vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));
        const int16x8_t ph0l =
                vmull_s8(vget_low_s8(v0_0hs), vget_low_s8(v1_0hs));
        const int16x8_t ph0h =
                vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));

        const int16x8_t pl1l =
                vmull_s8(vget_low_s8(v0_1ls), vget_low_s8(v1_1ls));
        const int16x8_t pl1h =
                vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1ls));
        const int16x8_t ph1l =
                vmull_s8(vget_low_s8(v0_1hs), vget_low_s8(v1_1hs));
        const int16x8_t ph1h =
                vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1hs));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)),
                            x0->d * y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)),
                            x1->d * y1->d);
#endif
    }
    return vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
}

}  // namespace opt
}  // namespace inferllm

#endif
