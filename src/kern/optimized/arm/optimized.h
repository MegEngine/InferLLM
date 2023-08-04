#pragma once

#include <assert.h>
#include "arm_neon.h"
#include "kern/kernel_define.h"

namespace inferllm {
namespace opt {

inline void elemwise_vector_add(
        const int n, const float* __restrict x, const float* __restrict y,
        float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

inline void elemwise_vector_mul(
        const int n, const float* __restrict x, const float* __restrict y,
        float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
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

inline void compute_src_offset_embd_matmul(
        const float* __restrict srcq_head, int offsetq,
        const float* __restrict srck_head, int offsetk, float* dst_head, int seqlen,
        int length, int sub_embd) {
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
}

inline float vec_vec_dot_q40_with_q80(
        const int n, const void* __restrict vx, const void* __restrict vy) {
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
        const int32x4_t p_0 =
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls), v0_0hs, v1_0hs);
        const int32x4_t p_1 =
                vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1ls), v0_1hs, v1_1hs);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), x0->d * y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), x1->d * y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8(v0_0ls), vget_low_s8(v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));
        const int16x8_t ph0l = vmull_s8(vget_low_s8(v0_0hs), vget_low_s8(v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));

        const int16x8_t pl1l = vmull_s8(vget_low_s8(v0_1ls), vget_low_s8(v1_1ls));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1ls));
        const int16x8_t ph1l = vmull_s8(vget_low_s8(v0_1hs), vget_low_s8(v1_1hs));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1hs));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0->d * y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), x1->d * y1->d);
#endif
    }
    return vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
}

inline void vec_vec_dot_q40_with_q80_packed(
        const int n, const void* __restrict vx, const void* __restrict vy, float* dst,
        const float* bias) {
    const int nb = n / QK80;

    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ40X8* __restrict x = (BlockQ40X8*)(vx);
    const BlockQ80* __restrict y = (BlockQ80*)vy;

    float32x4_t zero = vdupq_n_f32(0.0f);
    uint8x16_t m4b = vdupq_n_u8(0x0F);
    int8x16_t s8b = vdupq_n_s8(0x8);
    float bias_v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float* bias_ptr = bias ? bias : bias_v;

    float32x4_t sumv0 = vdupq_n_f32(bias_ptr[0]);
    float32x4_t sumv1 = vdupq_n_f32(bias_ptr[1]);
    float32x4_t sumv2 = vdupq_n_f32(bias_ptr[2]);
    float32x4_t sumv3 = vdupq_n_f32(bias_ptr[3]);
    float32x4_t sumv4 = vdupq_n_f32(bias_ptr[4]);
    float32x4_t sumv5 = vdupq_n_f32(bias_ptr[5]);
    float32x4_t sumv6 = vdupq_n_f32(bias_ptr[6]);
    float32x4_t sumv7 = vdupq_n_f32(bias_ptr[7]);

    for (int i = 0; i < nb; i++) {
        const BlockQ40X8* __restrict xptr = x + i;

        const uint8_t* __restrict x0 = xptr->qs;
        const uint8_t* __restrict x1 = xptr->qs + 1 * QK40 / 2;
        const uint8_t* __restrict x2 = xptr->qs + 2 * QK40 / 2;
        const uint8_t* __restrict x3 = xptr->qs + 3 * QK40 / 2;
        const uint8_t* __restrict x4 = xptr->qs + 4 * QK40 / 2;
        const uint8_t* __restrict x5 = xptr->qs + 5 * QK40 / 2;
        const uint8_t* __restrict x6 = xptr->qs + 6 * QK40 / 2;
        const uint8_t* __restrict x7 = xptr->qs + 7 * QK40 / 2;

        const float* scale = xptr->scale;

        const BlockQ80* __restrict y0 = &y[i];

        // load y
        int8x16_t vy0l = vld1q_s8(y0->qs);
        int8x16_t vy0h = vld1q_s8(y0->qs + 16);

        // interleave
        int8x16_t vyls = vuzp1q_s8(vy0l, vy0h);
        int8x16_t vyhs = vuzp2q_s8(vy0l, vy0h);

        uint8x16_t v0 = vld1q_u8(x0);
        uint8x16_t v1 = vld1q_u8(x1);
        uint8x16_t v2 = vld1q_u8(x2);
        uint8x16_t v3 = vld1q_u8(x3);
        uint8x16_t v4 = vld1q_u8(x4);
        uint8x16_t v5 = vld1q_u8(x5);
        uint8x16_t v6 = vld1q_u8(x6);
        uint8x16_t v7 = vld1q_u8(x7);

        // 4-bit -> 8-bit
        int8x16_t v0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
        int8x16_t v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));

        int8x16_t v1l = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
        int8x16_t v1h = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));

        int8x16_t v2l = vreinterpretq_s8_u8(vandq_u8(v2, m4b));
        int8x16_t v2h = vreinterpretq_s8_u8(vshrq_n_u8(v2, 4));

        int8x16_t v3l = vreinterpretq_s8_u8(vandq_u8(v3, m4b));
        int8x16_t v3h = vreinterpretq_s8_u8(vshrq_n_u8(v3, 4));

        int8x16_t v4l = vreinterpretq_s8_u8(vandq_u8(v4, m4b));
        int8x16_t v4h = vreinterpretq_s8_u8(vshrq_n_u8(v4, 4));

        int8x16_t v5l = vreinterpretq_s8_u8(vandq_u8(v5, m4b));
        int8x16_t v5h = vreinterpretq_s8_u8(vshrq_n_u8(v5, 4));
        
        int8x16_t v6l = vreinterpretq_s8_u8(vandq_u8(v6, m4b));
        int8x16_t v6h = vreinterpretq_s8_u8(vshrq_n_u8(v6, 4));
        
        int8x16_t v7l = vreinterpretq_s8_u8(vandq_u8(v7, m4b));
        int8x16_t v7h = vreinterpretq_s8_u8(vshrq_n_u8(v7, 4));

        // sub 8
        int8x16_t v0ls = vsubq_s8(v0l, s8b);
        int8x16_t v0hs = vsubq_s8(v0h, s8b);
        int8x16_t v1ls = vsubq_s8(v1l, s8b);
        int8x16_t v1hs = vsubq_s8(v1h, s8b);
        int8x16_t v2ls = vsubq_s8(v2l, s8b);
        int8x16_t v2hs = vsubq_s8(v2h, s8b);
        int8x16_t v3ls = vsubq_s8(v3l, s8b);
        int8x16_t v3hs = vsubq_s8(v3h, s8b);
        int8x16_t v4ls = vsubq_s8(v4l, s8b);
        int8x16_t v4hs = vsubq_s8(v4h, s8b);
        int8x16_t v5ls = vsubq_s8(v5l, s8b);
        int8x16_t v5hs = vsubq_s8(v5h, s8b);
        int8x16_t v6ls = vsubq_s8(v6l, s8b);
        int8x16_t v6hs = vsubq_s8(v6h, s8b);
        int8x16_t v7ls = vsubq_s8(v7l, s8b);
        int8x16_t v7hs = vsubq_s8(v7h, s8b);

        // dot product into int32x4_t
        int32x4_t p0 = vdotq_s32(vdotq_s32(zero, v0ls, vyls), v0hs, vyhs);
        int32x4_t p1 = vdotq_s32(vdotq_s32(zero, v1ls, vyls), v1hs, vyhs);
        int32x4_t p2 = vdotq_s32(vdotq_s32(zero, v2ls, vyls), v2hs, vyhs);
        int32x4_t p3 = vdotq_s32(vdotq_s32(zero, v3ls, vyls), v3hs, vyhs);
        int32x4_t p4 = vdotq_s32(vdotq_s32(zero, v4ls, vyls), v4hs, vyhs);
        int32x4_t p5 = vdotq_s32(vdotq_s32(zero, v5ls, vyls), v5hs, vyhs);
        int32x4_t p6 = vdotq_s32(vdotq_s32(zero, v6ls, vyls), v6hs, vyhs);
        int32x4_t p7 = vdotq_s32(vdotq_s32(zero, v7ls, vyls), v7hs, vyhs);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), scale[0] * y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), scale[1] * y0->d);
        sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), scale[2] * y0->d);
        sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), scale[3] * y0->d);
        sumv4 = vmlaq_n_f32(sumv4, vcvtq_f32_s32(p4), scale[4] * y0->d);
        sumv5 = vmlaq_n_f32(sumv5, vcvtq_f32_s32(p5), scale[5] * y0->d);
        sumv6 = vmlaq_n_f32(sumv6, vcvtq_f32_s32(p6), scale[6] * y0->d);
        sumv7 = vmlaq_n_f32(sumv7, vcvtq_f32_s32(p7), scale[7] * y0->d);
    }
    float32x4_t sumv_0 = vpaddq_f32(vpaddq_f32(sumv0, sumv1), vpaddq_f32(sumv2, sumv3));
    float32x4_t sumv_1 = vpaddq_f32(vpaddq_f32(sumv4, sumv5), vpaddq_f32(sumv6, sumv7));
    vst1q_f32(dst, sumv_0);
    vst1q_f32(dst + 4, sumv_1);
}

inline void vec_vec_dot_q40_with_q80_packed_asm(
        const int n, const void* __restrict vx, const void* __restrict vy, float* dst,
        const float* bias) {
    int nb = n / QK80;

    assert(n % QK80 == 0);
    assert(nb % 2 == 0 && nb > 0);

    const void* __restrict x = vx;
    const void* __restrict y = vy;

    float bias_v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float* bias_ptr = bias ? bias : bias_v;

    asm volatile(
            //! set all sum to 0
            "ld1r {v0.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v1.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v2.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v3.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v4.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v5.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v6.4s}, [%[bias_ptr]], #4\n"
            "ld1r {v7.4s}, [%[bias_ptr]], #4\n"

            //! main loop
            "1:\n"
            //! load y
            "ld1r {v10.4s}, [%[y]], #4\n"
            //! load constant 0x0f and 0x8
            "movi v8.16b, #0x0f\n"
            "movi v9.16b, #0x08\n"

            "ld1 {v28.16b}, [%[y]], #16\n"
            "ld1 {v29.16b}, [%[y]], #16\n"
            "uzp1 v30.16b, v28.16b, v29.16b\n"
            "uzp2 v31.16b, v28.16b, v29.16b\n"

            "ld1 {v11.16b}, [%[x]], #16\n"
            "ld1 {v13.16b}, [%[x]], #16\n"
            "ld1 {v15.16b}, [%[x]], #16\n"
            "ld1 {v17.16b}, [%[x]], #16\n"
            "ld1 {v19.16b}, [%[x]], #16\n"
            "ld1 {v21.16b}, [%[x]], #16\n"
            "ld1 {v23.16b}, [%[x]], #16\n"
            "ld1 {v25.16b}, [%[x]], #16\n"

            //! load the scale
            "ld1 {v26.4s}, [%[x]], #16\n"
            "ld1 {v27.4s}, [%[x]], #16\n"

            //! multiply scale
            "fmul v26.4s, v26.4s, v10.4s\n"
            "fmul v27.4s, v27.4s, v10.4s\n"

            //! 4-bit -> 8-bit
            "and v10.16b, v11.16b, v8.16b\n"
            "ushr v11.16b, v11.16b, #4\n"

            "and v12.16b, v13.16b, v8.16b\n"
            "ushr v13.16b, v13.16b, #4\n"

            "and v14.16b, v15.16b, v8.16b\n"
            "ushr v15.16b, v15.16b, #4\n"

            "and v16.16b, v17.16b, v8.16b\n"
            "ushr v17.16b, v17.16b, #4\n"

            "and v18.16b, v19.16b, v8.16b\n"
            "ushr v19.16b, v19.16b, #4\n"

            "and v20.16b, v21.16b, v8.16b\n"
            "ushr v21.16b, v21.16b, #4\n"

            "and v22.16b, v23.16b, v8.16b\n"
            "ushr v23.16b, v23.16b, #4\n"

            "and v24.16b, v25.16b, v8.16b\n"
            "ushr v25.16b, v25.16b, #4\n"

            //! sub 8
            "sub v10.16b, v10.16b, v9.16b\n"
            "sub v11.16b, v11.16b, v9.16b\n"

            "sub v12.16b, v12.16b, v9.16b\n"
            "sub v13.16b, v13.16b, v9.16b\n"

            "sub v14.16b, v14.16b, v9.16b\n"
            "sub v15.16b, v15.16b, v9.16b\n"

            "sub v16.16b, v16.16b, v9.16b\n"
            "sub v17.16b, v17.16b, v9.16b\n"

            "sub v18.16b, v18.16b, v9.16b\n"
            "sub v19.16b, v19.16b, v9.16b\n"

            "sub v20.16b, v20.16b, v9.16b\n"
            "sub v21.16b, v21.16b, v9.16b\n"

            "sub v22.16b, v22.16b, v9.16b\n"
            "sub v23.16b, v23.16b, v9.16b\n"

            "sub v24.16b, v24.16b, v9.16b\n"
            "sub v25.16b, v25.16b, v9.16b\n"

            //! dot product into int32x4_t
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            //! 0
            "sdot v8.4s, v10.16b, v30.16b\n"
            "sdot v8.4s, v11.16b, v31.16b\n"
            "scvtf v8.4s, v8.4s\n"
            "fmla v0.4s, v8.4s, v26.s[0]\n"

            //! 1
            "sdot v9.4s, v12.16b, v30.16b\n"
            "sdot v9.4s, v13.16b, v31.16b\n"
            "scvtf v9.4s, v9.4s\n"
            "fmla v1.4s, v9.4s, v26.s[1]\n"

            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"

            //! 2
            "sdot v28.4s, v14.16b, v30.16b\n"
            "sdot v28.4s, v15.16b, v31.16b\n"
            "scvtf v28.4s, v28.4s\n"
            "fmla v2.4s, v28.4s, v26.s[2]\n"

            //! 3
            "sdot v29.4s, v16.16b, v30.16b\n"
            "sdot v29.4s, v17.16b, v31.16b\n"
            "scvtf v29.4s, v29.4s\n"
            "fmla v3.4s, v29.4s, v26.s[3]\n"

            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"

            //! 4
            "sdot v8.4s, v18.16b, v30.16b\n"
            "sdot v8.4s, v19.16b, v31.16b\n"
            "scvtf v8.4s, v8.4s\n"
            "fmla v4.4s, v8.4s, v27.s[0]\n"

            //! 5
            "sdot v9.4s, v20.16b, v30.16b\n"
            "sdot v9.4s, v21.16b, v31.16b\n"
            "scvtf v9.4s, v9.4s\n"
            "fmla v5.4s, v9.4s, v27.s[1]\n"

            //! 6
            "sdot v28.4s, v22.16b, v30.16b\n"
            "sdot v28.4s, v23.16b, v31.16b\n"
            "scvtf v28.4s, v28.4s\n"
            "fmla v6.4s, v28.4s, v27.s[2]\n"

            //! 7
            "sdot v29.4s, v24.16b, v30.16b\n"
            "sdot v29.4s, v25.16b, v31.16b\n"
            "scvtf v29.4s, v29.4s\n"
            "fmla v7.4s, v29.4s, v27.s[3]\n"

            //! loop end
            "subs %w[nb], %w[nb], #1\n"
            "bne 1b\n"

            //! store
            "faddp v0.4s, v0.4s, v1.4s\n"
            "faddp v2.4s, v2.4s, v3.4s\n"
            "faddp v4.4s, v4.4s, v5.4s\n"
            "faddp v6.4s, v6.4s, v7.4s\n"

            "faddp v0.4s, v0.4s, v2.4s\n"
            "faddp v1.4s, v4.4s, v6.4s\n"

            "st1 {v0.4s, v1.4s}, [%[dst]]\n"

            : [x] "+r"(x), [y] "+r"(y), [dst] "+r"(dst), [bias_ptr] "+r"(bias_ptr),
              [nb] "+r"(nb)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "cc", "memory");
}

}  // namespace opt
}  // namespace inferllm
