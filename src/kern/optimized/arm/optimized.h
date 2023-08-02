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

// inline void vec_vec_dot_q40_with_q80_packed_asm(
//         const int n, const void* __restrict vx, const void* __restrict vy, float* dst,
//         const float* bias) {
//     const int nb = n / QK80;

//     assert(n % QK80 == 0);
//     assert(nb % 2 == 0);

//     const BlockQ40* __restrict x = (BlockQ40*)vx;
//     const BlockQ80* __restrict y = (BlockQ80*)vy;
//     int offset = 20;

//     float bias_v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
//     const float* bias_ptr = bias ? bias : bias_v;

//     asm volatile(
//             //! set all sum to 0
//             "eor v0.16b, v0.16b, v0.16b\n"
//             "eor v1.16b, v1.16b, v1.16b\n"
//             "eor v2.16b, v2.16b, v2.16b\n"
//             "eor v3.16b, v3.16b, v3.16b\n"
//             "eor v4.16b, v4.16b, v4.16b\n"
//             "eor v5.16b, v5.16b, v5.16b\n"
//             "eor v6.16b, v6.16b, v6.16b\n"
//             "eor v7.16b, v7.16b, v7.16b\n"

//             //! load bias
//             "ld1 {v8}, [%[bias]], #16\n"
//             "ld1 {v9}, [%[bias]], #16\n"

//             //! main loop
//             "1:\n"
//             "mov x0, %[x]\n"
//             "ld1 {v10.16b}, [%[x]], #20\n"
//             "ld1 {v12.16b}, [%[x]], #20\n"
//             "ld1 {v14.16b}, [%[x]], #20\n"
//             "ld1 {v16.16b}, [%[x]], #20\n"
//             "ld1 {v18.16b}, [%[x]], #20\n"
//             "ld1 {v20.16b}, [%[x]], #20\n"
//             "ld1 {v22.16b}, [%[x]], #20\n"
//             "ld1 {v24.16b}, [%[x]], #20\n"

            



//             "\n"
//             : [x] "+r"(x), [y] "+r"(y), [dst] "+r"(dst), [bias] "+r"(bias_ptr),
//               [nb] "+r"(nb), [offset] "+r"(offset)
//             :
//             : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
//               "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
//               "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
//               "cc", "memory");

//     float32x4_t zero = vdupq_n_f32(0.0f);
//     uint8x16_t m4b = vdupq_n_u8(0x0F);
//     int8x16_t s8b = vdupq_n_s8(0x8);


//     float32x4_t sumv0 = vdupq_n_f32(bias_ptr[0]);
//     float32x4_t sumv1 = vdupq_n_f32(bias_ptr[1]);
//     float32x4_t sumv2 = vdupq_n_f32(bias_ptr[2]);
//     float32x4_t sumv3 = vdupq_n_f32(bias_ptr[3]);
//     float32x4_t sumv4 = vdupq_n_f32(bias_ptr[4]);
//     float32x4_t sumv5 = vdupq_n_f32(bias_ptr[5]);
//     float32x4_t sumv6 = vdupq_n_f32(bias_ptr[6]);
//     float32x4_t sumv7 = vdupq_n_f32(bias_ptr[7]);

//     for (int i = 0; i < nb; i++) {
//         int id = i * 8;
//         const BlockQ40* __restrict x0 = &x[id + 0];
//         const BlockQ40* __restrict x1 = &x[id + 1];
//         const BlockQ40* __restrict x2 = &x[id + 2];
//         const BlockQ40* __restrict x3 = &x[id + 3];
//         const BlockQ40* __restrict x4 = &x[id + 4];
//         const BlockQ40* __restrict x5 = &x[id + 5];
//         const BlockQ40* __restrict x6 = &x[id + 6];
//         const BlockQ40* __restrict x7 = &x[id + 7];

//         const BlockQ80* __restrict y0 = &y[i];

//         // load y
//         int8x16_t vy0l = vld1q_s8(y0->qs);
//         int8x16_t vy0h = vld1q_s8(y0->qs + 16);

//         // interleave
//         int8x16_t vyls = vuzp1q_s8(vy0l, vy0h);
//         int8x16_t vyhs = vuzp2q_s8(vy0l, vy0h);

//         uint8x16_t v0 = vld1q_u8(x0->qs);
//         uint8x16_t v1 = vld1q_u8(x1->qs);
//         uint8x16_t v2 = vld1q_u8(x2->qs);
//         uint8x16_t v3 = vld1q_u8(x3->qs);
//         uint8x16_t v4 = vld1q_u8(x4->qs);
//         uint8x16_t v5 = vld1q_u8(x5->qs);
//         uint8x16_t v6 = vld1q_u8(x6->qs);
//         uint8x16_t v7 = vld1q_u8(x7->qs);

//         // 4-bit -> 8-bit
//         int8x16_t v0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
//         int8x16_t v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));

//         int8x16_t v1l = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
//         int8x16_t v1h = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));

//         int8x16_t v2l = vreinterpretq_s8_u8(vandq_u8(v2, m4b));
//         int8x16_t v2h = vreinterpretq_s8_u8(vshrq_n_u8(v2, 4));

//         int8x16_t v3l = vreinterpretq_s8_u8(vandq_u8(v3, m4b));
//         int8x16_t v3h = vreinterpretq_s8_u8(vshrq_n_u8(v3, 4));

//         int8x16_t v4l = vreinterpretq_s8_u8(vandq_u8(v4, m4b));
//         int8x16_t v4h = vreinterpretq_s8_u8(vshrq_n_u8(v4, 4));

//         int8x16_t v5l = vreinterpretq_s8_u8(vandq_u8(v5, m4b));
//         int8x16_t v5h = vreinterpretq_s8_u8(vshrq_n_u8(v5, 4));
        
//         int8x16_t v6l = vreinterpretq_s8_u8(vandq_u8(v6, m4b));
//         int8x16_t v6h = vreinterpretq_s8_u8(vshrq_n_u8(v6, 4));
        
//         int8x16_t v7l = vreinterpretq_s8_u8(vandq_u8(v7, m4b));
//         int8x16_t v7h = vreinterpretq_s8_u8(vshrq_n_u8(v7, 4));

//         // sub 8
//         int8x16_t v0ls = vsubq_s8(v0l, s8b);
//         int8x16_t v0hs = vsubq_s8(v0h, s8b);
//         int8x16_t v1ls = vsubq_s8(v1l, s8b);
//         int8x16_t v1hs = vsubq_s8(v1h, s8b);
//         int8x16_t v2ls = vsubq_s8(v2l, s8b);
//         int8x16_t v2hs = vsubq_s8(v2h, s8b);
//         int8x16_t v3ls = vsubq_s8(v3l, s8b);
//         int8x16_t v3hs = vsubq_s8(v3h, s8b);
//         int8x16_t v4ls = vsubq_s8(v4l, s8b);
//         int8x16_t v4hs = vsubq_s8(v4h, s8b);
//         int8x16_t v5ls = vsubq_s8(v5l, s8b);
//         int8x16_t v5hs = vsubq_s8(v5h, s8b);
//         int8x16_t v6ls = vsubq_s8(v6l, s8b);
//         int8x16_t v6hs = vsubq_s8(v6h, s8b);
//         int8x16_t v7ls = vsubq_s8(v7l, s8b);
//         int8x16_t v7hs = vsubq_s8(v7h, s8b);

//         // dot product into int32x4_t
//         int32x4_t p0 = vdotq_s32(vdotq_s32(zero, v0ls, vyls), v0hs, vyhs);
//         int32x4_t p1 = vdotq_s32(vdotq_s32(zero, v1ls, vyls), v1hs, vyhs);
//         int32x4_t p2 = vdotq_s32(vdotq_s32(zero, v2ls, vyls), v2hs, vyhs);
//         int32x4_t p3 = vdotq_s32(vdotq_s32(zero, v3ls, vyls), v3hs, vyhs);
//         int32x4_t p4 = vdotq_s32(vdotq_s32(zero, v4ls, vyls), v4hs, vyhs);
//         int32x4_t p5 = vdotq_s32(vdotq_s32(zero, v5ls, vyls), v5hs, vyhs);
//         int32x4_t p6 = vdotq_s32(vdotq_s32(zero, v6ls, vyls), v6hs, vyhs);
//         int32x4_t p7 = vdotq_s32(vdotq_s32(zero, v7ls, vyls), v7hs, vyhs);

//         sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), x0->d * y0->d);
//         sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), x1->d * y0->d);
//         sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), x2->d * y0->d);
//         sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), x3->d * y0->d);
//         sumv4 = vmlaq_n_f32(sumv4, vcvtq_f32_s32(p4), x4->d * y0->d);
//         sumv5 = vmlaq_n_f32(sumv5, vcvtq_f32_s32(p5), x5->d * y0->d);
//         sumv6 = vmlaq_n_f32(sumv6, vcvtq_f32_s32(p6), x6->d * y0->d);
//         sumv7 = vmlaq_n_f32(sumv7, vcvtq_f32_s32(p7), x7->d * y0->d);
//     }
//     float32x4_t sumv_0 = vpaddq_f32(vpaddq_f32(sumv0, sumv1), vpaddq_f32(sumv2, sumv3));
//     float32x4_t sumv_1 = vpaddq_f32(vpaddq_f32(sumv4, sumv5), vpaddq_f32(sumv6, sumv7));
//     vst1q_f32(dst, sumv_0);
//     vst1q_f32(dst + 4, sumv_1);
// }


}  // namespace opt
}  // namespace inferllm
