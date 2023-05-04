#pragma once
#if INFER_ARM
#include <assert.h>
#include "arm_neon.h"

#include "core/tensor.h"
#include "kern/kernel_define.h"
#include "kern/naive/naive.h"

namespace inferllm {
namespace opt {

inline void quantize_row_q4_0(const float* __restrict x, void* __restrict vy,
                              int k) {
    const int nb = k / QK40;

    BlockQ40* __restrict y = static_cast<BlockQ40*>(vy);
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int l = 0; l < 8; l++)
            srcv[l] = vld1q_f32(x + i * 32 + 4 * l);
        for (int l = 0; l < 8; l++)
            asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++)
            amaxv[2 * l] = vmaxq_f32(asrcv[2 * l], asrcv[2 * l + 1]);
        for (int l = 0; l < 2; l++)
            amaxv[4 * l] = vmaxq_f32(amaxv[4 * l], amaxv[4 * l + 2]);
        for (int l = 0; l < 1; l++)
            amaxv[8 * l] = vmaxq_f32(amaxv[8 * l], amaxv[8 * l + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const float32x4_t v = vmulq_n_f32(srcv[l], id);
            const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(8.5f));
            const int32x4_t vi = vcvtq_s32_f32(vf);

            y[i].qs[2 * l + 0] =
                    vgetq_lane_s32(vi, 0) | (vgetq_lane_s32(vi, 1) << 4);
            y[i].qs[2 * l + 1] =
                    vgetq_lane_s32(vi, 2) | (vgetq_lane_s32(vi, 3) << 4);
        }
    }
}

inline void dequantize_row_q4_0(const void* __restrict vx, float* __restrict y,
                                int k) {
    assert(k % QK40 == 0);
    const int nb = k / QK40;

    const BlockQ40* __restrict x = static_cast<const BlockQ40*>(vx);
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);

        const uint8_t* __restrict pp = x[i].qs;

        for (int l = 0; l < QK40; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l / 2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0f));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Convert to signed 8-bit integers
            const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
            const int8x8_t vs_1 = vreinterpret_s8_u8(v1);

            // Subtract 8 from each byte
            const int8x8_t vb_0 = vsub_s8(vs_0, vdup_n_s8(8));
            const int8x8_t vb_1 = vsub_s8(vs_1, vdup_n_s8(8));

            // Interleave and combine
            const int8x8_t vx_0 = vzip1_s8(vb_0, vb_1);
            const int8x8_t vx_1 = vzip2_s8(vb_0, vb_1);

            const int8x16_t vq = vcombine_s8(vx_0, vx_1);

            // convert to 2x int16x8_t
            const int16x8_t vi_0 = vmovl_s8(vget_low_s8(vq));
            const int16x8_t vi_1 = vmovl_s8(vget_high_s8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 =
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(vi_0)));
            const float32x4_t vf_1 =
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_0)));
            const float32x4_t vf_2 =
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(vi_1)));
            const float32x4_t vf_3 =
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_1)));

            // Multiply by d
            const float32x4_t r0 = vmulq_f32(vf_0, vd);
            const float32x4_t r1 = vmulq_f32(vf_1, vd);
            const float32x4_t r2 = vmulq_f32(vf_2, vd);
            const float32x4_t r3 = vmulq_f32(vf_3, vd);

            // Store
            vst1q_f32(y + i * QK40 + l + 0, r0);
            vst1q_f32(y + i * QK40 + l + 4, r1);
            vst1q_f32(y + i * QK40 + l + 8, r2);
            vst1q_f32(y + i * QK40 + l + 12, r3);
        }
    }
}

inline void quantize_row_q8_0(const float* __restrict x, void* __restrict vy,
                              int k) {
    assert(k % QK80 == 0);
    BlockQ80* y = static_cast<BlockQ80*>(vy);
    naive::quantize_row_q8_0_reference(x, y, k);
}

}  // namespace opt
}  // namespace inferllm
#endif