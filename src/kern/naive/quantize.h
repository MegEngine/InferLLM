#pragma once
#include <assert.h>
#include "kern/kernel_define.h"
#include "math.h"
#include "string.h"

namespace inferllm {
namespace naive {

inline void dequantize_row_q4_0_reference(
        const void* __restrict x, float* __restrict y, int k) {
    const int nb = k / QK40;
    const size_t bs = sizeof(float) + QK40 / 2;

    const uint8_t* __restrict pd = ((const uint8_t*)x + 0 * bs);
    const uint8_t* __restrict pb = ((const uint8_t*)x + 0 * bs + sizeof(float));

    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = *(const float*)(pd + i * bs);

        const uint8_t* __restrict pp = pb + i * bs;

        for (int l = 0; l < QK40; l += 2) {
            const uint8_t vi = pp[l / 2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8) * d;
            const float v1 = (vi1 - 8) * d;

            // printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n",
            // d, vi, vi0, vi1, v0, v1);

            y[i * QK40 + l + 0] = v0;
            y[i * QK40 + l + 1] = v1;
        }
    }
}

inline void dequantize_row_q8_0_reference(
        const void* __restrict x, float* __restrict y, int k) {
    const int nb = k / QK80;
    const size_t bs = sizeof(float) + QK40 / 2;

    const BlockQ80* xx = reinterpret_cast<const BlockQ80*>(x);

    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = xx[i].d;

        const int8_t* __restrict pp = xx[i].qs;

        for (int l = 0; l < QK80; l++) {
            y[i * QK80 + l] = pp[l] * d;
        }
    }
}

inline size_t quantize_row_q4_0_reference(const float* x, BlockQ40* y, int k) {
    const int nb = k / QK40;

    uint8_t pp[QK40 / 2];

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max

        for (int l = 0; l < QK40; l++) {
            const float v = x[i * QK40 + l];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK40; l += 2) {
            const float v0 = x[i * QK40 + l + 0] * id;
            const float v1 = x[i * QK40 + l + 1] * id;

            const uint8_t vi0 = (int8_t)roundf(v0) + 8;
            const uint8_t vi1 = (int8_t)roundf(v1) + 8;

            assert(vi0 < 16);
            assert(vi1 < 16);

            pp[l / 2] = vi0 | (vi1 << 4);
        }

        memcpy(y[i].qs, pp, sizeof(pp));
    }
    return nb * sizeof(BlockQ40);
}

//! quantize a row of float to int8
inline void quantize_row_q8_0_reference(const float* x, BlockQ80* y, int k) {
    assert(k % QK80 == 0);
    const int nb = k / QK80;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max

        for (int l = 0; l < QK80; l++) {
            const float v = x[i * QK80 + l];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK80; ++l) {
            const float v0 = x[i * QK80 + l] * id;

            y[i].qs[l] = roundf(v0);
        }
    }
}


inline float vec_vec_dot_q40_with_q80_reference(
        const int n, const void* vx, const void* vy) {
    const int nb = n / QK80;
    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ40* __restrict x = (BlockQ40*)(vx);
    const BlockQ80* __restrict y = (BlockQ80*)(vy);

    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        const uint8_t* __restrict p0 = x[i].qs;
        const int8_t* __restrict p1 = y[i].qs;

        int sumi = 0;
        for (int j = 0; j < QK80 / 2; j++) {
            const uint8_t v0 = p0[j];

            const int i0 = (int8_t)(v0 & 0x0F) - 8;
            const int i1 = (int8_t)(v0 >> 4) - 8;

            const int i2 = p1[2 * j + 0];
            const int i3 = p1[2 * j + 1];

            sumi += i0 * i2 + i1 * i3;
        }
        sumf += d0 * d1 * sumi;
    }
    return sumf;
}

inline void vec_vec_dot_q40_with_q80_packed_reference(
        const int n, const void* vx, const void* vy, float* dst, const float* bias) {
    const int nb = n / QK80;
    assert(n % QK80 == 0);

    const BlockQ40X8* __restrict x = (BlockQ40X8*)(vx);
    const BlockQ80* __restrict y = (BlockQ80*)(vy);
    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
    if (bias) {
        sum0 = bias[0], sum1 = bias[1], sum2 = bias[2], sum3 = bias[3], sum4 = bias[4],
        sum5 = bias[5], sum6 = bias[6], sum7 = bias[7];
    }
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
        const float d0 = scale[0];
        const float d1 = scale[1];
        const float d2 = scale[2];
        const float d3 = scale[3];
        const float d4 = scale[4];
        const float d5 = scale[5];
        const float d6 = scale[6];
        const float d7 = scale[7];

        const int8_t* __restrict y0 = y[i].qs;
        const float c0 = y[i].d;

        int i_sum0 = 0, i_sum1 = 0, i_sum2 = 0, i_sum3 = 0, i_sum4 = 0, i_sum5 = 0,
            i_sum6 = 0, i_sum7 = 0;
        for (int j = 0; j < QK80 / 2; j++) {
            const uint8_t v0 = x0[j], v1 = x1[j], v2 = x2[j], v3 = x3[j], v4 = x4[j],
                          v5 = x5[j], v6 = x6[j], v7 = x7[j];

            const int y2 = y0[2 * j + 0];
            const int y3 = y0[2 * j + 1];

            const int i00 = (int8_t)(v0 & 0x0F) - 8;
            const int i01 = (int8_t)(v0 >> 4) - 8;
            i_sum0 += i00 * y2 + i01 * y3;

            const int i10 = (int8_t)(v1 & 0x0F) - 8;
            const int i11 = (int8_t)(v1 >> 4) - 8;
            i_sum1 += i10 * y2 + i11 * y3;

            const int i20 = (int8_t)(v2 & 0x0F) - 8;
            const int i21 = (int8_t)(v2 >> 4) - 8;
            i_sum2 += i20 * y2 + i21 * y3;

            const int i30 = (int8_t)(v3 & 0x0F) - 8;
            const int i31 = (int8_t)(v3 >> 4) - 8;
            i_sum3 += i30 * y2 + i31 * y3;

            const int i40 = (int8_t)(v4 & 0x0F) - 8;
            const int i41 = (int8_t)(v4 >> 4) - 8;
            i_sum4 += i40 * y2 + i41 * y3;

            const int i50 = (int8_t)(v5 & 0x0F) - 8;
            const int i51 = (int8_t)(v5 >> 4) - 8;
            i_sum5 += i50 * y2 + i51 * y3;

            const int i60 = (int8_t)(v6 & 0x0F) - 8;
            const int i61 = (int8_t)(v6 >> 4) - 8;
            i_sum6 += i60 * y2 + i61 * y3;

            const int i70 = (int8_t)(v7 & 0x0F) - 8;
            const int i71 = (int8_t)(v7 >> 4) - 8;
            i_sum7 += i70 * y2 + i71 * y3;
        }
        sum0 += d0 * c0 * i_sum0;
        sum1 += d1 * c0 * i_sum1;
        sum2 += d2 * c0 * i_sum2;
        sum3 += d3 * c0 * i_sum3;
        sum4 += d4 * c0 * i_sum4;
        sum5 += d5 * c0 * i_sum5;
        sum6 += d6 * c0 * i_sum6;
        sum7 += d7 * c0 * i_sum7;
    }
    dst[0] = sum0;
    dst[1] = sum1;
    dst[2] = sum2;
    dst[3] = sum3;
    dst[4] = sum4;
    dst[5] = sum5;
    dst[6] = sum6;
    dst[7] = sum7;
}

inline float vec_vec_dot_q80_with_q80_reference(
        const int n, const void* vx, const void* vy) {
    const int nb = n / QK80;
    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ80* __restrict x = (BlockQ80*)(vx);
    const BlockQ80* __restrict y = (BlockQ80*)(vy);

    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        const int8_t* __restrict p0 = x[i].qs;
        const int8_t* __restrict p1 = y[i].qs;

        int sumi = 0;
        for (int j = 0; j < QK80; j++) {
            sumi += p0[j] * p1[j];
        }
        sumf += d0 * d1 * sumi;
    }
    return sumf;
}

inline float vec_vec_dot_float_with_float_reference(
        const int n, const float* x, const float* y) {
    float sumf = 0.0;
    for (int i = 0; i < n; i++) {
        sumf += x[i] * y[i];
    }
    return sumf;
}

}  // namespace naive
}  // namespace inferllm
