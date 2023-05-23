#pragma once
#include <assert.h>
#include "kern/kernel_define.h"
#include "math.h"
#include "string.h"

namespace inferllm {
namespace naive {

inline void dequantize_row_q4_0_reference(const void* __restrict x,
                                          float* __restrict y, int k) {
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

inline float vec_vec_dot_q40_with_q80_reference(const int n, const void* vx,
                                                const void* vy) {
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

inline float vec_vec_dot_float_with_float_reference(const int n, const float* x,
                                                    const float* y) {
    float sumf = 0.0;
    for (int i = 0; i < n; i++) {
        sumf += x[i] * y[i];
    }
    return sumf;
}

}  // namespace naive
}  // namespace inferllm