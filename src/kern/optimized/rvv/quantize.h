#pragma once

#include <cmath>

#include "core/tensor.h"
#include "kern/kernel_define.h"
#include "kern/naive/naive.h"
#include "kern/naive/quantize.h"

#include "common.h"

namespace inferllm {
namespace opt {

inline void dequantize_row_q4_0(const void* __restrict x, float* __restrict y, int k) {
    const int nb = k / QK40;
    const BlockQ40* vx = static_cast<const BlockQ40*>(x);

    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = vx[i].d;

        const uint8_t* pp = vx[i].qs;

        for (int l = 0; l < QK40; l += 2) {
            const uint8_t vi = pp[l / 2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8) * d;
            const float v1 = (vi1 - 8) * d;

            y[i * QK40 + l + 0] = v0;
            y[i * QK40 + l + 1] = v1;
        }
    }
}

inline void quantize_row_q8_0(const float* __restrict x, void* __restrict vy, int k) {
    BlockQ80* y = static_cast<BlockQ80*>(vy);

    const int nb = k / QK80;

    for (int i = 0; i < nb; i++) {
        float amax = vmaxabs(QK80, &x[i * QK80], 0.0);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK80; ++l) {
            const float v0 = x[i * QK80 + l] * id;

            y[i].qs[l] = roundf(v0);
        }
    }
}

}  // namespace opt
}  // namespace inferllm
