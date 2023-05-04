#pragma once
#if INFER_X86
#include <assert.h>
#include <immintrin.h>
#include "core/tensor.h"
#include "kern/kernel_define.h"
#include "common.h"

namespace inferllm {
namespace opt {

INFER_ATTRIBUTE_TARGET("avx2")
inline void quantize_row_q4_0(const float* __restrict x, void* __restrict vy,
                              int k) {
    const int nb = k / QK40;

    BlockQ40* __restrict y = static_cast<BlockQ40*>(vy);

    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                                 _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float d = maxScalar / 7.0f;
        y[i].d = d;
        const float id = (maxScalar != 0.0f) ? 7.0f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Apply the multiplier
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        // Convert int32 to int16
        i0 = _mm256_packs_epi32(
                i0,
                i1);  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32(
                i2, i3);  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23,
                          // 28, 29, 30, 31 Convert int16 to int8
        i0 = _mm256_packs_epi16(
                i0, i2);  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24,
                          // 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21,
                          // 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 ..
        // +15 ]
        const __m256i off = _mm256_set1_epi8(8);
        i0 = _mm256_add_epi8(i0, off);

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles(i0);
        _mm_storeu_si128((__m128i*)y[i].qs, res);
    }
}

INFER_ATTRIBUTE_TARGET("avx2")
inline void dequantize_row_q4_0(const void* __restrict vx, float* __restrict y,
                                int k) {
    assert(k % QK40 == 0);
    const int nb = k / QK40;

    const BlockQ40* __restrict x = static_cast<const BlockQ40*>(vx);

    for (int i = 0; i < nb; i++) {
        // scale factor
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);

        const uint8_t* __restrict pp = x[i].qs;

        for (int l = 0; l < QK40; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytesFromNibbles(pp + l / 2);

            // Subtract 8 from the integers
            vx8 = _mm256_sub_epi8(vx8, _mm256_set1_epi8(8));

            // Convert to 16-bit int
            const __m256i vx16_lo =
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi =
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                            _mm256_extracti128_si256(vx16_lo, 0))),
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                            _mm256_extracti128_si256(vx16_lo, 1))),
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                            _mm256_extracti128_si256(vx16_hi, 0))),
                    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                            _mm256_extracti128_si256(vx16_hi, 1)))};

            // Scale and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_mul_ps(vf[j], d_v);
                _mm256_storeu_ps(y + i * QK40 + l + j * 8, result);
            }
        }
    }
}

INFER_ATTRIBUTE_TARGET("avx")
inline void quantize_row_q4_0(const float* __restrict x, void* __restrict vy,
                              int k) {
    const int nb = k / QK40;

    BlockQ40* __restrict y = static_cast<BlockQ40*>(vy);

    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                                 _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float d = maxScalar / 7.0f;
        y[i].d = d;
        const float id = (maxScalar != 0.0f) ? 7.0f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Apply the multiplier
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128(i0);
        __m128i ni1 = _mm256_extractf128_si256(i0, 1);
        __m128i ni2 = _mm256_castsi256_si128(i1);
        __m128i ni3 = _mm256_extractf128_si256(i1, 1);
        __m128i ni4 = _mm256_castsi256_si128(i2);
        __m128i ni5 = _mm256_extractf128_si256(i2, 1);
        __m128i ni6 = _mm256_castsi256_si128(i3);
        __m128i ni7 = _mm256_extractf128_si256(i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32(ni0, ni1);
        ni2 = _mm_packs_epi32(ni2, ni3);
        ni4 = _mm_packs_epi32(ni4, ni5);
        ni6 = _mm_packs_epi32(ni6, ni7);
        // Convert int16 to int8
        ni0 = _mm_packs_epi16(ni0, ni2);
        ni4 = _mm_packs_epi16(ni4, ni6);

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 ..
        // +15 ]
        const __m128i off = _mm_set1_epi8(8);
        ni0 = _mm_add_epi8(ni0, off);
        ni4 = _mm_add_epi8(ni4, off);

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles(ni0, ni4);
        _mm_storeu_si128((__m128i*)y[i].qs, res);
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void quantize_row_q4_0(const float* __restrict x, void* __restrict vy,
                              int k) {
    const int nb = k / QK40;

    BlockQ40* __restrict y = static_cast<BlockQ40*>(vy);
    // scalar
    naive::quantize_row_q4_0_reference(x, y, k);
}

INFER_ATTRIBUTE_TARGET("default")
inline void dequantize_row_q4_0(const void* __restrict vx, float* __restrict y,
                                int k) {
    assert(k % QK40 == 0);
    const int nb = k / QK40;

    const BlockQ40* __restrict x = static_cast<const BlockQ40*>(vx);

    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        const uint8_t* __restrict pp = x[i].qs;

        for (int l = 0; l < QK40; l += 2) {
            const uint8_t vi = pp[l / 2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8) * d;
            const float v1 = (vi1 - 8) * d;

            y[i * QK40 + l + 0] = v0;
            y[i * QK40 + l + 1] = v1;

            assert(!isnan(y[i * QK40 + l + 0]));
            assert(!isnan(y[i * QK40 + l + 1]));
        }
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void quantize_row_q8_0(const float* __restrict x, void* __restrict vy,
                              int k) {
    assert(k % QK80 == 0);
    BlockQ80* y = static_cast<BlockQ80*>(vy);
    naive::quantize_row_q8_0_reference(x, y, k);
}

}  // namespace opt
}  // namespace inferllm
#endif