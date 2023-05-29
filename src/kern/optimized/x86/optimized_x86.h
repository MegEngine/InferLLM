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
inline float vec_vec_dot_q4_0(const int n, const void* __restrict x,
                               const void* __restrict y) {
    const int nb = n / QK40;

    assert(n % QK40 == 0);
    assert(nb % 2 == 0);

    const size_t bs = sizeof(float) + QK40 / 2;

    const uint8_t* __restrict pd0 = ((const uint8_t*)x + 0 * bs);
    const uint8_t* __restrict pd1 = ((const uint8_t*)y + 0 * bs);

    const uint8_t* __restrict pb0 =
            ((const uint8_t*)x + 0 * bs + sizeof(float));
    const uint8_t* __restrict pb1 =
            ((const uint8_t*)y + 0 * bs + sizeof(float));

    float sumf = 0.0;

    const size_t countBlocks = nb;

    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float* d0_0 = (const float*)(pd0 + i * bs);
        const float* d1_0 = (const float*)(pd1 + i * bs);

        const uint8_t* __restrict p0 = pb0 + i * bs;
        const uint8_t* __restrict p1 = pb1 + i * bs;

        // Compute combined scale for the block
        const __m256 scale = _mm256_mul_ps(_mm256_broadcast_ss(d0_0),
                                           _mm256_broadcast_ss(d1_0));

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        __m256i bx = bytesFromNibbles(p0);
        __m256i by = bytesFromNibbles(p1);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them
        // into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8(8);
        bx = _mm256_sub_epi8(bx, off);
        by = _mm256_sub_epi8(by, off);

        // Sign-extend first 16 signed bytes into int16_t
        __m256i x16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bx));
        __m256i y16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(by));
        // Compute products of int16_t integers, add pairwise
        __m256i i32 = _mm256_madd_epi16(x16, y16);

        // Sign-extend last 16 signed bytes into int16_t vectors
        x16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bx, 1));
        y16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(by, 1));
        // Accumulate products of int16_t integers
        i32 = _mm256_add_epi32(i32, _mm256_madd_epi16(x16, y16));

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps(i32);
        // Apply the scale, and accumulate
        acc = _mm256_fmadd_ps(scale, p, acc);
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps(acc, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));

    sumf = _mm_cvtss_f32(res);
    return sumf;
}

INFER_ATTRIBUTE_TARGET("default")
inline float vec_vec_dot_q4_0(const int n, const void* __restrict x,
                               const void* __restrict y) {
    const int nb = n / QK40;

    assert(n % QK40 == 0);
    assert(nb % 2 == 0);

    const size_t bs = sizeof(float) + QK40 / 2;

    const uint8_t* __restrict pd0 = ((const uint8_t*)x + 0 * bs);
    const uint8_t* __restrict pd1 = ((const uint8_t*)y + 0 * bs);

    const uint8_t* __restrict pb0 =
            ((const uint8_t*)x + 0 * bs + sizeof(float));
    const uint8_t* __restrict pb1 =
            ((const uint8_t*)y + 0 * bs + sizeof(float));

    float sumf = 0.0;

    // scalar
    for (int i = 0; i < nb; i++) {
        const float d0 = *(const float*)(pd0 + i * bs);
        const float d1 = *(const float*)(pd1 + i * bs);

        const uint8_t* __restrict p0 = pb0 + i * bs;
        const uint8_t* __restrict p1 = pb1 + i * bs;

        for (int j = 0; j < QK40 / 2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0 * ((int8_t)(v0 & 0xf) - 8);
            const float f1 = d0 * ((int8_t)(v0 >> 4) - 8);

            const float f2 = d1 * ((int8_t)(v1 & 0xf) - 8);
            const float f3 = d1 * ((int8_t)(v1 >> 4) - 8);

            sumf += f0 * f2 + f1 * f3;
        }
    }
    return sumf;
}

INFER_ATTRIBUTE_TARGET("avx2")
inline float vec_vec_dot_q40_with_q80(const int n, const void* __restrict vx,
                                      const void* __restrict vy) {
    const int nb = n / QK80;

    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ40* __restrict x = (const BlockQ40*)(vx);
    const BlockQ80* __restrict y = (const BlockQ80*)(vy);

    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        __m256i bx = bytesFromNibbles(x[i].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    return hsum_float_8(acc);
}

INFER_ATTRIBUTE_TARGET("avx")
inline float vec_vec_dot_q40_with_q80(const int n, const void* __restrict vx,
                                      const void* __restrict vy) {
    const int nb = n / QK80;

    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ40* __restrict x = (const BlockQ40*)(vx);
    const BlockQ80* __restrict y = (const BlockQ80*)(vy);
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        __m128i i32[2];
        for (int j = 0; j < 2; ++j) {
            // Load 8 bytes, and unpack 4 bit fields into bytes, making 16 bytes
            __m128i bx = bytes_from_nibbles_16(x[i].qs + 8*j);
            __m128i by = _mm_loadu_si128((const __m128i *)(y[i].qs + 16*j));

            // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
            const __m128i off = _mm_set1_epi8( 8 );
            bx = _mm_sub_epi8( bx, off );

            // Get absolute values of x vectors
            const __m128i ax = _mm_sign_epi8(bx, bx);

            // Sign the values of the y vectors
            const __m128i sy = _mm_sign_epi8(by, bx);

            // Perform multiplication and create 16-bit values
            const __m128i dot = _mm_maddubs_epi16(ax, sy);

            const __m128i ones = _mm_set1_epi16(1);
            i32[j] = _mm_madd_epi16(ones, dot);
        }

        // Convert int32_t to float
#if !defined(__GNUC__) || defined(__INTEL_COMPILER)
        __m256 p = _mm256_cvtepi32_ps( _mm256_set_m128i( i32[0], i32[1] ));
#else
        __m256 p = _mm256_cvtepi32_ps( _mm256_insertf128_si256( _mm256_castsi128_si256( i32[1] ), i32[0], 1 ));
#endif
        // Apply the scale, and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
    }
    return hsum_float_8(acc);
}

INFER_ATTRIBUTE_TARGET("default")
inline float vec_vec_dot_q40_with_q80(const int n, const void* __restrict vx,
                                      const void* __restrict vy) {
    const int nb = n / QK80;
    assert(n % QK80 == 0);
    assert(nb % 2 == 0);

    const BlockQ40* __restrict x = (const BlockQ40*)(vx);
    const BlockQ80* __restrict y = (const BlockQ80*)(vy);
    // scalar
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

INFER_ATTRIBUTE_TARGET("avx2")
inline void elemwise_vector_add(const int n, const float* __restrict x,
                                const float* __restrict y, float* __restrict z) {
    const int nb32 = n / 32;
    const int nb8 = (n - nb32 * 32) / 8;
    const int left = n - nb32 * 32 - nb8 * 8;
    for (int b32 = 0; b32 < nb32; b32++) {
        const float* __restrict x32 = (const float*)x + b32 * 32;
        const float* __restrict y32 = (const float*)y + b32 * 32;
        float* __restrict z32 = z + b32 * 32;
        __m256 vx0 = _mm256_loadu_ps(x32 + 0);
        __m256 vx1 = _mm256_loadu_ps(x32 + 8);
        __m256 vx2 = _mm256_loadu_ps(x32 + 16);
        __m256 vx3 = _mm256_loadu_ps(x32 + 24);
        __m256 vy0 = _mm256_loadu_ps(y32 + 0);
        __m256 vy1 = _mm256_loadu_ps(y32 + 8);
        __m256 vy2 = _mm256_loadu_ps(y32 + 16);
        __m256 vy3 = _mm256_loadu_ps(y32 + 24);

        __m256 vz0 = _mm256_add_ps(vx0, vy0);
        __m256 vz1 = _mm256_add_ps(vx1, vy1);
        __m256 vz2 = _mm256_add_ps(vx2, vy2);
        __m256 vz3 = _mm256_add_ps(vx3, vy3);

        _mm256_storeu_ps(z32 + 0, vz0);
        _mm256_storeu_ps(z32 + 8, vz1);
        _mm256_storeu_ps(z32 + 16, vz2);
        _mm256_storeu_ps(z32 + 24, vz3);
    }
    x = x + nb32 * 32;
    y = y + nb32 * 32;
    z = z + nb32 * 32;
    for (int b8 = 0; b8 < nb8; b8++) {
        const float* __restrict x8 = x + b8 * 8;
        const float* __restrict y8 = y + b8 * 8;
        float* __restrict z8 = z + b8 * 8;
        __m256 vx = _mm256_loadu_ps(x8);
        __m256 vy = _mm256_loadu_ps(y8);
        __m256 vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(z8, vz);
    }
    x = x + nb8 * 8;
    y = y + nb8 * 8;
    z = z + nb8 * 8;
    for (int i = 0; i < left; i++) {
        z[i] = x[i] + y[i];
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void elemwise_vector_add(const int n, const float* __restrict x,
                                const float* __restrict y,
                                float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

INFER_ATTRIBUTE_TARGET("avx2")
inline void elemwise_vector_mul(const int n, const float* __restrict x,
                                const float* __restrict y, float* __restrict z) {
    const int nb32 = n / 32;
    const int nb8 = (n - nb32 * 32) / 8;
    const int left = n - nb32 * 32 - nb8 * 8;
    for (int b32 = 0; b32 < nb32; b32++) {
        const float* __restrict x32 = (const float*)x + b32 * 32;
        const float* __restrict y32 = (const float*)y + b32 * 32;
        float* __restrict z32 = z + b32 * 32;
        __m256 vx0 = _mm256_loadu_ps(x32 + 0);
        __m256 vx1 = _mm256_loadu_ps(x32 + 8);
        __m256 vx2 = _mm256_loadu_ps(x32 + 16);
        __m256 vx3 = _mm256_loadu_ps(x32 + 24);
        __m256 vy0 = _mm256_loadu_ps(y32 + 0);
        __m256 vy1 = _mm256_loadu_ps(y32 + 8);
        __m256 vy2 = _mm256_loadu_ps(y32 + 16);
        __m256 vy3 = _mm256_loadu_ps(y32 + 24);

        __m256 vz0 = _mm256_mul_ps(vx0, vy0);
        __m256 vz1 = _mm256_mul_ps(vx1, vy1);
        __m256 vz2 = _mm256_mul_ps(vx2, vy2);
        __m256 vz3 = _mm256_mul_ps(vx3, vy3);

        _mm256_storeu_ps(z32 + 0, vz0);
        _mm256_storeu_ps(z32 + 8, vz1);
        _mm256_storeu_ps(z32 + 16, vz2);
        _mm256_storeu_ps(z32 + 24, vz3);
    }
    x = x + nb32 * 32;
    y = y + nb32 * 32;
    z = z + nb32 * 32;
    for (int b8 = 0; b8 < nb8; b8++) {
        const float* __restrict x8 = x + b8 * 8;
        const float* __restrict y8 = y + b8 * 8;
        float* __restrict z8 = z + b8 * 8;
        __m256 vx = _mm256_loadu_ps(x8);
        __m256 vy = _mm256_loadu_ps(y8);
        __m256 vz = _mm256_mul_ps(vx, vy);
        _mm256_storeu_ps(z8, vz);
    }
    x = x + nb8 * 8;
    y = y + nb8 * 8;
    z = z + nb8 * 8;
    for (int i = 0; i < left; i++) {
        z[i] = x[i] * y[i];
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void elemwise_vector_mul(const int n, const float* __restrict x,
                                const float* __restrict y,
                                float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

INFER_ATTRIBUTE_TARGET("avx2")
inline void elemwise_vector_silu(const int n, const float* __restrict x,
                                 float* __restrict z) {
    const int nb32 = n / 32;
    const int nb8 = (n - nb32 * 32) / 8;
    const int left = n - nb32 * 32 - nb8 * 8;
    __m256 zero = _mm256_setzero_ps(); 
    __m256 one = _mm256_set1_ps(1.0f);  
    for (int b32 = 0; b32 < nb32; b32++) {
        const float* __restrict x32 = x + b32 * 32;
        float* __restrict z32 = z + b32 * 32;
        __m256 vx0 = _mm256_loadu_ps(x32 + 0);
        __m256 vx1 = _mm256_loadu_ps(x32 + 8);
        __m256 vx2 = _mm256_loadu_ps(x32 + 16);
        __m256 vx3 = _mm256_loadu_ps(x32 + 24);

        __m256 neg_vx0 = _mm256_sub_ps(zero, vx0);
        __m256 neg_vx1 = _mm256_sub_ps(zero, vx1);
        __m256 neg_vx2 = _mm256_sub_ps(zero, vx2);
        __m256 neg_vx3 = _mm256_sub_ps(zero, vx3);

        __m256 exp_neg_vx0 = _mm256_add_ps(one, exp256_ps(neg_vx0));
        __m256 exp_neg_vx1 = _mm256_add_ps(one, exp256_ps(neg_vx1));
        __m256 exp_neg_vx2 = _mm256_add_ps(one, exp256_ps(neg_vx2));
        __m256 exp_neg_vx3 = _mm256_add_ps(one, exp256_ps(neg_vx3));

        __m256 vz0 = _mm256_mul_ps(vx0, _mm256_div_ps(one, exp_neg_vx0));
        __m256 vz1 = _mm256_mul_ps(vx1, _mm256_div_ps(one, exp_neg_vx1));
        __m256 vz2 = _mm256_mul_ps(vx2, _mm256_div_ps(one, exp_neg_vx2));
        __m256 vz3 = _mm256_mul_ps(vx3, _mm256_div_ps(one, exp_neg_vx3));

        _mm256_storeu_ps(z32 + 0, vz0);
        _mm256_storeu_ps(z32 + 8, vz1);
        _mm256_storeu_ps(z32 + 16, vz2);
        _mm256_storeu_ps(z32 + 24, vz3);
    }
    x = x + nb32 * 32;
    z = z + nb32 * 32;
    for (int b8 = 0; b8 < nb8; b8++) {
        const float* __restrict x8 = x + b8 * 8;
        float* __restrict z8 = z + b8 * 8;
        __m256 vx0 = _mm256_loadu_ps(x8);
        __m256 neg_vx0 = _mm256_sub_ps(zero, vx0);
        __m256 exp_neg_vx0 = _mm256_add_ps(one, exp256_ps(neg_vx0));
        __m256 vz0 = _mm256_mul_ps(vx0, _mm256_div_ps(one, exp_neg_vx0));
        _mm256_storeu_ps(z8, vz0);
    }
    x = x + nb8 * 8;
    z = z + nb8 * 8;
    for (int i = 0; i < left; i++) {
        z[i] = x[i] / (1 + exp(-x[i]));
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void elemwise_vector_silu(const int n, const float* __restrict x,
                                 float* __restrict z) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] / (1 + exp(-x[i]));
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void elemwise_vector_gelu(const int n, const float* __restrict x,
                                 float* __restrict z) {
    for (int i = 0; i < n; i++) {
        float src = x[i];
        z[i] = 0.5 * src *
               (1 + tanh(sqrt(2.0 / PI) * (src + PGELU * src * src * src)));
    }
}

INFER_ATTRIBUTE_TARGET("avx2")
inline void elemwise_vec_scale(const int n, const float* __restrict x,
                               float scale, float* __restrict z) {
    __m256 scalar_vec = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 result = _mm256_mul_ps(vx, scalar_vec);
        _mm256_storeu_ps(z + i, result);
    }
    for (; i < n; i++) {
        z[i] = x[i] * scale;
    }
}

INFER_ATTRIBUTE_TARGET("default")
inline void elemwise_vec_scale(const int n, const float* __restrict x,
                               float scale, float* __restrict z) {
    int i = 0;
    for (; i < n; i++) {
        z[i] = x[i] * scale;
    }
}

INFER_ATTRIBUTE_TARGET("avx2")
inline float reduce_square_sum(const int n, const float* __restrict x) {
    float result = 0.0f;
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 vx0 = _mm256_loadu_ps(x + i);
        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        __m256 vx2 = _mm256_loadu_ps(x + i + 16);
        __m256 vx3 = _mm256_loadu_ps(x + i + 24);
        __m256 x_squared0 = _mm256_mul_ps(vx0, vx0);
        __m256 x_squared1 = _mm256_mul_ps(vx1, vx1);
        __m256 x_squared2 = _mm256_mul_ps(vx2, vx2);
        __m256 x_squared3 = _mm256_mul_ps(vx3, vx3);
        __m256 sum_vec_tmp0 = _mm256_add_ps(x_squared0, x_squared1);
        __m256 sum_vec_tmp1 = _mm256_add_ps(x_squared2, x_squared3);
        sum_vec = _mm256_add_ps(sum_vec,
                                _mm256_add_ps(sum_vec_tmp0, sum_vec_tmp1));
    }
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 x_squared = _mm256_mul_ps(vx, vx);
        sum_vec = _mm256_add_ps(sum_vec, x_squared);
    }
    __m256 hadd1 = _mm256_hadd_ps(sum_vec, sum_vec);
    __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);
    __m128 low = _mm256_extractf128_ps(hadd2, 0);
    __m128 high = _mm256_extractf128_ps(hadd2, 1);
    __m128 sum = _mm_add_ps(low, high);
    result += _mm_cvtss_f32(sum);

    for (; i < n; i++) {
        result += x[i] * x[i];
    }
    return result;
}

INFER_ATTRIBUTE_TARGET("default")
inline float reduce_square_sum(const int n, const float* __restrict x) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

INFER_ATTRIBUTE_TARGET("avx2")
inline float reduce_max(const int n, const float* __restrict x) {
    float result = 0.0f;
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 vx0 = _mm256_loadu_ps(x + i);
        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        __m256 vx2 = _mm256_loadu_ps(x + i + 16);
        __m256 vx3 = _mm256_loadu_ps(x + i + 24);
        __m256 sum_vec_tmp0 = _mm256_max_ps(vx0, vx1);
        __m256 sum_vec_tmp1 = _mm256_max_ps(vx2, vx3);
        max_vec = _mm256_max_ps(max_vec,
                                _mm256_max_ps(sum_vec_tmp0, sum_vec_tmp1));
    }
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, vx);
    }
    __m128 low = _mm256_extractf128_ps(max_vec, 0);
    __m128 high = _mm256_extractf128_ps(max_vec, 1);
    __m128 max = _mm_max_ps(low, high);

    __m128 max1 = _mm_shuffle_ps(max, max, _MM_SHUFFLE(0, 0, 3, 2));
    __m128 max2 = _mm_max_ps(max, max1);
    __m128 max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 max4 = _mm_max_ps(max2, max3);

    result = _mm_cvtss_f32(max4);

    for (; i < n; i++) {
        result = std::max(x[i], result);
    }
    return result;
}

INFER_ATTRIBUTE_TARGET("default")
inline float reduce_max(const int n, const float* __restrict x) {
    float max = -INFINITY;
    for (int i = 0; i < n; i++) {
        max = std::max(max, x[i]);
    }
    return max;
}

INFER_ATTRIBUTE_TARGET("avx2")
inline float select_sub_max_and_reduce_sum(const int n,
                                           const float* __restrict x,
                                           float* __restrict y,
                                           const float max) {
    float result = 0.0f;
    __m256 infinit_v = _mm256_set1_ps(-INFINITY);
    __m256 zero_v = _mm256_setzero_ps();
    __m256 max_v = _mm256_set1_ps(max);
    __m256 sum_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 vx0 = _mm256_loadu_ps(x + i);
        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        __m256 vx2 = _mm256_loadu_ps(x + i + 16);
        __m256 vx3 = _mm256_loadu_ps(x + i + 24);
        //! mask = vx == infinit_v
        __m256 mask0 = _mm256_cmp_ps(vx0, infinit_v, _CMP_EQ_OQ);
        __m256 mask1 = _mm256_cmp_ps(vx1, infinit_v, _CMP_EQ_OQ);
        __m256 mask2 = _mm256_cmp_ps(vx2, infinit_v, _CMP_EQ_OQ);
        __m256 mask3 = _mm256_cmp_ps(vx3, infinit_v, _CMP_EQ_OQ);

        //! val = exp256_ps(vx - max_v) if mask == 0 else 0
        __m256 val0 = exp256_ps(_mm256_sub_ps(vx0, max_v));
        __m256 val1 = exp256_ps(_mm256_sub_ps(vx1, max_v));
        __m256 val2 = exp256_ps(_mm256_sub_ps(vx2, max_v));
        __m256 val3 = exp256_ps(_mm256_sub_ps(vx3, max_v));

        val0 = _mm256_blendv_ps(val0, zero_v, mask0);
        val1 = _mm256_blendv_ps(val1, zero_v, mask1);
        val2 = _mm256_blendv_ps(val2, zero_v, mask2);
        val3 = _mm256_blendv_ps(val3, zero_v, mask3);

        __m256 sum_vec_tmp0 = _mm256_add_ps(val0, val1);
        __m256 sum_vec_tmp1 = _mm256_add_ps(val2, val3);
        sum_v = _mm256_add_ps(sum_v, _mm256_add_ps(sum_vec_tmp0, sum_vec_tmp1));
        _mm256_storeu_ps(y + i, val0);
        _mm256_storeu_ps(y + i + 8, val1);
        _mm256_storeu_ps(y + i + 16, val2);
        _mm256_storeu_ps(y + i + 24, val3);
    }
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        //! mask = vx == infinit_v
        __m256 mask = _mm256_cmp_ps(vx, infinit_v, _CMP_EQ_OQ);
        //! val = exp256_ps(vx - max_v) if mask == 0 else 0
        __m256 val = exp256_ps(_mm256_sub_ps(vx, max_v));
        val = _mm256_blendv_ps(val, zero_v, mask);
        sum_v = _mm256_add_ps(sum_v, val);
        _mm256_storeu_ps(y + i, val);
    }
    __m256 hadd1 = _mm256_hadd_ps(sum_v, sum_v);
    __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);
    __m128 low = _mm256_extractf128_ps(hadd2, 0);
    __m128 high = _mm256_extractf128_ps(hadd2, 1);
    __m128 sum = _mm_add_ps(low, high);
    result = _mm_cvtss_f32(sum);

    for (; i < n; i++) {
        if (x[i] == -INFINITY) {
            y[i] = 0.0f;
        } else {
            float val = exp(x[i] - max);
            result += val;
            y[i] = val;
        }
    }
    return result;
}

INFER_ATTRIBUTE_TARGET("default")
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

INFER_ATTRIBUTE_TARGET("avx2")
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

            __m256 sum0v = _mm256_setzero_ps();
            __m256 sum1v = _mm256_setzero_ps();
            __m256 sum2v = _mm256_setzero_ps();
            __m256 sum3v = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 15 < sub_embd; k += 16) {
                __m256 qv0 = _mm256_loadu_ps(p_srcq + k);
                __m256 qv1 = _mm256_loadu_ps(p_srcq + k + 8);

                __m256 kv00 = _mm256_loadu_ps(p_srck0 + k);
                __m256 kv10 = _mm256_loadu_ps(p_srck1 + k);
                __m256 kv20 = _mm256_loadu_ps(p_srck2 + k);
                __m256 kv30 = _mm256_loadu_ps(p_srck3 + k);

                __m256 kv01 = _mm256_loadu_ps(p_srck0 + k + 8);
                __m256 kv11 = _mm256_loadu_ps(p_srck1 + k + 8);
                __m256 kv21 = _mm256_loadu_ps(p_srck2 + k + 8);
                __m256 kv31 = _mm256_loadu_ps(p_srck3 + k + 8);

                sum0v = _mm256_add_ps(sum0v,
                                      _mm256_add_ps(_mm256_mul_ps(qv0, kv00),
                                                    _mm256_mul_ps(qv1, kv01)));
                sum1v = _mm256_add_ps(sum1v,
                                      _mm256_add_ps(_mm256_mul_ps(qv0, kv10),
                                                    _mm256_mul_ps(qv1, kv11)));
                sum2v = _mm256_add_ps(sum2v,
                                      _mm256_add_ps(_mm256_mul_ps(qv0, kv20),
                                                    _mm256_mul_ps(qv1, kv21)));
                sum3v = _mm256_add_ps(sum3v,
                                      _mm256_add_ps(_mm256_mul_ps(qv0, kv30),
                                                    _mm256_mul_ps(qv1, kv31)));
            }
            for (; k + 7 < sub_embd; k += 8) {
                __m256 qv = _mm256_loadu_ps(p_srcq + k);
                __m256 kv0 = _mm256_loadu_ps(p_srck0 + k);
                __m256 kv1 = _mm256_loadu_ps(p_srck1 + k);
                __m256 kv2 = _mm256_loadu_ps(p_srck2 + k);
                __m256 kv3 = _mm256_loadu_ps(p_srck3 + k);

                sum0v = _mm256_add_ps(sum0v, _mm256_mul_ps(qv, kv0));
                sum1v = _mm256_add_ps(sum1v, _mm256_mul_ps(qv, kv1));
                sum2v = _mm256_add_ps(sum2v, _mm256_mul_ps(qv, kv2));
                sum3v = _mm256_add_ps(sum3v, _mm256_mul_ps(qv, kv3));
            }
            //! reduce sum
            float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            {
                __m256 hadd1 = _mm256_hadd_ps(sum0v, sum0v);
                __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);
                __m128 low = _mm256_extractf128_ps(hadd2, 0);
                __m128 high = _mm256_extractf128_ps(hadd2, 1);
                __m128 sum = _mm_add_ps(low, high);
                sum0 = _mm_cvtss_f32(sum);
            }
            {
                __m256 hadd1 = _mm256_hadd_ps(sum1v, sum1v);
                __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);
                __m128 low = _mm256_extractf128_ps(hadd2, 0);
                __m128 high = _mm256_extractf128_ps(hadd2, 1);
                __m128 sum = _mm_add_ps(low, high);
                sum1 = _mm_cvtss_f32(sum);
            }
            {
                __m256 hadd1 = _mm256_hadd_ps(sum2v, sum2v);
                __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);
                __m128 low = _mm256_extractf128_ps(hadd2, 0);
                __m128 high = _mm256_extractf128_ps(hadd2, 1);
                __m128 sum = _mm_add_ps(low, high);
                sum2 = _mm_cvtss_f32(sum);
            }
            {
                __m256 hadd1 = _mm256_hadd_ps(sum3v, sum3v);
                __m256 hadd2 = _mm256_hadd_ps(hadd1, hadd1);
                __m128 low = _mm256_extractf128_ps(hadd2, 0);
                __m128 high = _mm256_extractf128_ps(hadd2, 1);
                __m128 sum = _mm_add_ps(low, high);
                sum3 = _mm_cvtss_f32(sum);
            }
            for (; k < sub_embd; k++) {
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

INFER_ATTRIBUTE_TARGET("default")
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

//! because most case, the seqlen is 1, so we don't pack the srcv data to get
//! the best memory access. this optimize is only reuse the srcqk data and the
//! srcv data to compute the near dst data.
INFER_ATTRIBUTE_TARGET("avx2")
inline void comput_matmul_with_dst_uncontinue(
        float* __restrict dst, int offset_dst, const float* __restrict srcv,
        int offset_v, const float* __restrict srcqk, int seqlen, int length,
        int sub_embd) {
    uint32_t row = 0;
    for (; row + 3 < seqlen; row += 4) {
        auto p_qk0 = srcqk + row * length;
        auto p_qk1 = srcqk + (row + 1) * length;
        auto p_qk2 = srcqk + (row + 2) * length;
        auto p_qk3 = srcqk + (row + 3) * length;
        auto p_dst0 = dst + row * offset_dst;
        auto p_dst1 = dst + (row + 1) * offset_dst;
        auto p_dst2 = dst + (row + 2) * offset_dst;
        auto p_dst3 = dst + (row + 3) * offset_dst;
        uint32_t len = 0;
        for (; len + 15 < sub_embd; len += 16) {
            auto p_v = srcv + len;
            __m256 sum00 = _mm256_setzero_ps();
            __m256 sum01 = _mm256_setzero_ps();
            __m256 sum10 = _mm256_setzero_ps();
            __m256 sum11 = _mm256_setzero_ps();
            __m256 sum20 = _mm256_setzero_ps();
            __m256 sum21 = _mm256_setzero_ps();
            __m256 sum30 = _mm256_setzero_ps();
            __m256 sum31 = _mm256_setzero_ps();
            for (uint32_t k = 0; k < length; k++) {
                __m256 v0 = _mm256_loadu_ps(p_v + k * offset_v);
                __m256 v1 = _mm256_loadu_ps(p_v + k * offset_v + 8);
                __m256 qk0 = _mm256_set1_ps(p_qk0[k]);
                __m256 qk1 = _mm256_set1_ps(p_qk1[k]);
                __m256 qk2 = _mm256_set1_ps(p_qk2[k]);
                __m256 qk3 = _mm256_set1_ps(p_qk3[k]);
                sum00 = _mm256_add_ps(_mm256_mul_ps(v0, qk0), sum00);
                sum01 = _mm256_add_ps(_mm256_mul_ps(v1, qk0), sum01);
                sum10 = _mm256_add_ps(_mm256_mul_ps(v0, qk1), sum10);
                sum11 = _mm256_add_ps(_mm256_mul_ps(v1, qk1), sum11);
                sum20 = _mm256_add_ps(_mm256_mul_ps(v0, qk2), sum20);
                sum21 = _mm256_add_ps(_mm256_mul_ps(v1, qk2), sum21);
                sum30 = _mm256_add_ps(_mm256_mul_ps(v0, qk3), sum30);
                sum31 = _mm256_add_ps(_mm256_mul_ps(v1, qk3), sum31);
            }
            _mm256_storeu_ps(p_dst0 + len, sum00);
            _mm256_storeu_ps(p_dst0 + len + 8, sum01);
            _mm256_storeu_ps(p_dst1 + len, sum10);
            _mm256_storeu_ps(p_dst1 + len + 8, sum11);
            _mm256_storeu_ps(p_dst2 + len, sum20);
            _mm256_storeu_ps(p_dst2 + len + 8, sum21);
            _mm256_storeu_ps(p_dst3 + len, sum30);
            _mm256_storeu_ps(p_dst3 + len + 8, sum31);
        }
        for (; len < sub_embd; len++) {
            auto p_v = srcv + len;
            float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            for (uint32_t k = 0; k < length; k++) {
                sum0 += p_v[k * offset_v] * p_qk0[k];
                sum1 += p_v[k * offset_v] * p_qk1[k];
                sum2 += p_v[k * offset_v] * p_qk2[k];
                sum3 += p_v[k * offset_v] * p_qk3[k];
            }
            p_dst0[len] = sum0;
            p_dst1[len] = sum1;
            p_dst2[len] = sum2;
            p_dst3[len] = sum3;
        }
    }
    for (; row < seqlen; row++) {
        auto p_qk = srcqk + row * length;
        auto p_dst = dst + row * offset_dst;
        uint32_t len = 0;
        for (; len + 15 < sub_embd; len += 16) {
            auto p_v = srcv + len;
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            for (uint32_t k = 0; k < length; k++) {
                __m256 v0 = _mm256_loadu_ps(p_v + k * offset_v);
                __m256 v1 = _mm256_loadu_ps(p_v + k * offset_v + 8);
                __m256 qk = _mm256_set1_ps(p_qk[k]);
                sum0 = _mm256_add_ps(_mm256_mul_ps(v0, qk), sum0);
                sum1 = _mm256_add_ps(_mm256_mul_ps(v1, qk), sum1);
            }
            _mm256_storeu_ps(p_dst + len, sum0);
            _mm256_storeu_ps(p_dst + len + 8, sum1);
        }
        for (; len < sub_embd; len++) {
            auto p_v = srcv + len;
            float sum = 0;
            for (uint32_t k = 0; k < length; k++) {
                sum += p_v[k * offset_v] * p_qk[k];
            }
            p_dst[len] = sum;
        }
    }
}

INFER_ATTRIBUTE_TARGET("default")
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

}  // namespace opt
}  // namespace inferllm

#endif
