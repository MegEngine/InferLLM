#pragma once
#if INFER_X86
#include <assert.h>
#include <immintrin.h>
#include "core/tensor.h"
#include "kern/kernel_define.h"

namespace inferllm {
namespace opt {

INFER_ATTRIBUTE_TARGET("avx2")
static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
}

INFER_ATTRIBUTE_TARGET("avx")
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
INFER_ATTRIBUTE_TARGET("avx2")
static inline __m256i bytesFromNibbles( const uint8_t* rsi )
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}

INFER_ATTRIBUTE_TARGET("avx2")
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

INFER_ATTRIBUTE_TARGET("avx2")
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
}

INFER_ATTRIBUTE_TARGET("avx2")
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Unpack 16 4-bit fields into 16 bytes
// The output vector contains 16 bytes, each one in [ 0 .. 15 ] interval
INFER_ATTRIBUTE_TARGET("avx2")
static inline __m128i bytes_from_nibbles_16(const uint8_t * rsi)
{
    // Load 8 bytes from memory
    __m128i tmp = _mm_loadl_epi64( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m128i bytes = _mm_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m128i lowMask = _mm_set1_epi8( 0xF );
    __m128i high = _mm_andnot_si128( lowMask, bytes );
    __m128i low = _mm_and_si128( lowMask, bytes );
    high = _mm_slli_epi16( high, 4 );
    bytes = _mm_or_si128( low, high );
    return bytes;
}


/* __m128 is ugly to write */
typedef __m256 v8sf;   // vector of 8 float (avx)
typedef __m256i v8si;  // vector of 8 int   (avx)
typedef __m128i v4si;  // vector of 8 int   (avx)

#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))

#define _PS256_CONST(Name, Val)                                     \
    static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { \
            Val, Val, Val, Val, Val, Val, Val, Val}

#define _PI32_CONST256(Name, Val)                                    \
    static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = { \
            Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

INFER_ATTRIBUTE_TARGET("avx2")
static inline v8sf exp256_ps(v8sf x) {
    v8sf tmp = _mm256_setzero_ps(), fx;
    v8si imm0;
    v8sf one = *(v8sf*)_ps256_1;

    x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    // imm0 = _mm256_cvttps_epi32(fx);
    // tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, subtract 1 */
    // v8sf mask = _mm256_cmpgt_ps(tmp, fx);
    v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
    v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x, x);

    v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    v8sf pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

}  // namespace opt
}  // namespace inferllm

#endif