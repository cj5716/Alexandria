#pragma once
#include <cstdint>

#if defined(USE_SIMD)
#include <immintrin.h>
#endif

#if defined(USE_AVX512)
using VecEpi    = __m512i;
using Vec256Epi = __m256i;
using VecPs     = __m256;

inline VecEpi vec_set1_epi16 (const int16_t n) { return _mm512_set1_epi16(n); }
inline VecEpi vec_loadu_epi  (const VecEpi *src) { return _mm512_loadu_si512(src); }
inline void   vec_storeu_epi (VecEpi *dst, const VecEpi vec) { _mm512_storeu_si512(dst, vec); }
inline VecEpi vec_max_epi16  (const VecEpi vec0, const VecEpi vec1) { return _mm512_max_epi16(vec0, vec1); }
inline VecEpi vec_min_epi16  (const VecEpi vec0, const VecEpi vec1) { return _mm512_min_epi16(vec0, vec1); }
inline VecEpi vec_slli_epi16 (const VecEpi vec, const int shift) { return _mm512_slli_epi16(vec, shift); }
inline VecEpi vec_mulhi_epu16(const VecEpi vec0, const VecEpi vec1) { return _mm512_mulhi_epu16(vec0, vec1); }
inline VecEpi vec_madd_epi16 (const VecEpi vec0, const VecEpi vec1) { return _mm512_madd_epi16(vec0, vec1); }
inline VecEpi vec_add_epi32  (const VecEpi vec0, const VecEpi vec1) { return _mm512_add_epi32(vec0, vec1); }

inline VecPs vec_set1_ps   (const float n) { return _mm256_set1_ps(n); }
inline VecPs vec_loadu_ps  (const float *src) { return _mm256_loadu_ps(src); }
inline void  vec_storeu_ps (float *dst, const VecPs vec) { _mm256_storeu_ps(dst, vec); }
inline VecPs vec_cvtepi_ps (const Vec256Epi vec) { return _mm256_cvtepi32_ps(vec); }
inline VecPs vec_add_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_add_ps(vec0, vec1); }
inline VecPs vec_mul_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_mul_ps(vec0, vec1); }
inline VecPs vec_div_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_div_ps(vec0, vec1); }
inline VecPs vec_max_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_max_ps(vec0, vec1); }
inline VecPs vec_min_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_min_ps(vec0, vec1); }
inline VecPs vec_mul_add_ps(const VecPs vec0, const VecPs vec1, const VecPs vec2) { return _mm256_fmadd_ps(vec0, vec1, vec2); }
inline VecPs vec_hadd_ps   (const VecPs vec0, const VecPs vec1) { return _mm256_hadd_ps(vec0, vec1); }
inline VecPs vec_hadd_psx4 (const VecPs *vecs) {
    const VecPs sum01 = vec_hadd_ps(vecs[0], vecs[1]);
    const VecPs sum23 = vec_hadd_ps(vecs[2], vecs[3]);
    return vec_hadd_ps(sum01, sum23);
}
inline VecPs vec_comb_ps(const VecPs vec0, const VecPs vec1) {
    const __m128 vec0_low = _mm256_castps256_ps128(vec0);
    const __m128 vec0_hi = _mm256_extractf128_ps(vec0, 1);
    const __m128 vec0_m128 = _mm_add_ps(vec0_low, vec0_hi);

    const __m128 vec1_low = _mm256_castps256_ps128(vec1);
    const __m128 vec1_hi = _mm256_extractf128_ps(vec1, 1);
    const __m128 vec1_m128 = _mm_add_ps(vec1_low, vec1_hi);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(vec0_m128), vec1_m128, 1);
}
inline float vec_reduce_add_ps(const VecPs vec) {
    const __m128 upper_128 = _mm256_extractf128_ps(vec, 1);
    const __m128 lower_128 = _mm256_castps256_ps128(vec);
    const __m128 sum_128 = _mm_add_ps(upper_128, lower_128);

    const __m128 upper_64 = _mm_movehl_ps(sum_128, sum_128);
    const __m128 sum_64 = _mm_add_ps(upper_64, sum_128);

    const __m128 upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
    const __m128 sum_32 = _mm_add_ss(upper_32, sum_64);

    return _mm_cvtss_f32(sum_32);
}

inline Vec256Epi vec256_hadd_epi32  (const Vec256Epi vec0, const Vec256Epi vec1) { return _mm256_hadd_epi32(vec0, vec1); }
inline Vec256Epi vec256_hadd_epi32x4(const VecEpi *vecs) {
    auto cvt_vec512i_vec256i = [](const VecEpi vec) {
        const Vec256Epi upper256 = _mm512_extracti64x4_epi64(vec, 1); // same as _mm512_extracti32x8_epi32, but doesn't require AVX512DQ
        const Vec256Epi lower256 = _mm512_castsi512_si256(vec);
        return _mm256_add_epi32(lower256, upper256);
    };
    const Vec256Epi sum01 = _mm256_hadd_epi32(cvt_vec512i_vec256i(vecs[0]), cvt_vec512i_vec256i(vecs[1]));
    const Vec256Epi sum23 = _mm256_hadd_epi32(cvt_vec512i_vec256i(vecs[2]), cvt_vec512i_vec256i(vecs[3]));
    return _mm256_hadd_epi32(sum01, sum23);
}
inline Vec256Epi vec256_comb_epi32(const Vec256Epi vec0, const Vec256Epi vec1) {
    const __m128i vec0_low = _mm256_castsi256_si128(vec0);
    const __m128i vec0_hi = _mm256_extracti128_si256(vec0, 1);
    const __m128i vec0_m128 = _mm_add_epi32(vec0_low, vec0_hi);

    const __m128i vec1_low = _mm256_castsi256_si128(vec1);
    const __m128i vec1_hi = _mm256_extracti128_si256(vec1, 1);
    const __m128i vec1_m128 = _mm_add_epi32(vec1_low, vec1_hi);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(vec0_m128), vec1_m128, 1);
}

#elif defined(USE_AVX2)
using VecEpi    = __m256i;
using Vec256Epi = __m256i;
using VecPs     = __m256;

inline VecEpi vec_set1_epi16 (const int16_t n) { return _mm256_set1_epi16(n); }
inline VecEpi vec_loadu_epi  (const VecEpi *src) { return _mm256_loadu_si256(src); }
inline void   vec_storeu_epi (VecEpi *dst, const VecEpi vec) { _mm256_storeu_si256(dst, vec); }
inline VecEpi vec_max_epi16  (const VecEpi vec0, const VecEpi vec1) { return _mm256_max_epi16(vec0, vec1); }
inline VecEpi vec_min_epi16  (const VecEpi vec0, const VecEpi vec1) { return _mm256_min_epi16(vec0, vec1); }
inline VecEpi vec_slli_epi16 (const VecEpi vec, const int shift) { return _mm256_slli_epi16(vec, shift); }
inline VecEpi vec_mulhi_epu16(const VecEpi vec0, const VecEpi vec1) { return _mm256_mulhi_epu16(vec0, vec1); }
inline VecEpi vec_madd_epi16 (const VecEpi vec0, const VecEpi vec1) { return _mm256_madd_epi16(vec0, vec1); }
inline VecEpi vec_add_epi32  (const VecEpi vec0, const VecEpi vec1) { return _mm256_add_epi32(vec0, vec1); }

inline VecPs vec_set1_ps   (const float n) { return _mm256_set1_ps(n); }
inline VecPs vec_loadu_ps  (const float *src) { return _mm256_loadu_ps(src); }
inline void  vec_storeu_ps (float *dst, const VecPs vec) { _mm256_storeu_ps(dst, vec); }
inline VecPs vec_cvtepi_ps (const Vec256Epi vec) { return _mm256_cvtepi32_ps(vec); }
inline VecPs vec_add_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_add_ps(vec0, vec1); }
inline VecPs vec_mul_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_mul_ps(vec0, vec1); }
inline VecPs vec_div_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_div_ps(vec0, vec1); }
inline VecPs vec_max_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_max_ps(vec0, vec1); }
inline VecPs vec_min_ps    (const VecPs vec0, const VecPs vec1) { return _mm256_min_ps(vec0, vec1); }
inline VecPs vec_mul_add_ps(const VecPs vec0, const VecPs vec1, const VecPs vec2) { return _mm256_fmadd_ps(vec0, vec1, vec2); }
inline VecPs vec_hadd_ps   (const VecPs vec0, const VecPs vec1) { return _mm256_hadd_ps(vec0, vec1); }
inline VecPs vec_hadd_psx4 (const VecPs *vecs) {
    const VecPs sum01 = vec_hadd_ps(vecs[0], vecs[1]);
    const VecPs sum23 = vec_hadd_ps(vecs[2], vecs[3]);
    return vec_hadd_ps(sum01, sum23);
}
inline VecPs vec_comb_ps(const VecPs vec0, const VecPs vec1) {
    const __m128 vec0_low = _mm256_castps256_ps128(vec0);
    const __m128 vec0_hi = _mm256_extractf128_ps(vec0, 1);
    const __m128 vec0_m128 = _mm_add_ps(vec0_low, vec0_hi);

    const __m128 vec1_low = _mm256_castps256_ps128(vec1);
    const __m128 vec1_hi = _mm256_extractf128_ps(vec1, 1);
    const __m128 vec1_m128 = _mm_add_ps(vec1_low, vec1_hi);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(vec0_m128), vec1_m128, 1);
}
inline float vec_reduce_add_ps(const VecPs vec) {
    const __m128 upper_128 = _mm256_extractf128_ps(vec, 1);
    const __m128 lower_128 = _mm256_castps256_ps128(vec);
    const __m128 sum_128 = _mm_add_ps(upper_128, lower_128);

    const __m128 upper_64 = _mm_movehl_ps(sum_128, sum_128);
    const __m128 sum_64 = _mm_add_ps(upper_64, sum_128);

    const __m128 upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
    const __m128 sum_32 = _mm_add_ss(upper_32, sum_64);

    return _mm_cvtss_f32(sum_32);
}

inline Vec256Epi vec256_hadd_epi32  (const Vec256Epi vec0, const Vec256Epi vec1) { return _mm256_hadd_epi32(vec0, vec1); }
inline Vec256Epi vec256_hadd_epi32x4(const VecEpi *vecs) {
    const Vec256Epi sum01 = _mm256_hadd_epi32(vecs[0], vecs[1]);
    const Vec256Epi sum23 = _mm256_hadd_epi32(vecs[2], vecs[3]);
    return _mm256_hadd_epi32(sum01, sum23);
}
inline Vec256Epi vec256_comb_epi32(const Vec256Epi vec0, const Vec256Epi vec1) {
    const __m128i vec0_low = _mm256_castsi256_si128(vec0);
    const __m128i vec0_hi = _mm256_extracti128_si256(vec0, 1);
    const __m128i vec0_m128 = _mm_add_epi32(vec0_low, vec0_hi);

    const __m128i vec1_low = _mm256_castsi256_si128(vec1);
    const __m128i vec1_hi = _mm256_extracti128_si256(vec1, 1);
    const __m128i vec1_m128 = _mm_add_epi32(vec1_low, vec1_hi);

    return _mm256_inserti128_si256(_mm256_castsi128_si256(vec0_m128), vec1_m128, 1);
}
#endif