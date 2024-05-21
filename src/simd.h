#pragma once
#include <cstdint>

#if defined(USE_SIMD)
#include <immintrin.h>
#endif

#if defined(USE_AVX512)
using VecEpi    = __m512i;
using Vec256Epi = __m256i;
using VecPs     = __m256;

VecEpi vec_set1_epi16 (const int16_t n);
VecEpi vec_loadu_epi  (const VecEpi *src);
void   vec_storeu_epi (VecEpi *dst, const VecEpi vec);
VecEpi vec_max_epi16  (const VecEpi vec0, const VecEpi vec1);
VecEpi vec_min_epi16  (const VecEpi vec0, const VecEpi vec1);
VecEpi vec_slli_epi16 (const VecEpi vec, const int shift);
VecEpi vec_mulhi_epu16(const VecEpi vec0, const VecEpi vec1);
VecEpi vec_madd_epi16 (const VecEpi vec0, const VecEpi vec1);
VecEpi vec_add_epi32  (const VecEpi vec0, const VecEpi vec1);

VecPs vec_set1_ps   (const float n);
VecPs vec_loadu_ps  (const float *src);
void  vec_storeu_ps (float *dst, const VecPs vec);
VecPs vec_cvtepi_ps (const Vec256Epi vec);
VecPs vec_add_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_mul_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_div_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_max_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_min_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_mul_add_ps(const VecPs vec0, const VecPs vec1, const VecPs vec2);
VecPs vec_hadd_ps   (const VecPs vec0, const VecPs vec1);
VecPs vec_hadd_psx4 (const VecPs *vecs);
VecPs vec_comb_ps(const VecPs vec0, const VecPs vec1);
float vec_reduce_add_ps(const VecPs vec);

Vec256Epi vec256_hadd_epi32  (const Vec256Epi vec0, const Vec256Epi vec1);
Vec256Epi vec256_hadd_epi32x4(const VecEpi *vecs);
Vec256Epi vec256_comb_epi32(const Vec256Epi vec0, const Vec256Epi vec1);

#elif defined(USE_AVX2)
using VecEpi    = __m256i;
using Vec256Epi = __m256i;
using VecPs     = __m256;

VecEpi vec_set1_epi16 (const int16_t n);
VecEpi vec_loadu_epi  (const VecEpi *src);
void   vec_storeu_epi (VecEpi *dst, const VecEpi vec);
VecEpi vec_max_epi16  (const VecEpi vec0, const VecEpi vec1);
VecEpi vec_min_epi16  (const VecEpi vec0, const VecEpi vec1);
VecEpi vec_slli_epi16 (const VecEpi vec, const int shift);
VecEpi vec_mulhi_epu16(const VecEpi vec0, const VecEpi vec1);
VecEpi vec_madd_epi16 (const VecEpi vec0, const VecEpi vec1);
VecEpi vec_add_epi32  (const VecEpi vec0, const VecEpi vec1);

VecPs vec_set1_ps   (const float n);
VecPs vec_loadu_ps  (const float *src);
void  vec_storeu_ps (float *dst, const VecPs vec);
VecPs vec_cvtepi_ps (const Vec256Epi vec);
VecPs vec_add_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_mul_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_div_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_max_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_min_ps    (const VecPs vec0, const VecPs vec1);
VecPs vec_mul_add_ps(const VecPs vec0, const VecPs vec1, const VecPs vec2);
VecPs vec_hadd_ps   (const VecPs vec0, const VecPs vec1);
VecPs vec_hadd_psx4 (const VecPs *vecs);
VecPs vec_comb_ps(const VecPs vec0, const VecPs vec1);
float vec_reduce_add_ps(const VecPs vec);

Vec256Epi vec256_hadd_epi32  (const Vec256Epi vec0, const Vec256Epi vec1);
Vec256Epi vec256_hadd_epi32x4(const VecEpi *vecs);
Vec256Epi vec256_comb_epi32(const Vec256Epi vec0, const Vec256Epi vec1);
#endif