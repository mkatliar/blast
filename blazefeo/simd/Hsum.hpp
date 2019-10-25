#pragma once

#include <immintrin.h>


namespace blazefeo
{
    /// dst[127:64] = sum(v1)
    /// dst[63:0] = sum(v0)
    inline __m128d hsum(__m256d v0, __m256d v1)
    {
        __m256d v01 = _mm256_hadd_pd(v0, v1);
        return _mm_add_pd(_mm256_extractf128_pd(v01, 1), _mm256_castpd256_pd128(v01));
    }


    /// dst[255:192] = sum(v3)
    /// dst[191:128] = sum(v2)
    /// dst[127:64] = sum(v1)
    /// dst[63:0] = sum(v0)
    inline __m256d hsum(__m256d v0, __m256d v1, __m256d v2, __m256d v3)
    {
        __m128d v01 = hsum(v0, v1);
        __m128d v23 = hsum(v2, v3);
        return _mm256_insertf128_pd(_mm256_castpd128_pd256(v01), v23, 1);
    }
}