#pragma once

#include <blazefeo/math/simd/register_matrix/RegisterMatrix.hpp>
#include <blazefeo/Exception.hpp>

#include <cmath>


namespace blazefeo
{
    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::load(double beta, double const * ptr, size_t spacing, size_t m, size_t n)
    {
        if (n > 0)
            v_[0][0] = beta * _mm256_load_pd(ptr);

        if (n > 1)
            v_[0][1] = beta * _mm256_load_pd(ptr + 4);

        if (n > 2)
            v_[0][2] = beta * _mm256_load_pd(ptr + 8);

        if (n > 3)
            v_[0][3] = beta * _mm256_load_pd(ptr + 12);
    }


#if 1
    /// Magically, this function specialization is slightly faster than the default implementation of RegisterMatrix<>::store.
    template <>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 1, 4, 4>::store(double * ptr, size_t spacing, size_t m, size_t n) const
    {
        if (m >= 4)
        {
            if (n > 0)
                _mm256_store_pd(ptr, v_[0][0]);

            if (n > 1)
                _mm256_store_pd(ptr + 4, v_[0][1]);

            if (n > 2)
                _mm256_store_pd(ptr + 8, v_[0][2]);

            if (n > 3)
                _mm256_store_pd(ptr + 12, v_[0][3]);
        }
        else if (m > 0)
        {
            // Magically, the code below is significantly faster than this:
            // __m256i const mask = _mm256_cmpgt_epi64(_mm256_set_epi64x(m, m, m, m), _mm256_set_epi64x(3, 2, 1, 0)); 
            __m256i const mask = _mm256_set_epi64x(
                m > 3 ? 0x8000000000000000ULL : 0, 
                m > 2 ? 0x8000000000000000ULL : 0,
                m > 1 ? 0x8000000000000000ULL : 0,
                m > 0 ? 0x8000000000000000ULL : 0);

            if (n > 0)
                _mm256_maskstore_pd(ptr, mask, v_[0][0]);

            if (n > 1)
                _mm256_maskstore_pd(ptr + 4, mask, v_[0][1]);

            if (n > 2)
                _mm256_maskstore_pd(ptr + 8, mask, v_[0][2]);

            if (n > 3)
                _mm256_maskstore_pd(ptr + 12, mask, v_[0][3]);
        }
    }
#endif
}