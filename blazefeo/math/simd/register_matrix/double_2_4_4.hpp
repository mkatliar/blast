#pragma once

#include <blazefeo/math/simd/register_matrix/RegisterMatrix.hpp>
#include <blazefeo/Exception.hpp>

#include <blaze/util/Exception.h>

#include <immintrin.h>


namespace blazefeo
{
    template<>
    inline void RegisterMatrix<double, 2, 4, 4>::load(double beta, double const * ptr, size_t spacing, size_t m, size_t n)
    {
        if (n > 0)
        {
            v_[0][0] = beta * _mm256_load_pd(ptr);
            v_[1][0] = beta * _mm256_load_pd(ptr + spacing);
        }

        if (n > 1)
        {
            v_[0][1] = beta * _mm256_load_pd(ptr + 4);
            v_[1][1] = beta * _mm256_load_pd(ptr + spacing + 4);
        }

        if (n > 2)
        {
            v_[0][2] = beta * _mm256_load_pd(ptr + 8);
            v_[1][2] = beta * _mm256_load_pd(ptr + spacing + 8);
        }

        if (n > 3)
        {
            v_[0][3] = beta * _mm256_load_pd(ptr + 12);
            v_[1][3] = beta * _mm256_load_pd(ptr + spacing + 12);
        }
    }


#if 1
    template<>
    inline void RegisterMatrix<double, 2, 4, 4>::store(double * ptr, size_t spacing, size_t m, size_t n) const
    {
        for (size_t i = 0; i < 2; ++i)
        {
            if (m >= 4 * i + 4)
            {
                if (n > 0)
                    _mm256_store_pd(ptr + i * spacing, v_[i][0]);

                if (n > 1)
                    _mm256_store_pd(ptr + i * spacing + 4, v_[i][1]);

                if (n > 2)
                    _mm256_store_pd(ptr + i * spacing + 8, v_[i][2]);

                if (n > 3)
                    _mm256_store_pd(ptr + i * spacing + 12, v_[i][3]);
            }
            else if (m > 4 * i)
            {
                __m256i const mask = _mm256_set_epi64x(
                    m > 4 * i + 3 ? 0x8000000000000000ULL : 0, 
                    m > 4 * i + 2 ? 0x8000000000000000ULL : 0,
                    m > 4 * i + 1 ? 0x8000000000000000ULL : 0,
                    m > 4 * i + 0 ? 0x8000000000000000ULL : 0); 

                if (n > 0)
                    _mm256_maskstore_pd(ptr + i * spacing, mask, v_[i][0]);

                if (n > 1)
                    _mm256_maskstore_pd(ptr + i * spacing + 4, mask, v_[i][1]);

                if (n > 2)
                    _mm256_maskstore_pd(ptr + i * spacing + 8, mask, v_[i][2]);

                if (n > 3)
                    _mm256_maskstore_pd(ptr + i * spacing + 12, mask, v_[i][3]);
            }
        }
    }
#endif


    template <>
    template <>
    inline void RegisterMatrix<double, 2, 4, 4>::ger<false, true>(
        double alpha, double const * a, size_t sa, double const * b, size_t sb)
    {
        __m256d const a0 = alpha * _mm256_load_pd(a);
        __m256d const a4 = alpha * _mm256_load_pd(a + sa);                        
        
        __m256d bx = _mm256_broadcast_sd(b);
        v_[0][0] = _mm256_fmadd_pd(a0, bx, v_[0][0]);
        v_[1][0] = _mm256_fmadd_pd(a4, bx, v_[1][0]);

        bx = _mm256_broadcast_sd(b + 1);
        v_[0][1] = _mm256_fmadd_pd(a0, bx, v_[0][1]);
        v_[1][1] = _mm256_fmadd_pd(a4, bx, v_[1][1]);

        bx = _mm256_broadcast_sd(b + 2);
        v_[0][2] = _mm256_fmadd_pd(a0, bx, v_[0][2]);
        v_[1][2] = _mm256_fmadd_pd(a4, bx, v_[1][2]);

        bx = _mm256_broadcast_sd(b + 3);
        v_[0][3] = _mm256_fmadd_pd(a0, bx, v_[0][3]);
        v_[1][3] = _mm256_fmadd_pd(a4, bx, v_[1][3]);
    }


    template <>
    template <>
    inline void RegisterMatrix<double, 2, 4, 4>::ger<false, true>(
        double alpha, double const * a, size_t sa, double const * b, size_t sb, size_t m, size_t n)
    {
        __m256d const a0 = alpha * _mm256_load_pd(a);
        __m256d const a4 = alpha * _mm256_load_pd(a + sa);        
        __m256d bx;

        if (n > 0)
        {
            bx = _mm256_broadcast_sd(b);
            v_[0][0] = _mm256_fmadd_pd(a0, bx, v_[0][0]);
            v_[1][0] = _mm256_fmadd_pd(a4, bx, v_[1][0]);
        }

        if (n > 1)
        {
            bx = _mm256_broadcast_sd(b + 1);        
            v_[0][1] = _mm256_fmadd_pd(a0, bx, v_[0][1]);
            v_[1][1] = _mm256_fmadd_pd(a4, bx, v_[1][1]);
        }

        if (n > 2)
        {
            bx = _mm256_broadcast_sd(b + 2);
            v_[0][2] = _mm256_fmadd_pd(a0, bx, v_[0][2]);
            v_[1][2] = _mm256_fmadd_pd(a4, bx, v_[1][2]);
        }

        if (n > 3)
        {
            bx = _mm256_broadcast_sd(b + 3);
            v_[0][3] = _mm256_fmadd_pd(a0, bx, v_[0][3]);
            v_[1][3] = _mm256_fmadd_pd(a4, bx, v_[1][3]);
        }
    }
}