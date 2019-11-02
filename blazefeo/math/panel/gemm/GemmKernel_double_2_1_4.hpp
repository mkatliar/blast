#pragma once

#include <blazefeo/math/panel/gemm/GemmKernel.hpp>

#include <blazefeo/simd/Hsum.hpp>
#include <blazefeo/Exception.hpp>

#include <immintrin.h>


namespace blazefeo
{
    template <>
    class GemmKernel<double, 2, 1, 4>
    {
    public:
        GemmKernel()
        {            
        }


        explicit GemmKernel(double const * ptr, size_t spacing)
        {
            load(ptr, spacing);
        }


        void load(double const * ptr, size_t spacing)
        {
            v_[0][0] = _mm256_load_pd(ptr);
            v_[0][1] = _mm256_load_pd(ptr + 4);
            v_[0][2] = _mm256_load_pd(ptr + 8);
            v_[0][3] = _mm256_load_pd(ptr + 12);
            v_[1][0] = _mm256_load_pd(ptr + spacing);
            v_[1][1] = _mm256_load_pd(ptr + spacing + 4);
            v_[1][2] = _mm256_load_pd(ptr + spacing + 8);
            v_[1][3] = _mm256_load_pd(ptr + spacing + 12);
        }


        void load(double const * ptr, size_t spacing, size_t m, size_t n)
        {
            if (n > 0)
            {
                v_[0][0] = _mm256_load_pd(ptr);
                v_[1][0] = _mm256_load_pd(ptr + spacing);
            }

            if (n > 1)
            {
                v_[0][1] = _mm256_load_pd(ptr + 4);
                v_[1][1] = _mm256_load_pd(ptr + spacing + 4);
            }

            if (n > 2)
            {
                v_[0][2] = _mm256_load_pd(ptr + 8);
                v_[1][2] = _mm256_load_pd(ptr + spacing + 8);
            }

            if (n > 3)
            {
                v_[0][3] = _mm256_load_pd(ptr + 12);
                v_[1][3] = _mm256_load_pd(ptr + spacing + 12);
            }
        }


        void store(double * ptr, size_t spacing) const
        {
            _mm256_store_pd(ptr, v_[0][0]);
            _mm256_store_pd(ptr + 4, v_[0][1]);
            _mm256_store_pd(ptr + 8, v_[0][2]);
            _mm256_store_pd(ptr + 12, v_[0][3]);
            _mm256_store_pd(ptr + spacing, v_[1][0]);
            _mm256_store_pd(ptr + spacing + 4, v_[1][1]);
            _mm256_store_pd(ptr + spacing + 8, v_[1][2]);
            _mm256_store_pd(ptr + spacing + 12, v_[1][3]);
        }


        void store(double * ptr, size_t spacing, size_t m, size_t n) const
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


        template <bool TA, bool TB>
        void gemm(double const * a, size_t sa, double const * b, size_t sb);


        template <bool TA, bool TB>
        void gemm(double const * a, size_t sa, double const * b, size_t sb, size_t m, size_t n);


    private:
        __m256d v_[2][4];
    };


    template <>
    inline void GemmKernel<double, 2, 1, 4>::gemm<false, true>(
        double const * a, size_t sa, double const * b, size_t sb)
    {
        __m256d const a0 = _mm256_load_pd(a);
        __m256d const a4 = _mm256_load_pd(a + sa);                        
        
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
    inline void GemmKernel<double, 2, 1, 4>::gemm<false, true>(
        double const * a, size_t sa, double const * b, size_t sb, size_t m, size_t n)
    {
        __m256d const a0 = _mm256_load_pd(a);
        __m256d const a4 = _mm256_load_pd(a + sa);        
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