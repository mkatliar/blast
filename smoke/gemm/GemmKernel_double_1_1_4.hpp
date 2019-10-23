#pragma once

#include <smoke/gemm/GemmKernel.hpp>

#include <smoke/simd/Hsum.hpp>
#include <smoke/Exception.hpp>

#include <immintrin.h>


namespace smoke
{
    template <bool TA, bool TB>
    class GemmKernel<double, 1, 1, 4, TA, TB>
    {
    public:
        static size_t constexpr alignment = 0x20;
        using Traits = GemmKernelTraits<GemmKernel>;


        GemmKernel()
        {            
        }


        explicit GemmKernel(double const * ptr, size_t spacing)
        {
            load(ptr, spacing);
        }


        void load(double const * ptr, size_t spacing)
        {
            v_[0] = _mm256_load_pd(ptr);
            v_[1] = _mm256_load_pd(ptr + 4);
            v_[2] = _mm256_load_pd(ptr + 8);
            v_[3] = _mm256_load_pd(ptr + 12);
        }


        void store(double * ptr, size_t spacing) const
        {
            _mm256_store_pd(ptr, v_[0]);
            _mm256_store_pd(ptr + 4, v_[1]);
            _mm256_store_pd(ptr + 8, v_[2]);
            _mm256_store_pd(ptr + 12, v_[3]);
        }


        void store(double * ptr, size_t spacing, size_t m, size_t n) const
        {
            if (m >= 4)
            {
                if (n > 0)
                    _mm256_store_pd(ptr, v_[0]);

                if (n > 1)
                    _mm256_store_pd(ptr + 4, v_[1]);

                if (n > 2)
                    _mm256_store_pd(ptr + 8, v_[2]);

                if (n > 3)
                    _mm256_store_pd(ptr + 12, v_[3]);
            }
            else if (m > 0)
            {
                __m256i const mask = _mm256_set_epi64x(
                    m > 3 ? 0x8000000000000000ULL : 0, 
                    m > 2 ? 0x8000000000000000ULL : 0,
                    m > 1 ? 0x8000000000000000ULL : 0,
                    m > 0 ? 0x8000000000000000ULL : 0); 

                if (n > 0)
                    _mm256_maskstore_pd(ptr, mask, v_[0]);

                if (n > 1)
                    _mm256_maskstore_pd(ptr + 4, mask, v_[1]);

                if (n > 2)
                    _mm256_maskstore_pd(ptr + 8, mask, v_[2]);

                if (n > 3)
                    _mm256_maskstore_pd(ptr + 12, mask, v_[3]);
            }
        }


        void operator()(double const * a, size_t sa, double const * b, size_t sb);


    private:
        __m256d v_[4];
    };


    template <>
    inline void GemmKernel<double, 1, 1, 4, false, true>::operator()(double const * a, size_t sa, double const * b, size_t sb)
    {
        __m256d const a_v0 = _mm256_load_pd(a);
        v_[0] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b), v_[0]);
        v_[1] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 1), v_[1]);
        v_[2] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 2), v_[2]);
        v_[3] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 3), v_[3]);
    }
}