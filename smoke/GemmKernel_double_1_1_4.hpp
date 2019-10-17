#pragma once

#include <smoke/GemmKernel.hpp>

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
            v0_ = _mm256_load_pd(ptr);
            v1_ = _mm256_load_pd(ptr + 4);
            v2_ = _mm256_load_pd(ptr + 8);
            v3_ = _mm256_load_pd(ptr + 12);
        }


        void store(double * ptr, size_t spacing) const
        {
            _mm256_store_pd(ptr, v0_);
            _mm256_store_pd(ptr + 4, v1_);
            _mm256_store_pd(ptr + 8, v2_);
            _mm256_store_pd(ptr + 12, v3_);
        }


        void operator()(size_t K, double const * a, size_t sa, double const * b, size_t sb);


    private:
        __m256d v0_;
        __m256d v1_;
        __m256d v2_;
        __m256d v3_;
    };


    template <>
    inline void GemmKernel<double, 1, 1, 4, true, false>::operator()(
        size_t K, double const * a, size_t sa, double const * b, size_t sb)
    {
        size_t constexpr panel_size = 4;
        
        for (size_t k = 0; k + panel_size <= K; k += panel_size, a += sa, b += sb)
        {
            __m256d a_v0 = _mm256_load_pd(a);
            __m256d a_v1 = _mm256_load_pd(a + 4);
            __m256d a_v2 = _mm256_load_pd(a + 8);
            __m256d a_v3 = _mm256_load_pd(a + 12);

            __m256d b_v0 = _mm256_load_pd(b);
            __m256d b_v1 = _mm256_load_pd(b + 4);
            __m256d b_v2 = _mm256_load_pd(b + 8);
            __m256d b_v3 = _mm256_load_pd(b + 12);
            
            v0_ = _mm256_add_pd(v0_, hsum(
                _mm256_mul_pd(a_v0, b_v0),
                _mm256_mul_pd(a_v1, b_v0),
                _mm256_mul_pd(a_v2, b_v0),
                _mm256_mul_pd(a_v3, b_v0)
            ));

            v1_ = _mm256_add_pd(v1_, hsum(
                _mm256_mul_pd(a_v0, b_v1),
                _mm256_mul_pd(a_v1, b_v1),
                _mm256_mul_pd(a_v2, b_v1),
                _mm256_mul_pd(a_v3, b_v1)
            ));

            v2_ = _mm256_add_pd(v2_, hsum(
                _mm256_mul_pd(a_v0, b_v2),
                _mm256_mul_pd(a_v1, b_v2),
                _mm256_mul_pd(a_v2, b_v2),
                _mm256_mul_pd(a_v3, b_v2)
            ));

            v3_ = _mm256_add_pd(v3_, hsum(
                _mm256_mul_pd(a_v0, b_v3),
                _mm256_mul_pd(a_v1, b_v3),
                _mm256_mul_pd(a_v2, b_v3),
                _mm256_mul_pd(a_v3, b_v3)
            ));
        }
    }


    template <>
    inline void GemmKernel<double, 1, 1, 4, false, false>::operator()(
        size_t K, double const * a, size_t sa, double const * b, size_t sb)
    {
        size_t constexpr panel_size = 4;
        size_t constexpr block_element_count = panel_size * panel_size;

        for (size_t k = 0; k + panel_size <= K; k += panel_size, a += block_element_count, b += sb)
        {
            __m256d a_v0 = _mm256_load_pd(a);
            __m256d a_v1 = _mm256_load_pd(a + 4);
            __m256d a_v2 = _mm256_load_pd(a + 8);
            __m256d a_v3 = _mm256_load_pd(a + 12);

            __m256d bb = _mm256_load_pd(b);
            v0_ = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v0_);
            v0_ = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v0_);
            v0_ = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v0_);
            v0_ = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v0_);

            bb = _mm256_load_pd(b + 4);
            v1_ = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v1_);
            v1_ = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v1_);
            v1_ = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v1_);
            v1_ = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v1_);

            bb = _mm256_load_pd(b + 8);
            v2_ = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v2_);
            v2_ = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v2_);
            v2_ = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v2_);
            v2_ = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v2_);

            bb = _mm256_load_pd(b + 12);
            v3_ = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v3_);
            v3_ = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v3_);
            v3_ = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v3_);
            v3_ = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v3_);
        }
    }
    
    
    template <>
    inline void GemmKernel<double, 1, 1, 4, false, true>::operator()(
        size_t K, double const * a, size_t sa, double const * b, size_t sb)
    {
        size_t constexpr panel_size = 4;
        
        for (size_t k = 0; k < K; ++k, a += panel_size, b += panel_size)
        {
            __m256d const a_v0 = _mm256_load_pd(a);
            v0_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b), v0_);
            v1_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 1), v1_);
            v2_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 2), v2_);
            v3_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 3), v3_);
        }
    }
}