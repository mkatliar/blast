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
            if (m == 4)
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
            else
            {
                __m256i const mask = _mm256_set_epi64x(
                    m > 3 ? 0x8000000000000000ULL : 0, 
                    m > 2 ? 0x8000000000000000ULL : 0,
                    m > 1 ? 0x8000000000000000ULL : 0,
                    m > 0 ? 0x8000000000000000ULL : 0); 

                // switch (n)
                // {
                //     case 4:
                //         _mm256_maskstore_pd(ptr + 12, mask, v_[3]);
                //     case 3:
                //         _mm256_maskstore_pd(ptr + 8, mask, v_[2]);
                //     case 2:
                //         _mm256_maskstore_pd(ptr + 4, mask, v_[1]);
                //     case 1:
                //         _mm256_maskstore_pd(ptr, mask, v_[0]);
                // }                    

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


        void operator()(size_t K, double const * a, size_t sa, double const * b, size_t sb);


    private:
        __m256d v_[4];
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
            
            v_[0] = _mm256_add_pd(v_[0], hsum(
                _mm256_mul_pd(a_v0, b_v0),
                _mm256_mul_pd(a_v1, b_v0),
                _mm256_mul_pd(a_v2, b_v0),
                _mm256_mul_pd(a_v3, b_v0)
            ));

            v_[1] = _mm256_add_pd(v_[1], hsum(
                _mm256_mul_pd(a_v0, b_v1),
                _mm256_mul_pd(a_v1, b_v1),
                _mm256_mul_pd(a_v2, b_v1),
                _mm256_mul_pd(a_v3, b_v1)
            ));

            v_[2] = _mm256_add_pd(v_[2], hsum(
                _mm256_mul_pd(a_v0, b_v2),
                _mm256_mul_pd(a_v1, b_v2),
                _mm256_mul_pd(a_v2, b_v2),
                _mm256_mul_pd(a_v3, b_v2)
            ));

            v_[3] = _mm256_add_pd(v_[3], hsum(
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
            v_[0] = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v_[0]);
            v_[0] = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v_[0]);
            v_[0] = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v_[0]);
            v_[0] = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v_[0]);

            bb = _mm256_load_pd(b + 4);
            v_[1] = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v_[1]);
            v_[1] = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v_[1]);
            v_[1] = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v_[1]);
            v_[1] = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v_[1]);

            bb = _mm256_load_pd(b + 8);
            v_[2] = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v_[2]);
            v_[2] = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v_[2]);
            v_[2] = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v_[2]);
            v_[2] = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v_[2]);

            bb = _mm256_load_pd(b + 12);
            v_[3] = _mm256_fmadd_pd(a_v0, _mm256_permute4x64_pd(bb, 0b00000000), v_[3]);
            v_[3] = _mm256_fmadd_pd(a_v1, _mm256_permute4x64_pd(bb, 0b01010101), v_[3]);
            v_[3] = _mm256_fmadd_pd(a_v2, _mm256_permute4x64_pd(bb, 0b10101010), v_[3]);
            v_[3] = _mm256_fmadd_pd(a_v3, _mm256_permute4x64_pd(bb, 0b11111111), v_[3]);
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
            v_[0] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b), v_[0]);
            v_[1] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 1), v_[1]);
            v_[2] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 2), v_[2]);
            v_[3] = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 3), v_[3]);
        }
    }
}