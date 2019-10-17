#pragma once

#include <smoke/GemmKernel.hpp>

#include <smoke/simd/Hsum.hpp>
#include <smoke/Exception.hpp>

#include <immintrin.h>


namespace smoke
{
    template <bool TA, bool TB>
    class GemmKernel<double, 3, 1, 4, TA, TB>
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
            v_[0][0] = _mm256_load_pd(ptr);
            v_[0][1] = _mm256_load_pd(ptr + 4);
            v_[0][2] = _mm256_load_pd(ptr + 8);
            v_[0][3] = _mm256_load_pd(ptr + 12);
            v_[1][0] = _mm256_load_pd(ptr + spacing);
            v_[1][1] = _mm256_load_pd(ptr + spacing + 4);
            v_[1][2] = _mm256_load_pd(ptr + spacing + 8);
            v_[1][3] = _mm256_load_pd(ptr + spacing + 12);
            v_[2][0] = _mm256_load_pd(ptr + 2 * spacing);
            v_[2][1] = _mm256_load_pd(ptr + 2 * spacing + 4);
            v_[2][2] = _mm256_load_pd(ptr + 2 * spacing + 8);
            v_[2][3] = _mm256_load_pd(ptr + 2 * spacing + 12);
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
            _mm256_store_pd(ptr + 2 * spacing, v_[2][0]);
            _mm256_store_pd(ptr + 2 * spacing + 4, v_[2][1]);
            _mm256_store_pd(ptr + 2 * spacing + 8, v_[2][2]);
            _mm256_store_pd(ptr + 2 * spacing + 12, v_[2][3]);
        }


        void store(double * ptr, size_t spacing, size_t m, size_t n) const
        {
            for (size_t i = 0; i < 3; ++i)
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


        void operator()(size_t K, double const * a, size_t sa, double const * b, size_t sb);


    private:
        __m256d v_[3][4];
    };


    template <>
    inline void GemmKernel<double, 3, 1, 4, true, false>::operator()(
        size_t K, double const * a, size_t sa, double const * b, size_t sb)
    {
        size_t constexpr panel_size = 4;
        size_t constexpr block_element_count = panel_size * panel_size;
        
        for (size_t k = 0; k + panel_size <= K; k += panel_size, a += sa, b += sb)
        {
            __m256d a00 = _mm256_load_pd(a);
            __m256d a01 = _mm256_load_pd(a + 4);
            __m256d a02 = _mm256_load_pd(a + 8);
            __m256d a03 = _mm256_load_pd(a + 12);

            __m256d a40 = _mm256_load_pd(a + block_element_count);
            __m256d a41 = _mm256_load_pd(a + block_element_count + 4);
            __m256d a42 = _mm256_load_pd(a + block_element_count + 8);
            __m256d a43 = _mm256_load_pd(a + block_element_count + 12);

            __m256d b00 = _mm256_load_pd(b);
            __m256d b01 = _mm256_load_pd(b + 4);
            __m256d b02 = _mm256_load_pd(b + 8);
            __m256d b03 = _mm256_load_pd(b + 12);
            
            v_[0][0] = _mm256_add_pd(v_[0][0], hsum(
                _mm256_mul_pd(a00, b00),
                _mm256_mul_pd(a01, b00),
                _mm256_mul_pd(a02, b00),
                _mm256_mul_pd(a03, b00)
            ));

            v_[0][1] = _mm256_add_pd(v_[0][1], hsum(
                _mm256_mul_pd(a00, b01),
                _mm256_mul_pd(a01, b01),
                _mm256_mul_pd(a02, b01),
                _mm256_mul_pd(a03, b01)
            ));

            v_[0][2] = _mm256_add_pd(v_[0][2], hsum(
                _mm256_mul_pd(a00, b02),
                _mm256_mul_pd(a01, b02),
                _mm256_mul_pd(a02, b02),
                _mm256_mul_pd(a03, b02)
            ));

            v_[0][3] = _mm256_add_pd(v_[0][3], hsum(
                _mm256_mul_pd(a00, b03),
                _mm256_mul_pd(a01, b03),
                _mm256_mul_pd(a02, b03),
                _mm256_mul_pd(a03, b03)
            ));

            v_[1][0] = _mm256_add_pd(v_[1][0], hsum(
                _mm256_mul_pd(a40, b00),
                _mm256_mul_pd(a41, b00),
                _mm256_mul_pd(a42, b00),
                _mm256_mul_pd(a43, b00)
            ));

            v_[1][1] = _mm256_add_pd(v_[1][1], hsum(
                _mm256_mul_pd(a40, b01),
                _mm256_mul_pd(a41, b01),
                _mm256_mul_pd(a42, b01),
                _mm256_mul_pd(a43, b01)
            ));

            v_[1][2] = _mm256_add_pd(v_[1][2], hsum(
                _mm256_mul_pd(a40, b02),
                _mm256_mul_pd(a41, b02),
                _mm256_mul_pd(a42, b02),
                _mm256_mul_pd(a43, b02)
            ));

            v_[1][3] = _mm256_add_pd(v_[1][3], hsum(
                _mm256_mul_pd(a40, b03),
                _mm256_mul_pd(a41, b03),
                _mm256_mul_pd(a42, b03),
                _mm256_mul_pd(a43, b03)
            ));
        }
    }


    template <>
    inline void GemmKernel<double, 3, 1, 4, false, false>::operator()(
        size_t K, double const * a, size_t sa, double const * b, size_t sb)
    {
        size_t constexpr panel_size = 4;
        size_t constexpr block_element_count = panel_size * panel_size;

        for (size_t k = 0; k + panel_size <= K; k += panel_size, a += block_element_count, b += sb)
        {
            __m256d a00 = _mm256_load_pd(a);
            __m256d a01 = _mm256_load_pd(a + 4);
            __m256d a02 = _mm256_load_pd(a + 8);
            __m256d a03 = _mm256_load_pd(a + 12);

            __m256d a40 = _mm256_load_pd(a + sa);
            __m256d a41 = _mm256_load_pd(a + sa + 4);
            __m256d a42 = _mm256_load_pd(a + sa + 8);
            __m256d a43 = _mm256_load_pd(a + sa + 12);

            __m256d bb = _mm256_load_pd(b);
            v_[0][0] = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v_[0][0]);
            v_[0][0] = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v_[0][0]);
            v_[0][0] = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v_[0][0]);
            v_[0][0] = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v_[0][0]);
            v_[1][0] = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v_[1][0]);
            v_[1][0] = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v_[1][0]);
            v_[1][0] = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v_[1][0]);
            v_[1][0] = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v_[1][0]);

            bb = _mm256_load_pd(b + 4);
            v_[0][1] = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v_[0][1]);
            v_[0][1] = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v_[0][1]);
            v_[0][1] = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v_[0][1]);
            v_[0][1] = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v_[0][1]);
            v_[1][1] = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v_[1][1]);
            v_[1][1] = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v_[1][1]);
            v_[1][1] = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v_[1][1]);
            v_[1][1] = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v_[1][1]);

            bb = _mm256_load_pd(b + 8);
            v_[0][2] = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v_[0][2]);
            v_[0][2] = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v_[0][2]);
            v_[0][2] = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v_[0][2]);
            v_[0][2] = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v_[0][2]);
            v_[1][2] = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v_[1][2]);
            v_[1][2] = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v_[1][2]);
            v_[1][2] = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v_[1][2]);
            v_[1][2] = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v_[1][2]);

            bb = _mm256_load_pd(b + 12);
            v_[0][3] = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v_[0][3]);
            v_[0][3] = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v_[0][3]);
            v_[0][3] = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v_[0][3]);
            v_[0][3] = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v_[0][3]);
            v_[1][3] = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v_[1][3]);
            v_[1][3] = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v_[1][3]);
            v_[1][3] = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v_[1][3]);
            v_[1][3] = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v_[1][3]);
        }
    }


    template <>
    inline void GemmKernel<double, 3, 1, 4, false, true>::operator()(
        size_t K, double const * a, size_t sa, double const * b, size_t sb)
    {
        size_t constexpr panel_size = 4;
        
        for (size_t k = 0; k < K; ++k, a += panel_size, b += panel_size)
        {
            __m256d const a0 = _mm256_load_pd(a);
            __m256d const a4 = _mm256_load_pd(a + sa);
            __m256d const a8 = _mm256_load_pd(a + 2 * sa);
            
            __m256d bx = _mm256_broadcast_sd(b);
            v_[0][0] = _mm256_fmadd_pd(a0, bx, v_[0][0]);
            v_[1][0] = _mm256_fmadd_pd(a4, bx, v_[1][0]);
            v_[2][0] = _mm256_fmadd_pd(a8, bx, v_[2][0]);
            
            bx = _mm256_broadcast_sd(b + 1);
            v_[0][1] = _mm256_fmadd_pd(a0, bx, v_[0][1]);
            v_[1][1] = _mm256_fmadd_pd(a4, bx, v_[1][1]);
            v_[2][1] = _mm256_fmadd_pd(a8, bx, v_[2][1]);
            
            bx = _mm256_broadcast_sd(b + 2);
            v_[0][2] = _mm256_fmadd_pd(a0, bx, v_[0][2]);
            v_[1][2] = _mm256_fmadd_pd(a4, bx, v_[1][2]);
            v_[2][2] = _mm256_fmadd_pd(a8, bx, v_[2][2]);

            bx = _mm256_broadcast_sd(b + 3);
            v_[0][3] = _mm256_fmadd_pd(a0, bx, v_[0][3]);
            v_[1][3] = _mm256_fmadd_pd(a4, bx, v_[1][3]);
            v_[2][3] = _mm256_fmadd_pd(a8, bx, v_[2][3]);
        }
    }
}