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
    };


    inline void gemm(GemmKernel<double, 3, 1, 4, true, false>, size_t K,
        double const * a, size_t sa, double const * b, size_t sb,
        double const * c, size_t sc, double * d, size_t sd)
    {
        size_t constexpr panel_size = 4;
        size_t constexpr block_element_count = panel_size * panel_size;
        
        if (K % panel_size)
            SMOKE_THROW_EXCEPTION(std::logic_error("k is not a multiple of panel size"));

        __m256d v00_ = _mm256_load_pd(c);
        __m256d v01_ = _mm256_load_pd(c + 4);
        __m256d v02_ = _mm256_load_pd(c + 8);
        __m256d v03_ = _mm256_load_pd(c + 12);
        __m256d v40_ = _mm256_load_pd(c + sc);
        __m256d v41_ = _mm256_load_pd(c + sc + 4);
        __m256d v42_ = _mm256_load_pd(c + sc + 8);
        __m256d v43_ = _mm256_load_pd(c + sc + 12);
        __m256d v80_ = _mm256_load_pd(c + 2 * sc);
        __m256d v81_ = _mm256_load_pd(c + 2 * sc + 4);
        __m256d v82_ = _mm256_load_pd(c + 2 * sc + 8);
        __m256d v83_ = _mm256_load_pd(c + 2 * sc + 12);

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
            
            v00_ = _mm256_add_pd(v00_, hsum(
                _mm256_mul_pd(a00, b00),
                _mm256_mul_pd(a01, b00),
                _mm256_mul_pd(a02, b00),
                _mm256_mul_pd(a03, b00)
            ));

            v01_ = _mm256_add_pd(v01_, hsum(
                _mm256_mul_pd(a00, b01),
                _mm256_mul_pd(a01, b01),
                _mm256_mul_pd(a02, b01),
                _mm256_mul_pd(a03, b01)
            ));

            v02_ = _mm256_add_pd(v02_, hsum(
                _mm256_mul_pd(a00, b02),
                _mm256_mul_pd(a01, b02),
                _mm256_mul_pd(a02, b02),
                _mm256_mul_pd(a03, b02)
            ));

            v03_ = _mm256_add_pd(v03_, hsum(
                _mm256_mul_pd(a00, b03),
                _mm256_mul_pd(a01, b03),
                _mm256_mul_pd(a02, b03),
                _mm256_mul_pd(a03, b03)
            ));

            v40_ = _mm256_add_pd(v40_, hsum(
                _mm256_mul_pd(a40, b00),
                _mm256_mul_pd(a41, b00),
                _mm256_mul_pd(a42, b00),
                _mm256_mul_pd(a43, b00)
            ));

            v41_ = _mm256_add_pd(v41_, hsum(
                _mm256_mul_pd(a40, b01),
                _mm256_mul_pd(a41, b01),
                _mm256_mul_pd(a42, b01),
                _mm256_mul_pd(a43, b01)
            ));

            v42_ = _mm256_add_pd(v42_, hsum(
                _mm256_mul_pd(a40, b02),
                _mm256_mul_pd(a41, b02),
                _mm256_mul_pd(a42, b02),
                _mm256_mul_pd(a43, b02)
            ));

            v43_ = _mm256_add_pd(v43_, hsum(
                _mm256_mul_pd(a40, b03),
                _mm256_mul_pd(a41, b03),
                _mm256_mul_pd(a42, b03),
                _mm256_mul_pd(a43, b03)
            ));
        }

        _mm256_store_pd(d, v00_);
        _mm256_store_pd(d + 4, v01_);
        _mm256_store_pd(d + 8, v02_);
        _mm256_store_pd(d + 12, v03_);
        _mm256_store_pd(d + sd, v40_);
        _mm256_store_pd(d + sd + 4, v41_);
        _mm256_store_pd(d + sd + 8, v42_);
        _mm256_store_pd(d + sd + 12, v43_);
        _mm256_store_pd(d + 2 * sd, v80_);
        _mm256_store_pd(d + 2 * sd + 4, v81_);
        _mm256_store_pd(d + 2 * sd + 8, v82_);
        _mm256_store_pd(d + 2 * sd + 12, v83_);
    }


    inline void gemm(GemmKernel<double, 3, 1, 4, false, false>, size_t K,
        double const * a, size_t sa, double const * b, size_t sb,
        double const * c, size_t sc, double * d, size_t sd)
    {
        size_t constexpr panel_size = 4;
        size_t constexpr block_element_count = panel_size * panel_size;

        if (K % panel_size)
            SMOKE_THROW_EXCEPTION(std::logic_error("k is not a multiple of panel size"));

        __m256d v00_ = _mm256_load_pd(c);
        __m256d v01_ = _mm256_load_pd(c + 4);
        __m256d v02_ = _mm256_load_pd(c + 8);
        __m256d v03_ = _mm256_load_pd(c + 12);
        __m256d v40_ = _mm256_load_pd(c + sc);
        __m256d v41_ = _mm256_load_pd(c + sc + 4);
        __m256d v42_ = _mm256_load_pd(c + sc + 8);
        __m256d v43_ = _mm256_load_pd(c + sc + 12);
        __m256d v80_ = _mm256_load_pd(c + 2 * sc);
        __m256d v81_ = _mm256_load_pd(c + 2 * sc + 4);
        __m256d v82_ = _mm256_load_pd(c + 2 * sc + 8);
        __m256d v83_ = _mm256_load_pd(c + 2 * sc + 12);

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
            v00_ = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v00_);
            v00_ = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v00_);
            v00_ = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v00_);
            v00_ = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v00_);
            v40_ = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v40_);
            v40_ = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v40_);
            v40_ = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v40_);
            v40_ = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v40_);

            bb = _mm256_load_pd(b + 4);
            v01_ = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v01_);
            v01_ = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v01_);
            v01_ = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v01_);
            v01_ = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v01_);
            v41_ = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v41_);
            v41_ = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v41_);
            v41_ = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v41_);
            v41_ = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v41_);

            bb = _mm256_load_pd(b + 8);
            v02_ = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v02_);
            v02_ = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v02_);
            v02_ = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v02_);
            v02_ = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v02_);
            v42_ = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v42_);
            v42_ = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v42_);
            v42_ = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v42_);
            v42_ = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v42_);

            bb = _mm256_load_pd(b + 12);
            v03_ = _mm256_fmadd_pd(a00, _mm256_permute4x64_pd(bb, 0b00000000), v03_);
            v03_ = _mm256_fmadd_pd(a01, _mm256_permute4x64_pd(bb, 0b01010101), v03_);
            v03_ = _mm256_fmadd_pd(a02, _mm256_permute4x64_pd(bb, 0b10101010), v03_);
            v03_ = _mm256_fmadd_pd(a03, _mm256_permute4x64_pd(bb, 0b11111111), v03_);
            v43_ = _mm256_fmadd_pd(a40, _mm256_permute4x64_pd(bb, 0b00000000), v43_);
            v43_ = _mm256_fmadd_pd(a41, _mm256_permute4x64_pd(bb, 0b01010101), v43_);
            v43_ = _mm256_fmadd_pd(a42, _mm256_permute4x64_pd(bb, 0b10101010), v43_);
            v43_ = _mm256_fmadd_pd(a43, _mm256_permute4x64_pd(bb, 0b11111111), v43_);
        }

        _mm256_store_pd(d, v00_);
        _mm256_store_pd(d + 4, v01_);
        _mm256_store_pd(d + 8, v02_);
        _mm256_store_pd(d + 12, v03_);
        _mm256_store_pd(d + sd, v40_);
        _mm256_store_pd(d + sd + 4, v41_);
        _mm256_store_pd(d + sd + 8, v42_);
        _mm256_store_pd(d + sd + 12, v43_);
        _mm256_store_pd(d + 2 * sd, v80_);
        _mm256_store_pd(d + 2 * sd + 4, v81_);
        _mm256_store_pd(d + 2 * sd + 8, v82_);
        _mm256_store_pd(d + 2 * sd + 12, v83_);
    }


    inline void gemm(GemmKernel<double, 3, 1, 4, false, true>, size_t K,
        double const * a, size_t sa, double const * b, size_t sb,
        double const * c, size_t sc, double * d, size_t sd)
    {
        size_t constexpr panel_size = 4;
        if (K % panel_size)
            SMOKE_THROW_EXCEPTION(std::logic_error("k is not a multiple of panel size"));

        __m256d v00_ = _mm256_load_pd(c);
        __m256d v01_ = _mm256_load_pd(c + 4);
        __m256d v02_ = _mm256_load_pd(c + 8);
        __m256d v03_ = _mm256_load_pd(c + 12);
        __m256d v40_ = _mm256_load_pd(c + sc);
        __m256d v41_ = _mm256_load_pd(c + sc + 4);
        __m256d v42_ = _mm256_load_pd(c + sc + 8);
        __m256d v43_ = _mm256_load_pd(c + sc + 12);
        __m256d v80_ = _mm256_load_pd(c + 2 * sc);
        __m256d v81_ = _mm256_load_pd(c + 2 * sc + 4);
        __m256d v82_ = _mm256_load_pd(c + 2 * sc + 8);
        __m256d v83_ = _mm256_load_pd(c + 2 * sc + 12);

        for (size_t k = 0; k < K; ++k, a += panel_size, b += panel_size)
        {
            __m256d const a0 = _mm256_load_pd(a);
            __m256d const a4 = _mm256_load_pd(a + sa);
            __m256d const a8 = _mm256_load_pd(a + 2 * sa);
            
            __m256d bx = _mm256_broadcast_sd(b);
            v00_ = _mm256_fmadd_pd(a0, bx, v00_);
            v40_ = _mm256_fmadd_pd(a4, bx, v40_);
            v80_ = _mm256_fmadd_pd(a8, bx, v80_);
            
            bx = _mm256_broadcast_sd(b + 1);
            v01_ = _mm256_fmadd_pd(a0, bx, v01_);
            v41_ = _mm256_fmadd_pd(a4, bx, v41_);
            v81_ = _mm256_fmadd_pd(a8, bx, v81_);
            
            bx = _mm256_broadcast_sd(b + 2);
            v02_ = _mm256_fmadd_pd(a0, bx, v02_);
            v42_ = _mm256_fmadd_pd(a4, bx, v42_);
            v82_ = _mm256_fmadd_pd(a8, bx, v82_);

            bx = _mm256_broadcast_sd(b + 3);
            v03_ = _mm256_fmadd_pd(a0, bx, v03_);
            v43_ = _mm256_fmadd_pd(a4, bx, v43_);
            v83_ = _mm256_fmadd_pd(a8, bx, v83_);
        }        

        _mm256_store_pd(d, v00_);
        _mm256_store_pd(d + 4, v01_);
        _mm256_store_pd(d + 8, v02_);
        _mm256_store_pd(d + 12, v03_);
        _mm256_store_pd(d + sd, v40_);
        _mm256_store_pd(d + sd + 4, v41_);
        _mm256_store_pd(d + sd + 8, v42_);
        _mm256_store_pd(d + sd + 12, v43_);
        _mm256_store_pd(d + 2 * sd, v80_);
        _mm256_store_pd(d + 2 * sd + 4, v81_);
        _mm256_store_pd(d + 2 * sd + 8, v82_);
        _mm256_store_pd(d + 2 * sd + 12, v83_);
    }
}