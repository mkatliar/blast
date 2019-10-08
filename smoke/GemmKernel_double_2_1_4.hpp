#pragma once

#include <smoke/GemmKernel.hpp>

#include <smoke/simd/Hsum.hpp>
#include <smoke/Exception.hpp>

#include <immintrin.h>


namespace smoke
{
    template <>
    class GemmKernel<double, 2, 1, 4>
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
            v00_ = _mm256_load_pd(ptr);
            v01_ = _mm256_load_pd(ptr + 4);
            v02_ = _mm256_load_pd(ptr + 8);
            v03_ = _mm256_load_pd(ptr + 12);
            v40_ = _mm256_load_pd(ptr + spacing);
            v41_ = _mm256_load_pd(ptr + spacing + 4);
            v42_ = _mm256_load_pd(ptr + spacing + 8);
            v43_ = _mm256_load_pd(ptr + spacing + 12);
        }


        void store(double * ptr, size_t spacing) const
        {
            _mm256_store_pd(ptr, v00_);
            _mm256_store_pd(ptr + 4, v01_);
            _mm256_store_pd(ptr + 8, v02_);
            _mm256_store_pd(ptr + 12, v03_);
            _mm256_store_pd(ptr + spacing, v40_);
            _mm256_store_pd(ptr + spacing + 4, v41_);
            _mm256_store_pd(ptr + spacing + 8, v42_);
            _mm256_store_pd(ptr + spacing + 12, v43_);
        }


        void operator()(size_t K, double const * a, size_t sa, bool ta, double const * b, size_t sb, bool tb)
        {
            if (K % Traits::blockSize)
                SMOKE_THROW_EXCEPTION(std::logic_error("k is not a multiple of block size"));

            if (ta && !tb)
            {
                for (size_t k = 0; k + Traits::blockSize <= K; k += Traits::blockSize, a += sa, b += sb)
                {
                    __m256d a00 = _mm256_load_pd(a);
                    __m256d a01 = _mm256_load_pd(a + 4);
                    __m256d a02 = _mm256_load_pd(a + 8);
                    __m256d a03 = _mm256_load_pd(a + 12);

                    __m256d a40 = _mm256_load_pd(a + Traits::blockElementCount);
                    __m256d a41 = _mm256_load_pd(a + Traits::blockElementCount + 4);
                    __m256d a42 = _mm256_load_pd(a + Traits::blockElementCount + 8);
                    __m256d a43 = _mm256_load_pd(a + Traits::blockElementCount + 12);

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
            }
            else if (!ta && !tb)
            {
                for (size_t k = 0; k + Traits::blockSize <= K; k += Traits::blockSize, a += Traits::blockElementCount, b += sb)
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
            }
            else if (!ta && tb)
            {
                for (size_t k = 0; k + Traits::blockSize <= K; k += Traits::blockSize, a += Traits::blockElementCount, b += Traits::blockElementCount)
                {
                    {
                        __m256d const a00 = _mm256_load_pd(a);
                        __m256d const a40 = _mm256_load_pd(a + sa);
                        v00_ = _mm256_fmadd_pd(a00, _mm256_broadcast_sd(b), v00_);
                        v40_ = _mm256_fmadd_pd(a40, _mm256_broadcast_sd(b), v40_);
                        v01_ = _mm256_fmadd_pd(a00, _mm256_broadcast_sd(b + 1), v01_);
                        v41_ = _mm256_fmadd_pd(a40, _mm256_broadcast_sd(b + 1), v41_);
                        v02_ = _mm256_fmadd_pd(a00, _mm256_broadcast_sd(b + 2), v02_);
                        v42_ = _mm256_fmadd_pd(a40, _mm256_broadcast_sd(b + 2), v42_);
                        v03_ = _mm256_fmadd_pd(a00, _mm256_broadcast_sd(b + 3), v03_);
                        v43_ = _mm256_fmadd_pd(a40, _mm256_broadcast_sd(b + 3), v43_);
                    }

                    {
                        __m256d const a01 = _mm256_load_pd(a + 4);
                        __m256d const a41 = _mm256_load_pd(a + sa + 4);
                        v00_ = _mm256_fmadd_pd(a01, _mm256_broadcast_sd(b + 4), v00_);
                        v40_ = _mm256_fmadd_pd(a41, _mm256_broadcast_sd(b + 4), v40_);
                        v01_ = _mm256_fmadd_pd(a01, _mm256_broadcast_sd(b + 5), v01_);
                        v41_ = _mm256_fmadd_pd(a41, _mm256_broadcast_sd(b + 5), v41_);
                        v02_ = _mm256_fmadd_pd(a01, _mm256_broadcast_sd(b + 6), v02_);
                        v42_ = _mm256_fmadd_pd(a41, _mm256_broadcast_sd(b + 6), v42_);
                        v03_ = _mm256_fmadd_pd(a01, _mm256_broadcast_sd(b + 7), v03_);
                        v43_ = _mm256_fmadd_pd(a41, _mm256_broadcast_sd(b + 7), v43_);
                    }
                    
                    {
                        __m256d const a02 = _mm256_load_pd(a + 8);
                        __m256d const a42 = _mm256_load_pd(a + sa + 8);
                        v00_ = _mm256_fmadd_pd(a02, _mm256_broadcast_sd(b + 8), v00_);
                        v40_ = _mm256_fmadd_pd(a42, _mm256_broadcast_sd(b + 8), v40_);
                        v01_ = _mm256_fmadd_pd(a02, _mm256_broadcast_sd(b + 9), v01_);
                        v41_ = _mm256_fmadd_pd(a42, _mm256_broadcast_sd(b + 9), v41_);
                        v02_ = _mm256_fmadd_pd(a02, _mm256_broadcast_sd(b + 10), v02_);
                        v42_ = _mm256_fmadd_pd(a42, _mm256_broadcast_sd(b + 10), v42_);
                        v03_ = _mm256_fmadd_pd(a02, _mm256_broadcast_sd(b + 11), v03_);
                        v43_ = _mm256_fmadd_pd(a42, _mm256_broadcast_sd(b + 11), v43_);
                    }
                    
                    {
                        __m256d const a03 = _mm256_load_pd(a + 12);
                        __m256d const a43 = _mm256_load_pd(a + sa + 12);
                        v00_ = _mm256_fmadd_pd(a03, _mm256_broadcast_sd(b + 12), v00_);
                        v40_ = _mm256_fmadd_pd(a43, _mm256_broadcast_sd(b + 12), v40_);
                        v01_ = _mm256_fmadd_pd(a03, _mm256_broadcast_sd(b + 13), v01_);
                        v41_ = _mm256_fmadd_pd(a43, _mm256_broadcast_sd(b + 13), v41_);
                        v02_ = _mm256_fmadd_pd(a03, _mm256_broadcast_sd(b + 14), v02_);
                        v42_ = _mm256_fmadd_pd(a43, _mm256_broadcast_sd(b + 14), v42_);                    
                        v03_ = _mm256_fmadd_pd(a03, _mm256_broadcast_sd(b + 15), v03_);
                        v43_ = _mm256_fmadd_pd(a43, _mm256_broadcast_sd(b + 15), v43_);
                    }
                }
            }
            else
            {
                SMOKE_THROW_EXCEPTION(std::logic_error("Not implemented"));
            }
        }


    private:
        __m256d v00_;
        __m256d v01_;
        __m256d v02_;
        __m256d v03_;
        __m256d v40_;
        __m256d v41_;
        __m256d v42_;
        __m256d v43_;
    };
}