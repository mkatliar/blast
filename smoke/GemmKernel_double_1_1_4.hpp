#pragma once

#include <smoke/GemmKernel.hpp>

#include <smoke/simd/Hsum.hpp>
#include <smoke/Exception.hpp>

#include <immintrin.h>


namespace smoke
{
    template <>
    class GemmKernel<double, 1, 1, 4>
    {
    public:
        static size_t constexpr alignment = 0x20;
    };


    inline void gemm(GemmKernel<double, 1, 1, 4>, size_t K, 
        double const * a, size_t sa, bool ta, double const * b, size_t sb, bool tb, double const * c, size_t sc, double * d, size_t sd)
    {
        using Traits = GemmKernelTraits<GemmKernel<double, 1, 1, 4>>;
        if (K % Traits::blockSize)
            SMOKE_THROW_EXCEPTION(std::logic_error("k is not a multiple of block size"));

        __m256d v0_ = _mm256_load_pd(c);
        __m256d v1_ = _mm256_load_pd(c + 4);
        __m256d v2_ = _mm256_load_pd(c + 8);
        __m256d v3_ = _mm256_load_pd(c + 12);

        if (ta && !tb)
        {
            for (size_t k = 0; k + Traits::blockSize <= K; k += Traits::blockSize, a += sa, b += sb)
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
        else if (!ta && !tb)
        {
            for (size_t k = 0; k + Traits::blockSize <= K; k += Traits::blockSize, a += Traits::blockElementCount, b += sb)
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
        else if (!ta && tb)
        {
            // for (size_t k = 0; k + Traits::blockSize <= K; k += Traits::blockSize, a += Traits::blockElementCount, b += Traits::blockElementCount)
            for (size_t k = 0; k < K; ++k, a += Traits::blockSize, b += Traits::blockSize)
            {
                {
                    __m256d const a_v0 = _mm256_load_pd(a);
                    v0_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b), v0_);
                    v1_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 1), v1_);
                    v2_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 2), v2_);
                    v3_ = _mm256_fmadd_pd(a_v0, _mm256_broadcast_sd(b + 3), v3_);
                }

                // {
                //     __m256d const a_v1 = _mm256_load_pd(a + 4);
                //     v0_ = _mm256_fmadd_pd(a_v1, _mm256_broadcast_sd(b + 4), v0_);
                //     v1_ = _mm256_fmadd_pd(a_v1, _mm256_broadcast_sd(b + 5), v1_);
                //     v2_ = _mm256_fmadd_pd(a_v1, _mm256_broadcast_sd(b + 6), v2_);
                //     v3_ = _mm256_fmadd_pd(a_v1, _mm256_broadcast_sd(b + 7), v3_);
                // }

                // {
                //     __m256d const a_v2 = _mm256_load_pd(a + 8);
                //     v0_ = _mm256_fmadd_pd(a_v2, _mm256_broadcast_sd(b + 8), v0_);
                //     v1_ = _mm256_fmadd_pd(a_v2, _mm256_broadcast_sd(b + 9), v1_);
                //     v2_ = _mm256_fmadd_pd(a_v2, _mm256_broadcast_sd(b + 10), v2_);
                //     v3_ = _mm256_fmadd_pd(a_v2, _mm256_broadcast_sd(b + 11), v3_);
                // }

                // {
                //     __m256d const a_v3 = _mm256_load_pd(a + 12);
                //     v0_ = _mm256_fmadd_pd(a_v3, _mm256_broadcast_sd(b + 12), v0_);
                //     v1_ = _mm256_fmadd_pd(a_v3, _mm256_broadcast_sd(b + 13), v1_);
                //     v2_ = _mm256_fmadd_pd(a_v3, _mm256_broadcast_sd(b + 14), v2_);
                //     v3_ = _mm256_fmadd_pd(a_v3, _mm256_broadcast_sd(b + 15), v3_);
                // }
            }
        }
        else
        {
            SMOKE_THROW_EXCEPTION(std::logic_error("Not implemented"));
        }

        _mm256_store_pd(d, v0_);
        _mm256_store_pd(d + 4, v1_);
        _mm256_store_pd(d + 8, v2_);
        _mm256_store_pd(d + 12, v3_);
    }
}