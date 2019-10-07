#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/simd/Hsum.hpp>
#include <smoke/Exception.hpp>

#include <immintrin.h>


namespace smoke
{
    size_t constexpr panelSize = 4;


    // TODO: deprecate?
    template <typename T, size_t N = panelSize>
    class Panel
    {
    public:
        static size_t constexpr alignment = 0x1;


        Panel()
        {            
        }


        explicit Panel(T const * ptr)
        {
            load(ptr);
        }


        void load(T const * ptr)
        {
            for (size_t j = 0; j < N * N; ++j)
                v_[j] = ptr[j];
        }


        void store(T * ptr) const
        {
            for (size_t j = 0; j < N * N; ++j)
                ptr[j] = v_[j];
        }


        friend void gemm(Panel const& a, Panel const& b, Panel& c)
        {
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < N; ++i)
                    for (size_t k = 0; k < N; ++k)
                        c.v_[i + N * j] += a.v_[i + k * N] * b.v_[k + j * N];
        }


    private:
        T v_[N * N];
    };


    template <>
    class Panel<double, 4>
    {
    public:
        static size_t constexpr alignment = 0x20;


        Panel()
        {            
        }


        explicit Panel(double const * ptr)
        {
            load(ptr);
        }


        void load(double const * ptr)
        {
            v0_ = _mm256_load_pd(ptr);
            v1_ = _mm256_load_pd(ptr + 4);
            v2_ = _mm256_load_pd(ptr + 8);
            v3_ = _mm256_load_pd(ptr + 12);
        }


        void store(double * ptr) const
        {
            _mm256_store_pd(ptr, v0_);
            _mm256_store_pd(ptr + 4, v1_);
            _mm256_store_pd(ptr + 8, v2_);
            _mm256_store_pd(ptr + 12, v3_);
        }


        friend void gemm(Panel const& a, bool ta, Panel const& b, bool tb, Panel& c)
        {
            if (ta && !tb)
            {
                c.v0_ = _mm256_add_pd(c.v0_, hsum(
                    _mm256_mul_pd(a.v0_, b.v0_),
                    _mm256_mul_pd(a.v1_, b.v0_),
                    _mm256_mul_pd(a.v2_, b.v0_),
                    _mm256_mul_pd(a.v3_, b.v0_)
                ));

                c.v1_ = _mm256_add_pd(c.v1_, hsum(
                    _mm256_mul_pd(a.v0_, b.v1_),
                    _mm256_mul_pd(a.v1_, b.v1_),
                    _mm256_mul_pd(a.v2_, b.v1_),
                    _mm256_mul_pd(a.v3_, b.v1_)
                ));

                c.v2_ = _mm256_add_pd(c.v2_, hsum(
                    _mm256_mul_pd(a.v0_, b.v2_),
                    _mm256_mul_pd(a.v1_, b.v2_),
                    _mm256_mul_pd(a.v2_, b.v2_),
                    _mm256_mul_pd(a.v3_, b.v2_)
                ));

                c.v3_ = _mm256_add_pd(c.v3_, hsum(
                    _mm256_mul_pd(a.v0_, b.v3_),
                    _mm256_mul_pd(a.v1_, b.v3_),
                    _mm256_mul_pd(a.v2_, b.v3_),
                    _mm256_mul_pd(a.v3_, b.v3_)
                ));
            }
            else if (!ta && !tb)
            {
                c.v0_ = _mm256_fmadd_pd(a.v0_, _mm256_permute4x64_pd(b.v0_, 0b00000000), c.v0_);
                c.v0_ = _mm256_fmadd_pd(a.v1_, _mm256_permute4x64_pd(b.v0_, 0b01010101), c.v0_);
                c.v0_ = _mm256_fmadd_pd(a.v2_, _mm256_permute4x64_pd(b.v0_, 0b10101010), c.v0_);
                c.v0_ = _mm256_fmadd_pd(a.v3_, _mm256_permute4x64_pd(b.v0_, 0b11111111), c.v0_);

                c.v1_ = _mm256_fmadd_pd(a.v0_, _mm256_permute4x64_pd(b.v1_, 0b00000000), c.v1_);
                c.v1_ = _mm256_fmadd_pd(a.v1_, _mm256_permute4x64_pd(b.v1_, 0b01010101), c.v1_);
                c.v1_ = _mm256_fmadd_pd(a.v2_, _mm256_permute4x64_pd(b.v1_, 0b10101010), c.v1_);
                c.v1_ = _mm256_fmadd_pd(a.v3_, _mm256_permute4x64_pd(b.v1_, 0b11111111), c.v1_);

                c.v2_ = _mm256_fmadd_pd(a.v0_, _mm256_permute4x64_pd(b.v2_, 0b00000000), c.v2_);
                c.v2_ = _mm256_fmadd_pd(a.v1_, _mm256_permute4x64_pd(b.v2_, 0b01010101), c.v2_);
                c.v2_ = _mm256_fmadd_pd(a.v2_, _mm256_permute4x64_pd(b.v2_, 0b10101010), c.v2_);
                c.v2_ = _mm256_fmadd_pd(a.v3_, _mm256_permute4x64_pd(b.v2_, 0b11111111), c.v2_);

                c.v3_ = _mm256_fmadd_pd(a.v0_, _mm256_permute4x64_pd(b.v3_, 0b00000000), c.v3_);
                c.v3_ = _mm256_fmadd_pd(a.v1_, _mm256_permute4x64_pd(b.v3_, 0b01010101), c.v3_);
                c.v3_ = _mm256_fmadd_pd(a.v2_, _mm256_permute4x64_pd(b.v3_, 0b10101010), c.v3_);
                c.v3_ = _mm256_fmadd_pd(a.v3_, _mm256_permute4x64_pd(b.v3_, 0b11111111), c.v3_);
            }
            else
            {
                SMOKE_THROW_EXCEPTION(std::logic_error("Not implemented"));
            }
        }


    private:
        __m256d v0_;
        __m256d v1_;
        __m256d v2_;
        __m256d v3_;
    };
}