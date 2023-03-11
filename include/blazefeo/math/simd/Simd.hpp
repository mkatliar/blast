// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/util/Types.h>
#include <blaze/system/Inline.h>
#include <blaze/math/AlignmentFlag.h>

#include <immintrin.h>

#include <cstdint>
#include <type_traits>


namespace blazefeo
{
    using namespace blaze;


    template <typename T>
    struct Simd;


    template <>
    struct Simd<double>
    {
    private:
        class Index;


    public:
        using IntrinsicType = __m256d;
        using MaskType = __m256i;
        using IntType = long long;

        static size_t constexpr size = 4;
        static size_t constexpr registerCapacity = 16;


        //*******************************************************
        //
        // CUSTOM
        //
        //*******************************************************
        static Index index() noexcept
        {
            return Index {};
        }


    private:
        class Index
        {
        public:
            MaskType operator<(IntType n) const noexcept
            {
                return _mm256_cmpgt_epi64(_mm256_set1_epi64x(n), val_);
            }


            MaskType operator>(IntType n) const noexcept
            {
                return _mm256_cmpgt_epi64(val_, _mm256_set1_epi64x(n));
            }


            MaskType operator<=(IntType n) const noexcept
            {
                return *this < n + 1;
            }


            MaskType operator>=(IntType n) const noexcept
            {
                return *this > n - 1;
            }


        private:
            MaskType val_ = _mm256_set_epi64x(3, 2, 1, 0);
        };
    };


    template <>
    struct Simd<float>
    {
    private:
        class Index;


    public:
        using IntrinsicType = __m256;
        using MaskType = __m256i;
        using IntType = int;

        static size_t constexpr size = 8;
        static size_t constexpr registerCapacity = 16;


        //*******************************************************
        //
        // CUSTOM
        //
        //*******************************************************
        static Index index() noexcept
        {
            return Index {};
        }


    private:
        class Index
        {
        public:
            MaskType operator<(IntType n) const noexcept
            {
                return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), val_);
            }


            MaskType operator>(IntType n) const noexcept
            {
                return _mm256_cmpgt_epi32(val_, _mm256_set1_epi32(n));
            }


            MaskType operator<=(IntType n) const noexcept
            {
                return *this < n + 1;
            }


            MaskType operator>=(IntType n) const noexcept
            {
                return *this > n - 1;
            }


        private:
            MaskType val_ = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        };
    };


    template <typename T>
    using IntrinsicType_t = typename Simd<T>::IntrinsicType;


    template <typename T>
    struct SimdTraits;


    template <>
    struct SimdTraits<__m256d>
    {
        using ScalarType = double;
        using MaskType = __m256i;
        using IntType = long long;
        static size_t constexpr size = 4;
    };


    template <>
    struct SimdTraits<__m256>
    {
        using ScalarType = float;
        using MaskType = __m256i;
        using IntType = int;
        static size_t constexpr size = 8;
    };


    template <typename T>
    using ScalarType_t = typename SimdTraits<T>::ScalarType;


    template <typename T>
    using MaskType_t = typename SimdTraits<T>::MaskType;

    template <typename T>
    using IntType_t = typename SimdTraits<T>::IntType;


    template <typename T>
    size_t constexpr SimdSize_v = SimdTraits<T>::size;


    template <typename T>
    size_t constexpr RegisterCapacity_v = Simd<T>::registerCapacity;


    //*******************************************************
    //
    // LOAD
    //
    //*******************************************************

    template <bool AF, size_t SS, typename T>
    auto load(T const * ptr);


    template <size_t SS, typename T>
    auto broadcast(T const * ptr);


    template <>
    inline auto load<aligned, 4, double>(double const * ptr)
    {
        return _mm256_load_pd(ptr);
    }


    template <>
    inline auto load<unaligned, 4, double>(double const * ptr)
    {
        return _mm256_loadu_pd(ptr);
    }


    template <>
    inline auto load<aligned, 8, float>(float const * ptr)
    {
        return _mm256_load_ps(ptr);
    }


    template <>
    inline auto load<unaligned, 8, float>(float const * ptr)
    {
        return _mm256_loadu_ps(ptr);
    }


    inline auto maskload(double const * ptr, __m256i mask)
    {
        return _mm256_maskload_pd(ptr, mask);
    }


    inline auto maskload(float const * ptr, __m256i mask)
    {
        return _mm256_maskload_ps(ptr, mask);
    }


    template <>
    inline auto broadcast<4, double>(double const * ptr)
    {
        return _mm256_broadcast_sd(ptr);
    }


    template <>
    inline auto broadcast<8, float>(float const * ptr)
    {
        return _mm256_broadcast_ss(ptr);
    }


    //*******************************************************
    //
    // STORE
    //
    //*******************************************************
    template <bool AF>
    void store(double * ptr, __m256d a);


    template <>
    inline void store<aligned>(double * ptr, __m256d a)
    {
        _mm256_store_pd(ptr, a);
    }


    template <>
    inline void store<unaligned>(double * ptr, __m256d a)
    {
        _mm256_storeu_pd(ptr, a);
    }


    template <bool AF>
    void store(float * ptr, __m256 a);


    template <>
    inline void store<aligned>(float * ptr, __m256 a)
    {
        _mm256_store_ps(ptr, a);
    }


    template <>
    inline void store<unaligned>(float * ptr, __m256 a)
    {
        _mm256_storeu_ps(ptr, a);
    }


    inline void maskstore(double * ptr, __m256i m, __m256d a)
    {
        _mm256_maskstore_pd(ptr, m, a);
    }


    inline void maskstore(float * ptr, __m256i m, __m256 a)
    {
        _mm256_maskstore_ps(ptr, m, a);
    }


    //*******************************************************
    //
    // SET
    //
    //*******************************************************

    inline auto set(double a3, double a2, double a1, double a0)
    {
        return _mm256_set_pd(a3, a2, a1, a0);
    }


    inline auto set(float a7, float a6, float a5, float a4, float a3, float a2, float a1, float a0)
    {
        return _mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0);
    }


    inline auto set(long long a3, long long a2, long long a1, long long a0)
    {
        return _mm256_set_epi64x(a3, a2, a1, a0);
    }


    template <size_t N, typename T>
    auto set1(T a);


    template <>
    inline auto set1<4, double>(double a)
    {
        return _mm256_set1_pd(a);
    }


    template <>
    inline auto set1<8, float>(float a)
    {
        return _mm256_set1_ps(a);
    }


    template <>
    inline auto set1<8, int>(int val)
    {
        return _mm256_set1_epi32(val);
    }


    template <>
    inline auto set1<4, long long>(long long val)
    {
        return _mm256_set1_epi64x(val);
    }


    template <typename T, size_t N>
    auto setzero();


    template <>
    inline auto setzero<double, 4>()
    {
        return _mm256_setzero_pd();
    }


    template <>
    inline auto setzero<float, 8>()
    {
        return _mm256_setzero_ps();
    }


    //*******************************************************
    //
    // ARITHMETIC
    //
    //*******************************************************

    inline auto fmadd(__m256d a, __m256d b, __m256d c)
    {
        return _mm256_fmadd_pd(a, b, c);
    }


    inline auto fmadd(__m256 a, __m256 b, __m256 c)
    {
        return _mm256_fmadd_ps(a, b, c);
    }


    inline auto fnmadd(__m256d a, __m256d b, __m256d c)
    {
        return _mm256_fnmadd_pd(a, b, c);
    }


    inline auto fnmadd(__m256 a, __m256 b, __m256 c)
    {
        return _mm256_fnmadd_ps(a, b, c);
    }


    //*******************************************************
    //
    // COMPARE
    //
    //*******************************************************

    template <size_t N, typename MM>
    auto cmpgt(MM a, MM b);


    template <>
    inline auto cmpgt<4>(__m256i a, __m256i b)
    {
        return _mm256_cmpgt_epi64(a, b);
    }


    template <>
    inline auto cmpgt<8>(__m256i a, __m256i b)
    {
        return _mm256_cmpgt_epi32(a, b);
    }
}