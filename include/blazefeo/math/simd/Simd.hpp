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


    template <bool AF, typename T>
    inline auto load(T const * ptr)
    {
        return load<AF, Simd<T>::size>(ptr);
    }


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


    inline auto set1(double a)
    {
        return _mm256_set1_pd(a);
    }


    inline auto set1(float a)
    {
        return _mm256_set1_ps(a);
    }


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


    inline auto abs(__m256d a)
    {
        return _mm256_andnot_pd(set1(-0.), a);
    }


    inline auto abs(__m256 a)
    {
        return _mm256_andnot_ps(set1(-0.f), a);
    }


    /**
     * @brief Horizontal max
     *
     * The implementation is based on
     * https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
     *
     * @param x pack of 64-bit floating point numbers
     *
     * @return max(x[0], x[1], x[2], x[3])
     */
    inline double hmax(__m256d x)
    {
        __m256d y = _mm256_permute2f128_pd(x, x, 1); // permute 128-bit values
        __m256d m1 = _mm256_max_pd(x, y); // m1[0] = max(x[0], x[2]), m1[1] = max(x[1], x[3]), etc.
        __m256d m2 = _mm256_permute_pd(m1, 5); // set m2[0] = m1[1], m2[1] = m1[0], etc.
        __m256d m = _mm256_max_pd(m1, m2); // all m[0] ... m[3] contain the horizontal max(x[0], x[1], x[2], x[3])

        return m[0];
    }


    /**
     * @brief Horizontal max
     *
     * The implementation is based on
     * https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
     *
     * @param a pack of 32-bit floating point numbers
     *
     * @return max(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
     */
    inline float hmax(__m256 a)
    {
        __m256 v1 = a;                                          /* v1 = [H G F E | D C B A]                                                                         */
        __m256 v2 = _mm256_permute_ps(v1, 0b10'11'00'01);       /* v2 = [G H E F | C D A B]                                                                         */
        __m256 v3 = _mm256_max_ps(v1, v2);                      /* v3 = [W=max(G,H) W=max(G,H) Z=max(E,F) Z=max(E,F) | Y=max(C,D) Y=max(C,D) X=max(A,B) X=max(A,B)] */
                                                                /* v3 = [W W Z Z | Y Y X X]                                                                         */
        __m256 v4 = _mm256_permute_ps(v3, 0b00'00'10'10);       /* v4 = [Z Z W W | X X Y Y]                                                                         */
        __m256 v5 = _mm256_max_ps(v3, v4);                      /* v5 = [J=max(Z,W) J=max(Z,W) J=max(Z,W) J=max(Z,W) | I=max(X,Y) I=max(X,Y) I=max(X,Y) I=max(X,Y)] */
                                                                /* v5 = [J J J J | I I I I]                                                                         */
        __m128 v6 = _mm256_extractf128_ps(v5, 1);               /* v6 = [- - - - | J J J J]                                                                         */
        __m128 v7 = _mm_max_ps(_mm256_castps256_ps128(v5), v6); /* v7 = [- - - - | M=max(I,J) M=max(I,J) M=max(I,J) M=max(I,J)]                                     */

        return v7[0];
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


    //*******************************************************
    //
    // MISC
    //
    //*******************************************************

    inline void imax_packed(__m256d& a, __m256i& ia, __m256d b, __m256i ib)
    {
        __m256d mask = _mm256_cmp_pd(b, a, _CMP_GT_OQ);
        a = _mm256_blendv_pd(a, b, mask);
        ia = _mm256_blendv_epi8(ia, ib, _mm256_castpd_si256(mask));
    }


    inline void imax_packed(__m256& a, __m256i& ia, __m256 b, __m256i ib)
    {
        __m256 mask = _mm256_cmp_ps(b, a, _CMP_GT_OQ);
        a = _mm256_blendv_ps(a, b, mask);
        ia = _mm256_blendv_epi8(ia, ib, _mm256_castps_si256(mask));
    }
}