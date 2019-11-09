#pragma once

#include <blaze/util/Types.h>
#include <blaze/system/Inline.h>

#include <immintrin.h>

#include <cstdint>


namespace blazefeo
{
    using namespace blaze;


    template <typename T, size_t SIMD_SIZE>
    struct Simd;


    template <>
    struct Simd<double, 4>
    {
        using IntrinsicType = __m256d;
    };


    //*******************************************************
    //
    // LOAD
    //
    //*******************************************************

    template <size_t SS, typename T>
    auto load(T const * ptr);


    template <size_t SS, typename T>
    auto broadcast(T const * ptr);


    template <>
    inline auto load<4, double>(double const * ptr)
    {
        return _mm256_load_pd(ptr);
    }


    template <>
    inline auto broadcast<4, double>(double const * ptr)
    {
        return _mm256_broadcast_sd(ptr);
    }


    //*******************************************************
    //
    // STORE
    //
    //*******************************************************

    inline void store(double * ptr, __m256d a)
    {
        _mm256_store_pd(ptr, a);
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


    inline auto set(long long a3, long long a2, long long a1, long long a0)
    {
        return _mm256_set_epi64x(a3, a2, a1, a0);
    }


    template <typename T, size_t N>
    auto setzero();


    template <>
    inline auto setzero<double, 4>()
    {
        return _mm256_setzero_pd();
    }


    //*******************************************************
    //
    // ARITHMETIC
    //
    //*******************************************************

    inline auto fnmadd(__m256d a, __m256d b, __m256d c)
    {
        return _mm256_fnmadd_pd(a, b, c);
    }


    //*******************************************************
    //
    // COMPARE
    //
    //*******************************************************
    
    // inline auto operator>(__m256i a, __m256i b)
    // {
    //     return _mm256_cmpgt_epi64(a, b);
    // }
}