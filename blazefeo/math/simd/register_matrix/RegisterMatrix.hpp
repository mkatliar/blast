#pragma once

#include <blazefeo/math/simd/Simd.hpp>

#include <blaze/util/Types.h>
#include <blaze/util/Exception.h>
#include <blaze/system/Inline.h>

#include <cmath>

#include <immintrin.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename T, size_t M, size_t N, size_t SS>
    class RegisterMatrix
    {
    public:
        /// @brief Default ctor
        RegisterMatrix()
        {
        }


        static size_t constexpr rows()
        {
            return M * SS;
        }


        static size_t constexpr columns()
        {
            return N;
        }


        T& at(size_t i, size_t j)
        {
            return v_[i / SS][j][i % SS];
        }


        T at(size_t i, size_t j) const
        {
            return v_[i / SS][j][i % SS];
        }


        /// @brief load from memory
        void load(T beta, T const * ptr, size_t spacing);


        /// @brief load from memory with specified size
        void load(T beta, T const * ptr, size_t spacing, size_t m, size_t n);


        /// @brief store to memory
        void store(T * ptr, size_t spacing) const;


        /// @brief store to memory with specified size
        void store(T * ptr, size_t spacing, size_t m, size_t n) const;


        /// @brief Rank-1 update
        template <bool TA, bool TB>
        void ger(T alpha, T const * a, size_t sa, T const * b, size_t sb);


        /// @brief Rank-1 update of specified size
        template <bool TA, bool TB>
        void ger(T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n);


        /// @brief In-place Cholesky decomposition
        void potrf();


        /// @brief Triangular substitution
        template <bool LeftSide, bool Upper, bool TransA>
        void trsm(T const * l, size_t sl);


    private:
        using IntrinsicType = typename Simd<T, SS>::IntrinsicType;
        
        IntrinsicType v_[M][N];
    };


    template <typename Ker>
    struct RegisterMatrixTraits;


    template <typename T, size_t M, size_t N, size_t SS>
    struct RegisterMatrixTraits<RegisterMatrix<T, M, N, SS>>
    {
        static size_t constexpr simdSize = SS;
        static size_t constexpr rows = M * SS;
        static size_t constexpr columns = N;
        static size_t constexpr elementCount = rows * columns;
        
        using ElementType = T;
    };


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::load(T beta, T const * ptr, size_t spacing)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                v_[i][j] = blazefeo::load<SS>(ptr + spacing * i + SS * j);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::store(T * ptr, size_t spacing) const
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                blazefeo::store(ptr + spacing * i + SS * j, v_[i][j]);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::store(T * ptr, size_t spacing, size_t m, size_t n) const
    {
        for (size_t i = 0; i < M; ++i) if (SS * (i + 1) <= m)
            // The compile-time constant size of the j loop in combination with the if() expression
            // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
            for (size_t j = 0; j < N; ++j) if (j < n)
                blazefeo::store(ptr + spacing * i + SS * j, v_[i][j]);

        if (long long const rem = m % SS)
        {
            static_assert(SS == 4, "Partial store of RegisterMatrix for SIMD size is not implemented");

            __m256i const mask = set(rem, rem, rem, rem) > set(3LL, 2LL, 1LL, 0LL);
            size_t const i = m / SS;

            for (size_t j = 0; j < n && j < columns(); ++j)
                _mm256_maskstore_pd(ptr + spacing * i + SS * j, mask, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, size_t SS>
    template <bool LeftSide, bool Upper, bool TransA>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SS>::trsm(T const * l, size_t sl)
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j)
        {
            #pragma unroll
            for (size_t k = 0; k < j; ++k)
            {
                IntrinsicType const l_jk = broadcast<SS>(l + (j / SS) * sl + j % SS + k * SS);

                #pragma unroll
                for (size_t i = 0; i < M; ++i)
                    v_[i][j] = fnmadd(l_jk, v_[i][k], v_[i][j]);
            }

            IntrinsicType const l_jj = broadcast<SS>(l + (j / SS) * sl + j % SS + j * SS);
            
            #pragma unroll
            for (size_t i = 0; i < M; ++i)
                v_[i][j] /= l_jj;
        }
    }


    template <typename T, size_t M, size_t N, size_t SS>
    template <bool TA, bool TB>
    inline void RegisterMatrix<T, M, N, SS>::ger(T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented");
    }


    template <bool LeftSide, bool Upper, bool TransA, typename T, size_t M, size_t N, size_t SS>
    inline void trsm(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa)
    {
        ker.template trsm<LeftSide, Upper, TransA>(a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SS>::potrf()
    {
        static_assert(M * SS >= N, "potrf() not implemented for register matrices with columns more than rows");
        
        #pragma unroll
        for (size_t k = 0; k < N; ++k)
        {
            #pragma unroll
            for (size_t j = 0; j < k; ++j)
            {
                T const a_kj = v_[k / SS][j][k % SS];

                #pragma unroll
                for (size_t i = 0; i < M; ++i) if (i >= k / SS)
                    v_[i][k] = fnmadd(set(a_kj, a_kj, a_kj, a_kj), v_[i][j], v_[i][k]);
            }

            T const sqrt_a_kk = std::sqrt(v_[k / SS][k][k % SS]);
            
            #pragma unroll
            for (size_t i = 0; i < M; ++i) 
            {
                if (i < k / SS)
                    v_[i][k] = setzero<T, SS>();
                else
                    v_[i][k] /= sqrt_a_kk;
            }
        }     
    }


    /// @brief Rank-1 update
    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        ker.template ger<TA, TB>(alpha, a, sa, b, sb);
    }


    /// @brief Rank-1 update of specified size
    template <bool TA, bool TB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        ker.template ger<TA, TB>(alpha, a, sa, b, sb, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa)
    {
        ker.load(1.0, a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(1.0, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T beta, T const * a, size_t sa)
    {
        ker.load(beta, a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T beta, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(beta, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SS> const& ker, T * a, size_t sa)
    {
        ker.store(a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SS> const& ker, T * a, size_t sa, size_t m, size_t n)
    {
        ker.store(a, sa, m, n);
    }
}