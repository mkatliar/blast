// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/simd/MatrixPointer.hpp>
#include <blazefeo/math/simd/VectorPointer.hpp>
#include <blazefeo/math/dense/MatrixPointer.hpp>
#include <blazefeo/math/dense/VectorPointer.hpp>
#include <blazefeo/math/Side.hpp>
#include <blazefeo/math/UpLo.hpp>

#include <blaze/math/StorageOrder.h>
#include <blaze/math/Matrix.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Types.h>
#include <blaze/util/Exception.h>
#include <blaze/util/Assert.h>
#include <blaze/util/StaticAssert.h>

#include <cmath>


namespace blazefeo
{
    using namespace blaze;


    /// @brief Register-resident matrix
    ///
    /// @tparam T type of matrix elements
    /// @tparam M number of rows of the matrix. Must be a multiple of SS.
    /// @tparam N number of columns of the matrix.
    /// @tparam SO orientation of SIMD registers.
    template <typename T, size_t M, size_t N, bool SO = columnMajor>
    class RegisterMatrix
    :   public Matrix<RegisterMatrix<T, M, N, SO>, SO>
    {
    public:
        static_assert(SO == columnMajor, "Only column-major register matrices are currently supported");

        using BaseType = Matrix<RegisterMatrix<T, M, N, SO>, SO>;
        using BaseType::storageOrder;

        /// @brief Type of matrix elements
        using ElementType = T;
        using CompositeType = RegisterMatrix const&;              //!< Data type for composite expression templates.


        /// @brief Default ctor
        RegisterMatrix()
        {
            reset();
        }


        /// @brief Copying prohibited
        RegisterMatrix(RegisterMatrix const&) = delete;


        /// @brief Assignment prohibited
        RegisterMatrix& operator=(RegisterMatrix const&) = delete;


        /// @brief Number of matrix rows
        static size_t constexpr rows()
        {
            return M;
        }


        /// @brief Number of matrix columns
        static size_t constexpr columns()
        {
            return N;
        }


        /// @brief Number of matrix panels
        static size_t constexpr panels()
        {
            return RM;
        }


        /// @brief Number of registers used
        static size_t constexpr registers()
        {
            return RM * N;
        }


        /// @brief SIMD size
        static size_t constexpr simdSize()
        {
            return SS;
        }


        /// @brief Value of the matrix element at row \a i and column \a j
        T operator()(size_t i, size_t j) const noexcept
        {
            return at(i, j);
        }


        /// @brief Set all elements to 0.
        void reset() noexcept
        {
            for (size_t i = 0; i < RM; ++i)
                for (size_t j = 0; j < N; ++j)
                    v_[i][j] = setzero<T, SS>();
        }


        /// @brief Multiply all elements by a constant.
        void operator*=(T alpha) noexcept
        {
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                #pragma unroll
                for (size_t j = 0; j < RN; ++j)
                    v_[i][j] *= alpha;
        }


        /// @brief R += beta * A
        template <typename PA>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        void axpy(T beta, PA a) noexcept
        {
            #pragma unroll
            for (size_t j = 0; j < N; ++j)
                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] += beta * a(SS * i, j).load();
        }


        /// @brief R(0:m-1, 0:n-1) += beta * A
        template <typename PA>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        void axpy(T beta, PA a, size_t m, size_t n) noexcept
        {
            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n)
                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (i * RM < m)
                    v_[i][j] += beta * a(SS * i, j).load();
        }


        /// @brief load from memory
        // [[deprecated("Use load with a matrix argument instead")]]
        void load(T beta, T const * ptr, size_t spacing);


        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(P p) noexcept;


        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(T beta, P p) noexcept;


        /// @brief load from memory with specified size
        void load(T beta, T const * ptr, size_t spacing, size_t m, size_t n);


        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(T beta, P p, size_t m, size_t n) noexcept;


        /// @brief store to memory
        void store(T * ptr, size_t spacing) const;


        /// @brief Store matrix at location pointed by \a p
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void store(P p) const noexcept;


        /// @brief Store lower-triangular part of the matrix at location pointed by \a p.
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void storeLower(P p) const noexcept;


        /// @brief Store lower-triangular part of the matrix
        /// of size \a m by \a n at location pointed by \a p.
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void storeLower(P p, size_t m, size_t n) const noexcept;


        /// @brief store to memory with specified size
        void store(T * ptr, size_t spacing, size_t m, size_t n) const;


        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void store(P p, size_t m, size_t n) const noexcept;


        /// @brief Rank-1 update
        ///
        /// @tparam SOA storage order of the panels of the first matrix
        /// @tparam SOB storage order of the panels of the second matrix
        ///
        /// @param a pointer to the first element of the column of the first matrix. Must be aligned on panel boundary.
        /// @param sa pointer distance between the consecutive panels of the first matrix.
        /// @param b pointer to the first element of the row of the second matrix. Must be aligned on panel boundary.
        /// @param sb pointer distance between the consecutive panels of the second matrix.
        template <bool SOA, bool SOB>
        void ger(T alpha, T const * a, size_t sa, T const * b, size_t sb);


        template <typename PA, typename PB>
        requires
            VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
            VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
        void ger(T alpha, PA a, PB b) noexcept;


        template <typename PA, typename PB>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
        void ger(PA a, PB b) noexcept;


        /// @brief Rank-1 update of specified size
        ///
        /// @tparam SOA storage order of the panels of the first matrix
        /// @tparam SOB storage order of the panels of the second matrix
        ///
        /// @param a pointer to the first element of the column of the first matrix. Must be aligned on panel boundary.
        /// @param sa pointer distance between the consecutive panels of the first matrix.
        /// @param b pointer to the first element of the row of the second matrix. Must be aligned on panel boundary.
        /// @param sb pointer distance between the consecutive panels of the second matrix.
        template <bool SOA, bool SOB>
        void ger(T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n);


        template <typename PA, typename PB>
        requires
            VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
            VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
        void ger(T alpha, PA a, PB b, size_t m, size_t n) noexcept;


        template <typename PA, typename PB>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
        void ger(PA a, PB b, size_t m, size_t n) noexcept;


        /// @brief General matrix-matrix multiplication
        ///
        /// R += alpha * A * B
        ///
        template <typename PA, typename PB>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
        void gemm(size_t K, T alpha, PA a, PB b) noexcept;


		/// @brief General matrix-matrix multiplication with limited size
        ///
        /// R(0:md-1, 0:nd-1) += alpha * A * B
        ///
        template <typename PA, typename PB>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
        void gemm(size_t K, T alpha, PA a, PB b, size_t md, size_t nd) noexcept;


        /// @brief In-place Cholesky decomposition
        void potrf();


        /// @brief Triangular substitution, panel matrix pointer argument
        ///
        /// @brief l pointer to a triangular matrix
        ///
        template <bool LeftSide, bool Upper, bool TransA>
        void trsm(T const * l, size_t sl);


        /// @brief Triangular substitution, matrix pointer argument
        ///
        /// @brief a pointer to a triangular matrix
        ///
        template <typename P>
            requires MatrixPointer<P, T>
        void trsmRightUpper(P a);


        /// @brief Triangular matrix multiplication
        ///
        /// Performs the matrix-matrix operation
        ///
        /// R += alpha*A*B,
        ///
        /// where alpha is a scalar, B is an m by n matrix,
        /// A is an upper triangular matrix.
        ///
        /// @tparam P1 matrix A pointer type.
        /// @tparam P2 matrix B pointer type.
        ///
        /// @param a triangular matrix.
        /// @param b general matrix.
        ///
        template <typename P1, typename P2>
            requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
        void trmmLeftUpper(T alpha, P1 a, P2 b) noexcept;


        /// @brief Triangular matrix multiplication
        ///
        /// Performs the matrix-matrix operation
        ///
        /// R += alpha*B*A,
        ///
        /// where alpha is a scalar, B is an m by n matrix,
        /// A is a lower triangular matrix.
        ///
        /// @tparam P1 matrix A pointer type.
        /// @tparam P2 matrix B pointer type.
        ///
        /// @param a triangular matrix.
        /// @param b general matrix.
        ///
        template <typename P1, typename P2>
            requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
        void trmmRightLower(T alpha, P1 a, P2 b) noexcept;


    private:
        using SIMD = Simd<T>;
        using IntrinsicType = typename SIMD::IntrinsicType;
        using MaskType = typename SIMD::MaskType;
        using IntType = typename SIMD::IntType;

        // SIMD size
        static size_t constexpr SS = Simd<T>::size;

        // Numberf of SIMD registers required to store a single column of the matrix.
        static size_t constexpr RM = M / SS;
        static size_t constexpr RN = N;

        BLAZE_STATIC_ASSERT_MSG((RM > 0), "Number of rows must be not less than SIMD size");
        BLAZE_STATIC_ASSERT_MSG((RN > 0), "Number of columns must be positive");
        BLAZE_STATIC_ASSERT_MSG((M % SS == 0), "Number of rows must be a multiple of SIMD size");
        BLAZE_STATIC_ASSERT_MSG((RM * RN <= RegisterCapacity_v<T>), "Not enough registers for a RegisterMatrix");

        IntrinsicType v_[RM][RN];


        /// @brief Reference to the matrix element at row \a i and column \a j
        T& at(size_t i, size_t j)
        {
            return v_[i / SS][j][i % SS];
        }


        /// @brief Value of the matrix element at row \a i and column \a j
        T at(size_t i, size_t j) const
        {
            return v_[i / SS][j][i % SS];
        }
    };


    template <typename Ker>
    struct RegisterMatrixTraits;


    // TODO: deprecate
    template <typename T, size_t M, size_t N, bool SO>
    struct RegisterMatrixTraits<RegisterMatrix<T, M, N, SO>>
    {
        static size_t constexpr simdSize = RegisterMatrix<T, M, N, SO>::SS;
        static size_t constexpr rows = M;
        static size_t constexpr columns = N;
        static size_t constexpr elementCount = rows * columns;

        using ElementType = T;
    };


    template <typename T, size_t M, size_t N, bool SO>
    inline void RegisterMatrix<T, M, N, SO>::load(T beta, T const * ptr, size_t spacing)
    {
        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            #pragma unroll
            for (size_t j = 0; j < N; ++j)
                v_[i][j] = beta * blazefeo::load<aligned, SS>(ptr + spacing * i + SS * j);
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::load(P p) noexcept
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j)
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = p(SS * i, j).load();
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::load(T beta, P p) noexcept
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j)
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = beta * p(SS * i, j).load();
    }


    template <typename T, size_t M, size_t N, bool SO>
    inline void RegisterMatrix<T, M, N, SO>::load(T beta, T const * ptr, size_t spacing, size_t m, size_t n)
    {
        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n)
                v_[i][j] = beta * blazefeo::load<aligned, SS>(ptr + spacing * i + SS * j);
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::load(T beta, P p, size_t m, size_t n) noexcept
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n)
        {
            if constexpr (P::aligned && P::padded)
            {
                // If the storage is both aligned and padded, it should be safe to use unmasked loads, which are faster.
                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (SS * i < m)
                    v_[i][j] = beta * p(SS * i, j).load();
            }
            else
            {
                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (SS * i + SS <= m)
                    v_[i][j] = beta * p(SS * i, j).load();

                if (size_t const rem = m % SS)
                    v_[m / SS][j] = beta * p(m - rem, j).maskLoad(SIMD::index() < rem);
            }
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    inline void RegisterMatrix<T, M, N, SO>::store(T * ptr, size_t spacing) const
    {
        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            #pragma unroll
            for (size_t j = 0; j < N; ++j)
                blazefeo::store<aligned>(ptr + spacing * i + SS * j, v_[i][j]);
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::store(P p) const noexcept
    {
        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i < RM; ++i)
                p(SS * i, j).store(v_[i][j]);
    }


    template <typename T, size_t M, size_t N, bool SO>
    inline void RegisterMatrix<T, M, N, SO>::store(T * ptr, size_t spacing, size_t m, size_t n) const
    {
        BLAZE_STATIC_ASSERT_MSG((RM * RN + 2 <= RegisterCapacity_v<T>), "Not enough registers");
        BLAZE_INTERNAL_ASSERT(m > M - SS && m <= M, "Invalid number of rows in partial store");
        BLAZE_INTERNAL_ASSERT(n > 0 && n <= N, "Invalid number of columns in partial store");
        BLAZE_INTERNAL_ASSERT(m < M || n < N, "Partial store with full size");

        if (IntType const rem = m % SS)
        {
            #pragma unroll
            for (size_t i = 0; i < RM - 1; ++i)
                // The compile-time constant size of the j loop in combination with the if() expression
                // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
                #pragma unroll
                for (size_t j = 0; j < N; ++j) if (j < n)
                    blazefeo::store<aligned>(ptr + spacing * i + SS * j, v_[i][j]);

            MaskType const mask = SIMD::index() < rem;
            size_t constexpr i = RM - 1;

            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n)
                maskstore(ptr + spacing * i + SS * j, mask, v_[i][j]);
        }
        else
        {
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                // The compile-time constant size of the j loop in combination with the if() expression
                // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
                #pragma unroll
                for (size_t j = 0; j < N; ++j) if (j < n)
                    blazefeo::store<aligned>(ptr + spacing * i + SS * j, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::store(P p, size_t m, size_t n) const noexcept
    {
        // The compile-time constant size of the j loop in combination with the if() expression
        // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
        for (size_t j = 0; j < N; ++j) if (j < n)
            for (size_t i = 0; i < RM; ++i) if (SS * (i + 1) <= m)
                p(SS * i, j).store(v_[i][j]);

        if (IntType const rem = m % SS)
        {
            MaskType const mask = SIMD::index() < rem;
            size_t const i = m / SS;

            for (size_t j = 0; j < n && j < columns(); ++j)
                p(SS * i, j).maskStore(mask, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::storeLower(P p) const noexcept
    {
        for (size_t j = 0; j < N; ++j)
        {
            size_t ri = j / SS;
            IntType const skip = j % SS;

            if (skip && ri < RM)
            {
                MaskType const mask = SIMD::index() >= skip;
                p(SS * ri, j).maskStore(mask, v_[ri][j]);
                ++ri;
            }

            for(; ri < RM; ++ri)
                p(SS * ri, j).store(v_[ri][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::storeLower(P p, size_t m, size_t n) const noexcept
    {
        for (size_t j = 0; j < N; ++j) if (j < n)
        {
            for (size_t ri = j / SS; ri < RM; ++ri)
            {
                IntType const skip = j - ri * SS;
                IntType const rem = m - ri * SS;
                MaskType mask = SIMD::index() < rem;

                if (skip > 0)
                    mask &= SIMD::index() >= skip;

                p(SS * ri, j).maskStore(mask, v_[ri][j]);
            }
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <bool LeftSide, bool Upper, bool TransA>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trsm(T const * l, size_t sl)
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j)
        {
            #pragma unroll
            for (size_t k = 0; k < j; ++k)
            {
                IntrinsicType const l_jk = broadcast<SS>(l + (j / SS) * sl + j % SS + k * SS);

                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] = fnmadd(l_jk, v_[i][k], v_[i][j]);
            }

            IntrinsicType const l_jj = broadcast<SS>(l + (j / SS) * sl + j % SS + j * SS);

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] /= l_jj;
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trsmRightUpper(P a)
    {
        if constexpr (SO == columnMajor)
        {
            #pragma unroll
            for (size_t j = 0; j < N; ++j)
            {
                #pragma unroll
                for (size_t k = 0; k < j; ++k)
                {
                    IntrinsicType const a_kj = (~a)(k, j).broadcast();

                    #pragma unroll
                    for (size_t i = 0; i < RM; ++i)
                        v_[i][j] = fnmadd(a_kj, v_[i][k], v_[i][j]);
                }

                IntrinsicType const a_jj = (~a)(j, j).broadcast();

                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] /= a_jj;
            }
        }
        else
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <bool SOA, bool SOB>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        if (SOA == columnMajor && SOB == rowMajor)
        {
            BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= RegisterCapacity_v<T>), "Not enough registers for ger()");

            IntrinsicType ax[RM];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                ax[i] = alpha * blazefeo::load<aligned, SS>(a + i * sa);

            #pragma unroll
            for (size_t j = 0; j < N; ++j)
            {
                IntrinsicType bx = broadcast<SS>(b + (j / SS) * sb + (j % SS));

                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
            }
        }
        else
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
    requires
        VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
        VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(T alpha, PA a, PB b) noexcept
    {
        BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= RegisterCapacity_v<T>), "Not enough registers for ger()");

        IntrinsicType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = alpha * a(i * SS).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j)
        {
            IntrinsicType bx = (~b)(j).broadcast();

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(PA a, PB b) noexcept
    {
        BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= RegisterCapacity_v<T>), "Not enough registers for ger()");

        IntrinsicType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = a(i * SS, 0).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j)
        {
            IntrinsicType bx = (~b)(0, j).broadcast();

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <bool SOA, bool SOB>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        if (SOA == columnMajor && SOB == rowMajor)
        {
            BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= RegisterCapacity_v<T>), "Not enough registers for ger()");

            IntrinsicType ax[RM];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                ax[i] = alpha * blazefeo::load<aligned, SS>(a + i * sa);

            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n)
            {
                IntrinsicType bx = broadcast<SS>(b + (j / SS) * sb + (j % SS));

                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
            }
        }
        else
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
    requires
        VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
        VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(T alpha, PA a, PB b, size_t m, size_t n) noexcept
    {
        IntrinsicType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = alpha * a(i * SS).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n)
        {
            IntrinsicType bx = (~b)(j).broadcast();

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(PA a, PB b, size_t m, size_t n) noexcept
    {
        IntrinsicType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = a(i * SS, 0).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n)
        {
            IntrinsicType bx = (~b)(0, j).broadcast();

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::gemm(size_t K, T alpha, PA a, PB b) noexcept
    {
        auto bu = ~b;
        for (size_t k = 0; k < K; ++k)
        {
            ger(alpha, column(a), row(bu));
            a.hmove(1);
            bu.vmove(1);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PB, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::gemm(size_t K,
        T alpha, PA a, PB b, size_t md, size_t nd) noexcept
    {
        auto bu = ~b;
        for (size_t k = 0; k < K; ++k)
        {
            ger(alpha, column(a), row(bu), md, nd);
            a.hmove(1);
            bu.vmove(1);
        }
    }


    template <bool LeftSide, bool Upper, bool TransA, typename T, size_t M, size_t N, bool SO>
    inline void trsm(RegisterMatrix<T, M, N, SO>& ker, T const * a, size_t sa)
    {
        ker.template trsm<LeftSide, Upper, TransA>(a, sa);
    }


    template <typename T, size_t M, size_t N, bool SO>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::potrf()
    {
        static_assert(M >= N, "potrf() not implemented for register matrices with columns more than rows");
        static_assert(RM * RN + 2 <= RegisterCapacity_v<T>, "Not enough registers");

        #pragma unroll
        for (size_t k = 0; k < N; ++k)
        {
            #pragma unroll
            for (size_t j = 0; j < k; ++j)
            {
                T const a_kj = v_[k / SS][j][k % SS];

                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (i >= k / SS)
                    v_[i][k] = fnmadd(set1<SS>(a_kj), v_[i][j], v_[i][k]);
            }

            T const sqrt_a_kk = std::sqrt(v_[k / SS][k][k % SS]);

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
            {
                if (i < k / SS)
                    v_[i][k] = setzero<T, SS>();
                else
                    v_[i][k] /= sqrt_a_kk;
            }
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P1, typename P2>
        requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trmmLeftUpper(T alpha, P1 a, P2 b) noexcept
    {
        auto bu = ~b;

        #pragma unroll
        for (size_t k = 0; k < rows(); ++k)
        {
            IntrinsicType ax[RM];
            size_t const ii = (k + 1) / SS;
            size_t const rem = (k + 1) % SS;

            #pragma unroll
            for (size_t i = 0; i < ii; ++i)
                ax[i] = alpha * a(i * SS, 0).load();

            if (rem)
                ax[ii] = alpha * a(ii * SS, 0).maskLoad(SIMD::index() < rem);

            #pragma unroll
            for (size_t j = 0; j < N; ++j)
            {
                IntrinsicType bx = bu(0, j).broadcast();

                #pragma unroll
                for (size_t i = 0; i < ii; ++i)
                    v_[i][j] = fmadd(ax[i], bx, v_[i][j]);

                if (rem)
                    v_[ii][j] = fmadd(ax[ii], bx, v_[ii][j]);
            }

            a.hmove(1);
            bu.vmove(1);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PB, typename PA>
        requires MatrixPointer<PB, T> && (PB::storageOrder == columnMajor) && MatrixPointer<PA, T>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trmmRightLower(T alpha, PB b, PA a) noexcept
    {
        auto au = ~a;

        if constexpr (SO == columnMajor)
        {
            #pragma unroll
            for (size_t k = 0; k < N; ++k)
            {
                IntrinsicType bx[RM];

                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    bx[i] = alpha * b(i * SS, 0).load();

                #pragma unroll
                for (size_t j = 0; j <= k; ++j)
                {
                    IntrinsicType ax = au(0, j).broadcast();

                    #pragma unroll
                    for (size_t i = 0; i < RM; ++i)
                        v_[i][j] = fmadd(bx[i], ax, v_[i][j]);
                }

                b.hmove(1);
                au.vmove(1);
            }
        }
        else
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
        }
    }


    template <bool SOA, bool SOB, typename T, size_t M, size_t N, bool SO>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SO>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        ker.template ger<SOA, SOB>(alpha, a, sa, b, sb);
    }


    template <bool SOA, bool SOB, typename T, size_t M, size_t N, bool SO>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SO>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        ker.template ger<SOA, SOB>(alpha, a, sa, b, sb, m, n);
    }


    template <typename T, size_t M, size_t N, bool SO>
    // [[deprecated("Use load with a matrix argument instead")]]
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SO>& ker, T const * a, size_t sa)
    {
        ker.load(1., a, sa);
    }


    template <typename T, size_t M, size_t N, bool SO>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SO>& ker, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(1.0, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, bool SO>
    // [[deprecated("Use load with a matrix argument instead")]]
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SO>& ker, T beta, T const * a, size_t sa)
    {
        ker.load(beta, a, sa);
    }


    template <typename T, size_t M, size_t N, bool SO>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SO>& ker, T beta, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(beta, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, bool SO>
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SO> const& ker, T * a, size_t sa)
    {
        ker.store(a, sa);
    }


    template <typename T, size_t M, size_t N, bool SO>
    // [[deprecated("Use store with a matrix argument instead")]]
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SO> const& ker, T * a, size_t sa, size_t m, size_t n)
    {
        ker.store(a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2>
    inline bool operator==(RegisterMatrix<T, M, N, SO1> const& rm, Matrix<MT, SO2> const& m)
    {
        if (rows(m) != rm.rows() || columns(m) != rm.columns())
            return false;

        for (size_t i = 0; i < rm.rows(); ++i)
            for (size_t j = 0; j < rm.columns(); ++j)
                if (rm(i, j) != (*m)(i, j))
                    return false;

        return true;
    }


    template <typename MT, bool SO1, typename T, size_t M, size_t N, bool SO2>
    inline bool operator==(Matrix<MT, SO1> const& m, RegisterMatrix<T, M, N, SO2> const& rm)
    {
        return rm == m;
    }


    template <
        typename T, size_t M, size_t N, bool SO,
        typename PA, typename PB, typename PC, typename PD
    >
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
            && MatrixPointer<PB, T>
            && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, T alpha, PA a, PB b, T beta, PC c, PD d) noexcept
    {
        auto bu = ~b;
        ker.reset();

        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(a, bu);
            a.hmove(1);
            bu.vmove(1);
        }

        ker *= alpha;
        ker.axpy(beta, c);
        ker.store(d);
    }


    template <
        typename T, size_t M, size_t N, bool SO,
        typename PA, typename PB, typename PC, typename PD
    >
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
            && MatrixPointer<PB, T>
            && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, T alpha, PA a, PB b, T beta, PC c, PD d, size_t md, size_t nd) noexcept
    {
        auto bu = ~b;
        ker.reset();

        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(a, bu, md, nd);
            a.hmove(1);
            bu.vmove(1);
        }

        ker *= alpha;
        ker.axpy(beta, c, md, nd);
        ker.store(d, md, nd);
    }


    template <
        typename T, size_t M, size_t N, bool SO,
        typename PA, typename PB, typename PC, typename PD
    >
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
            && MatrixPointer<PB, T>
            && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, PA a, PB b, PC c, PD d) noexcept
    {
        ker.load(c);

        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(a, b);
            a.hmove(1);
            b.vmove(1);
        }

        ker.store(d);
    }


    template <
        typename T, size_t M, size_t N, bool SO,
        typename PA, typename PB, typename PC, typename PD
    >
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
            && MatrixPointer<PB, T>
            && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void gemm(RegisterMatrix<T, M, N, SO>& ker,
        size_t K, PA a, PB b, PC c, PD d, size_t md, size_t nd) noexcept
    {
        ker.load(c);

        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(a, b, md, nd);
            a.hmove(1);
            b.vmove(1);
        }

        ker.store(d, md, nd);
    }
}