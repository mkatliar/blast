// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/Simd.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/math/RowColumnVectorPointer.hpp>
#include <blast/math/Side.hpp>
#include <blast/math/UpLo.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/util/Types.hpp>
#include <blast/util/Exception.hpp>
#include <blast/system/Inline.hpp>

#include <stdexcept>
#include <type_traits>


namespace blast
{
    /// @brief Register-resident matrix
    ///
    /// The RegisterMatrix class provides basic linear algebra operations that can be performed
    /// on a register-resident matrix. Sizes of all loops are known at compile-time
    /// and the functions can be force-inlined. Optimized manually-written specializations of the RegisterMatrix
    /// functions can also be provided.
    ///
    /// @tparam T type of matrix elements
    /// @tparam M number of rows of the matrix. Must be a multiple of SS.
    /// @tparam N number of columns of the matrix.
    /// @tparam SO orientation of SIMD registers.
    ///
    template <typename T, size_t M, size_t N, bool SO = columnMajor>
    class RegisterMatrix
    {
    public:
        static_assert(SO == columnMajor, "Only column-major register matrices are currently supported");

        // TODO: change bool to StorageOrder
        static constexpr bool storageOrder = SO;

        /// @brief Type of matrix elements
        using ElementType = T;


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
        ///
        /// TODO: do we need it? deprecate?
        ///
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
        ///
        /// TODO: do we need it? deprecate?
        ///
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
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                #pragma unroll
                for (size_t j = 0; j < N; ++j)
                    v_[i][j].reset();
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
            SimdVecType const beta_simd {beta};

            #pragma unroll
            for (size_t j = 0; j < N; ++j)
                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] = fmadd(beta_simd, a(SS * i, j).load(), v_[i][j]);
        }


        /// @brief R(0:m-1, 0:n-1) += beta * A
        template <typename PA>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        void axpy(T beta, PA a, size_t m, size_t n) noexcept
        {
            SimdVecType const beta_simd {beta};

            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n)
                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (SS * i < m)
                    v_[i][j] = fmadd(beta_simd, a(SS * i, j).load(), v_[i][j]);
        }


        template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(P p) noexcept;


        template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(T beta, P p) noexcept;

        /**
         * @brief Load and multiply a matrix of specified size.
         *
         * @tparam P matrix pointer type
         *
         * @param beta multiplier
         * @param p matrix pointer to load from
         * @param m number of rows to load
         * @param n number of columns to load
         */
        template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(T beta, P p, size_t m, size_t n) noexcept;


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


        /// @brief store with specified size
        ///
        /// @tparam P matrix pointer type
        ///
        /// @param p matrix pointer to store to
        /// @param m number of rows to store
        /// @param n number of columns to store
        ///
        template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void store(P p, size_t m, size_t n) const noexcept;


        /// @brief Rank-1 update with multiplier
        ///
        /// m(i, j) += alpha * a(i) * b(j)
        /// for i=0...rows()-1, j=0...columns()-1
        ///
        /// @tparam PA first vector pointer type
        /// @tparam PB second vector pointer type
        ///
        /// @param alpha multiplier
        /// @param a column vector for rank-1 update
        /// @param b row vector for rank-1 update
        ///
        template <typename PA, typename PB>
        requires
            VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
            VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
        void ger(T alpha, PA a, PB b) noexcept;


        /// @brief Rank-1 update
        ///
        /// m(i, j) += a(i) * b(j)
        /// for i=0...rows()-1, j=0...columns()-1
        ///
        /// @tparam PA first vector pointer type
        /// @tparam PB second vector pointer type
        ///
        /// @param a column vector for rank-1 update
        /// @param b row vector for rank-1 update
        ///
        template <typename PA, typename PB>
        requires
            VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
            VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
        void ger(PA a, PB b) noexcept;


        /// @brief Rank-1 update of specified size with multiplier
        ///
        /// m(i, j) += alpha * a(i) * b(j)
        /// for i=0...m-1, j=0...n-1
        ///
        /// @tparam PA first vector pointer type
        /// @tparam PB second vector pointer type
        ///
        /// @param alpha multiplier
        /// @param a column vector for rank-1 update
        /// @param b row vector for rank-1 update
        ///
        template <typename PA, typename PB>
        requires
            VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
            VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
        void ger(T alpha, PA a, PB b, size_t m, size_t n) noexcept;


        /// @brief Rank-1 update of specified size
        ///
        /// m(i, j) += a(i) * b(j)
        /// for i=0...m-1, j=0...n-1
        ///
        /// @tparam PA first vector pointer type
        /// @tparam PB second vector pointer type
        ///
        /// @param alpha multiplier
        /// @param a column vector for rank-1 update
        /// @param b row vector for rank-1 update
        ///
        template <typename PA, typename PB>
        requires
            VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
            VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
        void ger(PA a, PB b, size_t m, size_t n) noexcept;


        /// @brief In-place Cholesky decomposition
        void potrf() noexcept;


        /// @brief Triangular substitution
        ///
        /// Solves
        /// X * A = B
        /// or
        /// A * X = B
        ///
        /// where A is either upper-triangular or lower-triangular.
        ///
        /// On entry, the register matrix contains the matrix B.
        /// On exit, the register matrix contains the solution matrix X.
        ///
        /// @param side specifies whether A appears on the left or right of X
        /// @param uplo specifies whether the matrix A is an upper or lower triangular matrix
        /// @param pointer pointer to matrix A. Depending on the @a uplo flag, onlny the diagonal
        ///     and the upper- or lower-triangular elements of A are referenced.
        ///
        template <typename P>
        requires MatrixPointer<P, T>
        void trsm(Side side, UpLo uplo, P A) noexcept;


        /// @brief Left multiplication with a triangular matrix
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
        /// @param alpha the scalar multiplier
        /// @param a triangular matrix
        /// @param uplo specifies whether the matrix A is an upper or lower triangular
        /// @param diagonal_unit specifies whether or not A is unit triangular
        /// @param b general matrix.
        ///
        template <typename P1, typename P2>
        requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
        void trmm(T alpha, P1 a, UpLo uplo, bool diagonal_unit, P2 b) noexcept;


        /// @brief Right multiplication with a triangular matrix
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
        /// @param b general matrix
        /// @param a triangular matrix
        /// @param uplo specifies whether the matrix A is an upper or lower triangular
        /// @param diagonal_unit specifies whether or not A is unit triangular
        ///
        template <typename P1, typename P2>
        requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
        void trmm(T alpha, P1 b, P2 a, UpLo uplo, bool diagonal_unit) noexcept;


        /// @brief Right multiplication with a triangular submatrix
        ///
        /// Performs the matrix-matrix operation
        ///
        /// R += alpha*B(0..m-1, 0..n-1)*A(0..n-1, 0..n-1),
        ///
        /// where alpha is a scalar, B is a general matrix,
        /// and A is a lower triangular matrix.
        ///
        /// @tparam P1 matrix A pointer type.
        /// @tparam P2 matrix B pointer type.
        ///
        /// @param b general matrix
        /// @param a triangular matrix
        /// @param uplo specifies whether the matrix A is an upper or lower triangular
        /// @param diagonal_unit specifies whether or not A is unit triangular
        ///
        template <typename PB, typename PA>
        requires MatrixPointer<PB, T> && (PB::storageOrder == columnMajor) && MatrixPointer<PA, T>
        void trmm(T alpha, PB b, PA a, UpLo uplo, bool diagonal_unit, size_t m, size_t n) noexcept;


    private:
        using Arch = xsimd::default_arch;
        using SimdVecType = SimdVec<T, Arch>;
        using IntrinsicType = typename SimdVecType::IntrinsicType;
        using MaskType = SimdMask<T, Arch>;
        using IntType = typename SimdIndex<T, Arch>::value_type;

        // SIMD size
        static size_t constexpr SS = SimdVecType::size();

        // Numberf of SIMD registers required to store a single column of the matrix.
        static size_t constexpr RM = M / SS;
        static size_t constexpr RN = N;

        static_assert(RM > 0, "Number of rows must be not less than SIMD size");
        static_assert(RN > 0, "Number of columns must be positive");
        static_assert(M % SS == 0, "Number of rows must be a multiple of SIMD size");
        static_assert(RM * RN <= registerCapacity(Arch {}), "Not enough registers for a RegisterMatrix");

        SimdVecType v_[RM][RN];


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


    /**
     * @brief Specialization for @a RegisterMatrix
     *
     * @tparam T type of matrix elements
     * @tparam M number of rows of the matrix. Must be a multiple of SS.
     * @tparam N number of columns of the matrix.
     * @tparam SO orientation of SIMD registers.
     */
    template <typename T, size_t M, size_t N, bool SO>
    struct StorageOrderHelper<RegisterMatrix<T, M, N, SO>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};


    // TODO: deprecate
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
    inline size_t constexpr rows(RegisterMatrix<T, M, N, SO> const& m) noexcept
    {
        return m.rows();
    }


    template <typename T, size_t M, size_t N, bool SO>
    inline size_t constexpr columns(RegisterMatrix<T, M, N, SO> const& m) noexcept
    {
        return m.columns();
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
                    v_[m / SS][j] = beta * p(m - rem, j).load(indexSequence<T, Arch>() < rem);
            }
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
    requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<T, M, N, SO>::store(P p) const noexcept
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j)
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                p(SS * i, j).store(v_[i][j]);
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
            MaskType const mask = indexSequence<T, Arch>() < rem;
            size_t const i = m / SS;

            for (size_t j = 0; j < n && j < columns(); ++j)
                p(SS * i, j).store(v_[i][j], mask);
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
                MaskType const mask = indexSequence<T, Arch>() >= skip;
                p(SS * ri, j).store(v_[ri][j], mask);
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
                MaskType mask = indexSequence<T, Arch>() < rem;

                if (skip > 0)
                    mask &= indexSequence<T, Arch>() >= skip;

                p(SS * ri, j).store(v_[ri][j], mask);
            }
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
    requires MatrixPointer<P, T>
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trsm(Side side, UpLo uplo, P A) noexcept
    {
        if constexpr (SO == columnMajor)
        {
            if (side == Side::Right && uplo == UpLo::Upper)
            {
                #pragma unroll
                for (size_t j = 0; j < N; ++j)
                {
                    #pragma unroll
                    for (size_t k = 0; k < j; ++k)
                    {
                        SimdVecType const a_kj = A[k, j];

                        #pragma unroll
                        for (size_t i = 0; i < RM; ++i)
                            v_[i][j] = fnmadd(a_kj, v_[i][k], v_[i][j]);
                    }

                    SimdVecType const a_jj = A[j, j];

                    #pragma unroll
                    for (size_t i = 0; i < RM; ++i)
                        v_[i][j] /= a_jj;
                }
            }
            else
            {
                // Not implemented
                assert(false);
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
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(T alpha, PA a, PB b) noexcept
    {
        BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= registerCapacity(Arch {})), "Not enough registers for ger()");

        SimdVecType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = alpha * a(i * SS).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j)
        {
            SimdVecType const bx = b[j];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
    requires
        VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
        VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(PA a, PB b) noexcept
    {
        BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= registerCapacity(Arch {})), "Not enough registers for ger()");

        SimdVecType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = a(i * SS).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j)
        {
            SimdVecType const bx = b[j];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
    requires
        VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
        VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(T alpha, PA a, PB b, size_t m, size_t n) noexcept
    {
        SimdVecType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = alpha * a(i * SS).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n)
        {
            SimdVecType const bx = b[j];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
    requires
        VectorPointer<PA, T> && (PA::transposeFlag == columnVector) &&
        VectorPointer<PB, T> && (PB::transposeFlag == rowVector)
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::ger(PA a, PB b, size_t m, size_t n) noexcept
    {
        SimdVecType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            ax[i] = a(i * SS).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n)
        {
            SimdVecType const bx = b[j];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::potrf() noexcept
    {
        static_assert(M >= N, "potrf() not implemented for register matrices with columns more than rows");
        static_assert(RM * RN + 2 <= registerCapacity(Arch {}), "Not enough registers");

        #pragma unroll
        for (size_t k = 0; k < N; ++k)
        {
            #pragma unroll
            for (size_t j = 0; j < k; ++j)
            {
                SimdVecType const a_kj = v_[k / SS][j][k % SS];

                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (i >= k / SS)
                    v_[i][k] = fnmadd(a_kj, v_[i][j], v_[i][k]);
            }

            T const sqrt_a_kk = std::sqrt(v_[k / SS][k][k % SS]);

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
            {
                if (i < k / SS)
                    v_[i][k].reset();
                else
                    v_[i][k] /= sqrt_a_kk;
            }
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P1, typename P2>
    requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trmm(T alpha, P1 a, UpLo uplo, bool diagonal_unit, P2 b) noexcept
    {
        if (diagonal_unit)
            BLAST_THROW_EXCEPTION(std::logic_error {"Unit diagonal matrices support not implemented in RegisterMatrix::trmm()"});

        if (uplo == UpLo::Upper)
        {
            auto bu = ~b;

            #pragma unroll
            for (size_t k = 0; k < rows(); ++k)
            {
                SimdVecType ax[RM];
                size_t const ii = (k + 1) / SS;
                size_t const rem = (k + 1) % SS;

                #pragma unroll
                for (size_t i = 0; i < ii; ++i)
                    ax[i] = alpha * a(i * SS, 0).load();

                if (rem)
                    ax[ii] = alpha * a(ii * SS, 0).load(indexSequence<T, Arch>() < rem);

                #pragma unroll
                for (size_t j = 0; j < N; ++j)
                {
                    SimdVecType const bx = bu[0, j];

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
        else
        {
            BLAST_THROW_EXCEPTION(std::logic_error {"Left multiplication with lower-triangular matrix not implemented in RegisterMatrix::trmm()"});
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PB, typename PA>
    requires MatrixPointer<PB, T> && (PB::storageOrder == columnMajor) && MatrixPointer<PA, T>
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trmm(T alpha, PB b, PA a, UpLo uplo, bool diagonal_unit) noexcept
    {
        if (diagonal_unit)
            BLAST_THROW_EXCEPTION(std::logic_error {"Unit diagonal matrices support not implemented in RegisterMatrix::trmm()"});

        if (uplo == UpLo::Lower)
        {
            auto au = ~a;

            if constexpr (SO == columnMajor)
            {
                #pragma unroll
                for (size_t k = 0; k < N; ++k)
                {
                    SimdVecType bx[RM];

                    #pragma unroll
                    for (size_t i = 0; i < RM; ++i)
                        bx[i] = alpha * b(i * SS, 0).load();

                    #pragma unroll
                    for (size_t j = 0; j <= k; ++j)
                    {
                        SimdVecType const ax = au[0, j];

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
        else
        {
            BLAST_THROW_EXCEPTION(std::logic_error {"Right multiplication with upper-triangular matrix not implemented in RegisterMatrix::trmm()"});
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PB, typename PA>
    requires MatrixPointer<PB, T> && (PB::storageOrder == columnMajor) && MatrixPointer<PA, T>
    BLAST_ALWAYS_INLINE void RegisterMatrix<T, M, N, SO>::trmm(T alpha, PB b, PA a, UpLo uplo, bool diagonal_unit, size_t m, size_t n) noexcept
    {
        // NOTE: this implementation uses unmasked loads from the matrix a,
        // and therefore will access rows of a beyond m-1.
        // This will result in undefined behavior on unpadded matrices.
        auto au = ~a;

        if constexpr (SO == columnMajor)
        {
            #pragma unroll
            for (size_t k = 0; k < N; ++k) if (k < n)
            {
                SimdVecType bx[RM];

                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    bx[i] = alpha * b(i * SS, 0).load();

                #pragma unroll
                for (size_t j = 0; j <= k; ++j)
                {
                    SimdVecType const ax = au[0, j];

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


    template <typename T, size_t M, size_t N, bool SO1, Matrix MT>
    inline bool operator==(RegisterMatrix<T, M, N, SO1> const& rm, MT const& m)
    {
        if (rows(m) != rm.rows() || columns(m) != rm.columns())
            return false;

        for (size_t i = 0; i < rm.rows(); ++i)
            for (size_t j = 0; j < rm.columns(); ++j)
                if (rm(i, j) != m(i, j))
                    return false;

        return true;
    }


    template <Matrix MT, typename T, size_t M, size_t N, bool SO2>
    inline bool operator==(MT const& m, RegisterMatrix<T, M, N, SO2> const& rm)
    {
        return rm == m;
    }
}
