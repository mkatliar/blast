// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/simd/MatrixPointer.hpp>
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


    /// @brief Register-resident matrix with dynamic size.
    ///
    /// @tparam T type of matrix elements
    /// @tparam M maximum number of rows of the matrix. Must be a multiple of SS.
    /// @tparam N maximum number of columns of the matrix.
    /// @tparam SO orientation of SIMD registers.
    template <typename T, size_t M, size_t N, bool SO = columnMajor>
    class DynamicRegisterMatrix
    :   public Matrix<DynamicRegisterMatrix<T, M, N, SO>, SO>
    {
    public:
        static_assert(SO == columnMajor, "Only column-major register matrices are currently supported");

        using BaseType = Matrix<DynamicRegisterMatrix<T, M, N, SO>, SO>;
        using BaseType::storageOrder;

        /// @brief Type of matrix elements
        using ElementType = T;
        using CompositeType = DynamicRegisterMatrix const&;              //!< Data type for composite expression templates.


        /// @brief Construct register matrix of specified size.
        DynamicRegisterMatrix(size_t m, size_t n)
        :   m_ {m}
        ,   n_ {n}
        {
            BLAZE_USER_ASSERT(m_ <= M && n_ <= N,
                "Invalid size of DynamicRegisterMatrix");

            reset();
        }


        /// @brief Copying prohibited
        DynamicRegisterMatrix(DynamicRegisterMatrix const&) = delete;


        /// @brief Assignment prohibited
        DynamicRegisterMatrix& operator=(DynamicRegisterMatrix const&) = delete;


        /// @brief Number of matrix rows
        size_t constexpr rows() const noexcept
        {
            return m_;
        }


        /// @brief Number of matrix columns
        size_t constexpr columns() const noexcept
        {
            return n_;
        }


        /// @brief Max number of matrix rows
        static size_t constexpr maxRows()
        {
            return M;
        }


        /// @brief Max number of matrix columns
        static size_t constexpr maxColumns()
        {
            return N;
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


        /// @brief R(0:m-1, 0:n-1) += beta * A
        template <typename PA>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        void axpy(T beta, PA a) noexcept
        {
            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n_)
                #pragma unroll
                for (size_t i = 0; i < RM; ++i) if (i * RM < m_)
                    v_[i][j] += beta * a.load(SS * i, j);
        }


        /// @brief Load from memory
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(P p) noexcept;


        /// @brief Load from memory
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void load(T beta, P p) noexcept;


        /// @brief Store matrix at location pointed by \a p
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void store(P p) const noexcept;


        /// @brief Store lower-triangular part of the matrix at location pointed by \a p.
        template <typename P>
            requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
        void storeLower(P p) const noexcept;


        /// @brief Rank-1 update with scalar coefficient
        ///
        /// R += alpha * column(a, 0) * row(b, 0)
        ///
        /// @param alpha scalar coefficient
        /// @param a pointer to the first element of the first matrix.
        /// @param b pointer to the first element of the second matrix.
        template <typename PA, typename PB>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
        void ger(T alpha, PA a, PB b) noexcept;


        /// @brief Rank-1 update
        ///
        /// R += column(a, 0) * row(b, 0)
        ///
        /// @param a pointer to the first element of the first matrix.
        /// @param b pointer to the first element of the second matrix.
        template <typename PA, typename PB>
            requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
        void ger(PA a, PB b) noexcept;


        /// @brief In-place Cholesky decomposition
        void potrf();


        /// @brief Triangular substitution
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
        BLAZE_STATIC_ASSERT_MSG((RM * RN <= RegisterCapacity_v<T>), "Not enough registers for a DynamicRegisterMatrix");

        IntrinsicType v_[RM][RN];
        size_t const m_;
        size_t const n_;


        /// @brief Reference to the matrix element at row \a i and column \a j
        T& at(size_t i, size_t j)
        {
            BLAZE_USER_ASSERT(i < m_ && j < n_, "Invalid index of DynamicRegisterMatrix");
            return v_[i / SS][j][i % SS];
        }


        /// @brief Value of the matrix element at row \a i and column \a j
        T at(size_t i, size_t j) const
        {
            BLAZE_USER_ASSERT(i < m_ && j < n_, "Invalid index of DynamicRegisterMatrix");
            return v_[i / SS][j][i % SS];
        }
    };


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void DynamicRegisterMatrix<T, M, N, SO>::load(P p) noexcept
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n_)
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = p(SS * i, j).load();
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void DynamicRegisterMatrix<T, M, N, SO>::load(T beta, P p) noexcept
    {
        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n_)
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] = beta * p.load(SS * i, j);
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void DynamicRegisterMatrix<T, M, N, SO>::store(P p) const noexcept
    {
        // The compile-time constant size of the j loop in combination with the if() expression
        // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
        for (size_t j = 0; j < N; ++j) if (j < n_)
            for (size_t i = 0; i < RM; ++i) if (SS * (i + 1) <= m_)
                p(SS * i, j).store(v_[i][j]);

        if (IntType const rem = m_ % SS)
        {
            MaskType const mask = SIMD::index() < rem;
            size_t const i = m_ / SS;

            for (size_t j = 0; j < n_ && j < columns(); ++j)
                p(SS * i, j).maskStore(mask, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename P>
        requires MatrixPointer<P, T> && (P::storageOrder == columnMajor)
    inline void DynamicRegisterMatrix<T, M, N, SO>::storeLower(P p) const noexcept
    {
        for (size_t j = 0; j < N; ++j) if (j < n_)
        {
            for (size_t ri = j / SS; ri < RM; ++ri)
            {
                IntType const skip = j - ri * SS;
                IntType const rem = m_ - ri * SS;
                MaskType mask = SIMD::index() < rem;

                if (skip > 0)
                    mask &= SIMD::index() >= skip;

                p(SS * ri, j).maskStore(mask, v_[ri][j]);
            }
        }
    }


    // template <typename T, size_t M, size_t N, bool SO>
    // template <typename P>
    //     requires MatrixPointer<P, T>
    // BLAZE_ALWAYS_INLINE void DynamicRegisterMatrix<T, M, N, SO>::trsmRightUpper(P a)
    // {
    //     if constexpr (SO == columnMajor)
    //     {
    //         #pragma unroll
    //         for (size_t j = 0; j < N; ++j)
    //         {
    //             #pragma unroll
    //             for (size_t k = 0; k < j; ++k)
    //             {
    //                 IntrinsicType const a_kj = a.broadcast(k, j);

    //                 #pragma unroll
    //                 for (size_t i = 0; i < RM; ++i)
    //                     v_[i][j] = fnmadd(a_kj, v_[i][k], v_[i][j]);
    //             }

    //             IntrinsicType const a_jj = a.broadcast(j, j);

    //             #pragma unroll
    //             for (size_t i = 0; i < RM; ++i)
    //                 v_[i][j] /= a_jj;
    //         }
    //     }
    //     else
    //     {
    //         BLAZE_THROW_LOGIC_ERROR("Not implemented");
    //     }
    // }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
    BLAZE_ALWAYS_INLINE void DynamicRegisterMatrix<T, M, N, SO>::ger(T alpha, PA a, PB b) noexcept
    {
        IntrinsicType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i) // TODO: !!! check i against m
            ax[i] = alpha * a(i * SS, 0).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n_)
        {
            IntrinsicType bx = b(0, j).broadcast();

            #pragma unroll
            for (size_t i = 0; i < RM; ++i) // TODO: !!! check i against m
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    template <typename T, size_t M, size_t N, bool SO>
    template <typename PA, typename PB>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor) && MatrixPointer<PB, T>
    BLAZE_ALWAYS_INLINE void DynamicRegisterMatrix<T, M, N, SO>::ger(PA a, PB b) noexcept
    {
        IntrinsicType ax[RM];

        #pragma unroll
        for (size_t i = 0; i < RM; ++i) // TODO: !!! check i against m
            ax[i] = a(i * SS, 0).load();

        #pragma unroll
        for (size_t j = 0; j < N; ++j) if (j < n_)
        {
            IntrinsicType bx = b(0, j).broadcast();

            #pragma unroll
            for (size_t i = 0; i < RM; ++i) // TODO: !!! check i against m
                v_[i][j] = fmadd(ax[i], bx, v_[i][j]);
        }
    }


    // template <typename T, size_t M, size_t N, bool SO>
    // BLAZE_ALWAYS_INLINE void DynamicRegisterMatrix<T, M, N, SO>::potrf()
    // {
    //     static_assert(M >= N, "potrf() not implemented for register matrices with columns more than rows");
    //     static_assert(RM * RN + 2 <= RegisterCapacity_v<T>, "Not enough registers");

    //     #pragma unroll
    //     for (size_t k = 0; k < N; ++k)
    //     {
    //         #pragma unroll
    //         for (size_t j = 0; j < k; ++j)
    //         {
    //             T const a_kj = v_[k / SS][j][k % SS];

    //             #pragma unroll
    //             for (size_t i = 0; i < RM; ++i) if (i >= k / SS)
    //                 v_[i][k] = fnmadd(set1<SS>(a_kj), v_[i][j], v_[i][k]);
    //         }

    //         T const sqrt_a_kk = std::sqrt(v_[k / SS][k][k % SS]);

    //         #pragma unroll
    //         for (size_t i = 0; i < RM; ++i)
    //         {
    //             if (i < k / SS)
    //                 v_[i][k] = setzero<T, SS>();
    //             else
    //                 v_[i][k] /= sqrt_a_kk;
    //         }
    //     }
    // }


    // template <typename T, size_t M, size_t N, bool SO>
    // template <typename P1, typename P2>
    //     requires MatrixPointer<P1, T> && (P1::storageOrder == columnMajor) && MatrixPointer<P2, T>
    // BLAZE_ALWAYS_INLINE void DynamicRegisterMatrix<T, M, N, SO>::trmmLeftUpper(T alpha, P1 a, P2 b) noexcept
    // {
    //     #pragma unroll
    //     for (size_t k = 0; k < rows(); ++k)
    //     {
    //         IntrinsicType ax[RM];
    //         size_t const ii = (k + 1) / SS;
    //         size_t const rem = (k + 1) % SS;

    //         #pragma unroll
    //         for (size_t i = 0; i < ii; ++i)
    //             ax[i] = alpha * a.load(i * SS, 0);

    //         if (rem)
    //             ax[ii] = alpha * a.maskLoad(ii * SS, 0, SIMD::index() < rem);

    //         #pragma unroll
    //         for (size_t j = 0; j < N; ++j)
    //         {
    //             IntrinsicType bx = b.broadcast(0, j);

    //             #pragma unroll
    //             for (size_t i = 0; i < ii; ++i)
    //                 v_[i][j] = fmadd(ax[i], bx, v_[i][j]);

    //             if (rem)
    //                 v_[ii][j] = fmadd(ax[ii], bx, v_[ii][j]);
    //         }

    //         a.hmove(1);
    //         b.vmove(1);
    //     }
    // }


    // template <typename T, size_t M, size_t N, bool SO>
    // template <typename PB, typename PA>
    //     requires MatrixPointer<PB, T> && (PB::storageOrder == columnMajor) && MatrixPointer<PA, T>
    // BLAZE_ALWAYS_INLINE void DynamicRegisterMatrix<T, M, N, SO>::trmmRightLower(T alpha, PB b, PA a) noexcept
    // {
    //     if constexpr (SO == columnMajor)
    //     {
    //         // for (size_t j = 0; j < columns(); ++j)
    //         // {
    //         //     ger(alpha, b.offset(0, j), a.offset(j, 0));
    //         // }

    //         #pragma unroll
    //         for (size_t k = 0; k < N; ++k)
    //         {
    //             IntrinsicType bx[RM];

    //             #pragma unroll
    //             for (size_t i = 0; i < RM; ++i)
    //                 bx[i] = alpha * b.load(i * SS, 0);

    //             #pragma unroll
    //             for (size_t j = 0; j <= k; ++j)
    //             {
    //                 IntrinsicType ax = a.broadcast(0, j);

    //                 #pragma unroll
    //                 for (size_t i = 0; i < RM; ++i)
    //                     v_[i][j] = fmadd(bx[i], ax, v_[i][j]);
    //             }

    //             b.hmove(1);
    //             a.vmove(1);
    //         }
    //     }
    //     else
    //     {
    //         BLAZE_THROW_LOGIC_ERROR("Not implemented");
    //     }
    // }


    template <typename T, size_t M, size_t N, bool SO1, typename MT, bool SO2>
    inline bool operator==(DynamicRegisterMatrix<T, M, N, SO1> const& rm, Matrix<MT, SO2> const& m)
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
    inline bool operator==(Matrix<MT, SO1> const& m, DynamicRegisterMatrix<T, M, N, SO2> const& rm)
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
    BLAZE_ALWAYS_INLINE void gemm(DynamicRegisterMatrix<T, M, N, SO>& ker,
        size_t K, T alpha, PA a, PB b, T beta, PC c, PD d) noexcept
    {
        ker.reset();

        for (size_t k = 0; k < K; ++k)
        {
            ker.ger(a, b);
            a.hmove(1);
            b.vmove(1);
        }

        ker *= alpha;
        ker.axpy(beta, c);
        ker.store(d);
    }
}