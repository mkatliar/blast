#pragma once

#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/panel/Forward.hpp>

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


    /// @brief Regiter-resident matrix
    ///
    /// @tparam T type of matrix elements
    /// @tparam M number of rows of the matrix. Must be a multiple of SS.
    /// @tparam N number of columns of the matrix.
    /// @tparam SS number of T elements that can be stored in a SIMD register.
    template <typename T, size_t M, size_t N, size_t SS>
    class RegisterMatrix
    :   public Matrix<RegisterMatrix<T, M, N, SS>, columnMajor>
    {
    public:
        /// @brief Type of matrix elements
        using ElementType = T;
        using CompositeType = RegisterMatrix const&;              //!< Data type for composite expression templates.


        /// @brief Default ctor
        RegisterMatrix()
        {
            reset();
        }


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
        T operator()(size_t i, size_t j) const
        {
            return at(i, j);
        }


        /// @brief Set all elements to 0.
        void reset()
        {
            for (size_t i = 0; i < RM; ++i)
                for (size_t j = 0; j < N; ++j)
                    v_[i][j] = setzero<T, SS>();
        }


        /// @brief load from memory
        // [[deprecated("Use load with a matrix argument instead")]]
        void load(T beta, T const * ptr, size_t spacing);


        template <typename MT>
        void load(T beta, PanelMatrix<MT, columnMajor> const& A, size_t i, size_t j)
        {
            BLAZE_INTERNAL_ASSERT(i + rows() <= (~A).rows(), "Invalid submatrix specification");
            BLAZE_INTERNAL_ASSERT(j + columns() <= (~A).columns(), "Invalid submatrix specification");
            BLAZE_INTERNAL_ASSERT((i + (~A).offset()) % SS == 0, "Invalid submatrix specification");

            size_t const p = (i + (~A).offset()) / SS;

            for (size_t ri = 0; ri < RM; ++ri)
                for (size_t rj = 0; rj < RN; ++rj)
                    v_[ri][rj] = beta * blazefeo::load<SS>(data(A) + (p + ri) * spacing(A) + SS * rj);
        }

        
        /// @brief Load register matrix from a submatrix of a memory matrix and multiply it by a constant;
        ///
        /// The requested size size (\a m,\a n) must be exceed the size of the register matrix.
        /// If the requested size is smaller than the register matrix, the register matrix is partially loaded,
        /// and the remaining elements are undefined.
        ///
        /// @param beta multiplication constant
        /// @param A memory matrix
        /// @param i0 index of the first row of the submatrix to be loaded
        /// @param j0 index of the first column of the submatrix to be loaded
        /// @param m number of rows to be loaded
        /// @param n number of columns to be loaded
        template <typename MT>
        void load(T beta, PanelMatrix<MT, columnMajor> const& A, size_t i, size_t j, size_t m, size_t n)
        {
            BLAZE_INTERNAL_ASSERT(i + m <= (~A).rows(), "Invalid submatrix specification");
            BLAZE_INTERNAL_ASSERT(j + n <= (~A).columns(), "Invalid submatrix specification");
            BLAZE_INTERNAL_ASSERT((i + (~A).offset()) % SS == 0, "Invalid submatrix specification");
            BLAZE_INTERNAL_ASSERT(m <= rows(), "Invalid number of rows");
            BLAZE_INTERNAL_ASSERT(n <= columns(), "Invalid number of columns");

            size_t const p = (i + (~A).offset()) / SS;

            for (size_t ri = 0; ri * SS < m; ++ri)
                for (size_t rj = 0; rj < n; ++rj)
                    v_[ri][rj] = beta * blazefeo::load<SS>(data(A) + (p + ri) * spacing(A) + SS * rj);
        }


        /// @brief load from memory with specified size
        void load(T beta, T const * ptr, size_t spacing, size_t m, size_t n);


        /// @brief store to memory
        void store(T * ptr, size_t spacing) const;


        /// @brief store to memory with specified size
        // [[deprecated("Use load with a matrix argument instead")]]
        void store(T * ptr, size_t spacing, size_t m, size_t n) const;


        /// @brief Store register matrix to a memory panel matrix;
        ///
        /// The size of the memory matrix \a A must be not smaller than the size of the register matrix.
        /// If the memory matrix is smaller than the register matrix, the register matrix is partially stored,
        /// and the remaining elements of \a A are unchanged.
        ///
        /// @param A memory matrix
        template <typename MT>
        void store(PanelMatrix<MT, columnMajor>& A) const
        {
            BLAZE_INTERNAL_ASSERT((~A).rows() <= rows(), "Invalid matrix size");
            BLAZE_INTERNAL_ASSERT((~A).columns() <= columns(), "Invalid matrix size");

            size_t const m = (~A).rows();
            size_t const n = (~A).columns();

            for (size_t i = 0; i < RM; ++i) if (SS * i < m)
                // The compile-time constant size of the j loop in combination with the if() expression
                // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
                for (size_t j = 0; j < N; ++j) if (j < n)
                    (~A).store(SS * i, j, v_[i][j]);
        }


        /// @brief Store register matrix to a memory dense matrix;
        ///
        /// The size of the memory matrix \a A must be not smaller than the size of the register matrix.
        /// If the memory matrix is smaller than the register matrix, the register matrix is partially stored,
        /// and the remaining elements of \a A are unchanged.
        ///
        /// @param A memory matrix
        template <typename MT>
        void store(DenseMatrix<MT, columnMajor>& A) const
        {
            size_t const m = (~A).rows();
            size_t const n = (~A).columns();
            size_t const sp = spacing(A);
            auto * ptr = data(A);

            BLAZE_STATIC_ASSERT_MSG((RM * RN + 2 <= RegisterCapacity_v<T, SS>), "Not enough registers");
            BLAZE_INTERNAL_ASSERT(m > M - SS && m <= M, "Invalid number of rows in partial store");
            BLAZE_INTERNAL_ASSERT(n > 0 && n <= N, "Invalid number of columns in partial store");
            // BLAZE_INTERNAL_ASSERT(m < M || n < N, "Partial store with full size");

            if (IntType const rem = m % SS)
            {
                #pragma unroll
                for (size_t i = 0; i < RM - 1; ++i)
                    // The compile-time constant size of the j loop in combination with the if() expression
                    // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
                    #pragma unroll
                    for (size_t j = 0; j < N; ++j) if (j < n)
                        blazefeo::store(ptr + SS * i + sp * j, v_[i][j]);
                        
                MaskType const mask = cmpgt<SS>(set1<SS>(rem), countUp<MaskType, SS>());
                size_t constexpr i = RM - 1;
            
                #pragma unroll
                for (size_t j = 0; j < N; ++j) if (j < n)
                    maskstore(ptr + SS * i + sp * j, mask, v_[i][j]);
            }
            else
            {
                #pragma unroll
                for (size_t i = 0; i < RM; ++i)
                    // The compile-time constant size of the j loop in combination with the if() expression
                    // prevent Clang from emitting memcpy() call here and produce good enough code with the loop unrolled.
                    #pragma unroll
                    for (size_t j = 0; j < N; ++j) if (j < n)
                        blazefeo::store(ptr + SS * i + sp * j, v_[i][j]);
            }
        }


        template <typename MT>
        void store(DenseMatrix<MT, columnMajor>&& A) const
        {
            this->store(~A);
        }


        /// @brief Store register matrix to a submatrix of a memory panel matrix;
        ///
        /// The size (m, n) of the submatrix be not bigger than the size of the register matrix.
        /// If (m, n) is smaller than the register matrix, the register matrix is partially stored,
        /// and the remaining elements of \a A are unchanged.
        ///
        /// @param A memory matrix
        /// @param i first row of A to store the register matrix to
        /// @param j first column of A to store the register matrix to
        /// @param m number of rows to store
        /// @param n number of columns to store
        template <typename MT>
        void store(PanelMatrix<MT, columnMajor>& A, size_t i, size_t j, size_t m, size_t n) const
        {
            BLAZE_INTERNAL_ASSERT((~A).rows() <= rows(), "Invalid matrix size");
            BLAZE_INTERNAL_ASSERT((~A).columns() <= columns(), "Invalid matrix size");

            auto sub_A = submatrix(~A, i, j, m, n);
            store(sub_A);
        }


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


        /// @brief In-place Cholesky decomposition
        void potrf();


        /// @brief Triangular substitution
        template <bool LeftSide, bool Upper, bool TransA>
        void trsm(T const * l, size_t sl);


    private:
        using IntrinsicType = typename Simd<T, SS>::IntrinsicType;
        using MaskType = typename Simd<T, SS>::MaskType;
        using IntType = typename Simd<T, SS>::IntType;

        // Numberf of SIMD registers required to store a single column of the matrix.
        static size_t constexpr RM = M / SS;
        static size_t constexpr RN = N;

        BLAZE_STATIC_ASSERT_MSG((RM > 0), "Number of rows must be not less than SIMD size");
        BLAZE_STATIC_ASSERT_MSG((RN > 0), "Number of columns must be positive");
        BLAZE_STATIC_ASSERT_MSG((M % SS == 0), "Number of rows must be a multiple of SIMD size");
        BLAZE_STATIC_ASSERT_MSG((RM * RN <= RegisterCapacity_v<T, SS>), "Not enough registers for a RegisterMatrix");
        
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


    template <typename T, size_t M, size_t N, size_t SS>
    struct RegisterMatrixTraits<RegisterMatrix<T, M, N, SS>>
    {
        static size_t constexpr simdSize = SS;
        static size_t constexpr rows = M;
        static size_t constexpr columns = N;
        static size_t constexpr elementCount = rows * columns;
        
        using ElementType = T;
    };


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::load(T beta, T const * ptr, size_t spacing)
    {
        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            #pragma unroll
            for (size_t j = 0; j < N; ++j)
                v_[i][j] = beta * blazefeo::load<SS>(ptr + spacing * i + SS * j);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::load(T beta, T const * ptr, size_t spacing, size_t m, size_t n)
    {
        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            #pragma unroll
            for (size_t j = 0; j < N; ++j) if (j < n)
                v_[i][j] = beta * blazefeo::load<SS>(ptr + spacing * i + SS * j);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::store(T * ptr, size_t spacing) const
    {
        #pragma unroll
        for (size_t i = 0; i < RM; ++i)
            #pragma unroll
            for (size_t j = 0; j < N; ++j)
                blazefeo::store(ptr + spacing * i + SS * j, v_[i][j]);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    inline void RegisterMatrix<T, M, N, SS>::store(T * ptr, size_t spacing, size_t m, size_t n) const
    {
        BLAZE_STATIC_ASSERT_MSG((RM * RN + 2 <= RegisterCapacity_v<T, SS>), "Not enough registers");
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
                    blazefeo::store(ptr + spacing * i + SS * j, v_[i][j]);
                    
            MaskType const mask = cmpgt<SS>(set1<SS>(rem), countUp<MaskType, SS>());
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
                    blazefeo::store(ptr + spacing * i + SS * j, v_[i][j]);
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
                for (size_t i = 0; i < RM; ++i)
                    v_[i][j] = fnmadd(l_jk, v_[i][k], v_[i][j]);
            }

            IntrinsicType const l_jj = broadcast<SS>(l + (j / SS) * sl + j % SS + j * SS);
            
            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                v_[i][j] /= l_jj;
        }
    }


    template <typename T, size_t M, size_t N, size_t SS>
    template <bool SOA, bool SOB>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SS>::ger(T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        if (SOA == columnMajor && SOB == rowMajor)
        {
            BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= RegisterCapacity_v<T, SS>), "Not enough registers for ger()");

            IntrinsicType ax[RM];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                ax[i] = alpha * blazefeo::load<SS>(a + i * sa);

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


    template <typename T, size_t M, size_t N, size_t SS>
    template <bool SOA, bool SOB>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SS>::ger(T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        if (SOA == columnMajor && SOB == rowMajor)
        {
            BLAZE_STATIC_ASSERT_MSG((RM * RN + RM + 1 <= RegisterCapacity_v<T, SS>), "Not enough registers for ger()");

            IntrinsicType ax[RM];

            #pragma unroll
            for (size_t i = 0; i < RM; ++i)
                ax[i] = alpha * blazefeo::load<SS>(a + i * sa);
            
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


    template <bool LeftSide, bool Upper, bool TransA, typename T, size_t M, size_t N, size_t SS>
    inline void trsm(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa)
    {
        ker.template trsm<LeftSide, Upper, TransA>(a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<T, M, N, SS>::potrf()
    {
        static_assert(M >= N, "potrf() not implemented for register matrices with columns more than rows");
        static_assert(RM * RN + 2 <= RegisterCapacity_v<T, SS>, "Not enough registers");
        
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


    /// @brief Rank-1 update
    template <bool SOA, bool SOB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb)
    {
        ker.template ger<SOA, SOB>(alpha, a, sa, b, sb);
    }


    /// @brief Rank-1 update of specified size
    template <bool SOA, bool SOB, typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void ger(RegisterMatrix<T, M, N, SS>& ker, T alpha, T const * a, size_t sa, T const * b, size_t sb, size_t m, size_t n)
    {
        ker.template ger<SOA, SOB>(alpha, a, sa, b, sb, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    // [[deprecated("Use load with a matrix argument instead")]]
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa)
    {
        ker.load(1., a, sa);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    BLAZE_ALWAYS_INLINE void load(RegisterMatrix<T, M, N, SS>& ker, T const * a, size_t sa, size_t m, size_t n)
    {
        ker.load(1.0, a, sa, m, n);
    }


    template <typename T, size_t M, size_t N, size_t SS>
    // [[deprecated("Use load with a matrix argument instead")]]
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
    // [[deprecated("Use store with a matrix argument instead")]]
    BLAZE_ALWAYS_INLINE void store(RegisterMatrix<T, M, N, SS> const& ker, T * a, size_t sa, size_t m, size_t n)
    {
        ker.store(a, sa, m, n);
    }
}