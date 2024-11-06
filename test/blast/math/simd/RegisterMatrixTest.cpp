// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/RegisterMatrix.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/math/Vector.hpp>
#include <blast/math/views/Submatrix.hpp>
#include <blast/math/reference/Ger.hpp>
#include <blast/math/reference/Gemm.hpp>
#include <blast/math/reference/Axpy.hpp>
#include <blast/math/reference/Trmm.hpp>
#include <blast/math/reference/Trsm.hpp>
#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/dense/StaticMatrix.hpp>
#include <blast/math/expressions/MatTransExpr.hpp>
#include <blast/math/algorithm/Randomize.hpp>
#include <blast/math/algorithm/MakePositiveDefinite.hpp>

#include <test/Testing.hpp>
#include <test/Tolerance.hpp>


namespace blast :: testing
{
    template <typename Ker>
    class RegisterMatrixTest
    :   public Test
    {
    };


    using MyTypes = Types<
        RegisterMatrix<double, 1 * SimdSize_v<double>, 4, columnMajor>,
        RegisterMatrix<double, 1 * SimdSize_v<double>, 2, columnMajor>,
        RegisterMatrix<double, 1 * SimdSize_v<double>, 1, columnMajor>,
        RegisterMatrix<double, 2 * SimdSize_v<double>, 4, columnMajor>,
        RegisterMatrix<double, 2 * SimdSize_v<double>, 2, columnMajor>,
        RegisterMatrix<double, 2 * SimdSize_v<double>, 1, columnMajor>,
        RegisterMatrix<double, 3 * SimdSize_v<double>, 4, columnMajor>,
        RegisterMatrix<double, 3 * SimdSize_v<double>, 2, columnMajor>,
        RegisterMatrix<double, 3 * SimdSize_v<double>, 1, columnMajor>,
        RegisterMatrix<float, 1 * SimdSize_v<float>, 4, columnMajor>,
        RegisterMatrix<float, 2 * SimdSize_v<float>, 4, columnMajor>,
        RegisterMatrix<float, 3 * SimdSize_v<float>, 4, columnMajor>
    >;


    TYPED_TEST_SUITE(RegisterMatrixTest, MyTypes);


    TYPED_TEST(RegisterMatrixTest, testDefaultCtor)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        RM ker;

        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), ET(0.));
    }


    TYPED_TEST(RegisterMatrixTest, testReset)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        randomize(A);

        RM ker;
        ET const beta = 0.1;
        ker.load(beta, ptr(A));

        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), beta * A(i, j));

        ker.reset();
        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), ET(0.));
    }


    TYPED_TEST(RegisterMatrixTest, testLoadPanel)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        randomize(A);

        RM ker;
        ET const beta = 0.1;
        ker.load(beta, ptr<aligned>(A, 0, 0));

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(ker(i, j), beta * A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST(RegisterMatrixTest, testPartialLoadPanel)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        randomize(A);

        for (size_t m = 0; m <= rows(A); ++m)
        {
            for (size_t n = 0; n <= columns(A); ++n)
            {
                RM ker;
                ET const beta = 0.1;
                ker.load(beta, ptr(A), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        ASSERT_EQ(ker(i, j), beta * A(i, j))
                        << "load error for size (" << m << ", " << n << "); "
                        << "element mismatch at (" << i << ", " << j << ")" ;
            }
        }
    }


    TYPED_TEST(RegisterMatrixTest, testPartialLoadDense)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        for (size_t i = 0; i < rows(A); ++i)
            for (size_t j = 0; j < columns(A); ++j)
                A(i, j) = 1000 * i + j;

        for (size_t m = 1; m <= rows(A); ++m)
        {
            for (size_t n = 1; n <= columns(A); ++n)
            {
                RM ker;
                ET const beta = 0.1;

                // Use lower right corner of the matrix to make sure that we don't read beyond the array bounds.
                ker.load(beta, ptr<unaligned>(A, rows(A) - m, columns(A) - n), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        ASSERT_EQ(ker(i, j), beta * A(rows(A) - m + i, columns(A) - n + j))
                        << "load error for size (" << m << ", " << n << "); "
                        << "element mismatch at (" << i << ", " << j << ")" ;
            }
        }
    }


    TYPED_TEST(RegisterMatrixTest, testLoadStore)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        randomize(A);

        RM ker;
        ker.load(ptr(A));
        ker.store(ptr(B));

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST(RegisterMatrixTest, testLoadStore2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B(0.);
        randomize(A);

        RM ker;
        ker.load(1., ptr<aligned>(A, 0, 0));
        ker.store(ptr<aligned>(B, 0, 0));

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST(RegisterMatrixTest, testLoadDynamicMatrix)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        RM ker;

        DynamicMatrix<ET, StorageOrder_v<RM>> A(ker.rows(), ker.columns());
        randomize(A);

        ker.load(1., ptr<aligned>(A, 0, 0));
        // store2(ker, B.data(), B.spacing());

        EXPECT_EQ(ker, A);

        // for (size_t i = 0; i < ker.rows(); ++i)
        //     for (size_t j = 0; j < ker.columns(); ++j)
        //         EXPECT_EQ(ker(i, j), A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST(RegisterMatrixTest, testStoreDynamicMatrix)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        RM ker;

        DynamicMatrix<ET, StorageOrder_v<RM>> A(ker.rows(), ker.columns());
        randomize(A);

        DynamicMatrix<ET, StorageOrder_v<RM>> B(ker.rows(), ker.columns());
        reset(B);

        ker.load(1., ptr<aligned>(A, 0, 0));
        ker.store(ptr<aligned>(B, 0, 0));

        EXPECT_EQ(B, A);
    }


    TYPED_TEST(RegisterMatrixTest, testLoadStaticMatrix)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        RM ker;

        StaticMatrix<ET, ker.rows(), ker.columns(), StorageOrder_v<RM>> A;
        randomize(A);

        ker.load(1., ptr<aligned>(A, 0, 0));

        EXPECT_EQ(ker, A);
    }


    TYPED_TEST(RegisterMatrixTest, testStoreStaticMatrix)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        RM ker;

        StaticMatrix<ET, ker.rows(), ker.columns(), StorageOrder_v<RM>> A, B;
        randomize(A);
        reset(B);

        ker.load(1., ptr<aligned>(A, 0, 0));
        ker.store(ptr<aligned>(B, 0, 0));

        EXPECT_EQ(B, A);
    }


    TYPED_TEST(RegisterMatrixTest, testPartialStore)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        assign(A, A_ref);

        RM ker;
        ker.load(ptr(A));

        for (size_t m = ker.rows() + 1 - ker.simdSize(); m <= Traits::rows; ++m)
            for (size_t n = 1; n <= Traits::columns; ++n)
            {
                if (m != Traits::rows && n != Traits::columns)
                {
                    B = 0.;
                    ker.store(ptr(B), m, n);

                    for (size_t i = 0; i < Traits::rows; ++i)
                        for (size_t j = 0; j < Traits::columns; ++j)
                            ASSERT_EQ(B(i, j), i < m && j < n ? A_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), "
                                << "store size = " << m << "x" << n;
                }
            }
    }


    TYPED_TEST(RegisterMatrixTest, testPartialStore2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        randomize(A);

        RM ker;
        ker.load(1., ptr<aligned>(A, 0, 0));

        for (size_t m = 0; m <= Traits::rows; ++m)
            for (size_t n = 0; n <= Traits::columns; ++n)
            {
                B = 0.;
                ker.store(ptr<aligned>(B, 0, 0), m, n);

                for (size_t i = 0; i < Traits::rows; ++i)
                    for (size_t j = 0; j < Traits::columns; ++j)
                        ASSERT_EQ(B(i, j), i < m && j < n ? A(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), "
                            << "store size = " << m << "x" << n;
            }
    }


    TYPED_TEST(RegisterMatrixTest, testGerNT)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, 1, columnMajor> A;
        StaticPanelMatrix<ET, Traits::columns, 1, columnMajor> B;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> C;

        randomize(A);
        randomize(B);
        randomize(C);

        TypeParam ker;
        ker.load(ptr(C));
        ker.ger(column(ptr(A)), row(ptr(B).trans()));

        reference::ger(rows(C), columns(C), 1., column(ptr(A)), column(ptr(B)).trans(), ptr(C), ptr(C));

        BLAST_EXPECT_APPROX_EQ(ker, C, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testGer)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicVector<ET, columnVector> a(Traits::rows);
        DynamicVector<ET, rowVector> b(Traits::columns);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> C;

        randomize(a);
        randomize(b);
        randomize(C);
        ET alpha {};
        randomize(alpha);

        TypeParam ker;
        ker.load(1., ptr(C));
        ker.ger(alpha, ptr(a), ptr(b));

        reference::ger(rows(C), columns(C), alpha, ptr(a), ptr(b), ptr(C), ptr(C));

        BLAST_EXPECT_APPROX_EQ(ker, C, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testPartialGerNT)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, 1, columnMajor> A;
        StaticPanelMatrix<ET, Traits::columns, 1, columnMajor> B;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> C;

        randomize(A);
        randomize(B);
        randomize(C);

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> D;
        reference::ger(Traits::rows, Traits::columns, 1., column(ptr(A)), column(ptr(B)).trans(), ptr(C), ptr(D));

        for (size_t m = 0; m <= rows(C); ++m)
        {
            for (size_t n = 0; n <= columns(C); ++n)
            {
                TypeParam ker;
                ker.load(ptr(C));
                ker.ger(column(ptr(A)), column(ptr(B)).trans(), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        BLAST_ASSERT_APPROX_EQ(ker(i, j), D(i, j), absTol<ET>(), relTol<ET>())
                            << "element mismatch at (" << i << ", " << j << "), "
                            << "store size = " << m << "x" << n;
            }
        }
    }


    TYPED_TEST(RegisterMatrixTest, testPartialGerNT2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicVector<ET, columnVector> a(Traits::rows);
        DynamicVector<ET, columnVector> b(Traits::columns);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> C;

        randomize(a);
        randomize(b);
        randomize(C);

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> D;
        reference::ger(Traits::rows, Traits::columns, 1., ptr(a), ptr(trans(b)), ptr(C), ptr(D));

        for (size_t m = 0; m <= rows(C); ++m)
        {
            for (size_t n = 0; n <= columns(C); ++n)
            {
                TypeParam ker;
                ker.load(1., ptr(C));
                ker.ger(ET(1.), ptr(a), ptr(trans(b)), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        BLAST_ASSERT_APPROX_EQ(ker(i, j), D(i, j), absTol<ET>(), relTol<ET>())
                            << "element mismatch at (" << i << ", " << j << "), "
                            << "store size = " << m << "x" << n;
            }
        }
    }


    TYPED_TEST(RegisterMatrixTest, testGerNT2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicVector<ET, columnVector> a(Traits::rows);
        DynamicVector<ET, columnVector> b(Traits::columns);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> C, D;

        randomize(a);
        randomize(b);
        randomize(C);

        TypeParam ker;
        ker.load(1., ptr(C));
        ker.ger(ET(1.), ptr(a), ptr(trans(b)));
        ker.store(ptr(D));

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> D_ref;
        reference::ger(Traits::rows, Traits::columns, 1., ptr(a), ptr(trans(b)), ptr(C), ptr(D_ref));
        BLAST_EXPECT_APPROX_EQ(D, D_ref, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testPotrf)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;
        using ET = typename Traits::ElementType;
        static size_t constexpr m = Traits::rows;
        static size_t constexpr n = Traits::columns;

        if constexpr (m >= n)
        {
            StaticMatrix<ET, m, n, columnMajor> A, L;
            makePositiveDefinite(submatrix<aligned>(A, 0, 0, n, n));
            randomize(submatrix<unaligned>(A, n, 0, m - n, n));

            TypeParam ker;
            ker.load(ptr(A));
            ker.potrf();
            ker.store(ptr(L));

            StaticMatrix<ET, m, m, columnMajor> LTL {};
            reference::gemm(1., L, trans(L), 0., LTL, LTL);

            BLAST_ASSERT_APPROX_EQ(submatrix<aligned>(LTL, 0, 0, m, n), A, absTol<ET>(), relTol<ET>());
        }
        else
        {
            std::clog << "RegisterMatrixTest.testPotrf not implemented for kernels with columns more than rows!" << std::endl;
        }
    }


    TYPED_TEST(RegisterMatrixTest, testTrsmRightLowerTransposePanel)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        RM ker;

        StaticPanelMatrix<ET, Traits::columns, Traits::columns, columnMajor> A;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> B, X;

        randomize(A);
        for (size_t i = 0; i < Traits::columns; ++i)
            A(i, i) += Traits::columns;  // Improve conditioning

        randomize(B);

        // True value
        reference::trsm(X, trans(A), UpLo::Upper, false, ET(1.), B);

        ker.load(ptr(B));
        ker.trsm(Side::Right, UpLo::Upper, ptr(A).trans());

        // TODO: should be strictly equal?
        BLAST_ASSERT_APPROX_EQ(ker, X, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testTrsmRightLowerTransposeDense)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        RM ker;

        StaticMatrix<ET, RM::columns(), RM::columns(), columnMajor> A;
        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> B, X;

        randomize(A);
        for (size_t i = 0; i < RM::columns(); ++i)
            A(i, i) += RM::columns();  // Improve conditioning

        randomize(B);

        ker.load(ptr(B));
        ker.trsm(Side::Right, UpLo::Upper, trans(ptr(A)));

        reference::trsm(X, trans(A), UpLo::Upper, false, ET(1.), B);

        // TODO: should be strictly equal?
        BLAST_ASSERT_APPROX_EQ(ker, X, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testTrmmLeftUpper)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, RM::rows(), RM::rows(), columnMajor> A;
        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> B, C;

        randomize(A);
        randomize(B);

        ET alpha {};
        randomize(alpha);

        RM ker;
        ker.trmm(alpha, ptr(A), UpLo::Upper, false, ptr(B));

        reference::trmm(alpha, A, UpLo::Upper, false, B, C);

        // TODO: should be strictly equal?
        BLAST_ASSERT_APPROX_EQ(ker, C, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testTrmmRightLower)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, RM::columns(), RM::columns(), columnMajor> A;
        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> B, C;

        randomize(A);
        randomize(B);

        ET alpha {};
        randomize(alpha);

        RM ker;
        ker.trmm(alpha, ptr(B), ptr(A), UpLo::Lower, false);

        reference::trmm(alpha, B, A, UpLo::Lower, false, C);

        // TODO: should be strictly equal?
        BLAST_ASSERT_APPROX_EQ(ker, C, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testAxpy)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> A, B;
        randomize(A);
        randomize(B);

        ET alpha {};
        randomize(alpha);

        RM ker;
        ker.load(ptr(B));
        ker.axpy(alpha, ptr(A));

        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> C;
        reference::axpy(alpha, A, B, C);

        BLAST_EXPECT_APPROX_EQ(ker, C, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testAxpyWithSize)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> A, B;
        randomize(A);
        randomize(B);

        ET alpha {};
        randomize(alpha);

        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> C;

        for (size_t m = 0; m < RM::rows(); ++m)
        {
            for (size_t n = 0; n < RM::columns(); ++n)
            {
                RM ker;
                ker.load(ptr(B));
                ker.axpy(alpha, ptr(A), ker.rows(), ker.columns());

                reference::axpy(alpha,
                    submatrix<aligned>(A, 0, 0, m, n),
                    submatrix<aligned>(B, 0, 0, m, n),
                    submatrix<aligned>(C, 0, 0, m, n)
                );

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        BLAST_ASSERT_APPROX_EQ(ker(i, j), C(i, j), absTol<ET>(), relTol<ET>())
                            << "element mismatch at (" << i << ", " << j << "), "
                            << "axpy() size = " << m << "x" << n;
            }
        }
    }
}
