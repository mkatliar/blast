// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/dense/MatrixPointer.hpp>
#include <blazefeo/math/dense/VectorPointer.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>


namespace blazefeo :: testing
{
    template <typename Ker>
    class RegisterMatrixTest
    :   public Test
    {
    };


    using MyTypes = Types<
        RegisterMatrix<double, 4, 4, columnMajor>,
        RegisterMatrix<double, 4, 2, columnMajor>,
        RegisterMatrix<double, 4, 1, columnMajor>,
        RegisterMatrix<double, 8, 4, columnMajor>,
        RegisterMatrix<double, 12, 4, columnMajor>,
        RegisterMatrix<float, 8, 4, columnMajor>,
        RegisterMatrix<float, 16, 4, columnMajor>,
        RegisterMatrix<float, 24, 4, columnMajor>
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
        ker.load(beta, A.ptr(0, 0), spacing(A));

        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), beta * A(i, j));

        ker.reset();
        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), ET(0.));
    }


    TYPED_TEST(RegisterMatrixTest, testLoad)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        randomize(A);

        RM ker;
        ET const beta = 0.1;
        ker.load(beta, A.ptr(0, 0), spacing(A));

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
                ker.load(beta, A.ptr(0, 0), spacing(A), m, n);

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
                (*A)(i, j) = 1000 * i + j;

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
        ker.load(1., A.ptr(0, 0), spacing(A));

        // std::cout << "A_ref=\n" << A_ref << std::endl;
        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "ker=\n" << ker << std::endl;

        ker.store(B.ptr(0, 0), spacing(B));
        // std::cout << "B=\n" << B << std::endl;

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
        A = A_ref;

        RM ker;
        load(ker, A.ptr(0, 0), A.spacing());

        for (size_t m = ker.rows() + 1 - ker.simdSize(); m <= Traits::rows; ++m)
            for (size_t n = 1; n <= Traits::columns; ++n)
            {
                if (m != Traits::rows && n != Traits::columns)
                {
                    B = 0.;
                    store(ker, B.ptr(0, 0), B.spacing(), m, n);

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

        blaze::DynamicMatrix<ET, blaze::columnMajor> ma(Traits::rows, 1);
        blaze::DynamicMatrix<ET, blaze::columnMajor> mb(Traits::columns, 1);
        blaze::StaticMatrix<ET, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;

        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticPanelMatrix<ET, Traits::rows, 1, columnMajor> A;
        StaticPanelMatrix<ET, Traits::columns, 1, columnMajor> B;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> C, D;

        A = ma;
        B = mb;
        C = mc;

        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "B=\n" << B << std::endl;
        // std::cout << "C=\n" << C << std::endl;

        TypeParam ker;
        ker.load(1., C.ptr(0, 0), spacing(C));
        ger<A.storageOrder, !B.storageOrder>(ker, ET(1.), A.ptr(0, 0), A.spacing(), B.ptr(0, 0), B.spacing());
        ker.store(D.ptr(0, 0), spacing(D));

        md = D;

        BLAZEFEO_EXPECT_EQ(md, evaluate(mc + ma * trans(mb)));
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
        blaze::randomize(alpha);

        TypeParam ker;
        ker.load(1., ptr(C));
        ker.ger(alpha, ptr(a), ptr(b));

        BLAZEFEO_EXPECT_APPROX_EQ(ker, evaluate(C + alpha * a * b), absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testPartialGerNT)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicMatrix<ET, columnMajor> ma(Traits::rows, 1);
        DynamicMatrix<ET, columnMajor> mb(Traits::columns, 1);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> mc;

        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> const D_ref = ma * trans(mb) + mc;

        StaticPanelMatrix<ET, Traits::rows, 1, columnMajor> A;
        StaticPanelMatrix<ET, Traits::columns, 1, columnMajor> B;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> C;

        A = ma;
        B = mb;
        C = mc;

        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "B=\n" << B << std::endl;
        // std::cout << "C=\n" << C << std::endl;

        for (size_t m = 0; m <= rows(C); ++m)
        {
            for (size_t n = 0; n <= columns(C); ++n)
            {
                TypeParam ker;
                load(ker, C.ptr(0, 0), C.spacing());
                ger<A.storageOrder, !B.storageOrder>(ker, ET(1.), A.ptr(0, 0), A.spacing(), B.ptr(0, 0), B.spacing(), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        ASSERT_EQ(ker(i, j), i < m && j < n ? D_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), "
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

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> const D_ref = a * trans(b) + C;

        for (size_t m = 0; m <= rows(C); ++m)
        {
            for (size_t n = 0; n <= columns(C); ++n)
            {
                TypeParam ker;
                ker.load(1., ptr(C));
                ker.ger(ET(1.), ptr(a), ptr(trans(b)), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        BLAZEFEO_ASSERT_APPROX_EQ(ker(i, j), D_ref(i, j), absTol<ET>(), relTol<ET>())
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

        BLAZEFEO_EXPECT_EQ(D, evaluate(C + a * trans(b)));
    }


    TYPED_TEST(RegisterMatrixTest, testPotrf)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;
        using ET = typename Traits::ElementType;
        static size_t constexpr m = Traits::rows;
        static size_t constexpr n = Traits::columns;

        if constexpr (m >= n)
        {
            StaticPanelMatrix<ET, m, n, columnMajor> A, L;
            StaticPanelMatrix<ET, m, m, columnMajor> A1;

            {
                StaticMatrix<ET, n, n, columnMajor> C0;
                makePositiveDefinite(C0);

                StaticMatrix<ET, m, n, columnMajor> C;
                submatrix(C, 0, 0, n, n) = C0;
                randomize(submatrix(C, n, 0, m - n, n));

                A = C;
            }

            {
                TypeParam ker;
                load(ker, A.ptr(0, 0), A.spacing());
                ker.potrf();
                store(ker, L.ptr(0, 0), L.spacing());
            }

            A1 = 0.;
            gemm_nt(L, L, A1, A1);

            // std::cout << "A=\n" << A << std::endl;
            // std::cout << "L=\n" << L << std::endl;
            // std::cout << "A1=\n" << A1 << std::endl;

            BLAZEFEO_ASSERT_APPROX_EQ(submatrix(A1, 0, 0, m, n), A, absTol<ET>(), relTol<ET>());
        }
        else
        {
            std::clog << "RegisterMatrixTest.testPotrf not implemented for kernels with columns more than rows!" << std::endl;
        }
    }


    TYPED_TEST(RegisterMatrixTest, testTrsmRltPanel)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        RM ker;

        using blaze::randomize;
        StaticPanelMatrix<ET, Traits::columns, Traits::columns, columnMajor> L;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> B, X, B1;

        for (size_t i = 0; i < Traits::columns; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                if (j <= i)
                {
                    randomize(L(i, j));
                    if (i == j)
                        L(i, j) += ET(1.);  // Improve conditioning
                }
                else
                    reset(L(i, j));

        randomize(B);

        // std::cout << "B=\n" << B << std::endl;
        // std::cout << "L=\n" << L << std::endl;

        StaticMatrix<ET, Traits::columns, Traits::columns, columnMajor> LL;
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> BB, XX;

        LL = L;
        BB = B;

        // std::cout << "BB=\n" << BB << std::endl;
        // std::cout << "LL=\n" << LL << std::endl;

        // True value
        XX = evaluate(BB * inv(trans(LL)));

        load(ker, B.ptr(0, 0), B.spacing());
        trsm<false, false, true>(ker, L.ptr(0, 0), spacing(L));
        store(ker, X.ptr(0, 0), X.spacing());

        // std::cout << "X=\n" << X << std::endl;
        // std::cout << "XX=\n" << XX << std::endl;

        // TODO: should be strictly equal?
        BLAZEFEO_ASSERT_APPROX_EQ(X, XX, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testTrsmRltDense)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;

        RM ker;

        using blaze::randomize;
        StaticMatrix<ET, RM::columns(), RM::columns(), columnMajor> L;
        StaticMatrix<ET, RM::rows(), RM::columns(), columnMajor> B, B1;

        for (size_t i = 0; i < RM::columns(); ++i)
            for (size_t j = 0; j < RM::columns(); ++j)
                if (j <= i)
                {
                    randomize(L(i, j));
                    if (i == j)
                        L(i, j) += ET(1.);  // Improve conditioning
                }
                else
                    reset(L(i, j));

        randomize(B);

        ker.load(1., ptr(B));
        ker.trsmRightUpper(trans(ptr(L)));

        // TODO: should be strictly equal?
        BLAZEFEO_ASSERT_APPROX_EQ(ker, evaluate(B * inv(trans(L))), absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testTrmmLeftUpper)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;


        DynamicMatrix<ET, columnMajor> A(RM::rows(), RM::rows());
        DynamicMatrix<ET, columnMajor> B(RM::rows(), RM::columns());

        randomize(A);
        randomize(B);

        ET alpha {};
        blaze::randomize(alpha);

        RM ker;
        ker.trmmLeftUpper(alpha, ptr(A), ptr(B));

        // Reset lower-triangular part
        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = 0; j < i && j < A.columns(); ++j)
                reset(A(i, j));

        // TODO: should be strictly equal?
        BLAZEFEO_ASSERT_APPROX_EQ(ker, alpha * A * B, absTol<ET>(), relTol<ET>());
    }


    TYPED_TEST(RegisterMatrixTest, testTrmmRightLower)
    {
        using RM = TypeParam;
        using ET = ElementType_t<RM>;


        DynamicMatrix<ET, columnMajor> A(RM::columns(), RM::columns());
        DynamicMatrix<ET, columnMajor> B(RM::rows(), RM::columns());

        randomize(A);
        randomize(B);

        ET alpha {};
        blaze::randomize(alpha);

        RM ker;
        ker.trmmRightLower(alpha, ptr(B), ptr(A));

        // Reset upper-triangular part
        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = i + 1; j < A.columns(); ++j)
                reset(A(i, j));

        // TODO: should be strictly equal?
        BLAZEFEO_ASSERT_APPROX_EQ(ker, alpha * B * A, absTol<ET>(), relTol<ET>());
    }
}