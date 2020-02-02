#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>
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


    TYPED_TEST_SUITE_P(RegisterMatrixTest);


    TYPED_TEST_P(RegisterMatrixTest, testLoadStore)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        RM ker;
        load(ker, A.ptr(0, 0), A.spacing());
        store(ker, B.ptr(0, 0), B.spacing());

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(RegisterMatrixTest, testLoadStore2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B(0.);
        randomize(A);

        RM ker;
        load2(ker, A.data(), A.spacing());
        store2(ker, B.data(), B.spacing());

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(RegisterMatrixTest, testPartialStore)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

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


    TYPED_TEST_P(RegisterMatrixTest, testPartialStore2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        randomize(A);

        RM ker;
        load2(ker, A.data(), A.spacing());

        for (size_t m = 0; m <= Traits::rows; ++m)
            for (size_t n = 0; n <= Traits::columns; ++n)
            {
                B = 0.;
                store2(ker, B.data(), B.spacing(), m, n);

                for (size_t i = 0; i < Traits::rows; ++i)
                    for (size_t j = 0; j < Traits::columns; ++j)
                        ASSERT_EQ(B(i, j), i < m && j < n ? A(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), " 
                            << "store size = " << m << "x" << n;
            }
    }


    TYPED_TEST_P(RegisterMatrixTest, testGerNT)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicMatrix<ET, columnMajor> ma(Traits::rows, 1);
        DynamicMatrix<ET, columnMajor> mb(Traits::columns, 1);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> mc, md;

        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticPanelMatrix<ET, Traits::rows, 1, columnMajor> A;
        StaticPanelMatrix<ET, Traits::columns, 1, columnMajor> B;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> C, D;

        A.pack(data(ma), spacing(ma));
        B.pack(data(mb), spacing(mb));
        C.pack(data(mc), spacing(mc));

        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "B=\n" << B << std::endl;
        // std::cout << "C=\n" << C << std::endl;

        TypeParam ker;
        load(ker, C.ptr(0, 0), C.spacing());
        ger<A.storageOrder, !B.storageOrder>(ker, ET(1.), A.ptr(0, 0), A.spacing(), B.ptr(0, 0), B.spacing());
        store(ker, D.ptr(0, 0), D.spacing());
        
        D.unpack(data(md), spacing(md));

        BLAZEFEO_EXPECT_EQ(md, evaluate(mc + ma * trans(mb)));
    }


    TYPED_TEST_P(RegisterMatrixTest, testPartialGerNT)
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

        A.pack(data(ma), spacing(ma));
        B.pack(data(mb), spacing(mb));
        C.pack(data(mc), spacing(mc));

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


    TYPED_TEST_P(RegisterMatrixTest, testPartialGerNT2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicMatrix<ET, columnMajor> A(Traits::rows, 1);
        DynamicMatrix<ET, columnMajor> B(Traits::columns, 1);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> C;

        randomize(A);
        randomize(B);
        randomize(C);

        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> const D_ref = A * trans(B) + C;
        
        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "B=\n" << B << std::endl;
        // std::cout << "C=\n" << C << std::endl;

        for (size_t m = 0; m <= rows(C); ++m)
        {
            for (size_t n = 0; n <= columns(C); ++n)
            {
                TypeParam ker;
                load2(ker, C.data(), C.spacing());
                ger2<A.storageOrder, !B.storageOrder>(ker, ET(1.), A.data(), A.spacing(), B.data(), B.spacing(), m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        ASSERT_EQ(ker(i, j), i < m && j < n ? D_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), " 
                            << "store size = " << m << "x" << n;
            }
        }
    }


    TYPED_TEST_P(RegisterMatrixTest, testGerNT2)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        DynamicMatrix<ET, columnMajor> A(Traits::rows, 1);
        DynamicMatrix<ET, columnMajor> B(Traits::columns, 1);
        StaticMatrix<ET, Traits::rows, Traits::columns, columnMajor> C, D;

        randomize(A);
        randomize(B);
        randomize(C);

        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "B=\n" << B << std::endl;
        // std::cout << "C=\n" << C << std::endl;

        TypeParam ker;
        load2(ker, C.data(), C.spacing());
        ger2<A.storageOrder, !B.storageOrder>(ker, ET(1.), A.data(), A.spacing(), B.data(), B.spacing());
        store2(ker, D.data(), D.spacing());

        BLAZEFEO_EXPECT_EQ(D, evaluate(C + A * trans(B)));
    }


    TYPED_TEST_P(RegisterMatrixTest, testPotrf)
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

                A.pack(data(C), spacing(C));
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


    TYPED_TEST_P(RegisterMatrixTest, testTrsmRLT)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        RM ker;

        using blaze::randomize;
        StaticPanelMatrix<ET, Traits::columns, Traits::columns, columnMajor> L;
        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> B, X, B1;            
        
        for (size_t i = 0; i < Traits::rows; ++i)
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

        L.unpack(LL.data(), LL.spacing());
        B.unpack(BB.data(), BB.spacing());

        // std::cout << "BB=\n" << BB << std::endl;
        // std::cout << "LL=\n" << LL << std::endl;

        // Workaround the bug:
        // https://bitbucket.org/blaze-lib/blaze/issues/301/error-in-evaluation-of-a-inv-trans-b
        XX = evaluate(BB * evaluate(inv(evaluate(trans(LL)))));
        
        load(ker, B.ptr(0, 0), B.spacing());
        trsm<false, false, true>(ker, L.ptr(0, 0), spacing(L));
        store(ker, X.ptr(0, 0), X.spacing());

        // std::cout << "X=\n" << X << std::endl;
        // std::cout << "XX=\n" << XX << std::endl;
        
        // TODO: should be strictly equal?
        BLAZEFEO_ASSERT_APPROX_EQ(X, XX, absTol<ET>(), relTol<ET>());
    }


    REGISTER_TYPED_TEST_SUITE_P(RegisterMatrixTest,
        testLoadStore,
        testLoadStore2,
        testPartialStore,
        testPartialStore2,
        testGerNT,
        testPartialGerNT,
        testPartialGerNT2,
        testGerNT2,
        testPotrf,
        testTrsmRLT
    );


    using RM_double_4_4_4 = RegisterMatrix<double, 4, 4, 4>;
    using RM_double_4_2_4 = RegisterMatrix<double, 4, 2, 4>;
    using RM_double_4_1_4 = RegisterMatrix<double, 4, 1, 4>;
    using RM_double_8_4_4 = RegisterMatrix<double, 8, 4, 4>;
    using RM_double_12_4_4 = RegisterMatrix<double, 12, 4, 4>;

    using RM_float_8_4_8 = RegisterMatrix<float, 8, 4, 8>;
    using RM_float_16_4_8 = RegisterMatrix<float, 16, 4, 8>;
    using RM_float_24_4_8 = RegisterMatrix<float, 24, 4, 8>;

    INSTANTIATE_TYPED_TEST_SUITE_P(double_4_4_4, RegisterMatrixTest, RM_double_4_4_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(double_4_2_4, RegisterMatrixTest, RM_double_4_2_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(double_4_1_4, RegisterMatrixTest, RM_double_4_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(double_8_4_4, RegisterMatrixTest, RM_double_8_4_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(double_12_4_4, RegisterMatrixTest, RM_double_12_4_4);

    INSTANTIATE_TYPED_TEST_SUITE_P(float_8_4_8, RegisterMatrixTest, RM_float_8_4_8);
    INSTANTIATE_TYPED_TEST_SUITE_P(float_16_4_8, RegisterMatrixTest, RM_float_16_4_8);
    INSTANTIATE_TYPED_TEST_SUITE_P(float_24_4_8, RegisterMatrixTest, RM_float_24_4_8);
}