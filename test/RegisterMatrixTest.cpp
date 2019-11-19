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


    TYPED_TEST_P(RegisterMatrixTest, testDefaultCtor)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        RM ker;
        
        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), ET(0.));
    }


    TYPED_TEST_P(RegisterMatrixTest, testReset)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        randomize(A);

        RM ker;
        ET const beta = 0.1;
        ker.load(beta, A, 0, 0, rows(A), columns(A));

        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), beta * A(i, j));

        ker.reset();
        for (size_t i = 0; i < ker.rows(); ++i)
            for (size_t j = 0; j < ker.columns(); ++j)
                ASSERT_EQ(ker(i, j), ET(0.));
    }


    TYPED_TEST_P(RegisterMatrixTest, testLoad)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A;
        randomize(A);

        RM ker;
        ET const beta = 0.1;
        ker.load(beta, A, 0, 0, rows(A), columns(A));

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(ker(i, j), beta * A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(RegisterMatrixTest, testPartialLoad)
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
                ker.load(beta, A, 0, 0, m, n);

                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        ASSERT_EQ(ker(i, j), beta * A(i, j)) 
                        << "load error for size (" << m << ", " << n << "); "
                        << "element mismatch at (" << i << ", " << j << ")" ;
            }
        }
    }


    TYPED_TEST_P(RegisterMatrixTest, testLoadStore)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        randomize(A);

        RM ker;
        ker.load(1., A, 0, 0, rows(A), columns(A));

        // std::cout << "A_ref=\n" << A_ref << std::endl;
        // std::cout << "A=\n" << A << std::endl;
        // std::cout << "ker=\n" << ker << std::endl;

        ker.store(B);
        // std::cout << "B=\n" << B << std::endl;

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(RegisterMatrixTest, testPartialStore)
    {
        using RM = TypeParam;
        using Traits = RegisterMatrixTraits<RM>;
        using ET = ElementType_t<RM>;

        blaze::StaticMatrix<ET, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<ET, Traits::rows, Traits::columns, columnMajor> A, B;
        A = A_ref;

        RM ker;
        ker.load(1., A, 0, 0, rows(A), columns(A));

        for (size_t m = ker.rows() + 1 - ker.simdSize(); m <= Traits::rows; ++m)
            for (size_t n = 1; n <= Traits::columns; ++n)
            {
                B = 0.;
                ker.store(B, 0, 0, m, n);

                for (size_t i = 0; i < Traits::rows; ++i)
                    for (size_t j = 0; j < Traits::columns; ++j)
                        ASSERT_EQ(B(i, j), i < m && j < n ? A_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), " 
                            << "store size = " << m << "x" << n;
            }
    }


    TYPED_TEST_P(RegisterMatrixTest, testGerNT)
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
        ker.load(1., C, 0, 0, rows(C), columns(C));
        ger<A.storageOrder, !B.storageOrder>(ker, ET(1.), A.ptr(0, 0), A.spacing(), B.ptr(0, 0), B.spacing());
        ker.store(D);
        
        md = D;

        BLAZEFEO_EXPECT_EQ(md, evaluate(mc + ma * trans(mb)));
    }


    TYPED_TEST_P(RegisterMatrixTest, testPotrf)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;
        using ET = typename Traits::ElementType;
        static size_t constexpr m = Traits::rows;
        static size_t constexpr n = Traits::columns;
        
        TypeParam ker;

        if constexpr (m >= n)
        {
            StaticPanelMatrix<ET, m, n, columnMajor> A, L;

            {
                blaze::StaticMatrix<ET, n, n, columnMajor> C0;
                makePositiveDefinite(C0);

                blaze::StaticMatrix<ET, m, n, columnMajor> C;
                submatrix(C, 0, 0, n, n) = C0;
                randomize(submatrix(C, n, 0, m - n, n));

                A = C;
            }

            ker.load(1., A, 0, 0, rows(A), columns(A));
            ker.potrf();
            ker.store(L);

            StaticMatrix<ET, m, n, columnMajor> LL;
            LL = L;
            // L.unpack(LL.data(), LL.spacing());

            StaticMatrix<ET, m, m, columnMajor> A1 = LL * trans(LL);

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

        LL = L;
        BB = B;

        // std::cout << "BB=\n" << BB << std::endl;
        // std::cout << "LL=\n" << LL << std::endl;

        // Workaround the bug:
        // https://bitbucket.org/blaze-lib/blaze/issues/301/error-in-evaluation-of-a-inv-trans-b
        XX = evaluate(BB * evaluate(inv(evaluate(trans(LL)))));
        
        ker.load(1., B, 0, 0, rows(B), columns(B));
        trsm<false, false, true>(ker, L.ptr(0, 0), spacing(L));
        ker.store(X);

        // std::cout << "X=\n" << X << std::endl;
        // std::cout << "XX=\n" << XX << std::endl;
        
        // TODO: should be strictly equal?
        BLAZEFEO_ASSERT_APPROX_EQ(X, XX, absTol<ET>(), relTol<ET>());
    }


    REGISTER_TYPED_TEST_SUITE_P(RegisterMatrixTest,
        testDefaultCtor,
        testReset,
        testLoad,
        testPartialLoad,
        testLoadStore,
        testPartialStore,
        testGerNT,
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