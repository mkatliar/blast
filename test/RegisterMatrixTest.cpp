#include <blazefeo/math/simd/RegisterMatrix.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


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
        using Traits = RegisterMatrixTraits<TypeParam>;

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        load(ker, A.tile(0, 0), A.spacing());
        store(ker, B.tile(0, 0), B.spacing());

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(RegisterMatrixTest, testPartialStore)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        load(ker, A.tile(0, 0), A.spacing());

        for (size_t m = 0; m <= Traits::rows; ++m)
            for (size_t n = 0; n <= Traits::columns; ++n)
            {
                B = 0.;
                store(ker, B.tile(0, 0), B.spacing(), m, n);

                for (size_t i = 0; i < Traits::rows; ++i)
                    for (size_t j = 0; j < Traits::columns; ++j)
                        ASSERT_EQ(B(i, j), i < m && j < n ? A_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), " 
                            << "store size = " << m << "x" << n;
            }
    }


    TYPED_TEST_P(RegisterMatrixTest, testGerNT)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;

        blaze::DynamicMatrix<double, blaze::columnMajor> ma(Traits::rows, 1);
        blaze::DynamicMatrix<double, blaze::columnMajor> mb(Traits::columns, 1);
        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;

        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticPanelMatrix<double, Traits::rows, 1, rowMajor> a;
        StaticPanelMatrix<double, Traits::columns, 1, rowMajor> b;
        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> c, d;

        a.pack(data(ma), spacing(ma));
        b.pack(data(mb), spacing(mb));
        c.pack(data(mc), spacing(mc));

        TypeParam ker;
        load(ker, c.tile(0, 0), c.spacing());
        ger<false, true>(ker, 1.0, a.tile(0, 0), a.spacing(), b.tile(0, 0), b.spacing());
        store(ker, d.tile(0, 0), d.spacing());
        
        d.unpack(data(md), spacing(md));

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
            StaticPanelMatrix<ET, m, n, rowMajor> A, L;
            StaticPanelMatrix<ET, m, m, rowMajor> A1;

            {
                blaze::StaticMatrix<ET, n, n, columnMajor> C0;
                makePositiveDefinite(C0);

                blaze::StaticMatrix<ET, m, n, columnMajor> C;
                submatrix(C, 0, 0, n, n) = C0;
                randomize(submatrix(C, n, 0, m - n, n));

                A.pack(data(C), spacing(C));
            }

            load(ker, A.tile(0, 0), A.spacing());
            ker.potrf();
            store(ker, L.tile(0, 0), L.spacing());

            A1 = 0.;
            gemm_nt(L, L, A1, A1);

            // std::cout << "A=\n" << A << std::endl;
            // std::cout << "L=\n" << L << std::endl;
            // std::cout << "A1=\n" << A1 << std::endl;

            BLAZEFEO_ASSERT_APPROX_EQ(submatrix(A1, 0, 0, m, n), A, 1e-15, 1e-15);
        }
        else
        {
            std::clog << "RegisterMatrixTest.testPotrf not implemented for kernels with columns more than rows!" << std::endl;
        }        
    }


    TYPED_TEST_P(RegisterMatrixTest, testTrsmRLT)
    {
        using Traits = RegisterMatrixTraits<TypeParam>;
        TypeParam ker;

        using blaze::randomize;
        StaticPanelMatrix<typename Traits::ElementType, Traits::columns, Traits::columns, rowMajor> L;
        StaticPanelMatrix<typename Traits::ElementType, Traits::rows, Traits::columns, rowMajor> B, X, B1;            
        
        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                if (j <= i)
                    randomize(L(i, j));
                else
                    reset(L(i, j));

        randomize(B);

        // std::cout << "B=" << B << std::endl;
        // std::cout << "L=" << L << std::endl;

        load(ker, B.tile(0, 0), B.spacing());
        trsm<false, false, true>(ker, L.tile(0, 0), spacing(L));
        store(ker, X.tile(0, 0), X.spacing());

        B1 = 0.;
        gemm_nt(X, L, B1, B1);

        // std::cout << B1 << std::endl;

        BLAZEFEO_ASSERT_APPROX_EQ(B1, B, 1e-11, 1e-11);
    }


    REGISTER_TYPED_TEST_SUITE_P(RegisterMatrixTest,
        testLoadStore,
        testPartialStore,
        testGerNT,
        testPotrf,
        testTrsmRLT
    );


    using RegisterMatrix_double_1_4_4 = RegisterMatrix<double, 1, 4, 4>;
    using RegisterMatrix_double_2_4_4 = RegisterMatrix<double, 2, 4, 4>;
    using RegisterMatrix_double_3_4_4 = RegisterMatrix<double, 3, 4, 4>;
    using RegisterMatrix_double_2_8_4 = RegisterMatrix<double, 2, 8, 4>;

    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_1_4_4, RegisterMatrixTest, RegisterMatrix_double_1_4_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_2_4_4, RegisterMatrixTest, RegisterMatrix_double_2_4_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_3_4_4, RegisterMatrixTest, RegisterMatrix_double_3_4_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(RegisterMatrix_double_2_8_4, RegisterMatrixTest, RegisterMatrix_double_2_8_4);
}