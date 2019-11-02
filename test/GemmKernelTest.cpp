#include <blazefeo/math/panel/gemm/GemmKernel_double_1_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_2_1_4.hpp>
#include <blazefeo/math/panel/gemm/GemmKernel_double_3_1_4.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blazefeo :: testing
{
    template <typename Ker>
    class GemmKernelTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(GemmKernelTest);


    TYPED_TEST_P(GemmKernelTest, testLoadStore)
    {
        using Traits = GemmKernelTraits<TypeParam>;

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        ker.load(A.tile(0, 0), A.spacing());
        ker.store(B.tile(0, 0), B.spacing());

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TYPED_TEST_P(GemmKernelTest, testStore)
    {
        using Traits = GemmKernelTraits<TypeParam>;

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticPanelMatrix<double, Traits::rows, Traits::columns, rowMajor> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        ker.load(A.tile(0, 0), A.spacing());

        for (size_t m = 0; m <= Traits::rows; ++m)
            for (size_t n = 0; n <= Traits::columns; ++n)
            {
                B = 0.;
                ker.store(B.tile(0, 0), B.spacing(), m, n);

                for (size_t i = 0; i < Traits::rows; ++i)
                    for (size_t j = 0; j < Traits::columns; ++j)
                        ASSERT_EQ(B(i, j), i < m && j < n ? A_ref(i, j) : 0.) << "element mismatch at (" << i << ", " << j << "), " 
                            << "store size = " << m << "x" << n;
            }
    }


    TYPED_TEST_P(GemmKernelTest, testGemmNT)
    {
        using Traits = GemmKernelTraits<TypeParam>;

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

        TypeParam ker(c.tile(0, 0), c.spacing());
        ker.template gemm<false, true>(a.tile(0, 0), a.spacing(), b.tile(0, 0), b.spacing());
        ker.store(d.tile(0, 0), d.spacing());
        
        d.unpack(data(md), spacing(md));

        BLAZEFEO_EXPECT_EQ(md, evaluate(mc + ma * trans(mb)));
    }


    REGISTER_TYPED_TEST_SUITE_P(GemmKernelTest,
        testLoadStore,
        testStore,
        testGemmNT
    );


    using GemmKernel_double_1_1_4 = GemmKernel<double, 1, 1, 4>;
    using GemmKernel_double_2_1_4 = GemmKernel<double, 2, 1, 4>;
    using GemmKernel_double_3_1_4 = GemmKernel<double, 3, 1, 4>;

    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_1_1_4, GemmKernelTest, GemmKernel_double_1_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_2_1_4, GemmKernelTest, GemmKernel_double_2_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_3_1_4, GemmKernelTest, GemmKernel_double_3_1_4);
}