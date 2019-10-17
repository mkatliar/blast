#include <smoke/GemmKernel_double_1_1_4.hpp>
#include <smoke/GemmKernel_double_2_1_4.hpp>
#include <smoke/GemmKernel_double_3_1_4.hpp>
#include <smoke/StaticMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace smoke :: testing
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

        StaticMatrix<double, Traits::rows, Traits::columns, Traits::blockSize, Traits::alignment> A, B;
        A.pack(data(A_ref), spacing(A_ref));

        TypeParam ker;
        ker.load(A.block(0, 0), A.spacing());
        ker.store(B.block(0, 0), B.spacing());

        for (size_t i = 0; i < Traits::rows; ++i)
            for (size_t j = 0; j < Traits::columns; ++j)
                EXPECT_EQ(B(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }
    TYPED_TEST_P(GemmKernelTest, testGemm)
    {
        using Traits = GemmKernelTraits<TypeParam>;
        size_t constexpr K = 3 * Traits::blockSize;

        blaze::DynamicMatrix<double, blaze::columnMajor> ma(
            Traits::tA ? K : Traits::rows,
            Traits::tA ? Traits::rows : K
        );

        blaze::DynamicMatrix<double, blaze::columnMajor> mb(
            Traits::tB ? Traits::columns : K,
            Traits::tB ? K : Traits::columns
        );

        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;

        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticMatrix<double, 
            Traits::tA ? K : Traits::rows,
            Traits::tA ? Traits::rows : K,
            Traits::blockSize, Traits::alignment> a;

        StaticMatrix<double, 
            Traits::tB ? Traits::columns : K,
            Traits::tB ? K : Traits::columns,
            Traits::blockSize, Traits::alignment> b;

        StaticMatrix<double, Traits::rows, Traits::columns, Traits::blockSize, Traits::alignment> c, d;
        a.pack(data(ma), spacing(ma));
        b.pack(data(mb), spacing(mb));
        c.pack(data(mc), spacing(mc));

        TypeParam ker(c.block(0, 0), c.spacing());
        ker(K, a.block(0, 0), a.spacing(), b.block(0, 0), b.spacing());
        ker.store(d.block(0, 0), d.spacing());
        
        d.unpack(data(md), spacing(md));

        if (Traits::tA)
            ma = trans(ma);

        if (Traits::tB)
            mb = trans(mb);

        SMOKE_EXPECT_EQ(md, evaluate(mc + ma * mb));
    }


    REGISTER_TYPED_TEST_SUITE_P(GemmKernelTest,
        testLoadStore,
        testGemm
    );


    using GemmKernel_double_1_1_4_TN = GemmKernel<double, 1, 1, 4, true, false>;
    using GemmKernel_double_1_1_4_NN = GemmKernel<double, 1, 1, 4, false, false>;
    using GemmKernel_double_1_1_4_NT = GemmKernel<double, 1, 1, 4, false, true>;
    using GemmKernel_double_2_1_4_TN = GemmKernel<double, 2, 1, 4, true, false>;
    using GemmKernel_double_2_1_4_NN = GemmKernel<double, 2, 1, 4, false, false>;
    using GemmKernel_double_2_1_4_NT = GemmKernel<double, 2, 1, 4, false, true>;
    using GemmKernel_double_3_1_4_TN = GemmKernel<double, 3, 1, 4, true, false>;
    using GemmKernel_double_3_1_4_NN = GemmKernel<double, 3, 1, 4, false, false>;
    using GemmKernel_double_3_1_4_NT = GemmKernel<double, 3, 1, 4, false, true>;

    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_1_1_4_TN, GemmKernelTest, GemmKernel_double_1_1_4_TN);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_1_1_4_NN, GemmKernelTest, GemmKernel_double_1_1_4_NN);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_1_1_4_NT, GemmKernelTest, GemmKernel_double_1_1_4_NT);
    
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_2_1_4_TN, GemmKernelTest, GemmKernel_double_2_1_4_TN);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_2_1_4_NN, GemmKernelTest, GemmKernel_double_2_1_4_NN);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_2_1_4_NT, GemmKernelTest, GemmKernel_double_2_1_4_NT);

    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_3_1_4_TN, GemmKernelTest, GemmKernel_double_3_1_4_TN);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_3_1_4_NN, GemmKernelTest, GemmKernel_double_3_1_4_NN);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_3_1_4_NT, GemmKernelTest, GemmKernel_double_3_1_4_NT);
}