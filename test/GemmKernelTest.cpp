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


    TYPED_TEST_P(GemmKernelTest, testGemmTN)
    {
        using Traits = GemmKernelTraits<TypeParam>;
        size_t constexpr K = 3 * Traits::blockSize;

        blaze::StaticMatrix<double, K, Traits::rows, blaze::columnMajor> ma;
        blaze::StaticMatrix<double, K, Traits::columns, blaze::columnMajor> mb;
        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;
        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticMatrix<double, K, Traits::rows, Traits::blockSize, Traits::alignment> a;
        StaticMatrix<double, K, Traits::columns, Traits::blockSize, Traits::alignment> b;
        StaticMatrix<double, Traits::rows, Traits::columns, Traits::blockSize, Traits::alignment> c, d;
        a.pack(data(ma), spacing(ma));
        b.pack(data(mb), spacing(mb));
        c.pack(data(mc), spacing(mc));
        
        gemm(TypeParam {}, K,
            a.block(0, 0), a.spacing(), true, b.block(0, 0), b.spacing(), false,
            c.block(0, 0), c.spacing(), d.block(0, 0), d.spacing());
        d.unpack(data(md), spacing(md));

        SMOKE_EXPECT_EQ(md, evaluate(mc + trans(ma) * mb));
    }


    TYPED_TEST_P(GemmKernelTest, testGemmNN)
    {
        using Traits = GemmKernelTraits<TypeParam>;
        size_t constexpr K = 3 * Traits::blockSize;

        blaze::StaticMatrix<double, Traits::rows, K, blaze::columnMajor> ma;
        blaze::StaticMatrix<double, K, Traits::columns, blaze::columnMajor> mb;
        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;
        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticMatrix<double, Traits::rows, K, Traits::blockSize, Traits::alignment> a;
        StaticMatrix<double, K, Traits::columns, Traits::blockSize, Traits::alignment> b;
        StaticMatrix<double, Traits::rows, Traits::columns, Traits::blockSize, Traits::alignment> c, d;
        a.pack(data(ma), spacing(ma));
        b.pack(data(mb), spacing(mb));
        c.pack(data(mc), spacing(mc));
        
        gemm(TypeParam {}, K, 
            a.block(0, 0), a.spacing(), false, b.block(0, 0), b.spacing(), false,
            c.block(0, 0), c.spacing(), d.block(0, 0), d.spacing());
        d.unpack(data(md), spacing(md));

        SMOKE_EXPECT_EQ(md, evaluate(mc + ma * mb));
    }


    TYPED_TEST_P(GemmKernelTest, testGemmNT)
    {
        using Traits = GemmKernelTraits<TypeParam>;
        size_t constexpr K = 5 * Traits::blockSize;

        blaze::StaticMatrix<double, Traits::rows, K, blaze::columnMajor> ma;
        blaze::StaticMatrix<double, Traits::columns, K, blaze::columnMajor> mb;
        blaze::StaticMatrix<double, Traits::rows, Traits::columns, blaze::columnMajor> mc, md;
        randomize(ma);
        randomize(mb);
        randomize(mc);

        StaticMatrix<double, Traits::rows, K, Traits::blockSize, Traits::alignment> a;
        StaticMatrix<double, Traits::columns, K, Traits::blockSize, Traits::alignment> b;
        StaticMatrix<double, Traits::rows, Traits::columns, Traits::blockSize, Traits::alignment> c, d;
        a.pack(data(ma), spacing(ma));
        b.pack(data(mb), spacing(mb));
        c.pack(data(mc), spacing(mc));
        
        gemm(TypeParam {}, K, 
            a.block(0, 0), a.spacing(), false, b.block(0, 0), b.spacing(), true,
            c.block(0, 0), c.spacing(), d.block(0, 0), d.spacing());
        d.unpack(data(md), spacing(md));

        SMOKE_EXPECT_EQ(md, evaluate(mc + ma * trans(mb)));
    }


    REGISTER_TYPED_TEST_SUITE_P(GemmKernelTest,
        testGemmTN,
        testGemmNN,
        testGemmNT
    );


    using GemmKernel_double_1_1_4 = GemmKernel<double, 1, 1, 4>;
    using GemmKernel_double_2_1_4 = GemmKernel<double, 2, 1, 4>;
    using GemmKernel_double_3_1_4 = GemmKernel<double, 3, 1, 4>;

    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_1_1_4, GemmKernelTest, GemmKernel_double_1_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_2_1_4, GemmKernelTest, GemmKernel_double_2_1_4);
    INSTANTIATE_TYPED_TEST_SUITE_P(GemmKernel_double_3_1_4, GemmKernelTest, GemmKernel_double_3_1_4);
}