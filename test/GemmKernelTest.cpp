#include <smoke/GemmKernel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

using namespace blaze;


namespace smoke :: testing
{
    TEST(GemmKernelTest, testGemmTN)
    {
        size_t constexpr N = 4;

        alignas(GemmKernel<double, 1, 1, N>::alignment) std::array<double, N * N> a, b, c;
        randomize(a);
        randomize(b);
        randomize(c);

        CustomMatrix<double, unaligned, unpadded, columnMajor> ma(a.data(), N, N), mb(b.data(), N, N), mc(c.data(), N, N);
        StaticMatrix<double, N, N> const mc0 = mc;

        GemmKernel<double, 1, 1, N> kc;
        kc.load(c.data(), c.size());
        kc(a.data(), true, b.data(), false);
        kc.store(c.data(), c.size());

        SMOKE_EXPECT_EQ(mc, evaluate(mc0 + trans(ma) * mb));
    }


    TEST(GemmKernelTest, testGemmNN)
    {
        size_t constexpr N = 4;

        alignas(GemmKernel<double, 1, 1, N>::alignment) std::array<double, N * N> a, b, c;
        randomize(a);
        randomize(b);
        randomize(c);

        CustomMatrix<double, unaligned, unpadded, columnMajor> ma(a.data(), N, N), mb(b.data(), N, N), mc(c.data(), N, N);
        StaticMatrix<double, N, N> const mc0 = mc;

        GemmKernel<double, 1, 1, N> kc;
        kc.load(c.data(), c.size());
        kc(a.data(), false, b.data(), false);
        kc.store(c.data(), c.size());

        SMOKE_EXPECT_EQ(mc, evaluate(mc0 + ma * mb));
    }


    TEST(GemmKernelTest, testGemmNT)
    {
        size_t constexpr N = 4;

        alignas(GemmKernel<double, 1, 1, N>::alignment) std::array<double, N * N> a, b, c;
        randomize(a);
        randomize(b);
        randomize(c);

        CustomMatrix<double, unaligned, unpadded, columnMajor> ma(a.data(), N, N), mb(b.data(), N, N), mc(c.data(), N, N);
        StaticMatrix<double, N, N> const mc0 = mc;

        GemmKernel<double, 1, 1, N> kc;
        kc.load(c.data(), c.size());
        kc(a.data(), false, b.data(), true);
        kc.store(c.data(), c.size());

        SMOKE_EXPECT_EQ(mc, evaluate(mc0 + ma * trans(mb)));
    }
}