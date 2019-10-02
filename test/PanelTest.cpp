#include <smoke/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

using namespace blaze;


namespace smoke :: testing
{
    TEST(PanelTest, testMultiply)
    {
        size_t constexpr N = 4;

        std::array<double, N * N> a, b, c;
        randomize(a);
        randomize(b);
        randomize(c);

        CustomMatrix<double, unaligned, unpadded, columnMajor> ma(a.data(), N, N), mb(b.data(), N, N), mc(c.data(), N, N);
        StaticMatrix<double, N, N> const mc0 = mc;

        Panel<double, N> pa, pb, pc;
        pa.load(a.data());
        pb.load(b.data());
        pc.load(c.data());

        gemm(pa, pb, pc);

        pc.store(c.data());

        SMOKE_ASSERT_EQ(mc, evaluate(mc0 + ma * mb));
    }
}