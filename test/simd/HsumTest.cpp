#include <blazefeo/math/simd/Hsum.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

#include <numeric>

using namespace blaze;


namespace blazefeo :: testing
{
    TEST(HsumTest, test4x256d)
    {
        std::array<double, 4> a, b, c, d, r;
        randomize(a);
        randomize(b);
        randomize(c);
        randomize(d);

        __m256d mma = _mm256_loadu_pd(a.data());
        __m256d mmb = _mm256_loadu_pd(b.data());
        __m256d mmc = _mm256_loadu_pd(c.data());
        __m256d mmd = _mm256_loadu_pd(d.data());        

        __m256d mmr = hsum(mma, mmb, mmc, mmd);
        _mm256_storeu_pd(r.data(), mmr);

        EXPECT_NEAR(r[0], std::accumulate(begin(a), end(a), 0.), 1.e-10);
        EXPECT_NEAR(r[1], std::accumulate(begin(b), end(b), 0.), 1.e-10);
        EXPECT_NEAR(r[2], std::accumulate(begin(c), end(c), 0.), 1.e-10);
        EXPECT_NEAR(r[3], std::accumulate(begin(d), end(d), 0.), 1.e-10);
    }
}