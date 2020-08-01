#include <blazefeo/math/dense/Trmm.hpp>
#include <blazefeo/math/dense/Gemm.hpp>

#include <blazefeo/Blaze.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(DenseTrmmTest, testLeftUpper)
    {
        for (size_t m = 1; m <= 20; m += 1)
            for (size_t n = 1; n <= 20; n += 1)
                for (size_t k = 1; k <= 20; ++k)
                {
                    // Init Blaze matrices
                    //
                    DynamicMatrix<double, columnMajor> A(m, k);
                    DynamicMatrix<double, rowMajor> B(k, n);
                    DynamicMatrix<double, columnMajor> C(m, n);
                    randomize(A);
                    randomize(B);

                    // Reset lower-triangular part of A
                    for (size_t i = 0; i < m; ++i)
                        for (size_t j = 0; j < i && j < k; ++j)
                            reset(A(i, j));

                    double alpha {};
                    blaze::randomize(alpha);

                    // Do trmm
                    blazefeo::trmm<Side::Left, UpLo::Upper>(alpha, A, B, C);

                    BLAZEFEO_ASSERT_APPROX_EQ(C, evaluate(alpha * A * B), 1e-10, 1e-10)
                        << "trmm error at size m,n,k=" << m << "," << n << "," << k;
                }
    }
}