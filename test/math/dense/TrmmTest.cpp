#include <blazefeo/math/dense/Trmm.hpp>
#include <blazefeo/math/dense/Gemm.hpp>

#include <blazefeo/Blaze.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(DenseTrmmTest, testNT_dynamic)
    {
        for (size_t m = 1; m <= 20; m += 1)
            for (size_t n = 1; n <= 20; n += 1)
            {
                // Init Blaze matrices
                //
                DynamicMatrix<double, columnMajor> A(m, m), B(n, m), C(m, n);
                randomize(A);
                randomize(B);

                double alpha {};
                blaze::randomize(alpha);

                // Do trmm
                blazefeo::trmm(alpha, A, trans(B), C);

                BLAZEFEO_ASSERT_APPROX_EQ(C, evaluate(alpha * A * trans(B)), 1e-10, 1e-10)
                    << "trmm error at size m,n=" << m << "," << n;
            }
    }
}