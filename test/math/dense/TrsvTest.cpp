#include <blazefeo/math/dense/Trsv.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    TEST(DenseTrsvTest, testLeftLower)
    {
        for (size_t n = 1; n <= 20; ++n)
        {
            // Init Blaze matrices
            //
            LowerMatrix<DynamicMatrix<double>> A(n, n);
            DynamicVector<double> b(n);
            DynamicVector<double> x(n);
            randomize(A);
            A += IdentityMatrix<double>(n); // Improve conditioning
            randomize(b);

            // Do trsv
            trsvLeftLower(A, b, x);

            BLAZEFEO_ASSERT_APPROX_EQ(evaluate(A * x), b, 1e-10, 1e-10)
                << "trsv error at size n=" << n;
        }
    }


    TEST(DenseTrsvTest, testLeftUpper)
    {
        for (size_t n = 1; n <= 20; ++n)
        {
            // Init Blaze matrices
            //
            UpperMatrix<DynamicMatrix<double>> A(n, n);
            DynamicVector<double> b(n);
            DynamicVector<double> x(n);
            randomize(A);
            A += IdentityMatrix<double>(n); // Improve conditioning
            randomize(b);

            // Do trsv
            trsvLeftUpper(A, b, x);

            BLAZEFEO_ASSERT_APPROX_EQ(evaluate(A * x), b, 1e-10, 1e-10)
                << "trsv error at size n=" << n;
        }
    }
}