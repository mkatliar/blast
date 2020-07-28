#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/dense/Potrf.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>


namespace blazefeo :: testing
{
    template <typename T>
    class DensePotrtTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(DensePotrtTest);


    TYPED_TEST_P(DensePotrtTest, testDynamicSize)
    {
        using Real = TypeParam;

        for (size_t M = 0; M <= 50; ++M)
        {
            // Init matrices
            //
            DynamicMatrix<Real, columnMajor> A(M, M), L(M, M);
            makePositiveDefinite(A);
            reset(L);
            
            // Do potrf
            potrf(A, L);
            // std::cout << "L=\n" << L << std::endl;

            // Check result
            DynamicMatrix<Real> L1;
            llh(A, L1);
            BLAZEFEO_EXPECT_APPROX_EQ(L, L1, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    REGISTER_TYPED_TEST_SUITE_P(DensePotrtTest,
        testDynamicSize
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DensePotrtTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, DensePotrtTest, float);
}