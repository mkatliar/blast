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


    TYPED_TEST_P(DensePotrtTest, testDynamic)
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
            blazefeo::potrf(A, L);
            // std::cout << "L=\n" << L << std::endl;

            // Check result
            DynamicMatrix<Real> L1;
            llh(A, L1);
            BLAZEFEO_EXPECT_APPROX_EQ(L, L1, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    TYPED_TEST_P(DensePotrtTest, testStatic)
    {
        using Real = TypeParam;

        size_t const M = 20;
        
        // Init matrices
        //
        StaticMatrix<Real, M, M, columnMajor> A, L;
        makePositiveDefinite(A);
        reset(L);
        
        // Do potrf
        blazefeo::potrf(A, L);
        // std::cout << "L=\n" << L << std::endl;

        // Check result
        DynamicMatrix<Real> L1;
        llh(A, L1);
        BLAZEFEO_EXPECT_APPROX_EQ(L, L1, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
    }


    TYPED_TEST_P(DensePotrtTest, testStaticInplace)
    {
        using Real = TypeParam;

        size_t const M = 3;
        
        // Init matrices
        //
        StaticMatrix<Real, M, M, columnMajor> A;
        makePositiveDefinite(A);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = i + 1; j < M; ++j)
                reset(A(i, j));

        // True result
        DynamicMatrix<Real> L1;
        llh(A, L1);
        
        // Do potrf
        blazefeo::potrf(A, A);
        // std::cout << "L=\n" << L << std::endl;

        // Check result
        BLAZEFEO_EXPECT_APPROX_EQ(A, L1, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
    }


    REGISTER_TYPED_TEST_SUITE_P(DensePotrtTest
        , testDynamic
        , testStatic
        , testStaticInplace
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, DensePotrtTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, DensePotrtTest, float);
}