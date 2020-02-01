#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>

namespace blazefeo :: testing
{
    template <typename T>
    class PotrfTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(PotrfTest);


    TYPED_TEST_P(PotrfTest, testDynamicSize)
    {
        using Real = TypeParam;

        for (size_t M = 0; M <= 50; ++M)
        {
            // Init matrices
            //
            DynamicMatrix<Real, columnMajor> blaze_A(M, M), blaze_L(M, M);
            makePositiveDefinite(blaze_A);
            llh(blaze_A, blaze_L);

            DynamicPanelMatrix<Real, columnMajor> A(M, M), L(M, M), A1(M, M);
            A.pack(data(blaze_A), spacing(blaze_A));
            
            // Do potrf
            potrf(A, L);

            // Check result
            A1 = 0.;
            gemm_nt(L, L, A1, A1);

            // std::cout << "L=\n" << L << std::endl;
            // std::cout << "blaze_L=\n" << blaze_L << std::endl;

            BLAZEFEO_EXPECT_APPROX_EQ(A1, A, absTol<Real>(), relTol<Real>()) << "potrf error for size " << M;
        }
    }


    REGISTER_TYPED_TEST_SUITE_P(PotrfTest,
        testDynamicSize
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_double, PotrfTest, double);
    // INSTANTIATE_TYPED_TEST_SUITE_P(Potrf_float, PotrfTest, float);
}