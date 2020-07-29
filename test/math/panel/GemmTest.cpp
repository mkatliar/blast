#define BLAZE_USER_ASSERTION 1

#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>


namespace blazefeo :: testing
{
    template <typename T>
    class PanelGemmTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(PanelGemmTest);


    TYPED_TEST_P(PanelGemmTest, testNT)
    {
        using Real = TypeParam;
        size_t const M_max = 20, N_max = 20, K_max = 20;

        for (size_t M = 1; M <= M_max; ++M)
        {
            for (size_t N = 5; N <= N_max; ++N)
            {
                for (size_t K = 1; K <= K_max; ++K)
                {
                    // Init Blaze matrices
                    //
                    blaze::DynamicMatrix<Real, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
                    randomize(blaze_A);
                    randomize(blaze_B);
                    randomize(blaze_C);

                    // Init BlazeFEO matrices
                    //
                    DynamicPanelMatrix<Real> A(M, K);
                    DynamicPanelMatrix<Real> B(N, K);
                    DynamicPanelMatrix<Real> C(M, N);
                    DynamicPanelMatrix<Real> D(M, N);

                    A = blaze_A;
                    B = blaze_B;
                    C = blaze_C;

                    // std::cout << "A=\n" << A << std::endl;
                    // std::cout << "B=\n" << B << std::endl;
                    // std::cout << "C=\n" << C << std::endl;
                    
                    // Do gemm with BlazeFEO
                    gemm_nt(A, B, C, D);

                    // Copy the resulting D matrix from BlazeFEO to Blaze
                    blaze_D = D;

                    // Print the result from BlazeFEO
                    // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

                    BLAZEFEO_ASSERT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), absTol<Real>(), relTol<Real>())
                        << "gemm error at size m,n,k=" << M << "," << N << "," << K;
                }
            }
        }
    }


    TYPED_TEST_P(PanelGemmTest, testNT_submatrix)
    {
        using Real = TypeParam;
        size_t const M = 8, N = 8, K = 3 * 8;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<Real, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init BlazeFEO matrices
        //
        StaticPanelMatrix<Real, M, K> A;
        StaticPanelMatrix<Real, N, K> B;
        StaticPanelMatrix<Real, M, N> C;
        StaticPanelMatrix<Real, M, N> D;

        A = blaze_A;
        B = blaze_B;
        C = blaze_C;
        
        // Do gemm with BlazeFEO
        auto D1 = submatrix(D, 0, 0, M, N);
        gemm_nt(submatrix(A, 0, 0, M, K), submatrix(B, 0, 0, N, K), submatrix(C, 0, 0, M, N), D1);

        // Copy the resulting D matrix from BlazeFEO to Blaze
        blaze_D = D;

        // Print the result from BlazeFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), absTol<Real>(), relTol<Real>());
    }


    REGISTER_TYPED_TEST_SUITE_P(PanelGemmTest,
        testNT,
        testNT_submatrix
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(double, PanelGemmTest, double);
    INSTANTIATE_TYPED_TEST_SUITE_P(float, PanelGemmTest, float);
}