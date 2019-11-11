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
    template <typename Ker>
    class GemmTest
    :   public Test
    {
    };


    TYPED_TEST_SUITE_P(GemmTest);


    TYPED_TEST_P(GemmTest, testNT)
    {
        using Real = TypeParam;
        size_t const M_max = 20, N_max = 20, K_max = 20;

        for (size_t M = 1; M <= M_max; ++M)
        {
            for (size_t N = 1; N <= N_max; ++N)
            {
                for (size_t K = 1; K <= K_max; ++K)
                {
                    // Init Blaze matrices
                    //
                    blaze::DynamicMatrix<Real, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
                    randomize(blaze_A);
                    randomize(blaze_B);
                    randomize(blaze_C);

                    // Init Smoke matrices
                    //
                    DynamicPanelMatrix<Real> A(M, K);
                    DynamicPanelMatrix<Real> B(N, K);
                    DynamicPanelMatrix<Real> C(M, N);
                    DynamicPanelMatrix<Real> D(M, N);

                    A.pack(data(blaze_A), spacing(blaze_A));
                    B.pack(data(blaze_B), spacing(blaze_B));
                    C.pack(data(blaze_C), spacing(blaze_C));
                    
                    // Do gemm with Smoke
                    gemm_nt(A, B, C, D);

                    // Copy the resulting D matrix from BLASFEO to Blaze
                    D.unpack(data(blaze_D), spacing(blaze_D));

                    // Print the result from BLASFEO
                    // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

                    BLAZEFEO_ASSERT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), absTol<Real>(), relTol<Real>())
                        << "gemm error at size m,n,k=" << M << "," << N << "," << K;
                }
            }
        }
    }


    TYPED_TEST_P(GemmTest, testNT_submatrix)
    {
        using Real = TypeParam;
        size_t const M = 8, N = 8, K = 3 * 8;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<Real, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticPanelMatrix<Real, M, K> A;
        StaticPanelMatrix<Real, N, K> B;
        StaticPanelMatrix<Real, M, N> C;
        StaticPanelMatrix<Real, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        auto D1 = submatrix(D, 0, 0, M, N);
        gemm_nt(submatrix(A, 0, 0, M, K), submatrix(B, 0, 0, N, K), submatrix(C, 0, 0, M, N), D1);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), absTol<Real>(), relTol<Real>());
    }


    REGISTER_TYPED_TEST_SUITE_P(GemmTest,
        testNT,
        testNT_submatrix
    );


    INSTANTIATE_TYPED_TEST_SUITE_P(Gemm_double, GemmTest, double);
    INSTANTIATE_TYPED_TEST_SUITE_P(Gemm_float, GemmTest, float);
}