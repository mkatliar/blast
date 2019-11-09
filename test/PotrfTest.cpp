#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/panel/Potrf.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(PotrfTest, testDynamicSize)
    {
        for (size_t M = 0; M <= 50; ++M)
        {
            // Init matrices
            //
            DynamicMatrix<double, columnMajor> blaze_A(M, M), blaze_L(M, M);
            makePositiveDefinite(blaze_A);
            llh(blaze_A, blaze_L);

            DynamicPanelMatrix<double, rowMajor> A(M, M), L(M, M), A1(M, M);
            A.pack(data(blaze_A), spacing(blaze_A));
            
            // Do potrf
            potrf(A, L);

            // Check result
            A1 = 0.;
            gemm_nt(L, L, A1, A1);

            // std::cout << "L=\n" << L << std::endl;
            // std::cout << "blaze_L=\n" << blaze_L << std::endl;

            BLAZEFEO_EXPECT_APPROX_EQ(A1, A, 1e-14, 1e-14) << "potrf error for size " << M;
        }
    }
}