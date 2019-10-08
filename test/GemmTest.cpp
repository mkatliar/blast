#include <smoke/StaticMatrix.hpp>
#include <smoke/Gemm.hpp>
#include <smoke/GemmKernel_double_1_1_4.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace smoke :: testing
{
    TEST(GemmTest, testTN)
    {
        size_t const M = 8, N = 8, K = 3 * 8;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(K, M), blaze_B(K, N), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticMatrix<double, K, M> A;
        StaticMatrix<double, K, N> B;
        StaticMatrix<double, M, N> C;
        StaticMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_tn<1, 1>(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + trans(blaze_A) * blaze_B), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNN)
    {
        size_t const M = 8, N = 8, K = 3 * 8;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(K, N), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticMatrix<double, M, K> A;
        StaticMatrix<double, K, N> B;
        StaticMatrix<double, M, N> C;
        StaticMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nn<1, 1>(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * blaze_B), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT)
    {
        size_t const M = 8, N = 8, K = 3 * 8;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticMatrix<double, M, K> A;
        StaticMatrix<double, N, K> B;
        StaticMatrix<double, M, N> C;
        StaticMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt<1, 1>(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }
}