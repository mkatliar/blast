#include <smoke/StaticPanelMatrix.hpp>
#include <smoke/Gemm.hpp>
#include <smoke/gemm/GemmKernel_double_1_1_4.hpp>
#include <smoke/gemm/GemmKernel_double_2_1_4.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace smoke :: testing
{
#if 0
    TEST(GemmTest, testTN_8_8_24)
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
        StaticPanelMatrix<double, K, M> A;
        StaticPanelMatrix<double, K, N> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm(GemmKernel<double, 1, 1, 4, true, false> {}, A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + trans(blaze_A) * blaze_B), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNN_8_8_24)
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
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, K, N> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm(GemmKernel<double, 1, 1, 4, false, false> {}, A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * blaze_B), 1e-10, 1e-10);
    }
#endif


    TEST(GemmTest, testNT_8_8_24)
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
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, N, K> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT_12_8_24)
    {
        size_t const M = 12, N = 8, K = 3 * 8;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, N, K> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT_19_15_17)
    {
        size_t const M = 19, N = 15, K = 17;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, N, K> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT_12_1_2)
    {
        size_t const M = 12, N = 1, K = 2;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, N, K> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT_3_2_11)
    {
        size_t const M = 3, N = 2, K = 11;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, N, K> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT_19_19_19)
    {
        size_t const M = 19, N = 19, K = 19;

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
        randomize(blaze_A);
        randomize(blaze_B);
        randomize(blaze_C);

        // Init Smoke matrices
        //
        StaticPanelMatrix<double, M, K> A;
        StaticPanelMatrix<double, N, K> B;
        StaticPanelMatrix<double, M, N> C;
        StaticPanelMatrix<double, M, N> D;

        A.pack(data(blaze_A), spacing(blaze_A));
        B.pack(data(blaze_B), spacing(blaze_B));
        C.pack(data(blaze_C), spacing(blaze_C));
        
        // Do gemm with Smoke
        gemm_nt(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }
}