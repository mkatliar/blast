#define BLAZE_USER_ASSERTION 1

#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>
#include <blazefeo/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
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
        gemm(RegisterMatrix<double, 1, 1, 4, true, false> {}, A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + trans(blaze_A) * blaze_B), 1e-10, 1e-10);
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
        gemm(RegisterMatrix<double, 1, 1, 4, false, false> {}, A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * blaze_B), 1e-10, 1e-10);
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

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
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

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
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

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
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

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
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

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
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

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }


    TEST(GemmTest, testNT_dynamic)
    {
        size_t const M_max = 20, N_max = 20, K_max = 20;

        for (size_t M = 1; M <= M_max; ++M)
            for (size_t N = 1; N <= N_max; ++N)
                for (size_t K = 1; K <= K_max; ++K)
                {
                    // Init Blaze matrices
                    //
                    blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(M, K), blaze_B(N, K), blaze_C(M, N), blaze_D(M, N);
                    randomize(blaze_A);
                    randomize(blaze_B);
                    randomize(blaze_C);

                    // Init Smoke matrices
                    //
                    DynamicPanelMatrix<double> A(M, K);
                    DynamicPanelMatrix<double> B(N, K);
                    DynamicPanelMatrix<double> C(M, N);
                    DynamicPanelMatrix<double> D(M, N);

                    A.pack(data(blaze_A), spacing(blaze_A));
                    B.pack(data(blaze_B), spacing(blaze_B));
                    C.pack(data(blaze_C), spacing(blaze_C));
                    
                    // Do gemm with Smoke
                    gemm_nt(A, B, C, D);

                    // Copy the resulting D matrix from BLASFEO to Blaze
                    D.unpack(data(blaze_D), spacing(blaze_D));

                    // Print the result from BLASFEO
                    // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

                    BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
                }
    }


    TEST(GemmTest, testNT_submatrix)
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
        auto D1 = submatrix(D, 0, 0, M, N);
        gemm_nt(submatrix(A, 0, 0, M, K), submatrix(B, 0, 0, N, K), submatrix(C, 0, 0, M, N), D1);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        BLAZEFEO_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * trans(blaze_B)), 1e-10, 1e-10);
    }
}