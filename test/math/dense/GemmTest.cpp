#define BLAZE_USER_ASSERTION 1

#include <blazefeo/math/dense/Gemm.hpp>

#include <blazefeo/Blaze.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(DenseGemmTest, testCr)
    {
        for (size_t m = 1; m <= 20; m += 1)
            for (size_t n = 1; n <= 20; n += 1)
                for (size_t k = 1; k <= 20; ++k)
                {
                    // Init Blaze matrices
                    //
                    DynamicMatrix<double, columnMajor> A(m, k), C(m, n), D(m, n);
                    DynamicMatrix<double, rowMajor> B(k, n);
                    randomize(A);
                    randomize(B);
                    randomize(C);

                    double alpha {}, beta {};
                    blaze::randomize(alpha);
                    blaze::randomize(beta);

                    /// Do gemm
                    gemm(alpha, A, B, beta, C, D);

                    BLAZEFEO_ASSERT_APPROX_EQ(D, evaluate(beta * C + alpha * A * B), 1e-10, 1e-10)
                        << "gemm error at size m,n,k=" << m << "," << n << "," << k;
                }
    }


    TEST(DenseGemmTest, testCc)
    {
        for (size_t m = 1; m <= 20; m += 1)
            for (size_t n = 1; n <= 20; n += 1)
                for (size_t k = 1; k <= 20; ++k)
                {
                    // Init Blaze matrices
                    //
                    DynamicMatrix<double, columnMajor> A(m, k), C(m, n), D(m, n);
                    DynamicMatrix<double, columnMajor> B(k, n);
                    randomize(A);
                    randomize(B);
                    randomize(C);

                    double alpha {}, beta {};
                    blaze::randomize(alpha);
                    blaze::randomize(beta);

                    // std::cout << "A=\n" << A << std::endl;
                    // std::cout << "B=\n" << B << std::endl;
                    // std::cout << "C=\n" << C << std::endl;
                    // std::cout << "C+A*trans(B)=\n" << C + A * trans(B) << std::endl;

                    // Do gemm
                    gemm(alpha, A, B, beta, C, D);

                    // Print the result from BLASFEO
                    // std::cout << "D=\n" << blaze_blasfeo_D;

                    BLAZEFEO_ASSERT_APPROX_EQ(D, evaluate(beta * C + alpha * A * B), 1e-10, 1e-10)
                        << "gemm error at size m,n,k=" << m << "," << n << "," << k;
                }
    }


    // TEST(DenseGemmTest, testNT_submatrix)
    // {
    //     size_t const M = 8, N = 8, K = 3 * 8;

    //     // Init Blaze matrices
    //     //
    //     DynamicMatrix<double, columnMajor> A(M, K), B(N, K), C(M, N), D(M, N);
    //     randomize(A);
    //     randomize(B);
    //     randomize(C);

    //     // Init Smoke matrices
    //     //
    //     StaticPanelMatrix<double, M, K> A;
    //     StaticPanelMatrix<double, N, K> B;
    //     StaticPanelMatrix<double, M, N> C;
    //     StaticPanelMatrix<double, M, N> D;

    //     A.pack(data(A), spacing(A));
    //     B.pack(data(B), spacing(B));
    //     C.pack(data(C), spacing(C));
        
    //     // Do gemm with Smoke
    //     auto D1 = submatrix(D, 0, 0, M, N);
    //     gemm_nt(submatrix(A, 0, 0, M, K), submatrix(B, 0, 0, N, K), submatrix(C, 0, 0, M, N), D1);

    //     // Copy the resulting D matrix from BLASFEO to Blaze
    //     D.unpack(data(D), spacing(D));

    //     // Print the result from BLASFEO
    //     // std::cout << "D=\n" << blaze_blasfeo_D;

    //     BLAZEFEO_EXPECT_APPROX_EQ(D, evaluate(C + A * trans(B)), 1e-10, 1e-10);
    // }
}