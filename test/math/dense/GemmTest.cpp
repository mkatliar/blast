#define BLAZE_USER_ASSERTION 1

#include <blazefeo/math/dense/Gemm.hpp>

#include <blaze/Math.h>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

namespace blazefeo :: testing
{
    TEST(DenseGemmTest, testNT_dynamic)
    {
        for (size_t m = 1; m <= 20; m += 1)
            for (size_t n = 1; n <= 20; n += 1)
                for (size_t k = 1; k <= 20; ++k)
                {
                    // Init Blaze matrices
                    //
                    DynamicMatrix<double, columnMajor> A(m, k), B(n, k), C(m, n), D(m, n);
                    randomize(A);
                    randomize(B);
                    randomize(C);

                    // std::cout << "A=\n" << A << std::endl;
                    // std::cout << "B=\n" << B << std::endl;
                    // std::cout << "C=\n" << C << std::endl;
                    // std::cout << "C+A*trans(B)=\n" << C + A * trans(B) << std::endl;

                    // Do gemm
                    gemm(1., A, trans(B), 1., C, D);

                    // Print the result from BLASFEO
                    // std::cout << "D=\n" << blaze_blasfeo_D;

                    BLAZEFEO_ASSERT_APPROX_EQ(D, evaluate(C + A * trans(B)), 1e-10, 1e-10)
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