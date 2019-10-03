#include <smoke/StaticMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace smoke :: testing
{
    TEST(StaticMatrixTest, testPanelRows)
    {
        EXPECT_EQ((StaticMatrix<double, 5, 7, 4>().panelRows()), 2);
        EXPECT_EQ((StaticMatrix<double, 8, 7, 4>().panelRows()), 2);
    }


    TEST(StaticMatrixTest, testPanelColumns)
    {
        EXPECT_EQ((StaticMatrix<double, 5, 2, 4>().panelColumns()), 1);
        EXPECT_EQ((StaticMatrix<double, 5, 7, 4>().panelColumns()), 2);
        EXPECT_EQ((StaticMatrix<double, 5, 8, 4>().panelRows()), 2);
    }


    TEST(StaticMatrixTest, testElementAccess)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;

        blaze::StaticMatrix<double, M, N> A_ref;
        randomize(A_ref);

        StaticMatrix<double, M, N, P> A;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                A(i, j) = A_ref(i, j);

        auto const& A_cref = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
            {
                EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
                EXPECT_EQ(A_cref(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
            }
    }


    TEST(StaticMatrixTest, testPack)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;

        blaze::StaticMatrix<double, M, N, blaze::columnMajor> A_ref;
        randomize(A_ref);

        StaticMatrix<double, M, N, P> A;
        A.pack(data(A_ref), spacing(A_ref));

        auto const& A_cref = A;

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TEST(StaticMatrixTest, testUnpack)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;        

        StaticMatrix<double, M, N, P> A;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                blaze::randomize(A(i, j));

        blaze::StaticMatrix<double, M, N, blaze::columnMajor> A1;
        A.unpack(data(A1), spacing(A1));

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A1(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    }


    TEST(StaticMatrixTest, testPanelLoad)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;

        blaze::StaticMatrix<double, M, N> A_ref;
        randomize(A_ref);

        StaticMatrix<double, M, N, P> A;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                A(i, j) = A_ref(i, j);

        for (size_t i = 0; i < A.panelRows(); ++i)
            for (size_t j = 0; j < A.panelColumns(); ++j)
            {
                Panel<double, P> const panel = A.load(i, j);
                
                alignas(Panel<double, P>::alignment) std::array<double, P * P> panel_data;
                panel.store(panel_data.data());

                for (size_t ii = 0; ii < P && i * P + ii < M; ++ii)
                    for (size_t jj = 0; jj < P && j * P + jj < N; ++jj)
                        EXPECT_EQ(panel_data[ii + jj * P], A_ref(i * P + ii, j * P + jj)) 
                            << "element mismatch at (" << i << ", " << j << ", " << ii << ", " << jj << ")";
            }
    }


    TEST(StaticMatrixTest, testPanelStore)
    {
        size_t constexpr M = 5;
        size_t constexpr N = 7;
        size_t constexpr P = 4;
        StaticMatrix<double, M, N, P> A;

        blaze::StaticMatrix<double, M, N> A_ref;
        randomize(A_ref);

        for (size_t i = 0; i < A.panelRows(); ++i)
            for (size_t j = 0; j < A.panelColumns(); ++j)
            {
                alignas(Panel<double, P>::alignment) std::array<double, P * P> panel_data;

                for (size_t ii = 0; ii < P && i * P + ii < M; ++ii)
                    for (size_t jj = 0; jj < P && j * P + jj < N; ++jj)
                        panel_data[ii + jj * P] = A_ref(i * P + ii, j * P + jj);

                A.store(i, j, Panel<double, P>(panel_data.data()));
            }

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                EXPECT_EQ(A(i, j), A_ref(i, j));
    }


    TEST(StaticMatrixTest, testGemmTN)
    {
        size_t const M = 8, N = 8, K = 8;

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
        gemm_tn(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + trans(blaze_A) * blaze_B), 1e-10, 1e-10);
    }


    TEST(StaticMatrixTest, testGemmNN)
    {
        size_t const M = 8, N = 8, K = 8;

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
        gemm_nn(A, B, C, D);

        // Copy the resulting D matrix from BLASFEO to Blaze
        D.unpack(data(blaze_D), spacing(blaze_D));

        // Print the result from BLASFEO
        // std::cout << "blaze_D=\n" << blaze_blasfeo_D;

        SMOKE_EXPECT_APPROX_EQ(blaze_D, evaluate(blaze_C + blaze_A * blaze_B), 1e-10, 1e-10);
    }
}