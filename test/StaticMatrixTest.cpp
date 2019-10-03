#include <smoke/StaticMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>

using namespace blaze;


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


    // TEST(StaticMatrixTest, testPanelLoad)
    // {
    //     size_t constexpr M = 5;
    //     size_t constexpr N = 7;
    //     size_t constexpr P = 4;

    //     blaze::StaticMatrix<double, M, N> A_ref;
    //     randomize(A_ref);

    //     StaticMatrix<double, M, N, P> A;
    //     for (size_t i = 0; i < M; ++i)
    //         for (size_t j = 0; j < N; ++j)
    //             A(i, j) = A_ref(i, j);

    //     auto const& A_cref = A;

    //     for (size_t i = 0; i < M; ++i)
    //         for (size_t j = 0; j < N; ++j)
    //         {
    //             EXPECT_EQ(A(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    //             EXPECT_EQ(A_cref(i, j), A_ref(i, j)) << "element mismatch at (" << i << ", " << j << ")";
    //         }
    // }
}