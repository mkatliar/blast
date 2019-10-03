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
}