#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/Panel.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blazefeo :: testing
{
    TEST(SubmatrixTest, testSubmatrixOfStaticPanelMatrix)
    {
        StaticPanelMatrix<double, 12, 12, rowMajor> A;
        auto B = submatrix(A, 4, 0, 8, 8);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfConstStaticPanelMatrix)
    {
        StaticPanelMatrix<double, 12, 12, rowMajor> const A;
        auto B = submatrix(A, 4, 0, 8, 8);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfDynamicPanelMatrix)
    {
        DynamicPanelMatrix<double, rowMajor> A(12, 12);
        auto B = submatrix(A, 4, 0, 8, 8);

        static_assert(std::is_same_v<decltype(tile(B, 0, 0)), double *>);
        tile(B, 0, 0);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfConstDynamicPanelMatrix)
    {
        DynamicPanelMatrix<double, rowMajor> const A(12, 12);
        auto B = submatrix(A, 4, 0, 8, 8);

        static_assert(std::is_same_v<decltype(tile(B, 0, 0)), double const *>);
        tile(B, 0, 0);
        
        // PanelSubmatrix<decltype(A), rowMajor> B(A, 4, 0, 8, 8);
        std::cout << B << std::endl;
    }


    TEST(SubmatrixTest, testSubmatrixOfSubmatrix)
    {
        DynamicPanelMatrix<double, rowMajor> A(12, 12);
        for (size_t i = 0; i < rows(A); ++i)
            for (size_t j = 0; j < columns(A); ++j)
                A(i, j) = -(100. * i + j);

        auto B = submatrix(A, 4, 0, 8, 8);
        ASSERT_EQ(&B.operand(), &A);

        auto B1 = submatrix(B, 4, 1, 2, 2);
        EXPECT_EQ(&B1.operand(), &A);
        EXPECT_EQ(B1.row(), B.row() + 4);
        EXPECT_EQ(B1.column(), B.column() + 1);

        static_assert(std::is_same_v<decltype(B), decltype(B1)>);

        for (size_t i = 0; i < rows(B1); ++i)
            for (size_t j = 0; j < columns(B1); ++j)
                B1(i, j) = 100. * i + j;

        for (size_t i = 0; i < rows(B1); ++i)
            for (size_t j = 0; j < columns(B1); ++j)
            {
                auto const val = 100. * i + j;
                ASSERT_EQ(B1(i, j), val);
                ASSERT_EQ(A(i + B1.row(), j + B1.column()), val);
            }
    }


    TEST(SubmatrixTest, testTile)
    {
        DynamicPanelMatrix<double, rowMajor> A(12, 12);
        A = 0.;
        auto B = submatrix(A, 4, 0, 8, 8);

        *tile(B, 0, 0) = 1.;
        *tile(B, 1, 0) = 2.;
        *tile(B, 0, 1) = 3.;
        *tile(B, 1, 1) = 4.;
        
        EXPECT_EQ(A(4, 0), 1.);
        EXPECT_EQ(A(8, 0), 2.);
        EXPECT_EQ(A(4, 4), 3.);
        EXPECT_EQ(A(8, 4), 4.);
    }
}