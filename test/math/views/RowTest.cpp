#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blazefeo :: testing
{
    TEST(RowTest, testSize)
    {
        StaticPanelMatrix<double, 12, 12, columnMajor> A;
        randomize(A);

        for (size_t i = 0; i < rows(A); ++i)
        {
            auto r = row(A, i);
            ASSERT_EQ(r.size(), A.columns());
        }
    }


    TEST(RowTest, testElementRead)
    {
        StaticPanelMatrix<double, 12, 12, columnMajor> A;
        randomize(A);

        for (size_t i = 0; i < rows(A); ++i)
        {
            {
                auto r = row(A, i);
                for (size_t j = 0; j < A.columns(); ++j)
                    ASSERT_EQ(r[j], A(i, j));
            }

            {
                auto r = row(std::as_const(A), i);
                for (size_t j = 0; j < A.columns(); ++j)
                    ASSERT_EQ(r[j], A(i, j));
            }
        }
    }


    TEST(RowTest, testElementWrite)
    {
        StaticPanelMatrix<double, 12, 12, columnMajor> A;
        randomize(A);

        for (size_t i = 0; i < rows(A); ++i)
        {
            auto r = row(A, i);
            for (size_t j = 0; j < A.columns(); ++j)
            {
                blaze::randomize(r[j]);
                ASSERT_EQ(A(i, j), r[j]);
            }
        }
    }
}