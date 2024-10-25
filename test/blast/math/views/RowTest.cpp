// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>

#include <test/Testing.hpp>
#include <blast/math/algorithm/Randomize.hpp>


namespace blast :: testing
{
    TEST(RowTest, testSize)
    {
        StaticPanelMatrix<double, 12, 12, columnMajor> A;
        randomize(A);

        for (size_t i = 0; i < rows(A); ++i)
        {
            auto r = row(A, i);
            ASSERT_EQ(size(r), A.columns());
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
