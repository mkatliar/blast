// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>

#include <blaze/Math.h>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>


namespace blast :: testing
{
    template <typename Real>
    class AssignPanelDenseTest
    :   public Test
    {
    protected:
        template <bool SO1, bool SO2>
        void testImpl()
        {
            size_t constexpr SS = PanelSize_v<Real>;
            size_t constexpr M = 2 * SS + 1;
            size_t constexpr N = 3 * SS + 2;

            blaze::StaticMatrix<Real, M, N, SO2> rhs;
            randomize(rhs);

            StaticPanelMatrix<Real, M, N, SO1> lhs;
            lhs = rhs;

            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    EXPECT_EQ(lhs(i, j), rhs(i, j)) << "element mismatch at (" << i << ", " << j << ")";
        }
    };


    using TestTypes = ::testing::Types<
        double,
        float
    >;

    TYPED_TEST_SUITE(AssignPanelDenseTest, TestTypes);


    TYPED_TEST(AssignPanelDenseTest, testColumnMajorColumnMajor)
    {
        this->template testImpl<columnMajor, columnMajor>();
    }


    TYPED_TEST(AssignPanelDenseTest, testColumnMajorRowMajor)
    {
        this->template testImpl<columnMajor, rowMajor>();
    }


    TYPED_TEST(AssignPanelDenseTest, testRowMajorColumnMajor)
    {
        this->template testImpl<rowMajor, columnMajor>();
    }


    TYPED_TEST(AssignPanelDenseTest, testRowMajorRowMajor)
    {
        this->template testImpl<rowMajor, rowMajor>();
    }
}