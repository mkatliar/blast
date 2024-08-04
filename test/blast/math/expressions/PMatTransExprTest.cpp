// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/panel/Potrf.hpp>
#include <blast/math/panel/Gemm.hpp>

#include <test/Testing.hpp>
#include <test/Randomize.hpp>
#include <test/Tolerance.hpp>

namespace blast :: testing
{
    template <typename T>
    class PMatTransExprTest;


    template <typename ET, size_t M, size_t N, bool SO>
    class PMatTransExprTest<StaticPanelMatrix<ET, M, N, SO>>
    :   public ::testing::Test
    {
    protected:
        StaticPanelMatrix<ET, M, N, SO> A_;


        void SetUp() override
        {
            randomize(A_);
        }
    };


    template <typename ET, bool SO>
    class PMatTransExprTest<DynamicPanelMatrix<ET, SO>>
    :   public ::testing::Test
    {
    protected:
        DynamicPanelMatrix<ET, SO> A_ {5, 7};


        void SetUp() override
        {
            randomize(A_);
        }
    };


    using TestMatrixTypes = ::testing::Types<
        DynamicPanelMatrix<double, columnMajor>,
        DynamicPanelMatrix<double, rowMajor>,
        DynamicPanelMatrix<float, columnMajor>,
        DynamicPanelMatrix<float, rowMajor>,
        StaticPanelMatrix<double, 5, 7, columnMajor>,
        StaticPanelMatrix<double, 5, 7, rowMajor>,
        StaticPanelMatrix<float, 5, 7, columnMajor>,
        StaticPanelMatrix<float, 5, 7, rowMajor>
    >;

    TYPED_TEST_SUITE(PMatTransExprTest, TestMatrixTypes);


    TYPED_TEST(PMatTransExprTest, testRows)
    {
        auto A_trans = trans(this->A_);
        ASSERT_EQ(A_trans.rows(), this->A_.columns());
    }


    TYPED_TEST(PMatTransExprTest, testColumns)
    {
        auto A_trans = trans(this->A_);
        ASSERT_EQ(A_trans.columns(), this->A_.rows());
    }


    TYPED_TEST(PMatTransExprTest, testStorageOrder)
    {
        auto A_trans = trans(this->A_);
        ASSERT_EQ(A_trans.storageOrder, !this->A_.storageOrder);
    }


    TYPED_TEST(PMatTransExprTest, testElementAccess)
    {
        auto A_trans = trans(this->A_);

        for (size_t i = 0; i < A_trans.rows(); ++i)
            for (size_t j = 0; j < A_trans.columns(); ++j)
                ASSERT_EQ(A_trans(i, j), this->A_(j, i));
    }
}
