// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/panel/DynamicPanelMatrixPointer.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>

#include <test/Testing.hpp>

#include <blaze/Math.h>


namespace blast :: testing
{
    template <typename P>
    class DynamicPanelMatrixPointerTest
    :   public Test
    {
    protected:
        DynamicPanelMatrixPointerTest()
        :   m_(3, 5)
        {
            randomize(m_);
        }


        using Pointer = P;
        using Real = typename Pointer::ElementType;
        static bool constexpr storageOrder = Pointer::storageOrder;

        DynamicPanelMatrix<Real, storageOrder> m_;
    };


    using MyTypes = Types<
        DynamicPanelMatrixPointer<double, columnMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<double, rowMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<float, columnMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<float, rowMajor, unaligned, padded>
    >;


    TYPED_TEST_SUITE(DynamicPanelMatrixPointerTest, MyTypes);


    TYPED_TEST(DynamicPanelMatrixPointerTest, testCtor)
    {
        typename TestFixture::Pointer p {this->m_.data(), this->m_.spacing()};
        EXPECT_EQ(p.spacing(), this->m_.spacing());
    }


    TYPED_TEST(DynamicPanelMatrixPointerTest, testPtr)
    {
        size_t const i = 1, j = 2;
        typename TestFixture::Pointer p = ptr<TestFixture::Pointer::aligned>(this->m_, i, j);
        EXPECT_EQ(p.spacing(), this->m_.spacing());
    }
}