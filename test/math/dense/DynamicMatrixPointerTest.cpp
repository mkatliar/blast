// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    template <typename P>
    class DynamicMatrixPointerTest
    :   public Test
    {
    protected:
        DynamicMatrixPointerTest()
        :   m_(3, 5)
        {
            randomize(m_);
        }


        using Pointer = P;
        using Real = typename Pointer::ElementType;
        static bool constexpr storageOrder = Pointer::storageOrder;

        DynamicMatrix<Real, storageOrder> m_;
    };


    using MyTypes = Types<
        DynamicMatrixPointer<double, columnMajor, unaligned, padded>,
        DynamicMatrixPointer<double, rowMajor, unaligned, padded>,
        DynamicMatrixPointer<float, columnMajor, unaligned, padded>,
        DynamicMatrixPointer<float, rowMajor, unaligned, padded>
    >;


    TYPED_TEST_SUITE(DynamicMatrixPointerTest, MyTypes);


    TYPED_TEST(DynamicMatrixPointerTest, testCtor)
    {
        typename TestFixture::Pointer p {this->m_.data(), this->m_.spacing()};
        EXPECT_EQ(p.spacing(), this->m_.spacing());
    }


    TYPED_TEST(DynamicMatrixPointerTest, testPtr)
    {
        size_t const i = 1, j = 2;
        typename TestFixture::Pointer p = ptr<TestFixture::Pointer::aligned>(this->m_, i, j);
        EXPECT_EQ(p.spacing(), this->m_.spacing());
    }
}