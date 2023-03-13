// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/StaticMatrixPointer.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    template <typename MT>
    class StaticMatrixPointerTest
    :   public Test
    {
    protected:
        StaticMatrixPointerTest()
        {
            randomize(m_);
        }


        using Matrix = MT;
        static bool constexpr storageOrder = MT::storageOrder;
        using Real = ElementType_t<MT>;
        using Pointer = StaticMatrixPointer<Real, MT::spacing(), storageOrder, IsAligned_v<MT>, IsPadded_v<MT>>;
        static size_t constexpr SS = Simd<Real>::size;

        Matrix m_;
    };


    using MyTypes = Types<
        StaticMatrix<double, 20, 20, columnMajor>,
        StaticMatrix<double, 20, 20, rowMajor>,
        StaticMatrix<float, 20, 20, columnMajor>,
        StaticMatrix<float, 20, 20, rowMajor>
    >;


    TYPED_TEST_SUITE(StaticMatrixPointerTest, MyTypes);


    TYPED_TEST(StaticMatrixPointerTest, testSpacing)
    {
        typename TestFixture::Pointer p {this->m_.data()};

        size_t constexpr s = p.spacing();
        EXPECT_EQ(s, this->m_.spacing());
    }


    TYPED_TEST(StaticMatrixPointerTest, testLoad)
    {
        typename TestFixture::Pointer p {this->m_.data()};
        ptrdiff_t constexpr i = 0, j = 0;
        auto const val = p(i, j).load();

        for (size_t k = 0; k < this->SS; ++k)
            EXPECT_EQ(val[k], this->storageOrder == columnMajor ?
                this->m_(i + k, j) : this->m_(i, j + k));
    }


    TYPED_TEST(StaticMatrixPointerTest, testPtr)
    {
        size_t const i = 0, j = 0;
        auto p = ptr<aligned>(this->m_, 0, 0);
        auto const val = p(i, j).load();

        EXPECT_EQ(p.spacing(), this->m_.spacing());

        for (size_t k = 0; k < this->SS; ++k)
            EXPECT_EQ(val[k], this->storageOrder == columnMajor ?
                this->m_(i + k, j) : this->m_(i, j + k));
    }


    TYPED_TEST(StaticMatrixPointerTest, testPtrTrans)
    {
        size_t const i = 0, j = 0;
        auto p = ptr<aligned>(trans(this->m_), 0, 0);
        auto const val = p(i, j).load();

        ASSERT_EQ(p.storageOrder, !this->storageOrder);
        EXPECT_EQ(p.spacing(), this->m_.spacing());

        for (size_t k = 0; k < this->SS; ++k)
            EXPECT_EQ(val[k], this->storageOrder == columnMajor ?
                this->m_(i + k, j) : this->m_(i, j + k)) << " at k=" << k;
    }
}