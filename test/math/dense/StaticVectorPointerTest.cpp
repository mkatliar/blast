// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/VectorPointer.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    template <typename Scalar>
    class StaticVectorPointerTest
    :   public Test
    {
    protected:
        using Real = Scalar;


        template <bool TF>
        void testSpacingImpl()
        {
            StaticVector<Real, 3, TF> v;
            auto p = ptr<aligned>(v, 0);
            EXPECT_EQ(p.spacing(), 1);
        }


        template <bool TF>
        void testGetImpl()
        {
            StaticVector<Real, 3, TF> v;
            size_t const i = 1;
            auto p = ptr<unaligned>(v, i);
            EXPECT_EQ(p.get(), &v[i]);
        }


        template <bool TF>
        void testOffsetImpl()
        {
            StaticVector<Real, 5, TF> v;
            size_t const i = 1;
            size_t const delta = 2;
            auto p = ptr<unaligned>(v, i);

            auto po = p(delta);
            EXPECT_EQ(po.get(), &v[i + delta]);
        }


        template <bool SO>
        void testMatrixRowSubvectorImpl()
        {
            StaticMatrix<Real, 5, 5, SO> A;

            size_t constexpr i = 1, j = 2;
            auto p = ptr<unaligned>(blaze::subvector<j, columns(A) - j>(blaze::row<i>(A)), 0);
            ASSERT_EQ(p.get(), &A(i, j));
            ASSERT_EQ(p.spacing(), SO == columnMajor ? A.spacing() : 1);
        }
    };


    using MyTypes = Types<double, float>;


    TYPED_TEST_SUITE(StaticVectorPointerTest, MyTypes);


    TYPED_TEST(StaticVectorPointerTest, testSpacingColumn)
    {
        this->template testSpacingImpl<columnVector>();
    }


    TYPED_TEST(StaticVectorPointerTest, testSpacingRow)
    {
        this->template testSpacingImpl<rowVector>();
    }


    TYPED_TEST(StaticVectorPointerTest, testGetColumn)
    {
        this->template testGetImpl<columnVector>();
    }


    TYPED_TEST(StaticVectorPointerTest, testGetRow)
    {
        this->template testGetImpl<rowVector>();
    }


    TYPED_TEST(StaticVectorPointerTest, testOffsetColumn)
    {
        this->template testOffsetImpl<columnVector>();
    }


    TYPED_TEST(StaticVectorPointerTest, testOffsetRow)
    {
        this->template testOffsetImpl<rowVector>();
    }


    TYPED_TEST(StaticVectorPointerTest, testMatrixRowSubvectorRowMajor)
    {
        this->template testMatrixRowSubvectorImpl<rowMajor>();
    }


    TYPED_TEST(StaticVectorPointerTest, testMatrixRowSubvectorColumnMajor)
    {
        this->template testMatrixRowSubvectorImpl<columnMajor>();
    }
}