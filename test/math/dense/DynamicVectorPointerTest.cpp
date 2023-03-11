// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/DynamicMatrixPointer.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    template <typename Scalar>
    class DynamicVectorPointerTest
    :   public Test
    {
    protected:
        using Real = Scalar;


        template <bool TF>
        void testSpacingImpl()
        {
            DynamicVector<Real, TF> v(3);
            auto p = ptr<aligned>(v, 0);
            EXPECT_EQ(p.spacing(), v.spacing());
        }


        template <bool TF>
        void testGetImpl()
        {
            DynamicVector<Real, TF> v(3);
            size_t const i = 1;
            auto p = ptr<aligned>(v, i);
            EXPECT_EQ(p.get(), &v[i]);
        }


        template <bool TF>
        void testOffsetImpl()
        {
            DynamicVector<Real, TF> v(5);
            size_t const i = 1;
            size_t const delta = 2;
            auto p = ptr<aligned>(v, i);

            if constexpr (TF == columnVector)
            {
                auto po = p.offset(delta, 0);
                EXPECT_EQ(po.get(), &v[i + delta]);
            }
            else
            {
                auto po = p.offset(0, delta);
                EXPECT_EQ(po.get(), &v[i + delta]);
            }
        }


        template <bool TF>
        void testMoveImpl()
        {
            DynamicVector<Real, TF> v(5);
            size_t const i = 1;
            size_t const delta = 2;
            auto p = ptr<aligned>(v, i);

            if constexpr (TF == columnVector)
            {
                p.vmove(delta);
                EXPECT_EQ(p.get(), &v[i + delta]);
            }
            else
            {
                p.hmove(delta);
                EXPECT_EQ(p.get(), &v[i + delta]);
            }
        }
    };


    using MyTypes = Types<double, float>;


    TYPED_TEST_SUITE(DynamicVectorPointerTest, MyTypes);


    TYPED_TEST(DynamicVectorPointerTest, testSpacingColumn)
    {
        this->template testSpacingImpl<columnVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testSpacingRow)
    {
        this->template testSpacingImpl<rowVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testGetColumn)
    {
        this->template testGetImpl<columnVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testGetRow)
    {
        this->template testGetImpl<rowVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testOffsetColumn)
    {
        this->template testOffsetImpl<columnVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testOffsetRow)
    {
        this->template testOffsetImpl<rowVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMoveColumn)
    {
        this->template testMoveImpl<columnVector>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMoveRow)
    {
        this->template testMoveImpl<rowVector>();
    }
}