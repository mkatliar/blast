// Copyright (c) 2019-2023 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blazefeo/math/dense/VectorPointer.hpp>

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
            EXPECT_EQ(p.spacing(), 1);
        }


        template <bool TF>
        void testGetImpl()
        {
            DynamicVector<Real, TF> v(3);
            size_t const i = 1;
            auto p = ptr<unaligned>(v, i);
            EXPECT_EQ(p.get(), &v[i]);
        }


        template <bool TF>
        void testOffsetImpl()
        {
            DynamicVector<Real, TF> v(5);
            size_t const i = 1;
            size_t const delta = 2;
            auto p = ptr<unaligned>(v, i);

            auto po = p(delta);
            EXPECT_EQ(po.get(), &v[i + delta]);
        }


        template <bool SO>
        void testMatrixRowImpl()
        {
            DynamicMatrix<Real, SO> A(5, 5);

            for (size_t i = 0; i < rows(A); ++i)
            {
                for (size_t j = 0; j < columns(A); ++j)
                {
                    auto p = ptr<unaligned>(row(A, i), j);
                    ASSERT_EQ(p.get(), &A(i, j));
                    ASSERT_EQ(p.spacing(), (SO == rowMajor) ? 1 : A.spacing());
                    ASSERT_EQ(p.transposeFlag, rowVector);
                }
            }
        }


        template <bool SO>
        void testMatrixRowSubvectorImpl()
        {
            DynamicMatrix<Real, SO> A(5, 5);

            for (size_t i = 0; i < rows(A); ++i)
            {
                for (size_t j = 0; j < columns(A); ++j)
                {
                    auto p = ptr<unaligned>(subvector(row(A, i), j, columns(A) - j), 0);
                    ASSERT_EQ(p.get(), &A(i, j));
                    ASSERT_EQ(p.spacing(), (SO == rowMajor) ? 1 : A.spacing());
                    ASSERT_EQ(p.transposeFlag, rowVector);
                }
            }
        }


        template <bool SO>
        void testMatrixColumnImpl()
        {
            DynamicMatrix<Real, SO> A(5, 5);

            for (size_t i = 0; i < rows(A); ++i)
            {
                for (size_t j = 0; j < columns(A); ++j)
                {
                    auto p = ptr<unaligned>(column(A, j), i);
                    ASSERT_EQ(p.get(), &A(i, j));
                    ASSERT_EQ(p.spacing(), (SO == columnMajor) ? 1 : A.spacing());
                    ASSERT_EQ(p.transposeFlag, columnVector);
                }
            }
        }


        template <bool SO>
        void testMatrixColumnSubvectorImpl()
        {
            DynamicMatrix<Real, SO> A(5, 5);

            for (size_t i = 0; i < rows(A); ++i)
            {
                for (size_t j = 0; j < columns(A); ++j)
                {
                    auto p = ptr<unaligned>(subvector(column(A, j), i, rows(A) - i), 0);
                    ASSERT_EQ(p.get(), &A(i, j));
                    ASSERT_EQ(p.spacing(), (SO == columnMajor) ? 1 : A.spacing());
                    ASSERT_EQ(p.transposeFlag, columnVector);
                }
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


    TYPED_TEST(DynamicVectorPointerTest, testMatrixRowRowMajor)
    {
        this->template testMatrixRowImpl<rowMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixRowColumnMajor)
    {
        this->template testMatrixRowImpl<columnMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixRowSubvectorRowMajor)
    {
        this->template testMatrixRowSubvectorImpl<rowMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixRowSubvectorColumnMajor)
    {
        this->template testMatrixRowSubvectorImpl<columnMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixColumnRowMajor)
    {
        this->template testMatrixColumnImpl<rowMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixColumnColumnMajor)
    {
        this->template testMatrixColumnImpl<columnMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixColumnSubvectorRowMajor)
    {
        this->template testMatrixColumnSubvectorImpl<rowMajor>();
    }


    TYPED_TEST(DynamicVectorPointerTest, testMatrixColumnSubvectorColumnMajor)
    {
        this->template testMatrixColumnSubvectorImpl<columnMajor>();
    }
}