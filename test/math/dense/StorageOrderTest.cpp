// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <blaze/math/typetraits/StorageOrder.h>
#include <blazefeo/math/dense/StorageOrder.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    TEST(StorageOrderTest, testStaticColumnVector)
    {
        StaticVector<double, 1, columnVector> v;
        ASSERT_EQ(StorageOrder_v<decltype(v)>, columnMajor);
    }


    TEST(StorageOrderTest, testStaticRowVector)
    {
        StaticVector<double, 1, rowVector> v;
        ASSERT_EQ(StorageOrder_v<decltype(v)>, rowMajor);
    }


    TEST(StorageOrderTest, testStaticColumnSubvector)
    {
        StaticVector<double, 10, columnVector> v;
        auto s = subvector<3, 2>(v);
        ASSERT_EQ(StorageOrder_v<decltype(s)>, columnMajor);
    }


    TEST(StorageOrderTest, testStaticRowSubvector)
    {
        StaticVector<double, 10, rowVector> v;
        auto s = subvector<3, 2>(v);
        ASSERT_EQ(StorageOrder_v<decltype(s)>, rowMajor);
    }


    TEST(StorageOrderTest, testStaticColumnMajorMatrixColumn)
    {
        StaticMatrix<double, 10, 10, columnMajor> A;
        auto c = column<1>(A);
        ASSERT_EQ(StorageOrder_v<decltype(c)>, columnMajor);
    }


    TEST(StorageOrderTest, testStaticColumnMajorMatrixRow)
    {
        StaticMatrix<double, 10, 10, columnMajor> A;
        auto r = row<1>(A);
        ASSERT_EQ(StorageOrder_v<decltype(r)>, columnMajor);
    }


    TEST(StorageOrderTest, testStaticMatrixRow)
    {
        StaticVector<double, 10, rowVector> v;
        auto s = subvector<3, 2>(v);
        ASSERT_EQ(StorageOrder_v<decltype(s)>, rowMajor);
    }
}