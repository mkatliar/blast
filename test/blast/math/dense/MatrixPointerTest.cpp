// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/Matrix.hpp>
#include <blast/math/StaticPanelMatrix.hpp>
#include <blast/math/DynamicPanelMatrix.hpp>
#include <blast/math/dense/StaticMatrix.hpp>
#include <blast/math/dense/DynamicMatrix.hpp>
#include <blast/math/algorithm/Randomize.hpp>
#include <blast/util/NextMultiple.hpp>

#include <test/Testing.hpp>


namespace blast :: testing
{
    /**
     * @brief Deduce matrix type for testing matrix pointer of a given type
     *
     * @tparam P matrix pointer type
     */
    template <typename P>
    struct MatrixType;


    template <typename T, StorageOrder SO, bool AF, bool PF>
    struct MatrixType<DynamicMatrixPointer<T, SO, AF, PF>>
    {
        using type = DynamicMatrix<T, SO>;
    };


    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    struct MatrixType<StaticMatrixPointer<T, S, SO, AF, PF>>
    {
    private:
        static size_t constexpr defaultRows = 13;
        static size_t constexpr defaultColumns = 11;
        static size_t constexpr M = SO == columnMajor ? S : defaultRows;
        static size_t constexpr N = SO == rowMajor ? S : defaultColumns;

    public:
        using type = StaticMatrix<T, M, N, SO>;
        static_assert(type::spacing() == S);
    };


    template <typename T, StorageOrder SO, bool AF, bool PF>
    struct MatrixType<DynamicPanelMatrixPointer<T, SO, AF, PF>>
    {
        using type = DynamicPanelMatrix<T, SO>;
    };


    template <typename T, size_t S, StorageOrder SO, bool AF, bool PF>
    struct MatrixType<StaticPanelMatrixPointer<T, S, SO, AF, PF>>
    {
    private:
        static size_t constexpr defaultRows = 13;
        static size_t constexpr defaultColumns = 11;
        static size_t constexpr M = SO == rowMajor ? S / SimdSize_v<T> : defaultRows;
        static size_t constexpr N = SO == columnMajor ? S / SimdSize_v<T> : defaultColumns;

    public:
        using type = StaticPanelMatrix<T, M, N, SO>;
        static_assert(type::spacing() == S);
    };


    template <typename P>
    class MatrixPointerTest
    :   public Test
    {
    protected:
        MatrixPointerTest()
        :   m_ {createMatrix()}
        {
            randomize(m_);
        }


        using Pointer = P;
        using Real = typename P::ElementType;
        using Matrix = typename MatrixType<P>::type;

        static bool constexpr storageOrder = P::storageOrder;
        static bool constexpr isAligned = P::aligned;
        static size_t constexpr SS = SimdSize_v<Real>;
        static size_t constexpr incI_ = storageOrder == columnMajor && isAligned ? SS : 1;
        static size_t constexpr incJ_ = storageOrder == rowMajor && isAligned ? SS : 1;
        static size_t constexpr SIMD_DIR_I = storageOrder == columnMajor ? 1 : 0;
        static size_t constexpr SIMD_DIR_J = storageOrder == rowMajor ? 1 : 0;
        static size_t constexpr SIMD_M = storageOrder == columnMajor ? SS : 1;
        static size_t constexpr SIMD_N = storageOrder == rowMajor ? SS : 1;


        Matrix m_;

        static Matrix createMatrix()
        {
            if constexpr (IsStatic_v<Matrix>)
                return Matrix {};
            else
                return Matrix(13, 11);
        }
    };


    using MyTypes = Types<
        DynamicMatrixPointer<double, columnMajor, aligned, padded>,
        DynamicMatrixPointer<double, columnMajor, unaligned, padded>,
        DynamicMatrixPointer<double, rowMajor, aligned, padded>,
        DynamicMatrixPointer<double, rowMajor, unaligned, padded>,
        StaticMatrixPointer<double, nextMultiple(13, SimdSize_v<double>), columnMajor, aligned, padded>,
        StaticMatrixPointer<double, nextMultiple(13, SimdSize_v<double>), columnMajor, unaligned, padded>,
        StaticMatrixPointer<double, nextMultiple(13, SimdSize_v<double>), rowMajor, aligned, padded>,
        StaticMatrixPointer<double, nextMultiple(13, SimdSize_v<double>), rowMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<double, columnMajor, aligned, padded>,
        DynamicPanelMatrixPointer<double, columnMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<double, rowMajor, aligned, padded>,
        DynamicPanelMatrixPointer<double, rowMajor, unaligned, padded>,
        StaticPanelMatrixPointer<double, SimdSize_v<double> * 13, columnMajor, aligned, padded>,
        StaticPanelMatrixPointer<double, SimdSize_v<double> * 13, columnMajor, unaligned, padded>,
        StaticPanelMatrixPointer<double, SimdSize_v<double> * 13, rowMajor, aligned, padded>,
        StaticPanelMatrixPointer<double, SimdSize_v<double> * 13, rowMajor, unaligned, padded>,

        DynamicMatrixPointer<float, columnMajor, aligned, padded>,
        DynamicMatrixPointer<float, rowMajor, aligned, padded>,
        DynamicMatrixPointer<float, columnMajor, unaligned, padded>,
        DynamicMatrixPointer<float, rowMajor, unaligned, padded>,
        StaticMatrixPointer<float, nextMultiple(13, SimdSize_v<float>), columnMajor, aligned, padded>,
        StaticMatrixPointer<float, nextMultiple(13, SimdSize_v<float>), rowMajor, aligned, padded>,
        StaticMatrixPointer<float, nextMultiple(13, SimdSize_v<float>), columnMajor, unaligned, padded>,
        StaticMatrixPointer<float, nextMultiple(13, SimdSize_v<float>), rowMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<float, columnMajor, aligned, padded>,
        DynamicPanelMatrixPointer<float, columnMajor, unaligned, padded>,
        DynamicPanelMatrixPointer<float, rowMajor, aligned, padded>,
        DynamicPanelMatrixPointer<float, rowMajor, unaligned, padded>,
        StaticPanelMatrixPointer<float, SimdSize_v<float> * 13, columnMajor, aligned, padded>,
        StaticPanelMatrixPointer<float, SimdSize_v<float> * 13, columnMajor, unaligned, padded>,
        StaticPanelMatrixPointer<float, SimdSize_v<float> * 13, rowMajor, unaligned, padded>,
        StaticPanelMatrixPointer<float, SimdSize_v<float> * 13, rowMajor, aligned, padded>
    >;


    TYPED_TEST_SUITE(MatrixPointerTest, MyTypes);


    TYPED_TEST(MatrixPointerTest, testIsStatic)
    {
        for (size_t i = 0; i < rows(this->m_); i += this->incI_)
            for (size_t j = 0; j < columns(this->m_); j += this->incJ_)
            {
                auto const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                EXPECT_EQ(p.isStatic, IsStatic_v<typename TestFixture::Matrix>);
            }
    }


    TYPED_TEST(MatrixPointerTest, testIsAlignedByDefault)
    {
        auto const p = ptr(this->m_);
        EXPECT_EQ(p.aligned, IsAligned_v<decltype(this->m_)>);
    }


    TYPED_TEST(MatrixPointerTest, testIsAligned)
    {
        for (size_t i = 0; i < rows(this->m_); i += this->incI_)
            for (size_t j = 0; j < columns(this->m_); j += this->incJ_)
            {
                auto const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                EXPECT_EQ(p.aligned, TestFixture::isAligned);
            }
    }


    TYPED_TEST(MatrixPointerTest, testGet)
    {
        for (size_t i = 0; i < rows(this->m_); i += this->incI_)
            for (size_t j = 0; j < columns(this->m_); j += this->incJ_)
            {
                typename TestFixture::Pointer const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                EXPECT_EQ(p.get(), &this->m_(i, j)) << " at (" << i << ", " << j << ")";
            }
    }


    TYPED_TEST(MatrixPointerTest, testSpacing)
    {
        for (size_t i = 0; i < rows(this->m_); i += this->incI_)
            for (size_t j = 0; j < columns(this->m_); j += this->incJ_)
            {
                typename TestFixture::Pointer const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                EXPECT_EQ(p.spacing(), this->m_.spacing()) << " at (" << i << ", " << j << ")";
            }
    }


    TYPED_TEST(MatrixPointerTest, testStorageOrder)
    {
        for (size_t i = 0; i < rows(this->m_); i += this->incI_)
            for (size_t j = 0; j < columns(this->m_); j += this->incJ_)
            {
                typename TestFixture::Pointer const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                EXPECT_EQ(p.storageOrder, this->storageOrder) << " at (" << i << ", " << j << ")";
            }
    }


    TYPED_TEST(MatrixPointerTest, testLoad)
    {
        for (size_t i = 0; i + this->SIMD_M <= rows(this->m_); i += this->incI_)
            for (size_t j = 0; j + this->SIMD_N <= columns(this->m_); j += this->incJ_)
            {
                typename TestFixture::Pointer const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                auto const val = p.load();

                for (size_t k = 0; k < this->SS; ++k)
                    EXPECT_EQ(val[k], this->m_(i + k * this->SIMD_DIR_I, j + k * this->SIMD_DIR_J))
                        << "at (i, j, k)=(" << i << ", " << j << ", " << k << ")\n"
                        << "m=\n" << this->m_;
            }
    }


    TYPED_TEST(MatrixPointerTest, testTrans)
    {
        for (size_t i = 0; i < rows(this->m_); i += this->incI_)
            for (size_t j = 0; j < columns(this->m_); j += this->incJ_)
            {
                typename TestFixture::Pointer const p = ptr<TestFixture::isAligned>(this->m_, i, j);
                auto const p_trans = p.trans();

                EXPECT_EQ(p_trans.get(), p.get());
                EXPECT_EQ(p_trans.spacing(), p.spacing());
                EXPECT_EQ(p_trans.aligned, p.aligned);
                EXPECT_EQ(p_trans.padded, p.padded);
                EXPECT_EQ(p_trans.isStatic, p.isStatic);
                EXPECT_EQ(p_trans.storageOrder, !p.storageOrder);
            }
    }
}
