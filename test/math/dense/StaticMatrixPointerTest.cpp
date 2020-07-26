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
        using Pointer = StaticMatrixPointer<Real, MT::spacing(), storageOrder>;

        Matrix m_;
    };


    using MyTypes = Types<
        StaticMatrix<double, 3, 5, columnMajor>,
        StaticMatrix<double, 3, 5, rowMajor>,
        StaticMatrix<float, 3, 5, columnMajor>,
        StaticMatrix<float, 3, 5, rowMajor>
    >;
        
        
    TYPED_TEST_SUITE(StaticMatrixPointerTest, MyTypes);


    TYPED_TEST(StaticMatrixPointerTest, testCtor)
    {
        typename TestFixture::Pointer p {this->m_.data()};
        EXPECT_EQ(p.get(), this->m_.data());

        size_t constexpr s = p.spacing();
        EXPECT_EQ(s, this->m_.spacing());
    }


    TYPED_TEST(StaticMatrixPointerTest, testPtr)
    {
        size_t const i = 1, j = 2;
        typename TestFixture::Pointer p = ptr(this->m_, i, j);
        EXPECT_EQ(p.get(), &this->m_(i, j));
        EXPECT_EQ(p.spacing(), this->m_.spacing());
    }
}