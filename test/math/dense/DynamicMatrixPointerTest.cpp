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
        DynamicMatrixPointer<double, columnMajor>,
        DynamicMatrixPointer<double, rowMajor>,
        DynamicMatrixPointer<float, columnMajor>,
        DynamicMatrixPointer<float, rowMajor>
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
        typename TestFixture::Pointer p = ptr(this->m_, i, j);
        EXPECT_EQ(p.spacing(), this->m_.spacing());
    }
}