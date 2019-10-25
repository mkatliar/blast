#include <blazefeo/PaddedSize.hpp>

#include <test/Testing.hpp>


namespace blazefeo :: testing
{
    TEST(PaddedSizeTest, testZeroRemainder)
    {
        size_t constexpr m = paddedSize(8, 4);
        EXPECT_EQ(m, 8);
    }


    TEST(PaddedSizeTest, testNonZeroRemainder)
    {
        size_t constexpr m = paddedSize(9, 4);
        EXPECT_EQ(m, 12);
    }
}