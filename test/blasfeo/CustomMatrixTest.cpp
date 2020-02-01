#include <blasfeo/CustomMatrix.hpp>
#include <test/Testing.hpp>

#include <vector>
#include <memory>


namespace blasfeo :: testing
{
	TEST(CustomMatrixTest, testCtor)
	{
		std::vector<double> data(6);
		blasfeo::CustomMatrix<double> m(data.data(), 2, 3);
	}


	TEST(CustomMatrixTest, testRows)
	{
		std::vector<double> data(6);
		blasfeo::CustomMatrix<double> m(data.data(), 2, 3);

		EXPECT_EQ(rows(m), 2);
	}


	TEST(CustomMatrixTest, testColumns)
	{
		std::vector<double> data(6);
		blasfeo::CustomMatrix<double> m(data.data(), 2, 3);

		EXPECT_EQ(columns(m), 3);
	}
}