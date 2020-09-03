// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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