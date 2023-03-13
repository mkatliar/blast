// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <blazefeo/Blaze.hpp>
#include <blaze/util/typetraits/IsArithmetic.h>

#include <iostream>
#include <cmath>
#include <type_traits>


namespace blazefeo :: testing
{
	using namespace ::testing;


	namespace detail
	{
		template <typename T>
		class ForcePrintImpl
		: 	public T
		{
			friend void PrintTo(const ForcePrintImpl &m, ::std::ostream *o)
			{
				*o << "\n" << m;
			}
		};
	}


	/*
	* Makes the Eigen3 matrix classes printable from GTest checking macros like EXPECT_EQ.
	* Usage: EXPECT_EQ(forcePrint(a), forcePrint(b))
	* Taken from this post: http://stackoverflow.com/questions/25146997/teach-google-test-how-to-print-eigen-matrix
	*/
	template <typename T>
	decltype(auto) forcePrint(T const& val)
	{
		return static_cast<detail::ForcePrintImpl<T> const&>(val);
	}


	MATCHER_P(FloatNearPointwise, tol, "Out of range")
	{
		return (std::get<0>(arg) > std::get<1>(arg) - tol && std::get<0>(arg) < std::get<1>(arg) + tol) ;
	}


	/// @brief Blaze matrix approx equality predicate
	template <typename MT1, bool SO1, typename MT2, bool SO2, typename Real>
	inline AssertionResult approxEqual(blaze::Matrix<MT1, SO1> const& lhs, blaze::Matrix<MT2, SO2> const& rhs, Real abs_tol, Real rel_tol = 0)
	{
		size_t const M = rows(lhs);
		size_t const N = columns(lhs);

		if (rows(rhs) != M || columns(rhs) != N)
			return AssertionFailure() << "Matrix size mismatch";

		for (size_t i = 0; i < M; ++i)
			for (size_t j = 0; j < N; ++j)
			{
				auto const a = (*lhs)(i, j);
				auto const b = (*rhs)(i, j);
				auto delta = a - b;

				if (std::isnan(a) != std::isnan(b)
					|| std::abs(delta) > abs_tol + rel_tol * std::abs(b))
					return AssertionFailure()
						<< "Actual value:\n" << lhs
						<< "Expected value:\n" << rhs
						<< "First mismatch at index (" << i << ", " << j << ")";
			}

		return AssertionSuccess();
	}


	/// @brief Blaze vector approx equality predicate
	template <typename VT1, typename VT2, bool TF, typename Real>
	inline std::enable_if_t<blaze::IsArithmetic_v<Real>, AssertionResult>
		approxEqual(blaze::Vector<VT1, TF> const& lhs, blaze::Vector<VT2, TF> const& rhs, Real abs_tol, Real rel_tol = 0)
	{
		size_t const N = size(lhs);

		if (size(rhs) != N)
			return AssertionFailure() << "Vector size mismatch";

		for (size_t j = 0; j < N; ++j)
			if (abs((*lhs)[j] - (*rhs)[j]) > abs_tol + rel_tol * abs((*rhs)[j]))
				return AssertionFailure()
					<< "Actual value:\n" << lhs
					<< "Expected value:\n" << rhs
					<< "First mismatch at index (" << j << ")";

		return AssertionSuccess();
	}


	/// @brief Vector approx equality predicate with absolute tolerances specified for each element.
	template <typename VT1, typename VT2, bool TF, typename Real>
	inline AssertionResult approxEqual(blaze::Vector<VT1, TF> const& lhs, blaze::Vector<VT2, TF> const& rhs, std::initializer_list<Real> abs_tol)
	{
		size_t const N = size(abs_tol);

		if (size(lhs) != N || size(rhs) != N)
			return AssertionFailure() << "Vector size mismatch";

		auto atol = begin(abs_tol);

		for (size_t j = 0; j < N; ++j, ++atol)
			if (abs((*lhs)[j] - (*rhs)[j]) > *atol)
				return AssertionFailure() << "First element mismatch at index "
					<< j << ", lhs=" << (*lhs)[j] << ", rhs=" << (*rhs)[j] << ", abs_tol=" << *atol;

		return AssertionSuccess();
	}


	/// @brief Vector approx equality predicate with absolute tolerances specified for each element.
	template <typename VT1, typename VT2, typename VT3, bool TF>
	inline AssertionResult approxEqual(blaze::Vector<VT1, TF> const& lhs, blaze::Vector<VT2, TF> const& rhs, blaze::Vector<VT3, TF> const& abs_tol)
	{
		size_t const N = size(abs_tol);

		if (size(lhs) != N || size(rhs) != N)
			return AssertionFailure() << "Vector size mismatch";

		for (size_t j = 0; j < N; ++j)
			if (abs((*lhs)[j] - (*rhs)[j]) > (*abs_tol)[j])
				return AssertionFailure() << "First element mismatch at index "
					<< j << ", lhs=" << (*lhs)[j] << ", rhs=" << (*rhs)[j] << ", abs_tol=" << (*abs_tol)[j];

		return AssertionSuccess();
	}


	/// @brief Scalar type approx equality predicate with specified tolerances.
	///
	template <typename Scalar>
	requires std::is_floating_point_v<Scalar>
	inline AssertionResult approxEqual(Scalar actual, Scalar expected, Scalar abs_tol, Scalar rel_tol)
	{
		if (std::abs(actual - expected) > abs_tol + rel_tol * std::abs(expected))
			return AssertionFailure()
				<< "Floating point values differ more than by the specified tolerance." << std::endl
				<< "Actual: " << actual << ", expected: " << expected;

		return AssertionSuccess();
	}


	/// @brief Exact equality comparison for matrices
	template <typename MT1, bool SO1, typename MT2, bool SO2>
	inline AssertionResult exactEqual(blaze::Matrix<MT1, SO1> const& lhs, blaze::Matrix<MT2, SO2> const& rhs)
	{
		size_t const M = rows(lhs);
		size_t const N = columns(lhs);

		if (rows(rhs) != M || columns(rhs) != N)
			return AssertionFailure() << "Matrix size mismatch";

		for (size_t i = 0; i < M; ++i)
			for (size_t j = 0; j < N; ++j)
				if (!((*lhs)(i, j) == (*rhs)(i, j)))
					return AssertionFailure() << "First element mismatch at index ("
						<< i << "," << j << "), lhs=" << (*lhs)(i, j) << ", rhs=" << (*rhs)(i, j);

		return AssertionSuccess();
	}


	/// @brief Exact equality comparison for vectors
	template <typename VT1, typename VT2, bool TF>
	inline AssertionResult exactEqual(blaze::Vector<VT1, TF> const& lhs, blaze::Vector<VT2, TF> const& rhs)
	{
		size_t const N = size(lhs);

		if (size(rhs) != N)
			return AssertionFailure() << "Vector size mismatch";

		for (size_t j = 0; j < N; ++j)
			if (!((*lhs)[j] == (*rhs)[j]))
				return AssertionFailure() << "First element mismatch at index "
					<< j << ", lhs=" << (*lhs)[j] << ", rhs=" << (*rhs)[j];

		return AssertionSuccess();
	}
}


#define BLAZEFEO_EXPECT_APPROX_EQ(val, expected, abs_tol, rel_tol) \
	EXPECT_TRUE(::blazefeo::testing::approxEqual(val, expected, abs_tol, rel_tol))

#define BLAZEFEO_ASSERT_APPROX_EQ(val, expected, abs_tol, rel_tol) \
	ASSERT_TRUE(::blazefeo::testing::approxEqual(val, expected, abs_tol, rel_tol))

#define BLAZEFEO_EXPECT_EQ(val, expected) \
	EXPECT_EQ(::blazefeo::testing::forcePrint(val), ::blazefeo::testing::forcePrint(expected))

#define BLAZEFEO_ASSERT_EQ(val, expected) \
	ASSERT_EQ(::blazefeo::testing::forcePrint(val), ::blazefeo::testing::forcePrint(expected))
