// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/util/Types.hpp>
#include <blast/util/NextMultiple.hpp>
#include <blast/math/Simd.hpp>


namespace blast
{
    template <typename T>
    class ColumnMajorLayout
    {
    public:
        constexpr ColumnMajorLayout(size_t m, size_t n) noexcept
        :   spacing_ {spacing(m)}
        {
        }


        static size_t constexpr capacity(size_t m, size_t n) noexcept
        {
            return spacing(m) * n;
        }


        T * operator()(T * p, ptrdiff_t i, ptrdiff_t j) noexcept
        {
            return p + i + spacing_ * j;
        }

    private:
        size_t const spacing_;

        static size_t constexpr spacing(size_t m) noexcept
        {
            return nextMultiple(m, SimdSize_v<T>)
        }
    };
}
