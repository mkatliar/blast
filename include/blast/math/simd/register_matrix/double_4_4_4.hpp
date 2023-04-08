// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/simd/register_matrix/RegisterMatrix.hpp>
#include <blast/Exception.hpp>

#include <cmath>


namespace blast
{
    template <>
    template <typename P>
    requires MatrixPointer<P, double> && (P::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 4, 4, columnMajor>::load(double beta, P ptr, size_t m, size_t n) noexcept
    {
        if (n > 0)
            v_[0][0] = beta * ptr.load();

        if (n > 1)
            v_[0][1] = beta * ptr(0, 1).load();

        if (n > 2)
            v_[0][2] = beta * ptr(0, 2).load();

        if (n > 3)
            v_[0][3] = beta * ptr(0, 3).load();
    }


#if 1
    /// Magically, this function specialization is slightly faster than the default implementation of RegisterMatrix<>::store.
    template <>
    template <typename P>
    requires MatrixPointer<P, double> && (P::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 4, 4, columnMajor>::store(P ptr, size_t m, size_t n) const noexcept
    {
        if (m >= 4)
        {
            if (n > 0)
                ptr.store(v_[0][0]);

            if (n > 1)
                ptr(0, 1).store(v_[0][1]);

            if (n > 2)
                ptr(0, 2).store(v_[0][2]);

            if (n > 3)
                ptr(0, 3).store(v_[0][3]);
        }
        else if (m > 0)
        {
            // Magically, the code below is significantly faster than this:
            // __m256i const mask = _mm256_cmpgt_epi64(_mm256_set_epi64x(m, m, m, m), _mm256_set_epi64x(3, 2, 1, 0));
            __m256i const mask = _mm256_set_epi64x(
                m > 3 ? 0x8000000000000000ULL : 0,
                m > 2 ? 0x8000000000000000ULL : 0,
                m > 1 ? 0x8000000000000000ULL : 0,
                m > 0 ? 0x8000000000000000ULL : 0);

            if (n > 0)
                ptr.store(v_[0][0], mask);

            if (n > 1)
                ptr(0, 1).store(v_[0][1], mask);

            if (n > 2)
                ptr(0, 2).store(v_[0][2], mask);

            if (n > 3)
                ptr(0, 3).store(v_[0][3], mask);
        }
    }
#endif
}