// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/dense/DynamicMatrixPointer.hpp>
#include <blast/math/dense/StaticMatrixPointer.hpp>
#include <blast/math/panel/DynamicPanelMatrixPointer.hpp>
#include <blast/math/panel/StaticPanelMatrixPointer.hpp>
#include <blast/util/Types.hpp>
#include <blast/system/Inline.hpp>

#include <iosfwd>
#include <stdexcept>
#include <random>


namespace blast
{
    /**
     * @brief Pointer to the first element of a matrix
     *
     * @tparam MT matrix type
     *
     * @param m matrix
     *
     * @return pointer to @a m(0, 0)
     */
    template <Matrix MT>
    BLAST_ALWAYS_INLINE auto ptr(MT& m)
    {
        return ptr<IsAligned_v<MT>>(m, 0, 0);
    }


    /**
     * @brief Pointer to the first element of a const matrix
     *
     * @tparam MT matrix type
     *
     * @param m matrix
     *
     * @return pointer to @a m(0, 0)
     */
    template <Matrix MT>
    BLAST_ALWAYS_INLINE auto ptr(MT const& m)
    {
        return ptr<IsAligned_v<MT>>(m, 0, 0);
    }
	
	
    template <Matrix M>
    inline void randomize(M& m) noexcept
    {
        std::mt19937 rng;
        std::uniform_real_distribution<ElementType_t<M>> dist;

        for (size_t i = 0; i < rows(m); ++i)
            for (size_t j = 0; j < columns(m); ++j)
                m(i, j) = dist(rng);
    }


    template <Matrix MA, Matrix MB>
    inline bool operator==(MA const& a, MB const& b)
    {
        size_t const M = rows(a);
        size_t const N = columns(a);

        if (M != rows(b) || N != columns(b))
            throw std::invalid_argument {"Inconsistent matrix sizes"};

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                if (a(i, j) != b(i, j))
                    return false;

        return true;
    }


    template <Matrix MA, Matrix MB>
    inline MA& assign(MA& a, MB const& b)
    {
        size_t const M = rows(a);
        size_t const N = columns(a);

        if (M != rows(b) || N != columns(b))
            throw std::invalid_argument {"Inconsistent matrix sizes"};

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                a(i, j) = b(i, j);

        return a;
    }


    template <Matrix M>
    inline std::ostream& operator<<(std::ostream& os, M const& m)
    {
        for (size_t i = 0; i < rows(m); ++i)
        {
            for (size_t j = 0; j < columns(m); ++j)
                os << m(i, j) << "\t";
            os << std::endl;
        }

        return os;
    }
}
