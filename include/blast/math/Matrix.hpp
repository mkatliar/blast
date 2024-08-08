// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/dense/DynamicMatrixPointer.hpp>
#include <blast/math/dense/StaticMatrixPointer.hpp>
#include <blast/math/panel/DynamicPanelMatrixPointer.hpp>
#include <blast/math/panel/StaticPanelMatrixPointer.hpp>
#include <blast/system/Inline.hpp>


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
}
