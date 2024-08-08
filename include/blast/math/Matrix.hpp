// Copyright (c) 2019-2024 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/typetraits/Matrix.hpp>
#include <blast/math/typetraits/IsDenseMatrix.hpp>
#include <blast/math/typetraits/IsStatic.hpp>
#include <blast/math/dense/StaticMatrix.hpp>
#include <blast/math/panel/StaticPanelMatrix.hpp>
#include <blast/util/Types.hpp>
#include <blast/system/Inline.hpp>

#include <iostream>


namespace blast
{
    template <Matrix M>
    inline void randomize(M& m) noexcept
    {

    }


    template <Matrix MA, Matrix MB>
    inline bool operator==(MA const& a, MB const& b) noexcept
    {
        return false;
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


    template <bool AF, Matrix MT>
    requires IsStatic_v<MT> && IsDenseMatrix_v<MT>
    BLAST_ALWAYS_INLINE auto ptr(MT& m, size_t i, size_t j) noexcept
    {
        return StaticMatrixPointer<ElementType_t<MT>, spacing(m), StorageOrder_v<MT>, AF, IsPadded_v<MT>>(data(m), i, j);
    }


    template <bool AF, Matrix MT>
    requires IsStatic_v<MT> && IsDenseMatrix_v<MT>
    BLAST_ALWAYS_INLINE auto ptr(MT const& m, size_t i, size_t j) noexcept
    {
        return StaticMatrixPointer<ElementType_t<MT> const, spacing(m), StorageOrder_v<MT>, AF, IsPadded_v<MT>>(data(m), i, j);
    }
}
