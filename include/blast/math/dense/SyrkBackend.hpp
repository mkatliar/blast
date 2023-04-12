// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/system/Tile.hpp>
#include <blast/math/register_matrix/RegisterMatrix.hpp>
#include <blast/math/dense/DynamicMatrixPointer.hpp>
#include <blast/math/dense/StaticMatrixPointer.hpp>

#include <blaze/util/Exception.h>
#include <blaze/util/constraints/SameType.h>
#include <blaze/math/DenseMatrix.h>

#include <type_traits>


namespace blast
{
    template <size_t KM, size_t KN, typename ST1, typename PA, typename T, typename PC, typename PD>
        requires MatrixPointer<PA, T> && (PA::storageOrder == columnMajor)
        && MatrixPointer<PC, T> && (PC::storageOrder == columnMajor)
        && MatrixPointer<PD, T> && (PD::storageOrder == columnMajor)
    BLAZE_ALWAYS_INLINE void syrk_ln_backend(size_t i, size_t M, size_t K, ST1 alpha, PA a, T beta, PC c, PD d)
    {
        using ET = std::remove_cv_t<ElementType_t<PA>>;
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        BLAZE_STATIC_ASSERT(KM % TILE_SIZE == 0);
        BLAZE_STATIC_ASSERT(KM >= KN);

        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<PC>, ET);
        BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE(ElementType_t<PD>, ET);

        RegisterMatrix<ET, KM, KN, columnMajor> ker;

        if (i + KM <= M)
        {
            auto ai = a.offset(i, 0);

            for (size_t j = 0; j < i; j += KN)
            {
                ker.load(beta, c.offset(i, j));
                ker.gemm(K, alpha, ai, trans(a).offset(0, j));
                ker.store(d.offset(i, j));
            }

            ker.load(beta, c.offset(i, i));
            ker.gemm(K, alpha, ai, trans(a).offset(0, i));
            ker.storeLower(d.offset(i, i));

            if constexpr (KM > KN)
            {
                syrk_ln_backend<KM - TILE_SIZE, KN>(0, KM - TILE_SIZE, K,
                    alpha, a.offset(i + KN, 0), beta, c.offset(i + KN, i + KN), d.offset(i + KN, i + KN));
            }
        }
        else
        {
            // Use partial save to calculate the bottom of the resulting matrix.
            size_t j = 0;

            for (; j < i; j += KN)
            {
                auto const md = M - i, nd = KN;
                ker.load(beta, c.offset(i, j), md, nd);
                ker.gemm(K, alpha, a.offset(i, 0), trans(a).offset(0, j), md, nd);
                ker.store(d.offset(i, j), md, nd);
            }

            if (j < M)
            {
                auto const md = M - i, nd = M - j;
                ker.load(beta, c.offset(i, j), md, nd);
                ker.gemm(K, alpha, a.offset(i, 0), trans(a).offset(0, j), md, nd);
                ker.storeLower(d.offset(i, j), md, nd);
            }

            if constexpr (KM > KN)
            {
                syrk_ln_backend<KM - TILE_SIZE, KN>(0, M - i - TILE_SIZE, K,
                    alpha, a.offset(i + KN, 0), beta, c.offset(i + KN, i + KN), d.offset(i + KN, i + KN));
            }
        }
    }
}