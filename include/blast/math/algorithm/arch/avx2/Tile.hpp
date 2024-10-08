// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/system/Tile.hpp>
#include <blast/system/Inline.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/math/RegisterMatrix.hpp>
#include <blast/util/Types.hpp>

#include <blast/math/Simd.hpp>


namespace blast :: detail
{
    template <typename ET, size_t KM, size_t KN, StorageOrder SO, typename FF, typename FP>
    BLAST_ALWAYS_INLINE void tile_backend(xsimd::avx2, size_t m, size_t n, size_t i, FF&& f_full, FP&& f_partial)
    {
        RegisterMatrix<ET, KM, KN, SO> ker;

        if (i + KM <= m)
        {
            size_t j = 0;

            for (; j + KN <= n; j += KN)
                f_full(ker, i, j);

            if (j < n)
                f_partial(ker, i, j, KM, n - j);
        }
        else
        {
            size_t j = 0;

            for (; j + KN <= n; j += KN)
                f_partial(ker, i, j, m - i, KN);

            if (j < n)
                f_partial(ker, i, j, m - i, n - j);
        }
    }


    template <typename ET, StorageOrder SO, typename FF, typename FP>
    BLAST_ALWAYS_INLINE void tile(xsimd::avx2 const& arch, StorageOrder traversal_order, std::size_t m, std::size_t n, FF&& f_full, FP&& f_partial)
    {
        size_t constexpr SS = SimdSize_v<ET>;
        size_t constexpr TILE_STEP = 4;    // TODO: this is almost arbitrary and needs to be properly determined

        static_assert(SO == columnMajor, "tile() for row-major matrices not implemented");

        if (traversal_order == columnMajor)
        {
            size_t j = 0;

            // Main part
            for (; j + TILE_STEP <= n; j += TILE_STEP)
            {
                size_t i = 0;

                // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
                // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
                for (; i + 3 * SS <= m && i + 4 * SS != m; i += 3 * SS)
                {
                    RegisterMatrix<ET, 3 * SS, TILE_STEP, SO> ker;
                    f_full(ker, i, j);
                }

                for (; i + 2 * SS <= m; i += 2 * SS)
                {
                    RegisterMatrix<ET, 2 * SS, TILE_STEP, SO> ker;
                    f_full(ker, i, j);
                }

                for (; i + 1 * SS <= m; i += 1 * SS)
                {
                    RegisterMatrix<ET, 1 * SS, TILE_STEP, SO> ker;
                    f_full(ker, i, j);
                }

                // Bottom side
                if (i < m)
                {
                    RegisterMatrix<ET, SS, TILE_STEP, SO> ker;
                    f_partial(ker, i, j, m - i, ker.columns());
                }
            }


            // Right side
            if (j < n)
            {
                size_t i = 0;

                // i + 4 * TILE_STEP != M is to improve performance in case when the remaining number of rows is 4 * TILE_STEP:
                // it is more efficient to apply 2 * TILE_STEP kernel 2 times than 3 * TILE_STEP + 1 * TILE_STEP kernel.
                for (; i + 3 * SS <= m && i + 4 * SS != m; i += 3 * SS)
                {
                    RegisterMatrix<ET, 3 * SS, TILE_STEP, SO> ker;
                    f_partial(ker, i, j, ker.rows(), n - j);
                }

                for (; i + 2 * SS <= m; i += 2 * SS)
                {
                    RegisterMatrix<ET, 2 * SS, TILE_STEP, SO> ker;
                    f_partial(ker, i, j, ker.rows(), n - j);
                }

                for (; i + 1 * SS <= m; i += 1 * SS)
                {
                    RegisterMatrix<ET, 1 * SS, TILE_STEP, SO> ker;
                    f_partial(ker, i, j, ker.rows(), n - j);
                }

                // Bottom-right corner
                if (i < m)
                {
                    RegisterMatrix<ET, SS, TILE_STEP, SO> ker;
                    f_partial(ker, i, j, m - i, n - j);
                }
            }
        }
        else
        {
            size_t i = 0;

            // i + 4 * SS != M is to improve performance in case when the remaining number of rows is 4 * SS:
            // it is more efficient to apply 2 * SS kernel 2 times than 3 * SS + 1 * SS kernel.
            for (; i + 2 * SS < m && i + 4 * SS != m; i += 3 * SS)
                tile_backend<ET, 3 * SS, TILE_STEP, SO>(arch, m, n, i, f_full, f_partial);

            for (; i + 1 * SS < m; i += 2 * SS)
                tile_backend<ET, 2 * SS, TILE_STEP, SO>(arch, m, n, i, f_full, f_partial);

            for (; i + 0 * SS < m; i += 1 * SS)
                tile_backend<ET, 1 * SS, TILE_STEP, SO>(arch, m, n, i, f_full, f_partial);
        }
    }
}
