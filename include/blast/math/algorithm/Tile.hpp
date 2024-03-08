// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <blast/system/Tile.hpp>
#include <blast/system/Inline.hpp>
#include <blast/math/StorageOrder.hpp>
#include <blast/math/simd/RegisterMatrix.hpp>

#include <cstdlib>


namespace blast
{
    /**
     * @brief Cover a matrix with tiles of different sizes in a performance-efficient way.
     *
     * The tile sizes and positions are chosen based on matrix element type, storage order, and current system architecture.
     * Positions and sizes of the tiles are such that the entire matrix is covered and tiles do not overlap.
     *
     * This function is helpful in implementing register-blocked matrix algorithms.
     *
     * For each tile one of the two specified functors @a f_full, @a f_partial is called:
     *
     * 1) @a f_full (ker, i, j);  // if tile size equals ker.columns() by ker.rows()
     * 2) @a f_partial (ker, i, j, km, kn);  // if tile size is smaller than ker.columns() by ker.rows()
     *
     * where ker is a RegisterMatrix object, (i, j) are indices of top left corner of the tile,
     * and (km, kn) are dimensions of the tile.
     *
     * @tparam ET type of matrix elements
     * @tparam SO matrix storage order
     * @tparam F functor type
     *
     * @param m number of matrix rows
     * @param n number of matrix columns
     * @param f_full functor to call on full tiles
     * @param f_partial functor to call on partial tiles
     */
    template <typename ET, StorageOrder SO, typename FF, typename FP>
    BLAST_ALWAYS_INLINE void tile(std::size_t m, std::size_t n, FF&& f_full, FP&& f_partial)
    {
        size_t constexpr TILE_SIZE = TileSize_v<ET>;

        size_t j = 0;

        // Main part
        for (; j + TILE_SIZE <= n; j += TILE_SIZE)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= m && i + 4 * TILE_SIZE != m; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_full(ker, i, j);
            }

            for (; i + 2 * TILE_SIZE <= m; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_full(ker, i, j);
            }

            for (; i + 1 * TILE_SIZE <= m; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_full(ker, i, j);
            }

            // Bottom side
            if (i < m)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_partial(ker, i, j, m - i, ker.columns());
            }
        }


        // Right side
        if (j < n)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= m && i + 4 * TILE_SIZE != m; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_partial(ker, i, j, ker.rows(), n - j);
            }

            for (; i + 2 * TILE_SIZE <= m; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_partial(ker, i, j, ker.rows(), n - j);
            }

            for (; i + 1 * TILE_SIZE <= m; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_partial(ker, i, j, ker.rows(), n - j);
            }

            // Bottom-right corner
            if (i < m)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                f_partial(ker, i, j, m - i, n - j);
            }
        }
    }
}