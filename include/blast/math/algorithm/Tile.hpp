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
#include <blast/math/StorageOrder.hpp>
#include <blast/math/simd/RegisterMatrix.hpp>

#include <cstdlib>


namespace blast
{
    /**
     * @brief Cover a matrix with tiles of different sizes in a performance-efficient way.
     *
     * The tile sizes and positions are chosen based on matrix element type, storage order, and current system architecture.
     * Positions and sizes of the tiles are such that the entire matrix is coveren and tiles do not overlap.
     *
     * This function is helpful in implementing register-blocked matrix algorithms.
     *
     * For each tile a specified functor @a func is called in one of the following ways:
     *
     * 1) func(ker, i, j);
     * 2) func(ker, i, j, km, kn);
     *
     * Where ker is a RegisterMatrix object, (i, j) are indices of top left corner of the tile,
     * and (km, kn) are dimensions of the tile.
     *
     * @tparam ET type of matrix elements
     * @tparam SO matrix storage order
     * @tparam F functor type
     *
     * @param m number of matrix rows
     * @param n number of matrix columns
     * @param func functor to call
     */
    template <typename ET, StorageOrder SO, typename F>
    inline void tile(std::size_t m, std::size_t n, F&& func)
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
                func(ker, i, j);
            }

            for (; i + 2 * TILE_SIZE <= m; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j);
            }

            for (; i + 1 * TILE_SIZE <= m; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j);
            }

            // Bottom edge
            if (i < m)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j, m - i, ker.columns());
            }
        }


        // Right edge
        if (j < n)
        {
            size_t i = 0;

            // i + 4 * TILE_SIZE != M is to improve performance in case when the remaining number of rows is 4 * TILE_SIZE:
            // it is more efficient to apply 2 * TILE_SIZE kernel 2 times than 3 * TILE_SIZE + 1 * TILE_SIZE kernel.
            for (; i + 3 * TILE_SIZE <= m && i + 4 * TILE_SIZE != m; i += 3 * TILE_SIZE)
            {
                RegisterMatrix<ET, 3 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j, ker.rows(), n - j);
            }

            for (; i + 2 * TILE_SIZE <= m; i += 2 * TILE_SIZE)
            {
                RegisterMatrix<ET, 2 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j, ker.rows(), n - j);
            }

            for (; i + 1 * TILE_SIZE <= m; i += 1 * TILE_SIZE)
            {
                RegisterMatrix<ET, 1 * TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j, ker.rows(), n - j);
            }

            // Bottom-right corner
            if (i < m)
            {
                RegisterMatrix<ET, TILE_SIZE, TILE_SIZE, columnMajor> ker;
                func(ker, i, j, m - i, n - j);
            }
        }
    }
}