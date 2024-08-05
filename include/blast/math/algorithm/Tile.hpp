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

#include <blast/math/Simd.hpp>

#if XSIMD_WITH_AVX2
#   include <blast/math/algorithm/arch/avx2/Tile.hpp>
#endif

#include <blast/math/StorageOrder.hpp>
#include <blast/system/Inline.hpp>

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
     * @tparam FF functor type for full tiles
     * @tparam FP functor type for partial tiles
     * @tparam Arch instruction set architecture
     *
     * @param m number of matrix rows
     * @param n number of matrix columns
     * @param f_full functor to call on full tiles
     * @param f_partial functor to call on partial tiles
     */
    template <typename ET, StorageOrder SO, typename FF, typename FP, typename Arch>
    BLAST_ALWAYS_INLINE void tile(Arch arch, StorageOrder traversal_order, std::size_t m, std::size_t n, FF&& f_full, FP&& f_partial)
    {
        detail::tile<ET, SO>(arch, traversal_order, m, n, f_full, f_partial);
    }
}
