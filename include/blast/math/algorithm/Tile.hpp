// Copyright 2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <blast/math/Simd.hpp>

#if XSIMD_WITH_AVX2
#   include <blast/math/algorithm/arch/avx2/Tile.hpp>
#endif

#include <blast/math/StorageOrder.hpp>

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
    inline void tile(Arch arch, StorageOrder traversal_order, std::size_t m, std::size_t n, FF&& f_full, FP&& f_partial)
    {
        detail::tile<ET, SO>(arch, traversal_order, m, n, f_full, f_partial);
    }
}
