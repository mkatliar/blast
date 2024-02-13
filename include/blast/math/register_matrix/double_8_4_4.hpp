// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/register_matrix/RegisterMatrix.hpp>
#include <blast/util/Exception.hpp>

#include <blaze/util/Exception.h>

#include <immintrin.h>


namespace blast
{
    template<>
    template <typename P>
    requires MatrixPointer<P, double> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<double, 8, 4, columnMajor>::load(double beta, P ptr, size_t m, size_t n) noexcept
    {
        if (n > 0)
        {
            v_[0][0] = beta * ptr.load();
            v_[1][0] = beta * ptr(SS, 0).load();
        }

        if (n > 1)
        {
            v_[0][1] = beta * ptr(0, 1).load();
            v_[1][1] = beta * ptr(SS, 1).load();
        }

        if (n > 2)
        {
            v_[0][2] = beta * ptr(0, 2).load();
            v_[1][2] = beta * ptr(SS, 2).load();
        }

        if (n > 3)
        {
            v_[0][3] = beta * ptr(0, 3).load();
            v_[1][3] = beta * ptr(SS, 3).load();
        }
    }


#if 1
    template<>
    template <typename P>
    requires MatrixPointer<P, double> && (P::storageOrder == columnMajor)
    inline void RegisterMatrix<double, 8, 4, columnMajor>::store(P ptr, size_t m, size_t n) const noexcept
    {
        #pragma unroll
        for (size_t i = 0; i < 2; ++i)
        {
            if (m >= 4 * i + 4)
            {
                if (n > 0)
                    ptr(SS * i, 0).store(v_[i][0]);

                if (n > 1)
                    ptr(SS * i, 1).store(v_[i][1]);

                if (n > 2)
                    ptr(SS * i, 2).store(v_[i][2]);

                if (n > 3)
                    ptr(SS * i, 3).store(v_[i][3]);
            }
            else if (m > 4 * i)
            {
                __m256i const mask = _mm256_set_epi64x(
                    m > 4 * i + 3 ? 0x8000000000000000ULL : 0,
                    m > 4 * i + 2 ? 0x8000000000000000ULL : 0,
                    m > 4 * i + 1 ? 0x8000000000000000ULL : 0,
                    m > 4 * i + 0 ? 0x8000000000000000ULL : 0);

                if (n > 0)
                    ptr(SS * i, 0).store(v_[i][0], mask);

                if (n > 1)
                    ptr(SS * i, 1).store(v_[i][1], mask);

                if (n > 2)
                    ptr(SS * i, 2).store(v_[i][2], mask);

                if (n > 3)
                    ptr(SS * i, 3).store(v_[i][3], mask);
            }
        }
    }
#endif


#ifdef BLAST_USE_CUSTOM_TRMM_RIGHT_LOWER
    template <>
    template <typename PB, typename PA>
        requires MatrixPointer<PB, double> && (PB::storageOrder == columnMajor) && MatrixPointer<PA, double>
    BLAZE_ALWAYS_INLINE void RegisterMatrix<double, 8, 4, columnMajor>::trmmRightLower(double alpha, PB b, PA a) noexcept
    {
        IntrinsicType bx[2], ax;

        // k == 0
        bx[0] = alpha * b.load(0 * SS, 0);
        bx[1] = alpha * b.load(1 * SS, 0);
        ax = a.broadcast(0, 0);
        v_[0][0] = fmadd(bx[0], ax, v_[0][0]);
        v_[1][0] = fmadd(bx[1], ax, v_[1][0]);

        // k == 1
        bx[0] = alpha * b.load(0 * SS, 1);
        bx[1] = alpha * b.load(1 * SS, 1);
        ax = a.broadcast(1, 0);
        v_[0][0] = fmadd(bx[0], ax, v_[0][0]);
        v_[1][0] = fmadd(bx[1], ax, v_[1][0]);
        ax = a.broadcast(1, 1);
        v_[0][1] = fmadd(bx[0], ax, v_[0][1]);
        v_[1][1] = fmadd(bx[1], ax, v_[1][1]);

        // k == 2
        bx[0] = alpha * b.load(0 * SS, 2);
        bx[1] = alpha * b.load(1 * SS, 2);
        ax = a.broadcast(2, 0);
        v_[0][0] = fmadd(bx[0], ax, v_[0][0]);
        v_[1][0] = fmadd(bx[1], ax, v_[1][0]);
        ax = a.broadcast(2, 1);
        v_[0][1] = fmadd(bx[0], ax, v_[0][1]);
        v_[1][1] = fmadd(bx[1], ax, v_[1][1]);
        ax = a.broadcast(2, 2);
        v_[0][2] = fmadd(bx[0], ax, v_[0][2]);
        v_[1][2] = fmadd(bx[1], ax, v_[1][2]);

        // k == 3
        bx[0] = alpha * b.load(0 * SS, 3);
        bx[1] = alpha * b.load(1 * SS, 3);
        ax = a.broadcast(3, 0);
        v_[0][0] = fmadd(bx[0], ax, v_[0][0]);
        v_[1][0] = fmadd(bx[1], ax, v_[1][0]);
        ax = a.broadcast(3, 1);
        v_[0][1] = fmadd(bx[0], ax, v_[0][1]);
        v_[1][1] = fmadd(bx[1], ax, v_[1][1]);
        ax = a.broadcast(3, 2);
        v_[0][2] = fmadd(bx[0], ax, v_[0][2]);
        v_[1][2] = fmadd(bx[1], ax, v_[1][2]);
        ax = a.broadcast(3, 3);
        v_[0][3] = fmadd(bx[0], ax, v_[0][3]);
        v_[1][3] = fmadd(bx[1], ax, v_[1][3]);
    }
#endif
}