// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/Simd.hpp>
#include <blast/math/panel/PanelSize.hpp>
#include <blast/math/Matrix.hpp>
#include <blast/math/AlignmentFlag.hpp>

#include <stdexcept>


namespace blast
{
    template <Matrix MT1, Matrix MT2>
    requires IsDenseMatrix_v<MT1> && IsPanelMatrix_v<MT2>
    inline void assign(MT1& lhs, MT2 const& rhs)
    {
        if (rows(lhs) != rows(rhs))
            throw std::invalid_argument {"Invalid number of rows"};

        if (columns(lhs) != columns(rhs))
            throw std::invalid_argument {"Invalid number of columns"};

        using ET1 = ElementType_t<MT1>;
        using ET2 = ElementType_t<MT2>;
        static size_t constexpr SS = SimdSize_v<ET2>;
        static size_t constexpr PANEL_SIZE = PanelSize_v<ET2>;
        static_assert(PANEL_SIZE % SS == 0);

        using MaskType = SimdMask<ET2, xsimd::default_arch>;
        using IntType = typename SimdIndex<ET2>::value_type;


        if constexpr (StorageOrder_v<MT1> == columnMajor && StorageOrder_v<MT2> == columnMajor)
        {
            size_t const m = rows(rhs);
            size_t const n = columns(rhs);
            size_t const s = spacing(lhs);

			for (size_t i = 0; i + SS <= m; i += SS)
            {
                auto pr = ptr<aligned>(rhs, i, 0);
                auto pl = ptr<aligned>(lhs, i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl(0, j).store(pr(0, j).load());
            }

            if (IntType const rem = m % SS)
            {
                MaskType const mask = indexSequence<ET2>() < rem;
                size_t const i = m - rem;
                auto pr = ptr<aligned>(rhs, i, 0);
                auto pl = ptr<aligned>(lhs, i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl(0, j).store(pr(0, j).load(), mask);
            }

		#if 0
            RegisterMatrix<ET2, SS, SS, SS> ker;

            for (size_t i = 0; i + SS <= m; i += SS)
            {
                for (size_t j = 0; j + SS <= n; ++j)
                {
                    ker.load(1., rhs, i, j, SS, SS);
                    ker.store(submatrix(*lhs, i, j, SS, SS));
                }

                if (size_t const rn = n % SS)
                {
                    size_t const j = n - rn;
                    ker.load(1., rhs, i, j, SS, rn);
                    ker.store(submatrix(*lhs, i, j, SS, rn));
                }
            }

            if (size_t const rm = m % SS)
            {
                size_t const i = m - rm;

                for (size_t j = 0; j + SS <= n; ++j)
                {
                    ker.load(1., rhs, i, j, rm, SS);
                    ker.store(submatrix(lhs, i, j, rm, SS));
                }

                if (size_t const rn = n % SS)
                {
                    size_t const j = n - rn;
                    ker.load(1., rhs, i, j, rm, rn);
                    ker.store(submatrix(lhs, i, j, rm, rn));
                }
            }
		#endif
        }
        else
        {
            for (size_t i = 0; i < rows(rhs); ++i)
                for (size_t j = 0; j < columns(rhs); ++j)
                    lhs(i, j) = rhs(i, j);
        }
    }


    template <Matrix MT1, Matrix MT2>
    requires IsPanelMatrix_v<MT1> && IsDenseMatrix_v<MT2>
    inline void assign(MT1& lhs, MT2 const& rhs)
    {
        size_t const m = rows(rhs);
        size_t const n = columns(rhs);
        size_t const s = spacing(rhs);

        if (rows(lhs) != m)
            throw std::invalid_argument {"Invalid number of rows"};
        if (columns(lhs) != n)
            throw std::invalid_argument {"Invalid number of columns"};

        using ET1 = ElementType_t<MT1>;
        using ET2 = ElementType_t<MT2>;
        static size_t constexpr SS = SimdSize_v<ET2>;
        static size_t constexpr PANEL_SIZE = PanelSize_v<ET2>;
        static_assert(PANEL_SIZE % SS == 0);

        using MaskType = SimdMask<ET2, xsimd::default_arch>;
        using IntType = typename SimdIndex<ET2>::value_type;

        if constexpr (StorageOrder_v<MT1> == columnMajor && StorageOrder_v<MT2> == columnMajor)
        {
			for (size_t i = 0; i + SS <= m; i += SS)
            {
                auto pr = ptr<aligned>(rhs, i, 0);
                auto pl = ptr<aligned>(lhs, i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl(0, j).store(pr(0, j).load());
            }

            if (IntType const rem = m % SS)
            {
                size_t const i = m - rem;
                ET2 const * pr = data(rhs) + i;
                ET1 * pl = &lhs(i, 0);

                for (size_t j = 0; j < n; ++j)
                    for (size_t i1 = 0; i1 < rem; ++i1)
                        pl[i1 + PANEL_SIZE * j] = pr[i1 + s * j];
            }

            // if (IntType const rem = m % SS)
            // {
            //     MaskType const mask = SIMD::index() < rem;
            //     size_t const i = m - rem;
            //     ET2 const * pr = ptr(rhs, i, 0);
            //     ET1 * pl = data(lhs) + i;

            //     for (size_t j = 0; j < n; ++j)
            //         maskstore(pl + PANEL_SIZE * j, mask, load<aligned, SS>(pr + s * j));
            // }
        }
        else if constexpr (StorageOrder_v<MT1> == columnMajor && StorageOrder_v<MT2> == rowMajor)
        {
			for (size_t i = 0; i < m; ++i)
            {
                ET2 const * pr = data(rhs) + s * i;
                ET1 * pl = &lhs(i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl[PANEL_SIZE * j] = pr[j];
            }
        }
        else
        {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    lhs(i, j) = rhs(i, j);
        }
    }
}
