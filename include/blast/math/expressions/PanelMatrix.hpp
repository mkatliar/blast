// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>
#include <blast/math/simd/SimdSize.hpp>
#include <blast/math/simd/SimdMask.hpp>
#include <blast/math/simd/SimdIndex.hpp>
#include <blast/math/panel/PanelSize.hpp>
#include <blast/math/Matrix.hpp>

#include <blast/blaze/Math.hpp>


namespace blast
{
    template <typename Derived, bool SO>
    struct PanelMatrix
    :   public blaze::Matrix<Derived, SO>
    {
    public:
        using TagType = Group0;


        template< typename Other >  // Data type of the foreign expression
        bool isAliased( const Other* alias ) const noexcept
        {
            return static_cast<const void*>( this ) == static_cast<const void*>( alias );
        }


        template< typename Other >  // Data type of the foreign expression
        bool canAlias( const Other* alias ) const noexcept
        {
            return static_cast<const void*>( this ) == static_cast<const void*>( alias );
        }
    };


    template <typename MT, bool SO>
    inline auto * data(PanelMatrix<MT, SO>& m) noexcept
    {
        return (*m).data();
    }


    template <typename MT, bool SO>
    inline auto const * data(PanelMatrix<MT, SO> const& m) noexcept
    {
        return (*m).data();
    }


    template <typename MT, bool SO>
    inline size_t spacing(PanelMatrix<MT, SO> const& m)
    {
        return (*m).spacing();
    }


    template <typename MT1, typename MT2, typename MT3, bool SO>
    inline auto assign(PanelMatrix<MT1, SO>& lhs,
        blaze::DMatDMatMultExpr<MT2, MT3, false, false, false, false> const& rhs)
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && IsPanelMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented");
    }


    template <typename MT1, typename MT2, typename MT3, bool SO>
    inline auto assign(PanelMatrix<MT1, SO>& lhs,
        blaze::DMatTDMatMultExpr<MT2, MT3, false, false, false, false> const& rhs)
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && blaze::IsRowMajorMatrix_v<MT2> && IsPanelMatrix_v<MT3> && blaze::IsRowMajorMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented 2");
    }


    template <typename MT1, bool SO1, typename MT2, bool SO2>
    inline void assign(blaze::DenseMatrix<MT1, SO1>& lhs, PanelMatrix<MT2, SO2> const& rhs)
    {
        BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
        BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );

        using ET1 = ElementType_t<MT1>;
        using ET2 = ElementType_t<MT2>;
        static size_t constexpr SS = SimdSize_v<ET2>;
        static size_t constexpr PANEL_SIZE = PanelSize_v<ET2>;
        static_assert(PANEL_SIZE % SS == 0);

        using MaskType = SimdMask<ET2, xsimd::default_arch>;
        using IntType = typename SimdIndex<ET2>::value_type;


        if constexpr (SO1 == columnMajor && SO2 == columnMajor)
        {
            size_t const m = (*rhs).rows();
            size_t const n = (*rhs).columns();
            size_t const s = spacing(lhs);

			for (size_t i = 0; i + SS <= m; i += SS)
            {
                auto pr = ptr<aligned>(*rhs, i, 0);
                auto pl = ptr<aligned>(*lhs, i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl(0, j).store(pr(0, j).load());
            }

            if (IntType const rem = m % SS)
            {
                MaskType const mask = indexSequence<ET2>() < rem;
                size_t const i = m - rem;
                auto pr = ptr<aligned>(*rhs, i, 0);
                auto pl = ptr<aligned>(*lhs, i, 0);

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
            for (size_t i = 0; i < (*rhs).rows(); ++i)
                for (size_t j = 0; j < (*rhs).columns(); ++j)
                    (*lhs)(i, j) = (*rhs)(i, j);
        }
    }


    template <typename MT1, bool SO1, typename MT2, bool SO2>
    inline void assign(PanelMatrix<MT1, SO1>& lhs, blaze::DenseMatrix<MT2, SO2> const& rhs)
    {
        size_t const m = (*rhs).rows();
        size_t const n = (*rhs).columns();
        size_t const s = spacing(rhs);

        BLAZE_INTERNAL_ASSERT((*lhs).rows() == m, "Invalid number of rows");
        BLAZE_INTERNAL_ASSERT((*lhs).columns() == n, "Invalid number of columns");

        using ET1 = ElementType_t<MT1>;
        using ET2 = ElementType_t<MT2>;
        static size_t constexpr SS = SimdSize_v<ET2>;
        static size_t constexpr PANEL_SIZE = PanelSize_v<ET2>;
        static_assert(PANEL_SIZE % SS == 0);

        using MaskType = SimdMask<ET2, xsimd::default_arch>;
        using IntType = typename SimdIndex<ET2>::value_type;

        if constexpr (SO1 == columnMajor && SO2 == columnMajor)
        {
			for (size_t i = 0; i + SS <= m; i += SS)
            {
                auto pr = ptr<aligned>(*rhs, i, 0);
                auto pl = ptr<aligned>(*lhs, i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl(0, j).store(pr(0, j).load());
            }

            if (IntType const rem = m % SS)
            {
                size_t const i = m - rem;
                ET2 const * pr = data(rhs) + i;
                ET1 * pl = &(*lhs)(i, 0);

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
        else if constexpr (SO1 == columnMajor && SO2 == rowMajor)
        {
			for (size_t i = 0; i < m; ++i)
            {
                ET2 const * pr = data(rhs) + s * i;
                ET1 * pl = &(*lhs)(i, 0);

                for (size_t j = 0; j < n; ++j)
                    pl[PANEL_SIZE * j] = pr[j];
            }
        }
        else
        {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    (*lhs)(i, j) = (*rhs)(i, j);
        }
    }
}
