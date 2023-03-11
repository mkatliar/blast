// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blazefeo/math/typetraits/IsPanelMatrix.hpp>
#include <blazefeo/math/simd/Simd.hpp>
#include <blazefeo/math/panel/PanelSize.hpp>
//#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <blaze/math/ReductionFlag.h>
#include <blaze/math/Matrix.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Types.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename Derived, bool SO>
    struct PanelMatrix
    :   public Matrix<Derived, SO>
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
    inline auto * ptr(PanelMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return (*m).ptr(i, j);
    }


    template <typename MT, bool SO>
    inline auto const * ptr(PanelMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return (*m).ptr(i, j);
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
        -> blaze::EnableIf_t<IsPanelMatrix_v<MT2> && IsRowMajorMatrix_v<MT2> && IsPanelMatrix_v<MT3> && IsRowMajorMatrix_v<MT3>>
    {
        BLAZE_THROW_LOGIC_ERROR("Not implemented 2");
    }


    template <typename MT1, bool SO1, typename MT2, bool SO2>
    inline void assign(DenseMatrix<MT1, SO1>& lhs, PanelMatrix<MT2, SO2> const& rhs)
    {
        BLAZE_INTERNAL_ASSERT( (*lhs).rows()    == (*rhs).rows()   , "Invalid number of rows"    );
        BLAZE_INTERNAL_ASSERT( (*lhs).columns() == (*rhs).columns(), "Invalid number of columns" );

        using ET1 = ElementType_t<MT1>;
        using ET2 = ElementType_t<MT2>;
        static size_t constexpr SS = Simd<ET2>::size;
        static size_t constexpr PANEL_SIZE = PanelSize_v<ET2>;
        static_assert(PANEL_SIZE % SS == 0);

        using MaskType = typename Simd<ET2>::MaskType;
        using IntType = typename Simd<ET2>::IntType;
        using SIMD = Simd<ET2>;


        if constexpr (SO1 == columnMajor && SO2 == columnMajor)
        {
            size_t const m = (*rhs).rows();
            size_t const n = (*rhs).columns();
            size_t const s = spacing(lhs);

			for (size_t i = 0; i + SS <= m; i += SS)
            {
                ET2 const * pr = ptr(rhs, i, 0);
                ET1 * pl = data(lhs) + i;

                for (size_t j = 0; j < n; ++j)
                    store<aligned>(pl + s * j, load<aligned, SS>(pr + PANEL_SIZE * j));
            }

            if (IntType const rem = m % SS)
            {
                MaskType const mask = SIMD::index() < rem;
                size_t const i = m - rem;
                ET2 const * pr = ptr(rhs, i, 0);
                ET1 * pl = data(lhs) + i;

                for (size_t j = 0; j < n; ++j)
                    maskstore(pl + s * j, mask, load<aligned, SS>(pr + PANEL_SIZE * j));
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
    inline void assign(PanelMatrix<MT1, SO1>& lhs, DenseMatrix<MT2, SO2> const& rhs)
    {
        size_t const m = (*rhs).rows();
        size_t const n = (*rhs).columns();
        size_t const s = spacing(rhs);

        BLAZE_INTERNAL_ASSERT((*lhs).rows() == m, "Invalid number of rows");
        BLAZE_INTERNAL_ASSERT((*lhs).columns() == n, "Invalid number of columns");

        using ET1 = ElementType_t<MT1>;
        using ET2 = ElementType_t<MT2>;
        static size_t constexpr SS = Simd<ET2>::size;
        static size_t constexpr PANEL_SIZE = PanelSize_v<ET2>;
        static_assert(PANEL_SIZE % SS == 0);

        using MaskType = typename Simd<ET2>::MaskType;
        using IntType = typename Simd<ET2>::IntType;

        if constexpr (SO1 == columnMajor && SO2 == columnMajor)
        {
			for (size_t i = 0; i + SS <= m; i += SS)
            {
                ET2 const * pr = data(rhs) + i;
                ET1 * pl = ptr(lhs, i, 0);

                for (size_t j = 0; j < n; ++j)
                    store<aligned>(pl + PANEL_SIZE * j, load<aligned, SS>(pr + s * j));
            }

            if (IntType const rem = m % SS)
            {
                size_t const i = m - rem;
                ET2 const * pr = data(rhs) + i;
                ET1 * pl = ptr(lhs, i, 0);

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
                ET1 * pl = ptr(lhs, i, 0);

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