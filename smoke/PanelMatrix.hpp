#pragma once

#include <smoke/SizeT.hpp>

#include <blaze/math/DenseMatrix.h>

#include <random>


namespace smoke
{
    template <typename Derived, size_t P>
    struct PanelMatrix
    :   public blaze::DenseMatrix<Derived, blaze::rowMajor>
    {
    };


    template <typename MT, size_t P>
    inline typename MT::ElementType * block(PanelMatrix<MT, P>& m, size_t i, size_t j)
    {
        return (~m).block(i, j);
    }


    template <typename MT, size_t P>
    inline typename MT::ElementType const * block(PanelMatrix<MT, P> const& m, size_t i, size_t j)
    {
        return (~m).block(i, j);
    }


    template <typename MT, size_t P>
    inline size_t spacing(PanelMatrix<MT, P> const& m)
    {
        return (~m).spacing();
    }


    template <typename MT, size_t P>
    inline size_t rows(PanelMatrix<MT, P> const& m)
    {
        return (~m).rows();
    }


    template <typename MT, size_t P>
    inline size_t columns(PanelMatrix<MT, P> const& m)
    {
        return (~m).columns();
    }
}