#pragma once

#include <smoke/SizeT.hpp>

#include <random>


namespace smoke
{
    template <typename Derived, size_t P>
    struct PanelMatrix
    {
        Derived& operator~() noexcept
        {
            return static_cast<Derived&>(*this);
        }


        Derived const& operator~() const noexcept
        {
            return static_cast<Derived const&>(*this);
        }
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


    template <typename MT, size_t P>
    static void randomize(PanelMatrix<MT, P>& A)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-1.0, 1.0);	

        for (size_t i = 0; i < rows(A); ++i)
            for (size_t j = 0; j < columns(A); ++j)
                (~A)(i, j) = dis(gen);
    }
}