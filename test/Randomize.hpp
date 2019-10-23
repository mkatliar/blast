#pragma once

#include <smoke/StaticPanelMatrix.hpp>
#include <blaze/Math.h>

#include <array>


namespace smoke
{
    template <typename T, std::size_t N>
    inline void randomize(std::array<T, N>& a)
    {
        for (T& v : a)
            blaze::randomize(v);
    }


    template <typename T, size_t M, size_t N, size_t P>
    static void randomize(StaticPanelMatrix<T, M, N, P>& A)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-1.0, 1.0);	

        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = 0; j < A.columns(); ++j)
                A(i, j) = dis(gen);
    }
}