#pragma once

#include <smoke/SizeT.hpp>


namespace smoke
{
    template <typename T, size_t N>
    class Panel
    {
    public:
        void load(T const * ptr)
        {
            for (size_t j = 0; j < N * N; ++j)
                v_[j] = ptr[j];
        }


        void store(T * ptr)
        {
            for (size_t j = 0; j < N * N; ++j)
                ptr[j] = v_[j];
        }


        friend void gemm(Panel const& a, Panel const& b, Panel& c)
        {
            for (size_t j = 0; j < N; ++j)
                for (size_t i = 0; i < N; ++i)
                    for (size_t k = 0; k < N; ++k)
                        c.v_[i + N * j] += a.v_[i + k * N] * b.v_[k + j * N];
        }


    private:
        T v_[N * N];
    };
}