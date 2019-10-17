#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Block.hpp>

#include <array>

namespace smoke
{
    template <typename T, size_t M, size_t N, size_t P = blockSize, size_t AL = blockAlignment>
    class StaticMatrix
    {
    public:
        StaticMatrix()
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            v_.fill(T {});
        }


        StaticMatrix& operator=(T val)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        T operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        T& operator()(size_t i, size_t j)
        {
            return v_[elementIndex(i, j)];
        }


        size_t constexpr rows() const
        {
            return M;
        }


        size_t constexpr columns() const
        {
            return N;
        }


        size_t constexpr spacing() const
        {
            return panelColumns_ * elementsPerPanel_;
        }


        size_t constexpr panelRows() const
        {
            return panelRows_;
        }


        size_t constexpr panelColumns() const
        {
            return panelColumns_;
        }


        void pack(T const * data, size_t lda)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(T * data, size_t lda) const
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        T * block(size_t i, size_t j)
        {
            return v_.data() + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        T const * block(size_t i, size_t j) const
        {
            return v_.data() + (i * panelColumns_ + j) * elementsPerPanel_;
        }


    private:
        static size_t constexpr panelRows_ = M / P + (M % P > 0);
        static size_t constexpr panelColumns_ = N / P + (N % P > 0);
        static size_t constexpr elementsPerPanel_ = P * P;

        alignas(AL) std::array<T, panelRows_ * panelColumns_ * elementsPerPanel_> v_;


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return (panel_i * panelColumns_ + panel_j) * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };
}