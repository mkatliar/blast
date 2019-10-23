#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Block.hpp>
#include <smoke/PaddedSize.hpp>
#include <smoke/PanelMatrix.hpp>

#include <memory>
#include <cstdlib>
#include <algorithm>


namespace smoke
{
    template <typename T, size_t P = blockSize, size_t AL = blockAlignment>
    class DynamicPanelMatrix
    :   public PanelMatrix<DynamicPanelMatrix<T, P, AL>, P>
    {
    public:
        using ElementType = T;

        
        DynamicPanelMatrix(size_t m, size_t n)
        :   m_(m)
        ,   n_(n)
        ,   spacing_(P * paddedSize(n, P))
        ,   capacity_(paddedSize(m, P) * paddedSize(n, P))
        ,   v_(reinterpret_cast<T *>(std::aligned_alloc(AL, capacity_ * sizeof(T))), &std::free)
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_.get(), capacity_, T {});
        }


        DynamicPanelMatrix& operator=(T val)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
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


        size_t rows() const
        {
            return m_;
        }


        size_t columns() const
        {
            return n_;
        }


        size_t spacing() const
        {
            return spacing_;
        }


        void pack(T const * data, size_t lda)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(T * data, size_t lda) const
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        T * block(size_t i, size_t j)
        {
            return v_.get() + i * spacing_ + j * elementsPerPanel_;
        }


        T const * block(size_t i, size_t j) const
        {
            return v_.get() + i * spacing_ + j * elementsPerPanel_;
        }


    private:
        static size_t constexpr elementsPerPanel_ = P * P;

        size_t m_;
        size_t n_;
        size_t spacing_;
        size_t capacity_;
        
        std::unique_ptr<T[], decltype(&std::free)> v_;


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return panel_i * spacing_ + panel_j * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };
}