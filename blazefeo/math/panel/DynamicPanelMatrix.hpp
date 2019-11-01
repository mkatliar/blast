#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/system/Tile.hpp>

#include <blaze/util/Memory.h>
#include <blaze/util/Types.h>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/system/Restrict.h>

#include <algorithm>


namespace blazefeo
{
    using namespace blaze;


    template <typename Type, bool SO = rowMajor>
    class DynamicPanelMatrix
    :   public PanelMatrix<DynamicPanelMatrix<Type, SO>, SO>
    {
    public:
        using ElementType = Type;

        
        explicit DynamicPanelMatrix(size_t m, size_t n)
        :   m_(m)
        ,   n_(n)
        ,   spacing_(tileSize_ * nextMultiple(n, tileSize_))
        ,   capacity_(nextMultiple(m, tileSize_) * nextMultiple(n, tileSize_))
        ,   v_(allocate<Type>(capacity_))
        {
            // Initialize padding elements to 0 to prevent denorms in calculations.
            // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
            std::fill_n(v_, capacity_, Type {});
        }


        DynamicPanelMatrix(DynamicPanelMatrix const& rhs)
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
        }


        DynamicPanelMatrix(DynamicPanelMatrix&& rhs) noexcept
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
        }


        ~DynamicPanelMatrix()
        {
            deallocate(v_);
        }


        DynamicPanelMatrix& operator=(Type val) noexcept
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = val;

            return *this;
        }


        DynamicPanelMatrix& operator=(DynamicPanelMatrix const& val)
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
            return *this;
        }


        DynamicPanelMatrix& operator=(DynamicPanelMatrix&& val) noexcept
        {
            BLAZE_THROW_LOGIC_ERROR("Not implemented");
            return *this;
        }


        Type operator()(size_t i, size_t j) const noexcept
        {
            return v_[elementIndex(i, j)];
        }


        Type& operator()(size_t i, size_t j) noexcept
        {
            return v_[elementIndex(i, j)];
        }


        size_t rows() const noexcept
        {
            return m_;
        }


        size_t columns() const noexcept
        {
            return n_;
        }


        size_t spacing() const
        {
            return spacing_;
        }


        void pack(Type const * data, size_t lda)
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(Type * data, size_t lda) const
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j < n_; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        Type * tile(size_t i, size_t j) noexcept
        {
            return v_ + i * spacing_ + j * elementsPerTile_;
        }


        Type const * tile(size_t i, size_t j) const noexcept
        {
            return v_ + i * spacing_ + j * elementsPerTile_;
        }


    private:
        static size_t constexpr tileSize_ = TileSize_v<Type>;
        static size_t constexpr elementsPerTile_ = tileSize_ * tileSize_;

        size_t m_;
        size_t n_;
        size_t spacing_;
        size_t capacity_;
        
        Type * BLAZE_RESTRICT v_;


        size_t elementIndex(size_t i, size_t j) const noexcept
        {
            size_t const panel_i = i / tileSize_;
            size_t const panel_j = j / tileSize_;
            size_t const subpanel_i = i % tileSize_;
            size_t const subpanel_j = j % tileSize_;

            return panel_i * spacing_ + panel_j * elementsPerTile_ + subpanel_i + subpanel_j * tileSize_;
        }
    };
}
