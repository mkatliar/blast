#pragma once

#include <blazefeo/math/PanelMatrix.hpp>
#include <blazefeo/math/views/submatrix/BaseTemplate.hpp>
#include <blazefeo/system/Tile.hpp>
#include <blazefeo/system/CacheLine.hpp>

#include <blaze/util/Memory.h>
#include <blaze/util/Types.h>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/system/Restrict.h>

#include <new>


namespace blazefeo
{
    using namespace blaze;


    /// @brief Panel matrix with dynamically defined size.
    ///
    /// @tparam Type element type of the matrix
    /// @tparam SO storage order of panel elements
    template <typename Type, bool SO = columnMajor>
    class DynamicPanelMatrix
    :   public PanelMatrix<DynamicPanelMatrix<Type, SO>, SO>
    {
        BLAZE_STATIC_ASSERT_MSG((SO == columnMajor), "Row-major panel matrices are not implemented");
        
    public:
        //**Type definitions****************************************************************************
        using This          = DynamicPanelMatrix<Type, SO>;   //!< Type of this StaticPanelMatrix instance.
        using BaseType      = PanelMatrix<This, SO>;        //!< Base type of this StaticPanelMatrix instance.
        using ResultType    = This;                        //!< Result type for expression template evaluations.
        using OppositeType  = DynamicPanelMatrix<Type, !SO>;  //!< Result type with opposite storage order for expression template evaluations.
        using TransposeType = DynamicPanelMatrix<Type, !SO>;  //!< Transpose type for expression template evaluations.
        using ElementType   = Type;                        //!< Type of the matrix elements.
        // using SIMDType      = SIMDTrait_t<ElementType>;    //!< SIMD type of the matrix elements.
        using ReturnType    = const Type&;                 //!< Return type for expression template evaluations.
        using CompositeType = const This&;                 //!< Data type for composite expression templates.

        using Reference      = Type&;        //!< Reference to a non-constant matrix value.
        using ConstReference = const Type&;  //!< Reference to a constant matrix value.
        using Pointer        = Type*;        //!< Pointer to a non-constant matrix value.
        using ConstPointer   = const Type*;  //!< Pointer to a constant matrix value.


        
        explicit DynamicPanelMatrix(size_t m, size_t n)
        :   m_(m)
        ,   n_(n)
        ,   spacing_(tileSize_ * nextMultiple(n, tileSize_))
        ,   capacity_(nextMultiple(m, tileSize_) * nextMultiple(n, tileSize_))
        // Initialize padding elements to 0 to prevent denorms in calculations.
        // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
        ,   v_(new(std::align_val_t {alignment_}) Type[capacity_] {})
        {
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
            delete[] v_;
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


        ConstReference operator()(size_t i, size_t j) const noexcept
        {
            return v_[elementIndex(i, j)];
        }


        Reference operator()(size_t i, size_t j) noexcept
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


        void unpackLower(Type * data, size_t lda) const
        {
            for (size_t i = 0; i < m_; ++i)
                for (size_t j = 0; j <= i; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


        Type * ptr(size_t i, size_t j)
        {
            // BLAZE_USER_ASSERT(i % tileSize_ == 0, "Row index not aligned to panel boundary");
            return v_ + elementIndex(i, j);
        }


        Type const * ptr(size_t i, size_t j) const
        {
            // BLAZE_USER_ASSERT(i % tileSize_ == 0, "Row index not aligned to panel boundary");
            return v_ + elementIndex(i, j);
        }


    private:
        static size_t constexpr alignment_ = CACHE_LINE_SIZE;
        static size_t constexpr tileSize_ = TileSize_v<Type>;
        static size_t constexpr elementsPerTile_ = tileSize_ * tileSize_;

        size_t m_;
        size_t n_;
        size_t spacing_;
        size_t capacity_;
        
        Type * BLAZE_RESTRICT v_;


        size_t elementIndex(size_t i, size_t j) const noexcept
        {
            return i / tileSize_ * spacing_ + i % tileSize_ + j * tileSize_;
        }
    };
}


namespace blaze
{
    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    template <typename T, bool SO>
    struct HasMutableDataAccess<blazefeo::DynamicPanelMatrix<T, SO>>
    :   public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************
}