// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/PanelMatrix.hpp>
#include <blast/math/views/submatrix/BaseTemplate.hpp>
#include <blast/math/simd/SimdSize.hpp>
#include <blast/math/TypeTraits.hpp>
#include <blast/system/CacheLine.hpp>

#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Memory.h>
#include <blaze/util/Types.h>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/system/Restrict.h>

#include <new>
#include <type_traits>


namespace blast
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
        :   m_ {m}
        ,   n_ {n}
        ,   spacing_ {SS * (SO == columnMajor ? n : m)}
        ,   capacity_ {spacing_ * nextMultiple(SO == columnMajor ? m : n, SS)}
        // Initialize padding elements to 0 to prevent denorms in calculations.
        // Initialize padding elements to 0 to prevent denorms in calculations.
        // Denorms can significantly impair performance, see https://github.com/giaf/blasfeo/issues/103
        ,   v_ {new(std::align_val_t {alignment_}) Type[capacity_] {}}
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


        template< typename MT    // Type of the right-hand side matrix
            , bool SO2 >      // Storage order of the right-hand side matrix
        DynamicPanelMatrix& operator=(blaze::Matrix<MT, SO2> const& rhs)
        {
            assign(*this, *rhs);
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


        size_t spacing() const noexcept
        {
            return spacing_;
        }


        Type * data() noexcept
        {
            return v_;
        }


        Type const * data() const noexcept
        {
            return v_;
        }


        /**
         * @brief Set all matrix elements to 0
         */
        void reset() noexcept
        {
            std::fill_n(v_, spacing_ * (SO == columnMajor ? n_ : m_), Type {});
        }


    private:
        static size_t constexpr alignment_ = CACHE_LINE_SIZE;
        static size_t constexpr SS = SimdSize_v<Type>;

        size_t m_;
        size_t n_;
        size_t spacing_;
        size_t capacity_;

        Type * BLAZE_RESTRICT v_;


        size_t elementIndex(size_t i, size_t j) const noexcept
        {
            return SO == columnMajor
                ? i / SS * spacing_ + i % SS + j * SS
                : j / SS * spacing_ + j % SS + i * SS;
        }
    };


    /**
     * @brief Specialization for @a DynamicPanelMatrix
     */
    template <typename T, bool SO>
    struct StorageOrderHelper<DynamicPanelMatrix<T, SO>> : std::integral_constant<StorageOrder, StorageOrder(SO)> {};
}


namespace blaze
{
    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    template <typename T, bool SO>
    struct HasMutableDataAccess<blast::DynamicPanelMatrix<T, SO>>
    :   public TrueType
    {};
    /*! \endcond */
    //*************************************************************************************************


    //=================================================================================================
    //
    //  IsStatic specialization
    //
    //=================================================================================================
    template <typename T, bool SO>
    struct IsStatic<blast::DynamicPanelMatrix<T, SO>>
    :   public FalseType
    {};


    //=================================================================================================
    //
    //  IsAligned specialization
    //
    //=================================================================================================
    template <typename T, bool SO>
    struct IsAligned<blast::DynamicPanelMatrix<T, SO>>
    :   public TrueType
    {};


    //=================================================================================================
    //
    //  IsPadded specialization
    //
    //=================================================================================================

    //*************************************************************************************************
    /*! \cond BLAZE_INTERNAL */
    template <typename T, bool SO>
    struct IsPadded<blast::DynamicPanelMatrix<T, SO>>
    : public TrueType
    {};


    //=================================================================================================
    //
    //  SubmatrixTrait specialization
    //
    //=================================================================================================
    template <typename T, bool SO>
    struct SubmatrixTrait<blast::DynamicPanelMatrix<T, SO>>
    {
        using Type = blast::DynamicPanelMatrix<T, SO>;
    };
}
