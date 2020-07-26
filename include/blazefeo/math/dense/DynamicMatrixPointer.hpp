#pragma once

#include <blazefeo/math/simd/MatrixPointer.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename T, bool SO>
    class DynamicMatrixPointer
    :   public MatrixPointerBase<DynamicMatrixPointer<T, SO>, SO>
    {
    public:
        using ElementType = T;
        static bool constexpr storageOrder = SO;

        
        constexpr DynamicMatrixPointer(T * ptr, size_t spacing) noexcept
        :   ptr_ {ptr}
        ,   spacing_ {spacing}
        {
        }


        DynamicMatrixPointer(DynamicMatrixPointer const&) = default;


        template <typename Other>
        constexpr DynamicMatrixPointer(DynamicMatrixPointer<Other, SO> const& other) noexcept
        :   ptr_ {other.ptr_}
        ,   spacing_ {other.spacing_}
        {
        }


        DynamicMatrixPointer& operator=(DynamicMatrixPointer const&) = default;


        T * get() const noexcept
        {
            return ptr_;
        }


        size_t spacing() const noexcept
        {
            return spacing_;
        }


        T * offset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            if (SO == columnMajor)
                return ptr_ + i + spacing_ * j;
            else
                return ptr_ + spacing_ * i + j;
        }


        void hmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == columnMajor)
                ptr_ += spacing_ * inc;
            else
                ptr_ += inc;
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing_ * inc;
            else
                ptr_ += inc;
        }


    private:
        T * ptr_;
        size_t spacing_;
    };


    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE std::enable_if_t<!IsStatic_v<MT>,
        DynamicMatrixPointer<ElementType_t<MT>, SO>>
        ptr(DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return {&(~m)(i, j), spacing(m)};
    }


    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE std::enable_if_t<!IsStatic_v<MT>,
        DynamicMatrixPointer<ElementType_t<MT> const, SO>>
        ptr(DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return  {&(~m)(i, j), spacing(m)};
    }


    template <bool SO, typename T>
    BLAZE_ALWAYS_INLINE DynamicMatrixPointer<T, SO> ptr(T * p, size_t spacing)
    {
        return {p, spacing};
    }
}