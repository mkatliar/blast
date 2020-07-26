#pragma once

#include <blazefeo/math/simd/MatrixPointer.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename T, size_t S, bool SO>
    class StaticMatrixPointer
    :   public MatrixPointer<StaticMatrixPointer<T, S, SO>, SO>
    {
    public:
        using ElementType = T;
        static bool constexpr storageOrder = SO;

        
        constexpr StaticMatrixPointer(T * ptr) noexcept
        :   ptr_ {ptr}
        {
        }


        StaticMatrixPointer(StaticMatrixPointer const&) = default;


        template <typename Other>
        constexpr StaticMatrixPointer(StaticMatrixPointer<Other, S, SO> const& other) noexcept
        :   ptr_ {other.ptr_}
        {
        }


        StaticMatrixPointer& operator=(StaticMatrixPointer const&) = default;


        T * get() const noexcept
        {
            return ptr_;
        }


        size_t constexpr spacing() const noexcept
        {
            return S;
        }


        void hmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == columnMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;
        }


        void vmove(ptrdiff_t inc) noexcept
        {
            if constexpr (SO == rowMajor)
                ptr_ += spacing() * inc;
            else
                ptr_ += inc;
        }


    private:
        T * ptr_;
    };


    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE std::enable_if_t<IsStatic_v<MT>,
        StaticMatrixPointer<ElementType_t<MT>, MT::spacing(), SO>> 
        ptr(DenseMatrix<MT, SO>& m, size_t i, size_t j)
    {
        return &(~m)(i, j);
    }


    template <typename MT, bool SO>
    BLAZE_ALWAYS_INLINE std::enable_if_t<IsStatic_v<MT>,
        StaticMatrixPointer<ElementType_t<MT> const, MT::spacing(), SO>> 
        ptr(DenseMatrix<MT, SO> const& m, size_t i, size_t j)
    {
        return &(~m)(i, j);
    }
}