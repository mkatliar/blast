#pragma once

#include <blazefeo/Blaze.hpp>
#include <blazefeo/math/simd/Simd.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename T, bool SO>
    class DynamicMatrixPointer
    {
    public:
        using ElementType = T;
        using IntrinsicType = typename Simd<std::remove_cv_t<T>>::IntrinsicType;
        using MaskType = typename Simd<std::remove_cv_t<T>>::MaskType;

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


        IntrinsicType load(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return blazefeo::load<SS>(ptrOffset(i, j));
        }


        IntrinsicType maskLoad(ptrdiff_t i, ptrdiff_t j, MaskType mask) const noexcept
        {
            return blazefeo::maskload(ptrOffset(i, j), mask);
        }


        IntrinsicType broadcast(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return blazefeo::broadcast<SS>(ptrOffset(i, j));
        }


        void store(ptrdiff_t i, ptrdiff_t j, IntrinsicType val) const noexcept
        {
            blazefeo::store(ptrOffset(i, j), val);
        }


        void maskStore(ptrdiff_t i, ptrdiff_t j, MaskType mask, IntrinsicType val) const noexcept
        {
            blazefeo::maskstore(ptrOffset(i, j), mask, val);
        }


        size_t spacing() const noexcept
        {
            return spacing_;
        }


        DynamicMatrixPointer offset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            return {ptrOffset(i, j), spacing_};
        }


        DynamicMatrixPointer<T, !SO> constexpr trans() const noexcept
        {
            return {ptr_, spacing_};
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
        static size_t constexpr SS = Simd<std::remove_cv_t<T>>::size;

        
        T * ptrOffset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            if (SO == columnMajor)
                return ptr_ + i + spacing_ * j;
            else
                return ptr_ + spacing_ * i + j;
        }


        T * ptr_;
        size_t spacing_;
    };


    template <bool SO, typename T>
    BLAZE_ALWAYS_INLINE auto trans(DynamicMatrixPointer<T, SO> const& p) noexcept
    {
        return p.trans();
    }


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