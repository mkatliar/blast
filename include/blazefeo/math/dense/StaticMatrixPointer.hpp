#pragma once

#include <blazefeo/Blaze.hpp>

#include <type_traits>


namespace blazefeo
{
    template <typename T, size_t S, bool SO>
    class StaticMatrixPointer
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


        StaticMatrixPointer constexpr offset(ptrdiff_t i, ptrdiff_t j) const noexcept
        {
            if (SO == columnMajor)
                return {ptr_ + i + spacing() * j};
            else
                return {ptr_ + spacing() * i + j};
        }


        StaticMatrixPointer<T, S, !SO> constexpr trans() const noexcept
        {
            return {ptr_};
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


    template <typename T, size_t S, bool SO>
    BLAZE_ALWAYS_INLINE auto trans(StaticMatrixPointer<T, S, SO> const& p) noexcept
    {
        return p.trans();
    }


    // NOTE:
    // IsStatic_v<...> for adapted static matrix types such as 
    // e.g. SymmetricMatrix<StaticMatrix<...>> evaluates to false;
    // therefore ptr() for these types will return a DynamicMatrixPointer,
    // which is not performance-optimal.
    //
    // See this issue: https://bitbucket.org/blaze-lib/blaze/issues/368
    //

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