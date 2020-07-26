#pragma once

#include <blazefeo/Blaze.hpp>


namespace blazefeo
{
    template <typename Derived, bool SO>
    class MatrixPointer
    {
    public:
        Derived& operator~() noexcept
        {
            return static_cast<Derived&>(*this);
        }


        Derived const& operator~() const noexcept
        {
            return static_cast<Derived const&>(*this);
        }


    protected:
        MatrixPointer() = default;
        MatrixPointer(MatrixPointer const&) = default;
    };
}