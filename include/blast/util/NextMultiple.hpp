#pragma once

#include <blast/util/Types.hpp>


namespace blast
{
    /**
     * @brief Rounds up an integral value to the next multiple of a given factor.
    //
    // @param value The integral value to be rounded up \f$[1..\infty)\f$.
    // @param factor The factor of the multiple \f$[1..\infty)\f$.
    // @return The next multiple of the given factor.
    //
    // This function rounds up the given integral value to the next multiple of the given integral
    // factor. In case the integral value is already a multiple of the given factor, the value itself
    // is returned.
     */
    inline size_t constexpr nextMultiple(size_t value, size_t factor) noexcept
    {
        return value + (factor - value % factor) % factor;
    }
}
