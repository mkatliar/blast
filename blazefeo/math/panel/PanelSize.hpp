#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/SIMDTrait.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename T>
    size_t constexpr PanelSize_v = SIMDTrait<T>::size;
}
