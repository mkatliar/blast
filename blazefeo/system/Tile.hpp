#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Vectorization.h>
#include <blaze/util/IntegralConstant.h>


namespace blazefeo
{
    using namespace blaze;


    template <typename T>
    struct TileSizeHelper
    {
    public:
        static constexpr size_t value = 1;
    };


    template <>
    struct TileSizeHelper<double>
    {
    public:
        static constexpr size_t value =
        #if BLAZE_AVX2_MODE
            4;
        #else
            1;
        #endif
    };
    

    template <typename T>
    struct TileSize
    :   public IntegralConstant<size_t, TileSizeHelper<T>::value>
    {};
    
    
    template <typename T>
    struct TileSize<const T>
    :   public IntegralConstant<size_t, TileSizeHelper<T>::value>
    {};
    
    
    template <typename T>
    struct TileSize<volatile T>
    :   public IntegralConstant<size_t, TileSizeHelper<T>::value>
    {};
    

    template <typename T>
    struct TileSize<volatile const T>
    :   public IntegralConstant<size_t, TileSizeHelper<T>::value>
    {};
    
    
    template <typename T>
    size_t constexpr TileSize_v = TileSize<T>::value;
}
