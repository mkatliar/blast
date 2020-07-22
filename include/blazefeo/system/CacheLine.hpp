#pragma once

#include <blaze/util/Types.h>


namespace blazefeo
{
    using namespace blaze;

    
    /// @brief Size of the cache line, depending on the architecture.
    size_t constexpr CACHE_LINE_SIZE = 0x40;
}