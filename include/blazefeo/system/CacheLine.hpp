// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blaze/util/Types.h>


namespace blazefeo
{
    using namespace blaze;

    
    /// @brief Size of the cache line, depending on the architecture.
    size_t constexpr CACHE_LINE_SIZE = 0x40;
}