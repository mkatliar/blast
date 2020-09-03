// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blazefeo
{
    /// @brief Defines the side of a matrix operation.
    ///
    enum class Side : bool
    {
        Left = false,
        Right = true
    };
}