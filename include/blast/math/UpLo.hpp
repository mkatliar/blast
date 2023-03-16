// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /// @brief Defines the part of a triangular matrix.
    ///
    enum class UpLo : bool
    {
        Lower = false,
        Upper = true
    };
}