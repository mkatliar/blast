// Copyright 2023-2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    /**
     * @brief Defines whether a vector is a row or a column
     *
     * The bool values match those of @a blaze::TransposeFlag
     */
    enum TransposeFlag : bool
    {
        rowVector = true,
        columnVector = false
    };


    inline TransposeFlag constexpr operator!(TransposeFlag tf)
    {
        return tf == TransposeFlag::rowVector ? TransposeFlag::columnVector : TransposeFlag::rowVector;
    }
}
