// Copyright 2023-2024 Mikhail Katliar. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    enum StorageOrder : bool
    {
        rowMajor = false,
        columnMajor = true
    };


    inline constexpr StorageOrder operator!(StorageOrder so)
    {
        return so == rowMajor ? columnMajor : rowMajor;
    }
}
