// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <type_traits>


namespace blast
{
    /// @brief Compile time check for panel matrix types.
    ///
    /// @tparam T matrix type
    ///
    /// This type trait tests whether or not the given template parameter is a panel matrix type.
    ///
    template <typename T>
    struct IsPanelMatrix : std::false_type {};


    /// @brief Auxiliary variable template for the IsPanelMatrix type trait.
    ///
    template <typename T>
    bool constexpr IsPanelMatrix_v = IsPanelMatrix<T>::value;
    //*************************************************************************************************
}
