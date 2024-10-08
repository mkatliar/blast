// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/util/Types.hpp>


namespace blast
{
    template< typename MT       // Type of the panel matrix
            , bool SO
            , size_t... CSAs >  // Compile time submatrix arguments
    class PanelSubmatrix
    {
    };
}
