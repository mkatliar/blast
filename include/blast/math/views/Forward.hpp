// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blast
{
    //=================================================================================================
    //
    //  ::blast NAMESPACE FORWARD DECLARATIONS
    //
    //=================================================================================================

    template< typename MT       // Type of the panel matrix
            , bool SO
            , size_t... CSAs >  // Compile time submatrix arguments
    class PanelSubmatrix;
}
