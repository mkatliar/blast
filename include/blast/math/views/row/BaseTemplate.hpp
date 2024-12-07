// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blast/math/TypeTraits.hpp>


namespace blast
{
    template <Matrix MT>
    requires IsPanelMatrix_v<MT>
    class PanelMatrixRow;
}
