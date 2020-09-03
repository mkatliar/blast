// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo_common.h>


namespace blasfeo
{
    /// @brief BLASFEO matrix type selector
    template <typename Real>
    struct BlasfeoMatrix;


    template <>
    struct BlasfeoMatrix<double>
    {
        using type = blasfeo_dmat;
    };


    template <>
    struct BlasfeoMatrix<float>
    {
        using type = blasfeo_smat;
    };


    /// @brief BLASFEO matrix type selector alias
    template <typename Real>
    using BlasfeoMatrix_t = typename BlasfeoMatrix<Real>::type;
}