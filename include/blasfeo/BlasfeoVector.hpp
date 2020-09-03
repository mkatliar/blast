// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo_common.h>


namespace blasfeo
{
    /// @brief BLASFEO vector type selector
    template <typename Real>
    struct BlasfeoVector;


    template <>
    struct BlasfeoVector<double>
    {
        using type = blasfeo_dvec;
    };


    template <>
    struct BlasfeoVector<float>
    {
        using type = blasfeo_svec;
    };


    /// @brief BLASFEO vector type selector alias
    template <typename Real>
    using BlasfeoVector_t = typename BlasfeoVector<Real>::type;
}