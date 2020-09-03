// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once


namespace blazefeo :: testing
{
    template <typename ET>
    ET absTol();


    template <typename ET>
    ET relTol();


    template <>
    inline double absTol<double>()
    {
        return 1e-11;
    }


    template <>
    inline double relTol<double>()
    {
        return 1e-11;
    }


    template <>
    inline float absTol<float>()
    {
        return 1e-5f;
    }


    template <>
    inline float relTol<float>()
    {
        return 1e-4f;
    }
}