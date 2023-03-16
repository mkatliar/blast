// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo_d_aux.h>
#include <blasfeo_s_aux.h>

#include <blasfeo/SizeT.hpp>


namespace blasfeo
{
    /// @brief Pack the vector x into the vector structure y
    inline void pack_vec(size_t m, double const * x, blasfeo_dvec& sy, size_t yi)
    {
        ::blasfeo_pack_dvec(m, const_cast<double *>(x), &sy, yi);
    }


    /// @brief Pack the vector x into the vector structure y
    inline void pack_vec(size_t m, float const * x, blasfeo_svec& sy, size_t yi)
    {
        ::blasfeo_pack_svec(m, const_cast<float *>(x), &sy, yi);
    }


    /// @brief Unpack the vector structure x into the vector y
    inline void unpack_vec(size_t m, blasfeo_dvec const& sx, size_t xi, double * y)
    {
        ::blasfeo_unpack_dvec(m, const_cast<blasfeo_dvec *>(&sx), xi, y);
    }


    /// @brief Unpack the vector structure x into the vector y
    inline void unpack_vec(size_t m, blasfeo_svec const& sx, size_t xi, float * y)
    {
        ::blasfeo_unpack_svec(m, const_cast<blasfeo_svec *>(&sx), xi, y);
    }


    /// @brief a <= alpha
    inline void vecse(size_t m, double alpha, blasfeo_dvec& sx, size_t xi)
    {
        ::blasfeo_dvecse(m, alpha, &sx, xi);
    }


    /// @brief a <= alpha
    inline void vecse(size_t m, float alpha, blasfeo_svec& sx, size_t xi)
    {
        ::blasfeo_svecse(m, alpha, &sx, xi);
    }


    /// @brief x <= alpha*x
    inline void vecsc(size_t m, double alpha, blasfeo_dvec& sx, size_t xi)
    {
        ::blasfeo_dvecsc(m, alpha, &sx, xi);
    }

    /// @brief x <= alpha*x
    inline void vecsc(size_t m, float alpha, blasfeo_svec& sx, size_t xi)
    {
        ::blasfeo_svecsc(m, alpha, &sx, xi);
    }
}