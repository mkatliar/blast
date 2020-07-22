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