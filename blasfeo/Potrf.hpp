#pragma once

#include <blasfeo_d_blasfeo_api.h>
#include <blasfeo_s_blasfeo_api.h>


namespace blasfeo
{
    /// @brief // D <= chol( C ) ; C, D lower triangular
    inline void potrf(size_t m,
        blasfeo_dmat const& sC, size_t ci, size_t cj, 
        blasfeo_dmat const& sD, size_t di, size_t dj)
    {
        blasfeo_dpotrf_l(m,
            const_cast<blasfeo_dmat *>(&sC), ci, cj,
            const_cast<blasfeo_dmat *>(&sD), di, dj);
    }


    /// @brief // D <= chol( C ) ; C, D lower triangular
    inline void potrf(size_t m,
        blasfeo_smat const& sC, size_t ci, size_t cj, 
        blasfeo_smat const& sD, size_t di, size_t dj)
    {
        blasfeo_spotrf_l(m,
            const_cast<blasfeo_smat *>(&sC), ci, cj,
            const_cast<blasfeo_smat *>(&sD), di, dj);
    }
}