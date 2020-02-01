#pragma once

#include <blasfeo_d_blasfeo_api.h>
#include <blasfeo_s_blasfeo_api.h>


namespace blasfeo
{
    /// @brief D <= chol( C + A * B' ) ; C, D lower triangular
    inline void syrk_potrf(size_t m, size_t k, 
        blasfeo_dmat const& sA, size_t ai, size_t aj, 
        blasfeo_dmat const& sB, size_t bi, size_t bj, 
        blasfeo_dmat const& sC, size_t ci, size_t cj, 
        blasfeo_dmat const& sD, size_t di, size_t dj)
    {
        blasfeo_dsyrk_dpotrf_ln(m, k, 
            const_cast<blasfeo_dmat *>(&sA), ai, aj, 
            const_cast<blasfeo_dmat *>(&sB), bi, bj, 
            const_cast<blasfeo_dmat *>(&sC), ci, cj,
            const_cast<blasfeo_dmat *>(&sD), di, dj);
    }


    /// @brief D <= chol( C + A * B' ) ; C, D lower triangular
    inline void syrk_potrf(size_t m, size_t k, 
        blasfeo_smat const& sA, size_t ai, size_t aj, 
        blasfeo_smat const& sB, size_t bi, size_t bj, 
        blasfeo_smat const& sC, size_t ci, size_t cj, 
        blasfeo_smat const& sD, size_t di, size_t dj)
    {
        blasfeo_ssyrk_spotrf_ln(m, k, 
            const_cast<blasfeo_smat *>(&sA), ai, aj, 
            const_cast<blasfeo_smat *>(&sB), bi, bj, 
            const_cast<blasfeo_smat *>(&sC), ci, cj,
            const_cast<blasfeo_smat *>(&sD), di, dj);
    }
}