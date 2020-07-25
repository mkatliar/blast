#pragma once

#include <blasfeo_d_blasfeo_api.h>
#include <blasfeo_s_blasfeo_api.h>

#include <blasfeo/SizeT.hpp>


namespace blasfeo
{
    /// @brief D <= beta * C + alpha * A * B^T ; C, D lower triangular
    inline void syrk_ln(size_t m, size_t k, double alpha,
        blasfeo_dmat const& sA, size_t ai, size_t aj,
        blasfeo_dmat const& sB, size_t bi, size_t bj,
        double beta, blasfeo_dmat const& sC, size_t ci, size_t cj,
        blasfeo_dmat& sD, size_t di, size_t dj)
    {
        ::blasfeo_dsyrk_ln(m, k, alpha,
            const_cast<blasfeo_dmat *>(&sA), ai, aj,
            const_cast<blasfeo_dmat *>(&sB), bi, bj, beta,
            const_cast<blasfeo_dmat *>(&sC), ci, cj,
            &sD, di, dj);
    }


    /// @brief D <= beta * C + alpha * A * B^T ; C, D lower triangular
    inline void syrk_ln(size_t m, size_t k, double alpha,
        blasfeo_smat const& sA, size_t ai, size_t aj,
        blasfeo_smat const& sB, size_t bi, size_t bj,
        double beta, blasfeo_smat const& sC, size_t ci, size_t cj,
        blasfeo_smat& sD, size_t di, size_t dj)
    {
        ::blasfeo_ssyrk_ln(m, k, alpha,
            const_cast<blasfeo_smat *>(&sA), ai, aj,
            const_cast<blasfeo_smat *>(&sB), bi, bj, beta,
            const_cast<blasfeo_smat *>(&sC), ci, cj,
            &sD, di, dj);
    }
}