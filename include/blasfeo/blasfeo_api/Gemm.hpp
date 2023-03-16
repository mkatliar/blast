// Copyright (c) 2019-2020 Mikhail Katliar All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <blasfeo_d_aux.h>
#include <blasfeo_s_aux.h>
#include <blasfeo_d_blasfeo_api.h>
#include <blasfeo_s_blasfeo_api.h>

#include <blasfeo/SizeT.hpp>


namespace blasfeo
{
    /// @brief D <= beta * C + alpha * A^T * B
    inline void gemm_tn(size_t m, size_t n, size_t k,
        double alpha,
        blasfeo_dmat const& sA, size_t ai, size_t aj,
        blasfeo_dmat const& sB, size_t bi, size_t bj,
        double beta,
        blasfeo_dmat const& sC, size_t ci, size_t cj,
        blasfeo_dmat& sD, size_t di, size_t dj)
    {
        blasfeo_dgemm_tn(m, n, k, 
            alpha, 
            const_cast<blasfeo_dmat *>(&sA), ai, aj, 
            const_cast<blasfeo_dmat *>(&sB), bi, bj, 
            beta, 
            const_cast<blasfeo_dmat *>(&sC), ci, cj, 
            &sD, di, dj);
    }


    /// @brief D <= beta * C + alpha * A^T * B
    inline void gemm_tn(size_t m, size_t n, size_t k,
        float alpha,
        blasfeo_smat const& sA, size_t ai, size_t aj,
        blasfeo_smat const& sB, size_t bi, size_t bj,
        float beta,
        blasfeo_smat const& sC, size_t ci, size_t cj,
        blasfeo_smat& sD, size_t di, size_t dj)
    {
        blasfeo_sgemm_tn(m, n, k, 
            alpha, 
            const_cast<blasfeo_smat *>(&sA), ai, aj, 
            const_cast<blasfeo_smat *>(&sB), bi, bj, 
            beta, 
            const_cast<blasfeo_smat *>(&sC), ci, cj, 
            &sD, di, dj);
    }


    /// @brief D <= beta * C + alpha * A * B
    inline void gemm_nn(size_t m, size_t n, size_t k,
        double alpha,
        blasfeo_dmat const& sA, size_t ai, size_t aj,
        blasfeo_dmat const& sB, size_t bi, size_t bj,
        double beta,
        blasfeo_dmat const& sC, size_t ci, size_t cj,
        blasfeo_dmat& sD, size_t di, size_t dj)
    {
        blasfeo_dgemm_nn(m, n, k, 
            alpha, 
            const_cast<blasfeo_dmat *>(&sA), ai, aj, 
            const_cast<blasfeo_dmat *>(&sB), bi, bj, 
            beta, 
            const_cast<blasfeo_dmat *>(&sC), ci, cj, 
            &sD, di, dj);
    }


    /// @brief D <= beta * C + alpha * A * B
    inline void gemm_nn(size_t m, size_t n, size_t k,
        float alpha,
        blasfeo_smat const& sA, size_t ai, size_t aj,
        blasfeo_smat const& sB, size_t bi, size_t bj,
        float beta,
        blasfeo_smat const& sC, size_t ci, size_t cj,
        blasfeo_smat& sD, size_t di, size_t dj)
    {
        blasfeo_sgemm_nn(m, n, k, 
            alpha, 
            const_cast<blasfeo_smat *>(&sA), ai, aj, 
            const_cast<blasfeo_smat *>(&sB), bi, bj, 
            beta, 
            const_cast<blasfeo_smat *>(&sC), ci, cj, 
            &sD, di, dj);
    }


    /// @brief D <= beta * C + alpha * A * B^T
    inline void gemm_nt(size_t m, size_t n, size_t k,
        double alpha,
        blasfeo_dmat const& sA, size_t ai, size_t aj,
        blasfeo_dmat const& sB, size_t bi, size_t bj,
        double beta,
        blasfeo_dmat const& sC, size_t ci, size_t cj,
        blasfeo_dmat& sD, size_t di, size_t dj)
    {
        blasfeo_dgemm_nt(m, n, k, 
            alpha, 
            const_cast<blasfeo_dmat *>(&sA), ai, aj, 
            const_cast<blasfeo_dmat *>(&sB), bi, bj, 
            beta, 
            const_cast<blasfeo_dmat *>(&sC), ci, cj, 
            &sD, di, dj);
    }


    /// @brief D <= beta * C + alpha * A * B^T
    inline void gemm_nt(size_t m, size_t n, size_t k,
        float alpha,
        blasfeo_smat const& sA, size_t ai, size_t aj,
        blasfeo_smat const& sB, size_t bi, size_t bj,
        float beta,
        blasfeo_smat const& sC, size_t ci, size_t cj,
        blasfeo_smat& sD, size_t di, size_t dj)
    {
        blasfeo_sgemm_nt(m, n, k, 
            alpha, 
            const_cast<blasfeo_smat *>(&sA), ai, aj, 
            const_cast<blasfeo_smat *>(&sB), bi, bj, 
            beta, 
            const_cast<blasfeo_smat *>(&sC), ci, cj, 
            &sD, di, dj);
    }
}
