/// @brief BLASFEO API backend
///

#pragma once

#include <blasfeo_d_aux.h>
#include <blasfeo_s_aux.h>

#include <blasfeo/SizeT.hpp>


namespace blasfeo
{
    /// \brief Number of rows
    inline size_t rows(blasfeo_dmat const& m)
    {
        return m.m;
    }


    /// \brief Number of rows
    inline size_t rows(blasfeo_smat const& m)
    {
        return m.m;
    }


    /// \brief Number of columns
    inline size_t columns(blasfeo_dmat const& m)
    {
        return m.n;
    }


    /// \brief Number of columns
    inline size_t columns(blasfeo_smat const& m)
    {
        return m.n;
    }


    // @brief returns the memory size (in bytes) needed for a BLASFEO matrix data with elements of type Real
    template <typename Real>
    size_t memsize_mat(size_t m, size_t n);


    // @brief returns the memory size (in bytes) needed for a double-precision BLASFEO matrix data
    template <>
    inline size_t memsize_mat<double>(size_t m, size_t n)
    {
        return blasfeo_memsize_dmat(m, n);
    }


    // @brief returns the memory size (in bytes) needed for a double-precision BLASFEO matrix data
    template <>
    inline size_t memsize_mat<float>(size_t m, size_t n)
    {
        return blasfeo_memsize_dmat(m, n);
    }


    /// @brief Create double-precision BLASFEO matrix
    inline void create_mat(size_t m, size_t n, blasfeo_dmat& sA, void * memory)
    {
        blasfeo_create_dmat(m, n, &sA, memory);
    }


    /// @brief Create single-precision BLASFEO matrix
    inline void create_mat(size_t m, size_t n, blasfeo_smat& sA, void * memory)
    {
        blasfeo_create_smat(m, n, &sA, memory);
    }


    /// \brief Matrix element access
    inline double& element(blasfeo_dmat& m, size_t i, size_t j)
    {
        return BLASFEO_DMATEL(&m, i, j);
    }


    /// \brief Matrix element access
    inline float& element(blasfeo_smat& m, size_t i, size_t j)
    {
        return BLASFEO_SMATEL(&m, i, j);
    }


    /// \brief Const matrix element access
    inline double const& element(blasfeo_dmat const& m, size_t i, size_t j)
    {
        return BLASFEO_DMATEL(&m, i, j);
    }


    /// \brief Const matrix element access
    inline float const& element(blasfeo_smat const& m, size_t i, size_t j)
    {
        return BLASFEO_SMATEL(&m, i, j);
    }


    // @brief Pack the column-major double-precision matrix A into BLASFEO matrix B
    inline void pack_mat(size_t m, size_t n, double const * A, size_t lda, blasfeo_dmat& sB, size_t bi, size_t bj)
    {
        blasfeo_pack_dmat(m, n, const_cast<double *>(A), lda, &sB, bi, bj);
    }


    // @brief Pack the column-major single-precision matrix A into BLASFEO matrix B
    inline void pack_mat(size_t m, size_t n, float const * A, size_t lda, blasfeo_smat& sB, size_t bi, size_t bj)
    {
        blasfeo_pack_smat(m, n, const_cast<float *>(A), lda, &sB, bi, bj);
    }


    // @brief Unpack the column-major double-precision BLASFEO matrix A into the column-major matrix B
    inline void unpack_mat(size_t m, size_t n, blasfeo_dmat& sA, size_t ai, size_t aj, double * B, size_t ldb)
    {
        blasfeo_unpack_dmat(m, n, &sA, ai, aj, B, ldb);
    }

    
    // @brief Unpack the column-major single-precision BLASFEO matrix A into the column-major matrix B
    inline void unpack_mat(size_t m, size_t n, blasfeo_smat& sA, size_t ai, size_t aj, float * B, size_t ldb)
    {
        blasfeo_unpack_smat(m, n, &sA, ai, aj, B, ldb);
    }
}