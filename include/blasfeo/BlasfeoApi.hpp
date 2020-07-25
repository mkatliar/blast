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


    /// @brief returns the memory size (in bytes) needed for a BLASFEO matrix data with elements of type Real
    template <typename Real>
    size_t memsize_mat(size_t m, size_t n);


    /// @brief returns the memory size (in bytes) needed for a double-precision BLASFEO matrix data
    template <>
    inline size_t memsize_mat<double>(size_t m, size_t n)
    {
        return blasfeo_memsize_dmat(m, n);
    }


    /// @brief returns the memory size (in bytes) needed for a double-precision BLASFEO matrix data
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


    /// @brief Pack the column-major double-precision matrix A into BLASFEO matrix B
    inline void pack_mat(size_t m, size_t n, double const * A, size_t lda, blasfeo_dmat& sB, size_t bi, size_t bj)
    {
        blasfeo_pack_dmat(m, n, const_cast<double *>(A), lda, &sB, bi, bj);
    }


    /// @brief Pack the column-major single-precision matrix A into BLASFEO matrix B
    inline void pack_mat(size_t m, size_t n, float const * A, size_t lda, blasfeo_smat& sB, size_t bi, size_t bj)
    {
        blasfeo_pack_smat(m, n, const_cast<float *>(A), lda, &sB, bi, bj);
    }


    /// @brief Transpose and pack the column-major matrix A into the matrix struct B
    inline void pack_tran_mat(size_t m, size_t n, double const * A, size_t lda, blasfeo_dmat& sB, size_t bi, size_t bj)
    {
        ::blasfeo_pack_tran_dmat(m, n, const_cast<double *>(A), lda, &sB, bi, bj);
    }


    /// @brief Transpose and pack the column-major matrix A into the matrix struct B
    inline void pack_tran_mat(size_t m, size_t n, float const * A, size_t lda, blasfeo_smat& sB, size_t bi, size_t bj)
    {
        ::blasfeo_pack_tran_smat(m, n, const_cast<float *>(A), lda, &sB, bi, bj);
    }


    /// @brief Unpack the column-major double-precision BLASFEO matrix A into the column-major matrix B
    inline void unpack_mat(size_t m, size_t n, blasfeo_dmat& sA, size_t ai, size_t aj, double * B, size_t ldb)
    {
        blasfeo_unpack_dmat(m, n, &sA, ai, aj, B, ldb);
    }

    
    /// @brief Unpack the column-major single-precision BLASFEO matrix A into the column-major matrix B
    inline void unpack_mat(size_t m, size_t n, blasfeo_smat& sA, size_t ai, size_t aj, float * B, size_t ldb)
    {
        blasfeo_unpack_smat(m, n, &sA, ai, aj, B, ldb);
    }


    /// @brief Transpose and unpack the matrix structure A into the column-major matrix B
    inline void unpack_tran_mat(size_t m, size_t n, blasfeo_dmat const& sA, size_t ai, size_t aj, double * B, size_t ldb)
    {
        ::blasfeo_unpack_tran_dmat(m, n, const_cast<blasfeo_dmat *>(&sA), ai, aj, B, ldb);
    }


    /// @brief Transpose and unpack the matrix structure A into the column-major matrix B
    inline void unpack_tran_mat(size_t m, size_t n, blasfeo_smat const& sA, size_t ai, size_t aj, float * B, size_t ldb)
    {
        ::blasfeo_unpack_tran_smat(m, n, const_cast<blasfeo_smat *>(&sA), ai, aj, B, ldb);
    }


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