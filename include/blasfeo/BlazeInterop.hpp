/// @brief Interoperability with Blaze
#pragma once

#include <blasfeo/BlasfeoApi.hpp>
#include <blazefeo/Blaze.hpp>

#include <type_traits>


namespace blasfeo
{
    template <typename T>
    struct IsBlasfeoMat
    {
        static bool constexpr value = 
            std::is_base_of_v<::blasfeo_dmat, T> ||
            std::is_base_of_v<::blasfeo_smat, T>;
    };


    template <typename T>
    inline bool constexpr IsBlasfeoMat_v = IsBlasfeoMat<T>::value;


    template <typename T>
    struct IsBlasfeoVec
    {
        static bool constexpr value = 
            std::is_base_of_v<::blasfeo_dvec, T> ||
            std::is_base_of_v<::blasfeo_svec, T>;
    };


    template <typename T>
    inline bool constexpr IsBlasfeoVec_v = IsBlasfeoVec<T>::value;


    // @brief Pack the matrix A into BLASFEO matrix B
    template <typename MT, bool SO, typename M>
    inline std::enable_if_t<IsBlasfeoMat_v<M>>
        pack(blaze::DenseMatrix<MT, SO> const& A, M& sB, size_t bi, size_t bj)
    {
        if constexpr (SO == blaze::columnMajor)
            pack_mat(rows(A), columns(A), data(A), spacing(A), sB, bi, bj);
        else
            pack_tran_mat(columns(A), rows(A), data(A), spacing(A), sB, bi, bj);
    }


    // // @brief Transpose and pack the matrix A into the BLASFEO matrix B
    // template <typename MT, bool SO, typename M>
    // inline std::enable_if_t<IsBlasfeoMat_v<M>>
    //     pack_tran_mat(blaze::DenseMatrix<MT, SO> const& A, M& sB, size_t bi, size_t bj)
    // {
    //     using Real = blaze::ElementType_t<MT>;

    //     if constexpr (SO == blaze::columnMajor)
    //         pack_tran_mat(rows(A), columns(A), const_cast<Real *>(A.data()), spacing(A), &sB, bi, bj);
    //     else
    //         pack_mat(rows(A), columns(A), const_cast<Real *>(A.data()), spacing(A), &sB, bi, bj);
    // }


    // @brief Pack the vector a into BLASFEO matrix B
    template <typename VT, bool TF, typename M>
    inline std::enable_if_t<IsBlasfeoMat_v<M>>
        pack(blaze::DenseVector<VT, TF> const& a, M& sB, size_t bi, size_t bj)
    {
        if constexpr (TF == blaze::columnVector)
            pack_mat(size(a), 1, data(a), size(a), sB, bi, bj);
        else
            pack_tran_mat(size(a), 1, data(a), size(a), sB, bi, bj);
    }


    // @brief Pack the vector x into the BLASFEO vector y
    template <typename VT, bool TF, typename V>
    inline std::enable_if_t<IsBlasfeoVec_v<V>>
        pack(blaze::DenseVector<VT, TF> const& x, V& sy, size_t yi)
    {
        pack_vec(size(x), data(x), sy, yi);
    }


    /// @brief Unpack the column-major double-precision BLASFEO matrix A into the matrix B
    template <typename M, typename MT, bool SO>
    inline std::enable_if_t<IsBlasfeoMat_v<M>>
        unpack(size_t m, size_t n, M& sA, size_t ai, size_t aj, blaze::DenseMatrix<MT, SO>& B)
    {
        resize(B, m, n);

        if constexpr (SO == blaze::columnMajor)
            unpack_mat(m, n, sA, ai, aj, data(B), spacing(B));
        else
            unpack_tran_mat(m, n, sA, ai, aj, data(B), spacing(B));
    }


    /// @brief Unpack and transpose the column-major double-precision BLASFEO matrix A into the matrix B
    template <typename M, typename MT, bool SO>
    inline std::enable_if_t<IsBlasfeoMat_v<M>>
        unpack_tran(size_t m, size_t n, M& sA, size_t ai, size_t aj, blaze::DenseMatrix<MT, SO>& B)
    {
        resize(B, n, m);

        if constexpr (SO == blaze::columnMajor)
            unpack_tran_mat(m, n, sA, ai, aj, data(B), spacing(B));
        else
            unpack_mat(m, n, sA, ai, aj, data(B), spacing(B));
    }


    /// @brief Unpack the vector structure x into the vector y
    template <typename V, typename VT, bool TF>
    inline std::enable_if_t<IsBlasfeoVec_v<V>>
        unpack(size_t m, blasfeo_dvec const& sx, size_t xi, blaze::DenseVector<VT, TF>& y)
    {
        resize(y, m);
        unpack_vec(m, sx, xi, data(y));
    }
}