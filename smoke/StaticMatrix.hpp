#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Panel.hpp>


namespace smoke
{
    template <typename T, size_t M, size_t N, size_t P = panelSize>
    class StaticMatrix
    {
    public:
        Panel<T, P> load(size_t i, size_t j) const
        {
            return Panel<T, P>(panelPtr(i, j));
        }


        void store(size_t i, size_t j, Panel<T, P> const& panel)
        {
            panel.store(panelPtr(i, j));
        }


        T operator()(size_t i, size_t j) const
        {
            return v_[elementIndex(i, j)];
        }


        T& operator()(size_t i, size_t j)
        {
            return v_[elementIndex(i, j)];
        }


        size_t constexpr rows() const
        {
            return M;
        }


        size_t constexpr columns() const
        {
            return N;
        }


        size_t constexpr panelRows() const
        {
            return panelRows_;
        }


        size_t constexpr panelColumns() const
        {
            return panelColumns_;
        }


        void pack(T const * data, size_t lda)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    (*this)(i, j) = data[i + lda * j];
        }


        void unpack(T * data, size_t lda) const
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    data[i + lda * j] = (*this)(i, j);
        }


    private:
        static size_t constexpr panelRows_ = M / P + (M % P > 0);
        static size_t constexpr panelColumns_ = N / P + (N % P > 0);
        static size_t constexpr elementsPerPanel_ = P * P;

        alignas(Panel<T, P>::alignment) T v_[panelRows_ * panelColumns_ * elementsPerPanel_];


        T * panelPtr(size_t i, size_t j)
        {
            return v_ + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        T const * panelPtr(size_t i, size_t j) const
        {
            return v_ + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return (panel_i * panelColumns_ + panel_j) * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };


    template <typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_nt(
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, N, K, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % P == 0);
        static_assert(N % P == 0);
        static_assert(K % P == 0);

        size_t const MM = M / P;
        size_t const NN = N / P;
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
            {
                Panel<T, P> p = C.load(i, j);

                for (size_t k = 0; k < KK; ++k)
                    gemm(A.load(i, k), false, B.load(j, k), true, p);

                D.store(i, j, p);
            }
    }


    template <typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_tn(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % P == 0);
        static_assert(N % P == 0);
        static_assert(K % P == 0);

        size_t const MM = M / P;
        size_t const NN = N / P;
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
            {
                Panel<T, P> p = C.load(i, j);

                for (size_t k = 0; k < KK; ++k)
                    gemm(A.load(k, i), true, B.load(k, j), false, p);

                D.store(i, j, p);
            }
    }
}