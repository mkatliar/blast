#pragma once

#include <smoke/SizeT.hpp>
#include <smoke/Panel.hpp>
#include <smoke/GemmKernel.hpp>


namespace smoke
{
    template <typename T, size_t M, size_t N, size_t P = panelSize>
    class StaticMatrix
    {
    public:
        Panel<T, P> load(size_t i, size_t j) const
        {
            return Panel<T, P>(block(i, j));
        }


        void store(size_t i, size_t j, Panel<T, P> const& panel)
        {
            panel.store(block(i, j));
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


        size_t constexpr spacing() const
        {
            return panelColumns_ * elementsPerPanel_;
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


        T * block(size_t i, size_t j)
        {
            return v_ + (i * panelColumns_ + j) * elementsPerPanel_;
        }


        T const * block(size_t i, size_t j) const
        {
            return v_ + (i * panelColumns_ + j) * elementsPerPanel_;
        }


    private:
        static size_t constexpr panelRows_ = M / P + (M % P > 0);
        static size_t constexpr panelColumns_ = N / P + (N % P > 0);
        static size_t constexpr elementsPerPanel_ = P * P;

        alignas(Panel<T, P>::alignment) T v_[panelRows_ * panelColumns_ * elementsPerPanel_];


        size_t elementIndex(size_t i, size_t j) const
        {
            size_t const panel_i = i / P;
            size_t const panel_j = j / P;
            size_t const subpanel_i = i % P;
            size_t const subpanel_j = j % P;

            return (panel_i * panelColumns_ + panel_j) * elementsPerPanel_ + subpanel_i + subpanel_j * P;
        }
    };


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_nt(
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, N, K, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
            {
                GemmKernel<T, KM, KN, P> ker;
                ker.load(C.block(KM * i, KN * j), C.spacing());

                for (size_t k = 0; k < KK; ++k)
                    ker(A.block(KM * i, k), false, B.block(KN * j, k), true);

                ker.store(D.block(KM * i, KN * j), D.spacing());
            }
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_tn(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
            {
                GemmKernel<T, KM, KN, P> ker;
                ker.load(C.block(KM * i, KN * j), C.spacing());

                for (size_t k = 0; k < KK; ++k)
                    ker(A.block(k, KM * i), true, B.block(k, KN * j), false);

                ker.store(D.block(i, j), D.spacing());
            }
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    inline void gemm_nn(
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        static_assert(M % (KM * P) == 0);
        static_assert(N % (KN * P) == 0);
        static_assert(K % P == 0);

        size_t const MM = M / (KM * P);
        size_t const NN = N / (KN * P);
        size_t const KK = K / P;

        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
            {
                GemmKernel<T, KM, KN, P> ker;
                ker.load(C.block(KM * i, KN * j), C.spacing());

                for (size_t k = 0; k < KK; ++k)
                    ker(A.block(KM * i, k), false, B.block(k, KN * j), false);

                ker.store(D.block(i, j), D.spacing());
            }

        /*
        for (size_t i = 0; i < MM; ++i)
            for (size_t j = 0; j < NN; ++j)
                D.store(i, j, C.load(i, j));

        for (size_t i = 0; i < MM; ++i)
            for (size_t k = 0; k < KK; ++k)
            {
                Panel<T, P> A_ik = A.load(i, k);

                for (size_t j = 0; j < NN; ++j)
                {
                    Panel<T, P> p = D.load(i, j);                
                    gemm(A_ik, false, B.load(k, j), false, p);
                    D.store(i, j, p);
                }
            }
        */
    }


    // template <typename T>
    // inline void gemm_tn(
    //     StaticMatrix<T, 8, 8, 4> const& A, StaticMatrix<T, 8, 8, 4> const& B, 
    //     StaticMatrix<T, 8, 8, 4> const& C, StaticMatrix<T, 8, 8, 4>& D)
    // {
    //     size_t const MM = 2;
    //     size_t const NN = 2;
    //     size_t const KK = 2;

    //     Panel<T, 4> p;
        
    //     p = C.load(0, 0);
    //     gemm(A.load(0, 0), true, B.load(0, 0), false, p);
    //     gemm(A.load(1, 0), true, B.load(1, 0), false, p);
    //     D.store(0, 0, p);

    //     p = C.load(0, 1);
    //     gemm(A.load(0, 0), true, B.load(0, 1), false, p);
    //     gemm(A.load(1, 0), true, B.load(1, 1), false, p);
    //     D.store(0, 1, p);

    //     p = C.load(1, 0);
    //     gemm(A.load(0, 1), true, B.load(0, 0), false, p);
    //     gemm(A.load(1, 1), true, B.load(1, 0), false, p);
    //     D.store(1, 0, p);

    //     p = C.load(1, 1);
    //     gemm(A.load(0, 1), true, B.load(0, 1), false, p);
    //     gemm(A.load(1, 1), true, B.load(1, 1), false, p);
    //     D.store(1, 1, p);
    // }
}