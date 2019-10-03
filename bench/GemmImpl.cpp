#include <smoke/StaticMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <random>
#include <memory>

#define INSTANTIATE_GEMM_TN(M, P) \
    template void gemm_tn_impl<double, M, M, M, P>(\
    StaticMatrix<double, M, M, P> const& A, StaticMatrix<double, M, M, P> const& B, \
    StaticMatrix<double, M, M, P> const& C, StaticMatrix<double, M, M, P>& D)


#define INSTANTIATE_GEMM_NN(M, P) \
    template void gemm_nn_impl<double, M, M, M, P>(\
    StaticMatrix<double, M, M, P> const& A, StaticMatrix<double, M, M, P> const& B, \
    StaticMatrix<double, M, M, P> const& C, StaticMatrix<double, M, M, P>& D)


namespace smoke :: benchmark
{
    template <typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_tn_impl(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        gemm_tn(A, B, C, D);
    }


    template <typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_nn_impl(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        gemm_nn(A, B, C, D);
    }


    INSTANTIATE_GEMM_TN(4, 4);
    INSTANTIATE_GEMM_TN(8, 4);
    INSTANTIATE_GEMM_TN(12, 4);
    INSTANTIATE_GEMM_TN(16, 4);
    INSTANTIATE_GEMM_TN(20, 4);
    INSTANTIATE_GEMM_TN(24, 4);
    INSTANTIATE_GEMM_TN(28, 4);
    INSTANTIATE_GEMM_TN(32, 4);
    INSTANTIATE_GEMM_TN(36, 4);
    INSTANTIATE_GEMM_TN(40, 4);

    INSTANTIATE_GEMM_NN(4, 4);
    INSTANTIATE_GEMM_NN(8, 4);
    INSTANTIATE_GEMM_NN(12, 4);
    INSTANTIATE_GEMM_NN(16, 4);
    INSTANTIATE_GEMM_NN(20, 4);
    INSTANTIATE_GEMM_NN(24, 4);
    INSTANTIATE_GEMM_NN(28, 4);
    INSTANTIATE_GEMM_NN(32, 4);
    INSTANTIATE_GEMM_NN(36, 4);
    INSTANTIATE_GEMM_NN(40, 4);
}