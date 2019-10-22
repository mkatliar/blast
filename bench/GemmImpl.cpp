#include <smoke/Gemm.hpp>
#include <smoke/GemmKernel_double_1_1_4.hpp>
#include <smoke/GemmKernel_double_2_1_4.hpp>
#include <smoke/GemmKernel_double_3_1_4.hpp>

#include <bench/Benchmark.hpp>

#include <random>
#include <memory>

#define INSTANTIATE_GEMM_TN(KM, KN, M, P) \
    template void gemm_tn_impl<KM, KN, double, M, M, M, P>(\
    StaticMatrix<double, M, M, P> const& A, StaticMatrix<double, M, M, P> const& B, \
    StaticMatrix<double, M, M, P> const& C, StaticMatrix<double, M, M, P>& D)


#define INSTANTIATE_GEMM_NN(KM, KN, M, P) \
    template void gemm_nn_impl<KM, KN, double, M, M, M, P>(\
    StaticMatrix<double, M, M, P> const& A, StaticMatrix<double, M, M, P> const& B, \
    StaticMatrix<double, M, M, P> const& C, StaticMatrix<double, M, M, P>& D)


#define INSTANTIATE_GEMM_NT(M, P) \
    template void gemm_nt_impl<double, M, M, M, P>(\
    StaticMatrix<double, M, M, P> const& A, StaticMatrix<double, M, M, P> const& B, \
    StaticMatrix<double, M, M, P> const& C, StaticMatrix<double, M, M, P>& D)


namespace smoke :: benchmark
{
    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_tn_impl(
        StaticMatrix<T, K, M, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        gemm(GemmKernel<T, KM, KN, P, true, false> {}, A, B, C, D);
    }


    template <size_t KM, size_t KN, typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_nn_impl(
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, K, N, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        gemm(GemmKernel<T, KM, KN, P, false, false> {}, A, B, C, D);
    }


    template <typename T, size_t M, size_t N, size_t K, size_t P>
    void gemm_nt_impl(
        StaticMatrix<T, M, K, P> const& A, StaticMatrix<T, N, K, P> const& B, 
        StaticMatrix<T, M, N, P> const& C, StaticMatrix<T, M, N, P>& D)
    {
        gemm_nt(A, B, C, D);
    }


    // INSTANTIATE_GEMM_TN(1, 1, 4, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 8, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 12, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 16, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 20, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 24, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 28, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 32, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 36, 4);
    // INSTANTIATE_GEMM_TN(1, 1, 40, 4);

    // INSTANTIATE_GEMM_TN(2, 1, 8, 4);
    // INSTANTIATE_GEMM_TN(2, 1, 16, 4);
    // INSTANTIATE_GEMM_TN(2, 1, 24, 4);
    // INSTANTIATE_GEMM_TN(2, 1, 32, 4);
    // INSTANTIATE_GEMM_TN(2, 1, 40, 4);

    // INSTANTIATE_GEMM_NN(1, 1, 4, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 8, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 12, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 16, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 20, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 24, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 28, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 32, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 36, 4);
    // INSTANTIATE_GEMM_NN(1, 1, 40, 4);

    INSTANTIATE_GEMM_NT(1, 4);
    INSTANTIATE_GEMM_NT(2, 4);
    INSTANTIATE_GEMM_NT(3, 4);
    INSTANTIATE_GEMM_NT(4, 4);
    INSTANTIATE_GEMM_NT(5, 4);
    INSTANTIATE_GEMM_NT(6, 4);
    INSTANTIATE_GEMM_NT(7, 4);
    INSTANTIATE_GEMM_NT(8, 4);
    INSTANTIATE_GEMM_NT(9, 4);
    INSTANTIATE_GEMM_NT(10, 4);
    INSTANTIATE_GEMM_NT(11, 4);
    INSTANTIATE_GEMM_NT(12, 4);
    INSTANTIATE_GEMM_NT(13, 4);
    INSTANTIATE_GEMM_NT(14, 4);
    INSTANTIATE_GEMM_NT(15, 4);
    INSTANTIATE_GEMM_NT(16, 4);
    INSTANTIATE_GEMM_NT(17, 4);
    INSTANTIATE_GEMM_NT(18, 4);
    INSTANTIATE_GEMM_NT(19, 4);
    INSTANTIATE_GEMM_NT(20, 4);
    INSTANTIATE_GEMM_NT(21, 4);
    INSTANTIATE_GEMM_NT(22, 4);
    INSTANTIATE_GEMM_NT(23, 4);
    INSTANTIATE_GEMM_NT(24, 4);
    INSTANTIATE_GEMM_NT(25, 4);
    INSTANTIATE_GEMM_NT(26, 4);
    INSTANTIATE_GEMM_NT(27, 4);
    INSTANTIATE_GEMM_NT(28, 4);
    INSTANTIATE_GEMM_NT(29, 4);
    INSTANTIATE_GEMM_NT(30, 4);
    INSTANTIATE_GEMM_NT(31, 4);
    INSTANTIATE_GEMM_NT(32, 4);
    INSTANTIATE_GEMM_NT(33, 4);
    INSTANTIATE_GEMM_NT(34, 4);
    INSTANTIATE_GEMM_NT(35, 4);
    INSTANTIATE_GEMM_NT(36, 4);
    INSTANTIATE_GEMM_NT(37, 4);
    INSTANTIATE_GEMM_NT(38, 4);
    INSTANTIATE_GEMM_NT(39, 4);
    INSTANTIATE_GEMM_NT(40, 4);
}