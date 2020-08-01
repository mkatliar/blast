#include <blazefeo/math/dense/StaticMatrixPointer.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_trmm(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        size_t constexpr K = 100;
        
        StaticMatrix<T, Kernel::rows(), Kernel::rows(), SO> A;
        StaticMatrix<T, Kernel::rows(), Kernel::columns(), columnMajor> B;

        randomize(A);
        randomize(B);

        Kernel ker;
        
        for (auto _ : state)
        {
            ker.template trmm<Side::Left, UpLo::Upper>(
                columns(A), T(1.), ptr(A, 0, 0), ptr(B, 0, 0));

            DoNotOptimize(ker);
        }

        state.counters["flops"] = Counter(M * N * (M + 1), Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, double, 4, 8, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, double, 12, 4, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, float, 24, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, float, 16, 5, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trmm, float, 16, 6, columnMajor);
}