#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, bool SO>
    static void BM_RegisterMatrix_potrf(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SO>;
        using Traits = RegisterMatrixTraits<Kernel>;
        size_t constexpr m = Traits::rows;
        size_t constexpr n = Traits::columns;
        
        DynamicPanelMatrix<T> a(m, n);
        randomize(a);

        Kernel ker;
        load(ker, a.ptr(0, 0), a.spacing());

        for (auto _ : state)
        {
            ker.potrf();
            DoNotOptimize(ker);
        }

        if (m >= n)
            setCounters(state.counters, complexityPotrf(m, n));
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 4, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 12, 4, columnMajor);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 8, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 16, 4, columnMajor);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 24, 4, columnMajor);
}