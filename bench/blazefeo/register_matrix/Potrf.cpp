#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>
#include <bench/Complexity.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t P>
    static void BM_RegisterMatrix_potrf(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, P>;
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


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 4, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 8, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, double, 12, 4, 4);

    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 8, 4, 8);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 16, 4, 8);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_potrf, float, 24, 4, 8);
}