#include <blazefeo/math/DynamicPanelMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t SS>
    static void BM_RegisterMatrix_store(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SS>;
        using Traits = RegisterMatrixTraits<Kernel>;

        Kernel ker;
        
        DynamicPanelMatrix<double, rowMajor> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
        randomize(c);

        load(ker, c.tile(0, 0), c.spacing());

        for (auto _ : state)
        {
            store(ker, d.tile(0, 0), d.spacing());
            DoNotOptimize(d);
        }

        state.counters["flops"] = Counter(ker.rows() * ker.columns(), Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, double, 1, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, double, 2, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, double, 3, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_store, double, 2, 8, 4);
}