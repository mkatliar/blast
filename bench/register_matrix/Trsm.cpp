#include <blazefeo/math/StaticPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t P>
    static void BM_RegisterMatrix_trsm(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, P>;
        using Traits = RegisterMatrixTraits<Kernel>;
        size_t constexpr m = Traits::rows;
        size_t constexpr n = Traits::columns;
        
        StaticPanelMatrix<double, m, n, rowMajor> L;
        randomize(L);

        StaticPanelMatrix<double, n, n, rowMajor> A;
        randomize(A);

        Kernel ker;
        load(ker, A.tile(0, 0), A.spacing());

        for (auto _ : state)
        {
            trsm<false, false, true>(ker, L.tile(0, 0), spacing(L));
            DoNotOptimize(ker);
        }

        // Algorithmic complexity of triangular substitution: (n^2-n)/2 additions (subtractions)
        // and (n^2-n)/2 multiplications, which results in n^2-n flops per row.
        // https://algowiki-project.org/en/Backward_substitution
        state.counters["flops"] = Counter((n * n - n) * m, Counter::kIsIterationInvariantRate);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 1, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 2, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 3, 4, 4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_trsm, double, 2, 8, 4);
}