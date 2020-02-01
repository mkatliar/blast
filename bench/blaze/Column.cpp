#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t N, bool CL, bool CR>
    static void BM_column(::benchmark::State& state)
    {
        blaze::StaticMatrix<double, N, N, blaze::columnMajor> A;
        randomize(A);

        for (auto _ : state)
        {
            size_t const k = N / 2;
            size_t const rs = N - k;

            auto D21 = submatrix(A, k, k, rs, 1, blaze::checked);
            auto const D20 = submatrix(A, k, 0, rs, k, blaze::checked);

            for (size_t j = 0; j < k; ++j)
                column(D21, 0, blaze::Check<CL> {}) -= (~A)(k, j) * column(D20, j, blaze::Check<CR> {});

            ::benchmark::DoNotOptimize(A(N - 1, N - 1));
        }
    }


    BENCHMARK_TEMPLATE(BM_column, double, 60, false, false);
    BENCHMARK_TEMPLATE(BM_column, double, 60, false, true);
    BENCHMARK_TEMPLATE(BM_column, double, 60, true, false);
    BENCHMARK_TEMPLATE(BM_column, double, 60, true, true);
}
