#include <blazefeo/Blaze.hpp>

#include <benchmark/benchmark.h>


namespace blazefeo :: benchmark
{
    template <typename Real, size_t N, bool SO>
    static void BM_LowerMatrixScalarMultiplyStatic_Submatrix(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, N, N, SO> A;        
        randomize(A);

        for (auto _ : state)
        {
            for (size_t k = 0; k < N; ++k)
            {
                size_t const rs = N - k - 1;
                auto A21 = submatrix(A, k + 1, k, rs, 1);

                A21 *= 1.1;
            }

            ::benchmark::DoNotOptimize(A(N - 1, N - 1));
        }
    }


    template <typename Real, size_t N, bool SO>
    static void BM_LowerMatrixScalarMultiplyStatic_SubmatrixColumn(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, N, N, SO> A;        
        randomize(A);

        for (auto _ : state)
        {
            for (size_t k = 0; k < N; ++k)
            {
                size_t const rs = N - k - 1;
                auto A21 = submatrix(A, k + 1, k, rs, 1);

                column(A21, 0) *= 1.1;
            }

            ::benchmark::DoNotOptimize(A(N - 1, N - 1));
        }
    }


    template <typename Real, size_t N, bool SO>
    static void BM_LowerMatrixScalarMultiplyStatic_ColumnSubvector(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, N, N, SO> A;        
        randomize(A);

        for (auto _ : state)
        {
            for (size_t k = 0; k < N; ++k)
            {
                size_t const rs = N - k - 1;
                auto A_k = column(A, k);

                subvector(A_k, k + 1, rs) *= 1.1;
            }

            ::benchmark::DoNotOptimize(A(N - 1, N - 1));
        }
    }


    template <typename Real, size_t N, bool SO>
    static void BM_LowerMatrixScalarMultiplyStatic_Loop(::benchmark::State& state)
    {
        blaze::StaticMatrix<Real, N, N, SO> A;
        randomize(A);

        for (auto _ : state)
        {
            for (size_t k = 0; k < N; ++k)
            {
                size_t const rs = N - k - 1;
                auto A21 = submatrix(A, k + 1, k, rs, 1);

                for (size_t i = 0; i < rs; ++i)
                    A21(i, 0) *= 1.1;
            }

            ::benchmark::DoNotOptimize(A(N - 1, N - 1));
        }
    }
    
    
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_Submatrix, double, 5, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_SubmatrixColumn, double, 5, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_ColumnSubvector, double, 5, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_Loop, double, 5, blaze::columnMajor);

    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_Submatrix, double, 50, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_SubmatrixColumn, double, 50, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_ColumnSubvector, double, 50, blaze::columnMajor);
    BENCHMARK_TEMPLATE(BM_LowerMatrixScalarMultiplyStatic_Loop, double, 50, blaze::columnMajor);
}
