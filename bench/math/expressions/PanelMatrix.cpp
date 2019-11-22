#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/StaticPanelMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <blaze/Math.h>


namespace blazefeo :: benchmark
{
    template <typename Real, bool SO1, bool SO2>
    static void BM_assign_Dense_Panel(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        DynamicPanelMatrix<Real, SO2> rhs(m, m);
        randomize(rhs);

        DynamicMatrix<Real, SO1> lhs(m, m);

        for (auto _ : state)
        {
            lhs = rhs;
            DoNotOptimize(rhs);
            DoNotOptimize(lhs);
        }

        state.counters["flops"] = Counter(m * m, Counter::kIsIterationInvariantRate);;
        state.counters["m"] = m;
    }


    template <typename Real, size_t M, bool SO1, bool SO2>
    static void BM_assign_Dense_Panel_static(::benchmark::State& state)
    {
        StaticPanelMatrix<Real, M, M, SO2> rhs;
        randomize(rhs);

        StaticMatrix<Real, M, M, SO1> lhs;

        for (auto _ : state)
        {
            lhs = rhs;
            DoNotOptimize(rhs);
            DoNotOptimize(lhs);
        }

        state.counters["flops"] = Counter(M * M, Counter::kIsIterationInvariantRate);;
        state.counters["m"] = M;
    }


    template <typename Real, bool SO1, bool SO2>
    static void BM_assign_Panel_Dense(::benchmark::State& state)
    {
        size_t const m = state.range(0);

        DynamicMatrix<Real, SO2> rhs(m, m);
        randomize(rhs);

        DynamicPanelMatrix<Real, SO1> lhs(m, m);

        for (auto _ : state)
        {
            lhs = rhs;
            DoNotOptimize(rhs);
            DoNotOptimize(lhs);
        }

        state.counters["flops"] = Counter(m * m, Counter::kIsIterationInvariantRate);;
        state.counters["m"] = m;
    }


    template <typename Real, size_t M, bool SO1, bool SO2>
    static void BM_assign_Panel_Dense_static(::benchmark::State& state)
    {
        StaticMatrix<Real, M, M, SO2> rhs;
        randomize(rhs);

        StaticPanelMatrix<Real, M, M, SO1> lhs;

        for (auto _ : state)
        {
            lhs = rhs;
            DoNotOptimize(rhs);
            DoNotOptimize(lhs);
        }

        state.counters["flops"] = Counter(M * M, Counter::kIsIterationInvariantRate);;
        state.counters["m"] = M;
    }


    BENCHMARK_TEMPLATE(BM_assign_Dense_Panel, double, columnMajor, columnMajor)->DenseRange(1, 300);
    BENCHMARK_TEMPLATE(BM_assign_Dense_Panel, float, columnMajor, columnMajor)->DenseRange(1, 300);

    BENCHMARK_TEMPLATE(BM_assign_Panel_Dense, double, columnMajor, columnMajor)->DenseRange(1, 300);
    BENCHMARK_TEMPLATE(BM_assign_Panel_Dense, float, columnMajor, columnMajor)->DenseRange(1, 300);

    BENCHMARK_TEMPLATE(BM_assign_Dense_Panel_static, double, 35, columnMajor, columnMajor);
    BENCHMARK_TEMPLATE(BM_assign_Panel_Dense_static, double, 35, columnMajor, columnMajor);
}
