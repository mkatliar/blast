#include <blazefeo/math/DynamicPanelMatrix.hpp>
#include <blazefeo/math/simd/RegisterMatrix.hpp>

#include <bench/Benchmark.hpp>

#include <test/Randomize.hpp>

#include <functional> 


namespace blazefeo :: benchmark
{
    template <typename T, size_t M, size_t N, size_t SS>
    static void BM_RegisterMatrix_partialStore(State& state)
    {
        using Kernel = RegisterMatrix<T, M, N, SS>;
        size_t const m = state.range(0);
        size_t const n = state.range(1);

        Kernel ker;
        
        DynamicPanelMatrix<double, rowMajor> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
        randomize(c);        

        load(ker, c.tile(0, 0), c.spacing());
        for (auto _ : state)
        {
            store(ker, d.tile(0, 0), d.spacing(), m, n);
            DoNotOptimize(d);   
        }

        state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
        state.counters["store_m"] = m;
        state.counters["store_n"] = n;
        state.counters["size_m"] = ker.rows();
        state.counters["size_n"] = ker.columns();
    }


    // template <typename T, size_t M, size_t N, size_t SS>
    // static void BM_RegisterMatrix_partialStoreAverage(State& state)
    // {
    //     using Kernel = RegisterMatrix<T, M, N, SS>;
    //     Kernel ker;
        
    //     DynamicPanelMatrix<double, rowMajor> c(ker.rows(), ker.columns()), d(ker.rows(), ker.columns());
    //     randomize(c);        

    //     load(ker, c.tile(0, 0), c.spacing());

    //     for (int m = 1; m <= ker.rows(); ++m)
    //         for (int n = 1; n <= ker.columns(); ++n)
    //             for (auto _ : state)
    //             {                    
    //                 store(ker, d.tile(0, 0), d.spacing(), m, n);
    //                 DoNotOptimize(d);
    //             }

    //     state.counters["flops"] = Counter(m * n, Counter::kIsIterationInvariantRate);
    // }


    static void args(internal::Benchmark * b, size_t M, size_t N) 
    {
        for (int i = 1; i <= M; ++i)
            for (int j = 1; j <= N; ++j)
                b->Args({i, j});
    }


    static void args_4_4(internal::Benchmark * b) 
    {
        args(b, 4, 4);
    }


    static void args_8_4(internal::Benchmark * b) 
    {
        args(b, 8, 4);
    }


    static void args_12_4(internal::Benchmark * b) 
    {
        args(b, 12, 4);
    }


    static void args_8_8(internal::Benchmark * b) 
    {
        args(b, 8, 8);
    }


    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 1, 4, 4)->Apply(args_4_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 2, 4, 4)->Apply(args_8_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 3, 4, 4)->Apply(args_12_4);
    BENCHMARK_TEMPLATE(BM_RegisterMatrix_partialStore, double, 2, 8, 4)->Apply(args_8_8);
}