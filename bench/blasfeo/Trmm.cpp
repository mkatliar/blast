#include <blasfeo/Blasfeo.hpp>

#include <benchmark/benchmark.h>

#include <random>
#include <memory>


#define ADD_BM_TRMM(m, n, p) BENCHMARK_CAPTURE(BM_trmm, m##x##n##x##p##_blasfeo, m, n, p)


namespace blasfeo :: benchmark
{
    template <typename MT>
    static void randomize(blasfeo::Matrix<MT>& A)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-1.0, 1.0);	

        for (size_t i = 0; i < rows(~A); ++i)
            for (size_t j = 0; j < columns(~A); ++j)
                (~A)(i, j) = dis(gen);
    }


    static void BM_trmm(::benchmark::State& state, size_t m, size_t n, size_t p)
    {
        blasfeo::DynamicMatrix<double> A(m, n), B(n, p), C(m, p);

        randomize(A);
        randomize(B);
        
        for (auto _ : state)
            trmm_rlnn(n /*m*/, p /*n*/, 1., A, 0, 0, B, 0, 0, C, 0, 0);
    }

    
    ADD_BM_TRMM(2, 2, 2);
    ADD_BM_TRMM(3, 3, 3);
    ADD_BM_TRMM(5, 5, 5);
    ADD_BM_TRMM(10, 10, 10);
    ADD_BM_TRMM(20, 20, 20);
    ADD_BM_TRMM(30, 30, 30);
}
