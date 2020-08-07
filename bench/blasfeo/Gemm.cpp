#include <blasfeo/Blasfeo.hpp>

#include <bench/Gemm.hpp>

#include <random>
#include <memory>


namespace blazefeo :: benchmark
{
    using namespace ::benchmark;

    
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


    template <typename Real>
    static void BM_gemm(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        size_t const n = m;
        size_t const k = m;

        blasfeo::DynamicMatrix<Real> A(k, m), B(k, n), C(m, n);

        randomize(A);
        randomize(B);
        randomize(C);

        /// @brief D <= beta * C + alpha * A^T * B
        // inline void gemm_tn(size_t m, size_t n, size_t k,
        //     double alpha,
        //     blasfeo_dmat const& sA, size_t ai, size_t aj,
        //     blasfeo_dmat const& sB, size_t bi, size_t bj,
        //     double beta,
        //     blasfeo_dmat const& sC, size_t ci, size_t cj,
        //     blasfeo_dmat& sD, size_t di, size_t dj);
        
        for (auto _ : state)
            gemm_nt(m, n, k, 1., A, 0, 0, B, 0, 0, 1., C, 0, 0, C, 0, 0);

        state.counters["flops"] = Counter(2 * m * m * m, Counter::kIsIterationInvariantRate);
        state.counters["m"] = m;
    }
    

    BENCHMARK_TEMPLATE(BM_gemm, double)->DenseRange(1, BENCHMARK_MAX_GEMM);
    BENCHMARK_TEMPLATE(BM_gemm, float)->DenseRange(1, BENCHMARK_MAX_GEMM);
}
