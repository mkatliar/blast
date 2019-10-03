#include <smoke/Panel.hpp>

#include <bench/Benchmark.hpp>

#include <random>
#include <memory>


namespace smoke :: benchmark
{
    template <bool TA, bool TB, typename T, size_t N>
    void gemm_impl(Panel<T, N> const& a, Panel<T, N> const& b, Panel<T, N>& c)
    {
        gemm(a, TA, b, TB, c);
    }


    template void gemm_impl<true, false, double, 4>(Panel<double, 4> const& a, Panel<double, 4> const& b, Panel<double, 4>& c);
    template void gemm_impl<false, false, double, 4>(Panel<double, 4> const& a, Panel<double, 4> const& b, Panel<double, 4>& c);
}