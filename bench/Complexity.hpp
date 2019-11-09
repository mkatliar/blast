#pragma once

#include <bench/Benchmark.hpp>

#include <cstdlib>
#include <string>
#include <map>
#include <stdexcept>


namespace blazefeo :: benchmark
{
    using Complexity = std::map<std::string, std::size_t>;

    /// @brief Algorithmic complexity of potrf
    inline Complexity complexityPotrf(std::size_t m, std::size_t n)
    {
        if (m < n)
            throw std::invalid_argument("Cannot calculate complexity of potrf for m < n");

        return {
            // Calculated as \sum _{k=0}^{n-1} \sum _{j=0}^{k-1} \sum _{i=k}^{m-1} 1
            {"add", (1 + 3 * m - 2 * n) * (n - 1) * n / 6},
            // Calculated as \sum _{k=0}^{n-1} \sum _{j=0}^{k-1} \sum _{i=k}^{m-1} 1
            {"mul", (1 + 3 * m - 2 * n) * (n - 1) * n / 6},
            // Calculated as \sum _{k=0}^{n-1} \sum _{i=k}^{m-1} 1
            {"div", n * (1 + 2 * m - n) / 2},
            // Calculated as \sum _{k=0}^{n-1} 1
            {"sqrt", n}
        };
    }


    /// @brief Algorithmic complexity of trsm
    inline Complexity complexityTrsm(std::size_t m, std::size_t n)
    {
        return {
            // Calculated as \sum _{j=0}^{n-1} \sum _{k=0}^{j-1} \sum _{i=0}^{m-1} 1
            {"add", m * (n - 1) * n / 2},
            // Calculated as \sum _{j=0}^{n-1} \sum _{k=0}^{j-1} \sum _{i=0}^{m-1} 1
            {"mul", m * (n - 1) * n / 2},
            // Calculated as \sum _{k=0}^{n-1} \sum _{i=k}^{m-1} 1
            {"div", n * (1 + 2 * m - n) / 2}
        };
    }


    template <typename Map>
    void setCounters(Map& counters, Complexity const& c)
    {
        for (auto const& v : c)
            counters["f." + v.first] = Counter(v.second, Counter::kIsIterationInvariantRate);

        std::size_t flops = 0;
        for (auto op : {"add", "mul"})
        {
            auto const it = c.find(op);
            if (it != c.end())
                flops += it->second;
        }

        counters["flops"] = Counter(flops, Counter::kIsIterationInvariantRate);;
    }
}