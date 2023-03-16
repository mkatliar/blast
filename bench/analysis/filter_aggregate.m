function benchmarks = filter_aggregate(benchmarks, name)
    ind = strcmp({benchmarks.aggregate_name}, name);
    benchmarks = benchmarks(ind);
end