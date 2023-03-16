factor = 1e+9; % Giga

data = jsondecode(fileread('../../partial_store.json'));
bench = filter_aggregate(data.benchmarks, 'mean');

flops = [];
for i = 1 : length(bench)
    flops(bench(i).store_m, bench(i).store_n) = bench(i).flops;
end

figure;
imagesc(flops / factor);
colormap gray;
axis equal;
axis ij;
xlabel n;
ylabel m;
colorbar;
title(sprintf('Gflops, avg=%.2f, min=%.2f, max=%.2f', ...
    mean(flops(:)) / factor, min(flops(:)) / factor, max(flops(:)) / factor));
