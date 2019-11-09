clf;
hold on;

factor = 1e+9; % Giga
line_width = 1;

% l1 = plot_benchmark('data/dpotrf-blazefeo-static-nounroll.json', factor);
% l1.DisplayName = 'no unroll';
% l1.LineWidth = line_width;
% 
% l2 = plot_benchmark('data/dpotrf-blazefeo-static-unroll.json', factor);
% l2.DisplayName = 'unroll';
% l2.LineWidth = line_width;

plot_rate('data/dpotrf-blazefeo-static-unroll.json', 'data/dpotrf-blazefeo-static-nounroll.json');

grid on;
l = legend('boxon');
% l.Orientation = 'vertical';
% l.Location = 'SouthEast';

xlabel 'matrix size';
ylabel 'performance [Gflops]';


function l = plot_benchmark(file_path, factor)
    data = jsondecode(fileread(file_path));
    benchmarks = filter_aggregate(data.benchmarks, 'mean');
    l = plot([benchmarks.m], [benchmarks.flops] / factor);
end


function l = plot_rate(file_path_1, file_path_2)
    data1 = jsondecode(fileread(file_path_1));
    data2 = jsondecode(fileread(file_path_2));
    benchmarks1 = filter_aggregate(data1.benchmarks, 'mean');
    benchmarks2 = filter_aggregate(data2.benchmarks, 'mean');
    
    m = [benchmarks1.m];    
    assert(isequal(m, [benchmarks2.m]));
    
    l = plot(m, [benchmarks1.flops] ./ [benchmarks2.flops]);
end