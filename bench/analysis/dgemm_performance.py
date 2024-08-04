import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def filter_aggregate(benchmarks, name):
    return [b for b in benchmarks if b['aggregate_name'] == name]


factor = 1e+9 # Giga
style = 'o-'
line_width = 1

# prop_cycle = plt.rcParams['axes.prop_cycle']
# c = prop_cycle.by_key()['color']

plots = [
    # {'data_file': 'dgemm-openblas.json', 'label': 'OpenBLAS'},
    {'data_file': 'dgemm-mkl.json', 'label': 'MKL'},
    {'data_file': 'dgemm-blasfeo.json', 'label': 'BLASFEO'},
    # {'data_file': 'dgemm-blasfeo-blas.json', 'label': 'BLASFEO*'},
    {'data_file': 'dgemm-libxsmm.json', 'label': 'LIBXSMM'},
    # {'data_file': 'dgemm-eigen-dynamic.json', 'label': 'Eigen (D)'},
    # {'data_file': 'dgemm-eigen-static.json', 'label': 'Eigen (S)'},
    # {'data_file': 'dgemm-blaze-dynamic.json', 'label': 'Blaze (D)'},
    {'data_file': 'dgemm-blaze-static.json', 'label': 'Blaze (S)'},
    {'data_file': 'dgemm-blast-static-panel.json', 'label': 'BLAST (SP)'},
    {'data_file': 'dgemm-blast-static-plain.json', 'label': 'BLAST (SD)'},
    {'data_file': 'dgemm-blast-dynamic-panel.json', 'label': 'BLAST (DP)'},
    {'data_file': 'dgemm-blast-dynamic-plain.json', 'label': 'BLAST (DD)'},
]

fig = plt.figure(figsize=[10, 6])
ax = fig.subplots()

for p in plots:
    file_name = 'bench_result/data/{}'.format(p['data_file'])
    print(f"Processing {file_name}")
    with open(file_name) as f:
        benchmarks = filter_aggregate(json.load(f)['benchmarks'], 'mean')
    l = ax.plot(
        [b['m'] for b in benchmarks],
        [b['flops'] / factor for b in benchmarks],
        style,
        label = p['label'],
        linewidth = line_width,
        markersize = 4
    )

ax.grid(True)
ax.legend()
ax.set_xlabel('matrix size')
ax.set_ylabel('performance [Gflops]')
ax.set_title('dgemm performance')

fig.savefig('bench_result/image/dgemm_performance.png')
