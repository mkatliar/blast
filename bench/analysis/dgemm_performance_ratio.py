import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def filter_aggregate(benchmarks, name):
    result = []
    for b in benchmarks:
        try:
            if b['aggregate_name'] == name:
                result.append(b)
        except KeyError:
            continue
    return result

def load_benchmark(file_name):
    with open(file_name) as f:
        return filter_aggregate(json.load(f)['benchmarks'], 'median')


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
]

reference = [
    {'data_file': 'dgemm-blast-static-panel.json', 'label': 'BLAST (SP)'},
    # {'data_file': 'dgemm-blast-dynamic-panel.json', 'label': 'BLAST (D)'},
]

fig = plt.figure(figsize=[10, 6])
ax = fig.subplots()

for r in reference:
    ax.set_title('Perormance relative to {}'.format(r['label']))
    ref = load_benchmark('bench_result/data/{}'.format(r['data_file']))
    m_ref = [b['m'] for b in ref]
    flops_ref = [b['flops'] for b in ref]

    for p in plots:
        benchmark = load_benchmark('bench_result/data/{}'.format(p['data_file']))

        ratio = []
        m = []
        for b in benchmark:
            if b['m'] in m_ref:
                ratio.append(b['flops'] / flops_ref[m_ref.index(b['m'])])
                m.append(b['m'])


        l = ax.plot(m, ratio, style, label=p['label'], linewidth=line_width)

ax.grid(True)
ax.legend()
ax.set_xlabel('matrix size')
ax.set_ylabel('performance ratio')
ax.set_title('dgemm performance relative to BLAST (S)')

fig.savefig('bench_result/image/dgemm_performance_ratio.png')
