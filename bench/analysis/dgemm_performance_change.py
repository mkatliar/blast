import matplotlib.pyplot as plt
import json
import os.path


def filter_aggregate(benchmarks, name):
    return [b for b in benchmarks if b['aggregate_name'] == name]


def load_benchmark(file_name):
    with open(file_name) as f:
        return filter_aggregate(json.load(f)['benchmarks'], 'mean')

def ref_data_file_name(data_file_name):
    data_file_base, data_file_ext = os.path.splitext(data_file_name)
    return data_file_base + ".ref" + data_file_ext


factor = 1e+9 # Giga
style = 'o-'
line_width = 1

plots = [
    {'data_file': 'dgemm-blast-static-panel.json', 'label': 'BLAST (SP)'},
    {'data_file': 'dgemm-blast-static-plain.json', 'label': 'BLAST (SD)'},
    {'data_file': 'dgemm-blast-dynamic-panel.json', 'label': 'BLAST (DP)'},
    {'data_file': 'dgemm-blast-dynamic-plain.json', 'label': 'BLAST (DD)'},
]

fig = plt.figure(figsize=[10, 6])
ax = fig.subplots()

ax.set_title('Relative perormance change')

for p in plots:
    try:
        benchmark = load_benchmark(f'bench_result/data/{p["data_file"]}')
        ref = load_benchmark(f'bench_result/data/{ref_data_file_name(p["data_file"])}')
        m_ref = [b['m'] for b in ref]
        flops_ref = [b['flops'] for b in ref]

        ratio = []
        m = []
        for b in benchmark:
            if b['m'] in m_ref:
                ratio.append(b['flops'] / flops_ref[m_ref.index(b['m'])])
                m.append(b['m'])

        l = ax.plot(m, ratio, style, label=p['label'], linewidth=line_width, markersize=3)
    except FileNotFoundError as e:
        print(f"Skipping {p['label']}: {e}")

ax.grid(True)
ax.legend()
ax.set_xlabel('matrix size')
ax.set_ylabel('performance ratio')
ax.set_title('dgemm performance change')

fig.savefig('bench_result/image/dgemm_performance_change.png')
