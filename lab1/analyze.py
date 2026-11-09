"""Rebuild figures and LaTeX tables from the recorded Polus measurements."""
import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, default=ROOT / 'results')
args = parser.parse_args()
R = args.results
P = ROOT / 'plots'
T = ROOT / 'report_latex' / 'tables'
I = ROOT / 'report_latex' / 'images'
for path in [P, T, I]:
    path.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 180})


def records(name):
    with (R / name).open() as f:
        return list(csv.DictReader(f))


def solution(name, m, n):
    path = R / name
    opener = gzip.open if not path.exists() else open
    if not path.exists():
        path = Path(str(path) + '.gz')
    with opener(path, 'rt') as f:
        data = np.genfromtxt(f, delimiter=',', names=True)
    return {key: data[key].reshape(m + 1, n + 1) for key in data.dtype.names}


def save(fig, name):
    fig.savefig(P / name)
    (I / name).write_bytes((P / name).read_bytes())
    plt.close(fig)


def boundary(ax):
    xx = np.linspace(-3, 3, 1601)
    s = np.sqrt(np.maximum(0, 9 - xx * xx))
    ax.plot(xx, 2 + s, color='#172b4d', lw=.7)
    ax.plot(xx, np.abs(s - 2), color='#172b4d', lw=.7)
    ax.set(xlabel='x', ylabel='y', aspect='equal', xlim=(-3.1, 3.1), ylim=(-.1, 5.1))


def f(value, digits=6):
    s = f'{float(value):.{digits}g}'
    if 'e' in s:
        a, exponent = s.split('e')
        s = a.replace('.', '{,}') + r'\cdot10^{' + str(int(exponent)) + '}'
    else:
        s = s.replace('.', '{,}')
    return '$' + s + '$'


def table(name, caption, headings, rows, alignment):
    text = '\\begin{table}[H]\n\\centering\\small\n\\caption{' + caption + '}\n'
    text += '\\begin{tabular}{' + alignment + '}\n\\toprule\n'
    text += ' & '.join(headings) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (T / (name + '.tex')).write_text(text)


groups = []
for experiment, filename in [('main', 'main_raw.csv'), ('openmp_large', 'openmp_large_raw.csv')]:
    bucket = {}
    for row in records(filename):
        assert row['converged'] == '1' and row['restarts'] == '0', row
        key = tuple(row[k] for k in ['mode', 'M', 'N', 'threads'])
        bucket.setdefault(key, []).append(row)
    for key, rows in bucket.items():
        mode, m, n, threads = key
        values = [float(r['seconds']) for r in rows]
        assert len({r['iterations'] for r in rows}) == 1
        groups.append(dict(mode=mode, M=int(m), N=int(n), threads=int(threads),
                           experiment=experiment, repeats=len(rows),
                           median_seconds=statistics.median(values), min_seconds=min(values),
                           max_seconds=max(values), iterations=int(rows[0]['iterations']),
                           residual=max(float(r['relative_residual']) for r in rows)))
for g in groups:
    base = next(x for x in groups if x['mode'] == 'serial' and x['experiment'] == g['experiment']
                and x['M'] == g['M'] and x['N'] == g['N'])
    g['speedup'] = base['median_seconds'] / g['median_seconds']
    g['efficiency'] = g['speedup'] / g['threads']
with (R / 'summary.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(groups[0]))
    writer.writeheader()
    writer.writerows(groups)
assert len(groups) == 16
assert sum(g['repeats'] for g in groups) == 60

serial = [g for g in groups if g['mode'] == 'serial' and g['experiment'] == 'main']
table('serial', 'Последовательная C++-версия, медианы пяти измерений.',
      [r'$M\times N$', 'Неизвестных', r'$\varepsilon$', 'Итераций', 'Время, с', r'$\|r\|_E/\|B\|_E$'],
      [[f"${g['M']}\\times{g['N']}$", (g['M']-1)*(g['N']-1), f(max(6/g['M'], 5/g['N'])**2),
        g['iterations'], f(g['median_seconds']), f(g['residual'], 3)] for g in serial], 'rrrrrr')
for which in ['main', 'openmp_large']:
    gs = [g for g in groups if g['mode'] == 'openmp' and g['experiment'] == which]
    table('openmp_main' if which == 'main' else 'openmp_large',
          'OpenMP на основной сетке.' if which == 'main' else 'Таблица 1 приложения: OpenMP, ключ -O1.',
          ['Нити', r'$M\times N$', 'Итераций', 'Время, с', r'$S$'],
          [[g['threads'], f"${g['M']}\\times{g['N']}$", g['iterations'], f(g['median_seconds']), f(g['speedup'], 4)]
           for g in gs], 'rrrrr')
base = [g for g in groups if g['mode'] == 'serial' and g['experiment'] == 'openmp_large']
table('large_base', 'Последовательная база больших сеток в том же задании.',
      [r'$M\times N$', 'Итераций', 'Время, с'],
      [[f"${g['M']}\\times{g['N']}$", g['iterations'], f(g['median_seconds'])] for g in base], 'rrr')
checks = records('checks.csv')
exact = [r for r in checks if r['manufactured'] == '1']
assert len(exact) == 3
table('manufactured', 'Погрешность в проверочной задаче с точным решением.',
      [r'$M\times N$', r'$\|w-u_{\rm exact}\|_E$', r'$\|w-u_{\rm exact}\|_\infty$', 'Отношение норм'],
      [[f"${r['M']}\\times{r['N']}$", f(r['sine_error_l2']), f(r['sine_error_max']),
        '---' if i == 0 else f(float(exact[i-1]['sine_error_l2'])/float(r['sine_error_l2']), 4)]
       for i, r in enumerate(exact)], 'rrrr')

ref80 = solution('solution_80.csv', 80, 80)['u']
validation = {}
for threads in [1, 4, 16]:
    u = solution(f'solution_openmp_{threads}.csv', 80, 80)['u']
    difference = float(np.max(np.abs(u-ref80)))
    assert difference < 1e-10
    validation[f'openmp_{threads}_max_difference'] = difference
histories = {n: np.genfromtxt(R / f'trace_{n}.csv', delimiter=',', names=True) for n in [20, 40, 80]}
for n, data in histories.items():
    assert np.array_equal(data['iteration'], np.arange(len(data)))
    assert data['step_norm'][-1] < 1e-9 and data['relative_residual'][-1] < 1e-8
    assert np.max(np.diff(data['functional'])) < 1e-7
    validation[f'iter_error_l2_{n}'] = float(data['iter_error_l2'][-1])
    validation[f'iter_error_max_{n}'] = float(data['iter_error_max'][-1])
(R / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
table('validation', 'Совпадение последовательной и OpenMP-версий на сетке $80\\times80$.',
      ['Нити', r'$\max|w_{\rm OpenMP}-w_{\rm serial}|$'],
      [[t, f(validation[f'openmp_{t}_max_difference'], 3)] for t in [1, 4, 16]], 'rr')
table('reference_error', 'Итерационная ошибка при остановке относительно $w_{\\rm ref}$ с $\\delta=10^{-12}$.',
      [r'$M\times N$', r'$\|w-w_{\rm ref}\|_E$', r'$\|w-w_{\rm ref}\|_\infty$'],
      [[f'${n}\\times{n}$', f(validation[f'iter_error_l2_{n}'], 4), f(validation[f'iter_error_max_{n}'], 4)]
       for n in [20, 40, 80]], 'rrr')

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout='constrained')
for n, d in histories.items():
    axes[0].semilogy(d['iteration'][1:], d['step_norm'][1:], label=f'{n} × {n}')
    axes[1].semilogy(d['iteration'], d['relative_residual'], label=f'{n} × {n}')
axes[0].axhline(1e-9, color='#9a3412', ls='--', label='δ = 10⁻⁹')
axes[0].set(title='Изменение решения', ylabel='Норма E изменения за шаг')
axes[1].set(title='Истинная невязка', ylabel='‖B − Aw‖E / ‖B‖E')
for ax in axes:
    ax.set(xlabel='Итерация');ax.grid(alpha=.2);ax.legend(fontsize=12)
save(fig, 'convergence.png')

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout='constrained')
for n, d in histories.items():
    axes[0].semilogy(d['iteration'], d['iter_error_l2'], label=f'{n} × {n}')
    axes[1].semilogy(d['iteration'], d['iter_error_max'], label=f'{n} × {n}')
for ax, norm in zip(axes, ['E', '∞']):
    ax.set(xlabel='Итерация', ylabel=f'‖wᵏ − wref‖{norm}', title=f'Итерационная ошибка, норма {norm}')
    ax.grid(alpha=.2);ax.legend(fontsize=12)
save(fig, 'iteration_error.png')

fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
for n in [20, 40, 80]:
    d = np.genfromtxt(R / f'exact_trace_{n}.csv', delimiter=',', names=True)
    ax.semilogy(d['iteration'], d['exact_error_max'], label=f'{n} × {n}')
ax.set(xlabel='Итерация', ylabel='max |wᵏ − sin(x+y)|', title='Проверочная задача: ошибка относительно точного решения')
ax.legend();ax.grid(alpha=.2)
save(fig, 'exact_error.png')

for name, m, n in [('solution_80.csv', 80, 80), ('solution_800_1200.csv', 800, 1200)]:
    d = solution(name, m, n)
    fig, ax = plt.subplots(figsize=(6.6, 5.2), layout='constrained')
    im = ax.pcolormesh(d['x'], d['y'], np.ma.array(d['u'], mask=d['in_domain']==0),
                       cmap='viridis', shading='auto', vmin=-1.05, vmax=1.6, rasterized=True)
    boundary(ax);ax.set_title(f'Приближённое решение, {m} × {n}')
    fig.colorbar(im, ax=ax, label='u(x, y)', shrink=.85)
    save(fig, name.replace('.csv', '.png'))

reference = solution('reference_80.csv', 80, 80)
frames = []
for path in sorted((R / 'snapshots').glob('snapshot_*.csv')):
    iteration = int(path.stem.split('_')[-1])
    d = solution(str(path.relative_to(R)), 80, 80)
    frames.append((iteration, np.abs(d['u']-reference['u'])))
assert frames[0][0] == 0 and frames[-1][0] == int(histories[80]['iteration'][-1])
fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), layout='constrained')
for ax, wanted in zip(axes.ravel(), [0, 50, 150, frames[-1][0]]):
    d = solution(f'snapshots/snapshot_{wanted:05d}.csv', 80, 80)
    im = ax.pcolormesh(d['x'], d['y'], np.ma.array(d['u'], mask=d['in_domain']==0),
                       cmap='viridis', vmin=-1.05, vmax=1.6, shading='auto')
    boundary(ax);ax.set_title(f'Итерация {wanted}')
fig.colorbar(im, ax=axes.ravel().tolist(), label='wᵏ(x, y), одна шкала')
save(fig, 'solution_iterations.png')
fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), layout='constrained')
for ax, wanted in zip(axes.ravel(), [0, 50, 150, frames[-1][0]]):
    iteration, error = next(frame for frame in frames if frame[0]==wanted)
    im = ax.pcolormesh(reference['x'], reference['y'],
                       np.ma.array(np.maximum(error, 1e-12), mask=reference['in_domain']==0),
                       norm=LogNorm(vmin=1e-10, vmax=2), cmap='magma', shading='auto')
    boundary(ax);ax.set_title(f'Итерация {iteration}')
fig.colorbar(im, ax=axes.ravel().tolist(), label='|wᵏ − wref|, одна шкала')
save(fig, 'spatial_error.png')
threshold = 1e-5
last_above = np.full(reference['u'].shape, -1, dtype=int)
for index, (_, error) in enumerate(frames):
    last_above[error>threshold] = index
sampled = np.zeros(reference['u'].shape)
for index, (iteration, _) in enumerate(frames):
    sampled[last_above==index-1] = iteration
sampled[last_above==len(frames)-1] = np.nan
fig, ax = plt.subplots(figsize=(6.7, 5.1), layout='constrained')
im = ax.pcolormesh(reference['x'], reference['y'],
                   np.ma.array(sampled, mask=reference['in_domain']==0),
                   cmap='viridis', shading='auto', vmin=0, vmax=frames[-1][0])
boundary(ax);ax.set_title('Сходимость в разных частях области, 80 × 80')
fig.colorbar(im, ax=ax, label='Первый снимок с устойчивой ошибкой ≤ 10⁻⁵', shrink=.84)
save(fig, 'spatial_convergence.png')

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout='constrained')
for ax, experiment in zip(axes, ['main', 'openmp_large']):
    for m, n in [(80, 80), (400, 600), (800, 1200)]:
        gs = sorted([g for g in groups if g['mode']=='openmp' and g['experiment']==experiment
                     and g['M']==m and g['N']==n], key=lambda g:g['threads'])
        if gs:
            ax.plot([g['threads'] for g in gs], [g['speedup'] for g in gs], 'o-', label=f'{m} × {n}')
    ax.axhline(1, color='#94a3b8', ls='--', lw=1)
    ax.set(xlabel='Число нитей', ylabel='Ускорение Tпосл / TOpenMP', ylim=(0, None),
           title='Основная сетка' if experiment=='main' else 'Большие сетки')
    ax.grid(alpha=.2);ax.legend(fontsize=12)
fig.suptitle('OpenMP, C++, ключ -O1; один узел и одна серия')
save(fig, 'speedup_openmp.png')
best = max((g for g in groups if g['mode']=='openmp'), key=lambda g:g['speedup'])
(T/'conclusion.tex').write_text('Наибольшее ускорение в измеренной серии составило '+f(best['speedup'],4)
    +' при '+str(best['threads'])+' нитях на сетке $'+str(best['M'])+'\\times'+str(best['N'])+'$.\n')
print(json.dumps({'configurations':len(groups), 'measurements':sum(g['repeats'] for g in groups),
                  'best':best, 'validation':validation}, indent=2))
