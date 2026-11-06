"""Figures for report_draft.tex.

Reads only saved results:
  results/rel_study/runs/*/*.json          per-episode miss of every trained model
  results/rel_study/baseline_analysis.json  pre-sweep measurements on the 10 Hz baseline
  results/rel_study/plots/figure_data.npz   winner braking and path traces
  results/rel_study/report_dart_stats.npz   demonstration-set statistics (report_dart_stats.py)
  results/rel_study/report_rollouts.npz     per-episode parking geometry (report_rollouts.py)

Writes PDF and PNG to <repo>/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'results' / 'rel_study'
OUT = ROOT.parents[2] / 'figures'
OUT.mkdir(exist_ok=True)

# Model identity is fixed across every figure (validated categorical slots 1-3 + neutral).
C = dict(baseline='#6f6e69', winner='#2a78d6', dart='#eb6834', offset='#1baf7a')
LS = dict(baseline=(0, (1, 1.5)), winner='-', dart=(0, (5, 2)), offset=(0, (6, 1.5, 1, 1.5)))
LABEL = dict(baseline='10 Hz baseline', winner='25 Hz winner', dart='+ high-amp. DART', offset='+ parking offsets')
INK, INK2, GRID = '#1f1f1d', '#52514e', '#dcdbd6'
COL_W, PAGE_W = 3.45, 7.16
WHEEL_R = 0.11

SWEEP = [
    # (label, role, name, family)
    ('10 Hz baseline', 'sweep', 'r10_p2_e0.5_hnone', 'Baseline'),
    ('5 Hz', 'sweep', 'r5_p2_e0.5_hnone', 'Rate'),
    ('25 Hz', 'sweep', 'r25_p2_e0.5_hnone', 'Rate'),
    ('50 Hz', 'sweep', 'r50_p2_e0.5_hnone', 'Rate'),
    ('Stack 0.5 s', 'sweep', 'r10_p2_e0.5_hstack0.5s', 'History'),
    ('Stack 2 s', 'sweep', 'r10_p2_e0.5_hstack2s', 'History'),
    ('GRU 2 s', 'sweep', 'r10_p2_e0.5_hgru2s', 'History'),
    ('Predict 1 s', 'sweep', 'r10_p1_e1_hnone', 'Timing'),
    ('Predict 5 s', 'sweep', 'r10_p5_e0.2_hnone', 'Timing'),
    ('Execute 0.5 s', 'sweep', 'r10_p2_e0.25_hnone', 'Timing'),
    ('Chunk, exec. 0.3 s', 'sweep', 'r10_p0.5_e0.5_hnone', 'Timing'),
    ('Chunk, exec. 0.1 s', 'sweep', 'r10_p0.5_e0.25_hnone', 'Timing'),
    ('25 Hz, predict 5 s', 'combination', 'r25_p5_e0.2_hnone', 'Combination'),
    ('50 Hz, predict 5 s', 'combination', 'r50_p5_e0.2_hnone', 'Combination'),
    ('25 Hz, pred. 5 s, GRU', 'combination', 'r25_p5_e0.2_hgru2s', 'Combination'),
]
FINAL = dict(baseline=('sweep', 'r10_p2_e0.5_hnone_d1000'), winner=('sweep', 'r25_p2_e0.5_hnone_d1000'),
             dart=('dart', 'r25_p2_e0.5_hnone_d2000'), offset=('offset', 'r25_p2_e0.5_hnone_d2000'))


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
        'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.4, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.titleweight': 'normal',
        'axes.titlelocation': 'left',
    })


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


def panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', color=INK, pad=4)


def runs(role, name):
    out = []
    for seed in (0, 1, 2):
        path = STUDY / 'runs' / role / f'{name}_rel_add_s{seed}.json'
        if not path.exists() and not name.endswith('_d1000') and not name.endswith('_d2000'):
            path = STUDY / 'runs' / role / f'{name}_d1000_rel_add_s{seed}.json'
        out.append(json.loads(path.read_text()))
    return out


def seed_stats(rs):
    miss = np.array([np.mean([e['min_dist'] * 1000 for e in r['episodes']]) for r in rs])
    catch = np.array([np.mean([e['min_dist'] < 0.01 for e in r['episodes']]) * 100 for r in rs])
    return miss, catch


def pooled(role, name):
    return np.concatenate([[e['min_dist'] * 1000 for e in r['episodes']] for r in runs(role, name)])


# ---------------------------------------------------------------- task
def fig_task():
    z = np.load(STUDY / 'report_dart_stats.npz')
    ex = 1
    origin = z[f'ex{ex}_clean_base'][0]
    ball = z[f'ex{ex}_ball'] - np.r_[origin, 0]
    base = z[f'ex{ex}_clean_base'] - origin
    k = int(z[f'ex{ex}_catch_k'])
    ball, base = ball[:k + 1], base[:k + 1]
    inside = ((np.abs(ball[:, 0]) <= 2.5) & (np.abs(ball[:, 1]) <= 2.5) & (ball[:, 2] >= 0.07) & (ball[:, 2] <= 1.30))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.38))
    a1.add_patch(plt.Rectangle((-2.5, -2.5), 5, 5, fill=False, ec=INK2, lw=0.8, ls=(0, (3, 2))))
    a1.plot(ball[:, 0], ball[:, 1], color=C['dart'], lw=1.1, label='ball')
    a1.plot(base[:, 0], base[:, 1], color=C['winner'], lw=1.6, label='expert base')
    a1.plot(0, 0, marker='s', ms=3.5, color=C['winner'])
    a1.plot(*ball[-1, :2], marker='x', ms=5, mew=1.2, color=INK)
    a1.set_xlim(-2.8, 2.8)
    a1.set_ylim(-2.8, 2.8)
    a1.set_aspect('equal')
    a1.set_xlabel('x (m)')
    a1.set_ylabel('y (m)')
    a1.legend(loc='lower left', fontsize=5.4, handlelength=1.3)
    panel(a1, 'a', 'Top view')
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1))]
    a2.axhspan(0.07, 1.30, color='#efeee9', lw=0)
    a2.plot(s, ball[:, 2], color=C['dart'], lw=0.9, ls=(0, (2, 1.5)))
    seg = np.where(inside, ball[:, 2], np.nan)
    a2.plot(s, seg, color=C['dart'], lw=1.6)
    a2.plot(s[-1], ball[-1, 2], marker='x', ms=5, mew=1.2, color=INK)
    a2.text(s[-1] - 0.1, ball[-1, 2] + 0.25, 'catch', ha='right', fontsize=5.8, color=INK)
    a2.text(0.1, 0.12, 'workspace height', fontsize=5.6, color=INK2)
    a2.set_xlabel('Distance along ball path (m)')
    a2.set_ylabel('Ball height (m)')
    panel(a2, 'b', 'Leaves the roof, re-enters')
    save(fig, 'fig_task')


# ---------------------------------------------------------------- baseline diagnosis
def fig_baseline():
    ba = json.loads((STUDY / 'baseline_analysis.json').read_text())
    cv = ba['command_variation']
    rates = [5, 10, 25, 50]
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.75), gridspec_kw=dict(wspace=0.42))

    ax = axes[0]
    vals = [cv[str(r)]['wheel_rms'] for r in rates]
    cols = [C['baseline'] if r == 10 else (C['winner'] if r == 25 else '#b9b8b2') for r in rates]
    ax.bar(range(4), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=6, color=INK2)
    ax.set_xticks(range(4), [f'{r} Hz' for r in rates])
    ax.set_xlabel('Policy rate (hold = 1/rate)')
    ax.set_ylabel('RMS wheel-command change\nwithin one held tick (rad/s)')
    ax.set_ylim(0, 0.8)
    ax.grid(axis='x', visible=False)
    panel(ax, 'a', 'Expert moves within a tick')

    ax = axes[1]
    mse = np.asarray(ba['test']['mse_horizon'])
    t = (np.arange(len(mse)) + 1) * 0.1
    ax.plot(t, mse / mse[0], color=C['baseline'], lw=1.4)
    ax.axvline(1.0, color=INK2, lw=0.6, ls=':')
    ax.text(0.97, 10.6, 'executed', fontsize=5.8, color=INK2, va='top', ha='right')
    ax.text(1.03, 10.6, 'discarded', fontsize=5.8, color=INK2, va='top')
    ax.set_xlabel('Step along predicted chunk (s)')
    ax.set_ylabel('Open-loop MSE / first step')
    ax.set_ylim(0, 11)
    panel(ax, 'b', 'Chunk error grows 10x')

    ax = axes[2]
    before = ba['test']['before']
    names = [b['name'].replace('–', '-') for b in before]
    wheel = [b['wheel'] for b in before]
    ax.bar(range(len(before)), wheel, color=C['baseline'], width=0.62)
    ax.set_xticks(range(len(before)), names)
    ax.set_xlabel('Time before the expert catch')
    ax.set_ylabel('Open-loop wheel MSE (norm.)')
    ax.grid(axis='x', visible=False)
    ax.text(3.0, wheel[-1] * 0.93, 'base\ndriving', ha='center', va='top', fontsize=6, color=INK2)
    ax.text(1, wheel[1] * 2.6, 'arm reach', ha='center', va='bottom', fontsize=6, color=INK2)
    panel(ax, 'c', 'Error sits in the drive')

    ax = axes[3]
    p3 = ba['part3']
    tau = np.array([p['tau'] for p in p3])
    mean = np.array([p['mean_miss_mm'] for p in p3])
    ax.plot(tau, mean, marker='o', ms=3.2, color=C['baseline'], lw=1.2, label='Expert drives, then\nlearner takes over')
    full = np.mean(seed_stats(runs('sweep', 'r10_p2_e0.5_hnone'))[0])
    ax.axhline(full, color=C['dart'], lw=1.0, ls=(0, (4, 2)))
    ax.text(0.45, full * 1.12, f'learner from the start: {full:.0f} mm', fontsize=5.8, color=INK2)
    ax.axhline(10, color=INK2, lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.5, 1, 2, 4, 8, 15], ['0.5', '1', '2', '4', '8', '15'])
    ax.set_ylim(1, 300)
    ax.set_xlabel('Learner control before catch (s)')
    ax.set_ylabel('Mean miss (mm)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.86), fontsize=5.8, handlelength=1.4)
    panel(ax, 'd', 'Late takeover catches')
    save(fig, 'fig_baseline_diagnosis')


# ---------------------------------------------------------------- sweep
def fig_sweep():
    rows = []
    for label, role, name, fam in SWEEP:
        miss, catch = seed_stats(runs(role, name))
        rows.append((label, fam, miss, catch))
        print(f'{label:24s} catch {catch.mean():4.1f}±{catch.std(ddof=1):3.1f}  miss {miss.mean():7.1f}±{miss.std(ddof=1):5.1f}')
    n = len(rows)
    y = np.arange(n)[::-1].astype(float)
    # visual gap between families
    fams = [r[1] for r in rows]
    shift = 0.0
    for i in range(1, n):
        if fams[i] != fams[i - 1]:
            shift += 0.6
        y[i] -= shift
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 3.15), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08))
    base_miss = rows[0][2].mean()
    for yi, (label, fam, miss, catch) in zip(y, rows):
        col = C['winner'] if label == '25 Hz' else (C['baseline'] if fam == 'Baseline' else '#8f8e88')
        mk = 'D' if label == '25 Hz' else 'o'
        a1.errorbar(miss.mean(), yi, xerr=miss.std(ddof=1), fmt=mk, ms=3.6, color=col, ecolor=col, elinewidth=0.9, capsize=1.5)
        a1.scatter(miss, [yi] * 3, s=4, color=col, alpha=0.45, lw=0, zorder=1)
        a2.errorbar(catch.mean(), yi, xerr=catch.std(ddof=1), fmt=mk, ms=3.6, color=col, ecolor=col, elinewidth=0.9, capsize=1.5)
    a1.axvline(base_miss, color=C['baseline'], lw=0.7, ls=':')
    a2.axvline(rows[0][3].mean(), color=C['baseline'], lw=0.7, ls=':')
    a1.set_xscale('log')
    a1.set_xlim(40, 2000)
    a1.set_xticks([50, 100, 200, 500, 1000], ['50', '100', '200', '500', '1000'])
    a1.set_xlabel('Mean miss (mm, log)')
    a2.set_xlabel('Catch within 10 mm (%)')
    a2.set_xlim(-0.5, 8.5)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.tick_params(axis='y', length=0)
    a2.tick_params(axis='y', length=0)
    a1.grid(axis='y', visible=False)
    a2.grid(axis='y', visible=False)
    seen = {}
    for yi, f in zip(y, fams):
        seen.setdefault(f, []).append(yi)
    for f, ys in seen.items():
        if f == 'Baseline':
            continue
        a2.text(8.4, max(ys) + 0.15, f, ha='right', va='bottom', fontsize=6, color=INK2, style='italic')
    save(fig, 'fig_sweep')


# ---------------------------------------------------------------- progression
def ecdf(ax, x, key, label=None):
    x = np.sort(x)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=C[key], ls=LS[key], lw=1.3,
            label=label or LABEL[key])


def fig_progression():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw=dict(width_ratios=[1.45, 1.15], wspace=0.42))
    stats = {}
    for key in ('baseline', 'winner', 'dart', 'offset'):
        role, name = FINAL[key]
        x = pooled(role, name)
        ecdf(a1, x, key)
        stats[key] = seed_stats(runs(role, name))
    a1.axvline(10, color=INK2, lw=0.6, ls=':')
    a1.axvline(50, color=INK2, lw=0.6, ls=':')
    a1.set_xlim(0, 200)
    a1.set_ylim(0, 1)
    a1.set_xlabel('Miss distance (mm)')
    a1.set_ylabel('Fraction of 600 episodes')
    a1.legend(loc='lower right', fontsize=5.6, handlelength=2.2)
    panel(a1, 'a', 'Miss distribution')
    keys = ['baseline', 'winner', 'dart', 'offset']
    for i, k in enumerate(keys):
        m = stats[k][0]
        a2.bar(i, m.mean(), color=C[k], width=0.62, alpha=0.9)
        a2.errorbar(i, m.mean(), yerr=m.std(ddof=1), color=INK, elinewidth=0.8, capsize=1.8)
        a2.text(i, m.mean() + m.std(ddof=1) + 3, f'{m.mean():.0f}', ha='center', fontsize=6, color=INK2)
    a2.set_xticks(range(4), ['10 Hz', '25 Hz', 'DART', 'Offset'], fontsize=6, rotation=35, ha='right',
                  rotation_mode='anchor')
    a2.set_ylabel('Mean miss (mm)')
    a2.set_ylim(0, 135)
    a2.grid(axis='x', visible=False)
    panel(a2, 'b', 'Mean miss')
    save(fig, 'fig_progression')


# ---------------------------------------------------------------- DART demonstrations
def fig_dart_demos():
    z = np.load(STUDY / 'report_dart_stats.npz')
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.85), gridspec_kw=dict(wspace=0.42, width_ratios=[1.25, 0.8, 1.05, 1.05]))
    ex = 1
    t = z[f'ex{ex}_t']
    cmd = z[f'ex{ex}_dart_cmd'].mean(axis=1) * WHEEL_R * 1000
    exe = z[f'ex{ex}_dart_exec'].mean(axis=1) * WHEEL_R * 1000
    ax = axes[0]
    ax.plot(t, exe, color=C['dart'], lw=0.8, label='executed (noisy)')
    ax.plot(t, cmd, color=INK, lw=1.2, label='clean expert label')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Forward command (mm/s)')
    ax.legend(loc='upper right', fontsize=5.8, handlelength=1.4)
    ax.set_xlim(0, t[-1])
    panel(ax, 'a', 'One DART demonstration')

    ax = axes[1]
    nr, cr = z['noise_rms'], z['cmd_rms']
    groups = [('Wheels', slice(0, 4)), ('Arm vel.', slice(11, 17))]
    ratio = [np.sqrt(np.mean(nr[s] ** 2)) / np.sqrt(np.mean(cr[s] ** 2)) * 100 for _, s in groups]
    ax.bar(range(2), ratio, color=C['dart'], width=0.55)
    for i, v in enumerate(ratio):
        ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=6, color=INK2)
    ax.set_xticks(range(2), [g for g, _ in groups])
    ax.set_ylabel('Noise RMS / command RMS (%)')
    ax.set_ylim(0, 110)
    ax.grid(axis='x', visible=False)
    panel(ax, 'b', 'Noise vs. signal')

    ax = axes[2]
    for key, kind, lw in (('baseline', 'clean', 1.4), ('offset', 'offset', 1.1), ('dart', 'dart', 1.0)):
        b = z[f'ex{ex}_{kind}_base']
        b = b - z[f'ex{ex}_clean_base'][0]
        lab = dict(clean='clean expert', offset='parking offset', dart='DART noise')[kind]
        col = INK if kind == 'clean' else C[key]
        ax.plot(b[:, 0], b[:, 1], color=col, lw=lw, ls='-' if kind != 'offset' else (0, (3, 1.5)), label=lab)
    k = int(z[f'ex{ex}_catch_k'])
    ball = z[f'ex{ex}_ball'][k] - np.r_[z[f'ex{ex}_clean_base'][0], 0]
    ax.plot(ball[0], ball[1], marker='x', ms=5, color=INK, mew=1.2, ls='none', label='catch point')
    kc = int(z[f'ex{ex}_catch_k'])
    for kind, col in (('clean', INK), ('offset', C['offset']), ('dart', C['dart'])):
        b = z[f'ex{ex}_{kind}_base'] - z[f'ex{ex}_clean_base'][0]
        ax.plot(*b[min(kc, len(b) - 1)], marker='o', ms=3.5, color=col, ls='none')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.6, 2.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.text(0.05, -0.5, 'clean and offset\nexperts (overlap)', fontsize=5.6, color=INK)
    ax.text(1.75, 1.45, 'DART\nnoise', fontsize=5.6, color=C['dart'])
    ax.text(2.13, 0.25, 'ball at\ncatch', fontsize=5.6, color=INK, ha='center')
    panel(ax, 'c', 'Base path, same scene')

    ax = axes[3]
    for key, field in (('offset', 'offset_base_mm'), ('dart', 'dart_base_mm')):
        x = np.sort(z[field])
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=C[key], ls=LS[key], lw=1.3)
    ax.text(6, 0.97, 'offset:\ncatches\n100%', fontsize=5.6, color=INK, ha='left', va='top')
    ax.text(1200, 0.32, 'DART:\ncatches\n0%', fontsize=5.6, color=INK, ha='left', va='top')
    ax.text(160, 0.04, '150 mm cap', fontsize=5.4, color=INK2, rotation=90, va='bottom', ha='left')
    ax.axvline(150, color=INK2, lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_xlim(5, 5000)
    ax.set_xlabel('Base distance from clean expert\nat its catch time (mm)')
    ax.set_ylabel('Fraction of 1,000 demos')
    panel(ax, 'd', 'Where the demos park')
    save(fig, 'fig_dart_demos')
    print('dart demo base median', np.median(z['dart_base_mm']), 'offset', np.median(z['offset_base_mm']),
          'ratio wheels/arm', ratio)


# ---------------------------------------------------------------- closed-loop parking
def load_roll():
    path = STUDY / 'report_rollouts.npz'
    if not path.exists():
        return None
    z = np.load(path)
    out = {}
    for key in ('baseline', 'winner', 'dart', 'offset'):
        if f'{key}__miss_mm' in z.files:
            out[key] = {f: z[f'{key}__{f}'] for f in ('scene', 'model_seed', 'success', 'miss_mm', 'base_along_mm',
                                                      'base_cross_mm', 'miss_along_mm', 'miss_cross_mm', 'miss_up_mm')}
    return out


def decompose(r):
    """Split the mean squared 3-D miss into the part a per-axis linear fit on the base error explains."""
    total = np.mean(r['miss_along_mm'] ** 2 + r['miss_cross_mm'] ** 2 + r['miss_up_mm'] ** 2)
    explained, r2, coef = [], [], []
    rest = 0.0
    for bname, mname in (('base_along_mm', 'miss_along_mm'), ('base_cross_mm', 'miss_cross_mm')):
        b, m = r[bname], r[mname]
        s = np.cov(b, m, ddof=0)[0, 1] / np.var(b)
        e = s ** 2 * np.var(b)
        explained.append(e)
        r2.append(e / np.var(m))
        coef.append(1 - s)
        rest += np.mean(m ** 2) - e
    up = np.mean(r['miss_up_mm'] ** 2)
    share = np.array([explained[0], explained[1], rest, up]) / total
    return dict(share=share, rms=float(np.sqrt(total)), r2=r2, coef=coef)


def slope_line(ax, b, m, col):
    s = np.cov(b, m, ddof=0)[0, 1] / np.var(b)
    xs = np.array([-150, 150])
    ax.plot(xs, s * xs + (np.mean(m) - s * np.mean(b)), color=col, lw=1.1)
    return 1 - s


def fig_transfer(roll):
    keys = [k for k in ('winner', 'dart', 'offset') if k in roll]
    fig, axes = plt.subplots(2, len(keys), figsize=(PAGE_W * len(keys) / 3, 3.3), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.18), squeeze=False)
    for j, k in enumerate(keys):
        r = roll[k]
        for i, (axis, bname, mname) in enumerate((('Along-track', 'base_along_mm', 'miss_along_mm'),
                                                    ('Cross-track', 'base_cross_mm', 'miss_cross_mm'))):
            ax = axes[i, j]
            b, m = r[bname], r[mname]
            ax.plot([-150, 150], [-150, 150], color=INK2, lw=0.7, ls=(0, (4, 2)))
            ax.axhline(0, color=INK2, lw=0.6)
            ax.scatter(b, m, s=3, color=C[k], alpha=0.35, lw=0)
            coef = slope_line(ax, b, m, C[k] if k != 'offset' else '#0e7a54')
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.set_aspect('equal')
            ax.text(-142, 140, f'{axis}\n$1-s$ = {coef:.2f}', fontsize=6, va='top', color=INK,
                    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))
            if i == 0:
                panel(ax, 'abc'[j], LABEL[k])
            if i == 1:
                ax.set_xlabel('Base parking error (mm)')
            if j == 0:
                ax.set_ylabel('Hand miss (mm)')
    save(fig, 'fig_transfer')


def fig_remaining(roll):
    keys = [k for k in ('baseline', 'winner', 'dart', 'offset') if k in roll]
    fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(PAGE_W, 2.05),
                                         gridspec_kw=dict(wspace=0.55, width_ratios=[1.05, 1, 0.9, 0.85]))
    # (a) parking error distribution
    for k in keys:
        r = roll[k]
        x = np.sort(np.hypot(r['base_along_mm'], r['base_cross_mm']))
        a1.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=C[k], ls=LS[k], lw=1.3, label=LABEL[k])
    a1.set_xlim(0, 250)
    a1.set_ylim(0, 1)
    a1.set_xlabel('Horizontal base error (mm)')
    a1.set_ylabel('Fraction of episodes')
    a1.legend(loc='lower right', fontsize=5.6, handlelength=2.2)
    panel(a1, 'a', 'Parking error')
    # (b) median miss by parking-error bin
    edges = np.array([0, 20, 40, 60, 90, 130, 400])
    mids = np.arange(len(edges) - 1)
    for k in keys:
        r = roll[k]
        b = np.hypot(r['base_along_mm'], r['base_cross_mm'])
        med, ok = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (b >= lo) & (b < hi)
            med.append(np.median(r['miss_mm'][sel]) if sel.sum() >= 10 else np.nan)
        a2.plot(mids, med, color=C[k], ls=LS[k], marker='o', ms=2.8, lw=1.2)
    a2.set_xticks(mids, [f'{lo}-{hi}' if hi < 400 else f'>{lo}' for lo, hi in zip(edges[:-1], edges[1:])], fontsize=5.6, rotation=35, ha='right', rotation_mode='anchor')
    a2.set_xlabel('Base error bin (mm)')
    a2.set_ylabel('Median miss (mm)')
    a2.set_ylim(0, None)
    panel(a2, 'b', 'Miss vs. parking error')
    # (c) where the hand miss is: component RMS
    comps = [('along', 'miss_along_mm'), ('cross', 'miss_cross_mm'), ('vertical', 'miss_up_mm')]
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        r = roll[k]
        vals = [np.median(np.abs(r[f])) for _, f in comps]
        a3.bar(np.arange(3) + (i - (len(keys) - 1) / 2) * w, vals, width=w * 0.9, color=C[k], label=LABEL[k])
    a3.set_xticks(range(3), [c for c, _ in comps])
    a3.set_ylabel('Median abs. hand miss (mm)')
    a3.grid(axis='x', visible=False)
    panel(a3, 'c', 'Miss by direction')
    # (d) share of the mean squared miss explained by parking error (per-axis linear fit)
    parts = [('parking, along', '#1c5cab'), ('parking, cross', '#6da7ec'), ('other horizontal', '#c9dcf3'),
             ('vertical', '#d9d8d3')]
    for i, k in enumerate(keys):
        sh = decompose(roll[k])
        bottom = 0.0
        for (name, col), v in zip(parts, sh['share']):
            a4.bar(i, v * 100, bottom=bottom, color=col, width=0.62, edgecolor='white', linewidth=0.6,
                   label=name if i == 0 else None)
            bottom += v * 100
        a4.text(i, 101, f"{sh['rms']:.0f}", ha='center', va='bottom', fontsize=5.6, color=INK2)
    a4.set_xticks(range(len(keys)), [dict(baseline='10 Hz', winner='25 Hz', dart='DART', offset='Offset')[k] for k in keys],
                  fontsize=6, rotation=35, ha='right', rotation_mode='anchor')
    a4.set_ylim(0, 112)
    a4.set_ylabel('Share of mean squared miss (%)')
    a4.grid(axis='x', visible=False)
    a4.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5.2, handlelength=1.0)
    panel(a4, 'd', 'Share from parking')
    save(fig, 'fig_remaining')
    for k in keys:
        sh = decompose(roll[k])
        print(f"{k:9s} share parking-along {sh['share'][0]:.2f} parking-cross {sh['share'][1]:.2f} "
              f"other-horiz {sh['share'][2]:.2f} vertical {sh['share'][3]:.2f}  rms {sh['rms']:.1f}  "
              f"R2 along {sh['r2'][0]:.2f} cross {sh['r2'][1]:.2f}  1-s along {sh['coef'][0]:.2f} cross {sh['coef'][1]:.2f}")
    for k in keys:
        r = roll[k]
        b = np.hypot(r['base_along_mm'], r['base_cross_mm'])
        m = r['miss_mm']
        # share of the hand miss that the base error would explain if the arm did not correct it
        rho = np.corrcoef(b, m)[0, 1]
        print(f"{k:9s} n={len(m)} mean miss {m.mean():.1f}  base med {np.median(b):.1f}  corr(|base|,miss) {rho:.2f}  "
              f"|along| {np.median(np.abs(r['miss_along_mm'])):.1f} |cross| {np.median(np.abs(r['miss_cross_mm'])):.1f} "
              f"|up| {np.median(np.abs(r['miss_up_mm'])):.1f}  catch {r['success'].mean():.3f}")


# ---------------------------------------------------------------- expert-parking counterfactual
def fig_handoff():
    path = STUDY / 'report_handoff.json'
    if not path.exists():
        print('report_handoff.json not ready; skipped')
        return
    rep = json.loads(path.read_text())
    keys = [k for k in ('baseline', 'winner', 'dart', 'offset') if k in rep['models']]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.9), gridspec_kw=dict(wspace=0.5))
    w = 0.36
    for i, k in enumerate(keys):
        role, name = FINAL[k]
        own, own_c = seed_stats(runs(role, name))
        ho = np.array([s['mean_miss_mm'] for s in rep['models'][k]])
        ho_c = np.array([s['catch_rate'] * 100 for s in rep['models'][k]])
        for ax, a, b in ((a1, own, ho), (a2, own_c, ho_c)):
            ax.bar(i - w / 2, a.mean(), width=w * 0.92, color=C[k])
            ax.bar(i + w / 2, b.mean(), width=w * 0.92, color='white', edgecolor=C[k], hatch='////', linewidth=0.9)
            ax.scatter([i - w / 2] * 3, a, s=4, color=INK, zorder=3, lw=0)
            ax.scatter([i + w / 2] * 3, b, s=4, color=INK, zorder=3, lw=0)
        a1.text(i + w / 2, ho.max() + 3, f'{ho.mean():.0f}', ha='center', fontsize=5.6, color=INK2)
        a1.text(i - w / 2, own.max() + 3, f'{own.mean():.0f}', ha='center', fontsize=5.6, color=INK2)
        print(f'{k:9s} own {own.mean():.1f} mm / {own_c.mean():.1f}%   expert-parked {ho.mean():.1f} mm / {ho_c.mean():.1f}%  '
              f'median {np.median(np.concatenate([s["miss_mm"] for s in rep["models"][k]])):.1f}')
    for ax in (a1, a2):
        ax.set_xticks(range(len(keys)), [dict(baseline='10 Hz', winner='25 Hz', dart='DART', offset='Offset')[k] for k in keys],
                      fontsize=6, rotation=35, ha='right', rotation_mode='anchor')
        ax.grid(axis='x', visible=False)
    a1.set_ylabel('Mean miss (mm)')
    a1.set_ylim(0, 130)
    a2.set_ylabel('Catch within 10 mm (%)')
    a2.set_ylim(0, 105)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc='#8f8e88', label='learner drives the whole episode'),
                        Patch(fc='white', ec='#8f8e88', hatch='////',
                              label=f"expert drives and parks; learner takes the last {rep['tau_s']:.0f} s")],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.8, handlelength=1.6)
    panel(a1, 'a', 'Miss')
    panel(a2, 'b', 'Catch rate')
    save(fig, 'fig_handoff')


# ---------------------------------------------------------------- winner driving behaviour
def fig_driving():
    z = np.load(STUDY / 'plots' / 'figure_data.npz')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.7), gridspec_kw=dict(wspace=0.5))
    prog = z['progress']
    cross = z['path_cross_mm'].reshape(-1, len(prog))
    lo, med, hi = np.nanpercentile(cross, [25, 50, 75], axis=0)
    a1.fill_between(prog, lo, hi, color=C['winner'], alpha=0.25, lw=0)
    a1.plot(prog, med, color=C['winner'], lw=1.3)
    a1.axhline(0, color=INK, lw=0.6)
    a1.set_xlabel('Progress along expert route')
    a1.set_ylabel('Cross-track offset (mm)')
    a1.set_ylim(-40, 40)
    panel(a1, 'a', 'Path: median, IQR')
    a2.plot(z['brake_t_s'], z['brake_expert_mm_s'], color=INK, lw=1.2, label='expert, learner state')
    a2.plot(z['brake_t_s'], z['brake_learner_mm_s'], color=C['winner'], lw=1.3, label='25 Hz learner')
    a2.axvline(0, color=INK2, lw=0.6, ls=':')
    a2.set_xlabel('Time from expert stop (s)')
    a2.set_ylabel('Forward cmd (mm/s)')
    a2.set_xlim(-2, 5)
    a2.legend(loc='upper right', fontsize=5.4, handlelength=1.3)
    panel(a2, 'b', 'Braking')
    save(fig, 'fig_driving')
    lag = z['brake_lag_s']
    lag = lag[np.isfinite(lag)]
    print('brake lag n', len(lag), 'median', np.median(lag), 'q', np.percentile(lag, [25, 75]))


def main():
    style()
    fig_baseline()
    fig_sweep()
    fig_progression()
    fig_dart_demos()
    fig_driving()
    fig_handoff()
    roll = load_roll()
    if roll:
        fig_transfer(roll)
        fig_remaining(roll)
    else:
        print('report_rollouts.npz not ready; skipped transfer/remaining figures')


if __name__ == '__main__':
    main()
