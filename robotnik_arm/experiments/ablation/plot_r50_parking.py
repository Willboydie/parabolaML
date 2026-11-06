"""Four views of the same parking-error sample: model base minus expert base."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results' / 'stage2' / 'r50_parking_errors.npz'
OUT = ROOT / 'results' / 'plots' / 'r50_parking'
MODEL = 'r50_p1_e0.5_hstack0.24s_d2000_deepsets'

INK = '#1f3a5f'
SEED_COLOR = {0: '#1f3a5f', 1: '#c46b2c', 2: '#2f6f4e'}
EXPERT = '#111111'


def summarize(z):
    along, cross, radial = z['along_mm'], z['cross_mm'], z['radial_mm']
    seeds = z['model_seed']
    def q(a):
        return {name: float(np.percentile(a, p)) for name, p in
                (('p10', 10), ('p25', 25), ('p50', 50), ('p75', 75), ('p90', 90))}
    per_seed = {}
    for seed in (0, 1, 2):
        m = seeds == seed
        per_seed[str(seed)] = dict(
            n=int(m.sum()),
            radial_median_mm=float(np.median(radial[m])),
            along_median_mm=float(np.median(along[m])),
            cross_abs_median_mm=float(np.median(np.abs(cross[m]))),
            frac_ahead=float(np.mean(along[m] > 0)),
        )
    return dict(
        n=int(len(along)),
        n_scenes=int(len(np.unique(z['scene']))),
        radial_median_mm=float(np.median(radial)),
        radial_mean_mm=float(np.mean(radial)),
        along_median_mm=float(np.median(along)),
        along_mean_mm=float(np.mean(along)),
        cross_median_mm=float(np.median(cross)),
        cross_abs_median_mm=float(np.median(np.abs(cross))),
        along_abs_median_mm=float(np.median(np.abs(along))),
        frac_ahead=float(np.mean(along > 0)),
        heading_abs_median_deg=float(np.median(np.abs(z['heading_deg']))),
        quantiles=dict(radial_mm=q(radial), along_mm=q(along), cross_mm=q(cross)),
        per_seed=per_seed,
    )


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#d0d5dd',
        'grid.linewidth': 0.6,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.axisbelow': True,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


FRAME_MM = 150


def _outside(values):
    return int(np.sum(np.abs(values) > FRAME_MM))


def scatter(z):
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    shown = np.ones(len(z['along_mm']), dtype=bool)
    for seed in (0, 1, 2):
        m = (z['model_seed'] == seed) & (np.abs(z['along_mm']) <= FRAME_MM) & (np.abs(z['cross_mm']) <= FRAME_MM)
        shown &= ~((z['model_seed'] == seed) & ~m)
        ax.scatter(z['along_mm'][m], z['cross_mm'][m], s=22, alpha=0.55, c=SEED_COLOR[seed],
                   edgecolors='none', label=f'Seed {seed}', zorder=2)
    hidden = int((~shown).sum())
    ax.axhline(0, color=EXPERT, lw=0.8, zorder=1)
    ax.axvline(0, color=EXPERT, lw=0.8, zorder=1)
    along_med = float(np.median(z['along_mm']))
    ax.axvline(along_med, color='#c46b2c', lw=1.0, ls='--', zorder=1)
    ax.scatter([0], [0], s=80, c=EXPERT, marker='+', linewidths=1.6, label='Expert', zorder=4)
    ax.annotate(f'median {along_med:+.1f} mm', xy=(along_med, FRAME_MM - 8),
                xytext=(8, 0), textcoords='offset points', color='#c46b2c', fontsize=10, va='top')
    ax.set_xlim(-FRAME_MM, FRAME_MM)
    ax.set_ylim(-FRAME_MM, FRAME_MM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Along-track parking error (mm)\npositive = ahead of the expert')
    ax.set_ylabel('Cross-track parking error (mm)\npositive = left of the expert route')
    ax.set_title('Where the base stops, relative to the expert')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=(0, 0.08, 0.86, 1))
    fig.text(0.43, 0.02, f'{hidden} of 600 episodes lie outside ±{FRAME_MM:.0f} mm and are omitted here. '
             f'Furthest radial error {np.max(z["radial_mm"]):.0f} mm.',
             ha='center', fontsize=9, color='#444444')
    save(fig, '01_scatter')


def histograms(z):
    bins = np.arange(-FRAME_MM, FRAME_MM + 1, 10)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    panels = (
        (axes[0], z['along_mm'], 'Along-track parking error (mm)'),
        (axes[1], z['cross_mm'], 'Cross-track parking error (mm)'),
    )
    for ax, values, xlabel in panels:
        clipped = values[np.abs(values) <= FRAME_MM]
        ax.hist(clipped, bins=bins, color=INK, alpha=0.9, edgecolor='white', linewidth=0.4)
        ax.axvline(0, color=EXPERT, lw=1.4, label='Expert (0 mm)')
        med = float(np.median(values))
        if abs(med) >= 8:
            ax.axvline(med, color='#c46b2c', lw=1.3, ls='--', label=f'Median {med:+.1f} mm')
            ax.legend(frameon=False, loc='upper left')
        else:
            ax.legend(frameon=False, loc='upper left')
            ax.set_title(f'Median {med:+.1f} mm', fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-FRAME_MM, FRAME_MM)
        omitted = _outside(values)
        ax.text(0.98, 0.98, f'{omitted} outside the axis', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='#444444')
    axes[0].set_ylabel('Episodes')
    fig.suptitle('Same 600 episodes, split into route-frame components', y=1.03)
    save(fig, '02_histograms')


def ecdf(z):
    radial = np.sort(z['radial_mm'])
    y = np.arange(1, len(radial) + 1) / len(radial)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.step(radial, y, where='post', color=INK, lw=1.8, label='Learned base')
    med = float(np.median(radial))
    ax.axvline(med, color='#c46b2c', lw=1.1, ls='--', label=f'Median {med:.1f} mm')
    ax.scatter([0], [0], s=36, c=EXPERT, zorder=4, label='Expert (0 mm on every episode)')
    at = float(np.mean(radial <= FRAME_MM))
    ax.set_xlim(0, FRAME_MM)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Parking distance from the expert (mm)')
    ax.set_ylabel('Fraction of episodes')
    ax.set_title('How far the learned base parks from the expert')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.5, 0.02, f'At {FRAME_MM:.0f} mm the curve is at {at:.1%}. '
             f'{int(np.sum(radial > FRAME_MM))} episodes continue past the axis, to {radial[-1]:.0f} mm.',
             ha='center', fontsize=9, color='#444444')
    save(fig, '03_ecdf')


def ranked(z):
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    series = (
        (axes[0], np.sort(z['along_mm']), 'Along-track error (mm)',
         'ordered from furthest behind to furthest ahead'),
        (axes[1], np.sort(z['cross_mm']), 'Cross-track error (mm)',
         'ordered from furthest right to furthest left'),
    )
    x = np.linspace(0, 100, len(z['along_mm']))
    for ax, values, ylabel, order in series:
        shown = np.clip(values, -FRAME_MM, FRAME_MM)
        ax.fill_between(x, shown, 0, where=shown >= 0, color=INK, alpha=0.25, interpolate=True)
        ax.fill_between(x, shown, 0, where=shown < 0, color='#9aa3b2', alpha=0.45, interpolate=True)
        ax.plot(x, shown, color=INK, lw=1.3)
        ax.axhline(0, color=EXPERT, lw=1.1, label='Expert')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-FRAME_MM, FRAME_MM)
        ax.text(0.99, 0.04, f'{order}\nends {values[0]:+.0f} mm and {values[-1]:+.0f} mm',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='#444444')
    axes[0].legend(frameon=False, loc='upper left')
    axes[1].set_xlabel('Percentile of the 600 episodes')
    axes[0].set_title('Signed parking error, ordered across the 600 episodes')
    save(fig, '04_ranked')


def main():
    style()
    with np.load(DATA) as z:
        arrays = {key: np.array(z[key]) for key in z.files}
    stats = summarize(arrays)
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(stats, indent=2))
    scatter(arrays)
    histograms(arrays)
    ecdf(arrays)
    ranked(arrays)
    print(json.dumps(stats, indent=2))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
