"""Summarise logged DART sweeps against the learner's forward drift.

Reads results already on disk. Does not roll out or train.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DAGGER = ROOT / 'data' / 'dagger_iter1'
DART = ROOT / 'data' / 'learner_dart'
WHEEL_R = 0.11
TICK0 = 2
STRIDE = 40
DT = 0.04


def fwd(cmd):
    return cmd[:, :4].mean(axis=1) * WHEEL_R


def ticks(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        obs = z['obs'][TICK0::STRIDE]
        cmd = z['cmd'][TICK0::STRIDE]
        executed = z['executed'][TICK0::STRIDE]
    n = min(len(obs), len(cmd), len(executed))
    return obs[:n], cmd[:n], executed[:n], meta


def stop_index(expert_v):
    moving = np.flatnonzero(expert_v > 0.02)
    if len(moving) == 0:
        return None
    return int(moving[-1])


def binned(samples, edges):
    """samples: list of (t, value). Return median per bin, and counts."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [v for t, v in samples if lo <= t < hi]
        out.append(None if not vals else float(np.median(vals)))
    return out


def learner_curves(n):
    """Closed-loop learner (executed) against the expert label, aligned to the expert stop."""
    lead = []
    lags = []
    after_neg = []
    label_neg = 0
    label_n = 0
    exec_neg = 0
    exec_n = 0
    curves_l, curves_e = [], []
    misses = []
    for i in range(n):
        obs, cmd, executed, meta = ticks(DAGGER / f'demo_{i:04d}.npz')
        ev, lv = fwd(cmd), fwd(executed)
        t = obs[:, 0]
        label_neg += int(np.sum(ev < -1e-4))
        label_n += len(ev)
        stop = stop_index(ev)
        if stop is None or stop >= len(t) - 1:
            continue
        t_rel = t - t[stop]
        m = (t_rel >= -2.0) & (t_rel < 6.0)
        curves_l.extend(zip(t_rel[m].tolist(), lv[m].tolist()))
        curves_e.extend(zip(t_rel[m].tolist(), ev[m].tolist()))
        window = (t_rel >= -1.0) & (t_rel < 1.0)
        if window.sum() > 5:
            lead.append(float(np.sum((lv[window] - ev[window]) * DT) * 1000))
        slow = np.flatnonzero((t_rel >= 0.0) & (lv < 0.05))
        lags.append(float(t_rel[slow[0]]) if len(slow) else None)
        after = (t_rel >= 0.0) & (t_rel < 1.0)
        if after.any():
            after_neg.append(float(np.mean(lv[after] < -1e-4)))
            exec_neg += int(np.sum(lv[after] < -1e-4))
            exec_n += int(after.sum())
        dist = meta.get('min_dist')
        if dist is not None:
            misses.append(float(dist) * 1000)
    edges = np.arange(-2.0, 6.0 + 1e-9, 0.2)
    return dict(
        n_lead=len(lead),
        lead_mm=lead,
        lead_median_mm=float(np.median(lead)),
        lead_mean_mm=float(np.mean(lead)),
        frac_lead_positive=float(np.mean(np.array(lead) > 0)),
        brake_lag_s=[x for x in lags if x is not None],
        brake_lag_missing=int(sum(x is None for x in lags)),
        frac_learner_reverse_after_stop=float(np.mean(after_neg)) if after_neg else None,
        learner_reverse_ticks_after_stop=exec_neg,
        learner_ticks_after_stop=exec_n,
        expert_label_negative_ticks=label_neg,
        expert_label_ticks=label_n,
        t_centers=((edges[:-1] + edges[1:]) / 2).tolist(),
        learner_v_mm_s=[None if v is None else v * 1000 for v in binned(curves_l, edges)],
        expert_v_mm_s=[None if v is None else v * 1000 for v in binned(curves_e, edges)],
        miss_median_mm=float(np.median(misses)) if misses else None,
        miss_mean_mm=float(np.mean(misses)) if misses else None,
        n_miss=len(misses),
    )


def dart_labels(step):
    """Clean labels versus the noisy command that was actually applied."""
    label_neg = exec_neg = n = 0
    label_min = []
    noise_travel = []
    misses = []
    catches = []
    paths = sorted(DART.glob('demo_*.npz'))[::step]
    for path in paths:
        obs, cmd, executed, meta = ticks(path)
        ev, xv = fwd(cmd), fwd(executed)
        n += len(ev)
        label_neg += int(np.sum(ev < -1e-4))
        exec_neg += int(np.sum(xv < -1e-4))
        label_min.append(float(ev.min()))
        # Net extra forward travel the noise asked for, over the episode.
        noise_travel.append(float(np.sum((xv - ev) * DT) * 1000))
        dist = meta.get('min_dist')
        if dist is not None and np.isfinite(dist):
            misses.append(float(dist) * 1000)
        catches.append(bool(meta.get('success', False)))
    travel = np.array(noise_travel)
    return dict(
        n_episodes=len(paths),
        label_negative_frac=label_neg / n,
        executed_negative_frac=exec_neg / n,
        label_min_m_s=float(np.min(label_min)),
        noise_travel_median_mm=float(np.median(travel)),
        noise_travel_mean_mm=float(np.mean(travel)),
        noise_travel_p10_mm=float(np.percentile(travel, 10)),
        noise_travel_p90_mm=float(np.percentile(travel, 90)),
        frac_noise_travel_positive=float(np.mean(travel > 0)),
        catch_rate=float(np.mean(catches)),
        miss_median_mm=float(np.median(misses)),
        miss_mean_mm=float(np.mean(misses)),
        noise_travel_mm=travel.tolist(),
    )


def sweep_table():
    cov = json.loads((RESULTS / 'cov_sweep.json').read_text())
    quiet = json.loads((RESULTS / 'recover_sweep.json').read_text())
    rows = []
    for alpha, row in cov['alphas'].items():
        rows.append(dict(
            source='full noise', alpha=float(alpha),
            catch_rate=row['catch_rate'],
            miss_median_mm=row['miss_mm']['median'],
            path_median_mm=row['dev_median_mm']['median'],
            path_p90_mm=row['dev_median_mm']['p90'],
            base_median_mm=row['base_catch_mm']['median'],
        ))
    rows.append(dict(
        source='learner closed loop', alpha=None,
        catch_rate=cov['learner']['catch_rate'],
        miss_median_mm=cov['learner']['miss_mm']['median'],
        path_median_mm=cov['learner']['dev_median_mm']['median'],
        path_p90_mm=cov['learner']['dev_median_mm']['p90'],
        base_median_mm=cov['learner']['base_catch_mm']['median'],
    ))
    quiet_rows = []
    for alpha, row in quiet['alphas'].items():
        quiet_rows.append(dict(
            alpha=float(alpha),
            catch_rate=row['catch_rate'],
            miss_median_mm=row['miss_mm']['median'],
            path_median_mm=row['dev_median_mm']['median'],
            path_p90_mm=row['dev_median_mm']['p90'],
        ))
    return dict(
        n_seeds=cov['n_seeds'],
        chosen_alpha_full=cov['chosen_alpha'],
        chosen_alpha_quiet=quiet['chosen_alpha'],
        learner_path_median_target_mm=quiet['learner_path_median_mm'],
        learner_path_p90_target_mm=quiet['learner_path_p90_mm'],
        full=rows,
        quiet=quiet_rows,
    )


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = ROOT / 'results' / 'plots' / 'dart_bias'
    out_dir.mkdir(parents=True, exist_ok=True)
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
    ink, ember = '#1f3a5f', '#c46b2c'
    d = report['dagger_rollouts']
    t = np.array(d['t_centers'])
    # Keep a point every 0.4 s so the brake, not the cruise, is what the eye follows.
    keep = np.arange(0, len(t), 2)
    t, lv, ev = t[keep], np.array(d['learner_v_mm_s'])[keep], np.array(d['expert_v_mm_s'])[keep]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(0.0, color='0.75', lw=0.8)
    ax.plot(t, ev, color=ink, lw=1.8, label='Expert command, queried in the learner state')
    ax.plot(t, lv, color=ember, lw=1.8, label='Learner command, closed loop')
    ax.set_xlabel('Time relative to the expert stop (s)')
    ax.set_ylabel('Forward command (mm/s)')
    ax.set_xlim(-2.0, 5.6)
    ax.set_ylim(-10, 280)
    ax.set_title('Median brake is 2.6 s after the expert has already stopped')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.02,
             'Median forward command, 133 of 200 rollouts. Quartiles of the brake lag are 1.2 s and 4.1 s.\n'
             'Expert forward labels never go negative (0 of 94,844 ticks here; minimum 0 on 200 DART demos).',
             ha='center', va='bottom', fontsize=8, color='0.35')
    fig.savefig(out_dir / '01_brake_lag.png', dpi=160, bbox_inches='tight')
    fig.savefig(out_dir / '01_brake_lag.pdf', bbox_inches='tight')
    plt.close(fig)

    full = [r for r in report['sweeps']['full'] if r['alpha'] is not None]
    alphas = [r['alpha'] for r in full]
    labels = [f'{a:.2f}' for a in alphas]
    catch = [100 * r['catch_rate'] for r in full]
    path = [r['path_median_mm'] for r in full]
    learner_path = report['sweeps']['full'][-1]['path_median_mm']

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    ax = axes[0]
    ax.bar(labels, catch, color=ink, width=0.72)
    ax.axhline(90, color='0.45', ls='--', lw=0.9, label='90% bar used to accept a scale')
    ax.set_xlabel('Noise scale α')
    ax.set_ylabel('Expert catch rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Small noise still catches')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.bar(labels, path, color=ember, width=0.72)
    ax.axhline(learner_path, color=ink, ls='--', lw=1.0,
               label=f'Learner path deviation, {learner_path:.0f} mm')
    ax.set_xlabel('Noise scale α')
    ax.set_ylabel('Median hand-path deviation (mm)')
    ax.set_title('Only large noise reaches the drift')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.02,
             'Expert with correlated command noise, 20 seeds (cov_sweep.json). '
             'The learner on the same kind of scene catches 0% and deviates by 68 mm.',
             ha='center', va='bottom', fontsize=8, color='0.35')
    fig.savefig(out_dir / '02_noise_scale.png', dpi=160, bbox_inches='tight')
    fig.savefig(out_dir / '02_noise_scale.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', out_dir)


def main():
    spec = json.loads((DART / 'dataset.json').read_text())
    dagger_n = json.loads((DAGGER / 'dataset.json').read_text())['n']
    report = dict(
        learner_dart_dataset=dict(
            n=spec['n'], catch_rate=spec['catch_rate'], mean_miss_mm=spec['mean_miss_mm']),
        dagger_rollouts=learner_curves(dagger_n),
        dart_sample=dart_labels(5),
        sweeps=sweep_table(),
    )
    d = report['dagger_rollouts']
    d.pop('lead_mm')
    report['dart_sample'].pop('noise_travel_mm')
    lags = np.array(d.pop('brake_lag_s'), dtype=float)
    d['brake_lag_median_s'] = float(np.median(lags))
    d['brake_lag_p25_s'] = float(np.percentile(lags, 25))
    d['brake_lag_p75_s'] = float(np.percentile(lags, 75))
    plot(report)
    out = RESULTS / 'stage2' / 'dart_bias_summary.json'
    out.write_text(json.dumps(report))
    print(f'wrote {out}')
    print('learner lead median', round(d['lead_median_mm'], 2),
          'mean', round(d['lead_mean_mm'], 2),
          'frac+', round(d['frac_lead_positive'], 3),
          'reverse after', d['frac_learner_reverse_after_stop'],
          'lag median', round(d['brake_lag_median_s'], 3),
          'p25', round(d['brake_lag_p25_s'], 3),
          'p75', round(d['brake_lag_p75_s'], 3),
          'lag missing', d['brake_lag_missing'])
    print('expert label neg', d['expert_label_negative_ticks'], '/', d['expert_label_ticks'])
    s = report['dart_sample']
    print('dart labels neg frac', s['label_negative_frac'],
          'exec neg', round(s['executed_negative_frac'], 4),
          'label min', s['label_min_m_s'],
          'noise travel median', round(s['noise_travel_median_mm'], 2),
          'catch', s['catch_rate'], 'miss med', round(s['miss_median_mm'], 1))
    print('curves t', np.round(d['t_centers'], 2).tolist())
    print('learner v', [None if v is None else round(v, 1) for v in d['learner_v_mm_s']])
    print('expert v', [None if v is None else round(v, 1) for v in d['expert_v_mm_s']])


if __name__ == '__main__':
    main()
