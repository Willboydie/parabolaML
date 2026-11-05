"""Base parking and the along/cross/vertical split of the miss.

Measured at the closest approach on the 200 test seeds, against the expert at
the same time. The horizontal base distance is the parking offset. The miss is
the end effector minus the ball, in the expert's route frame plus height.
"""
import json
from pathlib import Path

import numpy as np

from run_relative import (
    compensation, expert_episode, init_env, init_model, policy_episode, pool_map,
    route_axes, summarize_miss,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
CURRENT = [
    RESULTS / 'models' / f'r25_p2_e0.5_hstack0.5s_d2000_rel_add_s{seed}.npz'
    for seed in (0, 1, 2)
]
STAGE1 = RESULTS / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'


def miss_components(policy_rows, experts):
    """Median pieces of the miss, and the horizontal base offset, in millimetres."""
    base_n, along_b, cross_b = [], [], []
    along_m, cross_m, up_m, miss3 = [], [], [], []
    for row in policy_rows:
        best = row['best']
        if best is None:
            continue
        ex = experts[row['seed']]
        k = min(best['step'], len(ex['base']) - 1)
        along, cross = route_axes(ex['chord'], float(ex['bearing'][k]))
        delta = np.asarray(best['base'], float) - np.asarray(ex['base'][k], float)
        miss = np.asarray(best['ee'], float) - np.asarray(best['ball'], float)
        base_n.append(float(np.linalg.norm(delta) * 1000))
        along_b.append(float(np.dot(delta, along) * 1000))
        cross_b.append(float(np.dot(delta, cross) * 1000))
        along_m.append(float(np.dot(miss[:2], along) * 1000))
        cross_m.append(float(np.dot(miss[:2], cross) * 1000))
        up_m.append(float(miss[2] * 1000))
        miss3.append(float(np.linalg.norm(miss) * 1000))
    base_n = np.asarray(base_n)
    along_b, cross_b = np.asarray(along_b), np.asarray(cross_b)
    along_m, cross_m, up_m = np.asarray(along_m), np.asarray(cross_m), np.asarray(up_m)
    energy = along_m ** 2 + cross_m ** 2 + up_m ** 2
    total = float(np.mean(energy))
    up_abs = np.abs(up_m)
    return dict(
        n=int(len(base_n)),
        base_offset_median_mm=float(np.median(base_n)),
        base_offset_p90_mm=float(np.percentile(base_n, 90)),
        base_along_median_mm=float(np.median(along_b)),
        base_along_mean_mm=float(np.mean(along_b)),
        base_along_abs_median_mm=float(np.median(np.abs(along_b))),
        base_cross_abs_median_mm=float(np.median(np.abs(cross_b))),
        frac_ahead=float(np.mean(along_b > 0)),
        miss_along_median_mm=float(np.median(along_m)),
        miss_along_abs_median_mm=float(np.median(np.abs(along_m))),
        miss_cross_abs_median_mm=float(np.median(np.abs(cross_m))),
        miss_up_abs_median_mm=float(np.median(up_abs)),
        miss_3d_mean_mm=float(np.mean(miss3)),
        miss_3d_median_mm=float(np.median(miss3)),
        frac_up_at_least_10mm=float(np.mean(up_abs >= 10.0)),
        energy_share=dict(
            along=float(np.mean(along_m ** 2) / total),
            cross=float(np.mean(cross_m ** 2) / total),
            up=float(np.mean(up_m ** 2) / total),
        ),
    )


def expert_job(seed):
    return expert_episode(seed)


def load_experts(seeds):
    rows = pool_map(expert_job, [(int(s),) for s in seeds], init_env, label='expert')
    return {r['seed']: r for r in rows}


def eval_weights(weights, seeds, experts):
    rows = pool_map(policy_episode, [(int(s),) for s in seeds], init_model, (str(weights),), Path(weights).stem)
    stats = summarize_miss(rows)
    dists = [r['miss_mm'] for r in rows if r['miss_mm'] is not None]
    stats['catch_rate_50mm'] = float(np.mean([d < 50.0 for d in dists])) if dists else None
    stats['compensation'] = compensation(rows, experts)
    stats['split'] = miss_components(rows, experts)
    return stats


def aggregate(blocks):
    def col(path):
        vals = []
        for item in blocks:
            cur = item
            for part in path:
                cur = cur[part]
            vals.append(cur)
        return np.asarray(vals, float)

    paths = (
        ('miss_mm', ['miss_mm']),
        ('median_miss_mm', ['median_miss_mm']),
        ('catch_rate', ['catch_rate']),
        ('catch_rate_50mm', ['catch_rate_50mm']),
        ('base_offset_median_mm', ['split', 'base_offset_median_mm']),
        ('base_along_median_mm', ['split', 'base_along_median_mm']),
        ('frac_ahead', ['split', 'frac_ahead']),
        ('miss_along_abs_median_mm', ['split', 'miss_along_abs_median_mm']),
        ('miss_cross_abs_median_mm', ['split', 'miss_cross_abs_median_mm']),
        ('miss_up_abs_median_mm', ['split', 'miss_up_abs_median_mm']),
        ('frac_up_at_least_10mm', ['split', 'frac_up_at_least_10mm']),
        ('comp_along', ['compensation', 'along', 'regression']),
        ('comp_cross', ['compensation', 'cross', 'regression']),
    )
    out = {}
    for name, path in paths:
        vals = col(path)
        out[name] = float(np.mean(vals))
        out[name + '_sd'] = float(np.std(vals))
    return out


def main():
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    weights = [('stage1', STAGE1)] + [(f'd2000_s{i}', path) for i, path in enumerate(CURRENT)]
    missing = [path for _name, path in weights if not path.exists()]
    if missing:
        raise SystemExit('missing ' + ', '.join(str(p) for p in missing))
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    report = {}
    for name, path in weights:
        print(f'rollout {name}', flush=True)
        stats = eval_weights(path, test, experts)
        report[name] = stats
        split = stats['split']
        print(f"{name}: catch {stats['catch_rate']:.3f}  50mm {stats['catch_rate_50mm']:.3f}  "
              f"miss {stats['miss_mm']:.1f} mm  "
              f"base {split['base_offset_median_mm']:.0f} mm  "
              f"along {split['base_along_median_mm']:+.1f}  "
              f"ahead {split['frac_ahead']:.2f}  "
              f"miss along {split['miss_along_abs_median_mm']:.0f}  "
              f"cross {split['miss_cross_abs_median_mm']:.0f}  "
              f"up {split['miss_up_abs_median_mm']:.0f}", flush=True)
    report['d2000'] = aggregate([report[f'd2000_s{i}'] for i in range(3)])
    out = RESULTS / 'parking_split.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['d2000'], indent=2), flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
