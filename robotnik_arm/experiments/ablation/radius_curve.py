"""Catch rate against radius for the DeepSets one-ball ceiling.

The closest-approach distance is recorded once per episode. A radius catches the
episode when that distance is smaller than the radius. The ceiling radius is the
smallest one, from 10 mm to 200 mm in 5 mm steps, whose mean catch rate across
the three seeds is over 80%.
"""
import json
from pathlib import Path

import numpy as np

from parking_split import init_model, policy_episode, pool_map

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
TAG = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
RADII = list(range(10, 201, 5))


def rates(rows):
    dist = np.array([np.inf if r['miss_mm'] is None else r['miss_mm'] for r in rows])
    return {str(radius): float(np.mean(dist < radius)) for radius in RADII}


def main():
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    report = {}
    curves = []
    for seed in (0, 1, 2):
        weights = RESULTS / 'models' / f'{TAG}_s{seed}.npz'
        print(f'rollout seed {seed}', flush=True)
        rows = pool_map(policy_episode, [(int(s),) for s in test], init_model, (str(weights),), f's{seed}')
        curve = rates(rows)
        dists = [r['miss_mm'] for r in rows if r['miss_mm'] is not None]
        report[f's{seed}'] = dict(curve=curve, miss_mm=float(np.mean(dists)),
                                  median_miss_mm=float(np.median(dists)),
                                  distances_mm=[None if r['miss_mm'] is None else r['miss_mm'] for r in rows])
        curves.append(curve)
        print(f"s{seed}: 10 mm {curve['10']:.3f}  40 mm {curve['40']:.3f}  "
              f"80 mm {curve['80']:.3f}  120 mm {curve['120']:.3f}", flush=True)
    mean_curve = {str(radius): float(np.mean([c[str(radius)] for c in curves])) for radius in RADII}
    chosen = next((radius for radius in RADII if mean_curve[str(radius)] > 0.80), None)
    report['mean_curve'] = mean_curve
    report['radius_mm'] = chosen
    report['separation_mm'] = None if chosen is None else 4 * chosen
    if chosen is not None:
        report['ceiling_catch_rate'] = mean_curve[str(chosen)]
        report['ceiling_catch_at_10mm'] = mean_curve['10']
    out = RESULTS / 'stage2' / 'radius_curve.json'
    out.write_text(json.dumps(report))
    print(json.dumps(dict(radius_mm=chosen, separation_mm=report['separation_mm'],
                          ceiling=report.get('ceiling_catch_rate'), at_10mm=mean_curve['10']), indent=2),
          flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
