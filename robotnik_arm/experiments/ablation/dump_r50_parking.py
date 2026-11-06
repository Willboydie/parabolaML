"""Per-episode parking error of r50_p1_e0.5_hstack0.24s_d2000_deepsets vs the expert.

At the closest approach, the base offset is the learned base minus the expert
base, expressed in the expert's route frame. The expert's own parking error is
zero in that frame. Writes a compact array file; does not rescore the ledger.
"""
import json
import time
from pathlib import Path

import numpy as np

from measure_park_cloud import wrap
from parking_split import load_experts
from run_relative import init_model, policy_episode, pool_map, route_axes

ROOT = Path(__file__).resolve().parent
NAME = 'r50_p1_e0.5_hstack0.24s_d2000_deepsets'
OUT = ROOT / 'results' / 'stage2' / 'r50_parking_errors.npz'
SEEDS = (0, 1, 2)


def components(policy_rows, experts, model_seed):
    keys = ('scene', 'model_seed', 'along_mm', 'cross_mm', 'radial_mm', 'heading_deg',
            'miss_mm', 'miss_along_mm', 'miss_cross_mm', 'miss_up_mm')
    cols = {key: [] for key in keys}
    skipped = 0
    for row in policy_rows:
        best = row['best']
        if best is None or row['miss_mm'] is None:
            skipped += 1
            continue
        ex = experts[row['seed']]
        k = min(best['step'], len(ex['base']) - 1)
        ax_along, ax_cross = route_axes(ex['chord'], float(ex['bearing'][k]))
        delta = np.asarray(best['base'], float) - np.asarray(ex['base'][k], float)
        miss = np.asarray(best['ee'], float) - np.asarray(best['ball'], float)
        along = float(np.dot(delta, ax_along) * 1000)
        cross = float(np.dot(delta, ax_cross) * 1000)
        cols['scene'].append(int(row['seed']))
        cols['model_seed'].append(int(model_seed))
        cols['along_mm'].append(along)
        cols['cross_mm'].append(cross)
        cols['radial_mm'].append(float(np.hypot(along, cross)))
        cols['heading_deg'].append(float(wrap(best['bearing'] - float(ex['bearing'][k])) * 180.0 / np.pi))
        cols['miss_mm'].append(float(row['miss_mm']))
        cols['miss_along_mm'].append(float(np.dot(miss[:2], ax_along) * 1000))
        cols['miss_cross_mm'].append(float(np.dot(miss[:2], ax_cross) * 1000))
        cols['miss_up_mm'].append(float(miss[2] * 1000))
    packed = {key: np.asarray(cols[key], np.float64 if key not in ('scene', 'model_seed') else np.int64)
              for key in keys}
    packed['skipped'] = int(skipped)
    return packed


def main():
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    t0 = time.time()
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    parts = []
    for seed in SEEDS:
        weights = ROOT / 'results' / 'models' / f'{NAME}_s{seed}.npz'
        if not weights.exists():
            raise SystemExit(f'missing {weights}')
        print(f'rollout {weights.name}', flush=True)
        rows = pool_map(policy_episode, [(int(s),) for s in test], init_model, (str(weights),), weights.stem)
        part = components(rows, experts, seed)
        print(f"  s{seed}: n {len(part['along_mm'])} skipped {part['skipped']}  "
              f"radial median {np.median(part['radial_mm']):.1f} mm  "
              f"along median {np.median(part['along_mm']):+.1f} mm", flush=True)
        parts.append(part)
    packed = {key: np.concatenate([part[key] for part in parts])
              for key in ('scene', 'model_seed', 'along_mm', 'cross_mm', 'radial_mm', 'heading_deg',
                          'miss_mm', 'miss_along_mm', 'miss_cross_mm', 'miss_up_mm')}
    np.savez(OUT, **packed)
    print(f'wrote {OUT} n={len(packed["along_mm"])} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
