"""Per-episode closest-approach geometry for the report figures.

Re-runs the 200 test scenes for the 10 Hz baseline, the 25 Hz winner, and the
two 2,000-episode retrains (high-amplitude DART and parking offsets), three
seeds each, and stores every episode's base offset and hand miss in the
expert's route frame. three_models.json only kept medians; the report figures
need the per-episode scatter.

Output: results/rel_study/report_rollouts.npz
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from run_relative import init_model, policy_episode, pool_map, route_axes  # noqa: E402
from parking_split import load_experts  # noqa: E402

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'results' / 'rel_study'
OUT = STUDY / 'report_rollouts.npz'
MODELS = {
    'baseline': ('sweep', 'r10_p2_e0.5_hnone_d1000_rel_add'),
    'winner': ('sweep', 'r25_p2_e0.5_hnone_d1000_rel_add'),
    'dart': ('dart', 'r25_p2_e0.5_hnone_d2000_rel_add'),
    'offset': ('offset', 'r25_p2_e0.5_hnone_d2000_rel_add'),
}
SEEDS = (0, 1, 2)
FIELDS = ('scene', 'model_seed', 'success', 'miss_mm', 'base_along_mm', 'base_cross_mm',
          'miss_along_mm', 'miss_cross_mm', 'miss_up_mm', 't_best_s')


def geometry(rows, experts, model_seed):
    out = {k: [] for k in FIELDS}
    for row in rows:
        best = row['best']
        if best is None:
            continue
        ex = experts[row['seed']]
        k = min(best['step'], len(ex['base']) - 1)
        along, cross = route_axes(ex['chord'], float(ex['bearing'][k]))
        delta = np.asarray(best['base'], float) - np.asarray(ex['base'][k], float)
        miss = np.asarray(best['ee'], float) - np.asarray(best['ball'], float)
        out['scene'].append(int(row['seed']))
        out['model_seed'].append(model_seed)
        out['success'].append(bool(row['success']))
        out['miss_mm'].append(float(row['miss_mm']))
        out['base_along_mm'].append(float(np.dot(delta, along) * 1000))
        out['base_cross_mm'].append(float(np.dot(delta, cross) * 1000))
        out['miss_along_mm'].append(float(np.dot(miss[:2], along) * 1000))
        out['miss_cross_mm'].append(float(np.dot(miss[:2], cross) * 1000))
        out['miss_up_mm'].append(float(miss[2] * 1000))
        out['t_best_s'].append(best['step'] * 0.001)
    return out


def main():
    only = sys.argv[1:] or list(MODELS)
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    t0 = time.time()
    print(f'expert reference on {len(test)} scenes', flush=True)
    experts = load_experts(test)
    pack = {}
    if OUT.exists():
        with np.load(OUT) as z:
            pack = {k: z[k] for k in z.files}
    for name in only:
        role, cfg = MODELS[name]
        merged = {k: [] for k in FIELDS}
        for seed in SEEDS:
            weights = STUDY / 'models' / role / f'{cfg}_s{seed}.npz'
            print(f'{name} s{seed}  ({time.time() - t0:.0f} s)', flush=True)
            rows = pool_map(policy_episode, [(int(s),) for s in test], init_model, (str(weights),),
                            f'{name}_s{seed}', workers=22)
            g = geometry(rows, experts, seed)
            for k in FIELDS:
                merged[k].extend(g[k])
            print(f'  mean miss {np.mean(g["miss_mm"]):.1f} mm  catch {np.mean(g["success"]):.3f}', flush=True)
        for k in FIELDS:
            pack[f'{name}__{k}'] = np.asarray(merged[k])
        np.savez(OUT, **pack)
        print(f'saved {name} -> {OUT}', flush=True)
    print(f'done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
