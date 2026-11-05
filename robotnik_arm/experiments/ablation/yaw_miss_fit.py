"""Fit the sideways hand miss to the sideways base offset and the yaw error.

Same 50 learner episodes as the closest-approach comparison. Yaw error is the
learner's heading minus the expert's heading at that instant.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from closest_approach import N, init_worker, expert_at, load_learner, route_axes
from closest_approach import DEMOS

RESULTS_NAME = 'yaw_miss_fit.json'


def learner_bearing(index, step):
    with np.load(DEMOS / f'demo_{index:04d}.npz') as z:
        obs = z['obs']
    return float(np.arctan2(obs[step, 13], obs[step, 14]))


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fit(y, columns):
    x = np.column_stack([np.ones(len(y))] + columns)
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    return dict(intercept=float(coef[0]), slopes=[float(c) for c in coef[1:]],
                r2=1 - ss_res / ss_tot, ss_res=ss_res, ss_tot=ss_tot)


def main():
    from pathlib import Path
    learner = [r for r in load_learner() if r['ok']]
    for row in learner:
        row['bearing'] = learner_bearing(row['index'], row['step'])
    print(f'{len(learner)} episodes', flush=True)
    t0 = time.time()
    expert = {}
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        futs = {pool.submit(expert_at, r['seed'], r['step']): r['index'] for r in learner}
        for n, fut in enumerate(as_completed(futs), 1):
            expert[futs[fut]] = fut.result()
            if n % 10 == 0 or n == len(futs):
                print(f'  expert {n}/{len(futs)} ({time.time() - t0:.0f}s)', flush=True)

    miss_c, base_c, yaw = [], [], []
    for r in learner:
        ex = expert[r['index']]
        if not ex['reached']:
            continue
        along, cross = route_axes(ex)
        base_xy = r['base'][:2] - ex['base'][:2]
        miss_xy = (r['ee'] - r['ball'])[:2]
        miss_c.append(float(np.dot(miss_xy, cross) * 1000))
        base_c.append(float(np.dot(base_xy, cross) * 1000))
        yaw.append(float(np.rad2deg(wrap(r['bearing'] - ex['bearing']))))
    miss_c, base_c, yaw = map(np.asarray, (miss_c, base_c, yaw))
    base_only = fit(miss_c, [base_c])
    yaw_only = fit(miss_c, [yaw])
    both = fit(miss_c, [base_c, yaw])
    partial_yaw = (base_only['ss_res'] - both['ss_res']) / base_only['ss_res']
    partial_base = (yaw_only['ss_res'] - both['ss_res']) / yaw_only['ss_res']
    out = dict(
        n=int(len(miss_c)),
        yaw_deg=dict(median=float(np.median(yaw)), abs_median=float(np.median(np.abs(yaw))),
                     std=float(yaw.std())),
        corr_yaw_base=float(np.corrcoef(yaw, base_c)[0, 1]),
        base_only=dict(r2=base_only['r2'], slope_mm_per_mm=base_only['slopes'][0]),
        yaw_only=dict(r2=yaw_only['r2'], slope_mm_per_deg=yaw_only['slopes'][0]),
        both=dict(r2=both['r2'], base_mm_per_mm=both['slopes'][0], yaw_mm_per_deg=both['slopes'][1]),
        partial_r2_yaw_after_base=float(partial_yaw),
        partial_r2_base_after_yaw=float(partial_base),
    )
    path = Path(__file__).resolve().parent / 'results' / RESULTS_NAME
    path.write_text(json.dumps(out, indent=2))
    print(f"yaw error median {out['yaw_deg']['median']:+.2f}°  abs {out['yaw_deg']['abs_median']:.2f}°  "
          f"std {out['yaw_deg']['std']:.2f}°", flush=True)
    print(f"corr(yaw, base cross) {out['corr_yaw_base']:+.2f}", flush=True)
    print(f"R² base offset alone {base_only['r2']:.2f}   slope {base_only['slopes'][0]:+.2f} mm/mm", flush=True)
    print(f"R² yaw alone         {yaw_only['r2']:.2f}   slope {yaw_only['slopes'][0]:+.2f} mm/deg", flush=True)
    print(f"R² both              {both['r2']:.2f}   base {both['slopes'][0]:+.2f} mm/mm  "
          f"yaw {both['slopes'][1]:+.2f} mm/deg", flush=True)
    print(f"partial R² of yaw after base offset {partial_yaw:.2f}", flush=True)
    print(f"partial R² of base offset after yaw {partial_base:.2f}", flush=True)
    print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
