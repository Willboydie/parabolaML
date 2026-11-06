"""Expert-parking counterfactual for the report.

For each model, the expert drives and parks the base, and the learned policy
takes over TAU seconds before the expert's own catch time. Comparing that miss
with the model's own closed-loop miss on the same scenes isolates how much of the
error comes from the learner's driving and parking. Same 200 test scenes, three
seeds per model.

Output: results/rel_study/report_handoff.json
"""
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import run_relative  # noqa: E402
from run_relative import init_env, init_model, pool_map  # noqa: E402

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'results' / 'rel_study'
OUT = STUDY / 'report_handoff.json'
TAU = 2.0
MODELS = {
    'baseline': ('sweep', 'r10_p2_e0.5_hnone_d1000_rel_add'),
    'winner': ('sweep', 'r25_p2_e0.5_hnone_d1000_rel_add'),
    'dart': ('dart', 'r25_p2_e0.5_hnone_d2000_rel_add'),
    'offset': ('offset', 'r25_p2_e0.5_hnone_d2000_rel_add'),
}


def catch_job(seed):
    row = run_relative.expert_episode(seed)
    return dict(seed=row['seed'], catch_time=row['catch_time'])


def handoff_job(seed, catch_time):
    from catch_env import ExpertDriver, rollout
    from diagnose import HandoffDriver
    from np_policy import PolicyDriver
    model = run_relative._model
    t_switch = max(0.05, float(catch_time) - TAU)
    driver = HandoffDriver(ExpertDriver(), PolicyDriver(model), t_switch)
    out = rollout(run_relative._env, driver, int(seed), model.cfg.rate_hz, stop_on_catch=True)
    miss = out['min_dist']
    return dict(seed=int(seed), success=bool(out['success']),
                miss_mm=None if miss is None or not np.isfinite(miss) else float(miss) * 1000)


def main():
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    t0 = time.time()
    rows = pool_map(catch_job, [(int(s),) for s in test], init_env, label='expert', workers=22)
    catch = {r['seed']: r['catch_time'] for r in rows if r['catch_time'] is not None}
    print(f'expert catches {len(catch)}/{len(test)}', flush=True)
    report = dict(tau_s=TAU, n_scenes=len(catch), models={})
    for name, (role, cfg) in MODELS.items():
        per_seed = []
        for seed in (0, 1, 2):
            weights = STUDY / 'models' / role / f'{cfg}_s{seed}.npz'
            out = pool_map(handoff_job, [(s, catch[s]) for s in catch], init_model, (str(weights),),
                           f'{name}_s{seed}', workers=22)
            miss = [r['miss_mm'] for r in out if r['miss_mm'] is not None]
            per_seed.append(dict(seed=seed, mean_miss_mm=float(np.mean(miss)), median_miss_mm=float(np.median(miss)),
                                 catch_rate=float(np.mean([r['success'] for r in out])),
                                 within_50=float(np.mean(np.asarray(miss) < 50)), miss_mm=miss))
            print(f'{name} s{seed}: mean {np.mean(miss):.1f} mm  median {np.median(miss):.1f}  '
                  f'catch {per_seed[-1]["catch_rate"]:.3f}  ({time.time() - t0:.0f} s)', flush=True)
        report['models'][name] = per_seed
        OUT.write_text(json.dumps(report))
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
