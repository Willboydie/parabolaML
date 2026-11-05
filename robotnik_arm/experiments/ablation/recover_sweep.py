"""Re-sweep correlated command noise with the last 3 s quiet.

Judged on recovered catch rate. Path deviation is still reported so a passing
alpha in 0.15–0.25 can be matched to the learner's drift. 20 seeds, no training.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from sweep_cov_noise import DT, TAU, spread

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
ALPHAS = (0.15, 0.20, 0.25)
QUIET_S = 3.0
EE_COL = 39
LEARNER_PATH_MED = 68.0   # mm, measured on the same kind of seeds
LEARNER_PATH_P90 = 146.0

_env = None
_cov = None


def init_worker(cov):
    global _env, _cov
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()
    _cov = np.asarray(cov, dtype=np.float64)


def sweep_job(seed, noise_seed, alphas):
    from catch_env import CorrelatedCommandNoise, ExpertDriver, rollout
    clean = rollout(_env, ExpertDriver(), int(seed), 25, stop_on_catch=False, log=True)
    ee_c = clean['arrays']['obs'][:, EE_COL:EE_COL + 3]
    out = dict(seed=int(seed), alphas={})
    for i, alpha in enumerate(alphas):
        noise = CorrelatedCommandNoise(int(noise_seed) + i, _cov, TAU, DT, alpha=float(alpha))
        noisy = rollout(_env, ExpertDriver(), int(seed), 25, stop_on_catch=False, log=True,
                        command_noise=noise, quiet_tail_s=QUIET_S)
        ee_n = noisy['arrays']['obs'][:, EE_COL:EE_COL + 3]
        n = min(len(ee_n), len(ee_c))
        dev = np.linalg.norm(ee_n[:n] - ee_c[:n], axis=1) * 1000
        miss = noisy['min_dist']
        out['alphas'][str(alpha)] = dict(
            success=bool(noisy['success']),
            miss_mm=None if not np.isfinite(miss) else float(miss * 1000),
            dev_median_mm=float(np.median(dev)),
            dev_p90_mm=float(np.percentile(dev, 90)),
            dev_max_mm=float(np.max(dev)))
    return out


def choose(summary):
    """Passing means recovered catch >= 90% inside the requested alpha band."""
    passing = []
    for alpha, row in summary.items():
        if float(alpha) < 0.15 - 1e-9 or float(alpha) > 0.25 + 1e-9:
            continue
        if row['catch_rate'] + 1e-9 < 0.90:
            continue
        med = row['dev_median_mm']['median']
        tail = row['dev_median_mm']['p90']
        score = abs(med - LEARNER_PATH_MED) / LEARNER_PATH_MED + abs(tail - LEARNER_PATH_P90) / LEARNER_PATH_P90
        passing.append((score, float(alpha)))
    if not passing:
        return None
    passing.sort()
    return passing[0][1]


def main():
    from gen_learner_dart import seed_stream
    cov = np.array(json.loads((RESULTS / 'learner_cov.json').read_text())['cov'])
    scene = seed_stream(32, 20)
    noise_seeds = seed_stream(33, 20)
    print(f'quiet-tail {QUIET_S:.0f}s sweep alphas {ALPHAS} on {len(scene)} seeds', flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(16, initializer=init_worker, initargs=(cov.tolist(),)) as pool:
        futs = [pool.submit(sweep_job, s, n, ALPHAS) for s, n in zip(scene, noise_seeds)]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 5 == 0 or n == len(scene):
                print(f'  {n}/{len(scene)} ({time.time() - t0:.0f}s)', flush=True)
    summary = {}
    for alpha in ALPHAS:
        got = [r['alphas'][str(alpha)] for r in rows]
        summary[str(alpha)] = dict(
            catch_rate=float(np.mean([g['success'] for g in got])),
            miss_mm=spread(got, 'miss_mm'),
            dev_median_mm=spread(got, 'dev_median_mm'),
            dev_p90_mm=spread(got, 'dev_p90_mm'),
            dev_max_mm=spread(got, 'dev_max_mm'))
    picked = choose(summary)
    report = dict(quiet_tail_s=QUIET_S, alphas=summary, chosen_alpha=picked,
                  n_seeds=len(scene), learner_path_median_mm=LEARNER_PATH_MED,
                  learner_path_p90_mm=LEARNER_PATH_P90, elapsed_s=time.time() - t0)
    out = RESULTS / 'recover_sweep.json'
    out.write_text(json.dumps(report))
    print(f"{'alpha':>6} {'catch':>7} {'path med':>8} {'path p90':>8} {'path max':>8} {'miss med':>8}", flush=True)
    for alpha in ALPHAS:
        row = summary[str(alpha)]
        d = row['dev_median_mm']
        print(f"{alpha:6.2f} {row['catch_rate']:7.2f} {d['median']:8.0f} {d['p90']:8.0f} "
              f"{d['max']:8.0f} {row['miss_mm']['median']:8.0f}", flush=True)
    print(f'chosen alpha {picked}', flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
