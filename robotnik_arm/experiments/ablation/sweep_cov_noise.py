"""Fit the learner's 18x18 command-error covariance and sweep a noise scale.

Errors are policy command minus the expert's immediate command on the learner's
own states. Noise is an OU process per dimension, with correlated kicks from
the Cholesky factor of that covariance, and one time constant per group
(wheels 1.3 s, arm references 2.9 s, arm velocity references 0.74 s).

A scale alpha is applied to the noise. On 20 fresh seeds we compare each alpha
to the clean expert path and to the learner. No training.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
WEIGHTS = RESULTS / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'
ALPHAS = (0.1, 0.25, 0.5, 1.0)
BASE_COL = 10
TAU = np.array([1.3] * 4 + [2.9] * 7 + [0.74] * 7)
DT = 0.04  # 25 Hz command period
MASTER = 2026
EE_COL = 39

_env = None
_model = None
_cov = None


def init_fit(weights):
    global _env, _model
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _model = NumpyPolicy(weights)


def init_sweep(weights, cov):
    global _env, _model, _cov
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _model = NumpyPolicy(weights)
    _cov = np.asarray(cov, dtype=np.float64)


def fit_job(seed):
    from measure_learner_noise import rollout_errors
    row = rollout_errors(_env, _model, seed)
    err = row['error'][row['ready']] if row['ready'].any() else row['error']
    return err


def _stats(ee, base, ee_clean, base_clean, catch_time, result, dt):
    n = min(len(ee), len(ee_clean))
    dev = np.linalg.norm(ee[:n] - ee_clean[:n], axis=1)
    base_dev = np.linalg.norm(base[:n] - base_clean[:n], axis=1)
    k = n - 1 if catch_time is None else min(int(round(catch_time / dt)), n - 1)
    miss = result['min_dist']
    return dict(success=bool(result['success']),
                miss_mm=None if not np.isfinite(miss) else float(miss * 1000),
                dev_catch_mm=float(dev[k] * 1000),
                dev_median_mm=float(np.median(dev) * 1000),
                dev_p90_mm=float(np.percentile(dev, 90) * 1000),
                base_catch_mm=float(base_dev[k] * 1000))


def sweep_job(seed, noise_seed, alphas):
    from catch_env import CorrelatedCommandNoise, ExpertDriver, rollout
    from np_policy import PolicyDriver
    clean = rollout(_env, ExpertDriver(), seed, 25, stop_on_catch=False, log=True)
    obs_c = clean['arrays']['obs']
    ee_c, base_c = obs_c[:, EE_COL:EE_COL + 3], obs_c[:, BASE_COL:BASE_COL + 2]
    ct = clean['catch_time']
    learner = rollout(_env, PolicyDriver(_model), seed, 25, stop_on_catch=False, log=True)
    obs_l = learner['arrays']['obs']
    out = dict(seed=int(seed), learner=_stats(obs_l[:, EE_COL:EE_COL + 3], obs_l[:, BASE_COL:BASE_COL + 2],
                                             ee_c, base_c, ct, learner, _env.dt), alphas={})
    for i, alpha in enumerate(alphas):
        noise = CorrelatedCommandNoise(int(noise_seed) + i, _cov, TAU, DT, alpha=alpha)
        noisy = rollout(_env, ExpertDriver(), seed, 25, stop_on_catch=False, log=True, command_noise=noise)
        obs_n = noisy['arrays']['obs']
        out['alphas'][str(alpha)] = _stats(obs_n[:, EE_COL:EE_COL + 3], obs_n[:, BASE_COL:BASE_COL + 2],
                                           ee_c, base_c, ct, noisy, _env.dt)
    return out


def spread(rows, key):
    v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if len(v) == 0:
        return dict(median=None, p90=None, mean=None, max=None)
    return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                mean=float(np.mean(v)), max=float(np.max(v)))


def check_stationary(cov):
    """Marginal variances should match cov. Kicks use chol(cov), so this is the realizable part."""
    from catch_env import CorrelatedCommandNoise
    noise = CorrelatedCommandNoise(0, cov, np.full(18, 0.05), DT, alpha=1.0)
    xs = np.stack([noise(0.0) for _ in range(8000)])
    got = np.var(xs[2000:], axis=0)
    target = np.diag(cov)
    return float(np.max(np.abs(got - target) / np.maximum(target, 1e-8)))


def choose(summary, learner):
    """Prefer alpha with catch >= 90% whose path deviation matches the learner."""
    target_med = learner['dev_median_mm']['median']
    target_tail = learner['dev_median_mm']['p90']
    eligible = []
    for alpha, row in summary.items():
        if row['catch_rate'] + 1e-9 >= 0.90:
            med = row['dev_median_mm']['median']
            tail = row['dev_median_mm']['p90']
            score = abs(med - target_med) / max(target_med, 1.0) + abs(tail - target_tail) / max(target_tail, 1.0)
            eligible.append((score, float(alpha)))
    if not eligible:
        return None
    eligible.sort()
    return eligible[0][1]


def fit_cov():
    seeds = json.loads((DATA / 'seeds.json').read_text())['train'][:50]
    print(f'fitting covariance on {len(seeds)} learner rollouts', flush=True)
    t0 = time.time()
    errors = []
    with ProcessPoolExecutor(16, initializer=init_fit, initargs=(str(WEIGHTS),)) as pool:
        futs = [pool.submit(fit_job, int(s)) for s in seeds]
        for n, fut in enumerate(as_completed(futs), 1):
            errors.append(fut.result())
            if n % 10 == 0 or n == len(seeds):
                print(f'  fit {n}/{len(seeds)} ({time.time() - t0:.0f}s)', flush=True)
    stacked = np.concatenate(errors, axis=0)
    cov = np.cov(stacked, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    cov_path = RESULTS / 'learner_cov.json'
    cov_path.write_text(json.dumps(dict(
        n_ticks=int(stacked.shape[0]), n_episodes=len(errors),
        tau_s=TAU.tolist(), cov=cov.tolist(),
        sigma=np.sqrt(np.diag(cov)).tolist())))
    print(f'covariance from {stacked.shape[0]} ticks -> {cov_path}', flush=True)
    return cov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', default='0.1,0.25,0.5,1')
    parser.add_argument('--skip-fit', action='store_true')
    args = parser.parse_args()
    alphas = tuple(float(a) for a in args.alphas.split(',') if a)
    from gen_learner_dart import seed_stream
    rng = np.random.default_rng(1)
    toy = rng.normal(size=(18, 18))
    toy = toy @ toy.T
    rel = check_stationary(toy)
    print(f'stationary check max relative error {rel:.3f}', flush=True)
    if rel > 0.15:
        raise SystemExit('OU marginal variances do not match Sigma')
    cov_path = RESULTS / 'learner_cov.json'
    if args.skip_fit and cov_path.exists():
        cov = np.array(json.loads(cov_path.read_text())['cov'])
        print(f'loaded {cov_path}', flush=True)
    else:
        cov = fit_cov()
    t0 = time.time()

    scene = seed_stream(32, 20)
    noise_seeds = seed_stream(33, 20)
    print(f'sweeping alpha {alphas} on {len(scene)} seeds', flush=True)
    t1 = time.time()
    rows = []
    with ProcessPoolExecutor(16, initializer=init_sweep, initargs=(str(WEIGHTS), cov.tolist())) as pool:
        futs = [pool.submit(sweep_job, s, n, alphas) for s, n in zip(scene, noise_seeds)]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 5 == 0 or n == len(scene):
                print(f'  sweep {n}/{len(scene)} ({time.time() - t1:.0f}s)', flush=True)

    learner_rows = [r['learner'] for r in rows]
    summary = {}
    for alpha in alphas:
        got = [r['alphas'][str(alpha)] for r in rows]
        summary[str(alpha)] = dict(
            catch_rate=float(np.mean([g['success'] for g in got])),
            miss_mm=spread(got, 'miss_mm'),
            dev_catch_mm=spread(got, 'dev_catch_mm'),
            dev_median_mm=spread(got, 'dev_median_mm'),
            dev_p90_mm=spread(got, 'dev_p90_mm'),
            base_catch_mm=spread(got, 'base_catch_mm'))
    learner_summary = dict(
        catch_rate=float(np.mean([g['success'] for g in learner_rows])),
        miss_mm=spread(learner_rows, 'miss_mm'),
        dev_catch_mm=spread(learner_rows, 'dev_catch_mm'),
        dev_median_mm=spread(learner_rows, 'dev_median_mm'),
        dev_p90_mm=spread(learner_rows, 'dev_p90_mm'),
        base_catch_mm=spread(learner_rows, 'base_catch_mm'))
    picked = choose(summary, learner_summary)
    report = dict(alphas=summary, learner=learner_summary, chosen_alpha=picked,
                  n_seeds=len(scene), elapsed_s=time.time() - t0, episodes=rows)
    out = RESULTS / 'cov_sweep.json'
    out.write_text(json.dumps(report))
    p = learner_summary['dev_median_mm']
    print(f"learner: catch {learner_summary['catch_rate']:.2f}  "
          f"path median {p['median']:.0f}  p90 {p['p90']:.0f}  max {p['max']:.0f} mm  "
          f"base@catch {learner_summary['base_catch_mm']['median']:.0f} mm", flush=True)
    print(f"{'alpha':>6} {'catch':>7} {'path med':>8} {'path p90':>8} {'path max':>8} "
          f"{'base med':>8} {'miss med':>8}", flush=True)
    for alpha in alphas:
        row = summary[str(alpha)]
        d = row['dev_median_mm']
        print(f"{alpha:6.2f} {row['catch_rate']:7.2f} {d['median']:8.0f} {d['p90']:8.0f} "
              f"{d['max']:8.0f} {row['base_catch_mm']['median']:8.0f} {row['miss_mm']['median']:8.0f}", flush=True)
    print(f'chosen alpha {picked}', flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
