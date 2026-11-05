"""Measure learner action noise on policy-visited states (for DART).

Part 1 of diagnose.py scores the policy on expert states. That underestimates the
error the learner makes once it has drifted. Here we roll out the Stage 1 winner,
and at every policy tick ask the expert for its immediate command in that state.
The error series (policy − expert) over the 18 action dims gives per-dim σ and a
correlation time τ_c from the lag where the autocorrelation falls to 1/e.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from catch_env import CMD_DIM, FIRST_TICK_STEP
from np_policy import NumpyPolicy, PolicyDriver

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
WEIGHTS = RESULTS / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'
ACF_TARGET = np.e ** -1

ACT_NAMES = (
    [f'wheel_qd_{i}' for i in range(4)]
    + [f'arm_q_rel_{i}' for i in range(7)]
    + [f'arm_qd_{i}' for i in range(7)]
)
GROUPS = {'wheel_qd': slice(0, 4), 'arm_q_rel': slice(4, 11), 'arm_qd': slice(11, 18)}

_env = None
_model = None


def init_worker(weights):
    global _env, _model
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()
    _model = NumpyPolicy(weights)


class ExpertProbe:
    """Expert that senses every sim step and answers with an immediate command.

    Planning matches ExpertDriver (once, or again after a miss). Controllers are
    re-seeded from the measured state each query so the answer is about this
    state, not an expert that has been integrating its own open references while
    the learner was driving.
    """

    def __init__(self):
        from catch_env import ExpertDriver
        self.inner = ExpertDriver()

    def reset(self, env):
        self.inner.reset(env)

    def sense(self, env, t):
        self.inner.sense(env, t)

    def act(self, env, t, dt_ctrl):
        from robotnik_arm_control import BaseNavState
        ex = self.inner
        # Fresh nav phase from geometry toward the (already planned) goal.
        ex.base_state = BaseNavState()
        # Instantaneous arm reference starts at the measured angle.
        ex.q_des = env.q[4:11].copy()
        ex.qd_des = np.zeros(7)
        return ex.act(env, t, dt_ctrl)


def rollout_errors(env, model, seed):
    """Drive with the policy; record policy−expert error at every policy tick."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER
    policy = PolicyDriver(model)
    expert = ExpertProbe()
    env.reset(seed)
    policy.reset(env)
    expert.reset(env)
    rate = model.cfg.rate_hz
    stride = int(round(1.0 / (rate * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    errors, times, ready_flags = [], [], []
    catch_time = None
    step = 0
    while True:
        t = step * env.dt
        expert.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            pol, _ = policy.act(env, t, stride * env.dt)
            exp, ready = expert.act(env, t, stride * env.dt)
            pol = np.asarray(pol, dtype=np.float64)
            exp = np.asarray(exp, dtype=np.float64)
            # Joint refs compared as relative-to-measured-q, matching training labels.
            q = env.q[4:11]
            err = np.empty(CMD_DIM, dtype=np.float64)
            err[:4] = pol[:4] - exp[:4]
            err[4:11] = (pol[4:11] - q) - (exp[4:11] - q)
            err[11:18] = pol[11:18] - exp[11:18]
            errors.append(err)
            times.append(t)
            ready_flags.append(bool(ready))
            cmd = pol
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied[:4], applied[4:])
        step += 1
    return dict(
        seed=int(seed),
        success=catch_time is not None,
        min_dist=float(env.min_dist),
        catch_time=catch_time,
        duration=float(env.target.duration),
        t=np.asarray(times, dtype=np.float64),
        error=np.asarray(errors, dtype=np.float64),
        ready=np.asarray(ready_flags, dtype=bool),
    )


def job(seed):
    return rollout_errors(_env, _model, seed)


def acf_1d(x, max_lag):
    """Biased autocorrelation of a zero-mean series, normalized so lag 0 is 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-18 or len(x) < 2:
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    # full correlate; take non-negative lags
    full = np.correlate(x, x, mode='full')
    mid = len(x) - 1
    acf = full[mid:mid + max_lag + 1] / var
    if len(acf) < max_lag + 1:
        acf = np.r_[acf, np.full(max_lag + 1 - len(acf), np.nan)]
    return acf


def correlation_time(acf, dt):
    """First lag where ACF <= 1/e, linearly interpolated; returned in seconds."""
    if acf is None or len(acf) < 2 or not np.isfinite(acf[0]):
        return None, None
    target = ACF_TARGET
    for k in range(1, len(acf)):
        if not np.isfinite(acf[k]):
            break
        if acf[k] <= target:
            a0, a1 = acf[k - 1], acf[k]
            if a0 == a1:
                frac = 0.0
            else:
                frac = (a0 - target) / (a0 - a1)
            lag = (k - 1) + float(np.clip(frac, 0.0, 1.0))
            return lag, lag * dt
    # never crossed: report the last finite lag as a lower bound
    last = max(i for i in range(len(acf)) if np.isfinite(acf[i]))
    return float(last), last * dt


def fit_noise(rows, rate_hz, max_lag_s=4.0):
    dt = 1.0 / rate_hz
    max_lag = int(round(max_lag_s * rate_hz))
    # Prefer ticks where the expert has a plan; fall back to all ticks if needed.
    series = []
    for r in rows:
        err = r['error']
        ready = r['ready']
        use = err[ready] if ready.any() else err
        if len(use) >= 2:
            series.append(use)
    if not series:
        raise RuntimeError('no error samples')
    stacked = np.concatenate(series, axis=0)
    sigma = stacked.std(axis=0, ddof=1)
    mean = stacked.mean(axis=0)
    # Mean ACF across episodes (avoids stitching episode boundaries).
    acfs = np.zeros((CMD_DIM, max_lag + 1), dtype=np.float64)
    counts = np.zeros((CMD_DIM, max_lag + 1), dtype=np.float64)
    for err in series:
        for d in range(CMD_DIM):
            a = acf_1d(err[:, d], max_lag)
            m = np.isfinite(a)
            acfs[d, m] += a[m]
            counts[d, m] += 1.0
    with np.errstate(invalid='ignore'):
        mean_acf = np.where(counts > 0, acfs / np.maximum(counts, 1.0), np.nan)
    dims = []
    for d in range(CMD_DIM):
        lag_steps, tau_s = correlation_time(mean_acf[d], dt)
        dims.append(dict(
            dim=d,
            name=ACT_NAMES[d],
            mean=float(mean[d]),
            sigma=float(sigma[d]),
            tau_c_s=None if tau_s is None else float(tau_s),
            tau_c_steps=None if lag_steps is None else float(lag_steps),
            acf=mean_acf[d, : min(len(mean_acf[d]), int(rate_hz) + 1)].astype(float).tolist(),
        ))
    groups = {}
    for name, sl in GROUPS.items():
        rows_d = dims[sl]
        groups[name] = dict(
            sigma_mean=float(np.mean([r['sigma'] for r in rows_d])),
            sigma_median=float(np.median([r['sigma'] for r in rows_d])),
            tau_c_s_mean=float(np.mean([r['tau_c_s'] for r in rows_d if r['tau_c_s'] is not None])),
            tau_c_s_median=float(np.median([r['tau_c_s'] for r in rows_d if r['tau_c_s'] is not None])),
            dims=list(range(sl.start, sl.stop)),
        )
    return dict(
        n_episodes=len(rows),
        n_ticks=int(sum(len(r['error']) for r in rows)),
        n_ready_ticks=int(sum(int(r['ready'].sum()) for r in rows)),
        n_fit_ticks=int(stacked.shape[0]),
        catch_rate=float(np.mean([r['success'] for r in rows])),
        mean_miss_mm=float(np.mean([r['min_dist'] for r in rows]) * 1000),
        dt_s=dt,
        rate_hz=rate_hz,
        acf_target=float(ACF_TARGET),
        dims=dims,
        groups=groups,
        sigma=sigma.astype(float).tolist(),
        tau_c_s=[d['tau_c_s'] for d in dims],
        suggested=dict(
            # Compact DART proposal: per-group σ and a shared hold ≈ median τ_c.
            wheel_std=float(np.mean(sigma[0:4])),
            arm_q_std=float(np.mean(sigma[4:11])),
            arm_qd_std=float(np.mean(sigma[11:18])),
            hold_s=float(np.median([d['tau_c_s'] for d in dims if d['tau_c_s'] is not None])),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--out', type=Path, default=RESULTS / 'learner_noise.json')
    args = parser.parse_args()
    seeds = json.loads((DATA / 'seeds.json').read_text())['train'][:args.episodes]
    model = NumpyPolicy(args.weights)
    print(f'policy {args.weights.name}  {model.cfg.name}  rate={model.cfg.rate_hz} Hz  '
          f'episodes={len(seeds)}', flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker,
                              initargs=(str(args.weights),)) as pool:
        futs = {pool.submit(job, seed): seed for seed in seeds}
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0 or n == len(seeds):
                print(f'  {n}/{len(seeds)}  ({time.time() - t0:.0f}s)', flush=True)

    fit = fit_noise(rows, model.cfg.rate_hz)
    report = dict(
        weights=str(args.weights),
        config=model.cfg.to_dict(),
        n_requested=len(seeds),
        elapsed_s=time.time() - t0,
        noise=fit,
        per_episode=[dict(seed=r['seed'], success=r['success'], min_dist=r['min_dist'],
                          catch_time=r['catch_time'], n_ticks=len(r['error']),
                          n_ready=int(r['ready'].sum())) for r in rows],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky per-dim ACF tails from the printed summary path; keep full in JSON.
    args.out.write_text(json.dumps(report, indent=2))

    g = fit['groups']
    s = fit['suggested']
    print(f"\nrollouts: catch {fit['catch_rate']:.3f}  miss {fit['mean_miss_mm']:.1f} mm  "
          f"ticks {fit['n_fit_ticks']} ready-or-all", flush=True)
    print(f"{'dim':<16} {'σ':>10} {'τ_c (s)':>10} {'τ_c steps':>10}", flush=True)
    for d in fit['dims']:
        tau = '—' if d['tau_c_s'] is None else f"{d['tau_c_s']:.3f}"
        steps = '—' if d['tau_c_steps'] is None else f"{d['tau_c_steps']:.2f}"
        print(f"{d['name']:<16} {d['sigma']:10.4f} {tau:>10} {steps:>10}", flush=True)
    print('\ngroup summary (mean σ / median τ_c):', flush=True)
    for name, row in g.items():
        print(f"  {name:<10}  σ={row['sigma_mean']:.4f}  τ_c={row['tau_c_s_median']:.3f} s", flush=True)
    print('\nsuggested DART noise (group-averaged σ, median τ_c as hold):', flush=True)
    print(f"  wheel_std={s['wheel_std']:.4f} rad/s", flush=True)
    print(f"  arm_q_std={s['arm_q_std']:.4f} rad", flush=True)
    print(f"  arm_qd_std={s['arm_qd_std']:.4f} rad/s", flush=True)
    print(f"  hold_s={s['hold_s']:.3f} s   (was 0.05 s with wheel 0.08 / arm torque 0.2)", flush=True)
    print(f'wrote {args.out} in {report["elapsed_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
