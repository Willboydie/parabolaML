"""At the learner's closest approach, compare base offset with the miss.

Uses the 50 saved learner rollouts (dagger_iter1 demos 0–49, the same training
seeds as the noise measurement). The expert is rolled on those seeds only to
read its base and the route direction at that same time. No training.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEMOS = ROOT / 'data' / 'dagger_iter1'
RESULTS = ROOT / 'results'
N = 50
# Observation layout in the logged rows.
BALL = slice(1, 4)
BASE = slice(10, 13)
EE = slice(39, 42)


def closest_approach(obs):
    """Same catch window as CatchEnv.capture_distance: ball must leave the box and come back."""
    center = obs[0, BASE]
    lower = np.array([center[0] - 2.5, center[1] - 2.5, 0.07])
    upper = np.array([center[0] + 2.5, center[1] + 2.5, 1.30])
    ball = obs[:, BALL]
    outside = np.any(ball < lower - 1e-6, axis=1) | np.any(ball > upper + 1e-6, axis=1)
    has_left = np.cumsum(outside) > 0
    inside = has_left & ~outside
    if not np.any(inside):
        return None
    dist = np.linalg.norm(obs[:, EE] - ball, axis=1)
    dist = np.where(inside, dist, np.inf)
    k = int(np.argmin(dist))
    return dict(step=k, t=float(obs[k, 0]), dist=float(dist[k]),
                base=obs[k, BASE].astype(float), ee=obs[k, EE].astype(float),
                ball=obs[k, BALL].astype(float))


def load_learner():
    rows = []
    for i in range(N):
        with np.load(DEMOS / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            obs = z['obs']
            meta = json.loads(str(z['meta']))
        hit = closest_approach(obs)
        if hit is None:
            rows.append(dict(index=i, seed=meta['seed'], ok=False))
        else:
            hit.update(index=i, seed=int(meta['seed']), ok=True)
            rows.append(hit)
    return rows


_env = None


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def expert_at(seed, step_star):
    from catch_env import CAP_AFTER_ENDPOINT, FIRST_TICK_STEP, ExpertDriver
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    stride = int(round(1.0 / (1000 * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    drive_start = drive_end = None
    step = 0
    while True:
        t = step * env.dt
        if step == step_star or t >= end_time:
            break
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            phase = 'done' if driver.base_state.done else driver.base_state.phase
            if phase == 'drive':
                if drive_start is None:
                    drive_start = env.base_pos[:2].copy()
                drive_end = env.base_pos[:2].copy()
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None:
        chord = (drive_end - drive_start).astype(float)
    return dict(base=env.base_pos.astype(float), ee=env.ee.astype(float),
                bearing=float(env.bearing), chord=None if chord is None else chord.tolist(),
                reached=step == step_star)


def route_axes(expert):
    chord = None if expert['chord'] is None else np.asarray(expert['chord'], dtype=float)
    if chord is not None and np.linalg.norm(chord) > 0.05:
        along = chord / np.linalg.norm(chord)
    else:
        b = expert['bearing']
        along = np.array([np.cos(b), np.sin(b)])
    cross = np.array([-along[1], along[0]])  # left of the direction of travel
    return along, cross


def main():
    learner = load_learner()
    pending = [r for r in learner if r['ok']]
    print(f'{len(pending)}/{N} learner episodes have a closest approach', flush=True)
    t0 = time.time()
    expert = {}
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        futs = {pool.submit(expert_at, r['seed'], r['step']): r['index'] for r in pending}
        for n, fut in enumerate(as_completed(futs), 1):
            expert[futs[fut]] = fut.result()
            if n % 10 == 0 or n == len(futs):
                print(f'  expert {n}/{len(futs)} ({time.time() - t0:.0f}s)', flush=True)

    episodes = []
    for r in pending:
        ex = expert[r['index']]
        if not ex['reached']:
            continue
        along, cross = route_axes(ex)
        base_xy = r['base'][:2] - ex['base'][:2]
        miss = r['ee'] - r['ball']
        miss_xy = miss[:2]
        # Arm pose relative to the expert, after removing the base shift.
        ee_shift = r['ee'][:2] - ex['ee'][:2]
        arm_xy = ee_shift - base_xy
        episodes.append(dict(
            seed=r['seed'], t=r['t'],
            base_along_mm=float(np.dot(base_xy, along) * 1000),
            base_cross_mm=float(np.dot(base_xy, cross) * 1000),
            base_mm=float(np.linalg.norm(base_xy) * 1000),
            miss_along_mm=float(np.dot(miss_xy, along) * 1000),
            miss_cross_mm=float(np.dot(miss_xy, cross) * 1000),
            miss_up_mm=float(miss[2] * 1000),
            miss_xy_mm=float(np.linalg.norm(miss_xy) * 1000),
            miss_mm=float(np.linalg.norm(miss) * 1000),
            expert_hand_mm=float(np.linalg.norm(ex['ee'] - r['ball']) * 1000),
            arm_xy_mm=float(np.linalg.norm(arm_xy) * 1000),
            cos_base_miss=float(np.dot(base_xy, miss_xy) / (np.linalg.norm(base_xy) * np.linalg.norm(miss_xy) + 1e-12)),
        ))

    def col(name):
        return np.array([e[name] for e in episodes])

    base_a, base_c = col('base_along_mm'), col('base_cross_mm')
    miss_a, miss_c = col('miss_along_mm'), col('miss_cross_mm')
    base_n, miss_n = col('base_mm'), col('miss_mm')
    miss_xy, miss_up = col('miss_xy_mm'), col('miss_up_mm')
    arm, cos = col('arm_xy_mm'), col('cos_base_miss')
    along_dominates = np.abs(base_a) > 2 * np.abs(base_c)
    aligned = (cos > 0.7) & (np.abs(miss_xy - base_n) < 0.5 * np.maximum(miss_xy, 1))
    base_small_miss_large = (base_n < 30) & (miss_n > 50)
    summary = dict(
        n=len(episodes),
        base_mm=dict(median=float(np.median(base_n)), p90=float(np.percentile(base_n, 90))),
        base_along_mm=dict(median=float(np.median(base_a)), mean=float(np.mean(base_a)),
                           abs_median=float(np.median(np.abs(base_a)))),
        base_cross_mm=dict(median=float(np.median(base_c)), mean=float(np.mean(base_c)),
                           abs_median=float(np.median(np.abs(base_c)))),
        miss_mm=dict(median=float(np.median(miss_n)), mean=float(np.mean(miss_n))),
        miss_xy_mm=dict(median=float(np.median(miss_xy))),
        miss_up_mm=dict(median=float(np.median(miss_up)), abs_median=float(np.median(np.abs(miss_up)))),
        miss_along_mm=dict(median=float(np.median(miss_a)), abs_median=float(np.median(np.abs(miss_a)))),
        miss_cross_mm=dict(median=float(np.median(miss_c)), abs_median=float(np.median(np.abs(miss_c)))),
        arm_residual_xy_mm=dict(median=float(np.median(arm))),
        expert_hand_at_same_time_mm=dict(median=float(np.median(col('expert_hand_mm')))),
        corr_along=float(np.corrcoef(base_a, miss_a)[0, 1]),
        corr_cross=float(np.corrcoef(base_c, miss_c)[0, 1]),
        median_cos_base_miss=float(np.median(cos)),
        frac_along_dominates_base=float(np.mean(along_dominates)),
        frac_miss_aligned_with_base=float(np.mean(aligned)),
        frac_base_small_miss_large=float(np.mean(base_small_miss_large)),
        episodes=episodes,
    )
    out = RESULTS / 'closest_approach.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f"n={summary['n']}", flush=True)
    print(f"base offset  median {summary['base_mm']['median']:.0f} mm  "
          f"along {summary['base_along_mm']['median']:+.0f} (abs {summary['base_along_mm']['abs_median']:.0f})  "
          f"cross {summary['base_cross_mm']['median']:+.0f} (abs {summary['base_cross_mm']['abs_median']:.0f})", flush=True)
    print(f"miss         median {summary['miss_mm']['median']:.0f} mm  "
          f"xy {summary['miss_xy_mm']['median']:.0f}  up abs {summary['miss_up_mm']['abs_median']:.0f}", flush=True)
    print(f"miss along abs {summary['miss_along_mm']['abs_median']:.0f}  "
          f"cross abs {summary['miss_cross_mm']['abs_median']:.0f}", flush=True)
    print(f"corr along {summary['corr_along']:+.2f}  cross {summary['corr_cross']:+.2f}  "
          f"median cos {summary['median_cos_base_miss']:+.2f}", flush=True)
    print(f"arm residual xy median {summary['arm_residual_xy_mm']['median']:.0f} mm  "
          f"expert hand at that time {summary['expert_hand_at_same_time_mm']['median']:.0f} mm", flush=True)
    print(f"along-track dominates base offset in {summary['frac_along_dominates_base']:.0%} of episodes", flush=True)
    print(f"miss aligned with base offset in {summary['frac_miss_aligned_with_base']:.0%}", flush=True)
    print(f"base <30 mm and miss >50 mm in {summary['frac_base_small_miss_large']:.0%}", flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
