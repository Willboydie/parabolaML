"""Heading of the Stage 1 learner relative to the expert's planned drive heading.

On each of the 50 training seeds used for the noise measurement, roll the clean
expert and record the direction from the base to the planned goal while the
expert is in the drive phase. Roll the learner on the same seed and, at those
same times, measure wrap(planned heading − learner bearing).
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
WEIGHTS = RESULTS / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'
N = 50
RATE_HZ = 25

_env = None
_model = None


def init_worker(weights):
    global _env, _model
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _model = NumpyPolicy(weights)


def expert_drive(seed):
    """Times and planned headings while the clean expert is driving."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    from simulation_and_control import wrap_angle
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    rows = []
    step = 0
    while True:
        t = step * env.dt
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            phase = 'done' if driver.base_state.done else driver.base_state.phase
            if phase == 'drive':
                to_goal = driver.goal_pos - env.base_pos[:2]
                if np.linalg.norm(to_goal) > 1e-3:
                    planned = float(np.arctan2(to_goal[1], to_goal[0]))
                    rows.append((round(t, 5), planned, float(env.bearing),
                                 float(wrap_angle(planned - env.bearing))))
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
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
    return rows


def learner_bearing(seed, times):
    from catch_env import FIRST_TICK_STEP
    from np_policy import PolicyDriver
    env = _env
    policy = PolicyDriver(_model)
    env.reset(int(seed))
    policy.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    t_max = max(times) if times else 0.0
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    found = {}
    step = 0
    while True:
        t = step * env.dt
        if t > t_max + 1e-9:
            break
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            key = round(t, 5)
            if key in times:
                found[key] = float(env.bearing)
            cmd, _ = policy.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    return found


def job(seed):
    from simulation_and_control import wrap_angle
    drive = expert_drive(seed)
    times = {row[0] for row in drive}
    bearings = learner_bearing(seed, times)
    err = []
    expert_err = []
    for t, planned, _bearing, expert_e in drive:
        if t not in bearings:
            continue
        err.append(float(wrap_angle(planned - bearings[t])))
        expert_err.append(expert_e)
    err = np.asarray(err)
    expert_err = np.asarray(expert_err)
    if len(err) == 0:
        return dict(seed=int(seed), n=0)
    deg = np.rad2deg(err)
    abs_deg = np.abs(deg)
    return dict(seed=int(seed), n=int(len(deg)),
                median_deg=float(np.median(deg)),
                mean_deg=float(np.mean(deg)),
                frac_under_3=float(np.mean(abs_deg < 3)),
                frac_3_to_10=float(np.mean((abs_deg >= 3) & (abs_deg <= 10))),
                frac_over_10=float(np.mean(abs_deg > 10)),
                expert_median_deg=float(np.median(np.rad2deg(expert_err))))


def main():
    seeds = json.loads((ROOT / 'data' / 'seeds.json').read_text())['train'][:N]
    print(f'{N} seeds, learner heading vs expert drive heading', flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(16, initializer=init_worker, initargs=(str(WEIGHTS),)) as pool:
        futs = [pool.submit(job, int(s)) for s in seeds]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)} ({time.time() - t0:.0f}s)', flush=True)
    usable = [r for r in rows if r['n'] > 0]
    med = np.array([r['median_deg'] for r in usable])
    # Weight ticks equally for the pooled fractions.
    ns = np.array([r['n'] for r in usable])
    def wmean(key):
        return float(np.average([r[key] for r in usable], weights=ns))
    pos = int(np.sum(med > 1))
    neg = int(np.sum(med < -1))
    near = int(np.sum(np.abs(med) <= 1))
    summary = dict(
        n_seeds=len(usable),
        ticks=int(ns.sum()),
        convention='planned_heading - learner_bearing, degrees; positive means the learner is clockwise of the plan',
        per_seed_median_deg=dict(median=float(np.median(med)), mean=float(np.mean(med)),
                                 p10=float(np.percentile(med, 10)), p90=float(np.percentile(med, 90)),
                                 min=float(med.min()), max=float(med.max())),
        seeds_median_sign=dict(positive_gt_1deg=pos, negative_lt_m1deg=neg, within_1deg=near),
        tick_frac=dict(under_3deg=wmean('frac_under_3'), from_3_to_10=wmean('frac_3_to_10'),
                       over_10=wmean('frac_over_10')),
        expert_own_median_of_medians_deg=float(np.median([r['expert_median_deg'] for r in usable])),
        episodes=usable,
    )
    out = RESULTS / 'learner_heading.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f"seeds {len(usable)}  drive ticks {int(ns.sum())}", flush=True)
    print(f"per-seed median error: median {np.median(med):+.2f}°  mean {np.mean(med):+.2f}°  "
          f"p10 {np.percentile(med, 10):+.2f}°  p90 {np.percentile(med, 90):+.2f}°", flush=True)
    print(f"sign of per-seed median: {pos} above +1°, {neg} below -1°, {near} within 1°", flush=True)
    print(f"ticks |err|<3° {wmean('frac_under_3'):.0%}   3–10° {wmean('frac_3_to_10'):.0%}   "
          f">10° {wmean('frac_over_10'):.0%}", flush=True)
    print(f"expert's own drive error, median of per-seed medians "
          f"{summary['expert_own_median_of_medians_deg']:+.2f}°", flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
