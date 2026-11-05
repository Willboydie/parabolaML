"""Hand the policy the expert's state just before the stop, and measure the lead.

The expert drives until 2 s, 1 s or 0.5 s before its own stop. From there the
policy runs to the catch. At the switch the base is the expert's, so a lead at
the catch was produced by the policy's own braking. The along-track error is
the policy base minus the expert base, in the expert's route frame, at the
closest approach.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'stage2' / 'handoff_stop.json'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
SEEDS = (0, 1, 2)
TAUS = (2.0, 1.0, 0.5)
RATE_HZ = 25
MOVE_V = 0.02
WHEEL_RADIUS = 0.11

_env = None
_model = None


def forward_speed(cmd):
    wheels = np.asarray(cmd, dtype=np.float64)[..., :4]
    right = 0.5 * (wheels[..., 0] + wheels[..., 3])
    left = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return 0.5 * (right + left) * WHEEL_RADIUS


def init_worker(weights):
    global _env, _model
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _model = NumpyPolicy(weights)


def stop_time(times, speed):
    """First command time after the last drive command, or None."""
    moving = np.flatnonzero(np.asarray(speed) > MOVE_V)
    if len(moving) == 0:
        return None
    last = int(moving[-1])
    if last + 1 >= len(times):
        return None
    tail = np.asarray(speed)[last + 1:last + 11]
    if len(tail) < 5 or float(np.median(tail)) > 0.01:
        return None
    return float(times[last + 1])


def rollout(driver, seed, track_drive):
    """One episode at the policy rate. Returns the base path, the stop, and the closest approach."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    env = _env
    env.reset(int(seed))
    driver.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    base = []
    times, speed = [], []
    drive_start = drive_end = None
    best = None
    catch_time = None
    step = 0
    while True:
        t = step * env.dt
        base.append(env.base_pos[:2].copy())
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if best is None or dist < best['dist']:
                best = dict(dist=float(dist), step=step, base=env.base_pos[:2].copy())
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        if hasattr(driver, 'sense'):
            driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            phase = None
            if track_drive:
                phase = 'done' if driver.base_state.done else driver.base_state.phase
                if phase == 'drive':
                    if drive_start is None:
                        drive_start = env.base_pos[:2].copy()
                    drive_end = env.base_pos[:2].copy()
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
            times.append(t)
            speed.append(float(forward_speed(cmd)))
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None and np.linalg.norm(drive_end - drive_start) > 0.05:
        chord = (drive_end - drive_start).astype(np.float64)
        chord = chord / np.linalg.norm(chord)
    return dict(base=np.asarray(base, np.float64), times=np.asarray(times), speed=np.asarray(speed),
                chord=chord, best=best, catch_time=None if catch_time is None else float(catch_time),
                dt=float(env.dt))


def along_at(policy_base, expert_base, chord, step):
    if chord is None or step >= len(policy_base) or step >= len(expert_base):
        return None
    delta = policy_base[step] - expert_base[step]
    return float(np.dot(delta, chord) * 1000.0)


def job(seed):
    from catch_env import ExpertDriver
    from diagnose import HandoffDriver
    from np_policy import PolicyDriver
    expert = rollout(ExpertDriver(), seed, track_drive=True)
    stopped = stop_time(expert['times'], expert['speed'])
    rows = []
    if stopped is None or expert['chord'] is None:
        return dict(seed=int(seed), stop_time=stopped, rows=rows)
    for tau in TAUS:
        switch = stopped - tau
        if switch < 0.05:
            rows.append(dict(tau=tau, skipped='switch before the episode'))
            continue
        policy = rollout(HandoffDriver(ExpertDriver(), PolicyDriver(_model), switch), seed, track_drive=False)
        best = policy['best']
        catch_step = None if best is None else int(best['step'])
        switch_step = int(round(switch / expert['dt']))
        later_step = int(round((stopped + 1.0) / expert['dt']))
        rows.append(dict(
            tau=tau, switch=switch,
            along_at_switch_mm=along_at(policy['base'], expert['base'], expert['chord'], switch_step),
            along_1s_after_stop_mm=along_at(policy['base'], expert['base'], expert['chord'], later_step),
            along_at_catch_mm=None if catch_step is None else along_at(
                policy['base'], expert['base'], expert['chord'], catch_step),
            miss_mm=None if best is None else float(best['dist'] * 1000.0),
            caught=policy['catch_time'] is not None,
        ))
    return dict(seed=int(seed), stop_time=stopped, rows=rows)


def mean_ci(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = int(len(values))
    if n == 0:
        return dict(n=0, mean=None, median=None, ci95=(None, None), frac_ahead=None)
    mean = float(np.mean(values))
    se = 0.0 if n < 2 else float(np.std(values, ddof=1) / np.sqrt(n))
    return dict(n=n, mean=mean, median=float(np.median(values)),
                ci95=(mean - 1.96 * se, mean + 1.96 * se),
                frac_ahead=float(np.mean(values > 0)))


def summarise(episodes):
    by_tau = {tau: [] for tau in TAUS}
    for ep in episodes:
        for row in ep['rows']:
            if 'skipped' in row:
                continue
            by_tau[row['tau']].append(row)
    out = {}
    for tau, rows in by_tau.items():
        out[str(tau)] = dict(
            n=len(rows),
            at_switch=mean_ci([r['along_at_switch_mm'] for r in rows]),
            one_s_after_stop=mean_ci([r['along_1s_after_stop_mm'] for r in rows]),
            at_catch=mean_ci([r['along_at_catch_mm'] for r in rows]),
            miss_mm=mean_ci([r['miss_mm'] for r in rows]),
            catch_rate=float(np.mean([r['caught'] for r in rows])) if rows else None,
        )
    return out


def main():
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    from stage2_log import finish_run, start_run
    run_id, record_path, record = start_run(
        'handoff-stop', CEILING, dict(taus=list(TAUS), rate_hz=RATE_HZ), sys.argv,
        dict(test=len(test), seeds=list(SEEDS)),
        note='expert drives until 2, 1 and 0.5 s before its stop, then the policy; along-track error at the catch')
    t0 = time.time()
    report = dict(taus=list(TAUS), per_seed={})
    blocks = []
    for seed in SEEDS:
        weights = RESULTS / 'models' / f'{CEILING}_s{seed}.npz'
        if not weights.exists():
            raise SystemExit(f'missing {weights}')
        print(f'seed {seed} on {len(test)} scenes', flush=True)
        episodes = []
        with ProcessPoolExecutor(16, initializer=init_worker, initargs=(str(weights),)) as pool:
            futs = [pool.submit(job, int(s)) for s in test]
            for n, fut in enumerate(as_completed(futs), 1):
                episodes.append(fut.result())
                if n % 25 == 0 or n == len(futs):
                    print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
        stats = summarise(episodes)
        report['per_seed'][f's{seed}'] = stats
        blocks.append(stats)
        for tau, row in stats.items():
            catch = row['at_catch']
            print(f"  tau {tau}: catch along {catch['mean']} mm  "
                  f"ahead {catch['frac_ahead']}  1s-after {row['one_s_after_stop']['mean']}", flush=True)
    pooled = {}
    for tau in TAUS:
        key = str(tau)
        pooled[key] = dict(
            at_catch_mm=float(np.mean([b[key]['at_catch']['mean'] for b in blocks])),
            at_catch_frac_ahead=float(np.mean([b[key]['at_catch']['frac_ahead'] for b in blocks])),
            one_s_after_stop_mm=float(np.mean([b[key]['one_s_after_stop']['mean'] for b in blocks])),
            at_switch_mm=float(np.mean([b[key]['at_switch']['mean'] for b in blocks])),
            miss_mm=float(np.mean([b[key]['miss_mm']['mean'] for b in blocks])),
            catch_rate=float(np.mean([b[key]['catch_rate'] for b in blocks])),
        )
    report['pooled'] = pooled
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    finish_run(record_path, record, dict(
        miss_mm=pooled['1.0']['at_catch_mm'], catch_rate=pooled['1.0']['catch_rate']), [OUT])
    print(json.dumps(pooled, indent=2), flush=True)
    print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
