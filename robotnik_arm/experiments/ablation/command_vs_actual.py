"""Commanded forward speed against the base's actual forward speed.

Same hand-off as ``handoff_stop.py``: the expert drives at 25 Hz until one
second before its stop, then a ceiling seed takes over. Both the expert
reference and the policy log the commanded body speed and the actual body
speed at every millisecond. The question is whether the actual stop lags the
commanded one.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from chunk_replay import CEILING, SEEDS
from handoff_stop import MOVE_V, forward_speed, stop_time
from hold_replay import along_at, mean_ci

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'stage2' / 'command_vs_actual.json'
RATE_HZ = 25

_env = None
_models = None


def init_worker():
    global _env, _models
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _models = [NumpyPolicy(RESULTS / 'models' / f'{CEILING}_s{s}.npz') for s in SEEDS]


def body_forward(env):
    bearing = float(env.bearing)
    return float(env.base_vel[0] * np.cos(bearing) + env.base_vel[1] * np.sin(bearing))


def episode(driver, seed, track_drive):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    env = _env
    env.reset(int(seed))
    driver.reset(env)
    dt = float(env.dt)
    stride = int(round(1.0 / (RATE_HZ * dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    base, cmd_speed, actual = [], [], []
    tick_t, tick_v = [], []
    drive_start = drive_end = None
    catch_time = None
    step = 0
    while True:
        t = step * dt
        base.append(env.base_pos[:2].copy())
        if hasattr(driver, 'sense'):
            driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            if track_drive:
                phase = 'done' if driver.base_state.done else driver.base_state.phase
                if phase == 'drive':
                    pos = env.base_pos[:2].copy()
                    if drive_start is None:
                        drive_start = pos
                    drive_end = pos
            fresh, _ready = driver.act(env, t, stride * dt)
            cmd = np.asarray(fresh, dtype=float)
            tick_t.append(t)
            tick_v.append(float(forward_speed(cmd)))
        cmd_speed.append(float(forward_speed(cmd)))
        actual.append(body_forward(env))
        dist = env.capture_distance(t)
        if dist is not None and catch_time is None and dist < 0.01:
            catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None and np.linalg.norm(drive_end - drive_start) > 0.05:
        chord = (drive_end - drive_start).astype(np.float64)
        chord = chord / np.linalg.norm(chord)
    return dict(base=np.asarray(base, np.float64), cmd=np.asarray(cmd_speed), actual=np.asarray(actual),
                tick_t=np.asarray(tick_t), tick_v=np.asarray(tick_v), chord=chord, dt=dt)


def tracking(cmd, actual, dt, stop):
    """How the body speed sits relative to the command around ``stop``."""
    if stop is None or stop < 1 or stop >= len(actual):
        return None
    one = int(round(1.0 / dt))
    ramp = slice(max(0, stop - one), stop)
    after = slice(stop, min(len(actual), stop + one))
    fallen = np.flatnonzero(actual[stop:] <= MOVE_V)
    lag_s = None if len(fallen) == 0 else float(fallen[0] * dt)
    return dict(
        actual_minus_cmd_ramp_mm_s=float(np.mean(actual[ramp] - cmd[ramp]) * 1000.0),
        actual_at_stop_mm_s=float(actual[stop] * 1000.0),
        cmd_at_stop_mm_s=float(cmd[stop] * 1000.0),
        mean_actual_after_mm_s=float(np.mean(actual[after]) * 1000.0),
        coast_mm=float(np.sum(actual[after]) * dt * 1000.0),
        lag_s=lag_s,
    )


def job(seed):
    from catch_env import ExpertDriver
    from diagnose import HandoffDriver
    from np_policy import PolicyDriver
    expert = episode(ExpertDriver(), seed, track_drive=True)
    stopped = stop_time(expert['tick_t'], expert['tick_v'])
    if stopped is None or expert['chord'] is None:
        return dict(scene=int(seed), skipped='no quiet stop')
    stop = int(round(stopped / expert['dt']))
    later = stop + int(round(1.0 / expert['dt']))
    expert_track = tracking(expert['cmd'], expert['actual'], expert['dt'], stop)
    per_seed = {}
    for model_seed, model in zip(SEEDS, _models):
        driver = HandoffDriver(ExpertDriver(), PolicyDriver(model), stopped - 1.0)
        policy = episode(driver, seed, track_drive=False)
        row = tracking(policy['cmd'], policy['actual'], policy['dt'], stop)
        if row is None:
            per_seed[f's{model_seed}'] = None
            continue
        row['along_1s_mm'] = along_at(policy['base'], expert['base'], expert['chord'], later)
        per_seed[f's{model_seed}'] = row
    return dict(scene=int(seed), expert=expert_track, seeds=per_seed)


def column(rows, who, key):
    vals = []
    for row in rows:
        block = row.get(who) if who == 'expert' else row.get('seeds', {}).get(who)
        if not block or block.get(key) is None:
            continue
        vals.append(block[key])
    stat = mean_ci(vals)
    if key != 'along_1s_mm':
        stat.pop('frac_ahead', None)
    return stat


def summarise(rows):
    used = [r for r in rows if 'expert' in r]
    keys = ('actual_minus_cmd_ramp_mm_s', 'actual_at_stop_mm_s', 'cmd_at_stop_mm_s',
            'mean_actual_after_mm_s', 'coast_mm', 'lag_s', 'along_1s_mm')
    out = {'expert': {key: column(used, 'expert', key) for key in keys if key != 'along_1s_mm'}}
    for model_seed in SEEDS:
        name = f's{model_seed}'
        out[name] = {key: column(used, name, key) for key in keys}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:args.limit]
    full = args.limit == 200
    t0 = time.time()
    run_id = record_path = record = None
    if full:
        from stage2_log import start_run
        run_id, record_path, record = start_run(
            'command-vs-actual', CEILING, dict(rate_hz=RATE_HZ, switch_before_stop_s=1.0),
            sys.argv, dict(test=len(test), seeds=list(SEEDS)),
            note='commanded and actual forward speed on the 25 Hz hand-off')
    print(f'speed tracking on {len(test)} test scenes', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(job, int(s)) for s in test]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    stats = summarise(rows)
    skipped = [r for r in rows if 'skipped' in r]
    report = dict(n_test=len(test), n_used=len(rows) - len(skipped), n_skipped=len(skipped), stats=stats)

    def line(name, block):
        along = block.get('along_1s_mm', {}).get('mean')
        along_s = '' if along is None else f'  along {along:+.2f} mm'
        lag = block['lag_s']['mean']
        lag_s = 'n/a' if lag is None else f'{lag * 1000:.0f} ms'
        print(f"{name:<8} ramp {block['actual_minus_cmd_ramp_mm_s']['mean']:+.2f} mm/s  "
              f"at stop actual {block['actual_at_stop_mm_s']['mean']:+.1f}  "
              f"after {block['mean_actual_after_mm_s']['mean']:+.1f}  "
              f"coast {block['coast_mm']['mean']:+.2f} mm  "
              f"lag {lag_s}{along_s}", flush=True)

    line('expert', stats['expert'])
    for model_seed in SEEDS:
        line(f's{model_seed}', stats[f's{model_seed}'])
    if full:
        from stage2_log import finish_run
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2))
        finish_run(record_path, record, dict(miss_mm=stats['s0']['along_1s_mm']['mean']), [OUT])
        print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)
        print(f'run {run_id}', flush=True)
    else:
        print(f'smoke done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
