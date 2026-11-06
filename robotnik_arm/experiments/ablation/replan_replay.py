"""Replay the chunk the ceiling issues when it replans at the stop.

The first chunk is the one from ``chunk_replay.py``: queried on the expert
state one second before the stop, and executed for its one second. The base
there is about 2 mm ahead. This script queries the policy on that state and
holds the new chunk for 40 ms per step. The along-track gap one second later
is the instant at which the closed-loop hand-off is about 9.5 mm ahead.
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

from chunk_replay import CEILING, SEEDS, absolute_chunk, history_at
from hold_replay import (
    SWITCH_BEFORE_S, along_at, command_extra_mm, mean_ci, simulate, stop_index,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'stage2' / 'replan_replay.json'
HOLD_S = 0.040
Q_COL = None

_models = None


def init_worker():
    global _models, Q_COL
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv, OBS_NAMES
    from np_policy import NumpyPolicy
    import hold_replay
    hold_replay._env = CatchEnv()
    Q_COL = OBS_NAMES.index('arm_q_0')
    _models = []
    for seed in SEEDS:
        model = NumpyPolicy(RESULTS / 'models' / f'{CEILING}_s{seed}.npz')
        cfg = model.cfg
        if (cfg.rate_hz, cfg.horizon, cfg.execute, cfg.frames) != (25, 50, 25, 13):
            raise RuntimeError((cfg.rate_hz, cfg.horizon, cfg.execute, cfg.frames))
        _models.append(model)


def second_chunk_episode(seed, model, first, switch, replan):
    """Expert until ``switch``, first chunk until ``replan``, then the chunk queried there."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    from hold_replay import forward_speed
    env = __import__('hold_replay')._env
    env.reset(int(seed))
    driver = ExpertDriver()
    driver.reset(env)
    dt = float(env.dt)
    hold = int(round(HOLD_S / dt))
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    base, speed, obs = [], [], []
    second = None
    catch_time = None
    step = 0
    while True:
        t = step * dt
        base.append(env.base_pos[:2].copy())
        obs.append(env.obs(t).copy())
        if step == replan:
            frames = history_at(np.asarray(obs, np.float32), replan, model.cfg.frames, hold)
            q_ref = obs[-1][Q_COL:Q_COL + 7].astype(np.float64)
            second = absolute_chunk(model.predict(frames), q_ref)
        if step < switch:
            driver.sense(env, t)
            if step >= FIRST_TICK_STEP:
                fresh, _ready = driver.act(env, t, dt)
                cmd = np.asarray(fresh, dtype=np.float64)
        elif step < replan:
            cmd = first[(step - switch) // hold]
        else:
            k = min((step - replan) // hold, len(second) - 1)
            cmd = second[k]
        speed.append(float(forward_speed(cmd)))
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        dist = env.capture_distance(t)
        if dist is not None and catch_time is None and dist < 0.01:
            catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied[:4], applied[4:])
        step += 1
    return dict(base=np.asarray(base, np.float64), speed=np.asarray(speed, np.float64), dt=dt)


def job(seed):
    ref = simulate(seed, record_obs=True)
    stop = stop_index(ref['speed'], ref['dt'])
    switch = None if stop is None else stop - int(round(SWITCH_BEFORE_S / ref['dt']))
    if stop is None or ref['chord'] is None or switch is None or switch < 2:
        return dict(scene=int(seed), skipped='no quiet stop' if stop is None else 'switch too early')
    hold = int(round(HOLD_S / ref['dt']))
    replan = switch + _models[0].cfg.execute * hold
    later = replan + _models[0].cfg.execute * hold
    per_seed = {}
    for model_seed, model in zip(SEEDS, _models):
        frames = history_at(ref['obs'], switch, model.cfg.frames, hold)
        q_ref = ref['obs'][switch, Q_COL:Q_COL + 7].astype(np.float64)
        first = absolute_chunk(model.predict(frames), q_ref)
        rep = second_chunk_episode(seed, model, first, switch, replan)
        per_seed[f's{model_seed}'] = dict(
            at_replan_mm=along_at(rep['base'], ref['base'], ref['chord'], replan),
            one_s_later_mm=along_at(rep['base'], ref['base'], ref['chord'], later),
            command_extra_mm=command_extra_mm(rep['speed'], ref['speed'], ref['dt'], replan, later),
        )
    return dict(scene=int(seed), seeds=per_seed)


def summarise(rows):
    used = [r for r in rows if 'seeds' in r]
    keys = ('at_replan_mm', 'one_s_later_mm', 'command_extra_mm')
    per_seed = {}
    for model_seed in SEEDS:
        name = f's{model_seed}'
        block = {}
        for key in keys:
            stat = mean_ci([r['seeds'][name][key] for r in used])
            if key.startswith('command_'):
                stat.pop('frac_ahead', None)
            block[key] = stat
        per_seed[name] = block
    mean = {key: float(np.mean([per_seed[f's{s}'][key]['mean'] for s in SEEDS])) for key in keys}
    return per_seed, mean


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
            'replan-replay', CEILING, dict(switch_before_stop_s=SWITCH_BEFORE_S, hold_s=HOLD_S),
            sys.argv, dict(test=len(test), seeds=list(SEEDS)),
            note='second chunk, queried after the first chunk has run for 1 s')
    print(f'replan replay on {len(test)} test scenes', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(job, int(s)) for s in test]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    per_seed, mean = summarise(rows)
    skipped = [r for r in rows if 'skipped' in r]
    report = dict(n_test=len(test), n_used=len(rows) - len(skipped), n_skipped=len(skipped),
                  per_seed=per_seed, mean=mean)
    for model_seed in SEEDS:
        row = per_seed[f's{model_seed}']
        gap = row['one_s_later_mm']
        print(f"seed {model_seed}  at replan {row['at_replan_mm']['mean']:+.2f} mm  "
              f"1s later {gap['mean']:+.2f}  command {row['command_extra_mm']['mean']:+.2f}  "
              f"ahead {gap['frac_ahead']:.2f}", flush=True)
    print(f"mean     at replan {mean['at_replan_mm']:+.2f} mm  "
          f"1s later {mean['one_s_later_mm']:+.2f}  command {mean['command_extra_mm']:+.2f}", flush=True)
    if full:
        from stage2_log import finish_run
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2))
        finish_run(record_path, record, dict(miss_mm=mean['one_s_later_mm']), [OUT])
        print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)
        print(f'run {run_id}', flush=True)
    else:
        print(f'smoke done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
