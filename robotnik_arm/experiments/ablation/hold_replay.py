"""Replay the expert's commands open-loop through the stop.

The expert runs at 1 kHz and the stop is logged. From one second before that
stop, the same commands are applied three ways:

- every millisecond, as the expert applied them;
- sampled every 40 ms and held, as a 25 Hz policy executes;
- sampled every 20 ms and held, as a 50 Hz policy executes.

The hold grid starts at the switch, so the first sample is the expert command
at that instant. The along-track gap is the replay base minus the expert base,
in the expert's route frame. A 1 kHz gap near 0 mm means the replay is
deterministic. The 40 ms and 20 ms gaps are the hold itself, with no network.
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

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'stage2' / 'hold_replay.json'
MOVE_V = 0.02
STOP_TAIL_S = 0.4
WHEEL_RADIUS = 0.11
SWITCH_BEFORE_S = 1.0
MODES = (('khz', 0.001), ('hold40', 0.040), ('hold20', 0.020))

_env = None


def forward_speed(cmd):
    wheels = np.asarray(cmd, dtype=np.float64)[..., :4]
    right = 0.5 * (wheels[..., 0] + wheels[..., 3])
    left = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return 0.5 * (right + left) * WHEEL_RADIUS


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    _env = CatchEnv()


def stop_index(speed, dt):
    """First sample after the last driving command, if the command then stays near zero."""
    moving = np.flatnonzero(np.asarray(speed) > MOVE_V)
    if len(moving) == 0:
        return None
    stop = int(moving[-1]) + 1
    if stop >= len(speed):
        return None
    tail_n = int(round(STOP_TAIL_S / dt))
    tail = np.asarray(speed)[stop:stop + tail_n]
    if len(tail) < max(1, tail_n // 2) or float(np.median(tail)) > 0.01:
        return None
    return stop


def simulate(seed, logged=None, switch=None, hold=1, record_obs=False):
    """One episode. With ``logged`` set, commands from ``switch`` are replayed and held."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    env.reset(int(seed))
    driver = ExpertDriver()
    driver.reset(env)
    dt = float(env.dt)
    stride = int(round(1.0 / (1000.0 * dt)))
    if stride != 1 or abs(stride * dt * 1000.0 - 1.0) > 1e-9:
        raise RuntimeError(f'expert replay expects a 1 ms step, got dt={dt}')
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    base, speed, cmds, obs = [], [], [], []
    drive_start = drive_end = None
    best_dist, best_step = None, None
    catch_time = None
    live = logged is None
    step = 0
    while True:
        t = step * dt
        base.append(env.base_pos[:2].copy())
        if record_obs:
            obs.append(env.obs(t).copy())
        if live or step < switch:
            driver.sense(env, t)
            if step >= FIRST_TICK_STEP:
                fresh, _ready = driver.act(env, t, stride * dt)
                cmd = np.asarray(fresh, dtype=np.float64)
                phase = 'done' if driver.base_state.done else driver.base_state.phase
                if phase == 'drive':
                    pos = env.base_pos[:2].copy()
                    if drive_start is None:
                        drive_start = pos
                    drive_end = pos
        elif (step - switch) % hold == 0 and step < len(logged):
            cmd = np.asarray(logged[step], dtype=np.float64)
        speed.append(float(forward_speed(cmd)))
        if live:
            cmds.append(cmd.copy())
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if best_dist is None or dist < best_dist:
                best_dist, best_step = float(dist), step
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None and np.linalg.norm(drive_end - drive_start) > 0.05:
        chord = (drive_end - drive_start).astype(np.float64)
        chord = chord / np.linalg.norm(chord)
    out = dict(base=np.asarray(base, np.float64), speed=np.asarray(speed, np.float64),
               cmd=None if not cmds else np.asarray(cmds, np.float64), chord=chord,
               catch_step=best_step, dt=dt)
    if record_obs:
        out['obs'] = np.asarray(obs, np.float32)
    return out


def along_at(replay_base, expert_base, chord, step):
    if chord is None or step is None or step < 0 or step >= len(replay_base) or step >= len(expert_base):
        return None
    return float(np.dot(replay_base[step] - expert_base[step], chord) * 1000.0)


def command_extra_mm(replay_speed, expert_speed, dt, start, end):
    if start is None or end is None or start < 0 or end > len(replay_speed) or end > len(expert_speed):
        return None
    return float(np.sum(replay_speed[start:end] - expert_speed[start:end]) * dt * 1000.0)


def job(seed):
    ref = simulate(seed)
    stop = stop_index(ref['speed'], ref['dt'])
    switch = None if stop is None else stop - int(round(SWITCH_BEFORE_S / ref['dt']))
    if stop is None or ref['chord'] is None or switch is None or switch < 2:
        return dict(seed=int(seed), skipped='no quiet stop' if stop is None else 'switch too early')
    later = stop + int(round(1.0 / ref['dt']))
    modes = {}
    for name, period in MODES:
        hold = int(round(period / ref['dt']))
        rep = simulate(seed, logged=ref['cmd'], switch=switch, hold=hold)
        modes[name] = dict(
            at_switch_mm=along_at(rep['base'], ref['base'], ref['chord'], switch),
            at_stop_mm=along_at(rep['base'], ref['base'], ref['chord'], stop),
            one_s_after_stop_mm=along_at(rep['base'], ref['base'], ref['chord'], later),
            at_catch_mm=along_at(rep['base'], ref['base'], ref['chord'], ref['catch_step']),
            command_extra_mm=command_extra_mm(rep['speed'], ref['speed'], ref['dt'], switch, later),
        )
    return dict(seed=int(seed), stop_time=float(stop * ref['dt']), modes=modes)


def mean_ci(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = int(len(values))
    if n == 0:
        return dict(n=0, mean=None, median=None, ci95=(None, None), frac_ahead=None)
    mean = float(np.mean(values))
    se = 0.0 if n < 2 else float(np.std(values, ddof=1) / np.sqrt(n))
    return dict(n=n, mean=mean, median=float(np.median(values)),
                ci95=(mean - 1.96 * se, mean + 1.96 * se),
                frac_ahead=float(np.mean(values > 0.0)))


def summarise(rows):
    used = [r for r in rows if 'modes' in r]
    out = {}
    keys = ('at_switch_mm', 'at_stop_mm', 'one_s_after_stop_mm', 'at_catch_mm', 'command_extra_mm')
    for name, _period in MODES:
        block = {}
        for key in keys:
            stat = mean_ci([r['modes'][name][key] for r in used])
            if key == 'command_extra_mm':
                stat.pop('frac_ahead', None)
            block[key] = stat
        out[name] = block
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
            'hold-replay', 'expert', dict(switch_before_stop_s=SWITCH_BEFORE_S),
            sys.argv, dict(test=len(test)),
            note='open-loop replay of the expert command at 1 kHz, 40 ms hold and 20 ms hold')
    print(f'replay on {len(test)} test scenes', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(job, int(s)) for s in test]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    stats = summarise(rows)
    skipped = [r for r in rows if 'skipped' in r]
    report = dict(
        switch_before_stop_s=SWITCH_BEFORE_S,
        n_test=len(test), n_used=len(rows) - len(skipped), n_skipped=len(skipped),
        modes=stats,
    )
    labels = dict(khz='1 kHz', hold40='40 ms hold', hold20='20 ms hold')
    for name, label in labels.items():
        gap = stats[name]['one_s_after_stop_mm']
        extra = stats[name]['command_extra_mm']
        catch = stats[name]['at_catch_mm']
        print(f"{label:<12} 1s-after {gap['mean']:+.2f} mm  "
              f"median {gap['median']:+.2f}  ahead {gap['frac_ahead']:.2f}  "
              f"command {extra['mean']:+.2f} mm  at catch {catch['mean']:+.2f} mm", flush=True)
    if full:
        from stage2_log import finish_run
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2))
        gap = stats['hold40']['one_s_after_stop_mm']['mean']
        finish_run(record_path, record, dict(miss_mm=gap), [OUT])
        print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)
        print(f'run {run_id}', flush=True)
    else:
        print(f'smoke done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
