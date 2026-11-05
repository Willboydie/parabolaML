"""Park the expert sideways after it reaches its goal, and see how it still catches.

Once navigation reports done, the base is slid 50 mm or 100 mm to the left or
right of its parked heading. Yaw is unchanged and the phase machine is left
where it is. A catch with the base still sitting there is an arm reach. A catch
after the phase leaves done is a re-drive, which only happens on the miss replan.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
N = 20
RATE_HZ = 1000
LATERALS = (0.05, -0.05, 0.10, -0.10)

_env = None


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def slide(env, lateral):
    """Slide the parked base sideways by ``lateral`` metres. Positive is left of the heading."""
    client, rid = env.client, env.rid
    bearing = float(env.bearing)
    pos, orn = client.getBasePositionAndOrientation(rid)
    pos = [pos[0] - np.sin(bearing) * lateral, pos[1] + np.cos(bearing) * lateral, pos[2]]
    client.resetBasePositionAndOrientation(rid, pos, orn)
    client.resetBaseVelocity(rid, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    env._read_state()
    left = np.array([-np.sin(bearing), np.cos(bearing)])
    return left


def park_job(seed, lateral):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    shifted = False
    t_shift = None
    left = None
    post_base = post_ee = None
    best = None
    left_done = False
    t_left_done = None
    wheel_peak = 0.0
    step = 0
    while True:
        t = step * env.dt
        dist = env.capture_distance(t)
        if shifted and dist is not None:
            if best is None or dist < best['dist']:
                best = dict(dist=float(dist), base=env.base_pos[:2].copy(), ee=env.ee.copy(),
                            ball=env.ball(t).copy())
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
            if shifted:
                phase_done = bool(driver.base_state.done)
                if not phase_done and not left_done:
                    left_done = True
                    t_left_done = t
                wheel_peak = max(wheel_peak, float(np.max(np.abs(cmd[:4]))))
            elif driver.base_state.done:
                left = slide(env, lateral)
                post_base = env.base_pos[:2].copy()
                post_ee = env.ee.copy()
                t_shift = t
                shifted = True
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    miss = None if best is None else best['dist'] * 1000
    base_undo = ee_undo = arm_undo = None
    if shifted and best is not None:
        base_undo = float(-np.dot(best['base'] - post_base, left) * np.sign(lateral) * 1000)
        ee_undo = float(-np.dot(best['ee'][:2] - post_ee[:2], left) * np.sign(lateral) * 1000)
        arm_rel = (best['ee'][:2] - best['base']) - (post_ee[:2] - post_base)
        arm_undo = float(-np.dot(arm_rel, left) * np.sign(lateral) * 1000)
    if not shifted:
        how = 'never_parked'
    elif catch_time is not None and not left_done:
        how = 'reach'
    elif catch_time is not None and left_done:
        how = 'redrive'
    else:
        how = 'miss'
    return dict(seed=int(seed), lateral_mm=int(round(lateral * 1000)), how=how,
                success=catch_time is not None, miss_mm=miss,
                parked=shifted, left_done=left_done,
                t_shift=t_shift, t_left_done=t_left_done,
                catch_time=None if catch_time is None else float(catch_time),
                base_undo_mm=base_undo, ee_undo_mm=ee_undo, arm_undo_mm=arm_undo,
                wheel_peak=wheel_peak)


def _med(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return None if not vals else float(np.median(vals))


def summarize(rows):
    out = {}
    for lateral in LATERALS:
        got = [r for r in rows if r['lateral_mm'] == int(round(lateral * 1000))]
        counts = {name: sum(r['how'] == name for r in got) for name in ('reach', 'redrive', 'miss', 'never_parked')}
        caught = [r for r in got if r['success']]
        misses = [r['miss_mm'] for r in got if r['miss_mm'] is not None]
        out[str(int(round(lateral * 1000)))] = dict(
            n=len(got), catch_rate=float(np.mean([r['success'] for r in got])),
            miss_mm=None if not misses else float(np.mean(misses)),
            median_miss_mm=None if not misses else float(np.median(misses)),
            how=counts,
            base_undo_mm=_med(caught, 'base_undo_mm'),
            ee_undo_mm=_med(caught, 'ee_undo_mm'),
            arm_undo_mm=_med(caught, 'arm_undo_mm'),
            wheel_peak=_med(got, 'wheel_peak'))
    return out


def main():
    seeds = json.loads((ROOT / 'data' / 'seeds.json').read_text())['train'][:N]
    jobs = [(int(seed), lateral) for seed in seeds for lateral in LATERALS]
    print(f'{len(jobs)} parked-offset episodes, {N} seeds, {RATE_HZ} Hz', flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        futs = [pool.submit(park_job, seed, lateral) for seed, lateral in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 16 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)} ({time.time() - t0:.0f}s)', flush=True)
    summary = summarize(rows)
    out = RESULTS / 'sideways_parking.json'
    out.write_text(json.dumps(dict(rate_hz=RATE_HZ, n_seeds=N, summary=summary, rows=rows), indent=2))
    print(f'{"mm":>6} {"catch":>7} {"miss":>8} {"reach":>6} {"redrive":>8} {"missed":>6} '
          f'{"base":>8} {"hand":>8} {"arm":>8}', flush=True)
    for key, item in summary.items():
        miss = item['median_miss_mm']
        miss_s = 'n/a' if miss is None else f'{miss:.1f}'
        base = item['base_undo_mm']
        hand = item['ee_undo_mm']
        arm = item['arm_undo_mm']
        def fmt(v):
            return 'n/a' if v is None else f'{v:.0f}'
        print(f'{key:>6} {item["catch_rate"]:7.2f} {miss_s:>8} {item["how"]["reach"]:6d} '
              f'{item["how"]["redrive"]:8d} {item["how"]["miss"]:6d} '
              f'{fmt(base):>8} {fmt(hand):>8} {fmt(arm):>8}', flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
