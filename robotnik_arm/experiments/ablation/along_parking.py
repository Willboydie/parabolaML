"""Park the expert forward or back after it reaches its goal, and see how it catches.

Once navigation reports done, the base is slid 50 mm or 100 mm forward or back
along its parked heading. Yaw is unchanged. This is the sideways slide with the
other axis. A catch while the base stays parked is an arm reach. The ball's
offset from the planned intercept says whether the catch moved along the flight.
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
OUT = RESULTS / 'stage2' / 'along_parking.json'
RATE_HZ = 1000
ALONGS = (0.05, -0.05, 0.10, -0.10)

_env = None


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    _env = CatchEnv()


def slide(env, along):
    """Slide the parked base along its heading. Positive is forward."""
    client, rid = env.client, env.rid
    bearing = float(env.bearing)
    forward = np.array([np.cos(bearing), np.sin(bearing)])
    pos, orn = client.getBasePositionAndOrientation(rid)
    pos = [pos[0] + forward[0] * along, pos[1] + forward[1] * along, pos[2]]
    client.resetBasePositionAndOrientation(rid, pos, orn)
    client.resetBaseVelocity(rid, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    env._read_state()
    return forward


def park_job(seed, along):
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
    forward = None
    post_base = post_ee = intercept = None
    t_intercept = None
    best = None
    left_done = False
    step = 0
    while True:
        t = step * env.dt
        dist = env.capture_distance(t)
        if shifted and dist is not None:
            if best is None or dist < best['dist']:
                best = dict(dist=float(dist), base=env.base_pos[:2].copy(), ee=env.ee.copy(),
                            ball=env.ball(t)[:2].copy())
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            fresh, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(fresh, dtype=float)
            if shifted:
                if not driver.base_state.done and not left_done:
                    left_done = True
            elif driver.base_state.done:
                forward = slide(env, along)
                post_base = env.base_pos[:2].copy()
                post_ee = env.ee.copy()
                point = driver.expert.intercept
                intercept = None if point is None else np.asarray(point, float)[:2]
                t_intercept = driver.expert.intercept_time
                shifted = True
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    sign = np.sign(along)
    base_undo = ee_undo = arm_undo = ball_from_intercept = None
    if shifted and best is not None:
        base_undo = float(-np.dot(best['base'] - post_base, forward) * sign * 1000)
        ee_undo = float(-np.dot(best['ee'][:2] - post_ee[:2], forward) * sign * 1000)
        arm_rel = (best['ee'][:2] - best['base']) - (post_ee[:2] - post_base)
        arm_undo = float(-np.dot(arm_rel, forward) * sign * 1000)
        if intercept is not None:
            ball_from_intercept = float(np.dot(best['ball'] - intercept, forward) * 1000)
    if not shifted:
        how = 'never_parked'
    elif catch_time is not None and not left_done:
        how = 'reach'
    elif catch_time is not None and left_done:
        how = 'redrive'
    else:
        how = 'miss'
    return dict(seed=int(seed), along_mm=int(round(along * 1000)), how=how,
                success=catch_time is not None,
                miss_mm=None if best is None else best['dist'] * 1000,
                base_undo_mm=base_undo, ee_undo_mm=ee_undo, arm_undo_mm=arm_undo,
                ball_from_intercept_mm=ball_from_intercept,
                catch_minus_intercept_s=None if catch_time is None or t_intercept is None
                else float(catch_time - t_intercept))


def _med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return None if not vals else float(np.median(vals))


def summarize(rows):
    out = {}
    for along in ALONGS:
        key = int(round(along * 1000))
        got = [r for r in rows if r['along_mm'] == key]
        counts = {name: sum(r['how'] == name for r in got) for name in ('reach', 'redrive', 'miss', 'never_parked')}
        misses = [r['miss_mm'] for r in got if r['miss_mm'] is not None]
        caught = [r for r in got if r['success']]
        out[str(key)] = dict(
            n=len(got),
            catch_rate=float(np.mean([r['success'] for r in got])) if got else None,
            median_miss_mm=None if not misses else float(np.median(misses)),
            how=counts,
            base_undo_mm=_med(caught, 'base_undo_mm'),
            ee_undo_mm=_med(caught, 'ee_undo_mm'),
            arm_undo_mm=_med(caught, 'arm_undo_mm'),
            ball_from_intercept_mm=_med(caught, 'ball_from_intercept_mm'),
            catch_minus_intercept_s=_med(caught, 'catch_minus_intercept_s'))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    seeds = json.loads((ROOT / 'data' / 'seeds.json').read_text())['train'][:args.limit]
    jobs = [(int(seed), along) for seed in seeds for along in ALONGS]
    full = args.limit == 20
    t0 = time.time()
    run_id = record_path = record = None
    if full:
        from stage2_log import start_run
        run_id, record_path, record = start_run(
            'along-parking', 'expert', dict(along_mm=[int(a * 1000) for a in ALONGS], rate_hz=RATE_HZ),
            sys.argv, dict(train=len(seeds)),
            note='slide the parked expert 50 and 100 mm forward and back')
    print(f'{len(jobs)} along-track slides, {len(seeds)} seeds', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(park_job, seed, along) for seed, along in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 16 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    summary = summarize(rows)
    print(f'{"mm":>6} {"catch":>7} {"miss":>8} {"reach":>6} {"redrive":>8} {"missed":>6} '
          f'{"arm":>8} {"ball":>8}', flush=True)
    for key, item in summary.items():
        miss = item['median_miss_mm']
        arm = item['arm_undo_mm']
        ball = item['ball_from_intercept_mm']
        def fmt(v):
            return 'n/a' if v is None else f'{v:.0f}'
        miss_s = 'n/a' if miss is None else f'{miss:.2f}'
        print(f'{key:>6} {item["catch_rate"]:7.2f} {miss_s:>8} {item["how"]["reach"]:6d} '
              f'{item["how"]["redrive"]:8d} {item["how"]["miss"]:6d} {fmt(arm):>8} {fmt(ball):>8}', flush=True)
    if full:
        from stage2_log import finish_run
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(rate_hz=RATE_HZ, n_seeds=len(seeds), summary=summary), indent=2))
        caught = summary.get('50', {}).get('catch_rate')
        finish_run(record_path, record, dict(miss_mm=summary.get('50', {}).get('median_miss_mm'),
                                             catch_rate=caught), [OUT])
        print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)
        print(f'run {run_id}', flush=True)
    else:
        print(f'smoke done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
