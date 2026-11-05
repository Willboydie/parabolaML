"""Two checks on the expert's navigation phase, with no training.

Query consistency: roll a clean expert at 25 Hz and, at every tick, also ask a
freshly started expert for its command in that same state (same plan, navigation
phase reset to align, arm reference reset to the measured angle). Compare with
the command the running expert actually issued.

Perturbation: once the running expert is mid-drive, shift its heading by a few
degrees or its position by 5–10 cm, leave the phase machine where it is, and
see whether it keeps driving and still catches.
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
RATE_HZ = 25
WHEEL_RADIUS = 0.11
WHEEL_WIDTH = 0.66
ANG_TOL = 0.05  # rad, the align-phase tolerance in BaseControllerParams

_env = None


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def wheel_vw(cmd):
    rw, lw = float(cmd[0]), float(cmd[1])
    v = 0.5 * (rw + lw) * WHEEL_RADIUS
    w = (rw - lw) * WHEEL_RADIUS / WHEEL_WIDTH
    return v, w


def is_yaw(cmd):
    v, w = wheel_vw(cmd)
    return abs(v) < 0.02 and abs(w) > 0.2


def is_drive(cmd):
    v, _w = wheel_vw(cmd)
    return abs(v) > 0.04


def heading_error(env, driver):
    from simulation_and_control import wrap_angle
    to_goal = driver.goal_pos - env.base_pos[:2]
    if np.linalg.norm(to_goal) < 1e-6:
        return 0.0
    heading = np.arctan2(to_goal[1], to_goal[0])
    return float(wrap_angle(heading - env.bearing))


def fresh_wheels(env, driver, t, dt):
    """Command a freshly started phase machine would issue, then restore the running one."""
    from robotnik_arm_control import BaseNavState
    saved = dict(phase=driver.base_state.phase, done=driver.base_state.done,
                 q=driver.q_des.copy(), qd=driver.qd_des.copy(),
                 goal=driver.goal_pos.copy(), bearing=driver.goal_bearing,
                 planned=driver.planning_attempted)
    driver.base_state = BaseNavState()
    driver.q_des = env.q[4:11].copy()
    driver.qd_des = np.zeros(7)
    cmd, ready = driver.act(env, t, dt)
    driver.base_state = BaseNavState()
    driver.base_state.phase = saved['phase']
    driver.base_state.done = saved['done']
    driver.q_des = saved['q']
    driver.qd_des = saved['qd']
    driver.goal_pos = saved['goal']
    driver.goal_bearing = saved['bearing']
    driver.planning_attempted = saved['planned']
    return np.asarray(cmd, dtype=float), bool(ready)


def query_job(seed):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    counts = {p: dict(n=0, wheel_agree=0, fresh_yaw=0, issued_drive=0, fresh_yaw_issued_drive=0,
                      err_over_tol=0, wheel_l2=[]) for p in ('align', 'drive', 'orient', 'done')}
    step = 0
    while True:
        t = step * env.dt
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            phase = driver.base_state.phase if not driver.base_state.done else 'done'
            err = heading_error(env, driver)
            cmd, ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
            if ready:
                fresh, fresh_ready = fresh_wheels(env, driver, t, stride * env.dt)
                if fresh_ready:
                    bucket = counts[phase]
                    bucket['n'] += 1
                    diff = float(np.max(np.abs(cmd[:4] - fresh[:4])))
                    bucket['wheel_l2'].append(float(np.linalg.norm(cmd[:4] - fresh[:4])))
                    if diff < 0.05:
                        bucket['wheel_agree'] += 1
                    fresh_yaw = is_yaw(fresh)
                    issued_drive = is_drive(cmd)
                    if fresh_yaw:
                        bucket['fresh_yaw'] += 1
                    if issued_drive:
                        bucket['issued_drive'] += 1
                    if fresh_yaw and issued_drive:
                        bucket['fresh_yaw_issued_drive'] += 1
                    if abs(err) > ANG_TOL:
                        bucket['err_over_tol'] += 1
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
    for bucket in counts.values():
        series = bucket.pop('wheel_l2')
        bucket['wheel_l2_median'] = float(np.median(series)) if series else None
    miss = float(env.min_dist)
    return dict(seed=int(seed), success=catch_time is not None,
                miss_mm=None if not np.isfinite(miss) else miss * 1000, counts=counts)


def shift_pose(env, heading=0.0, lateral=0.0):
    """Yaw the base by ``heading`` radians, or slide it sideways by ``lateral`` metres."""
    client, rid = env.client, env.rid
    before = env.bearing
    pos, orn = client.getBasePositionAndOrientation(rid)
    if heading:
        roll, pitch, yaw = client.getEulerFromQuaternion(orn)
        orn = client.getQuaternionFromEuler([roll, pitch, yaw + heading])
    if lateral:
        pos = list(pos)
        pos[0] += -np.sin(before) * lateral
        pos[1] += np.cos(before) * lateral
    client.resetBasePositionAndOrientation(rid, pos, orn)
    env._read_state()
    from simulation_and_control import wrap_angle
    return float(wrap_angle(env.bearing - before))


def perturb_job(seed, kind, magnitude):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    drive_start = None
    applied = False
    achieved = None
    t_perturb = None
    t_first_align = None
    align_was_replan = None
    intercept_before_align = None
    after = dict(align=0, drive=0, orient=0, done=0, yaw=0, drive_cmd=0)
    first_vw = None
    first_phase = None
    step = 0
    while True:
        t = step * env.dt
        if (not applied and kind != 'none' and driver.base_state.phase == 'drive' and not driver.base_state.done):
            dist = float(np.linalg.norm(env.base_pos[:2] - driver.goal_pos))
            if drive_start is None:
                drive_start = dist
            elif drive_start - dist > 0.2 and dist > 0.4:
                kwargs = dict(heading=magnitude) if kind == 'heading' else dict(lateral=magnitude)
                achieved = shift_pose(env, **kwargs)
                applied = True
                t_perturb = t
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            pre_intercept = driver.expert.intercept_time
            pre_missed = (pre_intercept is not None and t > pre_intercept + 1.0
                          and driver.expert.intercept is not None
                          and np.linalg.norm(env.ball(t) - driver.expert.intercept) > 0.01)
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
            if applied:
                phase = 'done' if driver.base_state.done else driver.base_state.phase
                after[phase] += 1
                if is_yaw(cmd):
                    after['yaw'] += 1
                if is_drive(cmd):
                    after['drive_cmd'] += 1
                if first_vw is None:
                    first_vw = wheel_vw(cmd)
                    first_phase = phase
                if phase == 'align' and t_first_align is None:
                    t_first_align = t
                    align_was_replan = bool(pre_missed)
                    intercept_before_align = None if pre_intercept is None else float(pre_intercept)
        q_ref = env.clip_reference(cmd[4:11])
        applied_tau = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied_tau[:4], applied_tau[4:])
        step += 1
    miss = float(env.min_dist)
    return dict(seed=int(seed), kind=kind, magnitude=magnitude, applied=applied,
                achieved_heading_rad=achieved, t_perturb=t_perturb,
                success=catch_time is not None,
                miss_mm=None if not np.isfinite(miss) else miss * 1000,
                intercept_time=None if driver.expert.intercept_time is None else float(driver.expert.intercept_time),
                t_first_align=t_first_align, align_was_replan=align_was_replan,
                intercept_before_align=intercept_before_align, first_phase=first_phase,
                realigned=after['align'] > 0, after=after,
                first_v=None if first_vw is None else first_vw[0],
                first_w=None if first_vw is None else first_vw[1])


def summarize_perturb(rows):
    applied = [r for r in rows if r['applied'] or r['kind'] == 'none']
    misses = [r['miss_mm'] for r in applied if r['miss_mm'] is not None]
    return dict(n=len(rows), applied=len(applied),
                catch_rate=float(np.mean([r['success'] for r in applied])) if applied else None,
                miss_mm_median=float(np.median(misses)) if misses else None,
                miss_mm_max=float(np.max(misses)) if misses else None,
                realigned=int(sum(r['realigned'] for r in applied)),
                yaw_tick_frac=float(np.sum([r['after']['yaw'] for r in applied]) /
                                    max(1, np.sum([sum(r['after'][p] for p in ('align', 'drive', 'orient', 'done'))
                                                   for r in applied]))))


def main():
    seeds = json.loads((ROOT / 'data' / 'seeds.json').read_text())['train'][:N]
    conditions = [('none', 0.0)]
    for deg in (5.0, 10.0):
        for sign in (1.0, -1.0):
            conditions.append(('heading', sign * np.deg2rad(deg)))
    for cm in (0.05, 0.10):
        for sign in (1.0, -1.0):
            conditions.append(('lateral', sign * cm))
    print(f'{N} seeds, query consistency plus {len(conditions)} perturbation conditions', flush=True)
    t0 = time.time()
    queries, perturbs = [], []
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        futs = {pool.submit(query_job, int(s)): ('query', int(s)) for s in seeds}
        for seed in seeds:
            for kind, mag in conditions:
                futs[pool.submit(perturb_job, int(seed), kind, float(mag))] = ('perturb', kind, mag)
        done = 0
        for fut in as_completed(futs):
            kind = futs[fut][0]
            row = fut.result()
            (queries if kind == 'query' else perturbs).append(row)
            done += 1
            if done % 40 == 0 or done == len(futs):
                print(f'  {done}/{len(futs)} ({time.time() - t0:.0f}s)', flush=True)
    phases = {}
    for phase in ('align', 'drive', 'orient', 'done'):
        acc = dict(n=0, wheel_agree=0, fresh_yaw=0, issued_drive=0, fresh_yaw_issued_drive=0, err_over_tol=0)
        meds = []
        for row in queries:
            bucket = row['counts'][phase]
            for key in acc:
                acc[key] += bucket[key]
            if bucket['wheel_l2_median'] is not None:
                meds.append(bucket['wheel_l2_median'])
        acc['agree_frac'] = acc['wheel_agree'] / acc['n'] if acc['n'] else None
        acc['fresh_yaw_frac'] = acc['fresh_yaw'] / acc['n'] if acc['n'] else None
        acc['issued_drive_frac'] = acc['issued_drive'] / acc['n'] if acc['n'] else None
        acc['yaw_instead_of_drive_frac'] = acc['fresh_yaw_issued_drive'] / acc['n'] if acc['n'] else None
        acc['err_over_tol_frac'] = acc['err_over_tol'] / acc['n'] if acc['n'] else None
        acc['wheel_l2_median_across_eps'] = float(np.median(meds)) if meds else None
        phases[phase] = acc
    groups = {}
    for kind, mag in conditions:
        rows = [r for r in perturbs if r['kind'] == kind and abs(r['magnitude'] - mag) < 1e-12]
        label = 'none' if kind == 'none' else f'{kind} {mag:+.4f}'
        groups[label] = summarize_perturb(rows)
    report = dict(n_seeds=N, rate_hz=RATE_HZ, ang_tol_rad=ANG_TOL,
                  query_catch_rate=float(np.mean([r['success'] for r in queries])),
                  phases=phases, perturbations=groups, elapsed_s=time.time() - t0)
    out = RESULTS / 'expert_phase_tests.json'
    out.write_text(json.dumps(report, indent=2))
    print(f"query rollouts catch {report['query_catch_rate']:.2f} (restore sanity)", flush=True)
    print('phase     ticks  wheel agree  |err|>tol  issued drive  fresh yaw  yaw-not-drive', flush=True)
    for phase, acc in phases.items():
        if not acc['n']:
            continue
        print(f"{phase:8} {acc['n']:6d}  {acc['agree_frac']:11.2%}  {acc['err_over_tol_frac']:8.2%}  "
              f"{acc['issued_drive_frac']:12.2%}  {acc['fresh_yaw_frac']:9.2%}  "
              f"{acc['yaw_instead_of_drive_frac']:13.2%}", flush=True)
    print('condition                 applied  catch  miss med  miss max  realigned  yaw ticks', flush=True)
    for label, g in groups.items():
        catch = g['catch_rate']
        med, mx = g['miss_mm_median'], g['miss_mm_max']
        print(f"{label:24} {g['applied']:4d}  "
              f"{'n/a' if catch is None else f'{catch:.2f}'}  "
              f"{'n/a' if med is None else f'{med:7.2f}'}  "
              f"{'n/a' if mx is None else f'{mx:7.1f}'}  {g['realigned']:4d}  "
              f"{g['yaw_tick_frac']:.2%}", flush=True)
    print(f'wrote {out}', flush=True)


def detail():
    """When a 10 degree nudge misses, did align resume during the approach or only after the miss?"""
    seeds = json.loads((ROOT / 'data' / 'seeds.json').read_text())['train'][:N]
    jobs = [(int(s), 'heading', float(sign * np.deg2rad(10.0))) for s in seeds for sign in (1.0, -1.0)]
    rows = []
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        futs = [pool.submit(perturb_job, s, k, m) for s, k, m in jobs]
        for fut in as_completed(futs):
            rows.append(fut.result())
    fails = [r for r in rows if not r['success']]
    print(f'{len(rows)} episodes, {len(fails)} misses', flush=True)
    print('sign  miss_mm  t_nudge  t_align  intercept_before  replan  first_phase  first_v  first_w',
          flush=True)
    for r in sorted(fails, key=lambda r: (r['magnitude'], r['t_perturb'])):
        before = r['intercept_before_align']
        print(f"{np.rad2deg(r['magnitude']):+5.0f}  {r['miss_mm']:8.0f}  {r['t_perturb']:7.2f}  "
              f"{r['t_first_align']:7.2f}  {'n/a' if before is None else f'{before:16.2f}'}  "
              f"{str(r['align_was_replan']):5}  {r['first_phase']:8}  {r['first_v']:+7.3f}  {r['first_w']:+7.3f}",
              flush=True)
    caught = [r for r in rows if r['success']]
    n_realign_caught = sum(r['realigned'] for r in caught)
    print(f'catches {len(caught)}, of which realigned {n_realign_caught}', flush=True)
    if caught:
        print(f"caught first phase { {p: sum(r['first_phase']==p for r in caught) for p in ('drive','align','orient','done')} }",
              flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--detail':
        detail()
    else:
        main()
