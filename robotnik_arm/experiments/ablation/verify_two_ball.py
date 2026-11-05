"""Check the existing random-choice expert on two balls. It is not modified.

make_catchable_pair already keeps only balls it can plan a catch for, then draws
the slot uniformly. This runs that expert to the end and records the choice,
whether the choice stays put, and the miss to each ball.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results' / 'stage2'
MASTER = 2026
STREAM = 60
N = 100
SEPARATION = 0.180  # 4 x the 45 mm ceiling radius
RATE_HZ = 1000

_env = None


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def scene_seed(index, attempt):
    return int(np.random.SeedSequence([MASTER, STREAM, index, attempt]).generate_state(1, dtype=np.uint64)[0] >> 1)


def separated_pair(env, index):
    from robotnik_arm_control import make_catchable_pair
    rejected = []
    for attempt in range(40):
        targets, selected, plans = make_catchable_pair(
            scene_seed(index, attempt), None, env.lower, env.upper, env.init_base_pos,
            env.init_bearing, env.init_ee, env.dt, env.base_params, env.arm_params, target_count=2)
        gap = float(np.linalg.norm(plans[0].intercept - plans[1].intercept))
        if gap >= SEPARATION:
            return targets, int(selected), plans, rejected, gap
        rejected.append(dict(reason='intercepts_closer_than_4x_radius', gap_mm=gap * 1000))
    raise RuntimeError(f'no separated pair for scene {index}')


def job(index):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    targets, selected, plans, rejected, gap = separated_pair(env, index)
    env.client.restoreState(env.saved_state)
    env._read_state()
    env.targets = targets
    env.chosen_ball = selected
    env.target = targets[selected]
    env.has_left = False
    env.min_dist = np.inf
    driver = ExpertDriver()
    driver.reset(env)
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    end_time = max(t.duration for t in targets) + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    choices = []
    best = [None, None]
    left = [False, False]
    step = 0
    while True:
        t = step * env.dt
        for slot, target in enumerate(targets):
            ball = target.position(t)
            outside = np.any(ball < env.lower - 1e-6) or np.any(ball > env.upper + 1e-6)
            left[slot] |= bool(outside)
            if left[slot] and not outside:
                dist = float(np.linalg.norm(ball - env.ee))
                if best[slot] is None or dist < best[slot]:
                    best[slot] = dist
                if slot == selected and catch_time is None and dist < 0.01:
                    catch_time = t
        if driver.planning_attempted and getattr(env, 'chosen_ball', selected) != selected:
            choices.append(env.chosen_ball)
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    misses = [None if b is None else b * 1000 for b in best]
    nearer = int(np.linalg.norm(targets[1].position(0) - env.init_base_pos)
                 < np.linalg.norm(targets[0].position(0) - env.init_base_pos))
    earlier = int(plans[1].intercept_time < plans[0].intercept_time)
    return dict(
        index=index, chosen_ball=selected, choice_changed=bool(choices),
        reachable_attempts=[int(p.sample_attempts) for p in plans],
        separation_rejects=len(rejected), separation_mm=gap * 1000,
        launch_gap_s=0.0,
        miss_mm=misses, chosen_miss_mm=misses[selected],
        catch_10mm=misses[selected] is not None and misses[selected] < 10,
        catch_45mm=misses[selected] is not None and misses[selected] < 45,
        slot1_nearer=nearer, slot1_earlier=earlier, chose_slot1=int(selected == 1),
        intercept_times=[float(p.intercept_time) for p in plans])


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print(f'{N} two-ball expert episodes, separation {SEPARATION * 1000:.0f} mm', flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        futs = [pool.submit(job, i) for i in range(N)]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 25 == 0 or n == N:
                print(f'  {n}/{N} ({time.time() - t0:.0f}s)', flush=True)
    chosen = np.array([r['chose_slot1'] for r in rows])
    misses = [r['chosen_miss_mm'] for r in rows if r['chosen_miss_mm'] is not None]
    summary = dict(
        n=N, rate_hz=RATE_HZ, separation_required_mm=SEPARATION * 1000,
        chose_slot1=float(chosen.mean()),
        choice_changed=int(sum(r['choice_changed'] for r in rows)),
        catch_10mm=float(np.mean([r['catch_10mm'] for r in rows])),
        catch_45mm=float(np.mean([r['catch_45mm'] for r in rows])),
        chosen_miss_median_mm=float(np.median(misses)),
        chosen_miss_mean_mm=float(np.mean(misses)),
        separation_mm_median=float(np.median([r['separation_mm'] for r in rows])),
        separation_rejects_mean=float(np.mean([r['separation_rejects'] for r in rows])),
        reachable_attempts_mean=float(np.mean([a for r in rows for a in r['reachable_attempts']])),
        corr_choice_nearer=corr(chosen, [r['slot1_nearer'] for r in rows]),
        corr_choice_earlier=corr(chosen, [r['slot1_earlier'] for r in rows]),
        launch_gap_s=0.0,
        rule='make_catchable_pair rejects a ball until its own plan succeeds, then draws the slot with rng.integers',
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / 'expert_verify.json'
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
