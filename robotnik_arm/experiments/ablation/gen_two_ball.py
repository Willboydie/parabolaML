"""2,000 two-ball demos, plus 200 held-out scenes.

The mix matches the one-ball set the ceiling was trained on. The first half is
clean parking, and the odd episodes in that half get DART torque noise. The
second half parks off target (sideways 50 mm, along-track 20 mm, cap 150 mm,
no yaw change) with no noise. The logged command is the expert command that was
sent to the low level. DART noise is added to the applied torque and is not the
label.

Both balls are reachable, they launch together, and a scene is kept only when
the planned intercepts are at least 200 mm apart. Rejected draws are counted.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MASTER = 2026
SCENE_STREAM = 70
NOISE_STREAM = 71
OFFSET_STREAM = 72
TEST_STREAM = 80
ORACLE_STREAM = 81
SEPARATION = 0.200
SIDE_SIGMA = 0.050
ALONG_SIGMA = 0.020
OFFSET_CAP = 0.150
RATE_HZ = 1000

_env = None


def scene_seed(stream, index, attempt):
    return int(np.random.SeedSequence([MASTER, stream, index, attempt]).generate_state(1, dtype=np.uint64)[0] >> 1)


def sample_offset(index):
    rng = np.random.default_rng(scene_seed(OFFSET_STREAM, index, 0))
    side = float(rng.normal(0.0, SIDE_SIGMA))
    along = float(rng.normal(0.0, ALONG_SIGMA))
    mag = float(np.hypot(side, along))
    if mag > OFFSET_CAP:
        side *= OFFSET_CAP / mag
        along *= OFFSET_CAP / mag
    return along, side


def oracle_choice(index):
    return int(np.random.default_rng(scene_seed(ORACLE_STREAM, index, 0)).integers(0, 2))


def separated_pair(env, index, stream):
    from robotnik_arm_control import make_catchable_pair
    rejected = 0
    for attempt in range(80):
        seed = scene_seed(stream, index, attempt)
        targets, selected, plans = make_catchable_pair(
            seed, None, env.lower, env.upper, env.init_base_pos, env.init_bearing, env.init_ee,
            env.dt, env.base_params, env.arm_params, target_count=2)
        gap = float(np.linalg.norm(np.asarray(plans[0].intercept) - np.asarray(plans[1].intercept)))
        if gap >= SEPARATION:
            return dict(seed=int(seed), attempt=int(attempt), targets=targets, selected=int(selected),
                        plans=plans, rejected=int(rejected), gap=float(gap),
                        reachable=[int(p.sample_attempts) for p in plans])
        rejected += 1
    raise RuntimeError(f'no pair at least {SEPARATION * 1000:.0f} mm apart for scene {index}')


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def _rollout(env, driver, noise):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    stride = int(round(1.0 / (RATE_HZ * env.dt)))
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    ready = False
    end_time = max(target.duration for target in env.targets) + CAP_AFTER_ENDPOINT
    wheel_limit = (env.base_params.max_lin_vel + env.base_params.max_ang_vel * env.base_params.wheel_base_width / 2) / env.base_params.wheel_radius
    chosen = int(env.chosen_ball)
    best = [None, None]
    catch_time = None
    rows = dict(obs=[], cmd=[], applied=[], ready=[])
    step = 0
    while True:
        t = step * env.dt
        if hasattr(driver, 'sense'):
            driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        if noise is not None and ready:
            applied = applied + noise(t)
            applied[:4] = np.clip(applied[:4], -wheel_limit, wheel_limit)
            applied[4:] = np.clip(applied[4:], -env.arm_params.max_torque, env.arm_params.max_torque)
        for slot in range(2):
            dist = env.slot_distance(slot, t)
            if dist is not None:
                best[slot] = dist if best[slot] is None else min(best[slot], dist)
            if slot == chosen and dist is not None and catch_time is None and dist < 0.01:
                catch_time = t
        rows['obs'].append(env.obs_set(t))
        rows['cmd'].append(cmd.astype(np.float32))
        rows['applied'].append(applied.astype(np.float32))
        rows['ready'].append(bool(ready))
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied[:4], applied[4:])
        step += 1
    return rows, best, catch_time


def demo_job(index, dart, along, side, noise_seed, path):
    from catch_env import DartNoise, ExpertDriver
    from gen_offset_parking import OffsetGoalDriver
    try:
        scene = separated_pair(_env, index, SCENE_STREAM)
        env = _env
        env.set_scene(scene['targets'], scene['selected'])
        driver = OffsetGoalDriver(along, side) if along is not None else ExpertDriver()
        driver.reset(env)
        noise = DartNoise(int(noise_seed)) if dart else None
        rows, best, catch_time = _rollout(env, driver, noise)
        obs = np.stack(rows['obs']).astype(np.float32)
        cols = np.r_[0, 10:42]
        if float(np.max(np.abs(obs[:, 0, cols] - obs[:, 1, cols]))) != 0.0:
            raise RuntimeError('robot columns differ between the two balls')
        chosen_miss = best[scene['selected']]
        meta = dict(
            index=int(index), scene_seed=int(scene['seed']), attempt=int(scene['attempt']),
            chosen_ball=int(scene['selected']), choice_changed=bool(env.chosen_ball != scene['selected']),
            separation_mm=float(scene['gap'] * 1000), separation_rejects=int(scene['rejected']),
            reachable_attempts=[int(v) for v in scene['reachable']],
            dart=bool(dart), offset=along is not None,
            offset_along_m=None if along is None else float(along),
            offset_side_m=None if side is None else float(side),
            noise_seed=None if noise_seed is None else int(noise_seed),
            success=catch_time is not None,
            catch_10mm=catch_time is not None,
            chosen_miss_mm=None if chosen_miss is None else float(chosen_miss * 1000),
            miss_mm=[None if v is None else float(v * 1000) for v in best],
            durations_s=[float(t.duration) for t in scene['targets']],
            speeds=[float(t.speed) for t in scene['targets']],
            intercept_times_s=[float(p.intercept_time) for p in scene['plans']],
            launch_gap_s=0.0, rate_hz=RATE_HZ, steps=int(obs.shape[0]), sim_dt=float(env.dt))
        tmp = Path(str(path) + '.tmp.npz')
        np.savez(tmp, obs=obs, cmd=np.stack(rows['cmd']).astype(np.float32),
                 applied=np.stack(rows['applied']).astype(np.float32),
                 ready=np.asarray(rows['ready']), meta=np.array(json.dumps(meta)))
        tmp.replace(path)
        return meta
    except Exception:
        print(f'failed demo {index}', flush=True)
        raise


def sample_job(index):
    scene = separated_pair(_env, index, TEST_STREAM)
    return dict(index=int(index), seed=int(scene['seed']), attempt=int(scene['attempt']),
                oracle_slot=int(oracle_choice(index)), expert_slot=int(scene['selected']),
                separation_mm=float(scene['gap'] * 1000), separation_rejects=int(scene['rejected']),
                reachable_attempts=[int(v) for v in scene['reachable']],
                durations_s=[float(t.duration) for t in scene['targets']],
                speeds=[float(t.speed) for t in scene['targets']], launch_gap_s=0.0)


def build_grid(folder, n, grid_path):
    from catch_env import FIRST_TICK_STEP, OBS_NAMES
    from config import GRID_HZ
    stride = 1000 // GRID_HZ
    obs, cmd, chosen, demo_index, starts, lengths = [], [], [], [], [], []
    offset = 0
    for i in range(n):
        path = folder / f'demo_{i:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if int(meta['index']) != i:
                raise RuntimeError(f'{path.name} index {meta["index"]}')
            if float(meta['separation_mm']) < SEPARATION * 1000 - 1e-3:
                raise RuntimeError(f'{path.name} intercepts are {meta["separation_mm"]:.1f} mm apart')
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        if len(o) == 0:
            raise RuntimeError(f'{path.name} truncated to nothing')
        obs.append(o)
        cmd.append(c)
        chosen.append(np.full(len(o), int(meta['chosen_ball']), dtype=np.int8))
        demo_index.append(np.full(len(o), i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
    np.savez(grid_path, obs=np.concatenate(obs), cmd=np.concatenate(cmd), chosen=np.concatenate(chosen),
             demo=np.concatenate(demo_index), starts=np.asarray(starts), lengths=np.asarray(lengths),
             arm_q_col=np.int64(OBS_NAMES.index('arm_q_0')))
    return offset


def spec_dict(n, n_test):
    return dict(master=MASTER, scene_stream=SCENE_STREAM, noise_stream=NOISE_STREAM,
                offset_stream=OFFSET_STREAM, test_stream=TEST_STREAM, oracle_stream=ORACLE_STREAM,
                n=int(n), n_test=int(n_test), n_clean=int(n // 2), separation_m=SEPARATION,
                side_sigma_m=SIDE_SIGMA, along_sigma_m=ALONG_SIGMA, cap_m=OFFSET_CAP,
                dart='odd indices of the first half',
                labels='expert command sent to the low level; DART is torque noise and is not the label')


def summarize(metas, tests, grid_path):
    chosen = np.array([m['chosen_ball'] for m in metas])
    misses = [m['chosen_miss_mm'] for m in metas if m['chosen_miss_mm'] is not None]
    offsets = [(m['offset_along_m'], m['offset_side_m']) for m in metas if m['offset']]
    along = np.array([a for a, _s in offsets]) if offsets else np.zeros(0)
    side = np.array([s for _a, s in offsets]) if offsets else np.zeros(0)
    lengths = np.array([m['steps'] * m['sim_dt'] for m in metas])
    durations = np.array([np.mean(m['durations_s']) for m in metas])
    summary = dict(
        n=len(metas), n_test=len(tests), separation_required_mm=SEPARATION * 1000,
        n_dart=int(sum(m['dart'] for m in metas)), n_offset=int(sum(m['offset'] for m in metas)),
        expert_catch_10mm=float(np.mean([m['catch_10mm'] for m in metas])),
        expert_catch_50mm=float(np.mean([m['chosen_miss_mm'] is not None and m['chosen_miss_mm'] < 50 for m in metas])),
        chosen_miss_median_mm=None if not misses else float(np.median(misses)),
        chose_slot1=float(np.mean(chosen == 1)),
        choice_changed=int(sum(m['choice_changed'] for m in metas)),
        separation_mm_min=float(min(m['separation_mm'] for m in metas)),
        separation_mm_median=float(np.median([m['separation_mm'] for m in metas])),
        separation_rejects_total=int(sum(m['separation_rejects'] for m in metas)),
        separation_rejects_mean=float(np.mean([m['separation_rejects'] for m in metas])),
        scenes_resampled=int(sum(m['separation_rejects'] > 0 for m in metas)),
        test_separation_rejects_total=int(sum(t['separation_rejects'] for t in tests)),
        test_scenes_resampled=int(sum(t['separation_rejects'] > 0 for t in tests)),
        test_oracle_slot1=float(np.mean([t['oracle_slot'] == 1 for t in tests])),
        reachable_attempts_mean=float(np.mean([a for m in metas for a in m['reachable_attempts']])),
        offset_along_sd_mm=None if len(along) == 0 else float(along.std() * 1000),
        offset_side_sd_mm=None if len(side) == 0 else float(side.std() * 1000),
        episode_s_median=float(np.median(lengths)), flight_s_median=float(np.median(durations)),
        launch_gap_s=0.0, grid=str(grid_path),
        labels='clean expert command; DART noise is on applied torque only')
    one_grid = DATA / 'grid50.npz'
    if one_grid.exists():
        with np.load(one_grid) as z:
            one = np.asarray(z['lengths'][:1000], float) / 50.0
        summary['oneball_episode_s_median'] = float(np.median(one))
        summary['oneball_episode_s_mean'] = float(np.mean(one))
        summary['twoball_episode_s_mean'] = float(np.mean(lengths))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', type=int, default=2000)
    parser.add_argument('--test', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--out', type=Path, default=DATA / 'two_ball')
    parser.add_argument('--grid', type=Path, default=DATA / 'grid50_two_ball.npz')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    if args.demos < 2 or args.demos % 2:
        raise SystemExit('--demos must be an even number, at least 2')
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    spec_path = folder / 'dataset.json'
    spec = spec_dict(args.demos, args.test)
    if spec_path.exists():
        saved = json.loads(spec_path.read_text())
        keys = [k for k in spec if k != 'labels']
        if any(saved.get(k) != spec[k] for k in keys):
            raise SystemExit(f'{spec_path} does not match this request; use a fresh --out')
    spec_path.write_text(json.dumps(spec, indent=2))
    n_clean = args.demos // 2
    pending, test_pending = [], []
    for i in range(args.demos):
        path = folder / f'demo_{i:04d}.npz'
        if path.exists():
            continue
        dart = i < n_clean and (i % 2 == 1)
        if i >= n_clean:
            along, side = sample_offset(i - n_clean)
        else:
            along = side = None
        noise_seed = scene_seed(NOISE_STREAM, i, 0) if dart else None
        pending.append((i, dart, along, side, noise_seed, path))
    test_path = folder / 'test_scenes.json'
    have_test = False
    if test_path.exists():
        saved_tests = json.loads(test_path.read_text())
        have_test = len(saved_tests) == args.test and saved_tests[0].get('seed') == scene_seed(TEST_STREAM, 0, saved_tests[0]['attempt'])
    print(f'{args.demos} two-ball demos, {len(pending)} to generate, separation {SEPARATION * 1000:.0f} mm', flush=True)
    t0 = time.time()
    done = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(demo_job, *job) for job in pending]
        test_futs = [] if have_test else [pool.submit(sample_job, i) for i in range(args.test)]
        for k, fut in enumerate(as_completed(futs), 1):
            done.append(fut.result())
            if k == 50 and np.mean([m['catch_10mm'] for m in done]) < 0.5:
                raise RuntimeError('expert caught under half of the first 50 episodes; stopping')
            if k % 25 == 0 or k == len(futs):
                caught = float(np.mean([m['catch_10mm'] for m in done]))
                print(f'  demos {k}/{len(futs)}  catch {caught:.3f}  ({time.time() - t0:.0f}s)', flush=True)
        tests = saved_tests if have_test else [fut.result() for fut in test_futs]
    if not have_test:
        tests = sorted(tests, key=lambda row: row['index'])
        test_path.write_text(json.dumps(tests))
    metas = []
    for i in range(args.demos):
        with np.load(folder / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            metas.append(json.loads(str(z['meta'])))
    train_seeds = {m['scene_seed'] for m in metas}
    test_seeds = {t['seed'] for t in tests}
    if train_seeds & test_seeds:
        raise SystemExit('a training scene seed is also a test scene seed')
    one = DATA / 'seeds.json'
    if one.exists():
        held = json.loads(one.read_text())
        overlap = train_seeds & (set(held['train']) | set(held['test']))
        if overlap:
            raise SystemExit(f'{len(overlap)} scene seeds collide with the one-ball seed file')
    print('building the 50 Hz grid', flush=True)
    n_rows = build_grid(folder, args.demos, args.grid)
    summary = summarize(metas, tests, args.grid)
    summary['grid_rows'] = int(n_rows)
    summary['wall_s'] = time.time() - t0
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_log:
        from stage2_log import finish_run, start_run
        _run_id, record_path, record = start_run(
            'twoball-data', 'random-choice-expert', spec, ['gen_two_ball.py'],
            dict(train_stream=SCENE_STREAM, test_stream=TEST_STREAM, n=args.demos, n_test=args.test),
            note='200 mm separation, half-and-half one-ball recipe')
        out = ROOT / 'results' / 'stage2' / 'two_ball_data.json'
        out.write_text(json.dumps(dict(summary=summary, test_scenes=str(test_path)), indent=2))
        finish_run(record_path, record, dict(miss_mm=summary['chosen_miss_median_mm'],
                                              catch_rate=summary['expert_catch_10mm']),
                   [args.grid, test_path, out])
    if args.demos >= 100 and summary['expert_catch_10mm'] < 0.98:
        raise SystemExit(f"expert catch at 10 mm is {summary['expert_catch_10mm']:.3f}; not training on this")
    print(f'wrote {args.grid} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
