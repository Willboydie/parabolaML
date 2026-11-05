"""Catch rate, miss, between-miss, switches and commitment on the two-ball test scenes.

The one-ball ceiling is rolled out on the same scenes with only the ball the
oracle is told to catch. A catch is a capture-window distance under 50 mm.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SCENES = ROOT / 'data' / 'two_ball' / 'test_scenes.json'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
RADIUS_MM = 50.0
RADII = list(range(10, 201, 5))
PUBLISHED_CEILING = 0.865

_env = None
_models = None
_scenes = None


def check_agreement():
    """The summed numpy path matches a single ball, and matches torch on two balls."""
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    import torch
    from ballset import ball_features, robot_features
    from config import Config
    from np_policy import NumpyPolicy
    from policy import DeepSetsPolicy, load_policy
    from train import apply_oracle_flag, deepsets_forward

    balls = torch.zeros(4, 2, 22)
    chosen = torch.tensor([0, 1, 1, 0])
    flagged = apply_oracle_flag(balls, chosen, True)
    assert float(flagged[0, 0, 21]) == 1 and float(flagged[0, 1, 21]) == 0
    assert float(flagged[1, 1, 21]) == 1
    timed = apply_oracle_flag(torch.zeros(4, 3, 2, 22), chosen, True)
    assert torch.all(timed[1, :, 1, 21] == 1) and torch.all(timed[1, :, 0, 21] == 0)
    cfg = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                 demos=2000, encoder='deepsets', pool='sum', oracle=True)
    model = DeepSetsPolicy(cfg)
    frames = torch.randn(2, cfg.frames, 2, 42)
    frames[:, :, 1, 0] = frames[:, :, 0, 0]
    frames[:, :, 1, 10:] = frames[:, :, 0, 10:]
    same = deepsets_forward(model, frames, chosen[:2])
    swapped = deepsets_forward(model, frames[:, :, [1, 0]], 1 - chosen[:2])
    if not torch.allclose(same, swapped, atol=1e-5):
        raise SystemExit('swapping balls and the oracle flag changed the output')

    pt = RESULTS / 'models' / f'{CEILING}_s0.pt'
    policy = NumpyPolicy(pt.with_suffix('.npz'))
    torch_model = load_policy(pt)
    rng = np.random.default_rng(0)
    one = rng.normal(size=(policy.cfg.frames, 42)).astype(np.float32)
    one[:, 14] = 1
    gap_one = float(np.max(np.abs(policy.predict(one) - policy.predict_set(one[None], None))))
    both = rng.normal(size=(2, policy.cfg.frames, 42)).astype(np.float32)
    both[:, :, 14] = 1
    numpy_out = policy.predict_set(both, None)
    ball_t = ball_features(torch.from_numpy(both)).unsqueeze(0)
    robot_t = robot_features(torch.from_numpy(both[:1]))
    with torch.no_grad():
        torch_out = (torch_model.forward_set(robot_t, ball_t) * torch_model.act_std + torch_model.act_mean).numpy()[0]
    gap_two = float(np.max(np.abs(numpy_out - torch_out)))
    print(json.dumps(dict(one_ball_gap=gap_one, two_ball_gap=gap_two)), flush=True)
    if gap_one > 1e-4 or gap_two > 1e-4:
        raise SystemExit('numpy set path disagrees with the torch policy')


def init_worker(specs, scenes_path):
    global _env, _models, _scenes
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _models = [(name, NumpyPolicy(path), mode) for name, path, mode in specs]
    _scenes = json.loads(Path(scenes_path).read_text())


def _switches(aims, times):
    if len(aims) < 2:
        return 0, 0.0
    changes = [i for i in range(1, len(aims)) if aims[i] != aims[i - 1]]
    commitment = 0.0 if not changes else float(times[changes[-1]])
    return len(changes), commitment


def _rollout(driver, intercepts):
    from catch_env import CAP_AFTER_ENDPOINT, FIRST_TICK_STEP
    env = _env
    driver.reset(env)
    rate = driver.cfg.rate_hz
    stride = int(round(1.0 / (rate * env.dt)))
    end_time = max(target.duration for target in env.targets) + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    best = [None] * len(env.targets)
    nearest = None
    aims, aim_times = [], []
    step = 0
    while True:
        t = step * env.dt
        positions = None
        for slot in range(len(env.targets)):
            dist = env.slot_distance(slot, t)
            if dist is None:
                continue
            if best[slot] is None or dist < best[slot]:
                best[slot] = dist
            if nearest is None or dist < nearest['dist']:
                if positions is None:
                    positions = [target.position(t).copy() for target in env.targets]
                nearest = dict(dist=float(dist), ee=env.ee.copy(), balls=positions)
        if t >= end_time:
            break
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            if intercepts is not None:
                base = np.asarray(env.base_pos[:2], float)
                aims.append(int(np.argmin([np.linalg.norm(base - np.asarray(p[:2], float)) for p in intercepts])))
                aim_times.append(float(t))
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        env.step(cmd[:4], env.arm_torque(q_ref, cmd[11:18]))
        step += 1
    return best, nearest, aims, aim_times


def _between(nearest):
    if nearest is None or len(nearest['balls']) != 2:
        return None
    ee = np.asarray(nearest['ee'], float)
    balls = [np.asarray(p, float) for p in nearest['balls']]
    mid = (balls[0] + balls[1]) / 2
    distances = [float(np.linalg.norm(ee - p)) for p in balls]
    return bool(float(np.linalg.norm(ee - mid)) < min(distances))


def _row(best, nearest, aims, aim_times, oracle_slot, two_ball):
    misses = [None if v is None else float(v * 1000) for v in best]
    finite = [v for v in misses if v is not None]
    nearest_mm = None if not finite else float(min(finite))
    if two_ball:
        indicated = misses[oracle_slot]
        caught = [v is not None and v < RADIUS_MM for v in misses]
    else:
        indicated = misses[0]
        caught = [indicated is not None and indicated < RADIUS_MM]
    switches, commitment = (None, None) if not two_ball else _switches(aims, aim_times)
    return dict(
        miss_mm=misses, nearest_mm=nearest_mm, indicated_mm=indicated,
        catch_either=bool(any(caught)), catch_indicated=bool(indicated is not None and indicated < RADIUS_MM),
        n_caught=int(sum(caught)), between=_between(nearest) if two_ball else None,
        switches=switches, commitment_s=commitment, no_window=nearest is None)


def job(index):
    from np_policy import PolicyDriver, SetPolicyDriver
    from robotnik_arm_control import make_catchable_pair
    scene = _scenes[index]
    env = _env
    targets, _selected, plans = make_catchable_pair(
        int(scene['seed']), None, env.lower, env.upper, env.init_base_pos, env.init_bearing, env.init_ee,
        env.dt, env.base_params, env.arm_params, target_count=2)
    gap = float(np.linalg.norm(np.asarray(plans[0].intercept) - np.asarray(plans[1].intercept)) * 1000)
    if abs(gap - float(scene['separation_mm'])) > 1.0:
        raise RuntimeError(f'scene {index} reproduced a different pair ({gap:.1f} mm)')
    if gap < 200 - 1e-3:
        raise RuntimeError(f'scene {index} is only {gap:.1f} mm apart')
    oracle_slot = int(scene['oracle_slot'])
    intercepts = [np.asarray(plan.intercept, float) for plan in plans]
    out = dict(index=int(index), oracle_slot=oracle_slot)
    for name, model, mode in _models:
        if mode == 'ceiling':
            env.set_scene([targets[oracle_slot]], 0)
            driver = PolicyDriver(model)
            best, nearest, aims, aim_times = _rollout(driver, None)
            out[name] = _row(best, nearest, aims, aim_times, oracle_slot, False)
        else:
            env.set_scene(targets, 0)
            driver = SetPolicyDriver(model, oracle_slot)
            best, nearest, aims, aim_times = _rollout(driver, intercepts)
            out[name] = _row(best, nearest, aims, aim_times, oracle_slot, True)
    return out


def _curve(values):
    dist = np.array([np.inf if v is None else v for v in values], float)
    return {str(radius): float(np.mean(dist < radius)) for radius in RADII}


def _stats(rows, two_ball):
    nearest = [r['nearest_mm'] for r in rows if r['nearest_mm'] is not None]
    indicated = [r['indicated_mm'] for r in rows if r['indicated_mm'] is not None]
    misses = [r for r in rows if not r['catch_either']]
    between = [r for r in misses if r['between']]
    stats = dict(
        n=len(rows),
        catch_either=float(np.mean([r['catch_either'] for r in rows])),
        catch_indicated=float(np.mean([r['catch_indicated'] for r in rows])),
        miss_nearest_mm=None if not nearest else float(np.mean(nearest)),
        miss_nearest_median_mm=None if not nearest else float(np.median(nearest)),
        miss_indicated_mm=None if not indicated else float(np.mean(indicated)),
        n_miss=len(misses), n_between=len(between), n_no_window=int(sum(r['no_window'] for r in rows)),
        between_miss_rate=None if (not two_ball or not misses) else float(len(between) / len(misses)),
        between_episode_rate=float(np.mean([bool(r['between']) for r in rows])) if two_ball else None,
        radius_either=_curve([r['nearest_mm'] for r in rows]),
        radius_indicated=_curve([r['indicated_mm'] for r in rows]))
    if two_ball and rows and rows[0]['miss_mm'] is not None:
        stats['catch_slot0'] = float(np.mean([r['miss_mm'][0] is not None and r['miss_mm'][0] < RADIUS_MM for r in rows]))
        stats['catch_slot1'] = float(np.mean([r['miss_mm'][1] is not None and r['miss_mm'][1] < RADIUS_MM for r in rows]))
        switches = np.array([r['switches'] for r in rows], float)
        commit = np.array([r['commitment_s'] for r in rows], float)
        stats.update(switches_mean=float(switches.mean()), switches_median=float(np.median(switches)),
                     frac_switched=float(np.mean(switches > 0)),
                     commitment_s_mean=float(commit.mean()), commitment_s_median=float(np.median(commit)))
    return stats


def _mean(blocks, key):
    vals = [b[key] for b in blocks if b.get(key) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _reading(ceiling, oracle, mse):
    """Apply the plan's decision table, with a 5 point noise floor and a 10 point gap."""
    c, o, m = ceiling['catch_indicated'], oracle['catch_indicated'], mse['catch_either']
    between = mse['between_miss_rate']
    if c - o > 0.10:
        label = 'oracle_below_ceiling'
        text = 'The oracle is well below the ceiling on the same throws, so a second ball hurts before the choice is the issue.'
    elif c - m <= 0.05:
        label = 'mse_near_ceiling'
        text = 'The squared-error copier is close to the ceiling, so it is settling the choice early and averaging barely shows up.'
    elif c - m > 0.10 and between is not None and between >= 0.20:
        label = 'averaging'
        text = 'The oracle is near the ceiling and the squared-error copier is well below it, with many misses between the balls. The averaging failure is real.'
    else:
        label = 'mixed'
        text = 'The gap is real but it does not sit cleanly in one cell of the decision table.'
    return dict(label=label, text=text, ceiling=c, oracle_indicated=o, mse_either=m,
                mse_gap=c - m, oracle_gap=c - o, mse_between_miss_rate=between)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--scenes', type=Path, default=SCENES)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path, default=RESULTS / 'stage2' / 'two_ball_compare.json')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    if args.check:
        check_agreement()
        return
    scenes = json.loads(args.scenes.read_text())
    if args.limit:
        scenes = scenes[:args.limit]
        limit_path = args.scenes.with_name(args.scenes.stem + f'_limit{args.limit}.json')
        limit_path.write_text(json.dumps(scenes))
        scenes_path = limit_path
    else:
        scenes_path = args.scenes
    specs = []
    for seed in (0, 1, 2):
        mse = RESULTS / 'models' / f'twoball_mse_s{seed}.npz'
        oracle = RESULTS / 'models' / f'twoball_oracle_s{seed}.npz'
        ceiling = RESULTS / 'models' / f'{CEILING}_s{seed}.npz'
        for path in (mse, oracle, ceiling):
            if not path.exists():
                raise SystemExit(f'missing {path}')
        specs.append((f'mse_s{seed}', str(mse), 'mse'))
        specs.append((f'oracle_s{seed}', str(oracle), 'oracle'))
        specs.append((f'ceiling_s{seed}', str(ceiling), 'ceiling'))
    if args.limit:
        specs = [item for item in specs if item[0].endswith('_s0')]
    print(f'{len(scenes)} scenes, {len(specs)} models', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(specs, str(scenes_path))) as pool:
        futs = [pool.submit(job, i) for i in range(len(scenes))]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 5 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}', flush=True)
    rows.sort(key=lambda r: r['index'])
    report = dict(
        definitions=dict(
            catch='capture-window distance under 50 mm; either ball counts for the two-ball robots',
            indicated='the ball the oracle flag names; the ceiling sees only that ball',
            between_miss='among misses, the hand at its closest approach was nearer the midpoint of the two balls than either ball',
            switches='times the base crossed from being closer to one planned intercept to the other, sampled at the policy rate',
            commitment_s='time of the last such crossing; 0 if the base never changed its nearer intercept'),
        published_oneball_ceiling_50mm=PUBLISHED_CEILING,
        n_scenes=len(scenes), radius_mm=RADIUS_MM,
        oracle_slot1=float(np.mean([r['oracle_slot'] == 1 for r in rows])))
    flat_keys = ('catch_either', 'catch_indicated', 'miss_nearest_mm', 'miss_indicated_mm',
                 'between_miss_rate', 'switches_mean', 'commitment_s_mean', 'frac_switched',
                 'catch_slot0', 'catch_slot1')
    means = {}
    for role, two_ball in (('mse', True), ('oracle', True), ('ceiling', False)):
        block = {}
        seed_stats = []
        for seed in (0, 1, 2):
            name = f'{role}_s{seed}'
            if name not in rows[0]:
                continue
            stats = _stats([r[name] for r in rows], two_ball)
            block[f's{seed}'] = stats
            seed_stats.append(stats)
            miss = stats['miss_nearest_mm']
            miss_s = 'n/a' if miss is None else f'{miss:.1f}'
            print(f"{name}: either {stats['catch_either']:.3f}  indicated {stats['catch_indicated']:.3f}  "
                  f"miss {miss_s}", flush=True)
        mean = {}
        for key in flat_keys:
            value, spread = _mean(seed_stats, key)
            mean[key] = value
            mean[key + '_sd'] = spread
        if seed_stats:
            p = mean['catch_either']
            mean['binomial_se_n200'] = None if p is None else float(np.sqrt(p * (1 - p) / len(scenes)))
        block['mean'] = mean
        report[role] = block
        means[role] = mean
    if all(role in means and means[role].get('catch_indicated') is not None for role in ('ceiling', 'oracle', 'mse')):
        report['reading'] = _reading(means['ceiling'], means['oracle'], means['mse'])
        report['gaps_vs_published'] = dict(
            ceiling=means['ceiling']['catch_indicated'] - PUBLISHED_CEILING,
            oracle=means['oracle']['catch_indicated'] - PUBLISHED_CEILING,
            mse=means['mse']['catch_either'] - PUBLISHED_CEILING)
    report['episodes'] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))
    slim = {k: report[k] for k in report if k != 'episodes'}
    print(json.dumps(slim, indent=2), flush=True)
    print(f'wrote {args.out}', flush=True)
    if not args.no_log and not args.limit:
        from stage2_log import finish_run, start_run
        reading = report.get('reading', {})
        _run_id, record_path, record = start_run(
            'twoball-compare', 'mse-oracle-ceiling', dict(radius_mm=RADIUS_MM, n_scenes=len(scenes)),
            ['eval_two_ball.py'], dict(scenes=str(args.scenes)), note=reading.get('label', ''))
        finish_run(record_path, record, dict(miss_mm=means.get('mse', {}).get('miss_nearest_mm'),
                                              catch_rate=means.get('mse', {}).get('catch_either')), [args.out])


if __name__ == '__main__':
    main()
