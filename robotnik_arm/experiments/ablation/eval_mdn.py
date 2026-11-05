"""Roll out the two-ball mixture on the one-ball test set and the two-ball test scenes.

Selection is the component mean, not a sample. ``weight`` picks the highest mixture
weight. ``continue`` picks the component whose next second best matches the chunk
already being executed. Both replan every 1 s. The one-ball episodes stop at 10 mm
the same way the ceiling's test did. Two-ball episodes run to the end of the flight.
"""
import argparse
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SCENES = ROOT / 'data' / 'two_ball' / 'test_scenes.json'
SEEDS_PATH = ROOT / 'data' / 'seeds.json'
RADIUS_MM = 50.0
PUBLISHED_CEILING = 0.865

_env = None
_policies = None
_specs = None
_payload = None
_kind = None


def check_mdn():
    """Likelihood identity, ball-swap invariance, and numpy agreement with torch."""
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    import torch
    from config import Config
    from np_policy import NumpyPolicy, choose_component
    from policy import DeepSetsPolicy, export_numpy, gaussian_mixture_nll
    from train import mixture_inputs

    horizon, act = 50, 18
    target = torch.zeros(2, horizon, act)
    mu = torch.zeros(2, 1, horizon, act)
    sigma = torch.ones(2, 1, horizon, act)
    log_pi = torch.zeros(2, 1)
    nll = float(gaussian_mixture_nll(log_pi, mu, sigma, target))
    expected = 0.5 * horizon * act * math.log(2.0 * math.pi)
    if abs(nll - expected) > 1e-3:
        raise SystemExit(f'mixture likelihood at the mode is {nll}, expected {expected}')
    twin = torch.log_softmax(torch.zeros(2, 2), dim=-1)
    mu2 = torch.zeros(2, 2, horizon, act)
    sigma2 = torch.ones(2, 2, horizon, act)
    nll2 = float(gaussian_mixture_nll(twin, mu2, sigma2, target))
    if abs(nll2 - expected) > 1e-3:
        raise SystemExit('two identical components changed the likelihood')

    cfg = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                 demos=2000, encoder='deepsets', pool='sum', head='mdn', n_components=5, sigma_floor=0.01)
    torch.manual_seed(0)
    model = DeepSetsPolicy(cfg)
    with torch.no_grad():
        model.ball_mean.normal_()
        model.ball_std.copy_(torch.rand(model.ball_std.shape) + 0.5)
        model.robot_mean.normal_()
        model.robot_std.copy_(torch.rand(model.robot_std.shape) + 0.5)
        model.act_mean.normal_()
        model.act_std.copy_(torch.rand(model.act_std.shape) + 0.5)
    frames = torch.randn(2, cfg.frames, 2, 42)
    frames[:, :, 1, 0] = frames[:, :, 0, 0]
    frames[:, :, 1, 10:] = frames[:, :, 0, 10:]
    chosen = torch.zeros(2, dtype=torch.long)
    robot, balls = mixture_inputs(model, frames, chosen, True)
    with torch.no_grad():
        log_pi, mu, sigma = model.mixture(robot, balls)
        swapped = frames[:, :, [1, 0]]
        robot_s, balls_s = mixture_inputs(model, swapped, chosen, True)
        log_pi_s, mu_s, sigma_s = model.mixture(robot_s, balls_s)
    if not torch.allclose(mu, mu_s, atol=1e-5) or not torch.allclose(log_pi, log_pi_s, atol=1e-5):
        raise SystemExit('swapping the two balls changed the mixture')
    if float(sigma.min()) < cfg.sigma_floor - 1e-6:
        raise SystemExit('a component variance fell through the floor')

    pi = np.array([0.7, 0.3], np.float32)
    means = np.zeros((2, 4, 2), np.float32)
    previous = np.zeros((4, 2), np.float32)
    previous[2:] = 1
    means[0, :2] = 5
    means[1, :2] = 1
    if choose_component(pi, means, 'weight', previous, 2) != 0:
        raise SystemExit('highest-weight rule ignored the weights')
    if choose_component(pi, means, 'continue', None, 2) != 0:
        raise SystemExit('the first query did not take the highest weight')
    if choose_component(pi, means, 'continue', previous, 2) != 1:
        raise SystemExit('continuation rule did not follow the chunk in progress')

    with tempfile.TemporaryDirectory() as tmp:
        pt = Path(tmp) / 'mdn.pt'
        npz = Path(tmp) / 'mdn.npz'
        torch.save(dict(config=cfg.to_dict(), state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        meta={}), pt)
        export_numpy(pt, npz)
        policy = NumpyPolicy(npz)
        stacked = frames[0].permute(1, 0, 2).numpy().astype(np.float32)
        pi_np, mu_np = policy.mdn_components(stacked, None)
        pi_t = log_pi[0].exp().numpy()
        mu_t = mu[0].numpy()
        gap = max(float(np.max(np.abs(pi_np - pi_t))), float(np.max(np.abs(mu_np - mu_t))))
        print(json.dumps(dict(nll_gap=abs(nll - expected), numpy_gap=gap)), flush=True)
        if gap > 1e-4:
            raise SystemExit('numpy mixture disagrees with the torch policy')
    mse = RESULTS / 'models' / 'twoball_mse_s0.npz'
    if mse.exists():
        NumpyPolicy(mse)
    print('mdn check passed', flush=True)


def init_worker(specs, payload, kind):
    global _env, _policies, _specs, _payload, _kind
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _policies = {}
    for _name, path, _rule in specs:
        if path not in _policies:
            _policies[path] = NumpyPolicy(path)
    _specs, _payload, _kind = specs, payload, kind


def _one_ball(driver, seed):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    env = _env
    env.reset(int(seed))
    driver.reset(env)
    rate = driver.cfg.rate_hz
    stride = int(round(1.0 / (rate * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    best = None
    step = 0
    while True:
        t = step * env.dt
        dist = env.capture_distance(t)
        if dist is not None:
            if best is None or dist < best:
                best = dist
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        env.step(cmd[:4], env.arm_torque(q_ref, cmd[11:18]))
        step += 1
    nearest = None if best is None else dict(dist=float(best))
    return [best], nearest, [], []


def _two_ball(driver, intercepts):
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
            base = np.asarray(env.base_pos[:2], float)
            aims.append(int(np.argmin([np.linalg.norm(base - np.asarray(p[:2], float)) for p in intercepts])))
            aim_times.append(float(t))
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        env.step(cmd[:4], env.arm_torque(q_ref, cmd[11:18]))
        step += 1
    return best, nearest, aims, aim_times


def _annotate(row, driver):
    choices = list(driver.choices)
    counts = np.bincount(choices, minlength=driver.cfg.n_components) if choices else np.zeros(1)
    row['replans'] = len(choices)
    row['max_pi_mean'] = None if not driver.max_pi else float(np.mean(driver.max_pi))
    row['component_changes'] = int(sum(a != b for a, b in zip(choices, choices[1:])))
    row['components_used'] = int(np.count_nonzero(counts))
    row['dominant_frac'] = None if not choices else float(counts.max() / len(choices))
    row['choices'] = choices
    return row


def job(index):
    from np_policy import MDNDriver
    from eval_two_ball import _row
    env = _env
    if _kind == 'oneball':
        seed = int(_payload[index])
        oracle_slot = 0
        out = dict(index=int(index), seed=seed)
        for name, path, rule in _specs:
            driver = MDNDriver(_policies[path], rule)
            best, nearest, aims, aim_times = _one_ball(driver, seed)
            out[name] = _annotate(_row(best, nearest, aims, aim_times, oracle_slot, False), driver)
        return out
    from robotnik_arm_control import make_catchable_pair
    scene = _payload[index]
    targets, _selected, plans = make_catchable_pair(
        int(scene['seed']), None, env.lower, env.upper, env.init_base_pos, env.init_bearing, env.init_ee,
        env.dt, env.base_params, env.arm_params, target_count=2)
    gap = float(np.linalg.norm(np.asarray(plans[0].intercept) - np.asarray(plans[1].intercept)) * 1000)
    if abs(gap - float(scene['separation_mm'])) > 1.0 or gap < 200 - 1e-3:
        raise RuntimeError(f'scene {index} reproduced separation {gap:.1f} mm')
    oracle_slot = int(scene['oracle_slot'])
    intercepts = [np.asarray(plan.intercept, float) for plan in plans]
    out = dict(index=int(index), oracle_slot=oracle_slot)
    for name, path, rule in _specs:
        env.set_scene(targets, 0)
        driver = MDNDriver(_policies[path], rule)
        best, nearest, aims, aim_times = _two_ball(driver, intercepts)
        out[name] = _annotate(_row(best, nearest, aims, aim_times, oracle_slot, True), driver)
    return out


def _run(kind, payload, specs, workers):
    rows = []
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(specs, payload, kind)) as pool:
        futs = [pool.submit(job, i) for i in range(len(payload))]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 5 == 0 or n == len(futs):
                print(f'  {kind} {n}/{len(futs)}', flush=True)
    rows.sort(key=lambda r: r['index'])
    return rows


def _component_block(rows, names):
    block = {}
    for name in names:
        chosen = [c for r in rows for c in r[name]['choices']]
        hist = np.bincount(chosen, minlength=5).astype(float)
        hist = hist / max(hist.sum(), 1.0)
        block[name] = dict(
            max_pi_mean=float(np.mean([r[name]['max_pi_mean'] for r in rows])),
            dominant_frac_mean=float(np.mean([r[name]['dominant_frac'] for r in rows])),
            components_used_mean=float(np.mean([r[name]['components_used'] for r in rows])),
            component_changes_mean=float(np.mean([r[name]['component_changes'] for r in rows])),
            choice_frac=[float(v) for v in hist])
    return block


def _pack(rows, two_ball, seeds):
    from eval_two_ball import _mean, _stats
    block = {}
    seed_stats = []
    names = []
    for seed in seeds:
        for rule in ('weight', 'continue'):
            name = f'mdn_{rule}_s{seed}'
            if name not in rows[0]:
                continue
            names.append(name)
            stats = _stats([r[name] for r in rows], two_ball)
            block[name] = stats
            if rule == 'weight':
                seed_stats.append(stats)
            miss = stats['miss_nearest_mm']
            miss_s = 'n/a' if miss is None else f'{miss:.1f}'
            print(f"{name}: either {stats['catch_either']:.3f}  indicated {stats['catch_indicated']:.3f}  "
                  f"miss {miss_s}", flush=True)
    mean = {}
    for key in ('catch_either', 'catch_indicated', 'miss_nearest_mm', 'miss_indicated_mm',
                'between_miss_rate', 'switches_mean', 'commitment_s_mean', 'frac_switched',
                'catch_slot0', 'catch_slot1'):
        value, spread = _mean(seed_stats, key)
        mean[key] = value
        mean[key + '_sd'] = spread
    cont = []
    for seed in seeds:
        name = f'mdn_continue_s{seed}'
        if name in rows[0]:
            cont.append(_stats([r[name] for r in rows], two_ball))
    continue_mean = {}
    for key in ('catch_either', 'catch_indicated', 'miss_nearest_mm', 'between_miss_rate',
                'switches_mean', 'commitment_s_mean'):
        value, spread = _mean(cont, key)
        continue_mean[key] = value
        continue_mean[key + '_sd'] = spread
    agree = []
    for seed in seeds:
        a, b = f'mdn_weight_s{seed}', f'mdn_continue_s{seed}'
        if a in rows[0] and b in rows[0]:
            agree.append(float(np.mean([r[a]['catch_either'] == r[b]['catch_either'] for r in rows])))
    block['mean_weight'] = mean
    block['mean_continue'] = continue_mean
    block['rule_agreement'] = None if not agree else float(np.mean(agree))
    block['components'] = _component_block(rows, names)
    return block


def _reference():
    curve = json.loads((RESULTS / 'stage2' / 'radius_curve.json').read_text())
    misses = [curve[f's{seed}']['miss_mm'] for seed in (0, 1, 2)]
    compare = json.loads((RESULTS / 'stage2' / 'two_ball_compare.json').read_text())
    return dict(
        oneball_test_catch_50mm=float(curve['mean_curve']['50']),
        oneball_test_miss_mm=float(np.mean(misses)),
        published_ceiling_50mm=PUBLISHED_CEILING,
        twoball_ceiling_indicated=compare['ceiling']['mean']['catch_indicated'],
        twoball_oracle_indicated=compare['oracle']['mean']['catch_indicated'],
        twoball_mse_either=compare['mse']['mean']['catch_either'],
        twoball_mse_between=compare['mse']['mean']['between_miss_rate'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--out', type=Path, default=RESULTS / 'stage2' / 'mdn_compare.json')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    if args.check:
        check_mdn()
        return
    seeds = tuple(int(s) for s in args.seeds.split(',') if s)
    specs = []
    for seed in seeds:
        path = RESULTS / 'models' / f'twoball_mdn_s{seed}.npz'
        if not path.exists():
            raise SystemExit(f'missing {path}')
        for rule in ('weight', 'continue'):
            specs.append((f'mdn_{rule}_s{seed}', str(path), rule))
    one_seeds = json.loads(SEEDS_PATH.read_text())['test'][:200]
    scenes = json.loads(SCENES.read_text())
    if args.limit:
        one_seeds = one_seeds[:args.limit]
        scenes = scenes[:args.limit]
    ref = None if args.limit else _reference()
    print(f'one-ball test {len(one_seeds)} scenes, {len(specs)} rollouts', flush=True)
    one_rows = _run('oneball', one_seeds, specs, args.workers)
    print(f'two-ball test {len(scenes)} scenes', flush=True)
    two_rows = _run('twoball', scenes, specs, args.workers)
    report = dict(
        definitions=dict(
            catch='capture-window distance under 50 mm',
            oneball='the 200 one-ball test scenes the ceiling was measured on; a rollout stops at 10 mm, as that measurement did',
            twoball='the 200 two-ball test scenes; a catch is either ball; episodes run to the end of the flight',
            weight='execute the mean of the highest-weight component',
            continue_rule='at each replan, execute the component whose next second is closest, in normalized actions, to the unexecuted part of the chunk in progress',
            between_miss='among two-ball misses, the hand at its closest approach was nearer the midpoint than either ball'),
        train='two-ball episodes only, K=5 diagonal Gaussians, negative log-likelihood',
        reference=ref,
        oneball=_pack(one_rows, False, seeds),
        twoball=_pack(two_rows, True, seeds))
    if ref is not None:
        got = report['oneball']['mean_weight']['catch_either']
        report['oneball_gap_vs_ceiling'] = None if got is None else float(got - ref['oneball_test_catch_50mm'])
        two = report['twoball']['mean_weight']['catch_either']
        report['twoball_gap_vs_oracle'] = None if two is None else float(two - ref['twoball_oracle_indicated'])
    report['episodes'] = dict(oneball=one_rows, twoball=two_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))
    slim = {k: report[k] for k in report if k != 'episodes'}
    print(json.dumps(slim, indent=2), flush=True)
    print(f'wrote {args.out}', flush=True)
    if not args.no_log and not args.limit:
        from stage2_log import finish_run, start_run
        _run_id, record_path, record = start_run(
            'mdn-compare', 'mdn', dict(seeds=list(seeds)), ['eval_mdn.py'],
            dict(oneball=len(one_seeds), twoball=len(scenes)), note='weight and continue rules')
        finish_run(record_path, record, dict(
            miss_mm=report['oneball']['mean_weight'].get('miss_nearest_mm'),
            catch_rate=report['oneball']['mean_weight'].get('catch_either')), [args.out])


if __name__ == '__main__':
    main()
