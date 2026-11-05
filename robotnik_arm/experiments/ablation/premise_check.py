"""Was the one-ball miss a fair test of the mixture?

The mixture trained on two-ball rows. With DeepSets the head sees the sum of the
per-ball embeddings, so a one-ball input is a different vector. This script checks
that the training grid really is two balls on every row, measures how far the
executed chunk moves when the second ball is dropped, and rolls the two-ball
squared-error robot out on the one-ball test scenes.

The rollout uses the same one-ball loop and 50 mm catch as ``eval_mdn``. Pass
``--ceiling`` to run the DeepSets ceiling in that same loop, which checks the harness.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data'
SEEDS_PATH = DATA / 'seeds.json'
TRAIN_RECORD = RESULTS / 'stage2' / 'two_ball_mdn_train.json'
MDN_RECORD = RESULTS / 'stage2' / 'mdn_compare.json'
COMPARE_RECORD = RESULTS / 'stage2' / 'two_ball_compare.json'
CURVE_RECORD = RESULTS / 'stage2' / 'radius_curve.json'
GRID = DATA / 'grid50_two_ball.npz'
MDN = 'twoball_mdn_s{seed}'
MSE = 'twoball_mse_s{seed}'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets_s{seed}'
N_ROWS = 100
ROW_SEED = 0

_policies = None
_specs = None
_seeds = None


def _model(pattern, seed):
    path = RESULTS / 'models' / f'{pattern.format(seed=seed)}.npz'
    if not path.exists():
        raise SystemExit(f'missing {path}')
    return path


def audit_grid():
    """Every training row is one timestep of a two-ball episode."""
    record = json.loads(TRAIN_RECORD.read_text())
    with np.load(GRID) as z:
        obs = z['obs']
        chosen = np.asarray(z['chosen'])
        starts, lengths = np.asarray(z['starts']), np.asarray(z['lengths'])
    if obs.ndim != 3 or obs.shape[-1] != 42:
        raise SystemExit(f'expected two-ball observations (rows, balls, 42), got {obs.shape}')
    pos = obs[:, :, 1:4]
    sep = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1)
    robot = np.concatenate((obs[:, :, 0:1], obs[:, :, 10:]), axis=-1)
    robot_gap = float(np.max(np.abs(robot[:, 0] - robot[:, 1])))
    n_demos = int(record['config']['demos'])
    n_rows = int(starts[n_demos - 1] + lengths[n_demos - 1])
    summary_path = DATA / 'two_ball' / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return dict(
        train_record=str(TRAIN_RECORD),
        train_data=record['data'],
        grid=str(GRID),
        obs_shape=[int(v) for v in obs.shape],
        balls_per_row=int(obs.shape[1]),
        n_demos_in_grid=int(len(starts)),
        n_demos_trained=n_demos,
        n_rows_trained=n_rows,
        every_trained_row_has_two_balls=bool(obs.shape[1] == 2 and n_rows <= obs.shape[0]),
        chosen_values=[int(v) for v in np.unique(chosen[:n_rows])],
        rows_balls_differ=int(np.sum(sep[:n_rows] > 1e-3)),
        rows_balls_within_1mm=int(np.sum(sep[:n_rows] <= 1e-3)),
        timestep_separation_mm_min=float(sep[:n_rows].min() * 1000),
        timestep_separation_mm_median=float(np.median(sep[:n_rows]) * 1000),
        robot_columns_max_abs=robot_gap,
        demo_landing_separation_mm_min=summary.get('separation_mm_min'),
        one_ball_scenes_in_grid=0 if obs.shape[1] == 2 else None)


def _windows(obs, starts, lengths, demo, cfg, idx):
    hist_off = np.arange(cfg.frames - 1, -1, -1) * cfg.stride
    ep_start = starts[demo[idx]]
    h = np.maximum(idx[:, None] - hist_off[None, :], ep_start[:, None])
    return obs[h]


def drop_second_ball(seeds, n_rows):
    """Executed chunk with both balls, against the same row with only the chosen ball.

    This is the DeepSets shift: the head sees the sum of one embedding instead of two.
    """
    from config import Config
    from np_policy import NumpyPolicy

    with np.load(GRID) as z:
        obs = np.asarray(z['obs'])
        chosen = np.asarray(z['chosen'])
        starts = np.asarray(z['starts'])
        lengths = np.asarray(z['lengths'])
        demo = np.asarray(z['demo'])
    cfg = Config(**json.loads(TRAIN_RECORD.read_text())['config'])
    trained = int(starts[cfg.demos - 1] + lengths[cfg.demos - 1])
    idx = np.sort(np.random.default_rng(ROW_SEED).choice(trained, size=n_rows, replace=False))
    windows = _windows(obs, starts, lengths, demo, cfg, idx)
    chosen = chosen[idx]
    out = dict(n=int(n_rows), row_seed=ROW_SEED, rows=[int(v) for v in idx[:8]], per_seed={})
    for seed in seeds:
        policy = NumpyPolicy(_model(MDN, seed))
        chunk_gap, embed_gap, relative, flipped = [], [], [], []
        for window, slot in zip(windows, chosen):
            both = np.transpose(window, (1, 0, 2))
            one = both[int(slot):int(slot) + 1]
            _raw, pi, _mu, chunk = policy.mdn_stages(both)
            _raw1, pi1, _mu1, chunk1 = policy.mdn_stages(one)
            k, k1 = int(np.argmax(pi)), int(np.argmax(pi1))
            executed = chunk[k]
            executed_one = chunk1[k1]
            chunk_gap.append(np.abs(executed - executed_one))
            flipped.append(k != k1)
            z_both = policy._encode_set(both, None)
            z_one = policy._encode_set(one, None)
            delta = z_both - z_one
            embed_gap.append(float(np.max(np.abs(delta))))
            denom = float(np.linalg.norm(z_both))
            relative.append(float(np.linalg.norm(delta) / denom) if denom else 0.0)
        gap = np.stack(chunk_gap)
        flipped = np.asarray(flipped)
        same = gap[~flipped] if np.any(~flipped) else gap[:0]
        out['per_seed'][f's{seed}'] = dict(
            executed_chunk_max_abs=float(gap.max()),
            executed_chunk_mean_abs=float(gap.mean()),
            base_command_max_abs=float(gap[..., :4].max()),
            base_command_mean_abs=float(gap[..., :4].mean()),
            argmax_changed=float(flipped.mean()),
            same_component_chunk_max_abs=None if len(same) == 0 else float(same.max()),
            same_component_base_mean_abs=None if len(same) == 0 else float(same[..., :4].mean()),
            embedding_max_abs=float(np.max(embed_gap)),
            embedding_relative_l2_mean=float(np.mean(relative)))
    gaps = [out['per_seed'][f's{seed}'] for seed in seeds]
    mean = {}
    for key in gaps[0]:
        vals = [g[key] for g in gaps if g[key] is not None]
        mean[key] = None if not vals else float(np.mean(vals))
    out['mean'] = mean
    return out


def _published():
    mdn = json.loads(MDN_RECORD.read_text())['oneball']['mean_weight']
    mse = json.loads(COMPARE_RECORD.read_text())['mse']['mean']
    curve = json.loads(CURVE_RECORD.read_text())
    misses = [curve[f's{seed}']['miss_mm'] for seed in (0, 1, 2)]
    return dict(
        ceiling_catch_50mm=float(curve['mean_curve']['50']),
        ceiling_miss_mm=float(np.mean(misses)),
        mdn_oneball_catch=float(mdn['catch_either']),
        mdn_oneball_miss_mm=float(mdn['miss_nearest_mm']),
        twoball_mse_catch=float(mse['catch_either']),
        twoball_mse_miss_mm=float(mse['miss_nearest_mm']))


def _reading(audit, mse_catch, mse_miss, ceiling_catch, published):
    """What the control says about using one-ball scenes as evidence."""
    if not audit['every_trained_row_has_two_balls']:
        premise = 'The training grid is not two balls on every row. The premise check failed its own setup.'
        label = 'grid_unexpected'
    else:
        premise = (
            'The MDN trained on two-ball rows only. A one-ball input was not in its training set. '
            'With DeepSets, the head sees the sum of one ball embedding instead of two.')
        label = 'two_ball_training_only'
    ref = ceiling_catch if ceiling_catch is not None else published['ceiling_catch_50mm']
    ref_name = 'the ceiling in this harness' if ceiling_catch is not None else 'the published one-ball ceiling (86.5%)'
    if mse_catch is None:
        control = 'The squared-error control has not been rolled out.'
        control_label = 'not_run'
    elif mse_catch <= 0.25 or (mse_miss is not None and mse_miss >= 400):
        control = (
            f'The two-ball squared-error robot catches {mse_catch:.1%} on the one-ball test scenes '
            f'(mean miss {mse_miss:.0f} mm), against {ref:.1%} for {ref_name}. '
            'It collapses too, so one-ball scenes are not a clean test for either model. '
            'The argument has to rest on the two-ball results and the hand-off test.')
        control_label = 'not_clean'
    elif mse_catch >= 0.70 and ref - mse_catch <= 0.15:
        control = (
            f'The two-ball squared-error robot catches {mse_catch:.1%} on the one-ball test scenes '
            f'(mean miss {mse_miss:.0f} mm), against {ref:.1%} for {ref_name}. '
            'One-ball scenes are a fair test, and the MDN miss on them belongs to the MDN.')
        control_label = 'fair'
    else:
        control = (
            f'The two-ball squared-error robot catches {mse_catch:.1%} on the one-ball test scenes '
            f'(mean miss {mse_miss:.0f} mm), against {ref:.1%} for {ref_name}. '
            'That is neither a collapse nor a match to the ceiling, so one-ball scenes are a weaker test '
            'than the two-ball ones.')
        control_label = 'mixed'
    return dict(label=label, control=control_label, premise=premise, control_text=control)


def init_worker(specs, seeds):
    global _policies, _specs, _seeds
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import eval_mdn
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    eval_mdn._env = CatchEnv()
    _specs = specs
    _seeds = seeds
    _policies = {}
    for _name, path in specs:
        if path not in _policies:
            _policies[path] = NumpyPolicy(path)


def job(index):
    from eval_mdn import _one_ball
    from eval_two_ball import _row
    from np_policy import SetPolicyDriver
    seed = int(_seeds[index])
    out = dict(index=int(index), seed=seed)
    for name, path in _specs:
        driver = SetPolicyDriver(_policies[path])
        best, nearest, aims, aim_times = _one_ball(driver, seed)
        out[name] = _row(best, nearest, aims, aim_times, 0, False)
    return out


def _rollout(seeds, ceiling, limit, workers):
    from eval_two_ball import _mean, _stats
    test = json.loads(SEEDS_PATH.read_text())['test'][:200]
    if limit:
        test = test[:limit]
    specs = []
    for seed in seeds:
        specs.append((f'mse_s{seed}', str(_model(MSE, seed))))
        if ceiling:
            specs.append((f'ceiling_s{seed}', str(_model(CEILING, seed))))
    print(f'one-ball control: {len(test)} scenes, {len(specs)} models, {workers} workers', flush=True)
    rows = []
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(specs, test)) as pool:
        futs = [pool.submit(job, i) for i in range(len(test))]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0 or n == len(futs):
                print(f'  {n}/{len(test)}', flush=True)
    rows.sort(key=lambda r: r['index'])
    block = {}
    mse_stats, ceiling_stats = [], []
    for name, _path in specs:
        stats = _stats([r[name] for r in rows], False)
        block[name] = stats
        miss = stats['miss_nearest_mm']
        miss_s = 'n/a' if miss is None else f'{miss:.1f}'
        print(f"{name}: catch {stats['catch_either']:.3f}  miss {miss_s} mm", flush=True)
        (ceiling_stats if name.startswith('ceiling') else mse_stats).append(stats)
    mean = {}
    for key in ('catch_either', 'miss_nearest_mm'):
        value, spread = _mean(mse_stats, key)
        mean[key] = value
        mean[key + '_sd'] = spread
    ceiling_mean = None
    if ceiling_stats:
        ceiling_mean = {}
        for key in ('catch_either', 'miss_nearest_mm'):
            value, spread = _mean(ceiling_stats, key)
            ceiling_mean[key] = value
            ceiling_mean[key + '_sd'] = spread
    return dict(n_scenes=len(test), per_model=block, mse_mean=mean, ceiling_mean=ceiling_mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--rows', type=int, default=N_ROWS)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ceiling', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--out', type=Path, default=RESULTS / 'stage2' / 'mdn_premise.json')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(',') if s)
    audit = audit_grid()
    print(json.dumps(audit, indent=2), flush=True)
    drop = drop_second_ball(seeds, args.rows)
    print('drop second ball', json.dumps(drop['mean']), flush=True)
    published = _published()
    rollout = None
    if not args.audit_only:
        rollout = _rollout(seeds, args.ceiling, args.limit, args.workers)
    mse_catch = None if rollout is None else rollout['mse_mean']['catch_either']
    mse_miss = None if rollout is None else rollout['mse_mean']['miss_nearest_mm']
    ceiling_catch = None
    if rollout is not None and rollout['ceiling_mean'] is not None:
        ceiling_catch = rollout['ceiling_mean']['catch_either']
    reading = _reading(audit, mse_catch, mse_miss, ceiling_catch, published)
    report = dict(
        question='Was the MDN trained on any one-ball scene, and does the two-ball squared-error robot survive the one-ball test?',
        audit=audit, drop_second_ball=drop, published=published, rollout=rollout, reading=reading)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))
    print(json.dumps(reading, indent=2), flush=True)
    print(f'wrote {args.out}', flush=True)
    full = not args.limit and not args.audit_only and not args.no_log
    if full:
        from stage2_log import finish_run, start_run
        run_id, record_path, record = start_run(
            'mdn-premise', 'mse', dict(seeds=list(seeds), ceiling=bool(args.ceiling)),
            ['premise_check.py'], dict(scenes=rollout['n_scenes']),
            note='two-ball grid audit, and the two-ball squared-error robot on one-ball scenes')
        finish_run(record_path, record, dict(
            miss_mm=mse_miss, catch_rate=mse_catch), [args.out])


if __name__ == '__main__':
    main()
