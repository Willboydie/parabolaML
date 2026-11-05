"""Does the NumPy rollout match the trained mixture, stage by stage?

About 100 two-ball training rows go through the PyTorch model and through the
NumPy rollout. The maximum absolute difference is reported at each stage:

    trunk embedding
    raw logits, and the weights after softmax
    component means before un-standardising
    every component's chunk after un-standardising

A gap that starts at one stage names the fault. The usual ones are a reshape
order other than (component, time, action), softmax over the wrong axis, and
un-standardising with the wrong statistics. The six reshape orders are scored
on the flat mean head so a wrong order can be named.

The free comparison uses the same pass on one-ball training rows: the chunk the
MDN would execute against the DeepSets ceiling's chunk, and both against the
expert chunk. Those rows are in the ceiling's training set and not in the MDN's.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data'
TRAIN_RECORD = RESULTS / 'stage2' / 'two_ball_mdn_train.json'
CEILING_RECORD = RESULTS / 'stage2' / 'oneball_deepsets.json'
TWO_BALL = DATA / 'grid50_two_ball.npz'
MDN = 'twoball_mdn_s{seed}'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets_s{seed}'
N_ROWS = 100
ROW_SEED = 0
MATCH = 1e-4
ORDERS = ('KHA', 'KAH', 'HKA', 'HAK', 'AKH', 'AHK')
STAGES = ('embedding', 'logits', 'pi', 'mu_norm', 'chunk')


def _model(pattern, seed, suffix):
    path = RESULTS / 'models' / f'{pattern.format(seed=seed)}.{suffix}'
    if not path.exists():
        raise SystemExit(f'missing {path}')
    return path


def _windows(obs, starts, lengths, demo, cfg, idx):
    hist_off = np.arange(cfg.frames - 1, -1, -1) * cfg.stride
    ep_start = starts[demo[idx]]
    h = np.maximum(idx[:, None] - hist_off[None, :], ep_start[:, None])
    return obs[h]


def _sample(starts, lengths, demos, n):
    trained = int(starts[demos - 1] + lengths[demos - 1])
    if n > trained:
        raise SystemExit(f'asked for {n} rows and the training prefix has {trained}')
    idx = np.random.default_rng(ROW_SEED).choice(trained, size=n, replace=False)
    return np.sort(idx), trained


def _expert_chunk(cmd, obs, starts, lengths, demo, cfg, idx, q_col, two_ball):
    chunk_off = np.arange(cfg.horizon) * cfg.stride
    ep_last = (starts + lengths - 1)[demo[idx]]
    c = np.minimum(idx[:, None] + chunk_off[None, :], ep_last[:, None])
    y = np.array(cmd[c], copy=True)
    q_now = obs[idx, 0, q_col:q_col + 7] if two_ball else obs[idx, q_col:q_col + 7]
    y[:, :, 4:11] -= q_now[:, None, :]
    return y


def _reshape(flat, order, k, horizon, act):
    size = {'K': k, 'H': horizon, 'A': act}
    arr = np.asarray(flat, dtype=np.float32).reshape(tuple(size[c] for c in order))
    src = {name: i for i, name in enumerate(order)}
    return np.transpose(arr, (src['K'], src['H'], src['A']))


def _first_gap(gaps):
    for name in STAGES:
        if gaps[name] > MATCH:
            return name
    return None


def _torch_stages(model, windows, two_ball):
    import torch
    from train import mixture_inputs
    frames = torch.from_numpy(np.ascontiguousarray(windows))
    chosen = torch.zeros(frames.shape[0], dtype=torch.long)
    with torch.no_grad():
        robot, balls = mixture_inputs(model, frames, chosen, two_ball)
        encoded = model._encode(robot, balls)
        logits = model.pi_head(encoded)
        log_pi, mu, _sigma = model.mixture(robot, balls)
        pi = log_pi.exp()
        chunk = mu * model.act_std + model.act_mean
    return (encoded.numpy(), logits.numpy(), pi.numpy(), mu.numpy(), chunk.numpy())


def compare_seed(seed, windows):
    """Maximum gap at each stage, and which reshape order matches the mean head."""
    from np_policy import NumpyPolicy
    from policy import load_policy

    model = load_policy(_model(MDN, seed, 'pt'))
    policy = NumpyPolicy(_model(MDN, seed, 'npz'))
    encoded, logits, pi, mu, chunk = _torch_stages(model, windows, True)
    k, horizon, act = mu.shape[1:]
    gaps = {name: 0.0 for name in STAGES}
    order_gap = {name: 0.0 for name in ORDERS}
    argmax_agree = 0
    stats = dict(
        act_mean=float(np.max(np.abs(policy.act_mean - model.act_mean.numpy()))),
        act_std=float(np.max(np.abs(policy.act_std - model.act_std.numpy()))))
    for i, window in enumerate(windows):
        frames = np.transpose(window, (1, 0, 2))
        z = policy._encode_set(frames, None)
        raw, pi_np, mu_np, chunk_np = policy.mdn_stages(frames)
        gaps['embedding'] = max(gaps['embedding'], float(np.max(np.abs(z - encoded[i]))))
        gaps['logits'] = max(gaps['logits'], float(np.max(np.abs(raw - logits[i]))))
        gaps['pi'] = max(gaps['pi'], float(np.max(np.abs(pi_np - pi[i]))))
        gaps['mu_norm'] = max(gaps['mu_norm'], float(np.max(np.abs(mu_np - mu[i]))))
        gaps['chunk'] = max(gaps['chunk'], float(np.max(np.abs(chunk_np - chunk[i]))))
        argmax_agree += int(np.argmax(pi_np) == np.argmax(pi[i]))
        mu_w, mu_b = policy.mixture['mu']
        flat = z @ mu_w + mu_b
        for order in ORDERS:
            trial = _reshape(flat, order, k, horizon, act)
            order_gap[order] = max(order_gap[order], float(np.max(np.abs(trial - mu[i]))))
    best = min(ORDERS, key=lambda name: order_gap[name])
    return dict(
        n=int(len(windows)),
        max_abs=gaps,
        first_gap=_first_gap(gaps),
        argmax_agreement=float(argmax_agree / len(windows)),
        standardisation_max_abs=stats,
        reshape_max_abs=order_gap,
        reshape_best=best,
        reshape_matches_training=bool(best == 'KHA' and order_gap['KHA'] <= MATCH))


def _chunk_gap(a, b):
    diff = np.abs(a - b)
    row_max = diff.max(axis=(-1, -2))
    base_row = diff[..., :4].max(axis=(-1, -2))
    return dict(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        base_max_abs=float(diff[..., :4].max()),
        base_mean_abs=float(diff[..., :4].mean()),
        row_max_abs_median=float(np.median(row_max)),
        row_max_abs_p95=float(np.percentile(row_max, 95)),
        base_row_max_abs_median=float(np.median(base_row)),
        base_row_max_abs_p95=float(np.percentile(base_row, 95)))


def one_ball_chunks(seeds, n_rows):
    """MDN chunk against the ceiling chunk, on one-ball rows the ceiling trained on."""
    from np_policy import NumpyPolicy

    record = json.loads(CEILING_RECORD.read_text())
    grid = Path(record['data'])
    with np.load(grid) as z:
        obs = np.asarray(z['obs'])
        cmd = np.asarray(z['cmd'])
        starts = np.asarray(z['starts'])
        lengths = np.asarray(z['lengths'])
        demo = np.asarray(z['demo'])
        q_col = int(z['arm_q_col'])
    if obs.ndim != 2 or obs.shape[-1] != 42:
        raise SystemExit(f'expected one-ball observations (rows, 42), got {obs.shape}')
    ceiling = NumpyPolicy(_model(CEILING, seeds[0], 'npz'))
    idx, trained = _sample(starts, lengths, ceiling.cfg.demos, n_rows)
    windows = _windows(obs, starts, lengths, demo, ceiling.cfg, idx)
    expert = _expert_chunk(cmd, obs, starts, lengths, demo, ceiling.cfg, idx, q_col, False)
    ceilings = {}
    for seed in seeds:
        policy = NumpyPolicy(_model(CEILING, seed, 'npz'))
        ceilings[seed] = np.stack([policy.predict_set(window[None]) for window in windows])
    per_seed = {}
    for seed in seeds:
        mdn = NumpyPolicy(_model(MDN, seed, 'npz'))
        executed = []
        for window in windows:
            _raw, pi, _mu, chunk = mdn.mdn_stages(window[None])
            executed.append(chunk[int(np.argmax(pi))])
        executed = np.stack(executed)
        per_seed[f's{seed}'] = dict(
            mdn_vs_ceiling=_chunk_gap(executed, ceilings[seed]),
            mdn_vs_expert=_chunk_gap(executed, expert),
            ceiling_vs_expert=_chunk_gap(ceilings[seed], expert))
    spread = _chunk_gap(ceilings[seeds[0]], ceilings[seeds[-1]]) if len(seeds) > 1 else None
    return dict(
        grid=str(grid),
        n=int(n_rows),
        n_rows_trained=int(trained),
        note='These rows are in the ceiling training set. The MDN trained on two-ball rows only.',
        ceiling_seed_spread=spread,
        per_seed=per_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--rows', type=int, default=N_ROWS)
    parser.add_argument('--out', type=Path, default=RESULTS / 'stage2' / 'mdn_runtime.json')
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(',') if s)
    with np.load(TWO_BALL) as z:
        obs = np.asarray(z['obs'])
        starts = np.asarray(z['starts'])
        lengths = np.asarray(z['lengths'])
        demo = np.asarray(z['demo'])
    cfg_demos = json.loads(TRAIN_RECORD.read_text())['config']['demos']
    idx, trained = _sample(starts, lengths, cfg_demos, args.rows)
    from np_policy import NumpyPolicy
    cfg = NumpyPolicy(_model(MDN, seeds[0], 'npz')).cfg
    windows = _windows(obs, starts, lengths, demo, cfg, idx)
    print(f'{args.rows} two-ball training rows, seeds {seeds}', flush=True)
    per_seed = {}
    for seed in seeds:
        per_seed[f's{seed}'] = compare_seed(seed, windows)
        gap = per_seed[f's{seed}']
        print(f"s{seed}: first gap {gap['first_gap']}  max {json.dumps(gap['max_abs'])}  "
              f"reshape {gap['reshape_best']} {gap['reshape_max_abs'][gap['reshape_best']]:.3e}", flush=True)
    one_ball = one_ball_chunks(seeds, args.rows)
    for seed in seeds:
        row = one_ball['per_seed'][f's{seed}']
        print(f"s{seed} one-ball: mdn vs ceiling max {row['mdn_vs_ceiling']['max_abs']:.4f}  "
              f"mean {row['mdn_vs_ceiling']['mean_abs']:.4f}  "
              f"ceiling vs expert mean {row['ceiling_vs_expert']['mean_abs']:.4f}", flush=True)
    first = [per_seed[f's{seed}']['first_gap'] for seed in seeds]
    report = dict(
        question='Where do the PyTorch model and the NumPy rollout first disagree, and is the MDN chunk close to the ceiling on a one-ball row?',
        match_bar=MATCH,
        n_rows=int(args.rows),
        row_seed=ROW_SEED,
        n_rows_trained=int(trained),
        stages=per_seed,
        stages_agree=all(name is None for name in first),
        one_ball=one_ball)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))
    print(f'wrote {args.out}', flush=True)
    if not args.no_log:
        from stage2_log import finish_run, start_run
        run_id, record_path, record = start_run(
            'mdn-runtime', 'mdn', dict(seeds=list(seeds), rows=args.rows, stages_agree=report['stages_agree']),
            ['mdn_runtime.py'], dict(rows=args.rows),
            note='torch against numpy at four stages, then the MDN chunk against the ceiling on one-ball rows')
        finish_run(record_path, record, dict(miss_mm=None, catch_rate=None), [args.out])


if __name__ == '__main__':
    main()
