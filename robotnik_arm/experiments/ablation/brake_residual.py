"""Offline check: is the learned forward speed high on expert states near the stop?

Held-out scenes are the 200 test seeds. The expert is logged at 1 kHz and
downsampled onto the same 50 Hz grid the policies train on. The stop is the
first sample after the last driving command. On every grid state in the two
seconds up to and including that stop, each ceiling network predicts the
command it would issue now, and the residual is predicted forward speed minus
the expert's forward speed.

Forward speed is the body speed implied by the four wheel commands, the same
map the expert's base controller inverts when it writes those commands.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
GRID = DATA / 'heldout_expert_grid50.npz'
OUT = RESULTS / 'stage2' / 'brake_residual.json'
PLOT = RESULTS / 'stage2' / 'brake_residual.png'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
SEEDS = (0, 1, 2)

GRID_HZ = 50
WHEEL_RADIUS = 0.11
WHEEL_WIDTH = 0.66
WINDOW_S = 2.0
CRUISE_S = 4.0          # control window is [stop-4s, stop-2s]
MOVE_V = 0.02           # m/s; drive commands sit above the 0.05 m/s floor
STOP_TAIL_S = 0.4
_env = None


def forward_speed(cmd):
    """Body forward speed, m/s. ``cmd`` is (..., 4) wheels or (..., 18) commands.

    Order is front_right, front_left, back_left, back_right. Left and right are
    averaged across the axle, then averaged together, then scaled by the radius.
    """
    wheels = np.asarray(cmd, dtype=np.float64)[..., :4]
    right = 0.5 * (wheels[..., 0] + wheels[..., 3])
    left = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return 0.5 * (right + left) * WHEEL_RADIUS


def yaw_rate(cmd):
    """Body yaw rate, rad/s, from the same wheel commands."""
    wheels = np.asarray(cmd, dtype=np.float64)[..., :4]
    right = 0.5 * (wheels[..., 0] + wheels[..., 3])
    left = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return (right - left) * WHEEL_RADIUS / WHEEL_WIDTH


def _check_wheel_map():
    """The expert writes wheels from (v, w); the inverse must return that v and w."""
    v, w = 0.2, -0.15
    left = (v - w * WHEEL_WIDTH / 2) / WHEEL_RADIUS
    right = (v + w * WHEEL_WIDTH / 2) / WHEEL_RADIUS
    cmd = np.array([right, left, left, right])
    if abs(forward_speed(cmd) - v) > 1e-9 or abs(yaw_rate(cmd) - w) > 1e-9:
        raise RuntimeError('wheel map does not invert the base controller')


def init_worker():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    _env = CatchEnv()


def expert_grid_job(seed):
    from catch_env import FIRST_TICK_STEP, ExpertDriver, rollout
    out = rollout(_env, ExpertDriver(), int(seed), 1000, stop_on_catch=True, log=True)
    arrays = out['arrays']
    stride = int(round(0.001 / _env.dt)) * (1000 // GRID_HZ)
    # build_dataset keeps the command at every 50 Hz grid point of the 1 kHz log.
    obs = np.asarray(arrays['obs'][FIRST_TICK_STEP::stride], dtype=np.float32)
    cmd = np.asarray(arrays['cmd'][FIRST_TICK_STEP::stride], dtype=np.float32)
    return dict(seed=int(seed), obs=obs, cmd=cmd, success=bool(out['success']))


def build_grid(seeds, workers=16):
    if GRID.exists():
        with np.load(GRID) as z:
            saved = [int(s) for s in z['seeds']]
        if saved == [int(s) for s in seeds]:
            print(f'reusing {GRID.name}', flush=True)
            return
        raise SystemExit(f'{GRID.name} was built for a different seed list')
    print(f'expert logs for {len(seeds)} held-out seeds', flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        futs = [pool.submit(expert_grid_job, int(s)) for s in seeds]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(f'  expert {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    rows.sort(key=lambda r: seeds.index(r['seed']))
    obs = np.concatenate([r['obs'] for r in rows])
    cmd = np.concatenate([r['cmd'] for r in rows])
    demo, starts, lengths = [], [], []
    offset = 0
    for i, r in enumerate(rows):
        n = len(r['obs'])
        demo.append(np.full(n, i, dtype=np.int32))
        starts.append(offset)
        lengths.append(n)
        offset += n
    np.savez(GRID, obs=obs, cmd=cmd, demo=np.concatenate(demo),
             starts=np.asarray(starts, np.int64), lengths=np.asarray(lengths, np.int64),
             seeds=np.asarray([r['seed'] for r in rows], np.int64),
             success=np.asarray([r['success'] for r in rows]))
    print(f'wrote {GRID.name}: {len(rows)} episodes, {offset} rows, {time.time() - t0:.0f}s', flush=True)


def stop_index(cmd):
    """First grid index at which the forward command has finished the drive.

    The drive is the last run of commands faster than ``MOVE_V``. The stop is
    the next sample, and it only counts if the command then stays near zero.
    """
    v = forward_speed(cmd)
    moving = np.flatnonzero(v > MOVE_V)
    if len(moving) == 0:
        return None
    last = int(moving[-1])
    stop = last + 1
    tail_n = int(round(STOP_TAIL_S * GRID_HZ))
    tail = v[stop:stop + tail_n]
    if len(tail) < tail_n // 2 or float(np.median(tail)) > 0.01:
        return None
    return stop


def window_indices(n, stop, t0, t1):
    """Grid indices whose time relative to ``stop`` lies in [t0, t1]."""
    i0 = max(0, int(np.ceil(stop + t0 * GRID_HZ - 1e-9)))
    i1 = min(n - 1, int(np.floor(stop + t1 * GRID_HZ + 1e-9)))
    if i1 < i0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(i0, i1 + 1, dtype=np.int64)


def load_episodes():
    with np.load(GRID) as z:
        obs, cmd = z['obs'], z['cmd']
        starts, lengths = z['starts'], z['lengths']
        seeds = [int(s) for s in z['seeds']]
    episodes = []
    for i, seed in enumerate(seeds):
        a, n = int(starts[i]), int(lengths[i])
        episodes.append(dict(seed=seed, obs=obs[a:a + n], cmd=cmd[a:a + n]))
    return episodes


def gather_queries(episodes):
    """States in the brake window and the earlier cruise window, per episode."""
    brake_idx, cruise_idx = [], []
    kept, skipped = [], []
    for ep_i, ep in enumerate(episodes):
        stop = stop_index(ep['cmd'])
        if stop is None:
            skipped.append(ep['seed'])
            continue
        brake = window_indices(len(ep['cmd']), stop, -WINDOW_S, 0.0)
        # Cruise ends one sample before the brake window so the boundary is not in both.
        cruise = window_indices(len(ep['cmd']), stop, -CRUISE_S, -WINDOW_S - 1.0 / GRID_HZ)
        if len(brake) < int(1.5 * GRID_HZ):
            skipped.append(ep['seed'])
            continue
        kept.append(dict(ep=ep_i, stop=stop, brake=brake, cruise=cruise))
        brake_idx.append(np.column_stack((np.full(len(brake), ep_i), brake)))
        if len(cruise):
            cruise_idx.append(np.column_stack((np.full(len(cruise), ep_i), cruise)))
    if not kept:
        raise SystemExit('no held-out episode had a clean stop')
    return kept, np.concatenate(brake_idx), (np.concatenate(cruise_idx) if cruise_idx else np.zeros((0, 2), np.int64)), skipped


def predict_now(model, episodes, queries, device, batch=4096):
    """Immediate (horizon step 0) command at each (episode, index) query."""
    import torch
    cfg = model.cfg
    hist_off = np.arange(cfg.frames - 1, -1, -1) * cfg.stride
    model = model.to(device).eval()
    out = np.empty((len(queries), 18), dtype=np.float32)
    with torch.no_grad():
        for sl in range(0, len(queries), batch):
            q = queries[sl:sl + batch]
            frames = np.empty((len(q), cfg.frames, episodes[0]['obs'].shape[-1]), np.float32)
            for row, (ep_i, idx) in enumerate(q):
                ep = episodes[int(ep_i)]
                pos = np.maximum(int(idx) - hist_off, 0)
                frames[row] = ep['obs'][pos]
            pred = model.predict(torch.from_numpy(frames).to(device))
            out[sl:sl + len(q)] = pred[:, 0].detach().cpu().numpy()
    return out


def pack(episodes, queries, pred):
    """Align residuals onto a fixed grid of time-to-stop. NaN where a state is absent."""
    n_t = int(round(WINDOW_S * GRID_HZ)) + 1
    times = np.linspace(-WINDOW_S, 0.0, n_t)
    label_v = np.full((len(episodes), n_t), np.nan)
    pred_v = np.full_like(label_v, np.nan)
    label_w = np.full_like(label_v, np.nan)
    pred_w = np.full_like(label_v, np.nan)
    # queries may cover either window; only those inside [-WINDOW, 0] are stored here
    for row, (ep_i, idx) in enumerate(queries):
        ep = episodes[int(ep_i)]
        stop = stop_index(ep['cmd'])
        t = (int(idx) - stop) / GRID_HZ
        if t < -WINDOW_S - 1e-9 or t > 1e-9:
            continue
        k = int(round((t + WINDOW_S) * GRID_HZ))
        if not 0 <= k < n_t:
            continue
        label_v[int(ep_i), k] = forward_speed(ep['cmd'][int(idx)])
        pred_v[int(ep_i), k] = forward_speed(pred[row])
        label_w[int(ep_i), k] = yaw_rate(ep['cmd'][int(idx)])
        pred_w[int(ep_i), k] = yaw_rate(pred[row])
    return times, label_v, pred_v, label_w, pred_w


def cruise_residual(episodes, queries, pred):
    """Mean forward residual in the cruise window, one number per episode that has it."""
    by_ep = {}
    for row, (ep_i, idx) in enumerate(queries):
        ep = episodes[int(ep_i)]
        stop = stop_index(ep['cmd'])
        t = (int(idx) - stop) / GRID_HZ
        if t < -CRUISE_S - 1e-9 or t > -WINDOW_S + 1e-9:
            continue
        err = forward_speed(pred[row]) - forward_speed(ep['cmd'][int(idx)])
        by_ep.setdefault(int(ep_i), []).append(err)
    if not by_ep:
        return None
    means = np.array([np.mean(v) for v in by_ep.values()])
    return mean_ci(means * 1000.0)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(len(values))
    mean = float(np.mean(values))
    if n < 2:
        return dict(n=n, mean=mean, median=mean, std=0.0, ci95=(mean, mean), frac_positive=float(np.mean(values > 0)))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    return dict(n=n, mean=mean, median=float(np.median(values)), std=float(np.std(values, ddof=1)),
                ci95=(mean - 1.96 * se, mean + 1.96 * se),
                frac_positive=float(np.mean(values > 0)))


def curve_stats(times, label_v, pred_v, label_w, pred_w):
    """Episode-level residual of forward speed, plus the mean ramp."""
    resid = pred_v - label_v
    with np.errstate(all='ignore'):
        ep = np.nanmean(resid, axis=1)
    present = np.isfinite(resid).any(axis=1)
    ep = ep[present]
    summary = mean_ci(ep * 1000.0)
    # 0.1 s bins of the seed-averaged residual, for the "all the way down" check
    bin_w = 0.1
    edges = np.arange(-WINDOW_S, 1e-9, bin_w)
    bins = []
    for edge in edges:
        take = (times >= edge - 1e-9) & (times < edge + bin_w - 1e-9)
        if edge + bin_w >= -1e-9:
            take = (times >= edge - 1e-9) & (times <= 1e-9)
        col = resid[:, take]
        flat = col[np.isfinite(col)]
        if len(flat) == 0:
            continue
        bins.append(dict(t_s=float(edge + bin_w / 2), n=int(len(flat)),
                         residual_mm_s=float(np.mean(flat) * 1000),
                         label_mm_s=float(np.nanmean(label_v[:, take]) * 1000),
                         pred_mm_s=float(np.nanmean(pred_v[:, take]) * 1000)))
    yaw = pred_w - label_w
    yaw_ep = np.nanmean(yaw, axis=1)
    yaw_ep = yaw_ep[np.isfinite(yaw_ep)]
    labels = label_v[present]
    summary.update(
        frac_bins_positive=float(np.mean([b['residual_mm_s'] > 0 for b in bins])) if bins else None,
        n_bins=len(bins),
        bins=bins,
        yaw=mean_ci(yaw_ep),
        label_min_mm_s=float(np.nanmin(labels) * 1000),
        label_frac_negative=float(np.mean(labels[np.isfinite(labels)] < -1e-6)),
        samples=int(np.isfinite(resid[present]).sum()),
    )
    return summary


def judge(summary):
    """Consistently positive, centred on zero, or neither.

    The unit of the test is the episode, not the timestep: timesteps inside one
    approach are not independent. Centred means the 95% interval of the episode
    mean covers zero and the mean itself is under 1 mm/s. Consistently positive
    means that interval lies above zero, the median episode is positive, and the
    residual stays positive across the two-second ramp rather than in one bin.
    """
    lo, hi = summary['ci95']
    mean, median = summary['mean'], summary['median']
    bins_pos = summary['frac_bins_positive'] or 0.0
    if lo <= 0.0 <= hi and abs(mean) < 1.0:
        return 'centred'
    if lo > 0.0 and median > 0.0 and bins_pos >= 0.7:
        return 'positive'
    return 'inconclusive'


def save_plot(times, label_v, pred_v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    resid = (pred_v - label_v) * 1000.0
    with np.errstate(all='ignore'):
        mean_r = np.nanmean(resid, axis=0)
        mean_l = np.nanmean(label_v, axis=0) * 1000.0
        mean_p = np.nanmean(pred_v, axis=0) * 1000.0
        # episode spread of the residual
        p10 = np.nanpercentile(resid, 10, axis=0)
        p90 = np.nanpercentile(resid, 90, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    axes[0].plot(times, mean_l, label='expert label', color='0.25')
    axes[0].plot(times, mean_p, label='predicted', color='C0')
    axes[0].set_ylabel('forward speed (mm/s)')
    axes[0].legend(frameon=False)
    axes[0].set_title('Held-out expert states, last 2 s before the stop')
    axes[1].axhline(0.0, color='0.5', lw=0.8)
    axes[1].fill_between(times, p10, p90, color='C0', alpha=0.2, linewidth=0, label='episode 10–90%')
    axes[1].plot(times, mean_r, color='C0', label='mean residual')
    axes[1].set_ylabel('predicted − label (mm/s)')
    axes[1].set_xlabel('time relative to the stop (s)')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=120)
    plt.close(fig)


def main():
    _check_wheel_map()
    test = json.loads((DATA / 'seeds.json').read_text())['test'][:200]
    from stage2_log import finish_run, start_run
    run_id, record_path, record = start_run(
        'brake-residual', CEILING, dict(window_s=WINDOW_S, grid_hz=GRID_HZ, seeds=list(SEEDS)),
        sys.argv, dict(test=len(test)),
        note='predicted minus label forward speed on held-out states in the last 2s before the stop')
    t0 = time.time()
    build_grid(test)
    episodes = load_episodes()
    kept, brake_q, cruise_q, skipped = gather_queries(episodes)
    print(f'{len(kept)} episodes with a stop, {len(skipped)} skipped, '
          f'{len(brake_q)} brake states', flush=True)
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from policy import load_policy
    per_seed = []
    sum_label = sum_pred = sum_lw = sum_pw = None
    n_models = 0
    cruise = []
    for seed in SEEDS:
        path = RESULTS / 'models' / f'{CEILING}_s{seed}.pt'
        if not path.exists():
            raise SystemExit(f'missing {path}')
        print(f'predict seed {seed} on {device}', flush=True)
        model = load_policy(path)
        if model.cfg.encoder != 'deepsets' or model.cfg.head != 'mse':
            raise SystemExit(f'{path.name} is not the DeepSets MSE ceiling')
        pred_b = predict_now(model, episodes, brake_q, device)
        times, label_v, pred_v, label_w, pred_w = pack(episodes, brake_q, pred_b)
        # pack() writes into rows indexed by episode id, including skipped ones.
        # Restrict the summary to episodes that were kept.
        keep_rows = np.zeros(len(episodes), dtype=bool)
        for item in kept:
            keep_rows[item['ep']] = True
        stats = curve_stats(times, label_v[keep_rows], pred_v[keep_rows], label_w[keep_rows], pred_w[keep_rows])
        pred_c = predict_now(model, episodes, cruise_q, device) if len(cruise_q) else None
        cruise_stats = None if pred_c is None else cruise_residual(episodes, cruise_q, pred_c)
        per_seed.append(dict(seed=seed, brake=stats, cruise=cruise_stats, verdict=judge(stats)))
        print(f"  seed {seed}: residual {stats['mean']:+.2f} mm/s  "
              f"ci [{stats['ci95'][0]:+.2f}, {stats['ci95'][1]:+.2f}]  "
              f"median {stats['median']:+.2f}  bins+ {stats['frac_bins_positive']:.2f}  "
              f"{per_seed[-1]['verdict']}", flush=True)
        if sum_label is None:
            sum_label, sum_pred = label_v.copy(), pred_v.copy()
            sum_lw, sum_pw = label_w.copy(), pred_w.copy()
        else:
            sum_label += label_v
            sum_pred += pred_v
            sum_lw += label_w
            sum_pw += pred_w
        n_models += 1
        cruise.append(cruise_stats)
        del model
    label_m = sum_label / n_models
    pred_m = sum_pred / n_models
    # Labels do not depend on the network; the average equals each seed's labels.
    pooled = curve_stats(times, label_m[keep_rows], pred_m[keep_rows],
                         (sum_lw / n_models)[keep_rows], (sum_pw / n_models)[keep_rows])
    verdict = judge(pooled)
    cruise_means = [c['mean'] for c in cruise if c is not None]
    report = dict(
        definition=dict(
            states='200 held-out test seeds, expert logged at 1 kHz and sampled at 50 Hz',
            stop='first 50 Hz sample after the last wheel command faster than 0.02 m/s, requiring the following 0.4 s to stay near zero',
            residual='horizon-0 predicted body forward speed minus the expert command at the same state',
            window='grid states from 2 s before that stop through the stop sample',
            unit='episode mean, then the mean across episodes; the interval is the 95% interval of that mean',
        ),
        n_episodes=len(kept),
        n_skipped=len(skipped),
        pooled=pooled,
        verdict=verdict,
        per_seed=[{'seed': s['seed'], 'verdict': s['verdict'],
                   'mean_mm_s': s['brake']['mean'], 'median_mm_s': s['brake']['median'],
                   'ci95_mm_s': s['brake']['ci95'], 'frac_positive': s['brake']['frac_positive'],
                   'frac_bins_positive': s['brake']['frac_bins_positive'],
                   'yaw_mean_rad_s': s['brake']['yaw']['mean'],
                   'yaw_ci95': s['brake']['yaw']['ci95'],
                   'cruise_mean_mm_s': None if s['cruise'] is None else s['cruise']['mean'],
                   'cruise_ci95_mm_s': None if s['cruise'] is None else s['cruise']['ci95'],
                   'label_frac_negative': s['brake']['label_frac_negative'],
                   'label_min_mm_s': s['brake']['label_min_mm_s']} for s in per_seed],
        cruise_mean_mm_s=None if not cruise_means else float(np.mean(cruise_means)),
    )
    # Drop the bulky per-seed bin lists from the on-disk pooled copy's duplicate;
    # the pooled bins are the plot source and stay.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # per_seed stats contain bins too; keep them, they are small (20 bins).
    slim = json.loads(json.dumps(report, default=lambda x: list(x) if isinstance(x, tuple) else x))
    OUT.write_text(json.dumps(slim, indent=2))
    try:
        save_plot(times, label_m[keep_rows], pred_m[keep_rows])
    except Exception as exc:
        print(f'plot skipped: {exc}', flush=True)
    print(f"pooled: {pooled['mean']:+.2f} mm/s  "
          f"ci [{pooled['ci95'][0]:+.2f}, {pooled['ci95'][1]:+.2f}]  "
          f"median {pooled['median']:+.2f}  frac+ {pooled['frac_positive']:.2f}  "
          f"bins+ {pooled['frac_bins_positive']:.2f}  yaw {pooled['yaw']['mean']:+.4f} rad/s  "
          f"verdict {verdict}", flush=True)
    finish_run(record_path, record, dict(miss_mm=pooled['mean'], catch_rate=pooled['frac_positive']),
               [OUT, PLOT])
    print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
