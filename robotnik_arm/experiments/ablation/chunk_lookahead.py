"""Chunk lookahead on the held-out expert states. No new rollouts.

The previous check scored only the command the network would issue now. On the
robot the command at a moment comes from a chunk issued up to one second
earlier, so this scores every step of the chunk against the expert command at
that future time.

A query at grid index i predicts steps k = 0..H-1 at i + k * stride. The
predicted stop is the first step after the last predicted command faster than
0.02 m/s, the same rule as the expert's stop. If the chunk is still moving at
its last step, the stop is past the horizon and the query is censored.

The executed lead is the integral of predicted minus expert forward speed over
the steps the robot would actually run before it queries again.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import sys
import time
from pathlib import Path

import numpy as np

from brake_residual import (
    CEILING, GRID_HZ, MOVE_V, RESULTS, SEEDS, _check_wheel_map, forward_speed,
    load_episodes, mean_ci, stop_index,
)

OUT = RESULTS / 'stage2' / 'chunk_lookahead.json'
PLOT = RESULTS / 'stage2' / 'chunk_lookahead.png'
RATE50 = 'r50_p1_e0.5_hstack0.24s_d2000_deepsets'
BIN_S = 0.1


def predict_chunks(model, obs, indices, device):
    """Full denormalised command chunks. indices are grid rows in this episode."""
    import torch
    cfg = model.cfg
    hist_off = np.arange(cfg.frames - 1, -1, -1) * cfg.stride
    hist = np.maximum(indices[:, None] - hist_off[None, :], 0)
    frames = torch.from_numpy(np.ascontiguousarray(obs[hist])).to(device)
    with torch.no_grad():
        pred = model.predict(frames).detach().cpu().numpy()
    return pred


def stop_step(speed):
    """First stopped step, or -1 if ``speed`` is still above the floor at the end.

    ``speed`` is (N, H). Step -1 means the predicted stop is past the chunk.
    """
    moving = speed > MOVE_V
    any_moving = moving.any(axis=1)
    # Last True in each row. argmax on the reversed row is 0 when the last step moves.
    last = speed.shape[1] - 1 - np.argmax(moving[:, ::-1], axis=1)
    still = moving[:, -1]
    step = last + 1
    step = np.where(any_moving, step, 0)
    step = np.where(still, -1, step)
    return step.astype(np.int32)


def query_indices(stop, n, stride, horizon):
    """Policy ticks from one horizon before the stop through the stop."""
    start = max(0, int(stop) - horizon * stride)
    idx = np.arange(start, int(stop) + 1, stride, dtype=np.int64)
    idx = idx[idx < n]
    return idx


def open_query(stop, stride, execute):
    """Grid index of the chunk the robot is executing when the stop arrives.

    Replans fall on multiples of ``execute`` policy ticks, and the first tick of
    the episode is grid index 0. A stop that lands exactly on a replan uses the
    new chunk, because the driver queries before it uses that tick's command.
    """
    period = execute * stride
    if period <= 0:
        raise RuntimeError('execute period must be positive')
    return int(stop) // period * period


def bin_mean(tau, values, lo, hi, width):
    """Mean of ``values`` in bins of ``tau``. One row per bin that contains a sample."""
    rows = []
    edges = np.arange(lo, hi + 1e-9, width)
    for edge in edges:
        take = (tau >= edge - 1e-9) & (tau < edge + width - 1e-9)
        if edge + width >= hi - 1e-9:
            take = (tau >= edge - 1e-9) & (tau <= hi + 1e-9)
        if not np.any(take):
            continue
        rows.append((float(edge + width / 2), values[take]))
    return rows


def summarise_queries(tau, err, censored, speed_at_stop, horizon_s):
    """Stop error and the speed predicted for the stop instant, against lead time."""
    bins = []
    for centre, sel_tau in ((c, (tau >= c - BIN_S / 2) & (tau < c + BIN_S / 2))
                            for c in np.arange(0.0, horizon_s + 1e-9, BIN_S)):
        # rebuild properly below
        pass
    edges = np.arange(0.0, horizon_s + 1e-9, BIN_S)
    for edge in edges:
        take = (tau >= edge - 1e-9) & (tau < edge + BIN_S - 1e-9)
        if not np.any(take):
            continue
        e = err[take]
        c = censored[take]
        finite = e[np.isfinite(e)]
        centre = float(edge + BIN_S / 2)
        # The last partial bin is labelled on its own samples.
        if edge + BIN_S > horizon_s + 1e-9:
            centre = float(np.mean(tau[take]))
        bins.append(dict(
            tau_s=centre,
            n=int(take.sum()),
            stop_error_s=None if len(finite) == 0 else float(np.mean(finite)),
            stop_error_median_s=None if len(finite) == 0 else float(np.median(finite)),
            frac_later=float(np.mean(np.where(np.isfinite(e), e > 0.02, c))),
            frac_censored=float(np.mean(c)),
            speed_at_stop_mm_s=float(np.mean(speed_at_stop[take]) * 1000),
        ))
    return bins


def value_at(bins, tau, key):
    """Bin whose centre is closest to ``tau``."""
    if not bins:
        return None
    pick = min(bins, key=lambda row: abs(row['tau_s'] - tau))
    if abs(pick['tau_s'] - tau) > BIN_S:
        return None
    return pick[key]


def growth(bins):
    """How much the stop error grows up to 0.25 s, and from there to 1 s."""
    early = value_at(bins, 0.0, 'stop_error_median_s')
    quarter = value_at(bins, 0.25, 'stop_error_median_s')
    half = value_at(bins, 0.5, 'stop_error_median_s')
    one = value_at(bins, 1.0, 'stop_error_median_s')
    def gap(a, b):
        if a is None or b is None:
            return None
        return float(b - a)
    return dict(error_0s=early, error_0_25s=quarter, error_0_5s=half, error_1s=one,
                growth_0_to_0_25=gap(early, quarter), growth_0_25_to_1=gap(quarter, one))


def analyse(model, episodes, device):
    cfg = model.cfg
    stride, horizon, execute = cfg.stride, cfg.horizon, cfg.execute
    dt = 1.0 / cfg.rate_hz
    horizon_s = horizon * dt
    execute_s = execute * dt
    model = model.to(device).eval()

    leads = []
    tails = []
    phases = []
    # Residual against future time, for the chunk that is actually open at the stop.
    # Indexed by time relative to the stop, in policy steps, from -execute to +execute.
    span = execute + 1
    open_resid = []
    open_label = []
    open_pred = []
    open_t = []
    tau_all, err_all, cens_all, at_stop_all = [], [], [], []
    # Speed profiles for chunks issued at a few fixed leads, aligned to the stop.
    profiles = {key: [] for key in (0.25, 0.5, 1.0)}

    for ep in episodes:
        cmd, obs = ep['cmd'], ep['obs']
        stop = stop_index(cmd)
        if stop is None:
            continue
        idx = query_indices(stop, len(cmd), stride, horizon)
        if len(idx) == 0:
            continue
        pred = predict_chunks(model, obs, idx, device)
        fut = np.minimum(idx[:, None] + np.arange(horizon) * stride, len(cmd) - 1)
        v_hat = forward_speed(pred)
        v_true = forward_speed(cmd[fut])
        tau = (stop - idx) / GRID_HZ
        step = stop_step(v_hat)
        censored = step < 0
        t_hat = np.where(censored, np.nan, step * dt)
        err = t_hat - tau
        # Speed the chunk assigns to the true stop instant (expert speed there is ~0).
        k_stop = np.rint(tau / dt).astype(int)
        k_stop = np.clip(k_stop, 0, horizon - 1)
        at_stop = v_hat[np.arange(len(idx)), k_stop]
        keep = (tau >= -1e-9) & (tau <= horizon_s + 1e-9)
        tau_all.append(tau[keep])
        err_all.append(err[keep])
        cens_all.append(censored[keep])
        at_stop_all.append(at_stop[keep])

        q = open_query(stop, stride, execute)
        if q < 0 or q >= len(cmd):
            continue
        # The replan grid starts at episode index 0, so q need not be among the
        # lead-aligned queries above. Predict that one chunk on its own.
        open_pred_cmd = predict_chunks(model, obs, np.asarray([q]), device)[0]
        open_fut = np.minimum(q + np.arange(horizon) * stride, len(cmd) - 1)
        v_p_full = forward_speed(open_pred_cmd)
        v_y_full = forward_speed(cmd[open_fut])
        v_p = v_p_full[:execute]
        v_y = v_y_full[:execute]
        t_rel = (q + np.arange(execute) * stride - stop) / GRID_HZ
        lead = float(np.sum(v_p - v_y) * dt)
        after = t_rel >= -1e-12
        tail = float(np.sum((v_p - v_y)[after]) * dt)
        leads.append(lead)
        tails.append(tail)
        phases.append(float((stop - q) / (execute * stride)))
        open_resid.append(v_p - v_y)
        open_label.append(v_y)
        open_pred.append(v_p)
        open_t.append(t_rel)
        for key in profiles:
            nearest = int(np.argmin(np.abs(tau - key)))
            if abs(float(tau[nearest]) - key) > dt + 1e-9:
                continue
            r = int(nearest)
            t_prof = (idx[r] + np.arange(horizon) * stride - stop) / GRID_HZ
            profiles[key].append((t_prof, v_hat[r], v_true[r]))

    tau_all = np.concatenate(tau_all)
    err_all = np.concatenate(err_all)
    cens_all = np.concatenate(cens_all)
    at_stop_all = np.concatenate(at_stop_all)
    bins = summarise_queries(tau_all, err_all, cens_all, at_stop_all, horizon_s)
    lead_mm = np.asarray(leads) * 1000.0
    tail_mm = np.asarray(tails) * 1000.0
    # Open-chunk residual against time relative to the stop, 0.04 s bins.
    curve = []
    if open_t:
        tt = np.concatenate(open_t)
        rr = np.concatenate(open_resid)
        ll = np.concatenate(open_label)
        pp = np.concatenate(open_pred)
        for edge in np.arange(-execute_s, execute_s + 1e-9, dt):
            take = np.abs(tt - edge) < dt / 2 + 1e-9
            if not np.any(take):
                continue
            curve.append(dict(
                t_s=float(edge), n=int(take.sum()),
                residual_mm_s=float(np.mean(rr[take]) * 1000),
                label_mm_s=float(np.mean(ll[take]) * 1000),
                pred_mm_s=float(np.mean(pp[take]) * 1000),
            ))
    fixed = {}
    for key, rows in profiles.items():
        if not rows:
            continue
        acc_t, acc_p, acc_y = [], [], []
        for t_prof, v_p, v_y in rows:
            acc_t.append(t_prof)
            acc_p.append(v_p)
            acc_y.append(v_y)
        tt = np.concatenate(acc_t)
        pp = np.concatenate(acc_p)
        yy = np.concatenate(acc_y)
        series = []
        for edge in np.arange(-key, min(horizon_s - key, 1.0) + 1e-9, 0.04):
            take = np.abs(tt - edge) < 0.02 + 1e-9
            if take.sum() < 10:
                continue
            series.append(dict(t_s=float(edge),
                               pred_mm_s=float(np.mean(pp[take]) * 1000),
                               label_mm_s=float(np.mean(yy[take]) * 1000),
                               residual_mm_s=float(np.mean(pp[take] - yy[take]) * 1000)))
        fixed[f'{key:.2f}'] = series
    grown = growth(bins)
    quarter = grown['error_0_25s']
    one = grown['error_1s']
    # Flat after a quarter-second: the extra wait from 0.25 s to 1 s adds little
    # compared with the error already present at 0.25 s.
    if quarter is None or one is None:
        saturated = None
    else:
        saturated = abs(one - quarter) <= max(0.05, 0.25 * abs(quarter))
    return dict(
        rate_hz=cfg.rate_hz, predict_s=horizon_s, execute_s=execute_s,
        n_episodes=int(len(leads)),
        open_chunk=dict(
            phase_mean=float(np.mean(phases)) if phases else None,
            lead_mm=mean_ci(lead_mm),
            tail_after_stop_mm=mean_ci(tail_mm),
            curve=curve,
        ),
        by_lead=bins,
        growth=grown,
        error_flat_after_0_25s=saturated,
        profiles=fixed,
    )


def save_plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pooled = report['ceiling']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    xs = [row['tau_s'] for row in pooled['by_lead']]
    ys = [row['stop_error_median_s'] for row in pooled['by_lead']]
    axes[0].axhline(0.0, color='0.5', lw=0.8)
    axes[0].plot(xs, ys, color='C0')
    axes[0].set_xlabel('how far ahead the chunk was issued (s)')
    axes[0].set_ylabel('predicted stop − true stop (s)')
    axes[0].set_title('Stop error against lookahead')
    curve = pooled['open_chunk']['curve']
    axes[1].axhline(0.0, color='0.5', lw=0.8)
    axes[1].plot([row['t_s'] for row in curve], [row['label_mm_s'] for row in curve],
                 color='0.25', label='expert')
    axes[1].plot([row['t_s'] for row in curve], [row['pred_mm_s'] for row in curve],
                 color='C0', label='predicted')
    axes[1].set_xlabel('time relative to the stop (s)')
    axes[1].set_ylabel('forward speed (mm/s)')
    axes[1].set_title('Chunk that is open at the stop')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=120)
    plt.close(fig)


def main():
    _check_wheel_map()
    if not (RESULTS.parent / 'data' / 'heldout_expert_grid50.npz').exists() and not (
            Path(__file__).resolve().parent / 'data' / 'heldout_expert_grid50.npz').exists():
        raise SystemExit('held-out grid is missing; run brake_residual.py first')
    from stage2_log import finish_run, start_run
    run_id, record_path, record = start_run(
        'chunk-lookahead', CEILING, dict(bin_s=BIN_S, move_v=MOVE_V), sys.argv,
        dict(seeds=list(SEEDS), rate50=RATE50),
        note='full-chunk forward speed and predicted stop time on held-out expert states')
    t0 = time.time()
    episodes = load_episodes()
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from policy import load_policy
    per_seed = {}
    for seed in SEEDS:
        path = RESULTS / 'models' / f'{CEILING}_s{seed}.pt'
        print(f'ceiling seed {seed}', flush=True)
        model = load_policy(path)
        per_seed[f's{seed}'] = analyse(model, episodes, device)
        lead = per_seed[f's{seed}']['open_chunk']['lead_mm']
        print(f"  open-chunk lead {lead['mean']:+.1f} mm  "
              f"tail {per_seed[f's{seed}']['open_chunk']['tail_after_stop_mm']['mean']:+.1f} mm  "
              f"flat_after_0.25 {per_seed[f's{seed}']['error_flat_after_0_25s']}", flush=True)
        del model
    # Pool the three seeds by averaging their bin curves and their per-episode
    # summaries. The open-chunk curves are averaged pointwise on time.
    def col(key_path, cast=float):
        vals = []
        for seed in SEEDS:
            cur = per_seed[f's{seed}']
            for part in key_path:
                cur = cur[part]
            vals.append(cur)
        return vals
    # Rebuild the pooled lead from the stored means; the episode lists are not
    # kept, so the pooled interval is the interval across the three seed means.
    seed_leads = [per_seed[f's{s}']['open_chunk']['lead_mm']['mean'] for s in SEEDS]
    seed_tails = [per_seed[f's{s}']['open_chunk']['tail_after_stop_mm']['mean'] for s in SEEDS]
    # Average the by-lead bins on tau. Seeds share the same query times.
    by_tau = {}
    for seed in SEEDS:
        for row in per_seed[f's{seed}']['by_lead']:
            by_tau.setdefault(round(row['tau_s'], 5), []).append(row)
    pooled_bins = []
    for tau in sorted(by_tau):
        rows = by_tau[tau]
        def avg(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return None if not vals else float(np.mean(vals))
        pooled_bins.append(dict(
            tau_s=float(tau), n=int(np.mean([r['n'] for r in rows])),
            stop_error_s=avg('stop_error_s'), stop_error_median_s=avg('stop_error_median_s'),
            frac_later=avg('frac_later'), frac_censored=avg('frac_censored'),
            speed_at_stop_mm_s=avg('speed_at_stop_mm_s')))
    # Average open-chunk curves on time.
    by_t = {}
    for seed in SEEDS:
        for row in per_seed[f's{seed}']['open_chunk']['curve']:
            by_t.setdefault(round(row['t_s'], 5), []).append(row)
    pooled_curve = []
    for t_s in sorted(by_t):
        rows = by_t[t_s]
        pooled_curve.append(dict(
            t_s=float(t_s), n=int(np.mean([r['n'] for r in rows])),
            residual_mm_s=float(np.mean([r['residual_mm_s'] for r in rows])),
            label_mm_s=float(np.mean([r['label_mm_s'] for r in rows])),
            pred_mm_s=float(np.mean([r['pred_mm_s'] for r in rows]))))
    ceiling = dict(
        n_episodes=per_seed['s0']['n_episodes'],
        execute_s=per_seed['s0']['execute_s'], predict_s=per_seed['s0']['predict_s'],
        open_chunk=dict(
            phase_mean=float(np.mean([per_seed[f's{s}']['open_chunk']['phase_mean'] for s in SEEDS])),
            lead_mm=mean_ci(np.asarray(seed_leads)),
            tail_after_stop_mm=mean_ci(np.asarray(seed_tails)),
            curve=pooled_curve,
            per_seed_lead_mm=seed_leads,
            per_seed_tail_mm=seed_tails,
        ),
        by_lead=pooled_bins,
        growth=growth(pooled_bins),
    )
    ceiling['error_flat_after_0_25s'] = (
        None if ceiling['growth']['error_0_25s'] is None or ceiling['growth']['error_1s'] is None
        else abs(ceiling['growth']['error_1s'] - ceiling['growth']['error_0_25s'])
        <= max(0.05, 0.25 * abs(ceiling['growth']['error_0_25s'])))
    rate50 = None
    path50 = RESULTS / 'models' / f'{RATE50}_s0.pt'
    if path50.exists():
        print('50 Hz seed 0', flush=True)
        model = load_policy(path50)
        rate50 = analyse(model, episodes, device)
        lead = rate50['open_chunk']['lead_mm']
        print(f"  50 Hz open-chunk lead {lead['mean']:+.1f} mm  "
              f"flat_after_0.25 {rate50['error_flat_after_0_25s']}", flush=True)
        # Drop the bulky fixed profiles from the 50 Hz blob; the bins are enough.
        rate50.pop('profiles', None)
    for seed in SEEDS:
        per_seed[f's{seed}'].pop('profiles', None)
    report = dict(
        definition=dict(
            stop='first step after the last command faster than 0.02 m/s; censored if the chunk never gets there',
            lead='integral of predicted minus expert forward speed over the executed steps of the chunk that is open at the stop',
            tail='the same integral on executed steps at or after the true stop',
            flat='median stop error at 1 s minus the error at 0.25 s is within 0.05 s, or within a quarter of the 0.25 s error',
        ),
        ceiling=ceiling,
        per_seed={k: dict(open_chunk=v['open_chunk'], growth=v['growth'],
                          error_flat_after_0_25s=v['error_flat_after_0_25s'],
                          by_lead=v['by_lead']) for k, v in per_seed.items()},
        rate50_s0=None if rate50 is None else dict(
            open_chunk=rate50['open_chunk'], growth=rate50['growth'],
            error_flat_after_0_25s=rate50['error_flat_after_0_25s'], by_lead=rate50['by_lead']),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    try:
        save_plot(report)
    except Exception as exc:
        print(f'plot skipped: {exc}', flush=True)
    g = ceiling['growth']
    print(f"stop error at 0 / 0.25 / 0.5 / 1.0 s: "
          f"{g['error_0s']} / {g['error_0_25s']} / {g['error_0_5s']} / {g['error_1s']}", flush=True)
    print(f"flat after 0.25 s: {ceiling['error_flat_after_0_25s']}", flush=True)
    print(f"open-chunk lead seed means {seed_leads} mm", flush=True)
    finish_run(record_path, record, dict(
        miss_mm=ceiling['open_chunk']['lead_mm']['mean'],
        catch_rate=ceiling['open_chunk']['phase_mean']), [OUT, PLOT])
    print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
