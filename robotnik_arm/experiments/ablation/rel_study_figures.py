"""Redraw the five diagnostic figures on the flat rel_add 25 Hz winner.

The measurements match the original scripts. The weights are the three seeds of
r25_p2_e0.5_hnone_d1000_rel_add. Nothing here loads a DeepSets checkpoint.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'results' / 'rel_study'
OUT = STUDY / 'plots'
NAME = 'r25_p2_e0.5_hnone_d1000_rel_add'
WEIGHTS = [STUDY / 'models' / 'sweep' / f'{NAME}_s{seed}.npz' for seed in (0, 1, 2)]
PACK = OUT / 'figure_data.npz'
WHEEL_R = 0.11
MOVE_V = 0.02

from track_paths import P_GRID, SAMPLE_S, T_GRID, _path_offset, _same_time

_env = None
_models = None


def forward_speed(cmd):
    wheels = np.asarray(cmd, dtype=np.float64)[:4]
    return float(np.mean(wheels) * WHEEL_R)


def wrap(angle):
    return (float(angle) + np.pi) % (2 * np.pi) - np.pi


def init_worker(paths):
    global _env, _models
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _models = []
    for path in paths:
        model = NumpyPolicy(path)
        if model.cfg.encoder != 'flat' or model.cfg.features != 'rel_add':
            raise RuntimeError(f'{path} is {model.cfg.encoder} / {model.cfg.features}')
        if model.cfg.rate_hz != 25:
            raise RuntimeError(f'{path} is {model.cfg.rate_hz} Hz')
        _models.append(model)


def _stop_time(ts, speed):
    moving = np.flatnonzero(np.asarray(speed) > MOVE_V)
    if len(moving) == 0:
        return None
    last = int(moving[-1])
    tail = np.asarray(speed)[last + 1:last + 11]
    if last + 1 < len(ts) and len(tail) >= 5 and float(np.median(tail)) <= 0.01:
        return float(ts[last + 1])
    return None


def expert_roll(seed):
    """1 ms expert. Base is stored every step so parking uses the same index as expert_episode."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    dt = float(env.dt)
    sample_every = max(1, int(round(SAMPLE_S / dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    base, bearing = [], []
    xy, ts, speed = [], [], []
    drive_start = drive_end = None
    catch_time = None
    step = 0
    while True:
        t = step * dt
        base.append(env.base_pos[:2].copy())
        bearing.append(env.bearing)
        if step % sample_every == 0:
            xy.append(env.base_pos[:2].copy())
            ts.append(t)
            speed.append(forward_speed(cmd))
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % 1 == 0:
            phase = 'done' if driver.base_state.done else driver.base_state.phase
            if phase == 'drive':
                pos = env.base_pos[:2].copy()
                if drive_start is None:
                    drive_start = pos
                drive_end = pos
            fresh, _ready = driver.act(env, t, dt)
            cmd = np.asarray(fresh, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None and np.linalg.norm(drive_end - drive_start) > 0.05:
        chord = (np.asarray(drive_end) - np.asarray(drive_start)).astype(np.float64)
        chord = chord / np.linalg.norm(chord)
    return dict(
        base=np.asarray(base, np.float64), bearing=np.asarray(bearing, np.float64), chord=chord,
        xy=np.asarray(xy, np.float64), t=np.asarray(ts, np.float64), stop_t=_stop_time(ts, speed),
    )


def policy_roll(model, seed):
    """Closed loop at 25 Hz. The expert is queried in this state and is not executed."""
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    from measure_learner_noise import ExpertProbe
    from np_policy import PolicyDriver
    env = _env
    policy = PolicyDriver(model)
    expert = ExpertProbe()
    env.reset(int(seed))
    policy.reset(env)
    expert.reset(env)
    dt = float(env.dt)
    stride = int(round(1.0 / (model.cfg.rate_hz * dt)))
    sample_every = max(1, int(round(SAMPLE_S / dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    label = cmd.copy()
    xy, ts = [], []
    tick_t, tick_learner, tick_expert = [], [], []
    catch_time = None
    best = None
    step = 0
    while True:
        t = step * dt
        if step % sample_every == 0:
            xy.append(env.base_pos[:2].copy())
            ts.append(t)
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if best is None or dist < best['dist']:
                best = dict(dist=float(dist), step=step, base=env.base_pos[:2].copy(),
                            bearing=float(env.bearing))
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        expert.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            executed, _ = policy.act(env, t, stride * dt)
            executed = np.asarray(executed, dtype=float)
            probed, _ready = expert.act(env, t, stride * dt)
            probed = np.asarray(probed, dtype=float)
            cmd = executed
            label = probed
            tick_t.append(t)
            tick_learner.append(forward_speed(executed))
            tick_expert.append(forward_speed(probed))
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    miss = float(env.min_dist)
    return dict(
        xy=np.asarray(xy, np.float64), t=np.asarray(ts, np.float64), best=best,
        tick_t=np.asarray(tick_t, np.float64),
        learner_v=np.asarray(tick_learner, np.float64),
        expert_v=np.asarray(tick_expert, np.float64),
        miss_mm=None if not np.isfinite(miss) else miss * 1000.0,
    )


def _parking(expert, policy, model_seed, scene):
    from run_relative import route_axes
    best = policy['best']
    if best is None or policy['miss_mm'] is None:
        return None
    k = min(best['step'], len(expert['base']) - 1)
    along_hat, cross_hat = route_axes(expert['chord'], float(expert['bearing'][k]))
    delta = np.asarray(best['base'], float) - np.asarray(expert['base'][k], float)
    along = float(np.dot(delta, along_hat) * 1000)
    cross = float(np.dot(delta, cross_hat) * 1000)
    heading = wrap(best['bearing'] - float(expert['bearing'][k])) * 180.0 / np.pi
    return dict(scene=int(scene), model_seed=int(model_seed), along_mm=along, cross_mm=cross,
                radial_mm=float(np.hypot(along, cross)), heading_deg=float(heading),
                miss_mm=float(policy['miss_mm']))


def scene_job(seed):
    expert = expert_roll(int(seed))
    parking, brake, along, cross, path_cross, path_dist = [], [], [], [], [], []
    for i, model in enumerate(_models):
        policy = policy_roll(model, int(seed))
        parking.append(_parking(expert, policy, i, seed))
        brake.append(dict(t=policy['tick_t'], learner_v=policy['learner_v'], expert_v=policy['expert_v']))
        a, c = _same_time(expert, policy, expert['chord'], expert['stop_t'])
        pc, pd = _path_offset(expert, policy, expert['chord'], expert['stop_t'])
        along.append(a)
        cross.append(c)
        path_cross.append(pc)
        path_dist.append(pd)
    return dict(scene=int(seed), parking=parking, brake=brake,
                along=np.stack(along), cross=np.stack(cross),
                path_cross=np.stack(path_cross), path_dist=np.stack(path_dist))


def collect(seeds, workers):
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=( [str(p) for p in WEIGHTS],)) as pool:
        futs = [pool.submit(scene_job, int(s)) for s in seeds]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0 or n == len(futs):
                print(f'  scenes {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    order = {int(s): i for i, s in enumerate(seeds)}
    rows.sort(key=lambda r: order[r['scene']])
    park = [item for row in rows for item in row['parking'] if item is not None]
    packed = dict(
        scene=np.asarray([p['scene'] for p in park], np.int64),
        model_seed=np.asarray([p['model_seed'] for p in park], np.int64),
        along_mm=np.asarray([p['along_mm'] for p in park], np.float64),
        cross_mm=np.asarray([p['cross_mm'] for p in park], np.float64),
        radial_mm=np.asarray([p['radial_mm'] for p in park], np.float64),
        heading_deg=np.asarray([p['heading_deg'] for p in park], np.float64),
        miss_mm=np.asarray([p['miss_mm'] for p in park], np.float64),
        path_along_mm=np.stack([r['along'] for r in rows]),
        path_cross_time_mm=np.stack([r['cross'] for r in rows]),
        path_cross_mm=np.stack([r['path_cross'] for r in rows]),
        path_dist_mm=np.stack([r['path_dist'] for r in rows]),
        t_rel_s=T_GRID.astype(np.float64),
        progress=P_GRID.astype(np.float64),
    )
    curves_l, curves_e, lags = [], [], []
    label_neg = label_n = 0
    for row in rows:
        for episode in row['brake']:
            ev = episode['expert_v']
            lv = episode['learner_v']
            t = episode['t']
            label_neg += int(np.sum(ev < -1e-4))
            label_n += len(ev)
            moving = np.flatnonzero(ev > 0.02)
            if len(moving) == 0 or int(moving[-1]) >= len(t) - 1:
                lags.append(np.nan)
                continue
            stop = int(moving[-1])
            t_rel = t - t[stop]
            m = (t_rel >= -2.0) & (t_rel < 6.0)
            curves_l.extend(zip(t_rel[m].tolist(), lv[m].tolist()))
            curves_e.extend(zip(t_rel[m].tolist(), ev[m].tolist()))
            slow = np.flatnonzero((t_rel >= 0.0) & (lv < 0.05))
            lags.append(float(t_rel[slow[0]]) if len(slow) else np.nan)
    edges = np.arange(-2.0, 6.0 + 1e-9, 0.2)
    def binned(samples):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            vals = [v for tv, v in samples if lo <= tv < hi]
            out.append(np.nan if not vals else float(np.median(vals)) * 1000.0)
        return np.asarray(out, np.float64)
    packed['brake_t_s'] = ((edges[:-1] + edges[1:]) / 2).astype(np.float64)
    packed['brake_learner_mm_s'] = binned(curves_l)
    packed['brake_expert_mm_s'] = binned(curves_e)
    packed['brake_lag_s'] = np.asarray(lags, np.float64)
    packed['brake_label_neg'] = np.int64(label_neg)
    packed['brake_label_n'] = np.int64(label_n)
    packed['brake_n_episodes'] = np.int64(len(lags))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(PACK, **packed)
    print(f'wrote {PACK} in {time.time() - t0:.0f}s', flush=True)
    return packed


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#d0d5dd',
        'grid.linewidth': 0.6,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.axisbelow': True,
    })


def _save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'wrote {OUT / name}.png', flush=True)


def plot_parking(z):
    import plot_r50_parking as ref
    ref.OUT = OUT
    ref.style()
    arrays = {key: np.array(z[key]) for key in
              ('along_mm', 'cross_mm', 'radial_mm', 'heading_deg', 'model_seed', 'scene')}
    ref.scatter(arrays)
    ref.ranked(arrays)
    _parking_bias(arrays)


def _parking_bias(z):
    import matplotlib.pyplot as plt
    _style()
    along, cross = np.asarray(z['along_mm']), np.asarray(z['cross_mm'])
    frame = 150.0
    med = float(np.median(along))
    ahead = float(np.mean(along > 0))
    cross_med = float(np.median(cross))
    inside = (np.abs(along) <= frame) & (np.abs(cross) <= frame)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    ax = axes[0]
    ax.scatter(cross[inside], along[inside], s=12, c='#7eb6d9', alpha=0.75, edgecolors='none')
    ax.axhline(0, color='0.75', lw=0.8)
    ax.axvline(0, color='0.75', lw=0.8)
    ax.axhline(med, color='#d62728', lw=1.1)
    ax.text(0.04, 0.96, f'along-track median {med:+.1f} mm', transform=ax.transAxes,
            color='#d62728', fontsize=9, va='top')
    ax.set_xlim(-frame, frame)
    ax.set_ylim(-frame, frame)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('cross-track error (mm)')
    ax.set_ylabel('along-track error (mm)')
    ax.set_title('Where the base parks')

    ax = axes[1]
    bins = np.arange(-frame, frame + 1, 10)
    ax.hist(cross[np.abs(cross) <= frame], bins=bins, color='#4d4d4d', alpha=0.55,
            label=f'cross-track, median {cross_med:+.1f} mm')
    ax.hist(along[np.abs(along) <= frame], bins=bins, histtype='step', color='#1f77b4', lw=1.4,
            label=f'along-track, median {med:+.1f} mm')
    ax.axvline(med, color='#d62728', lw=1.1)
    direction = 'forward' if med >= 0 else 'back'
    ax.set_title(f'Along-track is shifted {direction} ({ahead:.0%} ahead)')
    ax.set_xlabel('signed error (mm), positive = ahead of the expert')
    ax.set_ylabel('episodes')
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(-frame, frame)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    n_scatter = int((~inside).sum())
    n_along = int(np.sum(np.abs(along) > frame))
    n_cross = int(np.sum(np.abs(cross) > frame))
    fig.text(0.5, 0.02,
             f'25 Hz flat rel_add winner, {len(along)} episodes. Tails past the axes are omitted: '
             f'{n_scatter} on the left, {n_along} along-track and {n_cross} cross-track on the right.',
             ha='center', va='bottom', fontsize=8, color='0.35')
    _save(fig, 'parking_bias')


def plot_brake(z, dart_note):
    import matplotlib.pyplot as plt
    _style()
    t = np.asarray(z['brake_t_s'])
    lv = np.asarray(z['brake_learner_mm_s'])
    ev = np.asarray(z['brake_expert_mm_s'])
    keep = np.arange(0, len(t), 2)
    t, lv, ev = t[keep], lv[keep], ev[keep]
    lags = np.asarray(z['brake_lag_s'], float)
    ok = np.isfinite(lags)
    med = float(np.median(lags[ok]))
    p25, p75 = np.percentile(lags[ok], [25, 75])
    n_ep = int(z['brake_n_episodes'])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(0.0, color='0.75', lw=0.8)
    ax.plot(t, ev, color='#1f3a5f', lw=1.8, label='Expert command, queried in the learner state')
    ax.plot(t, lv, color='#c46b2c', lw=1.8, label='Learner command, closed loop')
    ax.set_xlabel('Time relative to the expert stop (s)')
    ax.set_ylabel('Forward command (mm/s)')
    ax.set_xlim(-2.0, 5.6)
    finite = np.concatenate([lv[np.isfinite(lv)], ev[np.isfinite(ev)]])
    hi = max(280.0, float(np.nanmax(finite)) * 1.05) if len(finite) else 280.0
    ax.set_ylim(-10, hi)
    if med >= 0:
        ax.set_title(f'Median brake is {med:.1f} s after the expert has already stopped')
    else:
        ax.set_title(f'Median brake is {abs(med):.1f} s before the expert stops')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    neg = int(z['brake_label_neg'])
    nlab = int(z['brake_label_n'])
    fig.text(0.5, 0.02,
             f'Median forward command, {int(ok.sum())} of {n_ep} rollouts of the 25 Hz winner. '
             f'Quartiles of the brake lag are {p25:.1f} s and {p75:.1f} s.\n'
             f'Expert forward labels in these states are negative on {neg} of {nlab} ticks. {dart_note}',
             ha='center', va='bottom', fontsize=8, color='0.35')
    _save(fig, '01_brake_lag')
    return dict(median_s=med, p25_s=float(p25), p75_s=float(p75), n_stopped=int(ok.sum()), n_episodes=n_ep)


def plot_path(z):
    import matplotlib.pyplot as plt
    _style()
    path_cross = np.asarray(z['path_cross_mm']).reshape(-1, P_GRID.size)
    path_dist = np.asarray(z['path_dist_mm']).reshape(-1, P_GRID.size)
    progress = np.asarray(z['progress'])
    with np.errstate(all='ignore'):
        med = np.nanmedian(path_cross, axis=0)
        lo = np.nanpercentile(path_cross, 25, axis=0)
        hi = np.nanpercentile(path_cross, 75, axis=0)
        n = np.sum(np.isfinite(path_cross), axis=0)
        d_med = np.nanmedian(path_dist, axis=0)
        d_n = np.sum(np.isfinite(path_dist), axis=0)
    show = n >= 0.8 * path_cross.shape[0]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(0.0, color='0.25', lw=1.0, zorder=2, label='Expert path')
    rng = np.random.default_rng(0)
    pick = rng.choice(path_cross.shape[0], size=min(80, path_cross.shape[0]), replace=False)
    labeled = False
    for row in path_cross[pick]:
        ok = np.isfinite(row)
        ax.plot(progress[ok], row[ok], color='C0', alpha=0.15, lw=0.7, zorder=1,
                label='Test trajectories' if not labeled else None)
        labeled = True
    ax.fill_between(progress[show], lo[show], hi[show], color='C0', alpha=0.18, linewidth=0, zorder=2)
    ax.plot(progress[show], med[show], color='C0', lw=1.8, zorder=3, label='Median offset')
    ax.set_xlabel('Progress along the expert route')
    ax.set_ylabel('Sideways offset from the expert path (mm)')
    ax.set_title('The learner stays on the expert path')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-45, 45)
    cruise = (progress >= 0.2) & (progress <= 0.8) & (d_n >= 0.5 * path_dist.shape[0])
    cruise_mm = float(np.nanmedian(d_med[cruise])) if np.any(cruise) else float('nan')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    n_ep = path_cross.shape[0]
    fig.text(0.5, 0.02,
             f'Median distance to the expert path, middle of the route: {cruise_mm:.0f} mm.  '
             f'80 of {n_ep} rollouts drawn. 25 Hz flat rel_add winner.',
             ha='center', va='bottom', fontsize=8, color='0.35')
    _save(fig, 'path_adherence')
    return dict(n_rollouts=int(n_ep), path_distance_midroute_mm=cruise_mm)


def dart_label_note():
    """Clean expert labels on the high-amplitude DART half of the new grid."""
    path = ROOT / 'data' / 'grid50_rel_dart.npz'
    if not path.exists():
        return 'The new DART grid is not on disk.'
    with np.load(path) as z:
        start = int(z['starts'][1000])
        cmd = np.asarray(z['cmd'][start:], dtype=np.float64)
    forward = cmd[:, :4].mean(axis=1) * WHEEL_R
    neg = int(np.sum(forward < -1e-4))
    return (f'On the {len(forward)} clean labels of the new high-amplitude DART demos, '
            f'{neg} are negative (minimum {float(forward.min()):.3f} m/s).')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with np.load(PACK) as z:
            packed = {key: np.array(z[key]) for key in z.files}
    else:
        for path in WEIGHTS:
            if not path.exists():
                raise SystemExit(f'missing {path}')
        test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:args.limit]
        print(f'{len(test)} test scenes, {NAME}, 3 seeds', flush=True)
        packed = collect(test, args.workers)
    note = dart_label_note()
    plot_parking(packed)
    brake = plot_brake(packed, note)
    path = plot_path(packed)
    summary = dict(model=NAME, weights=[str(p) for p in WEIGHTS], brake=brake, path=path,
                   n_parking=int(len(packed['along_mm'])),
                   along_median_mm=float(np.median(packed['along_mm'])),
                   cross_median_mm=float(np.median(packed['cross_mm'])),
                   frac_ahead=float(np.mean(packed['along_mm'] > 0)),
                   dart_note=note)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
