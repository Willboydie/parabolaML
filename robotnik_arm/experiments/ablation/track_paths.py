"""Where the ceiling leaves the expert, and how close it stays to the expert path.

The saved parking logs are the closest approach only. This rolls the logged
DeepSets ceiling on the 200 test scenes, next to the expert on the same seeds,
and keeps the base path. Two comparisons:

- same clock time, in the expert's route frame: the along-track gap shows
  where the learner pulls ahead
- nearest point on the expert path: the cross-track offset shows whether the
  learner is still on that path

The expert is the 1 kHz controller used as the reference elsewhere. The policy
runs at its own 25 Hz.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'stage2' / 'track_paths.npz'
PLOT_DIR = RESULTS / 'stage2'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
SEEDS = (0, 1, 2)
MOVE_V = 0.02
WHEEL_RADIUS = 0.11
SAMPLE_S = 0.04
T_GRID = np.round(np.arange(-8.0, 2.0 + 1e-9, 0.1), 5)
P_GRID = np.linspace(0.0, 1.0, 51)

_env = None
_models = None


def forward_speed(cmd):
    wheels = np.asarray(cmd, dtype=np.float64)[..., :4]
    right = 0.5 * (wheels[..., 0] + wheels[..., 3])
    left = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return float(0.5 * (right + left) * WHEEL_RADIUS)


def init_worker():
    global _env, _models
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _models = []
    for seed in SEEDS:
        path = RESULTS / 'models' / f'{CEILING}_s{seed}.npz'
        model = NumpyPolicy(path)
        if model.cfg.rate_hz != 25:
            raise RuntimeError(f'{path.name} is {model.cfg.rate_hz} Hz, expected 25')
        _models.append(model)


def _roll(driver, seed, stride, track_drive):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    env = _env
    env.reset(int(seed))
    driver.reset(env)
    dt = float(env.dt)
    sample_every = max(1, int(round(SAMPLE_S / dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    xy, ts, speed = [], [], []
    drive_start = drive_end = None
    catch_time = None
    step = 0
    while True:
        t = step * dt
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
        if hasattr(driver, 'sense'):
            driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            if track_drive:
                phase = 'done' if driver.base_state.done else driver.base_state.phase
                if phase == 'drive':
                    pos = env.base_pos[:2].copy()
                    if drive_start is None:
                        drive_start = pos
                    drive_end = pos
            fresh, _ready = driver.act(env, t, stride * dt)
            cmd = np.asarray(fresh, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None and np.linalg.norm(drive_end - drive_start) > 0.05:
        chord = (drive_end - drive_start).astype(np.float64)
        chord = chord / np.linalg.norm(chord)
    moving = np.flatnonzero(np.asarray(speed) > MOVE_V)
    stop_t = None
    if len(moving):
        last = int(moving[-1])
        tail = np.asarray(speed)[last + 1:last + 11]
        if last + 1 < len(ts) and len(tail) >= 5 and float(np.median(tail)) <= 0.01:
            stop_t = float(ts[last + 1])
    return dict(xy=np.asarray(xy, np.float64), t=np.asarray(ts, np.float64),
                chord=chord, stop_t=stop_t, drive_start=drive_start)


def _interp(t_src, xy, t_query):
    out = np.full((len(t_query), 2), np.nan)
    if len(t_src) < 2:
        return out
    ok = (t_query >= t_src[0]) & (t_query <= t_src[-1])
    for axis in (0, 1):
        out[ok, axis] = np.interp(t_query[ok], t_src, xy[:, axis])
    return out


def _path_offset(expert, learner, chord, stop_t):
    """Signed cross-track and distance from the learner to the expert polyline.

    Samples are the learner's positions while the expert is still driving.
    Progress is arc length along the expert path, 0 at the first sample and 1
    at the expert stop.
    """
    cross_out = np.full(len(P_GRID), np.nan)
    dist_out = np.full(len(P_GRID), np.nan)
    if chord is None or stop_t is None:
        return cross_out, dist_out
    et, exy = expert['t'], expert['xy']
    keep = et <= stop_t + 1e-9
    if int(keep.sum()) < 4:
        return cross_out, dist_out
    poly = exy[keep]
    seg = np.diff(poly, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    good = seglen > 1e-6
    if int(good.sum()) < 2:
        return cross_out, dist_out
    a = poly[:-1][good]
    ab = seg[good]
    seglen = seglen[good]
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total < 0.05:
        return cross_out, dist_out
    lt, lxy = learner['t'], learner['xy']
    use = (lt >= et[0]) & (lt <= stop_t)
    if int(use.sum()) < 2:
        return cross_out, dist_out
    pts = lxy[use]
    pa = pts[:, None, :] - a[None, :, :]
    frac = np.clip(np.sum(pa * ab, axis=2) / ab2, 0.0, 1.0)
    proj = a + frac[..., None] * ab
    dist = np.linalg.norm(pts[:, None, :] - proj, axis=2)
    k = np.argmin(dist, axis=1)
    idx = np.arange(len(pts))
    nearest = dist[idx, k]
    direction = ab[k] / seglen[k, None]
    side = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    signed = np.sum((pts - proj[idx, k]) * side, axis=1)
    progress = (cum[k] + frac[idx, k] * seglen[k]) / total
    order = np.argsort(progress)
    progress, signed, nearest = progress[order], signed[order], nearest[order]
    # One value per progress bin: the sample closest to the bin centre.
    centres = P_GRID
    half = 0.5 * (centres[1] - centres[0])
    for i, centre in enumerate(centres):
        sel = np.flatnonzero(np.abs(progress - centre) <= half + 1e-9)
        if len(sel) == 0:
            continue
        j = sel[np.argmin(np.abs(progress[sel] - centre))]
        cross_out[i] = signed[j] * 1000.0
        dist_out[i] = nearest[j] * 1000.0
    return cross_out, dist_out


def _same_time(expert, learner, chord, stop_t):
    along = np.full(len(T_GRID), np.nan)
    cross = np.full(len(T_GRID), np.nan)
    if chord is None or stop_t is None:
        return along, cross
    query = stop_t + T_GRID
    exy = _interp(expert['t'], expert['xy'], query)
    lxy = _interp(learner['t'], learner['xy'], query)
    delta = lxy - exy
    side = np.array([-chord[1], chord[0]])
    ok = np.isfinite(delta[:, 0])
    along[ok] = delta[ok] @ chord * 1000.0
    cross[ok] = delta[ok] @ side * 1000.0
    return along, cross


def scene_job(seed):
    from catch_env import ExpertDriver
    from np_policy import PolicyDriver
    expert = _roll(ExpertDriver(), int(seed), stride=1, track_drive=True)
    along = np.full((len(SEEDS), len(T_GRID)), np.nan)
    cross = np.full_like(along, np.nan)
    path_cross = np.full((len(SEEDS), len(P_GRID)), np.nan)
    path_dist = np.full_like(path_cross, np.nan)
    for i, model in enumerate(_models):
        learner = _roll(PolicyDriver(model), int(seed), stride=int(round(1.0 / (model.cfg.rate_hz * _env.dt))),
                        track_drive=False)
        along[i], cross[i] = _same_time(expert, learner, expert['chord'], expert['stop_t'])
        path_cross[i], path_dist[i] = _path_offset(expert, learner, expert['chord'], expert['stop_t'])
    return dict(scene=int(seed), stop_t=expert['stop_t'],
                along_mm=along, cross_mm=cross, path_cross_mm=path_cross, path_dist_mm=path_dist)


def collect(seeds, workers):
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        futs = [pool.submit(scene_job, int(s)) for s in seeds]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 10 == 0 or n == len(futs):
                print(f'  scenes {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    rows.sort(key=lambda r: seeds.index(r['scene']))
    packed = dict(
        scene=np.asarray([r['scene'] for r in rows], np.int64),
        model_seed=np.asarray(SEEDS, np.int64),
        t_rel_s=T_GRID.astype(np.float64),
        progress=P_GRID.astype(np.float64),
        along_mm=np.stack([r['along_mm'] for r in rows]),
        cross_mm=np.stack([r['cross_mm'] for r in rows]),
        path_cross_mm=np.stack([r['path_cross_mm'] for r in rows]),
        path_dist_mm=np.stack([r['path_dist_mm'] for r in rows]),
        stop_t=np.asarray([np.nan if r['stop_t'] is None else r['stop_t'] for r in rows]),
    )
    # along_mm is (scene, seed, time)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **packed)
    print(f'wrote {OUT} in {time.time() - t0:.0f}s', flush=True)
    return packed


def _band(values):
    """values: (episode, time). Median and the 25/75 percentiles, ignoring NaN."""
    with np.errstate(all='ignore'):
        med = np.nanmedian(values, axis=0)
        lo = np.nanpercentile(values, 25, axis=0)
        hi = np.nanpercentile(values, 75, axis=0)
        n = np.sum(np.isfinite(values), axis=0)
    return med, lo, hi, n


def plot(packed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    along = packed['along_mm'].reshape(-1, packed['along_mm'].shape[-1])
    cross = packed['cross_mm'].reshape(-1, packed['cross_mm'].shape[-1])
    path_cross = packed['path_cross_mm'].reshape(-1, packed['path_cross_mm'].shape[-1])
    path_dist = packed['path_dist_mm'].reshape(-1, packed['path_dist_mm'].shape[-1])
    t = packed['t_rel_s']
    progress = packed['progress']

    a_med, a_lo, a_hi, a_n = _band(along)
    c_med, _, _, c_n = _band(cross)
    keep = (a_n >= 0.8 * along.shape[0]) & (c_n >= 0.8 * cross.shape[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.axhline(0.0, color='0.55', lw=0.8, zorder=0)
    ax.axvline(0.0, color='0.35', lw=0.9, zorder=1)
    ax.fill_between(t[keep], a_lo[keep], a_hi[keep], color='C0', alpha=0.18, linewidth=0, label='Along-track, middle half')
    ax.plot(t[keep], a_med[keep], color='C0', lw=1.8, label='Along-track, median')
    ax.plot(t[keep], c_med[keep], color='0.35', lw=1.4, label='Cross-track, median')
    ax.annotate('expert stops', xy=(0.0, 22), xytext=(0.15, 22),
                fontsize=8, color='0.25', va='bottom')
    ax.set_xlabel('Time relative to the expert stop (s)')
    ax.set_ylabel('Learner base − expert base (mm)')
    ax.set_title('Where the learner diverges from the expert')
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlim(float(t[keep][0]), float(t[keep][-1]))
    ax.set_ylim(-18, 26)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    n_ep = along.shape[0]
    fig.text(0.5, 0.02,
             f'DeepSets ceiling, {n_ep} rollouts. Same clock time, in the expert route frame. Positive is ahead.',
             ha='center', va='bottom', fontsize=8, color='0.35')
    diverge = PLOT_DIR / 'diverge_along_track.png'
    fig.savefig(diverge, dpi=140)
    plt.close(fig)

    p_med, p_lo, p_hi, p_n = _band(path_cross)
    d_med, _, _, d_n = _band(path_dist)
    show = p_n >= 0.8 * path_cross.shape[0]
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
    ax.fill_between(progress[show], p_lo[show], p_hi[show], color='C0', alpha=0.18, linewidth=0, zorder=2)
    ax.plot(progress[show], p_med[show], color='C0', lw=1.8, zorder=3, label='Median offset')
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
    fig.text(0.5, 0.02,
             f'Median distance to the expert path, middle of the route: {cruise_mm:.0f} mm.  80 of {n_ep} rollouts drawn.',
             ha='center', va='bottom', fontsize=8, color='0.35')
    adhere = PLOT_DIR / 'path_adherence.png'
    fig.savefig(adhere, dpi=140)
    plt.close(fig)

    summary = dict(
        n_rollouts=int(n_ep),
        along_at_m2s_mm=float(a_med[np.argmin(np.abs(t + 2.0))]),
        along_at_stop_mm=float(a_med[np.argmin(np.abs(t))]),
        along_at_p1s_mm=float(a_med[np.argmin(np.abs(t - 1.0))]),
        cross_at_stop_mm=float(c_med[np.argmin(np.abs(t))]),
        path_distance_midroute_mm=cruise_mm,
    )
    (PLOT_DIR / 'track_paths_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'wrote {diverge}', flush=True)
    print(f'wrote {adhere}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        with np.load(OUT) as z:
            packed = {key: np.array(z[key]) for key in z.files}
    else:
        test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:args.limit]
        print(f'{len(test)} test scenes, ceiling seeds {SEEDS}', flush=True)
        packed = collect(test, args.workers)
    plot(packed)


if __name__ == '__main__':
    main()
