"""Relative-frame features on the Stage 1 winner: two configs, three seeds.

rel_add keeps the 42-D observation and appends the ball in the base frame.
rel_only drops time and the world-frame ball, base and end-effector positions.

Reports miss on the 200 test seeds, the along-track and cross-track compensation
fractions against the expert at the closest approach, and the hand-off horizon
on the first 50 test seeds (largest lead time that still catches at least 90%).
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import Config, policy_frames, relative_ball

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
WINNER = dict(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5, demos=1000)
CONFIGS = (Config(**WINNER, features='rel_add'), Config(**WINNER, features='rel_only'))
SEEDS = (0, 1, 2)
TAUS = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0)
HORIZON_CATCH = 0.90
BASELINE_WEIGHTS = RESULTS / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'

_env = None
_model = None


def check_features():
    obs = np.zeros((2, 42), np.float32)
    obs[:, 14] = 1.0  # yaw 0, forward is +x
    obs[:, 1] = 1.0
    obs[:, 10] = 0.2
    obs[:, 4] = 0.5   # world ball velocity +x
    obs[:, 39] = 0.7  # ee x
    rel = relative_ball(obs)
    assert rel.shape == (2, 12), rel.shape
    assert abs(rel[0, 0] - 0.8) < 1e-5, rel[0, 0]
    assert abs(rel[0, 3] - 0.5) < 1e-5, rel[0, 3]
    assert abs(rel[0, 9] - 0.3) < 1e-5, rel[0, 9]
    for cfg, dim in ((CONFIGS[0], 54), (CONFIGS[1], 38)):
        out = policy_frames(obs, cfg)
        assert out.shape[-1] == dim == cfg.obs_dim, (cfg.features, out.shape, cfg.obs_dim)
    import torch
    got = policy_frames(torch.from_numpy(obs), CONFIGS[0])
    assert tuple(got.shape) == (2, 54)
    print(f'features ok  rel_add {CONFIGS[0].name}  rel_only {CONFIGS[1].name}', flush=True)


def train_all():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
            if ckpt.exists():
                print(f'reusing {ckpt.name}', flush=True)
            else:
                pending.append((cfg, seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            cfg, seed, ckpt = pending.pop(0)
            log_path = RESULTS / 'logs' / f'{cfg.name}_s{seed}.train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000',
                   '--data', str(ROOT / 'data' / 'grid50.npz')]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, cfg, seed, time.time()))
            print(f'training {cfg.features} seed {seed}', flush=True)
        time.sleep(2)
        still = []
        for proc, log, cfg, seed, t0 in active:
            if proc.poll() is None:
                still.append((proc, log, cfg, seed, t0))
                continue
            log.close()
            if proc.returncode:
                raise RuntimeError(f'train failed for {cfg.features} seed {seed}')
            print(f'trained {cfg.features} seed {seed} in {time.time() - t0:.0f}s', flush=True)
        active = still


def init_env():
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def init_model(weights):
    init_env()
    global _model
    from np_policy import NumpyPolicy
    _model = NumpyPolicy(weights)


def expert_episode(seed):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP, ExpertDriver
    env = _env
    driver = ExpertDriver()
    env.reset(int(seed))
    driver.reset(env)
    stride = 1
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    catch_time = None
    drive_start = drive_end = None
    base, bearing = [], []
    step = 0
    while True:
        t = step * env.dt
        base.append(env.base_pos[:2].copy())
        bearing.append(env.bearing)
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        driver.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            phase = 'done' if driver.base_state.done else driver.base_state.phase
            if phase == 'drive':
                if drive_start is None:
                    drive_start = env.base_pos[:2].copy()
                drive_end = env.base_pos[:2].copy()
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    chord = None
    if drive_start is not None and drive_end is not None and np.linalg.norm(drive_end - drive_start) > 0.05:
        chord = (drive_end - drive_start).astype(np.float32)
    return dict(seed=int(seed), base=np.asarray(base, np.float32), bearing=np.asarray(bearing, np.float32),
                chord=chord, catch_time=None if catch_time is None else float(catch_time))


def policy_episode(seed):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    from np_policy import PolicyDriver
    env = _env
    policy = PolicyDriver(_model)
    env.reset(int(seed))
    policy.reset(env)
    rate = _model.cfg.rate_hz
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
            env.min_dist = min(env.min_dist, dist)
            if best is None or dist < best['dist']:
                best = dict(dist=float(dist), step=step, base=env.base_pos[:2].copy(),
                            ee=env.ee.copy(), ball=env.ball(t).copy())
            if catch_time is None and dist < 0.01:
                catch_time = t
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            cmd, _ = policy.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        applied = np.concatenate((cmd[:4], env.arm_torque(q_ref, cmd[11:18])))
        env.step(applied[:4], applied[4:])
        step += 1
    miss = float(env.min_dist)
    return dict(seed=int(seed), success=catch_time is not None,
                miss_mm=None if not np.isfinite(miss) else miss * 1000, best=best)


def handoff_episode(seed, catch_time):
    from catch_env import ExpertDriver, rollout
    from diagnose import HandoffDriver
    from np_policy import PolicyDriver
    rate = _model.cfg.rate_hz
    rows = []
    for tau in TAUS:
        t_save = max(0.05, float(catch_time) - tau)
        driver = HandoffDriver(ExpertDriver(), PolicyDriver(_model), t_save)
        out = rollout(_env, driver, int(seed), rate, stop_on_catch=True)
        miss = out['min_dist']
        rows.append(dict(tau=tau, success=bool(out['success']),
                         miss_mm=None if miss is None or not np.isfinite(miss) else float(miss) * 1000))
    return dict(seed=int(seed), rows=rows)


def route_axes(chord, bearing):
    if chord is not None and np.linalg.norm(chord) > 0.05:
        along = np.asarray(chord, float)
        along = along / np.linalg.norm(along)
    else:
        along = np.array([np.cos(bearing), np.sin(bearing)])
    cross = np.array([-along[1], along[0]])
    return along, cross


def compensation(policy_rows, experts):
    along_b, along_m, cross_b, cross_m = [], [], [], []
    for row in policy_rows:
        best = row['best']
        if best is None:
            continue
        ex = experts[row['seed']]
        k = min(best['step'], len(ex['base']) - 1)
        along, cross = route_axes(ex['chord'], float(ex['bearing'][k]))
        delta = best['base'] - ex['base'][k]
        miss = best['ee'][:2] - best['ball'][:2]
        along_b.append(float(np.dot(delta, along) * 1000))
        cross_b.append(float(np.dot(delta, cross) * 1000))
        along_m.append(float(np.dot(miss, along) * 1000))
        cross_m.append(float(np.dot(miss, cross) * 1000))
    def fraction(base, miss):
        base, miss = np.asarray(base), np.asarray(miss)
        var = float(np.var(base))
        slope = float(np.cov(base, miss, ddof=0)[0, 1] / var) if var > 1 else None
        big = np.abs(base) > 20
        per = 1 - miss[big] / base[big] if big.any() else np.array([])
        return dict(regression=None if slope is None else 1 - slope,
                    median_when_offset=None if len(per) == 0 else float(np.median(per)),
                    n_offset=int(big.sum()),
                    base_abs_median=float(np.median(np.abs(base))),
                    miss_abs_median=float(np.median(np.abs(miss))))
    return dict(along=fraction(along_b, along_m), cross=fraction(cross_b, cross_m))


def horizon(by_tau):
    """Largest lead, in seconds, whose catch rate is still at least 90%."""
    good = [tau for tau in TAUS if by_tau[tau]['catch_rate'] >= HORIZON_CATCH]
    return None if not good else max(good)


def summarize_miss(rows):
    dists = [r['miss_mm'] for r in rows if r['miss_mm'] is not None]
    return dict(n=len(rows), catch_rate=float(np.mean([r['success'] for r in rows])),
                miss_mm=float(np.mean(dists)), median_miss_mm=float(np.median(dists)))


def pool_map(fn, jobs, initializer, initargs=(), label=''):
    out = []
    with ProcessPoolExecutor(16, initializer=initializer, initargs=initargs) as pool:
        futs = [pool.submit(fn, *job) for job in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            out.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(f'  {label} {n}/{len(futs)}', flush=True)
    return out


def main():
    check_features()
    t0 = time.time()
    train_all()
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test']
    miss_seeds = test[:200]
    hand_seeds = test[:50]
    print(f'expert reference on {len(miss_seeds)} test seeds', flush=True)
    expert_rows = pool_map(expert_episode, [(int(s),) for s in miss_seeds], init_env, label='expert')
    experts = {r['seed']: r for r in expert_rows}
    catch_time = {r['seed']: r['catch_time'] for r in expert_rows}

    from policy import export_numpy
    report = {}
    weight_paths = [('stage1', BASELINE_WEIGHTS)]
    for cfg in CONFIGS:
        for seed in SEEDS:
            ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
            weights = ckpt.with_suffix('.npz')
            export_numpy(ckpt, weights)
            weight_paths.append((f'{cfg.features}_s{seed}', weights))

    for name, weights in weight_paths:
        print(f'rollout {name}', flush=True)
        rows = pool_map(policy_episode, [(int(s),) for s in miss_seeds], init_model, (str(weights),), name)
        stats = summarize_miss(rows)
        stats['compensation'] = compensation(rows, experts)
        if name != 'stage1':
            print(f'handoff {name}', flush=True)
            jobs = [(int(s), catch_time[int(s)]) for s in hand_seeds if catch_time[int(s)] is not None]
            hands = pool_map(handoff_episode, jobs, init_model, (str(weights),), name + ' handoff')
            by_tau = {}
            for tau in TAUS:
                got = [h for ep in hands for h in ep['rows'] if h['tau'] == tau and h['miss_mm'] is not None]
                by_tau[tau] = dict(catch_rate=float(np.mean([h['success'] for h in got])),
                                   miss_mm=float(np.mean([h['miss_mm'] for h in got])),
                                   n=len(got))
            stats['handoff'] = {str(k): v for k, v in by_tau.items()}
            stats['handoff_horizon_s'] = horizon(by_tau)
        report[name] = stats
        comp = stats['compensation']
        print(f"{name}: catch {stats['catch_rate']:.3f}  miss {stats['miss_mm']:.1f} mm  "
              f"comp along {comp['along']['regression']}  cross {comp['cross']['regression']}", flush=True)

    # Aggregate the three seeds of each new config.
    for cfg in CONFIGS:
        keys = [f'{cfg.features}_s{s}' for s in SEEDS]
        block = [report[k] for k in keys]
        def col(path):
            vals = []
            for item in block:
                cur = item
                for part in path:
                    cur = cur[part]
                if cur is not None:
                    vals.append(cur)
            return vals
        report[cfg.features] = dict(
            miss_mm_mean=float(np.mean(col(['miss_mm']))),
            miss_mm_sd=float(np.std(col(['miss_mm']))),
            median_miss_mm_mean=float(np.mean(col(['median_miss_mm']))),
            catch_mean=float(np.mean(col(['catch_rate']))),
            comp_along=float(np.mean(col(['compensation', 'along', 'regression']))),
            comp_along_sd=float(np.std(col(['compensation', 'along', 'regression']))),
            comp_cross=float(np.mean(col(['compensation', 'cross', 'regression']))),
            comp_cross_sd=float(np.std(col(['compensation', 'cross', 'regression']))),
            horizon_s=[report[k]['handoff_horizon_s'] for k in keys],
        )
    out = RESULTS / 'relative_features.json'
    # Drop the bulky per-episode best poses before writing; the summaries are enough.
    slim = {}
    for key, item in report.items():
        if 'compensation' in item and isinstance(item.get('catch_rate'), float):
            slim[key] = {k: v for k, v in item.items()}
        else:
            slim[key] = item
    out.write_text(json.dumps(slim, indent=2))
    print(json.dumps({k: report[k] for k in ('rel_add', 'rel_only')}, indent=2), flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
