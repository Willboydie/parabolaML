"""Action-error, one-chunk replay and expert-to-policy hand-off on the Stage 2 winner.

Part 1 uses logged 1 kHz expert trajectories (training demos on disk; clean test
rollouts generated here). Parts 2–3 use the simulator. No retraining.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from catch_env import FIRST_TICK_STEP, OBS_NAMES
from config import GRID_HZ
from np_policy import NumpyPolicy, PolicyDriver

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
WEIGHTS = RESULTS / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'
Q_COL = OBS_NAMES.index('arm_q_0')
EE_COL = OBS_NAMES.index('ee_x')
TAUS = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0)
BEFORE_BINS = ((0.0, 1.0, '0–1 s'), (1.0, 2.0, '1–2 s'), (2.0, 4.0, '2–4 s'),
               (4.0, 8.0, '4–8 s'), (8.0, 1e9, '>8 s'))
TURN_THRESH = 2.0  # rad/s of left-right wheel difference: in-place spin vs drive

_env = None
_model = None


def init_worker(weights):
    global _env, _model
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()
    _model = NumpyPolicy(weights)


class HandoffDriver:
    """Expert until switch_time, then the policy. Policy history is the expert's ticks."""

    def __init__(self, expert, policy, switch_time):
        self.expert, self.policy, self.switch_time = expert, policy, float(switch_time)

    def reset(self, env):
        self.expert.reset(env)
        self.policy.reset(env)

    def sense(self, env, t):
        self.expert.sense(env, t)

    def act(self, env, t, dt_ctrl):
        if t + 1e-12 >= self.switch_time:
            return self.policy.act(env, t, dt_ctrl)
        self.policy.frames.append(env.obs(t))
        if len(self.policy.frames) > 64:
            del self.policy.frames[:-self.policy.cfg.frames]
        return self.expert.act(env, t, dt_ctrl)


def grid_from_1khz(obs, cmd):
    return obs[FIRST_TICK_STEP::20], cmd[FIRST_TICK_STEP::20]


def chunk_errors(model, obs, cmd, catch_time):
    """Policy-tick chunk errors on one 50 Hz episode. Matches training label construction.

    Training demos stop ~0.1 s after the catch, so chunk labels past the end are the last
    command. Test rollouts that run on to the ball's endpoint are truncated the same way.
    """
    if catch_time is not None and np.isfinite(catch_time):
        keep = obs[:, 0] <= catch_time + 0.1
        if keep.any():
            obs, cmd = obs[keep], cmd[keep]
    cfg = model.cfg
    s, H, F = cfg.stride, cfg.horizon, cfg.frames
    ticks = np.arange(0, len(obs), s)
    if len(ticks) == 0:
        return None
    hist = np.maximum(ticks[:, None] - np.arange(F - 1, -1, -1) * s, 0)
    chunk = np.minimum(ticks[:, None] + np.arange(H) * s, len(obs) - 1)
    frames = obs[hist]
    pred = model.predict_batch(frames)
    y = cmd[chunk].copy()
    y[:, :, 4:11] -= obs[ticks, Q_COL:Q_COL + 7][:, None]
    y_std = (y - model.act_mean) / model.act_std
    p_std = (pred - model.act_mean) / model.act_std
    se = (p_std - y_std) ** 2
    t = obs[ticks, 0]
    before = (catch_time - t) if catch_time is not None and np.isfinite(catch_time) else np.full(len(ticks), np.nan)
    turn = np.abs(cmd[ticks, 0] - cmd[ticks, 1])
    return dict(mse=se.mean(), mse_h=se.mean(axis=(0, 2)),
                wheel=np.mean(np.abs(pred[:, :, :4] - y[:, :, :4]), axis=(0, 2)),
                joint=np.mean(np.abs(pred[:, :, 4:11] - y[:, :, 4:11]), axis=(0, 2)),
                wheel_all=np.abs(pred[:, :, :4] - y[:, :, :4]).mean(axis=2),
                joint_all=np.abs(pred[:, :, 4:11] - y[:, :, 4:11]).mean(axis=2),
                se_all=se.mean(axis=2), before=before, turn=turn, t=t, n=len(ticks))


def reduce_errors(rows):
    """Average Part 1 rows, including per-horizon and time-before-catch splits."""
    mse = np.mean([r['mse'] for r in rows])
    mse_h = np.mean([r['mse_h'] for r in rows], axis=0)
    wheel_h = np.mean([r['wheel'] for r in rows], axis=0)
    joint_h = np.mean([r['joint'] for r in rows], axis=0)
    bins = []
    for lo, hi, name in BEFORE_BINS:
        se, wh, jn, n = [], [], [], 0
        for r in rows:
            m = np.isfinite(r['before']) & (r['before'] >= lo) & (r['before'] < hi)
            if m.any():
                se.append(r['se_all'][m].mean())
                wh.append(r['wheel_all'][m].mean())
                jn.append(r['joint_all'][m].mean())
                n += int(m.sum())
        bins.append(dict(name=name, n=n,
                         mse=float(np.mean(se)) if se else None,
                         wheel=float(np.mean(wh)) if wh else None,
                         joint=float(np.mean(jn)) if jn else None))
    turning = [r['se_all'][r['turn'] >= TURN_THRESH].mean() for r in rows if (r['turn'] >= TURN_THRESH).any()]
    driving = [r['se_all'][r['turn'] < TURN_THRESH].mean() for r in rows if (r['turn'] < TURN_THRESH).any()]
    return dict(n_episodes=len(rows), n_ticks=int(sum(r['n'] for r in rows)), mse=float(mse),
                mse_horizon=mse_h.astype(float).tolist(),
                wheel_horizon=wheel_h.astype(float).tolist(),
                joint_horizon=joint_h.astype(float).tolist(),
                wheel_mean=float(wheel_h.mean()), joint_mean=float(joint_h.mean()),
                before=bins,
                mse_turn=float(np.mean(turning)) if turning else None,
                mse_drive=float(np.mean(driving)) if driving else None)


def part1_train(model, n_demos=300):
    rows = []
    for i in range(n_demos):
        path = DATA / 'demos' / f'demo_{i:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            obs, cmd = grid_from_1khz(z['obs'], z['cmd'])
        err = chunk_errors(model, obs, cmd, meta.get('catch_time'))
        if err:
            err['dart'] = bool(meta.get('dart'))
            rows.append(err)
    out = {'all': reduce_errors(rows)}
    clean = [r for r in rows if not r['dart']]
    noisy = [r for r in rows if r['dart']]
    if clean:
        out['clean'] = reduce_errors(clean)
    if noisy:
        out['dart'] = reduce_errors(noisy)
    return out


def part1_test_job(seed):
    from catch_env import ExpertDriver, rollout
    out = rollout(_env, ExpertDriver(), seed, 1000, stop_on_catch=False, log=True)
    obs, cmd = grid_from_1khz(out['arrays']['obs'], out['arrays']['cmd'])
    err = chunk_errors(_model, obs, cmd, out['catch_time'])
    return err, dict(seed=int(seed), success=out['success'], catch_time=out['catch_time'],
                     min_dist=out['min_dist'], duration=out['duration'],
                     ee=out['arrays']['obs'][:, EE_COL:EE_COL + 3].copy())


def part23_job(seed, catch_time, expert_ee, taus):
    from catch_env import ExpertDriver, rollout
    cfg = _model.cfg
    rate = cfg.rate_hz
    chunks, handoffs = [], []
    for tau in taus:
        t_save = catch_time - tau
        if t_save < 0.05:
            t_save = 0.05
        # One-chunk: expert to t_save, policy for 1 s.
        driver = HandoffDriver(ExpertDriver(), PolicyDriver(_model), t_save)
        one = rollout(_env, driver, seed, rate, stop_on_catch=False, until=t_save + 1.0, log=True)
        t_end = one['arrays']['obs'][-1, 0]
        step = min(int(round(t_end / 0.001)), len(expert_ee) - 1)
        dev = float(np.linalg.norm(one['arrays']['obs'][-1, EE_COL:EE_COL + 3] - expert_ee[step]))
        chunks.append(dict(tau=tau, t_save=t_save, t_end=float(t_end), dev_mm=dev * 1000,
                           success=one['success'], min_dist=one['min_dist']))
        # Hand-off: expert to t_save, policy for the rest of the episode.
        driver = HandoffDriver(ExpertDriver(), PolicyDriver(_model), t_save)
        rest = rollout(_env, driver, seed, rate, stop_on_catch=True)
        handoffs.append(dict(tau=tau, t_save=t_save, success=rest['success'],
                             min_dist=rest['min_dist'], catch_time=rest['catch_time']))
    return dict(seed=int(seed), chunks=chunks, handoffs=handoffs)


def summarize_part2(rows):
    by_tau = defaultdict(list)
    for r in rows:
        for c in r['chunks']:
            by_tau[c['tau']].append(c['dev_mm'])
    return [dict(tau=tau, n=len(by_tau[tau]), mean_mm=float(np.mean(by_tau[tau])),
                 median_mm=float(np.median(by_tau[tau])),
                 p90_mm=float(np.percentile(by_tau[tau], 90)))
            for tau in TAUS if tau in by_tau]


def summarize_part3(rows):
    by_tau = defaultdict(list)
    catch = defaultdict(list)
    for r in rows:
        for h in r['handoffs']:
            by_tau[h['tau']].append(h['min_dist'] * 1000)
            catch[h['tau']].append(float(h['success']))
    return [dict(tau=tau, n=len(by_tau[tau]), mean_miss_mm=float(np.mean(by_tau[tau])),
                 median_miss_mm=float(np.median(by_tau[tau])),
                 catch_rate=float(np.mean(catch[tau])))
            for tau in TAUS if tau in by_tau]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    parser.add_argument('--train-demos', type=int, default=300)
    parser.add_argument('--test', type=int, default=200)
    parser.add_argument('--handoff-seeds', type=int, default=50)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    seeds = json.loads((DATA / 'seeds.json').read_text())
    test_seeds = seeds['test'][:args.test]
    handoff_seeds = seeds['test'][:args.handoff_seeds]
    model = NumpyPolicy(args.weights)
    print(f'policy {args.weights.name}  {model.cfg.name}  H={model.cfg.horizon} exec={model.cfg.execute} '
          f'frames={model.cfg.frames}', flush=True)

    t0 = time.time()
    print('Part 1 train…', flush=True)
    train = part1_train(model, args.train_demos)
    print(f"  train MSE {train['all']['mse']:.4f}  wheel {train['all']['wheel_mean']:.3f} rad/s  "
          f"joint {train['all']['joint_mean']:.4f} rad  ({time.time() - t0:.0f}s)", flush=True)

    print('Part 1 test + expert traces…', flush=True)
    expert_ee = {}
    expert_meta = {}
    test_rows = []
    t1 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(str(args.weights),)) as pool:
        futs = {pool.submit(part1_test_job, seed): seed for seed in test_seeds}
        for n, fut in enumerate(as_completed(futs), 1):
            seed = futs[fut]
            err, meta = fut.result()
            test_rows.append(err)
            expert_ee[seed] = meta.pop('ee')
            expert_meta[seed] = meta
            if n % 50 == 0:
                print(f'  {n}/{len(test_seeds)}', flush=True)
    test = reduce_errors(test_rows)
    print(f"  test MSE {test['mse']:.4f}  wheel {test['wheel_mean']:.3f} rad/s  "
          f"joint {test['joint_mean']:.4f} rad  ({time.time() - t1:.0f}s)", flush=True)

    print('Parts 2–3…', flush=True)
    t2 = time.time()
    p23 = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(str(args.weights),)) as pool:
        futs = []
        for seed in handoff_seeds:
            ct = expert_meta[seed]['catch_time']
            if ct is None:
                ct = expert_meta[seed]['duration']
            futs.append(pool.submit(part23_job, seed, ct, expert_ee[seed], TAUS))
        for n, fut in enumerate(as_completed(futs), 1):
            p23.append(fut.result())
            if n % 10 == 0:
                print(f'  {n}/{len(handoff_seeds)}', flush=True)
    part2 = summarize_part2(p23)
    part3 = summarize_part3(p23)
    print(f'  parts 2–3 done ({time.time() - t2:.0f}s)', flush=True)
    for row in part2:
        print(f"  chunk τ={row['tau']:>4}s  EE dev {row['mean_mm']:.1f} mm (median {row['median_mm']:.1f})", flush=True)
    for row in part3:
        print(f"  handoff τ={row['tau']:>4}s  miss {row['mean_miss_mm']:.1f} mm  catch {row['catch_rate']:.3f}",
              flush=True)

    report = dict(weights=str(args.weights), config=model.cfg.to_dict(),
                  train=train, test=test, part2=part2, part3=part3,
                  n_train_demos=args.train_demos, n_test=args.test, n_handoff=args.handoff_seeds,
                  elapsed_s=time.time() - t0)
    out = RESULTS / 'diagnose.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'wrote {out} in {report["elapsed_s"]:.0f}s')


if __name__ == '__main__':
    main()
