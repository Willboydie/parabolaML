"""First DAgger iteration: roll the Stage 1 winner out on 200 training seeds
and label every policy tick with the expert's immediate command in that state.

The learner's command is what gets executed. The logged command is the expert's,
so training clones the expert on the states the learner actually visits.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'dagger_iter1'
WEIGHTS = ROOT / 'results' / 'models' / 'r25_p2_e0.5_hstack0.5s_d1000_s1.npz'
N = 200

_env = None
_model = None


def init_worker(weights):
    global _env, _model
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _model = NumpyPolicy(weights)


def dagger_job(index, seed, path):
    from catch_env import CAP_AFTER_ENDPOINT, CATCH_LINGER, FIRST_TICK_STEP
    from measure_learner_noise import ExpertProbe
    from np_policy import PolicyDriver
    env = _env
    policy = PolicyDriver(_model)
    expert = ExpertProbe()
    env.reset(seed)
    policy.reset(env)
    expert.reset(env)
    rate = _model.cfg.rate_hz
    stride = int(round(1.0 / (rate * env.dt)))
    end_time = env.target.duration + CAP_AFTER_ENDPOINT
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    executed = cmd.copy()
    ready = False
    catch_time = None
    rows = dict(obs=[], cmd=[], executed=[], applied=[], ready=[])
    step = 0
    while True:
        t = step * env.dt
        expert.sense(env, t)
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            executed, _ = policy.act(env, t, stride * env.dt)
            executed = np.asarray(executed, dtype=float)
            cmd, ready = expert.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(executed[4:11])
        applied = np.concatenate((executed[:4], env.arm_torque(q_ref, executed[11:18])))
        dist = env.capture_distance(t)
        if dist is not None:
            env.min_dist = min(env.min_dist, dist)
            if catch_time is None and dist < 0.01:
                catch_time = t
        rows['obs'].append(env.obs(t))
        rows['cmd'].append(np.asarray(cmd, dtype=np.float32))
        rows['executed'].append(np.asarray(executed, dtype=np.float32))
        rows['applied'].append(applied.astype(np.float32))
        rows['ready'].append(ready)
        if (catch_time is not None and t >= catch_time + CATCH_LINGER) or t >= end_time:
            break
        env.step(applied[:4], applied[4:])
        step += 1
    miss = float(env.min_dist)
    if not np.isfinite(miss):
        miss = None
    meta = dict(seed=int(seed), index=index, dagger=True, success=catch_time is not None,
                min_dist=miss, catch_time=None if catch_time is None else float(catch_time),
                steps=step + 1, duration=float(env.target.duration), rate_hz=rate,
                sim_dt=float(env.dt), weights=str(WEIGHTS))
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, obs=np.asarray(rows['obs']), cmd=np.asarray(rows['cmd']),
             executed=np.asarray(rows['executed']), applied=np.asarray(rows['applied']),
             ready=np.asarray(rows['ready']), meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return meta


def build_grid(n):
    from catch_env import FIRST_TICK_STEP, OBS_NAMES
    from config import GRID_HZ
    stride = 1000 // GRID_HZ
    obs, cmd, demo_index, starts, lengths = [], [], [], [], []
    offset = 0
    for i in range(n):
        with np.load(OUT / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        obs.append(o)
        cmd.append(c)
        demo_index.append(np.full(len(o), i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
    grid = OUT / f'grid{GRID_HZ}.npz'
    np.savez(grid, obs=np.concatenate(obs), cmd=np.concatenate(cmd),
             demo=np.concatenate(demo_index), starts=np.array(starts), lengths=np.array(lengths),
             arm_q_col=np.int64(OBS_NAMES.index('arm_q_0')))
    return grid, offset, float(np.mean(lengths) / GRID_HZ)


def main():
    seeds = json.loads((DATA / 'seeds.json').read_text())['train'][:N]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'dataset.json').write_text(json.dumps(dict(n=N, seeds=seeds, weights=str(WEIGHTS))))
    pending = [(i, int(s), OUT / f'demo_{i:04d}.npz') for i, s in enumerate(seeds)
               if not (OUT / f'demo_{i:04d}.npz').exists()]
    print(f'{N} DAgger rollouts, {len(pending)} to collect', flush=True)
    t0 = time.time()
    if pending:
        with ProcessPoolExecutor(16, initializer=init_worker, initargs=(str(WEIGHTS),)) as pool:
            futs = [pool.submit(dagger_job, i, s, path) for i, s, path in pending]
            for k, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if k % 25 == 0 or k == len(pending):
                    print(f'  {k}/{len(pending)} ({time.time() - t0:.0f}s)', flush=True)
    metas = []
    for i in range(N):
        with np.load(OUT / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            metas.append(json.loads(str(z['meta'])))
    grid, n_rows, mean_len = build_grid(N)
    catch = float(np.mean([m['success'] for m in metas]))
    dists = [m['min_dist'] for m in metas if m['min_dist'] is not None]
    print(f'{N} rollouts, learner catch {catch:.3f}, '
          f'mean miss {np.mean(dists) * 1000:.1f} mm, mean length {mean_len:.1f}s, '
          f'{n_rows} grid rows -> {grid}', flush=True)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
