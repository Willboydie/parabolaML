"""1,000 one-ball DART demos using the measured learner command noise.

The expert is queried at 25 Hz (the policy rate). An independent OU process is
added to each of the 18 command dimensions, with the per-dimension σ and τ_c
from results/learner_noise.json. The 1 ms loop holds the noisy command. Logged
``cmd`` is the clean expert command; ``executed`` is what the low level tracked.

Episode seeds are a new stream, disjoint from the Stage 1 train and test seeds.
Existing demos in data/demos/ are not touched. Re-running skips finished files.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'learner_dart'
NOISE_PATH = ROOT / 'results' / 'learner_noise.json'
MASTER_SEED = 2026
EPISODE_STREAM = 30
NOISE_STREAM = 31
RATE_HZ = 25

_env = None
_sigma = None
_tau = None


def init_worker(sigma, tau):
    global _env, _sigma, _tau
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()
    _sigma = np.asarray(sigma, dtype=np.float64)
    _tau = np.asarray(tau, dtype=np.float64)


def seed_stream(stream, count):
    return [int(np.random.SeedSequence([MASTER_SEED, stream, i]).generate_state(1, dtype=np.uint64)[0] >> 1)
            for i in range(count)]


def demo_job(index, seed, noise_seed, path):
    from catch_env import ExpertDriver, LearnerCommandNoise, rollout
    noise = LearnerCommandNoise(noise_seed, _sigma, _tau, 1.0 / RATE_HZ)
    out = rollout(_env, ExpertDriver(), seed, RATE_HZ, command_noise=noise, log=True)
    arrays = out.pop('arrays')
    if not np.isfinite(out['min_dist']):
        out['min_dist'] = None
    meta = dict(out, index=index, dart=True, noise='ou_learner', noise_seed=int(noise_seed),
                rate_hz=RATE_HZ, sim_dt=float(_env.dt))
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, obs=arrays['obs'], cmd=arrays['cmd'], executed=arrays['executed'],
             applied=arrays['applied'], ready=arrays['ready'], meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return meta


def build_grid(n):
    """50 Hz anchor grid, same layout as build_dataset.py, over this demo directory."""
    from catch_env import FIRST_TICK_STEP, OBS_NAMES
    from config import GRID_HZ
    stride = 1000 // GRID_HZ
    obs, cmd, demo_index, starts, lengths = [], [], [], [], []
    offset = 0
    for i in range(n):
        path = OUT / f'demo_{i:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta['index'] != i:
                raise RuntimeError(f'{path.name} has index {meta["index"]}')
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        obs.append(o)
        cmd.append(c)
        demo_index.append(np.full(len(o), i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
    grid_path = OUT / f'grid{GRID_HZ}.npz'
    np.savez(grid_path, obs=np.concatenate(obs), cmd=np.concatenate(cmd),
             demo=np.concatenate(demo_index), starts=np.array(starts), lengths=np.array(lengths),
             arm_q_col=np.int64(OBS_NAMES.index('arm_q_0')))
    return grid_path, offset, float(np.mean(lengths) / GRID_HZ)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    noise = json.loads(NOISE_PATH.read_text())['noise']
    sigma = noise['sigma']
    tau = noise['tau_c_s']
    if len(sigma) != 18 or len(tau) != 18:
        raise SystemExit('learner noise file does not have 18 dimensions')
    seeds = seed_stream(EPISODE_STREAM, args.episodes)
    noise_seeds = seed_stream(NOISE_STREAM, args.episodes)
    heldout = json.loads((DATA / 'seeds.json').read_text())
    overlap = set(seeds) & (set(heldout['train']) | set(heldout['test']))
    if overlap:
        raise SystemExit(f'{len(overlap)} episode seeds overlap train/test')
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT / 'dataset.json'
    spec = dict(master=MASTER_SEED, episode_stream=EPISODE_STREAM, noise_stream=NOISE_STREAM,
                rate_hz=RATE_HZ, n=args.episodes, noise='ou_learner',
                source=str(NOISE_PATH), sigma=sigma, tau_c_s=tau,
                seeds=seeds, noise_seeds=noise_seeds)
    if spec_path.exists():
        saved = json.loads(spec_path.read_text())
        if saved.get('seeds') != seeds or saved.get('noise_seeds') != noise_seeds:
            raise SystemExit(f'{spec_path} seeds do not match this request')
        saved.update({k: spec[k] for k in spec if k not in ('seeds', 'noise_seeds')})
        spec = saved
    spec_path.write_text(json.dumps(spec))

    pending = []
    for i, (seed, nseed) in enumerate(zip(seeds, noise_seeds)):
        path = OUT / f'demo_{i:04d}.npz'
        if not path.exists():
            pending.append((i, seed, nseed, path))
    print(f'{args.episodes} demos, {len(pending)} to generate, rate {RATE_HZ} Hz, '
          f'wheel σ={np.mean(sigma[:4]):.3f} rad/s, arm q σ={np.mean(sigma[4:11]):.4f} rad, '
          f'arm qd σ={np.mean(sigma[11:]):.4f} rad/s', flush=True)
    t0 = time.time()
    new_metas = []
    if pending:
        with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(sigma, tau)) as pool:
            futs = {pool.submit(demo_job, i, seed, nseed, path): i
                    for i, seed, nseed, path in pending}
            for n, fut in enumerate(as_completed(futs), 1):
                new_metas.append(fut.result())
                if n % 25 == 0 or n == len(pending):
                    print(f'  {n}/{len(pending)}  ({time.time() - t0:.0f}s)', flush=True)
    metas = []
    for i in range(args.episodes):
        with np.load(OUT / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            metas.append(json.loads(str(z['meta'])))
    catch = float(np.mean([m['success'] for m in metas]))
    dists = [m['min_dist'] for m in metas if m['min_dist'] is not None]
    miss = float(np.mean(dists) * 1000) if dists else None
    grid_path, n_rows, mean_len = build_grid(args.episodes)
    spec['catch_rate'] = catch
    spec['mean_miss_mm'] = miss
    spec['grid_rows'] = int(n_rows)
    spec['mean_length_s'] = mean_len
    spec_path.write_text(json.dumps(spec))
    miss_s = 'n/a' if miss is None else f'{miss:.1f}'
    print(f'{args.episodes} demos, catch {catch:.3f}, mean miss {miss_s} mm, '
          f'mean length {mean_len:.1f}s, {n_rows} grid rows -> {grid_path}', flush=True)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
