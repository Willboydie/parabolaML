"""1,000 one-ball demos at the quiet-tail noise scale, truncated at 300 mm drift.

Noise is the correlated OU from results/learner_cov.json, scaled by the alpha
chosen in results/recover_sweep.json, and switched off for the last 3 s. A demo
stops at the first step whose end effector is more than 300 mm from the clean
expert at that same time.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gen_learner_dart import seed_stream
from sweep_cov_noise import DT, TAU

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'quiet_dart'
MASTER_SEED = 2026
EPISODE_STREAM = 35
NOISE_STREAM = 36
RATE_HZ = 25
QUIET_S = 3.0
DRIFT_M = 0.30
EE_COL = 39

_env = None
_cov = None
_alpha = None


def init_worker(cov, alpha):
    global _env, _cov, _alpha
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()
    _cov = np.asarray(cov, dtype=np.float64)
    _alpha = float(alpha)


def demo_job(index, seed, noise_seed, path):
    from catch_env import CorrelatedCommandNoise, ExpertDriver, rollout
    clean = rollout(_env, ExpertDriver(), seed, RATE_HZ, stop_on_catch=False, log=True)
    ee = clean['arrays']['obs'][:, EE_COL:EE_COL + 3]
    del clean
    noise = CorrelatedCommandNoise(noise_seed, _cov, TAU, DT, alpha=_alpha)
    out = rollout(_env, ExpertDriver(), seed, RATE_HZ, stop_on_catch=True, log=True,
                  command_noise=noise, quiet_tail_s=QUIET_S, drift_ee=ee, drift_limit=DRIFT_M)
    arrays = out.pop('arrays')
    if not np.isfinite(out['min_dist']):
        out['min_dist'] = None
    meta = dict(out, index=index, dart=True, noise='ou_correlated_quiet', alpha=_alpha,
                noise_seed=int(noise_seed), rate_hz=RATE_HZ, sim_dt=float(_env.dt),
                quiet_tail_s=QUIET_S, drift_limit_m=DRIFT_M)
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, obs=arrays['obs'], cmd=arrays['cmd'], executed=arrays['executed'],
             applied=arrays['applied'], ready=arrays['ready'], meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return meta


def build_grid(n):
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
                raise RuntimeError(f'{path.name} index {meta["index"]}')
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        if len(o) == 0:
            raise RuntimeError(f'{path.name} truncated to nothing')
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
    report = json.loads((ROOT / 'results' / 'recover_sweep.json').read_text())
    alpha = report.get('chosen_alpha')
    if alpha is None:
        raise SystemExit('no alpha in 0.15–0.25 recovered a 90% catch; not generating')
    cov = json.loads((ROOT / 'results' / 'learner_cov.json').read_text())['cov']
    n = 1000
    seeds = seed_stream(EPISODE_STREAM, n)
    noise_seeds = seed_stream(NOISE_STREAM, n)
    heldout = json.loads((DATA / 'seeds.json').read_text())
    overlap = set(seeds) & (set(heldout['train']) | set(heldout['test']))
    if overlap:
        raise SystemExit(f'{len(overlap)} episode seeds overlap train/test')
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT / 'dataset.json'
    spec = dict(master=MASTER_SEED, episode_stream=EPISODE_STREAM, noise_stream=NOISE_STREAM,
                rate_hz=RATE_HZ, n=n, noise='ou_correlated_quiet', alpha=alpha,
                quiet_tail_s=QUIET_S, drift_limit_m=DRIFT_M, tau_s=TAU.tolist(),
                seeds=seeds, noise_seeds=noise_seeds)
    if spec_path.exists():
        saved = json.loads(spec_path.read_text())
        if saved.get('seeds') != seeds or saved.get('alpha') != alpha:
            raise SystemExit(f'{spec_path} does not match this alpha and seed list')
    spec_path.write_text(json.dumps(spec))
    pending = [(i, s, ns, OUT / f'demo_{i:04d}.npz')
               for i, (s, ns) in enumerate(zip(seeds, noise_seeds))
               if not (OUT / f'demo_{i:04d}.npz').exists()]
    print(f'{n} demos, {len(pending)} to generate, alpha {alpha}, quiet {QUIET_S:.0f}s, '
          f'truncate at {DRIFT_M * 1000:.0f} mm', flush=True)
    t0 = time.time()
    if pending:
        with ProcessPoolExecutor(16, initializer=init_worker, initargs=(cov, alpha)) as pool:
            futs = [pool.submit(demo_job, i, s, ns, path) for i, s, ns, path in pending]
            for k, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if k % 25 == 0 or k == len(pending):
                    print(f'  {k}/{len(pending)} ({time.time() - t0:.0f}s)', flush=True)
    metas = []
    for i in range(n):
        with np.load(OUT / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            metas.append(json.loads(str(z['meta'])))
    catch = float(np.mean([m['success'] for m in metas]))
    dists = [m['min_dist'] for m in metas if m.get('min_dist') is not None]
    truncated = float(np.mean([m.get('drift_truncated', False) for m in metas]))
    grid_path, n_rows, mean_len = build_grid(n)
    spec.update(catch_rate=catch, mean_miss_mm=None if not dists else float(np.mean(dists) * 1000),
                truncated_frac=truncated, grid_rows=int(n_rows), mean_length_s=mean_len)
    spec_path.write_text(json.dumps(spec))
    miss = spec['mean_miss_mm']
    miss_s = 'n/a' if miss is None else f'{miss:.1f}'
    print(f'{n} demos, catch {catch:.3f}, mean miss {miss_s} mm, truncated {truncated:.3f}, '
          f'mean length {mean_len:.1f}s -> {grid_path}', flush=True)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
