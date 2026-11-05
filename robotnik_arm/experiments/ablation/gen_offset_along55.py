"""1,000 offset-parking demos with a wider along-track spread.

Same scenes and cap as the first offset set. Sideways stays at about 48 mm.
Along-track is about 55 mm. The parking heading is unchanged and the labels
are the commands that were executed. The grid is the original 1,000 demos
plus these, not the earlier offset set.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gen_offset_parking import OffsetGoalDriver, init_worker

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'offset_parking_along55'
MASTER_SEED = 2026
OFFSET_STREAM = 41
N = 1000
SIDE_SIGMA = 0.048
ALONG_SIGMA = 0.055
OFFSET_CAP = 0.150
GRID = DATA / 'grid50_offset_along55.npz'


def sample_offset(index):
    seed = int(np.random.SeedSequence([MASTER_SEED, OFFSET_STREAM, index]).generate_state(1, dtype=np.uint64)[0] >> 1)
    rng = np.random.default_rng(seed)
    side = float(rng.normal(0.0, SIDE_SIGMA))
    along = float(rng.normal(0.0, ALONG_SIGMA))
    mag = float(np.hypot(side, along))
    if mag > OFFSET_CAP:
        side *= OFFSET_CAP / mag
        along *= OFFSET_CAP / mag
    return along, side


def demo_job(index, seed, along_m, side_m, path):
    from catch_env import rollout
    env = _worker_env()
    out = rollout(env, OffsetGoalDriver(along_m, side_m), seed, 1000, log=True)
    arrays = out.pop('arrays')
    if not np.isfinite(out['min_dist']):
        out['min_dist'] = None
    meta = dict(out, index=index, dart=False, rate_hz=1000, sim_dt=float(env.dt),
                offset_along_m=along_m, offset_side_m=side_m, offset_cap_m=OFFSET_CAP,
                offset_stream=OFFSET_STREAM, side_sigma_m=SIDE_SIGMA, along_sigma_m=ALONG_SIGMA)
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, obs=arrays['obs'], cmd=arrays['cmd'], applied=arrays['applied'], ready=arrays['ready'],
             meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return meta


def _worker_env():
    from gen_offset_parking import _env
    return _env


def build_combined(n_new):
    from catch_env import FIRST_TICK_STEP
    stride = 20
    with np.load(DATA / 'grid50.npz') as z:
        n_rows = int(z['starts'][999] + z['lengths'][999])
        obs = [np.array(z['obs'][:n_rows], copy=True)]
        cmd = [np.array(z['cmd'][:n_rows], copy=True)]
        demo = [np.array(z['demo'][:n_rows], copy=True)]
        starts = [int(v) for v in z['starts'][:1000]]
        lengths = [int(v) for v in z['lengths'][:1000]]
        q_col = int(z['arm_q_col'])
    if int(demo[0].max()) != 999 or len(starts) != 1000:
        raise RuntimeError('grid50.npz is not the original 1,000 demos')
    offset = int(starts[-1] + lengths[-1])
    for i in range(n_new):
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
        demo.append(np.full(len(o), 1000 + i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
    np.savez(GRID, obs=np.concatenate(obs), cmd=np.concatenate(cmd), demo=np.concatenate(demo),
             starts=np.asarray(starts), lengths=np.asarray(lengths), arm_q_col=np.int64(q_col))
    return offset


def main():
    if not (DATA / 'grid50.npz').exists():
        raise SystemExit('missing data/grid50.npz')
    heldout = json.loads((DATA / 'seeds.json').read_text())
    seeds = heldout['train'][:N]
    if len(seeds) < N:
        raise SystemExit(f'only {len(seeds)} training seeds')
    if set(seeds) & set(heldout['test']):
        raise SystemExit('training seeds overlap the test set')
    offsets = [sample_offset(i) for i in range(N)]
    sides = np.array([s for _a, s in offsets])
    alongs = np.array([a for a, _s in offsets])
    print(f'offset sideways sd {sides.std() * 1000:.1f} mm, along sd {alongs.std() * 1000:.1f} mm, '
          f'max {np.hypot(sides, alongs).max() * 1000:.1f} mm', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT / 'dataset.json'
    spec = dict(master=MASTER_SEED, offset_stream=OFFSET_STREAM, n=N, rate_hz=1000,
                side_sigma_m=SIDE_SIGMA, along_sigma_m=ALONG_SIGMA, cap_m=OFFSET_CAP,
                seeds=seeds, along_m=[a for a, _s in offsets], side_m=[s for _a, s in offsets])
    if spec_path.exists():
        saved = json.loads(spec_path.read_text())
        if saved.get('seeds') != seeds or saved.get('side_m') != spec['side_m']:
            raise SystemExit(f'{spec_path} does not match this offset list')
    spec_path.write_text(json.dumps(spec))
    pending = [(i, int(seed), offsets[i][0], offsets[i][1], OUT / f'demo_{i:04d}.npz')
               for i, seed in enumerate(seeds) if not (OUT / f'demo_{i:04d}.npz').exists()]
    print(f'{N} along-track offset demos, {len(pending)} to generate', flush=True)
    t0 = time.time()
    if pending:
        with ProcessPoolExecutor(16, initializer=init_worker) as pool:
            futs = [pool.submit(demo_job, *job) for job in pending]
            for k, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if k % 25 == 0 or k == len(pending):
                    print(f'  {k}/{len(pending)} ({time.time() - t0:.0f}s)', flush=True)
    metas = []
    for i in range(N):
        with np.load(OUT / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            metas.append(json.loads(str(z['meta'])))
    catch = float(np.mean([m['success'] for m in metas]))
    dists = [m['min_dist'] for m in metas if m.get('min_dist') is not None]
    n_rows = build_combined(N)
    spec.update(catch_rate=catch,
                mean_miss_mm=None if not dists else float(np.mean(dists) * 1000),
                median_miss_mm=None if not dists else float(np.median(dists) * 1000),
                realized_side_sd_mm=float(sides.std() * 1000),
                realized_along_sd_mm=float(alongs.std() * 1000),
                grid_rows=int(n_rows), grid=str(GRID))
    spec_path.write_text(json.dumps(spec))
    miss = spec['median_miss_mm']
    miss_s = 'n/a' if miss is None else f'{miss:.2f}'
    print(f'{N} demos, catch {catch:.3f}, median miss {miss_s} mm, {n_rows} grid rows -> {GRID}', flush=True)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
