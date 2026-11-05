"""1,000 goal-shift demos drawn from the DeepSets parking cloud.

The expert plans the optimal park, then the goal is moved by the shared offset.
Heading is unchanged. Writes data/offset_parking_matched and grid50_offset_matched.npz.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gen_offset_parking import OffsetGoalDriver, init_worker
from variety_data import (
    MATCHED_DIR, MATCHED_GRID, MASTER_SEED, OFFSET_CAP, OFFSET_STREAM, N,
    assert_writable, build_grid, load_offsets, training_seeds,
)

OUT = MATCHED_DIR


def demo_job(index, seed, along_m, side_m, path):
    from catch_env import rollout
    from gen_offset_parking import _env
    path = assert_writable(path)
    driver = OffsetGoalDriver(along_m, side_m)
    out = rollout(_env, driver, seed, 1000, log=True)
    arrays = out.pop('arrays')
    if not np.isfinite(out['min_dist']):
        out['min_dist'] = None
    meta = dict(out, index=index, dart=False, rate_hz=1000, sim_dt=float(_env.dt),
                offset_along_m=along_m, offset_side_m=side_m, offset_cap_m=OFFSET_CAP,
                offset_stream=OFFSET_STREAM, arm='goal_shift', heading_applied_rad=0.0)
    tmp = Path(str(path) + '.tmp.npz')
    assert_writable(tmp)
    np.savez(tmp, obs=arrays['obs'], cmd=arrays['cmd'], applied=arrays['applied'], ready=arrays['ready'],
             meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return meta


def main():
    if not (OUT.parent / 'grid50.npz').exists():
        raise SystemExit('missing data/grid50.npz')
    spec_in = load_offsets()
    seeds = training_seeds()
    along = [float(v) for v in spec_in['along_m']]
    side = [float(v) for v in spec_in['side_m']]
    print(f"shared offsets: along sd {np.std(along) * 1000:.1f} mm, "
          f"side sd {np.std(side) * 1000:.1f} mm, include_heading {spec_in['include_heading']} "
          f"(goal-shift leaves heading unchanged)", flush=True)
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = assert_writable(OUT / 'dataset.json')
    spec = dict(master=MASTER_SEED, offset_stream=OFFSET_STREAM, n=N, rate_hz=1000, arm='goal_shift',
                cap_m=OFFSET_CAP, include_heading=False,
                heading_abs_median_deg=spec_in['heading_abs_median_deg'],
                seeds=seeds, along_m=along, side_m=side)
    if spec_path.exists():
        saved = json.loads(spec_path.read_text())
        if saved.get('seeds') != seeds or saved.get('side_m') != side or saved.get('along_m') != along:
            raise SystemExit(f'{spec_path} does not match this offset list')
    spec_path.write_text(json.dumps(spec))
    pending = [(i, int(seed), along[i], side[i], OUT / f'demo_{i:04d}.npz')
               for i, seed in enumerate(seeds) if not (OUT / f'demo_{i:04d}.npz').exists()]
    print(f'{N} goal-shift demos, {len(pending)} to generate', flush=True)
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
    n_rows = build_grid(OUT, MATCHED_GRID, N)
    spec.update(catch_rate=catch,
                mean_miss_mm=None if not dists else float(np.mean(dists) * 1000),
                median_miss_mm=None if not dists else float(np.median(dists) * 1000),
                grid_rows=int(n_rows), grid=str(MATCHED_GRID))
    spec_path.write_text(json.dumps(spec))
    miss = spec['median_miss_mm']
    miss_s = 'n/a' if miss is None else f'{miss:.2f}'
    print(f'{N} demos, catch {catch:.3f}, median miss {miss_s} mm, {n_rows} grid rows -> {MATCHED_GRID}',
          flush=True)
    print(f'done in {time.time() - t0:.0f}s', flush=True)
    if catch < 0.98:
        raise SystemExit(f'expert catch at 10 mm is {catch:.3f}; not training on this')


if __name__ == '__main__':
    main()
