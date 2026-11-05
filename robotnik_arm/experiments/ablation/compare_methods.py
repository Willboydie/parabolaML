"""Train the Stage 1 winner architecture on the quiet-tail DART set and on the
first DAgger iteration, then roll both out on the original 200 test seeds.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from config import Config
from run_ablation import EVAL_CHUNK, eval_chunk, init_worker

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
TRAIN_SEED = 1
WINNER = dict(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5)


def train(data, demos, ckpt, log_path):
    if ckpt.exists():
        print(f'reusing {ckpt.name}', flush=True)
        return
    cfg = Config(**WINNER, demos=demos)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
           '--seed', str(TRAIN_SEED), '--out', str(ckpt), '--data', str(data), '--steps', '20000']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, 'w') as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise RuntimeError(f'train failed, see {log_path}')
    print(f'trained {ckpt.name} in {time.time() - t0:.0f}s', flush=True)


def summarize(episodes):
    dists = [e['min_dist'] for e in episodes if e['min_dist'] is not None and np.isfinite(e['min_dist'])]
    return dict(n=len(episodes),
                catch_rate=float(np.mean([e['success'] for e in episodes])),
                miss_mm=float(np.mean(dists) * 1000),
                median_miss_mm=float(np.median(dists) * 1000))


def evaluate(pool, weights, seeds):
    futs = [pool.submit(eval_chunk, str(weights), list(map(int, seeds[i:i + EVAL_CHUNK])), 0.01)
            for i in range(0, len(seeds), EVAL_CHUNK)]
    episodes = []
    for fut in as_completed(futs):
        rows, _wall = fut.result()
        for ep in rows:
            dist = ep['min_dist']
            episodes.append(dict(seed=int(ep['seed']), success=bool(ep['success']),
                                 min_dist=None if dist is None or not np.isfinite(dist) else float(dist)))
    return episodes


def main():
    from policy import export_numpy
    jobs = []
    dagger_grid = ROOT / 'data' / 'dagger_iter1' / 'grid50.npz'
    quiet_grid = ROOT / 'data' / 'quiet_dart' / 'grid50.npz'
    if dagger_grid.exists():
        jobs.append(('dagger', dagger_grid, 200, RESULTS / 'models' / 'dagger_iter1_s1.pt'))
    if quiet_grid.exists():
        n = len(json.loads((ROOT / 'data' / 'quiet_dart' / 'dataset.json').read_text())['seeds'])
        jobs.append(('quiet_dart', quiet_grid, n, RESULTS / 'models' / 'quiet_dart_s1.pt'))
    if not jobs:
        raise SystemExit('no dataset grid to train')
    for name, grid, demos, ckpt in jobs:
        train(grid, demos, ckpt, RESULTS / 'logs' / f'{name}.train.log')
    test_seeds = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    report = {}
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        for name, _grid, demos, ckpt in jobs:
            weights = ckpt.with_suffix('.npz')
            meta = export_numpy(ckpt, weights)
            print(f'eval {name} on {len(test_seeds)} test seeds', flush=True)
            t0 = time.time()
            episodes = evaluate(pool, weights, test_seeds)
            stats = summarize(episodes)
            stats.update(demos=demos, train_time_s=meta['train_time_s'], final_loss=meta['final_loss'],
                         eval_s=time.time() - t0)
            report[name] = stats
            print(f"{name}: catch {stats['catch_rate']:.3f}  miss {stats['miss_mm']:.1f} mm  "
                  f"median {stats['median_miss_mm']:.1f} mm", flush=True)
    out = RESULTS / 'method_compare.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
