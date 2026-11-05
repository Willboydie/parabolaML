"""Train relative-features-added on the original 1,000 demos plus the 1,000 offset parks.

Three seeds. Miss and the along-track and cross-track compensation fractions are
measured on the same 200 test seeds as the earlier relative-feature round.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from config import Config
from run_relative import compensation, init_env, init_model, pool_map, policy_episode, summarize_miss

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data' / 'grid50_offset.npz'
CFG = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
             demos=2000, features='rel_add')
SEEDS = (0, 1, 2)


def train_all():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{CFG.name}_s{seed}.pt'
        if ckpt.exists():
            print(f'reusing {ckpt.name}', flush=True)
        else:
            pending.append((seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            seed, ckpt = pending.pop(0)
            log_path = RESULTS / 'logs' / f'{CFG.name}_s{seed}.train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(CFG.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000',
                   '--data', str(DATA)]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, seed, time.time()))
            print(f'training {CFG.name} seed {seed}', flush=True)
        time.sleep(2)
        still = []
        for proc, log, seed, t0 in active:
            if proc.poll() is None:
                still.append((proc, log, seed, t0))
                continue
            log.close()
            if proc.returncode:
                raise RuntimeError(f'train failed for seed {seed}')
            print(f'trained seed {seed} in {time.time() - t0:.0f}s', flush=True)
        active = still


def expert_job(seed):
    from run_relative import expert_episode
    return expert_episode(seed)


def main():
    if not DATA.exists():
        raise SystemExit(f'missing {DATA}; generate the offset demos first')
    t0 = time.time()
    train_all()
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    print(f'expert reference on {len(test)} test seeds', flush=True)
    expert_rows = pool_map(expert_job, [(int(s),) for s in test], init_env, label='expert')
    experts = {r['seed']: r for r in expert_rows}
    from policy import export_numpy
    report = {}
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{CFG.name}_s{seed}.pt'
        weights = ckpt.with_suffix('.npz')
        export_numpy(ckpt, weights)
        print(f'rollout seed {seed}', flush=True)
        rows = pool_map(policy_episode, [(int(s),) for s in test], init_model, (str(weights),), f's{seed}')
        stats = summarize_miss(rows)
        stats['compensation'] = compensation(rows, experts)
        report[f's{seed}'] = stats
        comp = stats['compensation']
        print(f"s{seed}: catch {stats['catch_rate']:.3f}  miss {stats['miss_mm']:.1f} mm  "
              f"median {stats['median_miss_mm']:.1f}  "
              f"comp along {comp['along']['regression']:.2f}  cross {comp['cross']['regression']:.2f}",
              flush=True)
    block = [report[f's{s}'] for s in SEEDS]
    def col(path):
        vals = []
        for item in block:
            cur = item
            for part in path:
                cur = cur[part]
            vals.append(cur)
        return vals
    report['rel_add_d2000'] = dict(
        miss_mm_mean=float(np.mean(col(['miss_mm']))),
        miss_mm_sd=float(np.std(col(['miss_mm']))),
        median_miss_mm_mean=float(np.mean(col(['median_miss_mm']))),
        catch_mean=float(np.mean(col(['catch_rate']))),
        comp_along=float(np.mean(col(['compensation', 'along', 'regression']))),
        comp_along_sd=float(np.std(col(['compensation', 'along', 'regression']))),
        comp_cross=float(np.mean(col(['compensation', 'cross', 'regression']))),
        comp_cross_sd=float(np.std(col(['compensation', 'cross', 'regression']))),
    )
    out = RESULTS / 'offset_parking_train.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['rel_add_d2000'], indent=2), flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
