"""Train relative-features-added on the original 1,000 plus the wider along-track offsets.

The checkpoint name is separate from today's 2,000-demo run, which used the
narrower along-track set. Same 200 test seeds, same three seeds.
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
from parking_split import aggregate, eval_weights, load_experts

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data' / 'grid50_offset_along55.npz'
CFG = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
             demos=2000, features='rel_add')
TAG = 'r25_p2_e0.5_hstack0.5s_d2000_rel_add_along55'
SEEDS = (0, 1, 2)


def train_all():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{TAG}_s{seed}.pt'
        if ckpt.exists():
            print(f'reusing {ckpt.name}', flush=True)
        else:
            pending.append((seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            seed, ckpt = pending.pop(0)
            log_path = RESULTS / 'logs' / f'{TAG}_s{seed}.train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(CFG.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000',
                   '--data', str(DATA)]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, seed, time.time()))
            print(f'training {TAG} seed {seed}', flush=True)
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


def main():
    if not DATA.exists():
        raise SystemExit(f'missing {DATA}; generate the along-track offset demos first')
    t0 = time.time()
    train_all()
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    from policy import export_numpy
    report = {}
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{TAG}_s{seed}.pt'
        weights = ckpt.with_suffix('.npz')
        export_numpy(ckpt, weights)
        print(f'rollout seed {seed}', flush=True)
        stats = eval_weights(weights, test, experts)
        report[f's{seed}'] = stats
        split = stats['split']
        comp = stats['compensation']
        print(f"s{seed}: catch {stats['catch_rate']:.3f}  miss {stats['miss_mm']:.1f} mm  "
              f"median {stats['median_miss_mm']:.1f}  "
              f"comp along {comp['along']['regression']:.2f}  cross {comp['cross']['regression']:.2f}  "
              f"base {split['base_offset_median_mm']:.0f}  "
              f"along {split['miss_along_abs_median_mm']:.0f}  "
              f"cross {split['miss_cross_abs_median_mm']:.0f}  "
              f"up {split['miss_up_abs_median_mm']:.0f}", flush=True)
    report['rel_add_along55'] = aggregate([report[f's{s}'] for s in SEEDS])
    out = RESULTS / 'offset_along55_train.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['rel_add_along55'], indent=2), flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
