"""Retrain the one-ball policy with the DeepSets encoder.

The data is the set that won step 0a. Three seeds, the same 200 test seeds.
The comparison is the flat relative-feature model at 38.2 mm mean miss.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import Config
from parking_split import aggregate, eval_weights, load_experts
from policy import load_policy
from stage2_log import finish_run, start_run

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
NARROW = ROOT / 'data' / 'grid50_offset.npz'
WIDE = ROOT / 'data' / 'grid50_offset_along55.npz'
BEST_MISS = 38.23595621013194
SEEDS = (0, 1, 2)


def choose_data():
    """Keep the 38.2 mm dataset unless the wider along-track round is clearly better."""
    path = RESULTS / 'offset_along55_train.json'
    if not path.exists():
        raise SystemExit('along-track round has not written its result yet')
    wide = json.loads(path.read_text())['rel_add_along55']['miss_mm']
    if wide + 1.0 < BEST_MISS:
        return WIDE, wide, 'along55'
    return NARROW, wide, 'narrow_offset'


def weight_hash(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    digest = hashlib.sha256()
    for key, value in sorted(blob['state_dict'].items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.numpy()).tobytes())
    return digest.hexdigest()


def permutation_gap(path):
    model = load_policy(path)
    torch.manual_seed(0)
    robot = torch.randn(4, model.cfg.frames, 33)
    balls = torch.randn(4, 2, model.cfg.frames, 22)
    with torch.no_grad():
        a = model.forward_set(robot, balls)
        b = model.forward_set(robot, balls.flip(1))
    return float((a - b).abs().max())


def train_all(cfg, data):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
        if ckpt.exists():
            print(f'reusing {ckpt.name}', flush=True)
        else:
            pending.append((seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            seed, ckpt = pending.pop(0)
            log_path = RESULTS / 'logs' / f'{cfg.name}_s{seed}.train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000', '--data', str(data)]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, seed, time.time()))
            print(f'training {cfg.name} seed {seed}', flush=True)
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
    data, wide_miss, choice = choose_data()
    cfg = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                 demos=2000, encoder='deepsets', pool='sum')
    print(f'data choice {choice}: wide-along miss {wide_miss:.2f} mm, training on {data.name}', flush=True)
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    run_id, record_path, record = start_run(
        'oneball-deepsets', cfg.name, cfg.to_dict(), sys.argv, dict(test_hash=hashlib.sha256(
            json.dumps(test).encode()).hexdigest(), train_seeds='grid prefix 2000'),
        note=f'data {choice}, wide miss {wide_miss:.2f}')
    t0 = time.time()
    train_all(cfg, data)
    hashes = {}
    perm = {}
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
        hashes[f's{seed}'] = weight_hash(ckpt)
        perm[f's{seed}'] = permutation_gap(ckpt)
        print(f's{seed} hash {hashes[f"s{seed}"][:16]}  permutation {perm[f"s{seed}"]:.3e}', flush=True)
    if len(set(hashes.values())) != 3:
        raise SystemExit('the three DeepSets checkpoints are not distinct')
    from policy import export_numpy
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    report = dict(data=str(data), data_choice=choice, wide_along_miss_mm=wide_miss,
                  reference_miss_mm=BEST_MISS, hashes=hashes, permutation_max_abs_diff=perm)
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
        weights = ckpt.with_suffix('.npz')
        export_numpy(ckpt, weights)
        print(f'rollout seed {seed}', flush=True)
        stats = eval_weights(weights, test, experts)
        report[f's{seed}'] = stats
        split = stats['split']
        print(f"s{seed}: catch {stats['catch_rate']:.3f}  miss {stats['miss_mm']:.1f} mm  "
              f"median {stats['median_miss_mm']:.1f}  "
              f"along {split['miss_along_abs_median_mm']:.0f}  "
              f"cross {split['miss_cross_abs_median_mm']:.0f}  "
              f"up {split['miss_up_abs_median_mm']:.0f}", flush=True)
    summary = aggregate([report[f's{s}'] for s in SEEDS])
    summary['clearly_worse_than_38mm'] = bool(summary['miss_mm'] > BEST_MISS + 5.0)
    report['deepsets'] = summary
    out = RESULTS / 'stage2' / 'oneball_deepsets.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    worse = summary['clearly_worse_than_38mm']
    finish_run(record_path, record, dict(miss_mm=summary['miss_mm'], catch_rate=summary['catch_rate']),
               [out] + [RESULTS / 'models' / f'{cfg.name}_s{s}.pt' for s in SEEDS])
    print(json.dumps(summary, indent=2), flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)
    if worse:
        raise SystemExit('DeepSets one-ball miss is clearly worse than 38.2 mm; stop before two-ball data')


if __name__ == '__main__':
    main()
