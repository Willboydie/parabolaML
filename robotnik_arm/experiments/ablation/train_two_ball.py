"""Train the two-ball MSE copier and the oracle on two-ball data only.

Three seeds each. Same backbone as the one-ball ceiling: DeepSets, 25 Hz,
2 s chunks, half of each chunk executed, 0.5 s of stacked history, 20,000 steps.
The checkpoints are not written under the ceiling's filename.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from config import Config
from policy import export_numpy
from stage2_log import finish_run, start_run
from train_deepsets_oneball import weight_hash

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'grid50_two_ball.npz'
SUMMARY = ROOT / 'data' / 'two_ball' / 'summary.json'
RESULTS = ROOT / 'results'
SEEDS = (0, 1, 2)


def launch(cfg, role, data):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'twoball_{role}_s{seed}.pt'
        if ckpt.exists():
            print(f'reusing {ckpt.name}', flush=True)
        else:
            pending.append((seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            seed, ckpt = pending.pop(0)
            log_path = RESULTS / 'logs' / f'twoball_{role}_s{seed}.train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000', '--data', str(data)]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, seed, time.time()))
            print(f'training {role} seed {seed}', flush=True)
        time.sleep(2)
        still = []
        for proc, log, seed, t0 in active:
            if proc.poll() is None:
                still.append((proc, log, seed, t0))
                continue
            log.close()
            if proc.returncode:
                raise RuntimeError(f'{role} seed {seed} failed')
            print(f'trained {role} seed {seed} in {time.time() - t0:.0f}s', flush=True)
        active = still


def main():
    if not DATA.exists() or not SUMMARY.exists():
        raise SystemExit('two-ball grid is missing; generate it first')
    summary = json.loads(SUMMARY.read_text())
    if summary.get('expert_catch_10mm', 0) < 0.98:
        raise SystemExit('expert check on the two-ball demos is below 98%')
    if int(summary.get('n', 0)) != 2000 or float(summary.get('separation_mm_min', 0)) < 199:
        raise SystemExit('the two-ball summary is not the 2,000-episode, 200 mm set')
    mse = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                 demos=2000, encoder='deepsets', pool='sum', oracle=False)
    oracle = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                    demos=2000, encoder='deepsets', pool='sum', oracle=True)
    _run_id, record_path, record = start_run(
        'twoball-train', 'mse-and-oracle', dict(mse=mse.to_dict(), oracle=oracle.to_dict()),
        sys.argv, dict(seeds=list(SEEDS), data=str(DATA)),
        note='two-ball data only, 3 seeds, no one-ball rows')
    t0 = time.time()
    launch(mse, 'mse', DATA)
    launch(oracle, 'oracle', DATA)
    hashes = {}
    for role in ('mse', 'oracle'):
        for seed in SEEDS:
            ckpt = RESULTS / 'models' / f'twoball_{role}_s{seed}.pt'
            hashes[f'{role}_s{seed}'] = weight_hash(ckpt)
            export_numpy(ckpt, ckpt.with_suffix('.npz'))
            print(f'{role} s{seed} {hashes[f"{role}_s{seed}"][:16]}', flush=True)
        if len({hashes[f'{role}_s{seed}'] for seed in SEEDS}) != 3:
            raise SystemExit(f'{role} seeds are not distinct checkpoints')
    out = RESULTS / 'stage2' / 'two_ball_train.json'
    report = dict(data=str(DATA), hashes=hashes, wall_s=time.time() - t0)
    out.write_text(json.dumps(report, indent=2))
    finish_run(record_path, record, dict(miss_mm=None, catch_rate=None),
               [out] + [RESULTS / 'models' / f'twoball_{role}_s{seed}.pt' for role in ('mse', 'oracle') for seed in SEEDS])
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
