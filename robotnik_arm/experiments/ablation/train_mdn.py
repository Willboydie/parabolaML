"""Train the two-ball mixture density network, three seeds, two-ball data only.

Same backbone as the squared-error copier. The head is five diagonal Gaussians
over the 50 by 18 action chunk. Checkpoints are twoball_mdn_s*.pt.
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


def launch(cfg, data):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'twoball_mdn_s{seed}.pt'
        if ckpt.exists():
            print(f'reusing {ckpt.name}', flush=True)
        else:
            pending.append((seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            seed, ckpt = pending.pop(0)
            log_path = RESULTS / 'logs' / f'twoball_mdn_s{seed}.train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000', '--data', str(data)]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, seed, time.time()))
            print(f'training mdn seed {seed}', flush=True)
        time.sleep(2)
        still = []
        for proc, log, seed, t0 in active:
            if proc.poll() is None:
                still.append((proc, log, seed, t0))
                continue
            log.close()
            if proc.returncode:
                raise RuntimeError(f'mdn seed {seed} failed')
            print(f'trained mdn seed {seed} in {time.time() - t0:.0f}s', flush=True)
        active = still


def main():
    from eval_mdn import check_mdn
    check_mdn()
    if not DATA.exists() or not SUMMARY.exists():
        raise SystemExit('two-ball grid is missing; generate it first')
    summary = json.loads(SUMMARY.read_text())
    if summary.get('expert_catch_10mm', 0) < 0.98:
        raise SystemExit('expert check on the two-ball demos is below 98%')
    if int(summary.get('n', 0)) != 2000 or float(summary.get('separation_mm_min', 0)) < 199:
        raise SystemExit('the two-ball summary is not the 2,000-episode, 200 mm set')
    cfg = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                 demos=2000, encoder='deepsets', pool='sum', oracle=False, head='mdn', n_components=5,
                 sigma_floor=0.01)
    _run_id, record_path, record = start_run(
        'twoball-mdn', 'mdn', cfg.to_dict(), sys.argv, dict(seeds=list(SEEDS), data=str(DATA)),
        note='five diagonal Gaussians, negative log-likelihood, two-ball data only')
    t0 = time.time()
    launch(cfg, DATA)
    hashes, tails = {}, {}
    import torch
    for seed in SEEDS:
        ckpt = RESULTS / 'models' / f'twoball_mdn_s{seed}.pt'
        hashes[f's{seed}'] = weight_hash(ckpt)
        export_numpy(ckpt, ckpt.with_suffix('.npz'))
        blob = torch.load(ckpt, map_location='cpu', weights_only=False)
        diags = blob['meta'].get('diagnostics') or []
        tails[f's{seed}'] = diags[-1] if diags else None
        print(f'mdn s{seed} {hashes[f"s{seed}"][:16]}', flush=True)
    if len(set(hashes.values())) != 3:
        raise SystemExit('mdn seeds are not distinct checkpoints')
    out = RESULTS / 'stage2' / 'two_ball_mdn_train.json'
    report = dict(data=str(DATA), config=cfg.to_dict(), hashes=hashes, last_diagnostic=tails,
                  wall_s=time.time() - t0)
    out.write_text(json.dumps(report, indent=2))
    finish_run(record_path, record, dict(miss_mm=None, catch_rate=None),
               [out] + [RESULTS / 'models' / f'twoball_mdn_s{seed}.pt' for seed in SEEDS])
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
