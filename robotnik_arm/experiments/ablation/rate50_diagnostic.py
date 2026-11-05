"""50 Hz smaller-chunk diagnostic on the DeepSets one-ball ceiling.

Step 0 measures signed along-track parking on the existing 25 Hz ceiling.
If that ceiling still parks ahead, seed 0 of a same-architecture 50 Hz policy
(1 s predict / 0.5 s execute, 0.24 s history) is trained and compared. Seeds
1–2 and a 25 Hz shorter-execute control run only if the committed tail moved.
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
from policy import export_numpy
from stage2_log import finish_run, start_run

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data' / 'grid50_offset.npz'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
SEEDS = (0, 1, 2)
OUT = RESULTS / 'stage2' / 'rate50_diagnostic.json'


def primary_cfg():
    return Config(rate_hz=50, predict_s=1.0, execute_frac=0.5, history='stack', history_s=0.24,
                  demos=2000, encoder='deepsets', pool='sum')


def control_cfg():
    return Config(rate_hz=25, predict_s=2.0, execute_frac=0.25, history='stack', history_s=0.5,
                  demos=2000, encoder='deepsets', pool='sum')


def still_ahead(split):
    """The overshoot story applies if the base is systematically in front of the expert."""
    return float(split['frac_ahead']) >= 0.55 or float(split['base_along_median_mm']) >= 5.0


def tail_moved(ceiling, trial):
    """Shorter execute should cut ahead-rate or shrink the signed along offset."""
    c, t = ceiling['split'], trial['split']
    ahead_drop = float(c['frac_ahead']) - float(t['frac_ahead'])
    along_improved = abs(float(t['base_along_median_mm'])) + 3.0 <= abs(float(c['base_along_median_mm']))
    miss_worse = float(trial['miss_mm']) > float(ceiling['miss_mm']) + 5.0
    if miss_worse and ahead_drop < 0.05 and not along_improved:
        return False
    return ahead_drop >= 0.05 or along_improved


def describe(stats):
    split = stats['split']
    return (f"catch10 {stats['catch_rate']:.3f}  catch50 {stats['catch_rate_50mm']:.3f}  "
            f"miss {stats['miss_mm']:.1f} mm  along {split['base_along_median_mm']:+.1f}  "
            f"ahead {split['frac_ahead']:.2f}")


def train_seeds(cfg, seeds):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in seeds:
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
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000', '--data', str(DATA)]
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
                raise RuntimeError(f'train failed for {cfg.name} seed {seed}')
            print(f'trained {cfg.name} seed {seed} in {time.time() - t0:.0f}s', flush=True)
        active = still


def eval_cfg(cfg, seeds, test, experts):
    blocks = []
    report = {}
    for seed in seeds:
        ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
        weights = ckpt.with_suffix('.npz')
        export_numpy(ckpt, weights)
        print(f'rollout {cfg.name} seed {seed}', flush=True)
        stats = eval_weights(weights, test, experts)
        report[f's{seed}'] = stats
        blocks.append(stats)
        print(f'  {describe(stats)}', flush=True)
    if len(blocks) > 1:
        report['mean'] = aggregate(blocks)
    return report


def main():
    if not DATA.exists():
        raise SystemExit(f'missing {DATA}')
    pcfg, ccfg = primary_cfg(), control_cfg()
    assert pcfg.horizon == 50 and pcfg.execute == 25 and pcfg.frames == 13, (
        pcfg.horizon, pcfg.execute, pcfg.frames)
    assert ccfg.execute == 13, ccfg.execute
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    run_id, record_path, record = start_run(
        'rate50-diagnostic', pcfg.name, dict(primary=pcfg.to_dict(), control=ccfg.to_dict()),
        sys.argv, dict(test=len(test), train_seeds=list(SEEDS)),
        note='signed parking on DeepSets ceiling, then 50 Hz 1s/0.5s if still ahead')
    t0 = time.time()
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)

    ceiling = {}
    ceiling_blocks = []
    for seed in SEEDS:
        weights = RESULTS / 'models' / f'{CEILING}_s{seed}.npz'
        if not weights.exists():
            raise SystemExit(f'missing ceiling {weights}')
        print(f'rollout ceiling seed {seed}', flush=True)
        stats = eval_weights(weights, test, experts)
        ceiling[f's{seed}'] = stats
        ceiling_blocks.append(stats)
        print(f'  {describe(stats)}', flush=True)
    ceiling['mean'] = aggregate(ceiling_blocks)
    print(f"ceiling mean: miss {ceiling['mean']['miss_mm']:.1f} mm  "
          f"along {ceiling['mean']['base_along_median_mm']:+.1f}  "
          f"ahead {ceiling['mean']['frac_ahead']:.2f}  "
          f"catch50 {ceiling['mean']['catch_rate_50mm']:.3f}", flush=True)

    report = dict(ceiling=ceiling, primary_config=pcfg.to_dict(), control_config=ccfg.to_dict(),
                  bias_present=still_ahead(ceiling['mean']), expanded=False, control_ran=False)
    if not report['bias_present']:
        report['stop'] = 'ceiling is not systematically ahead; overshoot diagnostic does not apply'
        print(report['stop'], flush=True)
        OUT.write_text(json.dumps(report, indent=2))
        finish_run(record_path, record, dict(miss_mm=ceiling['mean']['miss_mm'],
                                             catch_rate=ceiling['mean']['catch_rate_50mm']),
                   [OUT])
        print(f'wrote {OUT} in {time.time() - t0:.0f}s', flush=True)
        return

    train_seeds(pcfg, (0,))
    primary = eval_cfg(pcfg, (0,), test, experts)
    report['primary'] = primary
    moved = tail_moved(ceiling['s0'], primary['s0'])
    report['tail_moved'] = moved
    print(f"seed 0 vs ceiling s0: moved={moved}", flush=True)
    if not moved:
        report['stop'] = '50 Hz seed 0 did not cut the committed tail; not expanding'
        print(report['stop'], flush=True)
        OUT.write_text(json.dumps(report, indent=2))
        finish_run(record_path, record, dict(miss_mm=primary['s0']['miss_mm'],
                                             catch_rate=primary['s0']['catch_rate_50mm']),
                   [OUT, RESULTS / 'models' / f'{pcfg.name}_s0.pt'])
        print(f'wrote {OUT} in {time.time() - t0:.0f}s', flush=True)
        return

    report['expanded'] = True
    train_seeds(pcfg, (1, 2))
    primary = eval_cfg(pcfg, SEEDS, test, experts)
    report['primary'] = primary
    train_seeds(ccfg, SEEDS)
    report['control'] = eval_cfg(ccfg, SEEDS, test, experts)
    report['control_ran'] = True
    OUT.write_text(json.dumps(report, indent=2))
    outputs = [OUT]
    for seed in SEEDS:
        outputs.append(RESULTS / 'models' / f'{pcfg.name}_s{seed}.pt')
        outputs.append(RESULTS / 'models' / f'{ccfg.name}_s{seed}.pt')
    finish_run(record_path, record, dict(miss_mm=primary['mean']['miss_mm'],
                                         catch_rate=primary['mean']['catch_rate_50mm']),
               outputs)
    print(f"primary mean miss {primary['mean']['miss_mm']:.1f} mm  "
          f"along {primary['mean']['base_along_median_mm']:+.1f}  "
          f"ahead {primary['mean']['frac_ahead']:.2f}", flush=True)
    print(f'wrote {OUT} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
