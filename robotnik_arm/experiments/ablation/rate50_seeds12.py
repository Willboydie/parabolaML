"""Train and score seeds 1 and 2 of the 50 Hz, 0.5 s DeepSets backbone.

Same Config and train.py arguments as primary_cfg() in rate50_diagnostic.py.
That diagnostic stopped after seed 0; this is the expansion it skipped.
"""
import json
import sys
import time
from pathlib import Path

from parking_split import load_experts
from rate50_diagnostic import eval_cfg, primary_cfg, train_seeds
from stage2_log import finish_run, start_run

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'stage2' / 'rate50_s1_s2.json'
SEEDS = (1, 2)


def main():
    cfg = primary_cfg()
    assert (cfg.horizon, cfg.execute, cfg.frames, cfg.name) == (
        50, 25, 13, 'r50_p1_e0.5_hstack0.24s_d2000_deepsets',
    ), (cfg.horizon, cfg.execute, cfg.frames, cfg.name)
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    run_id, record_path, record = start_run(
        'rate50-s1-s2', cfg.name, cfg.to_dict(), sys.argv,
        dict(test=len(test), train_seeds=list(SEEDS)),
        note='seeds 1 and 2 of the 50 Hz 1s predict / 0.5s execute DeepSets backbone')
    t0 = time.time()
    train_seeds(cfg, SEEDS)
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    primary = eval_cfg(cfg, SEEDS, test, experts)
    report = dict(config=cfg.to_dict(), primary=primary)
    OUT.write_text(json.dumps(report, indent=2))
    outputs = [OUT]
    for seed in SEEDS:
        outputs.append(ROOT / 'results' / 'models' / f'{cfg.name}_s{seed}.pt')
        outputs.append(ROOT / 'results' / 'models' / f'{cfg.name}_s{seed}.npz')
    finish_run(record_path, record, dict(
        miss_mm=primary['mean']['miss_mm'],
        catch_rate=primary['mean']['catch_rate_50mm']), outputs)
    mean = primary['mean']
    print(f"mean miss {mean['miss_mm']:.1f} mm  "
          f"along {mean['base_along_median_mm']:+.1f}  "
          f"ahead {mean['frac_ahead']:.3f}  "
          f"catch50 {mean['catch_rate_50mm']:.3f}", flush=True)
    print(f'wrote {OUT} in {time.time() - t0:.0f}s', flush=True)
    print(f'run {run_id}', flush=True)


if __name__ == '__main__':
    main()
