"""Train DeepSets on the two new parking-variety grids and compare them to the ceiling.

Checkpoints use the matched and planned tags. The current one-ball weights and
oneball_deepsets.json are read and not rewritten. An arm whose expert caught
under 98% at 10 mm is not trained.
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

from config import Config
from parking_split import compensation, load_experts, miss_components
from run_relative import init_model, policy_episode, pool_map, summarize_miss
from variety_data import (
    COMPARE, DATA, DEEPSETS_TAG, MATCHED_DIR, MATCHED_GRID, MATCHED_TAG, MODELS, PLANNED_DIR,
    PLANNED_GRID, PLANNED_TAG, RESULTS, assert_writable, test_seeds,
)

SEEDS = (0, 1, 2)
ARMS = (
    ('matched', MATCHED_TAG, MATCHED_GRID, MATCHED_DIR / 'dataset.json'),
    ('planned', PLANNED_TAG, PLANNED_GRID, PLANNED_DIR / 'dataset.json'),
)
CEILING_MISS = 36.06269104689875


def cfg_for(tag):
    if tag == DEEPSETS_TAG or tag.startswith(DEEPSETS_TAG + '_s'):
        raise SystemExit(f'refusing to train over the ceiling tag {tag}')
    return Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
                  demos=2000, encoder='deepsets', pool='sum')


def weight_hash(path):
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    digest = hashlib.sha256()
    for key, value in sorted(blob['state_dict'].items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.numpy()).tobytes())
    return digest.hexdigest()


def stamp(paths):
    out = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            stat = path.stat()
            out[str(path)] = (stat.st_mtime_ns, stat.st_size)
    return out


def train_arm(tag, data):
    cfg = cfg_for(tag)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    pending = []
    for seed in SEEDS:
        ckpt = assert_writable(MODELS / f'{tag}_s{seed}.pt')
        if ckpt.exists():
            print(f'reusing {ckpt.name}', flush=True)
        else:
            pending.append((seed, ckpt))
    active = []
    while pending or active:
        while pending and len(active) < 2:
            seed, ckpt = pending.pop(0)
            log_path = assert_writable(RESULTS / 'logs' / f'{tag}_s{seed}.train.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(DATA.parent / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                   '--seed', str(seed), '--out', str(ckpt), '--steps', '20000', '--data', str(data)]
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=DATA.parent, stdout=log, stderr=subprocess.STDOUT, env=env)
            active.append((proc, log, seed, time.time()))
            print(f'training {tag} seed {seed}', flush=True)
        time.sleep(2)
        still = []
        for proc, log, seed, t0 in active:
            if proc.poll() is None:
                still.append((proc, log, seed, t0))
                continue
            log.close()
            if proc.returncode:
                raise RuntimeError(f'train failed for {tag} seed {seed}')
            print(f'trained {tag} seed {seed} in {time.time() - t0:.0f}s', flush=True)
        active = still


def eval_weights(weights, seeds, experts):
    rows = pool_map(policy_episode, [(int(s),) for s in seeds], init_model, (str(weights),), Path(weights).stem,
                    workers=8)
    stats = summarize_miss(rows)
    caught = [r['miss_mm'] is not None and r['miss_mm'] < 50.0 for r in rows]
    stats['catch_50'] = float(np.mean(caught)) if rows else None
    stats['compensation'] = compensation(rows, experts)
    stats['split'] = miss_components(rows, experts)
    return stats


def aggregate(blocks):
    def col(path):
        vals = []
        for item in blocks:
            cur = item
            for part in path:
                cur = cur[part]
            vals.append(cur)
        return np.asarray(vals, float)

    paths = (
        ('miss_mm', ['miss_mm']),
        ('median_miss_mm', ['median_miss_mm']),
        ('catch_rate', ['catch_rate']),
        ('catch_50', ['catch_50']),
        ('base_offset_median_mm', ['split', 'base_offset_median_mm']),
        ('miss_along_abs_median_mm', ['split', 'miss_along_abs_median_mm']),
        ('miss_cross_abs_median_mm', ['split', 'miss_cross_abs_median_mm']),
        ('miss_up_abs_median_mm', ['split', 'miss_up_abs_median_mm']),
        ('comp_along', ['compensation', 'along', 'regression']),
        ('comp_cross', ['compensation', 'cross', 'regression']),
    )
    out = {}
    for name, path in paths:
        vals = col(path)
        out[name] = float(np.mean(vals))
        out[name + '_sd'] = float(np.std(vals))
    return out


def helps(summary, ceiling_miss):
    if summary is None:
        return False
    return bool(ceiling_miss - summary['miss_mm'] > 2.0 and summary['comp_along'] > 0.55)


def reading(goal_shift, planner, ceiling_miss):
    goal_helps = helps(goal_shift, ceiling_miss)
    plan_helps = helps(planner, ceiling_miss)
    if goal_helps and not plan_helps:
        text = ('Goal-shift clears the bar and the planner does not. The missing ingredient was the '
                'shape of the parking cloud. That recipe is the candidate for later two-ball data.')
    elif plan_helps and not goal_helps:
        text = ('The planner clears the bar and the goal-shift does not. Shifting the goal after the '
                'plan is the wrong way to show the correction.')
    elif goal_helps and plan_helps:
        better = 'goal-shift' if goal_shift['miss_mm'] <= planner['miss_mm'] else 'planner'
        text = (f'Both clear the bar. The lower mean miss is the {better}. '
                'Either recipe is a candidate for later two-ball data.')
    else:
        text = ('Neither beats the ceiling by more than 2 mm with along compensation above 0.55. '
                'Leave the two-ball recipe on the current narrow offsets. The residual is not this '
                'parking distribution.')
    return dict(goal_shift_helps=goal_helps, planner_helps=plan_helps, text=text)


def arm_ready(dataset_path, grid_path):
    if not grid_path.exists() or not dataset_path.exists():
        return False, 'missing grid or dataset.json'
    spec = json.loads(dataset_path.read_text())
    catch = spec.get('catch_rate')
    if catch is None or float(catch) < 0.98:
        return False, f'expert catch at 10 mm is {catch}; not training'
    return True, spec


def ceiling_record():
    path = RESULTS / 'stage2' / 'oneball_deepsets.json'
    recorded = json.loads(path.read_text())['deepsets']
    radius_path = RESULTS / 'stage2' / 'radius_curve.json'
    catch_50 = None
    if radius_path.exists():
        curve = json.loads(radius_path.read_text()).get('mean_curve', {})
        if '50' in curve:
            catch_50 = float(curve['50'])
    return dict(miss_mm=float(recorded['miss_mm']), comp_along=float(recorded['comp_along']),
                comp_cross=float(recorded['comp_cross']),
                base_offset_median_mm=float(recorded['base_offset_median_mm']),
                catch_50=catch_50, source=str(path))


def main():
    from variety_data import PROTECTED
    before = stamp(PROTECTED)
    ceiling = ceiling_record()
    if abs(ceiling['miss_mm'] - CEILING_MISS) > 1e-6:
        raise SystemExit('oneball_deepsets.json does not match the recorded 36.1 mm ceiling')
    old_hashes = set(json.loads((RESULTS / 'stage2' / 'oneball_deepsets.json').read_text())['hashes'].values())
    test = test_seeds()
    print(f'expert reference on {len(test)} test seeds', flush=True)
    experts = load_experts(test)
    report = dict(ceiling=ceiling, arms={})
    for name, tag, grid, dataset in ARMS:
        ok, info = arm_ready(dataset, grid)
        if not ok:
            print(f'skip {name}: {info}', flush=True)
            report['arms'][name] = dict(trained=False, reason=info)
            continue
        print(f'train {name} on {grid.name}', flush=True)
        train_arm(tag, grid)
        hashes = {}
        block = dict(trained=True, expert_catch_10mm=float(info['catch_rate']), hashes={})
        if name == 'planned':
            block['kept_fraction'] = info.get('kept_fraction')
            block['refuse_counts'] = info.get('refuse_counts')
        for seed in SEEDS:
            ckpt = MODELS / f'{tag}_s{seed}.pt'
            digest = weight_hash(ckpt)
            if digest in old_hashes:
                raise SystemExit(f'{ckpt.name} matches a ceiling checkpoint')
            hashes[f's{seed}'] = digest
            weights = assert_writable(ckpt.with_suffix('.npz'))
            from policy import export_numpy
            export_numpy(ckpt, weights)
            print(f'rollout {name} seed {seed}', flush=True)
            stats = eval_weights(weights, test, experts)
            block[f's{seed}'] = stats
            comp = stats['compensation']
            print(f"{name} s{seed}: catch10 {stats['catch_rate']:.3f}  catch50 {stats['catch_50']:.3f}  "
                  f"miss {stats['miss_mm']:.1f} mm  "
                  f"comp along {comp['along']['regression']:.2f}  cross {comp['cross']['regression']:.2f}",
                  flush=True)
        if len(set(hashes.values())) != 3:
            raise SystemExit(f'{name} checkpoints are not distinct')
        block['hashes'] = hashes
        block['summary'] = aggregate([block[f's{s}'] for s in SEEDS])
        report['arms'][name] = block
    goal = report['arms'].get('matched', {}).get('summary')
    planner = report['arms'].get('planned', {}).get('summary')
    report['reading'] = reading(goal, planner, ceiling['miss_mm'])
    after = stamp(PROTECTED)
    changed = [path for path, mark in before.items() if after.get(path) != mark]
    created = [path for path in after if path not in before]
    if changed or created:
        raise SystemExit('protected files changed: ' + ', '.join(changed + created))
    report['protected_unchanged'] = True
    assert_writable(COMPARE)
    COMPARE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['reading'], indent=2), flush=True)
    print(f'wrote {COMPARE}', flush=True)


if __name__ == '__main__':
    main()
