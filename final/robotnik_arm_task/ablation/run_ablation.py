"""Stage 1 ablation runner: train on the GPU, evaluate rollouts on 16 CPU worker processes.

Each (config, training seed) is trained once, saved under results/models, then evaluated
on the same 200 test seeds. Configs shared by several stages (the baseline appears in
A-D) are run once. Finished runs are skipped on restart.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from config import BASELINE, Config

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
TRAIN_SEEDS = (0, 1, 2)
EVAL_CHUNK = 25


def stage_configs(excluded_rates=()):
    b = BASELINE
    stages = {
        'A': [replace(b, rate_hz=r) for r in (5, 10, 25) if r not in excluded_rates],
        'B': [replace(b, predict_s=p, execute_frac=f) for p in (0.5, 2.0, 5.0) for f in (0.25, 0.5)],
        'C': [b, replace(b, history='stack', history_s=0.5), replace(b, history='stack', history_s=2.0),
              replace(b, history='gru', history_s=2.0)],
        'D': [replace(b, demos=n) for n in (125, 250, 500, 1000)],
    }
    return stages


_env = None
_models = {}


def init_worker():
    """Rollout workers never import torch; NumPy/BLAS are limited to one thread."""
    global _env
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def eval_chunk(weights, seeds, radius):
    from catch_env import rollout
    from np_policy import NumpyPolicy, PolicyDriver
    if weights not in _models:
        _models.clear()
        _models[weights] = NumpyPolicy(weights)
    model = _models[weights]
    t0 = time.time()
    rows = []
    for seed in seeds:
        out = rollout(_env, PolicyDriver(model), seed, model.cfg.rate_hz, radius=radius)
        rows.append(out)
    return rows, time.time() - t0


def run_path(cfg, seed):
    return RESULTS / 'runs' / f'{cfg.name}_s{seed}.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', default='A,B,C,D')
    parser.add_argument('--configs', type=Path, help='JSON list of extra Config dicts (Stage E)')
    parser.add_argument('--exclude-rates', default='')
    parser.add_argument('--seeds', default=','.join(map(str, TRAIN_SEEDS)))
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--gpu-jobs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--radius', type=float, default=0.01)
    parser.add_argument('--test-limit', type=int, default=None)
    parser.add_argument('--data', type=Path, default=DATA / 'grid50.npz')
    parser.add_argument('--skip-configs', type=Path, help='JSON list of Config dicts handled by another runner')
    args = parser.parse_args()
    excluded = [int(r) for r in args.exclude_rates.split(',') if r]
    configs = []
    if args.stages != 'none':
        stages = stage_configs(excluded)
        for s in args.stages.split(','):
            configs += stages[s]
    if args.configs:
        configs += [Config(**c) for c in json.loads(args.configs.read_text())]
    configs = list(dict.fromkeys(configs))
    if args.skip_configs:
        skip = {Config(**c) for c in json.loads(args.skip_configs.read_text())}
        configs = [c for c in configs if c not in skip]
    seeds = [int(s) for s in args.seeds.split(',')]
    test_seeds = json.loads((DATA / 'seeds.json').read_text())['test'][:args.test_limit]
    jobs = deque((cfg, seed) for seed in seeds for cfg in configs if not run_path(cfg, seed).exists())
    print(f'{len(configs)} unique configs, {len(jobs)} runs to do', flush=True)
    (RESULTS / 'models').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'runs').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'logs').mkdir(parents=True, exist_ok=True)

    from policy import export_numpy

    t_start = time.time()
    training = {}
    evaluating = {}
    py = sys.executable
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')

    def submit_eval(pool, cfg, seed, ckpt, t0):
        weights = ckpt.with_suffix('.npz')
        meta = export_numpy(ckpt, weights)
        futs = [pool.submit(eval_chunk, str(weights), test_seeds[i:i + EVAL_CHUNK], args.radius)
                for i in range(0, len(test_seeds), EVAL_CHUNK)]
        evaluating[(cfg, seed)] = dict(futs=futs, meta=meta, t_train_done=time.time(), t_train_start=t0)

    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        while jobs or training or evaluating:
            while jobs and len(training) < args.gpu_jobs:
                cfg, seed = jobs.popleft()
                ckpt = RESULTS / 'models' / f'{cfg.name}_s{seed}.pt'
                if ckpt.exists():
                    try:
                        submit_eval(pool, cfg, seed, ckpt, time.time())
                        print(f'[{time.time() - t_start:6.0f}s] reusing trained {cfg.name} s{seed}', flush=True)
                        continue
                    except Exception as exc:
                        print(f'retraining {cfg.name} s{seed}: unreadable checkpoint ({exc})', flush=True)
                log = open(RESULTS / 'logs' / f'{cfg.name}_s{seed}.train.log', 'w')
                cmd = [py, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()), '--seed', str(seed),
                       '--out', str(ckpt), '--steps', str(args.steps), '--data', str(args.data)]
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                training[proc] = (cfg, seed, ckpt, log, time.time())
            for proc in [p for p in training if p.poll() is not None]:
                cfg, seed, ckpt, log, t0 = training.pop(proc)
                log.close()
                if proc.returncode:
                    print(f'TRAIN FAILED {cfg.name} s{seed}; see logs', flush=True)
                    continue
                submit_eval(pool, cfg, seed, ckpt, t0)
                print(f'[{time.time() - t_start:6.0f}s] trained {cfg.name} s{seed} in {time.time() - t0:.0f}s', flush=True)
            for key in [k for k, v in evaluating.items() if all(f.done() for f in v['futs'])]:
                cfg, seed = key
                info = evaluating.pop(key)
                episodes, cpu = [], 0.0
                for f in info['futs']:
                    rows, wall = f.result()
                    episodes += rows
                    cpu += wall
                meta = info['meta']
                result = dict(config=cfg.to_dict(), name=cfg.name, seed=seed, radius=args.radius,
                              catch_rate=float(np.mean([e['success'] for e in episodes])),
                              miss_mm=float(np.mean([e['min_dist'] for e in episodes]) * 1000),
                              n_episodes=len(episodes), train_time_s=meta['train_time_s'], final_loss=meta['final_loss'],
                              eval_cpu_s=cpu, eval_wall_s=time.time() - info['t_train_done'],
                              run_wall_s=time.time() - info['t_train_start'], episodes=episodes)
                run_path(cfg, seed).write_text(json.dumps(result))
                print(f"[{time.time() - t_start:6.0f}s] {cfg.name} s{seed}: catch {result['catch_rate']:.3f} "
                      f"miss {result['miss_mm']:.1f} mm (train {meta['train_time_s']:.0f}s, eval wall "
                      f"{result['eval_wall_s']:.0f}s, eval cpu {cpu:.0f}s)", flush=True)
            time.sleep(1.0)
    print(f'all done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
