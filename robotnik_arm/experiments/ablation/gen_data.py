"""Generate D1 demos (1 kHz logs), the fixed test seeds, and the Stage 0 expert-rate check.

Training demos come from the expert acting every simulation step. Odd-indexed demos get
DART-style action noise with clean labels, so every nested prefix keeps a 50/50 mix.
Test seeds come from a different seed stream and are never used for training.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MASTER_SEED = 2026
EXPERT_RATES = (5, 10, 25, 1000)

_env = None


def init_worker():
    global _env
    os.environ['OMP_NUM_THREADS'] = '1'
    from catch_env import CatchEnv
    _env = CatchEnv()


def seed_stream(stream, count):
    return [int(np.random.SeedSequence([MASTER_SEED, stream, i]).generate_state(1, dtype=np.uint64)[0] >> 1)
            for i in range(count)]


def demo_job(index, seed, noise_seed, dart, path):
    from catch_env import ExpertDriver, DartNoise, rollout
    noise = DartNoise(noise_seed) if dart else None
    out = rollout(_env, ExpertDriver(), seed, 1000, noise=noise, log=True)
    arrays = out.pop('arrays')
    meta = dict(out, index=index, dart=dart, noise_seed=noise_seed, rate_hz=1000, sim_dt=_env.dt)
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, obs=arrays['obs'], cmd=arrays['cmd'], applied=arrays['applied'], ready=arrays['ready'],
             meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return 'demo', meta


def expert_job(rate, seed):
    from catch_env import ExpertDriver, rollout
    t0 = time.time()
    out = rollout(_env, ExpertDriver(), seed, rate)
    out.update(rate=rate, wall=time.time() - t0)
    return 'expert', out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', type=int, default=1000)
    parser.add_argument('--test', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    (DATA / 'demos').mkdir(parents=True, exist_ok=True)
    train_seeds = seed_stream(10, args.demos)
    noise_seeds = seed_stream(11, args.demos)
    test_seeds = seed_stream(20, args.test)
    assert not set(train_seeds) & set(test_seeds), 'train/test seed overlap'
    seeds_path = DATA / 'seeds.json'
    if seeds_path.exists():
        saved = json.loads(seeds_path.read_text())
        assert saved['test'] == test_seeds and saved['train'][:args.demos] == train_seeds, 'seed file mismatch'
    else:
        seeds_path.write_text(json.dumps(dict(master=MASTER_SEED, train=train_seeds, test=test_seeds), indent=1))
    checks_path = DATA / 'expert_rate_check.json'
    done_checks = json.loads(checks_path.read_text()) if checks_path.exists() else []
    done_keys = {(r['rate'], r['seed']) for r in done_checks}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futures = []
        # Expert checks first: they decide which rates survive.
        for rate in EXPERT_RATES:
            for seed in test_seeds:
                if (rate, seed) not in done_keys:
                    futures.append(pool.submit(expert_job, rate, seed))
        for i, (seed, nseed) in enumerate(zip(train_seeds, noise_seeds)):
            path = DATA / 'demos' / f'demo_{i:04d}.npz'
            if not path.exists():
                futures.append(pool.submit(demo_job, i, seed, nseed, bool(i % 2), path))
        checks, demos = list(done_checks), []
        for n, fut in enumerate(as_completed(futures), 1):
            kind, out = fut.result()
            (checks if kind == 'expert' else demos).append(out)
            if kind == 'expert' and len(checks) % 50 == 0:
                checks_path.write_text(json.dumps(checks))
            if n % 100 == 0:
                print(f'[{n}/{len(futures)}] {time.time() - t0:.0f}s', flush=True)
    checks_path.write_text(json.dumps(checks))
    print(f'done in {time.time() - t0:.0f}s')
    for rate in EXPERT_RATES:
        rows = [r for r in checks if r['rate'] == rate]
        if rows:
            print(f"expert {rate:4d} Hz: catch {np.mean([r['success'] for r in rows]):.3f} "
                  f"miss {np.mean([r['min_dist'] for r in rows]) * 1000:.2f} mm  n={len(rows)}")
    if demos:
        print(f"demos: {len(demos)} new, success {np.mean([d['success'] for d in demos]):.3f}")


if __name__ == '__main__':
    main()
