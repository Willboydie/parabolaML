"""Train the Stage 1 winner on the learner-noise demos, 5-fold 85/15.

One shuffle (seed 2026). Fold k holds out a disjoint block of 150 demos (15%)
and trains on the other 850. The same architecture and training seed as the
winner (25 Hz, 2 s chunk, execute 1 s, 0.5 s stack, seed 1). Each fold is
rolled out on its 150 held-out scene seeds, and on the original 200 test seeds.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

from config import Config
from run_ablation import EVAL_CHUNK, eval_chunk, init_worker

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'learner_dart'
RESULTS = ROOT / 'results'
N_FOLDS = 5
TEST_FRAC = 0.15
SPLIT_SEED = 2026
TRAIN_SEED = 1
WINNER = dict(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5)


def splits(n):
    """Five disjoint 15% test blocks. The last 250 demos stay in every train set."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_test = int(round(n * TEST_FRAC))
    out = []
    for k in range(N_FOLDS):
        test = np.sort(perm[k * n_test:(k + 1) * n_test])
        train = np.setdiff1d(np.arange(n), test)
        out.append((train, test))
    return out


def write_train_grid(train_ids, path):
    if path.exists():
        return
    src = DATA / 'grid50.npz'
    with np.load(src) as z:
        obs, cmd, starts, lengths = z['obs'], z['cmd'], z['starts'], z['lengths']
        q_col = int(z['arm_q_col'])
        pieces_o, pieces_c, demo_i, new_starts, new_lengths = [], [], [], [], []
        offset = 0
        for new_i, old_i in enumerate(train_ids):
            s, length = int(starts[old_i]), int(lengths[old_i])
            pieces_o.append(obs[s:s + length])
            pieces_c.append(cmd[s:s + length])
            demo_i.append(np.full(length, new_i, dtype=np.int32))
            new_starts.append(offset)
            new_lengths.append(length)
            offset += length
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, obs=np.concatenate(pieces_o), cmd=np.concatenate(pieces_c),
             demo=np.concatenate(demo_i), starts=np.array(new_starts), lengths=np.array(new_lengths),
             arm_q_col=np.int64(q_col))
    tmp.replace(path)


def slim(ep):
    dist = ep['min_dist']
    dist = None if dist is None or not np.isfinite(dist) else float(dist)
    ct = ep['catch_time']
    return dict(seed=int(ep['seed']), success=bool(ep['success']), min_dist=dist,
                catch_time=None if ct is None else float(ct))


def summarize(episodes):
    dists = [e['min_dist'] for e in episodes if e['min_dist'] is not None]
    return dict(n=len(episodes), n_finite=len(dists),
                catch_rate=float(np.mean([e['success'] for e in episodes])),
                miss_mm=float(np.mean(dists) * 1000) if dists else None,
                median_miss_mm=float(np.median(dists) * 1000) if dists else None)


def eval_seeds(pool, weights, seeds):
    futs = [pool.submit(eval_chunk, str(weights), list(map(int, seeds[i:i + EVAL_CHUNK])), 0.01)
            for i in range(0, len(seeds), EVAL_CHUNK)]
    episodes = []
    for fut in as_completed(futs):
        rows, _wall = fut.result()
        episodes.extend(slim(r) for r in rows)
    return episodes


def main():
    spec = json.loads((DATA / 'dataset.json').read_text())
    scene_seeds = spec['seeds']
    fold_splits = splits(len(scene_seeds))
    heldout = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:200]
    (RESULTS / 'models').mkdir(parents=True, exist_ok=True)
    print(f'{N_FOLDS} folds, {len(fold_splits[0][0])} train / {len(fold_splits[0][1])} test, '
          f'train seed {TRAIN_SEED}', flush=True)

    pending = []
    for k, (train_ids, _test_ids) in enumerate(fold_splits):
        grid = DATA / f'fold{k}_train.npz'
        ckpt = RESULTS / 'models' / f'learner_dart_fold{k}_s{TRAIN_SEED}.pt'
        print(f'fold {k}: writing train grid', flush=True)
        write_train_grid(train_ids, grid)
        if ckpt.exists():
            print(f'fold {k}: reusing {ckpt.name}', flush=True)
        else:
            pending.append(k)
    # Two GPU jobs at a time, matching the ablation runner's limit of a few.
    running = {}
    while pending or running:
        while pending and len(running) < 2:
            k = pending.pop(0)
            grid = DATA / f'fold{k}_train.npz'
            ckpt = RESULTS / 'models' / f'learner_dart_fold{k}_s{TRAIN_SEED}.pt'
            log_path = RESULTS / 'logs' / f'learner_dart_fold{k}.train.log'
            cfg = Config(**WINNER, demos=850)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
            cmd = [sys.executable, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                   '--seed', str(TRAIN_SEED), '--out', str(ckpt), '--data', str(grid), '--steps', '20000']
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, 'w')
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[proc] = (k, log, time.time())
            print(f'fold {k}: training', flush=True)
        finished = [p for p in running if p.poll() is not None]
        if not finished:
            time.sleep(2.0)
            continue
        for proc in finished:
            k, log, t0 = running.pop(proc)
            log.close()
            if proc.returncode:
                raise RuntimeError(f'fold {k} train failed, see results/logs/learner_dart_fold{k}.train.log')
            print(f'fold {k}: trained in {time.time() - t0:.0f}s', flush=True)

    from policy import export_numpy
    t0 = time.time()
    fold_rows = []
    with ProcessPoolExecutor(16, initializer=init_worker) as pool:
        for k, (train_ids, test_ids) in enumerate(fold_splits):
            ckpt = RESULTS / 'models' / f'learner_dart_fold{k}_s{TRAIN_SEED}.pt'
            weights = ckpt.with_suffix('.npz')
            meta = export_numpy(ckpt, weights)
            test_seeds = [scene_seeds[i] for i in test_ids]
            print(f'fold {k}: eval {len(test_seeds)} held-out + {len(heldout)} stage-1 test', flush=True)
            held = eval_seeds(pool, weights, test_seeds)
            stage1 = eval_seeds(pool, weights, heldout)
            row = dict(fold=k, train_ids=train_ids.astype(int).tolist(),
                       test_ids=test_ids.astype(int).tolist(),
                       train_time_s=meta['train_time_s'], final_loss=meta['final_loss'],
                       train_rows=meta['train_rows'],
                       heldout=summarize(held), stage1_test=summarize(stage1),
                       heldout_episodes=held, stage1_episodes=stage1)
            fold_rows.append(row)
            print(f"fold {k}: held-out catch {row['heldout']['catch_rate']:.3f} "
                  f"miss {row['heldout']['miss_mm']:.1f} mm | "
                  f"stage-1 test catch {row['stage1_test']['catch_rate']:.3f} "
                  f"miss {row['stage1_test']['miss_mm']:.1f} mm", flush=True)

    def fold_record(row):
        kept = {k: v for k, v in row.items() if not k.endswith('episodes')}
        kept['heldout_median_miss_mm'] = row['heldout']['median_miss_mm']
        kept['stage1_median_miss_mm'] = row['stage1_test']['median_miss_mm']
        return kept

    def across(key, field):
        vals = np.array([r[key][field] for r in fold_rows], dtype=float)
        return dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                    min=float(vals.min()), max=float(vals.max()))

    report = dict(weights_seed=TRAIN_SEED, split_seed=SPLIT_SEED, n_folds=N_FOLDS,
                  test_frac=TEST_FRAC, config={**WINNER, 'demos': 850},
                  heldout_catch=across('heldout', 'catch_rate'),
                  heldout_miss_mm=across('heldout', 'miss_mm'),
                  heldout_median_miss_mm=float(np.median(
                      [e['min_dist'] for r in fold_rows for e in r['heldout_episodes'] if e['min_dist'] is not None]) * 1000),
                  stage1_catch=across('stage1_test', 'catch_rate'),
                  stage1_miss_mm=across('stage1_test', 'miss_mm'),
                  stage1_median_miss_mm=float(np.median(
                      [e['min_dist'] for r in fold_rows for e in r['stage1_episodes'] if e['min_dist'] is not None]) * 1000),
                  folds=[fold_record(r) for r in fold_rows],
                  elapsed_s=time.time() - t0)
    # Keep per-episode records beside the summary, not inside the small folds list.
    out = RESULTS / 'learner_dart_cv.json'
    out.write_text(json.dumps(report))
    (RESULTS / 'learner_dart_cv_episodes.json').write_text(json.dumps([
        dict(fold=r['fold'], heldout=r['heldout_episodes'], stage1=r['stage1_episodes']) for r in fold_rows]))
    h, s = report['heldout_catch'], report['stage1_catch']
    hm, sm = report['heldout_miss_mm'], report['stage1_miss_mm']
    print(f"held-out 15%: catch {h['mean']:.3f} ± {h['sd']:.3f}  "
          f"miss {hm['mean']:.1f} ± {hm['sd']:.1f} mm  median {report['heldout_median_miss_mm']:.1f} mm", flush=True)
    print(f"stage-1 200:  catch {s['mean']:.3f} ± {s['sd']:.3f}  "
          f"miss {sm['mean']:.1f} ± {sm['sd']:.1f} mm  median {report['stage1_median_miss_mm']:.1f} mm", flush=True)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
