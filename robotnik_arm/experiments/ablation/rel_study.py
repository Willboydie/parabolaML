"""Flat rel_add ablation. No DeepSets encoder is trained, loaded, or cited.

The factor grid is fixed in this file. The sweep does not start until
results/rel_study/hypotheses.json exists, and it does not read those
predictions when it chooses configs. The winner is rank() from report.py:
higher mean catch rate at 10 mm, and lower mean miss when the catch
intervals overlap.

Phases: baseline, analyze, sweep, combine, dart, compare, all.
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

from config import Config
from report import overlap, rank, summarize
from run_ablation import EVAL_CHUNK, eval_chunk, init_worker

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
STUDY = ROOT / 'results' / 'rel_study'
MODELS = STUDY / 'models'
RUNS = STUDY / 'runs'
LOGS = STUDY / 'logs'
GRID50 = DATA / 'grid50.npz'
OFFSET_GRID = DATA / 'grid50_offset.npz'
DART_GRID = DATA / 'grid50_rel_dart.npz'
DART_DEMOS = DATA / 'rel_study_dart'
HYPOTHESES = STUDY / 'hypotheses.json'
WINNER_PATH = STUDY / 'winner.json'
SEEDS = (0, 1, 2)
FACTORS = ('rate', 'history', 'predict_horizon', 'execute_window', 'short_chunk')
FAMILY_FIELDS = {
    'rate': ('rate_hz',),
    'history': ('history', 'history_s'),
    'timing': ('predict_s', 'execute_frac'),
}
ALLOWED_DATA = {
    'baseline': GRID50,
    'sweep': GRID50,
    'combination': GRID50,
    'dart': DART_GRID,
    'offset': OFFSET_GRID,
}


def baseline_config():
    return Config(rate_hz=10, predict_s=2.0, execute_frac=0.5, history='none', history_s=0.0,
                  demos=1000, features='rel_add', encoder='flat')


def factor_grid(base=None):
    """One-factor levels. The baseline itself is not repeated here."""
    b = baseline_config() if base is None else base
    return {
        'rate': [replace(b, rate_hz=r) for r in (5, 25, 50)],
        'history': [replace(b, history='stack', history_s=0.5),
                    replace(b, history='stack', history_s=2.0),
                    replace(b, history='gru', history_s=2.0)],
        'predict_horizon': [replace(b, predict_s=1.0, execute_frac=1.0),
                            replace(b, predict_s=5.0, execute_frac=0.2)],
        'execute_window': [replace(b, execute_frac=0.25)],
        'short_chunk': [replace(b, predict_s=0.5, execute_frac=0.5),
                        replace(b, predict_s=0.5, execute_frac=0.25)],
    }


def assert_flat(cfg):
    if cfg.encoder != 'flat' or cfg.features != 'rel_add' or 'deepsets' in cfg.name:
        raise SystemExit(f'refusing {cfg.name}: encoder={cfg.encoder} features={cfg.features}')
    if cfg.rate_hz not in (5, 10, 25, 50):
        raise SystemExit(f'{cfg.name} rate {cfg.rate_hz} is not on the 50 Hz grid used here')
    return cfg


def hypotheses_ready():
    if not HYPOTHESES.exists():
        raise SystemExit(f'sweep refused: {HYPOTHESES} does not exist. Analyse the baseline first.')
    doc = json.loads(HYPOTHESES.read_text())
    missing = [k for k in FACTORS if k not in doc.get('factors', {})]
    if missing:
        raise SystemExit(f'sweep refused: hypotheses.json has no prediction for {missing}')
    return doc


def run_path(role, cfg, seed):
    return RUNS / role / f'{cfg.name}_s{seed}.json'


def ckpt_path(role, cfg, seed):
    return MODELS / role / f'{cfg.name}_s{seed}.pt'


def write_grid_snapshot():
    STUDY.mkdir(parents=True, exist_ok=True)
    base = baseline_config()
    grid = factor_grid(base)
    doc = {
        'baseline': base.to_dict(),
        'factors': {name: [c.to_dict() for c in cfgs] for name, cfgs in grid.items()},
        'ranking': 'mean catch rate at 10 mm; if mean +/- seed sd intervals overlap, lower mean miss',
        'encoder': 'flat',
        'features': 'rel_add',
        'data': str(GRID50),
    }
    (STUDY / 'grid.json').write_text(json.dumps(doc, indent=2))


def load_role(role):
    runs = {}
    folder = RUNS / role
    if not folder.exists():
        return runs
    for path in sorted(folder.glob('*.json')):
        row = json.loads(path.read_text())
        runs.setdefault(row['name'], []).append(row)
    return runs


def train_role(role, jobs, data, steps, gpu_jobs, workers, radius, test_limit):
    """jobs: list of (factor_or_none, cfg). Trains every seed whose run file is missing."""
    data = Path(data).resolve()
    allowed = ALLOWED_DATA[role].resolve()
    if data != allowed:
        raise SystemExit(f'{role} refused data {data}; only {allowed} is allowed')
    if not data.exists():
        raise SystemExit(f'missing {data}')
    for _factor, cfg in jobs:
        assert_flat(cfg)
    test_seeds = json.loads((DATA / 'seeds.json').read_text())['test'][:test_limit]
    pending = deque()
    for factor, cfg in jobs:
        for seed in SEEDS:
            if not run_path(role, cfg, seed).exists():
                pending.append((factor, cfg, seed))
    print(f'{role}: {len(jobs)} configs, {len(pending)} runs to do, data {data.name}', flush=True)
    if not pending:
        return
    for folder in (MODELS / role, RUNS / role, LOGS / role):
        folder.mkdir(parents=True, exist_ok=True)
    from policy import export_numpy
    t_start = time.time()
    training, evaluating = {}, {}
    py = sys.executable
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')

    def submit_eval(pool, factor, cfg, seed, ckpt, t0):
        weights = ckpt.with_suffix('.npz')
        meta = export_numpy(ckpt, weights)
        if NumpyGuard(weights):
            pass
        futs = [pool.submit(eval_chunk, str(weights), test_seeds[i:i + EVAL_CHUNK], radius)
                for i in range(0, len(test_seeds), EVAL_CHUNK)]
        evaluating[(cfg.name, seed)] = dict(
            futs=futs, meta=meta, cfg=cfg, factor=factor, seed=seed,
            t_train_done=time.time(), t_train_start=t0)

    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        while pending or training or evaluating:
            while pending and len(training) < gpu_jobs:
                factor, cfg, seed = pending.popleft()
                ckpt = ckpt_path(role, cfg, seed)
                if ckpt.exists():
                    try:
                        submit_eval(pool, factor, cfg, seed, ckpt, time.time())
                        print(f'[{time.time() - t_start:6.0f}s] reusing {ckpt.name}', flush=True)
                        continue
                    except Exception as exc:
                        print(f'retraining {ckpt.name}: unreadable checkpoint ({exc})', flush=True)
                log = open(LOGS / role / f'{cfg.name}_s{seed}.train.log', 'w')
                cmd = [py, str(ROOT / 'train.py'), '--config', json.dumps(cfg.to_dict()),
                       '--seed', str(seed), '--out', str(ckpt), '--steps', str(steps), '--data', str(data)]
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                training[proc] = (factor, cfg, seed, ckpt, log, time.time())
            for proc in [p for p in training if p.poll() is not None]:
                factor, cfg, seed, ckpt, log, t0 = training.pop(proc)
                log.close()
                if proc.returncode:
                    print(f'TRAIN FAILED {cfg.name} s{seed}; see {LOGS / role}', flush=True)
                    continue
                submit_eval(pool, factor, cfg, seed, ckpt, t0)
                print(f'[{time.time() - t_start:6.0f}s] trained {cfg.name} s{seed}', flush=True)
            done = [k for k, v in evaluating.items() if all(f.done() for f in v['futs'])]
            for key in done:
                info = evaluating.pop(key)
                cfg, seed, factor = info['cfg'], info['seed'], info['factor']
                episodes, cpu = [], 0.0
                for fut in info['futs']:
                    rows, wall = fut.result()
                    episodes += rows
                    cpu += wall
                meta = info['meta']
                result = dict(
                    config=cfg.to_dict(), name=cfg.name, seed=seed, radius=radius,
                    catch_rate=float(np.mean([e['success'] for e in episodes])),
                    miss_mm=float(np.mean([e['min_dist'] for e in episodes]) * 1000),
                    n_episodes=len(episodes), train_time_s=meta['train_time_s'], final_loss=meta['final_loss'],
                    eval_cpu_s=cpu, eval_wall_s=time.time() - info['t_train_done'],
                    run_wall_s=time.time() - info['t_train_start'], episodes=episodes,
                    study_role=role, study_factor=factor, data=str(data))
                run_path(role, cfg, seed).write_text(json.dumps(result))
                print(f"[{time.time() - t_start:6.0f}s] {role} {cfg.name} s{seed}: "
                      f"catch {result['catch_rate']:.3f} miss {result['miss_mm']:.1f} mm", flush=True)
            time.sleep(1.0)
    print(f'{role} done in {time.time() - t_start:.0f}s', flush=True)


class NumpyGuard:
    """Reject a DeepSets export before it is rolled out."""

    def __init__(self, weights):
        from np_policy import NumpyPolicy
        cfg = NumpyPolicy(weights).cfg
        assert_flat(cfg)


def scored(configs, runs):
    rows = []
    for cfg in configs:
        row = summarize(cfg, runs)
        if row and row['n'] == len(SEEDS):
            rows.append(row)
    return rows


def beats(level, base_row, runs):
    """True when rank() places this level ahead of the baseline."""
    row = summarize(level, runs)
    if row is None or row['n'] != len(SEEDS):
        return False
    return rank([row, base_row])[0]['cfg'] == level


def choose_combination(runs):
    """Top two families that beat the baseline, their top two levels, plus the best guess.

    Timing levels share predict_s and execute_frac, so horizon, execute window and
    the short chunk are one family. Rate and history are the other two.
    """
    base = baseline_config()
    grid = factor_grid(base)
    base_row = summarize(base, runs)
    if base_row is None or base_row['n'] != len(SEEDS):
        raise SystemExit('baseline is not fully scored')
    families = {
        'rate': grid['rate'],
        'history': grid['history'],
        'timing': grid['predict_horizon'] + grid['execute_window'] + grid['short_chunk'],
    }
    moving = {}
    for name, levels in families.items():
        present = scored([base] + levels, runs)
        if len(present) != 1 + len(levels):
            raise SystemExit(f'{name} is missing runs')
        order = rank(present)
        if order[0]['cfg'] == base:
            continue
        best = max(s['catch'] for s in present)
        worst = min(s['catch'] for s in present)
        resolved = not overlap(max(present, key=lambda s: s['catch']), min(present, key=lambda s: s['catch']))
        miss_range = max(s['miss'] for s in present) - min(s['miss'] for s in present)
        moving[name] = dict(order=order, resolved=resolved, catch=best - worst, miss=miss_range)
    if not moving:
        return [], base
    effect = lambda name: (moving[name]['catch'] if moving[name]['resolved'] else 0.0, moving[name]['miss'])
    movers = sorted(moving, key=effect, reverse=True)[:2]
    top2 = {}
    for name in movers:
        top2[name] = [s['cfg'] for s in moving[name]['order'][:2]]
    configs = []
    if len(movers) == 1:
        configs.extend(top2[movers[0]])
    else:
        for left in top2[movers[0]]:
            for right in top2[movers[1]]:
                cfg = base
                for name, src in ((movers[0], left), (movers[1], right)):
                    cfg = replace(cfg, **{field: getattr(src, field) for field in FAMILY_FIELDS[name]})
                configs.append(cfg)
    guess = base
    for name in moving:
        src = moving[name]['order'][0]['cfg']
        guess = replace(guess, **{field: getattr(src, field) for field in FAMILY_FIELDS[name]})
    configs.append(guess)
    configs = [assert_flat(c) for c in list(dict.fromkeys(configs)) if c != base]
    return configs, None


def overall_winner(runs):
    base = baseline_config()
    configs = [base]
    for levels in factor_grid(base).values():
        configs.extend(levels)
    comb_path = STUDY / 'combination_configs.json'
    if comb_path.exists():
        configs.extend(Config(**row) for row in json.loads(comb_path.read_text()))
    seen = set()
    unique = []
    for cfg in configs:
        if cfg.name not in seen:
            seen.add(cfg.name)
            unique.append(cfg)
    configs = unique
    rows = scored(configs, runs)
    if not rows:
        raise SystemExit('no scored configs')
    return rank(rows)[0]


def write_winner(row):
    doc = dict(config=row['cfg'].to_dict(), name=row['cfg'].name,
               catch=row['catch'], catch_sd=row['catch_sd'], miss=row['miss'], miss_sd=row['miss_sd'],
               median_miss=row['median_miss'], n=row['n'])
    WINNER_PATH.write_text(json.dumps(doc, indent=2))
    print(f"winner {doc['name']}: catch {doc['catch']:.3f} miss {doc['miss']:.1f} mm", flush=True)
    return doc


def expected_sweep_jobs():
    grid = factor_grid()
    jobs = [(None, baseline_config())]
    for name, levels in grid.items():
        jobs.extend((name, cfg) for cfg in levels)
    return jobs


def require_complete(role, jobs):
    missing = [f'{cfg.name} s{seed}' for _f, cfg in jobs for seed in SEEDS
               if not run_path(role, cfg, seed).exists()]
    if missing:
        raise SystemExit(f'{role} incomplete, missing {len(missing)}: {missing[:6]}')


def offset_seeds():
    spec = DATA / 'offset_parking' / 'dataset.json'
    if not spec.exists():
        raise SystemExit(f'missing {spec}')
    seeds = json.loads(spec.read_text())['seeds']
    if len(seeds) < 1000:
        raise SystemExit(f'offset dataset has {len(seeds)} seeds')
    test = set(json.loads((DATA / 'seeds.json').read_text())['test'])
    if set(seeds[:1000]) & test:
        raise SystemExit('offset training seeds overlap the test set')
    path = STUDY / 'offset_seeds.json'
    path.write_text(json.dumps(seeds[:1000]))
    return path


def fill_tau(report):
    sigma = [float(v) for v in report['noise']['sigma']]
    tau = report['noise']['tau_c_s']
    known = [float(t) for t in tau if t is not None]
    fill = float(np.median(known)) if known else 0.05
    out = [fill if t is None else float(t) for t in tau]
    return dict(noise=dict(sigma=sigma, tau_c_s=out)), fill, sum(t is None for t in tau)


def build_dart_grid():
    """Original 1,000 demos plus the 1,000 high-amplitude DART demos. grid50.npz is not modified."""
    from catch_env import FIRST_TICK_STEP
    if DART_GRID.exists():
        with np.load(DART_GRID) as z:
            if len(z['starts']) == 2000:
                print(f'reusing {DART_GRID}', flush=True)
                return
    stride = 20
    with np.load(GRID50) as z:
        n_rows = int(z['starts'][999] + z['lengths'][999])
        obs = [np.array(z['obs'][:n_rows], copy=True)]
        cmd = [np.array(z['cmd'][:n_rows], copy=True)]
        demo = [np.array(z['demo'][:n_rows], copy=True)]
        starts = [int(v) for v in z['starts'][:1000]]
        lengths = [int(v) for v in z['lengths'][:1000]]
        q_col = int(z['arm_q_col'])
    if int(demo[0].max()) != 999 or len(starts) != 1000:
        raise SystemExit('grid50.npz is not the original 1,000 demos')
    offset = int(starts[-1] + lengths[-1])
    for i in range(1000):
        path = DART_DEMOS / f'demo_{i:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta['index'] != i or not meta.get('dart'):
                raise SystemExit(f'{path.name} is not high-amplitude DART demo {i}')
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        obs.append(o)
        cmd.append(c)
        demo.append(np.full(len(o), 1000 + i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
    np.savez(DART_GRID, obs=np.concatenate(obs), cmd=np.concatenate(cmd), demo=np.concatenate(demo),
             starts=np.asarray(starts), lengths=np.asarray(lengths), arm_q_col=np.int64(q_col))
    print(f'wrote {DART_GRID} ({offset} rows)', flush=True)


def phase_baseline(args):
    write_grid_snapshot()
    train_role('sweep', [(None, baseline_config())], GRID50, args.steps, args.gpu_jobs, args.workers, args.radius, None)


def phase_analyze(args):
    base = baseline_config()
    weights = ckpt_path('sweep', base, 1).with_suffix('.npz')
    if not weights.exists():
        raise SystemExit(f'missing baseline weights {weights}')
    import rel_baseline_analysis
    rel_baseline_analysis.run(weights, workers=args.workers)


def phase_sweep(args):
    hypotheses_ready()
    jobs = expected_sweep_jobs()
    train_role('sweep', jobs, GRID50, args.steps, args.gpu_jobs, args.workers, args.radius, None)
    require_complete('sweep', jobs)


def phase_combine(args):
    hypotheses_ready()
    require_complete('sweep', expected_sweep_jobs())
    runs = load_role('sweep')
    configs, early = choose_combination(runs)
    (STUDY / 'combination_configs.json').write_text(json.dumps([c.to_dict() for c in configs], indent=2))
    print(f'combination: {len(configs)} configs', flush=True)
    if configs:
        train_role('combination', [(None, c) for c in configs], GRID50, args.steps, args.gpu_jobs, args.workers, args.radius, None)
        require_complete('combination', [(None, c) for c in configs])
        runs.update(load_role('combination'))
    elif early is not None:
        pass
    row = overall_winner(runs if configs else load_role('sweep'))
    # overall_winner reloads combination from disk; merge sweep runs if combination was empty.
    if not configs:
        row = overall_winner(load_role('sweep'))
    else:
        merged = load_role('sweep')
        merged.update(load_role('combination'))
        # names are unique across these roles; a shared name would be a bug.
        row = overall_winner(merged)
    write_winner(row)


def phase_dart(args):
    if not WINNER_PATH.exists():
        raise SystemExit('no winner.json; run combine first')
    winner = json.loads(WINNER_PATH.read_text())
    cfg = assert_flat(Config(**winner['config']))
    weights = None
    for seed in (1, 0, 2):
        path = ckpt_path('sweep', cfg, seed).with_suffix('.npz')
        if not path.exists():
            path = ckpt_path('combination', cfg, seed).with_suffix('.npz')
        if path.exists():
            weights = path
            break
    if weights is None:
        raise SystemExit(f'no weights for winner {cfg.name}')
    noise_out = STUDY / 'learner_noise.json'
    if not noise_out.exists():
        cmd = [sys.executable, str(ROOT / 'measure_learner_noise.py'), '--weights', str(weights),
               '--episodes', '50', '--workers', str(args.workers), '--out', str(noise_out)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    report = json.loads(noise_out.read_text())
    usable, fill, n_missing = fill_tau(report)
    usable['tau_fill_s'] = fill
    usable['tau_missing'] = n_missing
    usable['weights'] = str(weights)
    dart_noise = STUDY / 'dart_noise.json'
    dart_noise.write_text(json.dumps(usable))
    seeds = offset_seeds()
    cmd = [sys.executable, str(ROOT / 'gen_learner_dart.py'), '--episodes', '1000',
           '--workers', str(args.workers), '--rate', str(cfg.rate_hz), '--out', str(DART_DEMOS),
           '--noise', str(dart_noise), '--seed-list', str(seeds)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    build_dart_grid()
    dart_cfg = assert_flat(replace(cfg, demos=2000))
    train_role('dart', [(None, dart_cfg)], DART_GRID, args.steps, args.gpu_jobs, args.workers, args.radius, None)
    train_role('offset', [(None, dart_cfg)], OFFSET_GRID, args.steps, args.gpu_jobs, args.workers, args.radius, None)
    require_complete('dart', [(None, dart_cfg)])
    require_complete('offset', [(None, dart_cfg)])


def _factor_rows(runs, hypo):
    base = baseline_config()
    grid = factor_grid(base)
    base_row = summarize(base, runs)
    lines = []
    flags = {}
    for name in FACTORS:
        levels = grid[name]
        present = scored([base] + levels, runs)
        order = rank(present)
        pred = hypo['factors'][name]
        best = order[0]
        confirmed = None
        if pred['prediction'] == 'not_indicated':
            outcome = 'no prediction'
        elif pred['prediction'] == 'higher_rate_lowers_miss':
            by_rate = {c.rate_hz: summarize(c, runs) for c in [base] + levels}
            ok = (rank([by_rate[25], by_rate[10]])[0]['cfg'].rate_hz == 25
                  and rank([by_rate[50], by_rate[10]])[0]['cfg'].rate_hz == 50
                  and rank([by_rate[5], by_rate[10]])[0]['cfg'].rate_hz == 10)
            confirmed = bool(ok)
            outcome = 'confirmed' if ok else 'refuted'
        elif pred['prediction'] == 'stack_0.5s_lowers_miss':
            stack = next(c for c in levels if c.history == 'stack' and c.history_s == 0.5)
            confirmed = rank([summarize(stack, runs), base_row])[0]['cfg'] == stack
            outcome = 'confirmed' if confirmed else 'refuted'
        elif pred['prediction'] == 'shorter_predict_lowers_miss':
            short = next(c for c in levels if c.predict_s == 1.0)
            long = next(c for c in levels if c.predict_s == 5.0)
            confirmed = rank([summarize(short, runs), summarize(long, runs)])[0]['cfg'] == short
            outcome = 'confirmed' if confirmed else 'refuted'
        elif pred['prediction'] == 'shorter_execute_lowers_miss':
            shorter = levels[0]
            confirmed = rank([summarize(shorter, runs), base_row])[0]['cfg'] == shorter
            outcome = 'confirmed' if confirmed else 'refuted'
        elif pred['prediction'] == 'short_chunk_increases_miss':
            worse = all(rank([summarize(c, runs), base_row])[0]['cfg'] == base for c in levels)
            confirmed = bool(worse)
            outcome = 'confirmed' if worse else 'refuted'
        else:
            outcome = 'no prediction'
        flags[name] = dict(prediction=pred['prediction'], outcome=outcome, value=pred['value'],
                           best=best['cfg'].name, best_catch=best['catch'], best_miss=best['miss'])
        lines.append(
            f"| {name} | {pred['prediction']} | {outcome} | {best['cfg'].name} | "
            f"{best['catch']:.3f} | {best['miss']:.1f} | {base_row['miss']:.1f} |")
    return lines, flags


def _role_summary(role, cfg):
    runs = load_role(role)
    row = summarize(cfg, runs)
    if row is None:
        raise SystemExit(f'{role} {cfg.name} is not scored')
    return row


def phase_compare(args):
    from parking_split import aggregate, eval_weights, load_experts
    hypo = hypotheses_ready()
    if not WINNER_PATH.exists():
        raise SystemExit('no winner.json')
    winner = json.loads(WINNER_PATH.read_text())
    cfg = Config(**winner['config'])
    runs = load_role('sweep')
    runs.update(load_role('combination'))
    lines, flags = _factor_rows(runs, hypo)
    header = ['| Factor | Prediction from the baseline | Outcome | Best level | Catch | Miss mm | Baseline miss mm |',
              '| --- | --- | --- | --- | ---: | ---: | ---: |']
    dart_cfg = replace(cfg, demos=2000)
    test = json.loads((DATA / 'seeds.json').read_text())['test'][:200]
    print(f'parking split on {len(test)} test scenes', flush=True)
    experts = load_experts(test)
    blocks = {}
    for role, model in (('winner', cfg), ('dart', dart_cfg), ('offset', dart_cfg)):
        role_dir = 'sweep' if role == 'winner' and run_path('sweep', model, 0).exists() else (
            'combination' if role == 'winner' else role)
        stats = []
        for seed in SEEDS:
            weights = ckpt_path(role_dir, model, seed).with_suffix('.npz')
            if not weights.exists():
                raise SystemExit(f'missing {weights}')
            print(f'rollout {role} s{seed}', flush=True)
            one = eval_weights(weights, test, experts)
            blocks.setdefault(role, []).append(one)
        blocks[role + '_mean'] = aggregate(blocks[role])
    def plain(obj):
        if isinstance(obj, dict):
            return {str(k): plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [plain(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    (STUDY / 'three_models.json').write_text(json.dumps(plain(blocks), indent=2))
    three = {role: blocks[role + '_mean'] for role in ('winner', 'dart', 'offset')}
    def separated(a, b):
        return a['miss_mm'] + a['miss_mm_sd'] < b['miss_mm']
    offsets_beat_winner = separated(three['offset'], three['winner'])
    offsets_beat_dart = separated(three['offset'], three['dart'])
    dart_beats_winner = separated(three['dart'], three['winner'])
    cross_gain = three['offset']['comp_cross'] - three['dart']['comp_cross']
    along_gain = three['offset']['comp_along'] - three['dart']['comp_along']
    if offsets_beat_winner and not dart_beats_winner and (cross_gain > 0.1 or along_gain > 0.1):
        mechanism = ('Offset demos lower the miss and raise the parking-compensation slope relative to '
                     'high-amplitude command noise. The noise matches the winner\'s per-dimension command '
                     'error and does not move the parking goal.')
    elif dart_beats_winner:
        mechanism = ('High-amplitude DART lowered the miss. The report does not say that DART failed.')
    else:
        mechanism = ('The miss and the compensation slopes do not support "DART failed and offsets fixed '
                     'the parking error." Report the three rows and stop there.')
    model_lines = ['| Model | Miss mm | Catch at 10 mm | Along slope | Sideways slope |',
                   '| --- | ---: | ---: | ---: | ---: |']
    for role in ('winner', 'dart', 'offset'):
        row = three[role]
        model_lines.append(
            f"| {role} | {row['miss_mm']:.1f} ± {row['miss_mm_sd']:.1f} | "
            f"{row['catch_rate']:.3f} ± {row['catch_rate_sd']:.3f} | "
            f"{row['comp_along']:.2f} | {row['comp_cross']:.2f} |")
    text = '\n'.join([
        '# Flat rel_add study',
        '',
        'Encoder is flat. Features are rel_add. DeepSets weights are not inputs.',
        '',
        f"Ablation winner: `{winner['name']}` catch {winner['catch']:.3f} miss {winner['miss']:.1f} mm.",
        '',
        '## Predictions against the sweep',
        '',
        *header, *lines, '',
        'A prediction is confirmed only by the inequality written in hypotheses.json.',
        'A factor marked not_indicated was still swept. It has no confirmation to claim.',
        '',
        '## Winner, high-amplitude DART, parking offsets',
        '',
        *model_lines, '',
        mechanism, '',
        'DART and the offset set each add 1,000 episodes on the offset dataset\'s training seeds.',
        'The ablation winner stays the 1,000-demo model.',
    ])
    out = STUDY / 'verdict.md'
    out.write_text(text + '\n')
    (STUDY / 'verdict.json').write_text(json.dumps(dict(
        factors=flags, models=three, mechanism=mechanism,
        offsets_beat_winner=offsets_beat_winner, offsets_beat_dart=offsets_beat_dart,
        dart_beats_winner=dart_beats_winner), indent=2))
    print(text, flush=True)
    print(f'wrote {out}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('baseline', 'analyze', 'sweep', 'combine', 'dart', 'compare', 'all'))
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--gpu-jobs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--radius', type=float, default=0.01)
    args = parser.parse_args()
    phases = {
        'baseline': phase_baseline, 'analyze': phase_analyze, 'sweep': phase_sweep,
        'combine': phase_combine, 'dart': phase_dart, 'compare': phase_compare,
    }
    if args.phase == 'all':
        for name in ('baseline', 'analyze', 'sweep', 'combine', 'dart', 'compare'):
            print(f'\n=== {name} ===', flush=True)
            phases[name](args)
    else:
        phases[args.phase](args)


if __name__ == '__main__':
    main()
