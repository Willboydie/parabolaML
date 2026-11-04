"""Per-stage tables, ranking, and Stage E config selection from results/runs/*.json.

Ranking: mean catch rate over training seeds. If two configs' mean +/- seed std
intervals overlap, the one with the lower mean miss distance wins.
"""
import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import BASELINE, Config
from run_ablation import stage_configs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FACTORS = {
    'A': ('rate_hz',),
    'B': ('predict_s', 'execute_frac'),
    'C': ('history', 'history_s'),
    'D': ('demos',),
}


def load_runs(radius=None):
    runs = {}
    for f in sorted((RESULTS / 'runs').glob('*.json')):
        r = json.loads(f.read_text())
        if radius is not None:
            r['catch_rate'] = float(np.mean([e['min_dist'] < radius for e in r['episodes']]))
        runs.setdefault(r['name'], []).append(r)
    return runs


def summarize(cfg, runs):
    rows = runs.get(cfg.name, [])
    if not rows:
        return None
    c = np.array([r['catch_rate'] for r in rows])
    m = np.array([r['miss_mm'] for r in rows])
    return dict(cfg=cfg, n=len(rows), catch=c.mean(), catch_sd=c.std(ddof=1) if len(c) > 1 else 0.0,
                catch_min=c.min(), catch_max=c.max(), miss=m.mean(), miss_sd=m.std(ddof=1) if len(m) > 1 else 0.0,
                train=np.mean([r['train_time_s'] for r in rows]), eval_wall=np.mean([r['eval_wall_s'] for r in rows]),
                median_miss=np.median([e['min_dist'] * 1000 for r in rows for e in r['episodes']]))


def overlap(a, b):
    return a['catch'] - a['catch_sd'] <= b['catch'] + b['catch_sd'] and b['catch'] - b['catch_sd'] <= a['catch'] + a['catch_sd']


def rank(summaries):
    """Order by the stated rule: take the best mean catch, then prefer any config within its spread with lower miss."""
    left, order = [s for s in summaries if s], []
    while left:
        top = max(left, key=lambda s: s['catch'])
        tied = [s for s in left if overlap(s, top)]
        win = min(tied, key=lambda s: s['miss'])
        order.append(win)
        left.remove(win)
    return order


def label(stage, cfg):
    if stage == 'A':
        return f'{cfg.rate_hz} Hz'
    if stage == 'B':
        return f'predict {cfg.predict_s:g} s ({cfg.horizon} steps), execute {cfg.execute_frac:g} = {cfg.execute / cfg.rate_hz:g} s ({cfg.execute} steps)'
    if stage == 'C':
        return 'none' if cfg.history == 'none' else f'{cfg.history} {cfg.history_s:g} s ({cfg.frames} frames)'
    if stage == 'D':
        return f'{cfg.demos} demos'
    return (f'{cfg.rate_hz} Hz, predict {cfg.predict_s:g} s / exec {cfg.execute_frac:g}, '
            f"hist {'none' if cfg.history == 'none' else f'{cfg.history} {cfg.history_s:g}s'}, {cfg.demos} demos")


def table(stage, summaries):
    lines = ['| Rank | Config | Seeds | Catch rate mean ± sd [min–max] | Miss distance mm mean ± sd | Median miss mm | Train time s |',
             '|---:|---|---:|---|---|---:|---:|']
    for i, s in enumerate(rank(summaries), 1):
        cfg = s['cfg']
        base = ' (baseline)' if cfg == BASELINE else ''
        lines.append(f"| {i} | {label(stage, cfg)}{base} | {s['n']} | {s['catch']:.3f} ± {s['catch_sd']:.3f} "
                     f"[{s['catch_min']:.3f}–{s['catch_max']:.3f}] | {s['miss']:.1f} ± {s['miss_sd']:.1f} | "
                     f"{s['median_miss']:.1f} | {s['train']:.0f} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude-rates', default='')
    parser.add_argument('--radius', type=float, default=None, help='Re-score catches at a smaller radius (m)')
    parser.add_argument('--write-stage-e', action='store_true')
    args = parser.parse_args()
    excluded = [int(r) for r in args.exclude_rates.split(',') if r]
    runs = load_runs(args.radius)
    stages = stage_configs(excluded)
    out, winners, effects, top2 = [], {}, {}, {}
    for stage, cfgs in stages.items():
        sums = [summarize(c, runs) for c in cfgs]
        sums = [s for s in sums if s]
        if not sums:
            continue
        out.append(f'### Stage {stage}\n\n' + table(stage, sums))
        order = rank(sums)
        winners[stage] = order[0]['cfg']
        top2[stage] = [s['cfg'] for s in order[:2]]
        best, worst = max(sums, key=lambda s: s['catch']), min(sums, key=lambda s: s['catch'])
        effects[stage] = dict(catch=best['catch'] - worst['catch'], resolved=not overlap(best, worst),
                              miss=max(s['miss'] for s in sums) - min(s['miss'] for s in sums))
    # A catch-rate range inside the seed spread is not a resolved effect; such factors are ordered by miss range.
    effect_key = lambda k: (effects[k]['catch'] if effects[k]['resolved'] else 0.0, effects[k]['miss'])
    if effects:
        out.append('### Factor effects\n\nCatch-rate range counts only when the best and worst values are outside each '
                   "other's seed spread; otherwise the factor is ordered by its miss-distance range.\n\n"
                   + '\n'.join(f"- Stage {k}: catch range {v['catch']:.3f} ({'resolved' if v['resolved'] else 'within spread'}), "
                               f"miss range {v['miss']:.1f} mm" for k, v in sorted(effects.items(), key=lambda kv: effect_key(kv[0]), reverse=True)))
    e_cfgs = []
    if len(effects) == 4:
        movers = sorted(effects, key=effect_key, reverse=True)[:2]
        for a, b in itertools.product(top2[movers[0]], top2[movers[1]]):
            cfg = BASELINE
            for stage, src in ((movers[0], a), (movers[1], b)):
                cfg = replace(cfg, **{f: getattr(src, f) for f in FACTORS[stage]})
            e_cfgs.append(cfg)
        guess = BASELINE
        for stage, win in winners.items():
            guess = replace(guess, **{f: getattr(win, f) for f in FACTORS[stage]})
        e_cfgs.append(guess)
        e_cfgs = list(dict.fromkeys(e_cfgs))
        out.append(f'Stage E movers: {movers}; grid of top-2 values plus best single guess:\n\n'
                   + '\n'.join(f'- {label("E", c)}' for c in e_cfgs))
        if args.write_stage_e:
            (RESULTS / 'stage_e_configs.json').write_text(json.dumps([c.to_dict() for c in e_cfgs], indent=1))
    e_sums = [summarize(c, runs) for c in e_cfgs]
    if any(e_sums):
        out.append('### Stage E\n\n' + table('E', [s for s in e_sums if s]))
    text = '\n\n'.join(out)
    print(text)
    (RESULTS / ('summary.md' if args.radius is None else f'summary_r{args.radius * 1000:g}mm.md')).write_text(text + '\n')


if __name__ == '__main__':
    main()
