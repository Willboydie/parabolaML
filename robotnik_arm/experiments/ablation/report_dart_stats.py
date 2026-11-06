"""Statistics of the three 1,000-episode demonstration sets for the report.

Same 1,000 training scenes, three versions: the clean expert (data/demos), the
parking-offset expert (data/offset_parking) and the expert executed under the
high-amplitude DART noise (data/rel_study_dart). For each scene we record the
expert's miss, and how far the base is from the clean expert's base at the
clean catch time. For the DART set we also record the RMS of the injected noise
and of the clean command per channel group.

Output: results/rel_study/report_dart_stats.npz
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'rel_study' / 'report_dart_stats.npz'
BASE = slice(10, 12)
N = 1000
EXAMPLES = 6


def load(kind, i):
    with np.load(DATA / kind / f'demo_{i:04d}.npz') as z:
        meta = json.loads(str(z['meta']))
        out = dict(meta=meta, base=np.array(z['obs'][:, BASE]), t=np.array(z['obs'][:, 0]),
                   ready=np.array(z['ready'], bool))
        if 'executed' in z.files:
            out['cmd'] = np.array(z['cmd'])
            out['executed'] = np.array(z['executed'])
        if kind == 'demos':
            ball = np.array(z['obs'][:, 1:4])
            out['ball'] = ball
    return out


def main():
    rows = {k: [] for k in ('seed', 'clean_miss', 'offset_miss', 'dart_miss', 'clean_ok', 'offset_ok', 'dart_ok',
                            'offset_base_mm', 'dart_base_mm', 'dart_base_max_mm', 't_catch')}
    sq_noise = np.zeros(18)
    sq_cmd = np.zeros(18)
    n_ticks = 0
    examples = {}
    for i in range(N):
        c = load('demos', i)
        o = load('offset_parking', i)
        d = load('rel_study_dart', i)
        assert c['meta']['seed'] == o['meta']['seed'] == d['meta']['seed'], i
        tc = c['meta']['catch_time']
        k = len(c['base']) - 1 if tc is None else min(int(round(tc / 0.001)), len(c['base']) - 1)
        ref = c['base'][k]
        ko = min(k, len(o['base']) - 1)
        kd = min(k, len(d['base']) - 1)
        rows['seed'].append(int(c['meta']['seed']))
        rows['t_catch'].append(np.nan if tc is None else float(tc))
        for name, demo in (('clean', c), ('offset', o), ('dart', d)):
            rows[f'{name}_miss'].append(float(demo['meta']['min_dist']) * 1000)
            rows[f'{name}_ok'].append(bool(demo['meta']['success']))
        rows['offset_base_mm'].append(float(np.linalg.norm(o['base'][ko] - ref) * 1000))
        rows['dart_base_mm'].append(float(np.linalg.norm(d['base'][kd] - ref) * 1000))
        n = min(len(c['base']), len(d['base']))
        rows['dart_base_max_mm'].append(float(np.max(np.linalg.norm(d['base'][:n] - c['base'][:n], axis=1)) * 1000))
        r = d['ready']
        sq_noise += np.sum((d['executed'][r] - d['cmd'][r]) ** 2, axis=0)
        sq_cmd += np.sum(d['cmd'][r] ** 2, axis=0)
        n_ticks += int(r.sum())
        if i < EXAMPLES:
            examples[f'ex{i}_clean_base'] = c['base'][::20]
            examples[f'ex{i}_offset_base'] = o['base'][::20]
            examples[f'ex{i}_dart_base'] = d['base'][::20]
            examples[f'ex{i}_ball'] = c['ball'][::20]
            examples[f'ex{i}_t'] = d['t'][::20]
            examples[f'ex{i}_dart_cmd'] = d['cmd'][::20, :4]
            examples[f'ex{i}_dart_exec'] = d['executed'][::20, :4]
            examples[f'ex{i}_catch_k'] = np.int64(k // 20)
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{N}', flush=True)
    pack = {k: np.asarray(v) for k, v in rows.items()}
    pack['noise_rms'] = np.sqrt(sq_noise / n_ticks)
    pack['cmd_rms'] = np.sqrt(sq_cmd / n_ticks)
    pack.update(examples)
    np.savez(OUT, **pack)
    for name in ('clean', 'offset', 'dart'):
        print(name, 'catch', np.mean(pack[f'{name}_ok']), 'mean miss', np.mean(pack[f'{name}_miss']))
    print('offset base median', np.median(pack['offset_base_mm']), 'dart base median', np.median(pack['dart_base_mm']))
    print('noise rms', np.round(pack['noise_rms'], 3))
    print('cmd rms', np.round(pack['cmd_rms'], 3))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
