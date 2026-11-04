"""Downsample the 1 kHz demo logs once to the shared 50 Hz anchor grid.

Every policy rate (5, 10, 25 Hz) divides 50 Hz, and policy ticks start at the same
simulation step as the grid, so each rate's history frames and chunk labels are exact
grid samples. Demo order is preserved so nested subsets are prefixes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from catch_env import FIRST_TICK_STEP, OBS_NAMES
from policy import GRID_HZ

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require', type=int, default=1000, help='Fail unless this many contiguous demos exist')
    parser.add_argument('--out', type=Path, default=DATA / f'grid{GRID_HZ}.npz')
    args = parser.parse_args()
    files = []
    while (DATA / 'demos' / f'demo_{len(files):04d}.npz').exists():
        files.append(DATA / 'demos' / f'demo_{len(files):04d}.npz')
    if len(files) < args.require:
        raise SystemExit(f'only {len(files)} contiguous demos, need {args.require}')
    stride = 1000 // GRID_HZ
    obs, cmd, demo_index, starts, lengths, metas = [], [], [], [], [], []
    offset = 0
    for i, f in enumerate(files):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            assert meta['index'] == i
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        obs.append(o)
        cmd.append(c)
        demo_index.append(np.full(len(o), i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
        metas.append(meta)
    out = args.out
    np.savez(out, obs=np.concatenate(obs), cmd=np.concatenate(cmd), demo=np.concatenate(demo_index),
             starts=np.array(starts), lengths=np.array(lengths),
             arm_q_col=np.int64(OBS_NAMES.index('arm_q_0')))
    success = np.mean([m['success'] for m in metas])
    print(f'{len(files)} demos, {offset} grid samples, mean length {np.mean(lengths) / GRID_HZ:.1f}s, '
          f'expert success {success:.3f} -> {out}')


if __name__ == '__main__':
    main()
