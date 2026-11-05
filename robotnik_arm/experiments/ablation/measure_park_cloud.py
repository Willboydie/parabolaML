"""Signed parking errors of the current DeepSets ceiling, and the shared offset list.

Rolls the three existing checkpoints on the 200 test seeds. At the closest approach
the base error is written in the expert's route frame, plus the heading error.
One thousand draws from that cloud, capped at 150 mm, are saved for both generators.
"""
import json
from pathlib import Path

import numpy as np

from parking_split import load_experts
from run_relative import init_model, policy_episode, pool_map, route_axes
from variety_data import (
    CLOUD, HEADING_GATE_DEG, MASTER_SEED, MODELS, N, OFFSET_CAP, OFFSET_STREAM, OFFSETS,
    DEEPSETS_TAG, assert_writable,
)

SEEDS = (0, 1, 2)


def wrap(angle):
    return (float(angle) + np.pi) % (2 * np.pi) - np.pi


def components(policy_rows, experts, model_seed):
    along, cross, heading, seeds = [], [], [], []
    skipped = 0
    for row in policy_rows:
        best = row['best']
        if best is None:
            skipped += 1
            continue
        ex = experts[row['seed']]
        k = min(best['step'], len(ex['base']) - 1)
        ax_along, ax_cross = route_axes(ex['chord'], float(ex['bearing'][k]))
        delta = np.asarray(best['base'], float) - np.asarray(ex['base'][k], float)
        along.append(float(np.dot(delta, ax_along)))
        cross.append(float(np.dot(delta, ax_cross)))
        heading.append(wrap(best['bearing'] - float(ex['bearing'][k])))
        seeds.append(int(row['seed']))
    return dict(
        along_m=np.asarray(along, np.float64),
        cross_m=np.asarray(cross, np.float64),
        heading_rad=np.asarray(heading, np.float64),
        seed=np.asarray(seeds, np.int64),
        model_seed=np.full(len(seeds), int(model_seed), np.int64),
        skipped=int(skipped),
    )


def draw_offsets(along_m, cross_m, heading_rad):
    seed = int(np.random.SeedSequence([MASTER_SEED, OFFSET_STREAM]).generate_state(1, dtype=np.uint64)[0] >> 1)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(along_m), size=N)
    along = along_m[index].copy()
    side = cross_m[index].copy()
    heading = heading_rad[index].copy()
    mag = np.hypot(along, side)
    over = mag > OFFSET_CAP
    along[over] *= OFFSET_CAP / mag[over]
    side[over] *= OFFSET_CAP / mag[over]
    heading_abs_median_deg = float(np.median(np.abs(heading_rad)) * 180.0 / np.pi)
    return dict(
        master=MASTER_SEED, offset_stream=OFFSET_STREAM, n=N, cap_m=OFFSET_CAP,
        include_heading=bool(heading_abs_median_deg > HEADING_GATE_DEG),
        heading_abs_median_deg=heading_abs_median_deg,
        heading_gate_deg=HEADING_GATE_DEG,
        n_capped=int(over.sum()),
        along_m=along.tolist(), side_m=side.tolist(), heading_rad=heading.tolist(),
        source_index=index.astype(int).tolist(),
    )


def main():
    from variety_data import test_seeds
    weights = [MODELS / f'{DEEPSETS_TAG}_s{seed}.npz' for seed in SEEDS]
    missing = [path for path in weights if not path.exists()]
    if missing:
        raise SystemExit('missing ' + ', '.join(str(p) for p in missing))
    test = test_seeds()
    if CLOUD.exists():
        with np.load(CLOUD) as z:
            packed = {key: np.array(z[key]) for key in ('along_m', 'cross_m', 'heading_rad', 'seed', 'model_seed')}
        print(f'reusing {CLOUD} ({len(packed["along_m"])} samples)', flush=True)
    else:
        print(f'expert reference on {len(test)} test seeds', flush=True)
        experts = load_experts(test)
        parts = []
        for seed, path in zip(SEEDS, weights):
            print(f'rollout {path.name}', flush=True)
            rows = pool_map(policy_episode, [(int(s),) for s in test], init_model, (str(path),), path.stem)
            part = components(rows, experts, seed)
            print(f"  s{seed}: {len(part['along_m'])} parks, {part['skipped']} without a closest approach",
                  flush=True)
            parts.append(part)
        packed = {key: np.concatenate([part[key] for part in parts]) for key in
                  ('along_m', 'cross_m', 'heading_rad', 'seed', 'model_seed')}
        assert_writable(CLOUD)
        tmp = Path(str(CLOUD) + '.tmp.npz')
        assert_writable(tmp)
        np.savez(tmp, **packed)
        tmp.replace(CLOUD)
        print(f'wrote {CLOUD}', flush=True)
    if len(packed['along_m']) == 0:
        raise SystemExit('the parking cloud is empty')
    base = np.hypot(packed['along_m'], packed['cross_m']) * 1000
    print(f"cloud n {len(base)}  base median {np.median(base):.1f} mm  "
          f"along {np.median(np.abs(packed['along_m'])) * 1000:.1f}  "
          f"cross {np.median(np.abs(packed['cross_m'])) * 1000:.1f}  "
          f"heading {np.median(np.abs(packed['heading_rad'])) * 180 / np.pi:.1f} deg", flush=True)
    spec = draw_offsets(packed['along_m'], packed['cross_m'], packed['heading_rad'])
    spec['cloud'] = str(CLOUD)
    spec['cloud_n'] = int(len(packed['along_m']))
    spec['cloud_base_median_mm'] = float(np.median(base))
    assert_writable(OFFSETS)
    OFFSETS.write_text(json.dumps(spec))
    drawn = np.hypot(spec['along_m'], spec['side_m'])
    print(f"drew {N} offsets, sideways sd {np.std(spec['side_m']) * 1000:.1f} mm, "
          f"along sd {np.std(spec['along_m']) * 1000:.1f} mm, max {np.max(drawn) * 1000:.1f} mm, "
          f"capped {spec['n_capped']}, include_heading {spec['include_heading']}", flush=True)
    print(f'wrote {OFFSETS}', flush=True)


if __name__ == '__main__':
    main()
