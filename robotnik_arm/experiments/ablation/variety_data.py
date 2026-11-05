"""Shared paths for the parking-variety comparison.

New files only. The current one-ball grids, raw demos, checkpoints and recorded
results are read, never written. ``grid50_offset.npz`` is on that list because
``gen_offset_parking.build_combined`` writes it; these scripts must not.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
MODELS = RESULTS / 'models'

MASTER_SEED = 2026
OFFSET_STREAM = 42
OFFSET_CAP = 0.150
N = 1000
HEADING_GATE_DEG = 10.0

CLOUD = RESULTS / 'learner_park_cloud.npz'
OFFSETS = RESULTS / 'learner_park_offsets.json'
COMPARE = RESULTS / 'offset_variety_compare.json'

MATCHED_DIR = DATA / 'offset_parking_matched'
MATCHED_GRID = DATA / 'grid50_offset_matched.npz'
PLANNED_DIR = DATA / 'offset_parking_planned'
PLANNED_GRID = DATA / 'grid50_offset_planned.npz'

BASE_GRID = DATA / 'grid50.npz'
DEEPSETS_TAG = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
MATCHED_TAG = DEEPSETS_TAG + '_matched'
PLANNED_TAG = DEEPSETS_TAG + '_planned'


def _protected():
    paths = [
        BASE_GRID,
        DATA / 'grid50_offset.npz',
        DATA / 'grid50_offset_along55.npz',
        DATA / 'grid50_two_ball.npz',
        DATA / 'seeds.json',
        DATA / 'offset_parking',
        DATA / 'offset_parking_along55',
        DATA / 'two_ball',
        RESULTS / 'stage2' / 'oneball_deepsets.json',
        RESULTS / 'stage2' / 'radius_curve.json',
        RESULTS / 'offset_along55_train.json',
        RESULTS / 'offset_parking_train.json',
        RESULTS / 'parking_split.json',
    ]
    for tag in (
        DEEPSETS_TAG,
        'r25_p2_e0.5_hstack0.5s_d2000_rel_add',
        'r25_p2_e0.5_hstack0.5s_d2000_rel_add_along55',
        'r25_p2_e0.5_hstack0.5s_d1000',
    ):
        for seed in (0, 1, 2):
            for ext in ('.pt', '.npz'):
                paths.append(MODELS / f'{tag}_s{seed}{ext}')
    return paths


PROTECTED = _protected()


def assert_writable(path):
    """Raise if ``path`` is a protected file or sits inside a protected directory."""
    path = Path(path).resolve()
    for item in PROTECTED:
        item = item.resolve()
        if path == item or item in path.parents:
            raise SystemExit(f'refusing to write protected path {path}')
    return path


def load_offsets():
    if not OFFSETS.exists():
        raise SystemExit(f'missing {OFFSETS}; run measure_park_cloud.py first')
    spec = json.loads(OFFSETS.read_text())
    if int(spec['n']) != N or len(spec['along_m']) != N or len(spec['side_m']) != N:
        raise SystemExit(f'{OFFSETS} does not hold {N} offsets')
    if abs(float(spec['cap_m']) - OFFSET_CAP) > 1e-12:
        raise SystemExit(f'{OFFSETS} cap is not {OFFSET_CAP}')
    return spec


def training_seeds():
    heldout = json.loads((DATA / 'seeds.json').read_text())
    seeds = heldout['train'][:N]
    if len(seeds) < N:
        raise SystemExit(f'only {len(seeds)} training seeds')
    if set(seeds) & set(heldout['test']):
        raise SystemExit('training seeds overlap the test set')
    return seeds


def test_seeds():
    heldout = json.loads((DATA / 'seeds.json').read_text())
    return heldout['test'][:200]


def build_grid(demo_dir, grid_path, n_new):
    """Original 1,000 rows of grid50.npz, then ``n_new`` new demos. Never the old offset grids."""
    from catch_env import FIRST_TICK_STEP
    grid_path = assert_writable(grid_path)
    demo_dir = assert_writable(demo_dir)
    if grid_path.exists() and grid_path.is_dir():
        raise SystemExit(f'{grid_path} is a directory')
    stride = 20
    with np.load(BASE_GRID) as z:
        n_rows = int(z['starts'][999] + z['lengths'][999])
        obs = [np.array(z['obs'][:n_rows], copy=True)]
        cmd = [np.array(z['cmd'][:n_rows], copy=True)]
        demo = [np.array(z['demo'][:n_rows], copy=True)]
        starts = [int(v) for v in z['starts'][:1000]]
        lengths = [int(v) for v in z['lengths'][:1000]]
        q_col = int(z['arm_q_col'])
    if int(demo[0].max()) != 999 or len(starts) != 1000:
        raise RuntimeError('grid50.npz is not the original 1,000 demos')
    offset = int(starts[-1] + lengths[-1])
    for i in range(n_new):
        path = demo_dir / f'demo_{i:04d}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta['index'] != i:
                raise RuntimeError(f'{path.name} index {meta["index"]}')
            o = z['obs'][FIRST_TICK_STEP::stride]
            c = z['cmd'][FIRST_TICK_STEP::stride]
        if len(o) == 0:
            raise RuntimeError(f'{path.name} truncated to nothing')
        obs.append(o)
        cmd.append(c)
        demo.append(np.full(len(o), 1000 + i, dtype=np.int32))
        starts.append(offset)
        lengths.append(len(o))
        offset += len(o)
    tmp = Path(str(grid_path) + '.tmp.npz')
    assert_writable(tmp)
    np.savez(tmp, obs=np.concatenate(obs), cmd=np.concatenate(cmd), demo=np.concatenate(demo),
             starts=np.asarray(starts), lengths=np.asarray(lengths), arm_q_col=np.int64(q_col))
    tmp.replace(grid_path)
    return offset


def world_offset(along_m, side_m, nominal_xy, base_xy, bearing):
    """Metres in the world frame. ``side_m`` is to the left of the route, as in OffsetGoalDriver."""
    drive = np.asarray(nominal_xy[:2], float) - np.asarray(base_xy[:2], float)
    n = float(np.linalg.norm(drive))
    if n > 1e-3:
        fwd = drive / n
    else:
        bearing = float(bearing)
        fwd = np.array([np.cos(bearing), np.sin(bearing)])
    left = np.array([-fwd[1], fwd[0]])
    return float(along_m) * fwd + float(side_m) * left


def pose_required(goal, bearing, base_pos, base_bearing, ee_pos, intercept, base_params, arm_params):
    """Drive time plus reach time, the same model ``arrival_options`` uses."""
    from simulation_and_control import wrap_angle
    drive = np.asarray(goal[:2], float) - np.asarray(base_pos[:2], float)
    distance = float(np.linalg.norm(drive))
    heading = np.arctan2(drive[1], drive[0]) if distance > base_params.pos_tol else float(base_bearing)
    turns = abs(wrap_angle(heading - base_bearing)) + abs(wrap_angle(bearing - heading))
    base_time = distance / (0.8 * base_params.max_lin_vel) + turns / (0.65 * base_params.max_ang_vel)
    if distance > base_params.pos_tol or turns > base_params.ang_tol:
        base_time += 2.0
    yaw = wrap_angle(float(bearing) - float(base_bearing))
    c, s = np.cos(yaw), np.sin(yaw)
    relative = np.asarray(ee_pos[:2], float) - np.asarray(base_pos[:2], float)
    carried_xy = np.asarray(goal[:2], float) + np.array([[c, -s], [s, c]]) @ relative
    carried_ee = np.r_[carried_xy, float(ee_pos[2])]
    arm_time = np.linalg.norm(np.asarray(intercept, float) - carried_ee) / (0.65 * arm_params.max_cartesian_speed) + 2.5
    return base_time + arm_time + 1.0
