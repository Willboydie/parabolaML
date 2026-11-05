"""1,000 demos where the planner commits to the shared offset, or refuses it.

The expert still plans the optimal park first. The offset pose is kept only when
the same drive-time and reach-time model says it arrives by the chosen intercept.
Otherwise the optimal park is driven. Writes data/offset_parking_planned and
grid50_offset_planned.npz.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from gen_offset_parking import init_worker
from variety_data import (
    MASTER_SEED, N, OFFSET_CAP, OFFSET_STREAM, PLANNED_DIR, PLANNED_GRID,
    assert_writable, build_grid, load_offsets, pose_required, training_seeds, world_offset,
)

OUT = PLANNED_DIR


class PlannedOffsetDriver:
    """ExpertDriver that commits to one offset pose when the timing model allows it."""

    def __init__(self, along_m, side_m, heading_rad):
        from catch_env import ExpertDriver
        self.inner = ExpertDriver()
        self.along_m = float(along_m)
        self.side_m = float(side_m)
        self.heading_rad = float(heading_rad)
        self.offset_xy = None
        self.n_plans = 0
        self.n_committed = 0
        self.committed = False
        self.refuse_reason = 'no_plan'
        self.required_s = None
        self.delay_s = None
        self.shortfall_m = None

    def reset(self, env):
        self.inner.reset(env)
        self.offset_xy = None
        self.n_plans = 0
        self.n_committed = 0
        self.committed = False
        self.refuse_reason = 'no_plan'
        self.required_s = None
        self.delay_s = None
        self.shortfall_m = None
        expert = self.inner.expert
        orig = expert.plan
        driver = self

        def plan(time, position, base_pos, base_bearing, ee_pos, base_params, arm_params, max_delay=240.0):
            ok = orig(time, position, base_pos, base_bearing, ee_pos, base_params, arm_params, max_delay=max_delay)
            if ok:
                driver._decide(time, base_pos, base_bearing, ee_pos, base_params, arm_params)
            return ok

        expert.plan = plan

    def _decide(self, time, base_pos, base_bearing, ee_pos, base_params, arm_params):
        from simulation_and_control import wrap_angle
        expert = self.inner.expert
        nominal = np.asarray(expert.goal_pos[:2], float).copy()
        bearing = float(expert.goal_bearing)
        if self.offset_xy is None:
            self.offset_xy = world_offset(self.along_m, self.side_m, nominal, np.asarray(base_pos[:2], float), bearing)
        target = nominal + self.offset_xy
        target_bearing = float(wrap_angle(bearing + self.heading_rad))
        delay = float(expert.intercept_time - time)
        required = pose_required(target, target_bearing, base_pos, base_bearing, ee_pos,
                                 expert.intercept, base_params, arm_params)
        self.n_plans += 1
        self.required_s = float(required)
        self.delay_s = delay
        if required <= delay + 1e-9:
            expert.goal_pos = target.copy()
            expert.goal_bearing = target_bearing
            self.committed = True
            self.n_committed += 1
            self.refuse_reason = None
            self.shortfall_m = 0.0
        else:
            self.committed = False
            self.refuse_reason = 'time'
            self.shortfall_m = float(np.linalg.norm(self.offset_xy))

    def sense(self, env, t):
        self.inner.sense(env, t)

    def act(self, env, t, dt_ctrl):
        return self.inner.act(env, t, dt_ctrl)


def demo_job(index, seed, along_m, side_m, heading_rad, path):
    from catch_env import rollout
    from gen_offset_parking import _env
    path = assert_writable(path)
    driver = PlannedOffsetDriver(along_m, side_m, heading_rad)
    out = rollout(_env, driver, seed, 1000, log=True)
    arrays = out.pop('arrays')
    if not np.isfinite(out['min_dist']):
        out['min_dist'] = None
    meta = dict(out, index=index, dart=False, rate_hz=1000, sim_dt=float(_env.dt),
                offset_along_m=along_m, offset_side_m=side_m, offset_cap_m=OFFSET_CAP,
                offset_stream=OFFSET_STREAM, arm='planned', heading_applied_rad=heading_rad,
                committed=bool(driver.committed), n_plans=int(driver.n_plans),
                n_committed=int(driver.n_committed), refuse_reason=driver.refuse_reason,
                required_s=driver.required_s, delay_s=driver.delay_s, shortfall_m=driver.shortfall_m)
    tmp = Path(str(path) + '.tmp.npz')
    assert_writable(tmp)
    np.savez(tmp, obs=arrays['obs'], cmd=arrays['cmd'], applied=arrays['applied'], ready=arrays['ready'],
             meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return meta


def main():
    spec_in = load_offsets()
    seeds = training_seeds()
    along = [float(v) for v in spec_in['along_m']]
    side = [float(v) for v in spec_in['side_m']]
    include = bool(spec_in['include_heading'])
    heading = [float(v) if include else 0.0 for v in spec_in['heading_rad']]
    print(f"shared offsets: along sd {np.std(along) * 1000:.1f} mm, "
          f"side sd {np.std(side) * 1000:.1f} mm, include_heading {include}", flush=True)
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = assert_writable(OUT / 'dataset.json')
    spec = dict(master=MASTER_SEED, offset_stream=OFFSET_STREAM, n=N, rate_hz=1000, arm='planned',
                cap_m=OFFSET_CAP, include_heading=include,
                heading_abs_median_deg=spec_in['heading_abs_median_deg'],
                seeds=seeds, along_m=along, side_m=side, heading_rad=heading)
    if spec_path.exists():
        saved = json.loads(spec_path.read_text())
        if saved.get('seeds') != seeds or saved.get('side_m') != side or saved.get('along_m') != along:
            raise SystemExit(f'{spec_path} does not match this offset list')
        if saved.get('include_heading') != include:
            raise SystemExit(f'{spec_path} heading gate does not match the cloud')
    spec_path.write_text(json.dumps(spec))
    pending = [(i, int(seed), along[i], side[i], heading[i], OUT / f'demo_{i:04d}.npz')
               for i, seed in enumerate(seeds) if not (OUT / f'demo_{i:04d}.npz').exists()]
    print(f'{N} planner-committed demos, {len(pending)} to generate', flush=True)
    t0 = time.time()
    if pending:
        with ProcessPoolExecutor(16, initializer=init_worker) as pool:
            futs = [pool.submit(demo_job, *job) for job in pending]
            for k, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if k % 25 == 0 or k == len(pending):
                    print(f'  {k}/{len(pending)} ({time.time() - t0:.0f}s)', flush=True)
    metas = []
    for i in range(N):
        with np.load(OUT / f'demo_{i:04d}.npz', allow_pickle=False) as z:
            metas.append(json.loads(str(z['meta'])))
    catch = float(np.mean([m['success'] for m in metas]))
    dists = [m['min_dist'] for m in metas if m.get('min_dist') is not None]
    kept = float(np.mean([bool(m.get('committed')) for m in metas]))
    reasons = {}
    for m in metas:
        if not m.get('committed'):
            reason = m.get('refuse_reason') or 'unknown'
            reasons[reason] = reasons.get(reason, 0) + 1
    n_rows = build_grid(OUT, PLANNED_GRID, N)
    spec.update(catch_rate=catch,
                mean_miss_mm=None if not dists else float(np.mean(dists) * 1000),
                median_miss_mm=None if not dists else float(np.median(dists) * 1000),
                kept_fraction=kept, refuse_counts=reasons,
                grid_rows=int(n_rows), grid=str(PLANNED_GRID))
    spec_path.write_text(json.dumps(spec))
    miss = spec['median_miss_mm']
    miss_s = 'n/a' if miss is None else f'{miss:.2f}'
    print(f'{N} demos, catch {catch:.3f}, kept {kept:.3f}, median miss {miss_s} mm, '
          f'refused {reasons}, {n_rows} grid rows -> {PLANNED_GRID}', flush=True)
    print(f'done in {time.time() - t0:.0f}s', flush=True)
    if catch < 0.98:
        raise SystemExit(f'expert catch at 10 mm is {catch:.3f}; not training on this')


if __name__ == '__main__':
    main()
