"""Hand the two-ball robots the expert's state at points through the drive.

Times are measured from the start of the episode, not backwards from the catch.
The learner parks long before the ball arrives, and a miss of about a metre is a
base that never reached the parking pose. The times are:

    launch                  the policy drives the whole episode
    before_commit           0.5 s before the oracle's last change of nearer ball
    after_commit            0.5 s after that change
    drive_mid               halfway through the expert's drive phase
    expert_stop             the tick the expert's base finishes parking
    arm                     0.5 s before the expert ball's planned catch

The squared-error robot, the oracle and the mixture each take over at every
time. A poor mixture number is read against the other two, so a bad hand-off
is not mistaken for a bad mixture.
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SCENES = ROOT / 'data' / 'two_ball' / 'test_scenes.json'
OUT = RESULTS / 'stage2' / 'handoff_drive.json'
ROLES = ('mse', 'oracle', 'mdn')
NAMES = {
    'mse': 'twoball_mse_s{seed}',
    'oracle': 'twoball_oracle_s{seed}',
    'mdn': 'twoball_mdn_s{seed}',
}
ORDER = ('launch', 'before_commit', 'after_commit', 'drive_mid', 'expert_stop', 'arm')
RADIUS_MM = 50.0
MARGIN = 0.5
DIVERGE_M = 0.20
LATE = ('expert_stop', 'arm')
EARLY = ('launch', 'before_commit')

_env = None
_models = None
_scenes = None


def _model_path(role, seed):
    path = RESULTS / 'models' / f'{NAMES[role].format(seed=seed)}.npz'
    if not path.exists():
        raise SystemExit(f'missing {path}')
    return path


def init_worker(seed, scenes_path):
    global _env, _models, _scenes
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    _env = CatchEnv()
    _models = {role: NumpyPolicy(_model_path(role, seed)) for role in ROLES}
    _scenes = json.loads(Path(scenes_path).read_text())


def _scene(index):
    from robotnik_arm_control import make_catchable_pair
    scene = _scenes[index]
    env = _env
    targets, selected, plans = make_catchable_pair(
        int(scene['seed']), None, env.lower, env.upper, env.init_base_pos, env.init_bearing, env.init_ee,
        env.dt, env.base_params, env.arm_params, target_count=2)
    gap = float(np.linalg.norm(np.asarray(plans[0].intercept) - np.asarray(plans[1].intercept)) * 1000)
    if abs(gap - float(scene['separation_mm'])) > 1.0 or int(selected) != int(scene['expert_slot']):
        raise RuntimeError(f'scene {index} did not reproduce')
    intercepts = [np.asarray(plan.intercept[:2], float) for plan in plans]
    return scene, targets, plans, intercepts


def _nearer(base, intercepts):
    xy = np.asarray(base[:2], float)
    return int(np.argmin([np.linalg.norm(xy - p) for p in intercepts]))


def _driver(role, oracle_slot):
    from np_policy import MDNDriver, SetPolicyDriver
    if role == 'mdn':
        return MDNDriver(_models[role], 'weight')
    slot = oracle_slot if role == 'oracle' else None
    return SetPolicyDriver(_models[role], slot)


def _pose_error(base_xy, park_xy, park_bearing):
    if base_xy is None or park_xy is None:
        return None, None, None, None
    delta = np.asarray(base_xy[:2], float) - np.asarray(park_xy, float)
    park_mm = float(np.linalg.norm(delta) * 1000)
    along = cross = heading = None
    if park_bearing is not None:
        forward = np.array([np.cos(park_bearing), np.sin(park_bearing)])
        left = np.array([-forward[1], forward[0]])
        along = float(np.dot(delta, forward) * 1000)
        cross = float(np.dot(delta, left) * 1000)
        if len(base_xy) > 2:
            err = (base_xy[2] - park_bearing + np.pi) % (2 * np.pi) - np.pi
            heading = float(np.degrees(err))
    return park_mm, along, cross, heading


def _score(best, nearest, expert_slot, final_base, catch_base, park_xy, park_bearing,
           tick_aims, choices, choice_aims, diverge_s, switch_times):
    from eval_two_ball import _between
    misses = [None if v is None else float(v * 1000) for v in best]
    finite = [v for v in misses if v is not None]
    caught = [v is not None and v < RADIUS_MM for v in misses]
    either = bool(any(caught))
    expert_hit = bool(caught[expert_slot]) if expert_slot < len(caught) else False
    switches = match = aligned = near_diverge = None
    if choices:
        switches = int(sum(a != b for a, b in zip(choices, choices[1:])))
        dominant = Counter(choices).most_common(1)[0][0]
        aimed = [a == expert_slot for a, c in zip(choice_aims, choices) if c == dominant]
        match = None if not aimed else float(np.mean(aimed))
        aim_changes = [i for i in range(1, len(choice_aims)) if choice_aims[i] != choice_aims[i - 1]]
        comp_changes = [i for i in range(1, len(choices)) if choices[i] != choices[i - 1]]
        if comp_changes:
            aligned = float(np.mean([any(abs(i - j) <= 1 for j in aim_changes) for i in comp_changes]))
        if diverge_s is not None and switch_times:
            near_diverge = bool(any(abs(t - diverge_s) <= 1.0 for t in switch_times))
        elif diverge_s is not None:
            near_diverge = False
    park_mm, along, cross, heading = _pose_error(final_base, park_xy, park_bearing)
    catch_mm, _, _, _ = _pose_error(catch_base, park_xy, park_bearing)
    final_aim = None if not tick_aims else int(tick_aims[-1])
    return dict(
        catch_either=either,
        catch_expert=expert_hit,
        miss_either_mm=None if not finite else float(min(finite)),
        miss_expert_mm=misses[expert_slot] if expert_slot < len(misses) else None,
        between=False if either else _between(nearest),
        component_switches=switches,
        winning_matches_expert=match,
        switch_near_aim=aligned,
        switch_at_diverge=None if near_diverge is None else float(near_diverge),
        final_aim_expert=None if final_aim is None else bool(final_aim == expert_slot),
        park_error_mm=park_mm,
        park_at_catch_mm=catch_mm,
        park_along_mm=along,
        park_cross_mm=cross,
        heading_error_deg=heading,
        diverge_s=diverge_s)


def _capture(env, step, cmd, frames, t):
    return dict(
        state_id=int(env.client.saveState()), step=int(step), t=float(t),
        cmd=np.array(cmd, copy=True), frames=[row.copy() for row in frames],
        slot_left=list(env.slot_left))


def _run_from(env, targets, expert_slot, intercepts, end_time, driver, snap, expert_path):
    """Continue one episode from a saved expert state. ``snap`` None starts at launch."""
    from catch_env import FIRST_TICK_STEP
    if snap is None:
        env.set_scene(targets, expert_slot)
        driver.reset(env)
        step = 0
        cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    else:
        env.client.restoreState(snap['state_id'])
        env._read_state()
        env.slot_left = list(snap['slot_left'])
        driver.reset(env)
        driver.frames = [row.copy() for row in snap['frames']]
        step = int(snap['step'])
        cmd = np.array(snap['cmd'], copy=True)
    stride = int(round(1.0 / (driver.cfg.rate_hz * env.dt)))
    best = [None] * len(targets)
    nearest = None
    catch_base = None
    tick_aims, tick_times = [], []
    choice_aims, switch_times = [], []
    diverge_s = None
    while True:
        t = step * env.dt
        positions = None
        for slot in range(len(targets)):
            dist = env.slot_distance(slot, t)
            if dist is None:
                continue
            if best[slot] is None or dist < best[slot]:
                best[slot] = dist
            if nearest is None or dist < nearest['dist']:
                if positions is None:
                    positions = [target.position(t).copy() for target in targets]
                nearest = dict(dist=float(dist), ee=env.ee.copy(), balls=positions)
                catch_base = np.array([env.base_pos[0], env.base_pos[1], env.bearing], float)
        if t >= end_time:
            break
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            aim = _nearer(env.base_pos, intercepts)
            tick_aims.append(aim)
            tick_times.append(float(t))
            if expert_path is not None and step in expert_path and diverge_s is None:
                delta = env.base_pos[:2] - expert_path[step]
                if float(np.linalg.norm(delta)) > DIVERGE_M:
                    diverge_s = float(t)
            n_before = len(getattr(driver, 'choices', []))
            cmd, _ready = driver.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
            choices = getattr(driver, 'choices', None)
            if choices is not None and len(choices) > n_before:
                choice_aims.append(aim)
                if len(choices) >= 2 and choices[-1] != choices[-2]:
                    switch_times.append(float(t))
        q_ref = env.clip_reference(cmd[4:11])
        env.step(cmd[:4], env.arm_torque(q_ref, cmd[11:18]))
        step += 1
    final_base = np.array([env.base_pos[0], env.base_pos[1], env.bearing], float)
    if catch_base is None:
        catch_base = final_base
    choices = list(getattr(driver, 'choices', []))
    return dict(best=best, nearest=nearest, final_base=final_base, catch_base=catch_base,
                tick_aims=tick_aims, tick_times=tick_times, choices=choices,
                choice_aims=choice_aims, diverge_s=diverge_s, switch_times=switch_times)


def _expert_pass(env, targets, expert_slot, known, until=None, discover=True):
    """Run the expert once. Save the hand-off states and the base path.

    ``known`` holds the times that do not depend on the expert (just before and
    just after the oracle commits, and half a second before the catch). The
    drive midpoint and the parking tick are found on this same pass. The leave
    latch for each ball is updated every step, so a late hand-off still counts
    a ball that left the box while the expert was driving.
    """
    from catch_env import ExpertDriver, FIRST_TICK_STEP
    env.set_scene(targets, expert_slot)
    expert = ExpertDriver()
    expert.reset(env)
    stride = int(round(1.0 / (25.0 * env.dt)))
    pending = sorted((t, name) for name, t in known.items())
    snaps = {}
    frames = []
    path = {}
    park = None
    drive_on = drive_off = None
    drive_marks = []
    drive_ticks = 0
    drive_closed = False
    cmd = np.concatenate((np.zeros(4), env.q_init_arm, np.zeros(7)))
    step = 0
    end_time = max(target.duration for target in targets) + 1.0

    def close_drive():
        nonlocal drive_marks, drive_closed
        if not drive_marks or drive_closed:
            drive_closed = True
            return
        mid = 0.5 * (drive_on + drive_off)
        keep = min(drive_marks, key=lambda item: abs(item[0] - mid))
        snaps['drive_mid'] = keep[1]
        for _t, mark in drive_marks:
            if mark['state_id'] != keep[1]['state_id']:
                env.client.removeState(mark['state_id'])
        drive_marks = []
        drive_closed = True

    while True:
        t = step * env.dt
        for slot in range(len(targets)):
            env.slot_distance(slot, t)
        expert.sense(env, t)
        if t >= end_time or (until is not None and not pending and t > until + 1e-9):
            break
        if step >= FIRST_TICK_STEP and (step - FIRST_TICK_STEP) % stride == 0:
            while pending and t + 1e-12 >= pending[0][0]:
                _when, name = pending.pop(0)
                snaps[name] = _capture(env, step, cmd, frames, t)
            phase = 'done' if expert.base_state.done else expert.base_state.phase
            if discover and not drive_closed and phase == 'drive':
                drive_on = t if drive_on is None else drive_on
                drive_off = t
                # A mark every fifth tick is 0.2 s, enough to name the middle of the drive.
                if drive_ticks % 5 == 0:
                    drive_marks.append((float(t), _capture(env, step, cmd, frames, t)))
                drive_ticks += 1
            elif discover and not drive_closed and drive_on is not None:
                close_drive()
            if discover and expert.base_state.done and park is None:
                park = dict(xy=env.base_pos[:2].copy(), bearing=float(env.bearing), t=float(t))
                snaps['expert_stop'] = _capture(env, step, cmd, frames, t)
            path[step] = env.base_pos[:2].copy()
            frames.append(env.obs_set(t))
            if len(frames) > 64:
                del frames[:-64]
            cmd, _ready = expert.act(env, t, stride * env.dt)
            cmd = np.asarray(cmd, dtype=float)
        q_ref = env.clip_reference(cmd[4:11])
        env.step(cmd[:4], env.arm_torque(q_ref, cmd[11:18]))
        step += 1
    if discover:
        close_drive()
    for _when, name in pending:
        snaps.setdefault(name, None)
    return dict(snaps=snaps, park=park, drive_on=drive_on, drive_off=drive_off, path=path)


def _times(commit, drive_on, drive_off, stop_t, catch_t):
    times = {}
    if commit is not None and commit - MARGIN > 0.15:
        times['before_commit'] = float(commit - MARGIN)
    if commit is not None:
        times['after_commit'] = float(commit + MARGIN)
    if drive_on is not None and drive_off is not None:
        times['drive_mid'] = float(0.5 * (drive_on + drive_off))
    if stop_t is not None:
        times['expert_stop'] = float(stop_t)
    if catch_t is not None and catch_t - MARGIN > 0.15:
        times['arm'] = float(catch_t - MARGIN)
    return times


def _commit(aims, aim_times):
    changes = [i for i in range(1, len(aims)) if aims[i] != aims[i - 1]]
    if not changes:
        return 0.0, False
    return float(aim_times[changes[-1]]), True


def _finish(run, expert_slot, park_xy, park_bearing):
    return _score(
        run['best'], run['nearest'], expert_slot, run['final_base'], run['catch_base'],
        park_xy, park_bearing, run['tick_aims'], run['choices'], run['choice_aims'],
        run['diverge_s'], run['switch_times'])


def job(index):
    scene, targets, plans, intercepts = _scene(index)
    env = _env
    expert_slot = int(scene['expert_slot'])
    oracle_slot = int(scene['oracle_slot'])
    catch_t = float(plans[expert_slot].intercept_time)
    end_time = max(target.duration for target in targets) + 1.0
    # The arm time is known from the plan. Drive midpoint and the parking tick
    # are found on this pass, which also records the expert's base path.
    known = {}
    if catch_t - MARGIN > 0.15:
        known['arm'] = float(catch_t - MARGIN)
    taken = _expert_pass(env, targets, expert_slot, known)
    park = taken['park']
    park_xy = None if park is None else park['xy']
    park_bearing = None if park is None else park['bearing']
    path = taken['path']
    oracle = _driver('oracle', oracle_slot)
    oracle_run = _run_from(env, targets, expert_slot, intercepts, end_time, oracle, None, path)
    commit, switched = _commit(oracle_run['tick_aims'], oracle_run['tick_times'])
    early = {'after_commit': float(commit + MARGIN)}
    if commit - MARGIN > 0.15:
        early['before_commit'] = float(commit - MARGIN)
    early_pass = _expert_pass(
        env, targets, expert_slot, early, until=max(early.values()), discover=False)
    for name, snap in early_pass['snaps'].items():
        if snap:
            taken['snaps'][name] = snap
    out = dict(
        index=int(index), expert_slot=expert_slot, oracle_slot=oracle_slot,
        commit_s=commit, commit_switched=switched,
        drive_mid_s=None if not taken['snaps'].get('drive_mid') else taken['snaps']['drive_mid']['t'],
        expert_stop_s=None if park is None else park['t'],
        arm_s=known.get('arm'), catch_s=catch_t)
    roles = {'oracle': {'launch': _finish(oracle_run, expert_slot, park_xy, park_bearing)}}
    for role in ('mse', 'mdn'):
        driver = _driver(role, oracle_slot)
        run = _run_from(env, targets, expert_slot, intercepts, end_time, driver, None, taken['path'])
        roles[role] = {'launch': _finish(run, expert_slot, park_xy, park_bearing)}
    for name in ORDER:
        if name == 'launch':
            continue
        snap = taken['snaps'].get(name)
        if not snap:
            for role in ROLES:
                roles[role][name] = dict(skipped='no state at this time')
            continue
        for role in ROLES:
            driver = _driver(role, oracle_slot)
            run = _run_from(
                env, targets, expert_slot, intercepts, end_time, driver, snap, taken['path'])
            roles[role][name] = _finish(run, expert_slot, park_xy, park_bearing)
        env.client.removeState(snap['state_id'])
    out['roles'] = roles
    return out


def _kept(values):
    return [v for v in values if v is not None and np.isfinite(v)]


def _mean(values):
    vals = _kept(values)
    if not vals:
        return None
    return float(np.mean(vals))


def _median(values):
    vals = _kept(values)
    if not vals:
        return None
    return float(np.median(vals))


def _pool(rows, role, name):
    got = []
    for row in rows:
        cell = row['roles'][role].get(name)
        if cell and 'skipped' not in cell:
            got.append(cell)
    misses = [c for c in got if not c['catch_either']]
    between = [c for c in misses if c['between']]
    return dict(
        n=len(got),
        n_skipped=sum(1 for row in rows if 'skipped' in row['roles'][role].get(name, {})),
        catch_either=_mean([c['catch_either'] for c in got]),
        catch_expert=_mean([c['catch_expert'] for c in got]),
        miss_either_mm=_mean([c['miss_either_mm'] for c in got]),
        between_miss_rate=None if not misses else float(len(between) / len(misses)),
        component_switches=_mean([c['component_switches'] for c in got]),
        winning_matches_expert=_mean([c['winning_matches_expert'] for c in got]),
        switch_near_aim=_mean([c['switch_near_aim'] for c in got]),
        switch_at_diverge=_mean([c['switch_at_diverge'] for c in got]),
        final_aim_expert=_mean([float(c['final_aim_expert']) for c in got if c['final_aim_expert'] is not None]),
        park_error_mm=_mean([c['park_error_mm'] for c in got]),
        park_error_median_mm=_median([c['park_error_mm'] for c in got]),
        park_at_catch_mm=_mean([c['park_at_catch_mm'] for c in got]),
        park_along_mm=_mean([c['park_along_mm'] for c in got]),
        park_cross_mm=_mean([c['park_cross_mm'] for c in got]),
        diverge_s=_mean([c['diverge_s'] for c in got]),
        diverge_rate=None if not got else float(np.mean([c['diverge_s'] is not None for c in got])))


def _timing(rows):
    def col(key):
        return _mean([r[key] for r in rows if r.get(key) is not None])
    switched = [r for r in rows if r['commit_switched']]
    return dict(
        n=len(rows),
        commit_switched=float(np.mean([r['commit_switched'] for r in rows])) if rows else None,
        commit_s_mean=col('commit_s'),
        commit_s_when_switched=_mean([r['commit_s'] for r in switched]),
        drive_mid_s=col('drive_mid_s'),
        expert_stop_s=col('expert_stop_s'),
        arm_s=col('arm_s'),
        catch_s=col('catch_s'))


def _gap(block, name, role, other):
    a = block[name][role]['catch_either']
    b = block[name][other]['catch_either']
    if a is None or b is None:
        return None
    return float(b - a)


def _reading(block):
    """Which of the three hand-off stories the catch rates support."""
    def catch(name, role):
        cell = block.get(name, {}).get(role, {})
        return cell.get('catch_either')
    late_mdn = [catch(name, 'mdn') for name in LATE]
    late_mse = [catch(name, 'mse') for name in LATE]
    late_oracle = [catch(name, 'oracle') for name in LATE]
    early_mdn = catch('launch', 'mdn')
    early_oracle = catch('launch', 'oracle')
    after = block.get('after_commit', {}).get('mdn', {})
    stories = []
    if all(v is not None for v in late_mdn + late_mse) and all(m + 0.15 < s for m, s in zip(late_mdn, late_mse)):
        stories.append('bad_from_good_states')
    if (all(v is not None for v in late_mdn + late_oracle) and early_mdn is not None and early_oracle is not None
            and all(abs(m - o) <= 0.15 for m, o in zip(late_mdn, late_oracle))
            and early_mdn + 0.20 < min(late_mdn)
            and early_oracle + 0.15 >= min(late_oracle)):
        stories.append('builds_during_drive')
    keeps = after.get('winning_matches_expert')
    if keeps is None:
        keeps = after.get('final_aim_expert')
    if (after.get('catch_expert') is not None and after['catch_expert'] >= 0.50 and keeps is not None and keeps >= 0.60
            and early_mdn is not None and early_mdn < 0.20):
        stories.append('carries_a_choice')
    text = {
        'bad_from_good_states':
            'The mixture is worse than squared error even from the late hand-offs, so it acts badly from states the expert has already made good. The next check is the weights on held-out states: the highest-weight component against the best-matching one.',
        'builds_during_drive':
            'The mixture matches the oracle from the late hand-offs and fails from the start, while the oracle does not. The error builds during the drive. See whether component switches line up with the base leaving the expert\'s parking pose.',
        'carries_a_choice':
            'From the hand-off just after the oracle commits, the mixture keeps the expert\'s ball and catches, and from the launch it does not. It can carry a choice and does not make one. That points at a head that first classifies the ball.',
    }
    return dict(stories=stories, text=[text[name] for name in stories])


def _print_block(block):
    print(f'{"handoff":<16} {"role":<8} {"either":>8} {"expert":>8} {"miss":>8} {"between":>8} {"park":>8} {"aim":>6} {"sw":>6}', flush=True)
    for name in ORDER:
        for role in ROLES:
            cell = block[name][role]
            def fmt(key, scale=1, digits=3):
                value = cell.get(key)
                if value is None:
                    return f'{"—":>8}'
                return f'{value * scale:>8.{digits}f}'
            print(f'{name:<16} {role:<8} {fmt("catch_either")} {fmt("catch_expert")} '
                  f'{fmt("miss_either_mm", 1, 0)} {fmt("between_miss_rate")} {fmt("park_error_mm", 1, 0)} '
                  f'{fmt("final_aim_expert")} {fmt("component_switches", 1, 2)}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(',') if s)
    scenes = json.loads(SCENES.read_text())
    n = len(scenes) if not args.limit else min(args.limit, len(scenes))
    report = dict(
        definitions=dict(
            launch='the policy acts from the first tick',
            before_commit='0.5 s before the oracle last changes which ball its base is nearer',
            after_commit='0.5 s after that change',
            drive_mid='halfway from the expert entering its drive phase to leaving it',
            expert_stop='the tick the expert base first reports that parking is done',
            arm='0.5 s before the planned catch of the ball the expert is heading for',
            catch_expert='hand within 50 mm of the ball the expert was heading for',
            catch_either='hand within 50 mm of either ball',
            between='among misses, the hand at its closest approach was nearer the midpoint than either ball',
            winning_matches_expert='while the most-used component was selected, the share of replans whose nearer landing was the expert ball',
            park_error='base position at the closest approach, against where the expert had parked, in mm',
            commitment='measured on the oracle seed of this run, from its own launch hand-off'),
        seeds=list(seeds), n_scenes=n, per_seed={})
    from stage2_log import finish_run, start_run
    log = not args.no_log
    record_path = record = None
    if log:
        _run_id, record_path, record = start_run(
            'handoff-drive', 'mdn', dict(seeds=list(seeds), scenes=n), ['handoff_drive.py'],
            dict(scenes=n), note='expert hand-off at launch, commitment, mid-drive, stop and arm; mse, oracle and mdn')
    for seed in seeds:
        print(f'seed {seed} on {n} scenes', flush=True)
        rows = []
        with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(seed, str(SCENES))) as pool:
            futs = [pool.submit(job, i) for i in range(n)]
            for done, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if done % 10 == 0 or done == n:
                    print(f'  {done}/{n}', flush=True)
                if done % 20 == 0:
                    partial = args.out.with_name(args.out.stem + f'.s{seed}.partial.json')
                    partial.write_text(json.dumps(dict(seed=seed, n_done=done, rows=rows)))
        rows.sort(key=lambda r: r['index'])
        block = {}
        for name in ORDER:
            block[name] = {role: _pool(rows, role, name) for role in ROLES}
        report['per_seed'][f's{seed}'] = dict(timing=_timing(rows), handoffs=block)
        _print_block(block)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report))
    pooled = {}
    for name in ORDER:
        pooled[name] = {}
        for role in ROLES:
            cells = [report['per_seed'][f's{seed}']['handoffs'][name][role] for seed in seeds]
            pooled[name][role] = {
                key: _mean([c[key] for c in cells]) for key in cells[0] if key != 'n_skipped'}
            pooled[name][role]['n'] = int(round(_mean([c['n'] for c in cells])))
    report['pooled'] = pooled
    report['reading'] = _reading(pooled)
    report['timing'] = {
        key: _mean([report['per_seed'][f's{seed}']['timing'][key] for seed in seeds])
        for key in report['per_seed'][f's{seeds[0]}']['timing']
    }
    args.out.write_text(json.dumps(report))
    print(json.dumps(dict(timing=report['timing'], reading=report['reading']), indent=2), flush=True)
    print(f'wrote {args.out}', flush=True)
    if log:
        launch = pooled['launch']['mdn']
        finish_run(record_path, record, dict(
            miss_mm=launch.get('miss_either_mm'), catch_rate=launch.get('catch_either')), [args.out])


if __name__ == '__main__':
    main()
