"""Reproducible episode collection with clean expert labels on noisy visited states."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np

SPLITS = ('train_dart', 'train_normal', 'validation', 'test')
FEATURES = [('target_history', 18), ('target_history_mask', 6), ('history_time_offsets', 3),
            ('base_pos', 3), ('base_ori_xyzw', 4), ('bearing_sin_cos', 2),
            ('base_velocity_world', 6), ('joint_pos', 17), ('joint_vel', 17),
            ('ee_pos', 3), ('previous_applied_action', 11), ('selected_target_mask', 2),
            ('target_mask', 2), ('time', 1), ('dt', 1)]
PHASES = ('align', 'drive', 'orient', 'done')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def episode_jobs(counts, seed):
    jobs = []
    for split_id, (split, count) in enumerate(zip(SPLITS, counts)):
        for index in range(count):
            seeds = np.random.SeedSequence([seed, split_id, index]).generate_state(2, dtype=np.uint64)
            jobs.append(dict(split=split, index=index, seed=int(seeds[0]), noise_seed=int(seeds[1])))
    if len({job['seed'] for job in jobs}) != len(jobs):
        raise RuntimeError('Episode seed collision')
    return jobs


class EpisodeRecorder:
    def __init__(self, job, config):
        self.job, self.config = job, config
        self.rng = np.random.default_rng(job['noise_seed'])
        self.history = deque(maxlen=3)
        self.previous = np.zeros(11)
        self.noise = np.zeros(11)
        self.noise_until = -np.inf
        self.rows = []
        self.pending = None
        self.step = 0
        self.metadata = {}

    def start(self, targets, selected, dt, joint_names, lower, upper):
        self.dt = float(dt)
        self.stride = max(1, round(1 / (dt * self.config['sample_hz'])))
        self.mask = np.zeros(2, dtype=np.float32)
        self.mask[:len(targets)] = 1
        self.selected = np.eye(2, dtype=np.float32)[selected]
        self.metadata = dict(self.job, schema_version=1, config_hash=self.config['config_hash'],
                             balls=len(targets), selected_target=selected, simulation_dt=dt,
                             record_stride=self.stride, effective_sample_hz=1 / (dt * self.stride),
                             joint_names=joint_names, lower=np.asarray(lower).tolist(), upper=np.asarray(upper).tolist(),
                             trajectories=[dict(start=t.start.tolist(), end=t.end.tolist(), control=t.control.tolist(),
                                                duration=t.duration, launch_speed=t.speed) for t in targets])

    def command(self, state, clean, base_params, arm_params):
        clean = np.asarray(clean, dtype=float)
        target_positions = np.zeros((2, 3), dtype=np.float32)
        target_positions[:len(state['targets'])] = state['targets']
        self.history.append((state['time'], target_positions.copy()))
        positions = np.zeros((3, 2, 3), dtype=np.float32)
        masks = np.zeros((3, 2), dtype=np.float32)
        offsets = np.zeros(3, dtype=np.float32)
        for index, (t, pos) in enumerate(self.history, start=3 - len(self.history)):
            positions[index] = pos
            masks[index] = self.mask
            offsets[index] = t - state['time']
        applied = clean.copy()
        if self.job['split'] == 'train_dart' and state['ready']:
            if state['time'] >= self.noise_until:
                std = np.r_[np.full(4, self.config['wheel_noise_std']), np.full(7, self.config['arm_noise_std'])]
                self.noise = np.clip(self.rng.normal(size=11), -3, 3) * std
                self.noise_until = state['time'] + self.config['noise_hold_seconds']
            applied[:11] += self.noise
            wheel_limit = (base_params.max_lin_vel + base_params.max_ang_vel * base_params.wheel_base_width / 2) / base_params.wheel_radius
            applied[:4] = np.clip(applied[:4], -wheel_limit, wheel_limit)
            applied[4:11] = np.clip(applied[4:11], -arm_params.max_torque, arm_params.max_torque)
        parts = [positions.ravel(), masks.ravel(), offsets, state['base_pos'], state['base_ori'],
                 [np.sin(state['base_bearing']), np.cos(state['base_bearing'])], state['base_velocity'],
                 state['joint_pos'], state['joint_vel'], state['ee_pos'], self.previous.copy(),
                 self.selected, self.mask, [state['time']], [self.dt]]
        x = np.concatenate(parts).astype(np.float32)
        if x.shape != (sum(width for _, width in FEATURES),) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(applied)):
            raise ValueError('Invalid observation/action; episode not saved')
        intercept = np.full(3, np.nan) if state['intercept'] is None else np.asarray(state['intercept']).copy()
        self.pending = dict(X=x, y=clean[:11].astype(np.float32), applied_action=applied[:11].astype(np.float32),
                            expert_command=clean.astype(np.float32), applied_command=applied.astype(np.float32),
                            action_noise=(applied[:11] - clean[:11]).astype(np.float32),
                            time=np.float64(state['time']), target_positions=target_positions,
                            target_mask=self.mask.copy(), selected_target=np.int8(state['selected']),
                            target_history=positions, target_history_mask=masks, history_time_offsets=offsets,
                            base_pos=np.asarray(state['base_pos'], dtype=np.float32),
                            base_ori=np.asarray(state['base_ori'], dtype=np.float32),
                            base_bearing=np.float32(state['base_bearing']),
                            base_velocity=np.asarray(state['base_velocity'], dtype=np.float32),
                            joint_pos=np.asarray(state['joint_pos'], dtype=np.float32),
                            joint_vel=np.asarray(state['joint_vel'], dtype=np.float32),
                            ee_pos=np.asarray(state['ee_pos'], dtype=np.float32),
                            previous_applied_action=self.previous.astype(np.float32),
                            q_des=np.asarray(state['q_des'], dtype=np.float32),
                            qd_des=np.asarray(state['qd_des'], dtype=np.float32),
                            expert_phase=np.int8(3 if state['nav_done'] else PHASES.index(state['phase'])),
                            expert_ready=np.bool_(state['ready']), nav_done=np.bool_(state['nav_done']),
                            intercept=intercept.astype(np.float32),
                            intercept_time=np.float64(np.nan if state['intercept_time'] is None else state['intercept_time']),
                            base_goal=np.asarray(state['base_goal'], dtype=np.float32),
                            base_goal_bearing=np.float32(state['base_goal_bearing']),
                            ee_error=np.float32(state['ee_error']), simulation_step=np.int64(self.step))
        self.previous = applied[:11].copy()
        return applied

    def after_step(self, q_next, base_next, targets_next):
        padded = np.zeros((2, 3), dtype=np.float32)
        padded[:len(targets_next)] = targets_next
        self.pending.update(next_joint_pos=np.asarray(q_next, dtype=np.float32),
                            next_base_pos=np.asarray(base_next, dtype=np.float32), next_target_positions=padded)
        if self.step < 3 or self.step % self.stride == 0:
            self.rows.append(self.pending)
        self.step += 1

    def finish(self, reason, time, error):
        if self.pending is not None and (not self.rows or self.rows[-1]['simulation_step'] != self.pending['simulation_step']):
            self.rows.append(self.pending)
        success = str(reason).startswith('Target reached')
        self.metadata.update(outcome='success' if success else 'step_cap' if 'step cap' in str(reason) else 'stopped',
                             success=success, reason=reason, final_time=time, final_ee_error=error,
                             physics_steps=self.step, samples=len(self.rows))

    def save(self, path):
        if not self.rows or 'outcome' not in self.metadata:
            raise ValueError('Incomplete episode')
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        arrays['terminated'] = np.zeros(len(self.rows), dtype=bool)
        arrays['truncated'] = np.zeros(len(self.rows), dtype=bool)
        arrays['terminated'][-1] = self.metadata['success']
        arrays['truncated'][-1] = not self.metadata['success']
        arrays['metadata_json'] = np.array(json.dumps(self.metadata))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
        return self.metadata


def run_worker(jobfile):
    payload = json.loads(Path(jobfile).read_text())
    os.environ['ROBOENV_HEADLESS'] = '1'
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'robotnik-collection-mpl'))
    from robotnik_arm_task import main
    recorder = EpisodeRecorder(payload['job'], payload['config'])
    main(dict(recorder=recorder, seed=payload['job']['seed'], balls=payload['config']['balls'],
              max_steps=payload['config']['max_steps'], assets_dir=payload['config']['assets_dir'],
              target_speed=payload['config']['target_speed']))
    recorder.save(payload['destination'])


def read_episode(path, job, config_hash):
    with np.load(path, allow_pickle=False) as episode:
        metadata = json.loads(str(episode['metadata_json']))
        if (metadata['config_hash'] != config_hash or metadata['seed'] != job['seed']
                or metadata['split'] != job['split'] or metadata['index'] != job['index']):
            raise ValueError(f'Episode configuration mismatch: {path}')
        if episode['X'].shape != (metadata['samples'], 96) or episode['y'].shape != (metadata['samples'], 11):
            raise ValueError(f'Invalid episode shapes: {path}')
    return metadata


def dependency_check():
    """Check the actual worker interpreter once before scheduling episodes."""
    env = os.environ.copy()
    env['ROBOENV_HEADLESS'] = '1'
    env.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'robotnik-collection-mpl'))
    result = subprocess.run([sys.executable, '-c', 'import robotnik_arm_task'],
                            cwd=Path(__file__).resolve().parent, env=env,
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        details = '\n'.join((result.stderr or result.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f'Simulation dependencies are unavailable in {sys.executable}.\n'
                           f'{details}\nRun the collector with the RoboEnv Python environment used for the simulation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('expert_dataset'))
    parser.add_argument('--balls', type=int, choices=(1, 2), default=2)
    parser.add_argument('--dart', type=int, default=200, help='DART training episodes')
    parser.add_argument('--normal', type=int, default=200, help='Clean training episodes')
    parser.add_argument('--validation', type=int, default=50)
    parser.add_argument('--test', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=60000)
    parser.add_argument('--target-speed', type=float, default=None, help='Optional fixed launch speed in m/s; default random 0.5--0.7')
    parser.add_argument('--sample-hz', type=float, default=100.)
    parser.add_argument('--wheel-noise-std', type=float, default=.08, help='Wheel angular velocity std, rad/s')
    parser.add_argument('--arm-noise-std', type=float, default=.2, help='Arm torque std, N m')
    parser.add_argument('--noise-hold-seconds', type=float, default=.05)
    parser.add_argument('--workers', type=int, default=1, help='Independent subprocesses')
    parser.add_argument('--assets-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--episode-timeout', type=float, default=600.)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
        return
    counts = [args.dart, args.normal, args.validation, args.test]
    if min(counts) < 0 or sum(counts) == 0 or args.seed < 0 or args.max_steps < 1 or args.workers < 1:
        parser.error('Counts must be nonnegative with at least one episode; seed nonnegative; steps/workers positive')
    for name in ('sample_hz', 'noise_hold_seconds', 'episode_timeout'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    for name in ('wheel_noise_std', 'arm_noise_std'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be finite and nonnegative')
    if args.target_speed is not None and (not np.isfinite(args.target_speed) or args.target_speed <= 0):
        parser.error('target-speed must be finite and positive')
    jobs = episode_jobs(counts, args.seed)
    print(f'Plan: {dict(zip(SPLITS, counts))}; {args.balls} ball(s), {len(jobs)} episodes', flush=True)
    if args.dry_run:
        return
    try:
        dependency_check()
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    assets = args.assets_dir.resolve()
    if not (assets / 'configs' / 'robotnik_arm.json').is_file() or not (assets / 'models').is_dir():
        parser.error('--assets-dir must contain configs/robotnik_arm.json and models/')
    folder = args.output.resolve()
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        parser.error('Output folder is not empty; use --resume or a fresh folder')
    folder.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in ('balls', 'seed', 'max_steps', 'sample_hz', 'wheel_noise_std', 'arm_noise_std', 'noise_hold_seconds', 'target_speed')}
    config.update(counts=dict(zip(SPLITS, counts)), assets_dir=str(assets), schema_version=1,
                  source_hashes={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ('collect_expert.py', 'robotnik_arm_task.py', 'robotnik_arm_control.py', 'moving_target.py')})
    config['config_hash'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = folder / 'dataset.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        if args.resume and not any(folder.glob('*/*.npz')):
            atomic_json(folder / 'dataset.previous.json', json.loads(config_path.read_text()))
            print('No episode data was saved; refreshing configuration for this resumed run.', flush=True)
        else:
            parser.error('Existing dataset configuration/source differs; use a fresh output folder')
    atomic_json(config_path, config)
    schema = dict(inputs=[], outputs=[dict(name='wheel_velocity', width=4, units='rad/s'), dict(name='arm_torque', width=7, units='N m')],
                  note='X contains observed state/history and selected target cue; privileged expert state is stored separately.')
    offset = 0
    for name, width in FEATURES:
        schema['inputs'].append(dict(name=name, start=offset, stop=offset + width))
        offset += width
    atomic_json(folder / 'schema.json', schema)
    (folder / '.jobs').mkdir(exist_ok=True)
    (folder / 'logs').mkdir(exist_ok=True)
    results = []
    stop = threading.Event()
    process_lock = threading.Lock()
    processes = set()

    def collect(job):
        tag = f"{job['split']}_{job['index']:05d}"
        destination = folder / job['split'] / f"episode_{job['index']:05d}.npz"
        if destination.exists():
            return read_episode(destination, job, config['config_hash'])
        jobfile = folder / '.jobs' / f'{tag}.json'
        atomic_json(jobfile, dict(job=job, config=config, destination=str(destination)))
        with (folder / 'logs' / f'{tag}.log').open('w') as log:
            with process_lock:
                if stop.is_set():
                    raise RuntimeError('Collection interrupted')
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(jobfile)],
                                           stdout=log, stderr=subprocess.STDOUT)
                processes.add(process)
            try:
                code = process.wait(timeout=args.episode_timeout)
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            finally:
                with process_lock:
                    processes.discard(process)
        return read_episode(destination, job, config['config_hash'])

    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(collect, job): job for job in jobs}
    try:
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log_path = folder / 'logs' / f"{job['split']}_{job['index']:05d}.log"
                details = '\n'.join(log_path.read_text(errors='replace').splitlines()[-12:]) if log_path.exists() else ''
                result = dict(job, outcome='error', error=str(exc), log=str(log_path), details=details)
                print(f"Episode failed: {exc}\nLog: {log_path}\n{details}", file=sys.stderr, flush=True)
            results.append(result)
            atomic_json(folder / 'manifest.json', sorted(results, key=lambda row: (SPLITS.index(row['split']), row['index'])))
            print(f"[{len(results)}/{len(jobs)}] {job['split']} {job['index']}: {result['outcome']}", flush=True)
    except KeyboardInterrupt:
        stop.set()
        with process_lock:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        print('Interrupted; completed episode files are retained. Use --resume with the same settings.', flush=True)
        raise SystemExit(130)
    finally:
        pool.shutdown(wait=True, cancel_futures=stop.is_set())
    failures = sum(row['outcome'] == 'error' for row in results)
    print(f'Finished: {len(results) - failures} episodes saved, {failures} errors. Dataset: {folder}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
