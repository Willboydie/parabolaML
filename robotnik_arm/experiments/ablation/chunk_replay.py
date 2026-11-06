"""Replay the ceiling's chunk open-loop from the expert state before the stop.

The expert runs at 1 kHz. One second before its stop the DeepSets ceiling is
shown the expert's own history and queried once. The whole predicted chunk,
2 s at 25 Hz, is then held for 40 ms per step. Nothing is replanned. The
position one second after the stop is the end of that chunk; the position at
the stop is the end of the 1 s the robot would execute before replanning.

The along-track gap is the replay base minus the expert base, in the expert's
route frame, the same measurement as ``hold_replay.py``.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hold_replay import (
    SWITCH_BEFORE_S, along_at, command_extra_mm, mean_ci, simulate, stop_index,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'stage2' / 'chunk_replay.json'
CEILING = 'r25_p2_e0.5_hstack0.5s_d2000_deepsets'
SEEDS = (0, 1, 2)
HOLD_S = 0.040

_models = None


def init_worker():
    global _models
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from catch_env import CatchEnv
    from np_policy import NumpyPolicy
    import hold_replay
    hold_replay._env = CatchEnv()
    _models = []
    for seed in SEEDS:
        path = RESULTS / 'models' / f'{CEILING}_s{seed}.npz'
        model = NumpyPolicy(path)
        cfg = model.cfg
        if (cfg.rate_hz, cfg.horizon, cfg.execute, cfg.frames) != (25, 50, 25, 13):
            raise RuntimeError((cfg.rate_hz, cfg.horizon, cfg.execute, cfg.frames))
        _models.append(model)


def history_at(obs, switch, frames, hold):
    """``frames`` expert observations ending at ``switch``, spaced one hold apart."""
    idx = switch - np.arange(frames - 1, -1, -1) * hold
    idx = np.clip(idx, 0, len(obs) - 1)
    return obs[idx]


def absolute_chunk(chunk, q_ref):
    """Arm joint targets are stored relative to the arm position at the query."""
    out = np.array(chunk, dtype=np.float64, copy=True)
    out[:, 4:11] += q_ref
    return out


def held_commands(expert_cmd, chunk, switch, hold):
    """Expert commands up to the switch, then the chunk, one step held for ``hold`` ticks."""
    logged = np.array(expert_cmd, dtype=np.float64, copy=True)
    for step in range(switch, len(logged)):
        k = min((step - switch) // hold, len(chunk) - 1)
        logged[step] = chunk[k]
    return logged


def job(seed):
    from catch_env import OBS_NAMES
    ref = simulate(seed, record_obs=True)
    stop = stop_index(ref['speed'], ref['dt'])
    switch = None if stop is None else stop - int(round(SWITCH_BEFORE_S / ref['dt']))
    if stop is None or ref['chord'] is None or switch is None or switch < 2:
        return dict(scene=int(seed), skipped='no quiet stop' if stop is None else 'switch too early')
    hold = int(round(HOLD_S / ref['dt']))
    later = stop + int(round(1.0 / ref['dt']))
    q_col = OBS_NAMES.index('arm_q_0')
    q_ref = ref['obs'][switch, q_col:q_col + 7].astype(np.float64)
    per_seed = {}
    for model_seed, model in zip(SEEDS, _models):
        frames = history_at(ref['obs'], switch, model.cfg.frames, hold)
        chunk = absolute_chunk(model.predict(frames), q_ref)
        logged = held_commands(ref['cmd'], chunk, switch, hold)
        rep = simulate(seed, logged=logged, switch=switch, hold=1)
        per_seed[f's{model_seed}'] = dict(
            at_switch_mm=along_at(rep['base'], ref['base'], ref['chord'], switch),
            at_stop_mm=along_at(rep['base'], ref['base'], ref['chord'], stop),
            one_s_after_stop_mm=along_at(rep['base'], ref['base'], ref['chord'], later),
            at_catch_mm=along_at(rep['base'], ref['base'], ref['chord'], ref['catch_step']),
            command_extra_mm=command_extra_mm(rep['speed'], ref['speed'], ref['dt'], switch, later),
            command_extra_execute_mm=command_extra_mm(rep['speed'], ref['speed'], ref['dt'], switch, stop),
        )
    return dict(scene=int(seed), stop_time=float(stop * ref['dt']), seeds=per_seed)


def summarise(rows):
    used = [r for r in rows if 'seeds' in r]
    keys = ('at_switch_mm', 'at_stop_mm', 'one_s_after_stop_mm', 'at_catch_mm',
            'command_extra_mm', 'command_extra_execute_mm')
    per_seed = {}
    for model_seed in SEEDS:
        name = f's{model_seed}'
        block = {}
        for key in keys:
            stat = mean_ci([r['seeds'][name][key] for r in used])
            if key.startswith('command_'):
                stat.pop('frac_ahead', None)
            block[key] = stat
        per_seed[name] = block
    mean = {}
    for key in keys:
        vals = [per_seed[f's{s}'][key]['mean'] for s in SEEDS]
        mean[key] = float(np.mean(vals))
    return per_seed, mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    test = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][:args.limit]
    full = args.limit == 200
    t0 = time.time()
    run_id = record_path = record = None
    if full:
        from stage2_log import start_run
        run_id, record_path, record = start_run(
            'chunk-replay', CEILING, dict(switch_before_stop_s=SWITCH_BEFORE_S, hold_s=HOLD_S),
            sys.argv, dict(test=len(test), seeds=list(SEEDS)),
            note='one open-loop chunk of the DeepSets ceiling from 1 s before the expert stop')
    print(f'chunk replay on {len(test)} test scenes', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(job, int(s)) for s in test]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 25 == 0 or n == len(futs):
                print(f'  {n}/{len(futs)}  {time.time() - t0:.0f}s', flush=True)
    per_seed, mean = summarise(rows)
    skipped = [r for r in rows if 'skipped' in r]
    report = dict(
        switch_before_stop_s=SWITCH_BEFORE_S, hold_s=HOLD_S,
        chunk='full 2 s prediction, each step held 40 ms, no replan',
        n_test=len(test), n_used=len(rows) - len(skipped), n_skipped=len(skipped),
        per_seed=per_seed, mean=mean,
    )
    for model_seed in SEEDS:
        gap = per_seed[f's{model_seed}']['one_s_after_stop_mm']
        stop = per_seed[f's{model_seed}']['at_stop_mm']
        extra = per_seed[f's{model_seed}']['command_extra_mm']
        print(f"seed {model_seed}  1s-after {gap['mean']:+.2f} mm  "
              f"at stop {stop['mean']:+.2f}  command {extra['mean']:+.2f}  "
              f"ahead {gap['frac_ahead']:.2f}", flush=True)
    print(f"mean     1s-after {mean['one_s_after_stop_mm']:+.2f} mm  "
          f"at stop {mean['at_stop_mm']:+.2f}  command {mean['command_extra_mm']:+.2f}", flush=True)
    if full:
        from stage2_log import finish_run
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2))
        finish_run(record_path, record, dict(miss_mm=mean['one_s_after_stop_mm']), [OUT])
        print(f'wrote {OUT.name} in {time.time() - t0:.0f}s', flush=True)
        print(f'run {run_id}', flush=True)
    else:
        print(f'smoke done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
