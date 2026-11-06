"""Action analysis of the flat rel_add 10 Hz baseline, and the hypotheses it supports.

Thresholds below are the decision rules. They are not fit to the ablation.
The sweep driver refuses to start until this script has written hypotheses.json.
"""
import json
import time
from pathlib import Path

import numpy as np

import diagnose
from catch_env import FIRST_TICK_STEP
from np_policy import NumpyPolicy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
STUDY = ROOT / 'results' / 'rel_study'

# Pre-registered decision thresholds. A factor is "not indicated" unless the
# inequality holds. The sweep still runs every factor; this file only records
# what the baseline measurements predict.
RATE_RATIO = 1.5          # wheel-command RMS over 100 ms vs over 20 ms
HORIZON_RATIO = 2.0       # open-loop MSE at the last chunk step vs the first
EXECUTE_RATIO = 1.5       # hand-off miss after 1.0 s of control vs 0.5 s
HISTORY_RATIO = 2.0       # open-loop MSE in the turn vs on the drive
SHORT_RATIO = 2.0         # expert ||cmd(t+0.5s) - cmd(t)|| vs ||cmd(t+0.1s) - cmd(t)||
FACTORS = ('rate', 'history', 'predict_horizon', 'execute_window', 'short_chunk')


def _rms(diff):
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def command_variation(n_demos=100):
    """How far the expert command moves inside one tick of each policy rate.

    Holds are counted on the 50 Hz grid: 20 ms, 40 ms, 100 ms, 200 ms.
    """
    holds = {50: 1, 25: 2, 10: 5, 5: 10}
    wheel = {hz: [] for hz in holds}
    full = {hz: [] for hz in holds}
    short_ratios = []
    used = 0
    for i in range(n_demos):
        path = DATA / 'demos' / f'demo_{i:04d}.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            cmd = np.asarray(z['cmd'][FIRST_TICK_STEP::20], dtype=np.float64)
        if len(cmd) < 30:
            continue
        used += 1
        for hz, step in holds.items():
            if len(cmd) <= step:
                continue
            delta = cmd[step:] - cmd[:-step]
            wheel[hz].append(delta[:, :4])
            full[hz].append(delta)
        # 0.5 s is 25 grid steps; 0.1 s is 5. Sample at the baseline's 10 Hz ticks.
        idx = np.arange(0, len(cmd) - 25, 5)
        if len(idx):
            far = cmd[idx + 25] - cmd[idx]
            near = cmd[idx + 5] - cmd[idx]
            far_n = np.linalg.norm(far[:, :4], axis=1)
            near_n = np.linalg.norm(near[:, :4], axis=1)
            ok = near_n > 1e-3
            if ok.any():
                short_ratios.append(far_n[ok] / near_n[ok])
    out = {}
    for hz in holds:
        w = np.concatenate(wheel[hz]) if wheel[hz] else np.zeros((1, 4))
        f = np.concatenate(full[hz]) if full[hz] else np.zeros((1, 18))
        out[str(hz)] = dict(wheel_rms=_rms(w), command_rms=_rms(f), n=int(len(w)))
    ratios = np.concatenate(short_ratios) if short_ratios else np.array([np.nan])
    out['n_demos'] = used
    out['wheel_rms_100ms_over_20ms'] = out['10']['wheel_rms'] / max(out['50']['wheel_rms'], 1e-9)
    out['wheel_rms_200ms_over_100ms'] = out['5']['wheel_rms'] / max(out['10']['wheel_rms'], 1e-9)
    out['short_chunk_median_ratio'] = float(np.median(ratios))
    out['short_chunk_n'] = int(ratios.size)
    return out


def _tau(rows, tau):
    for row in rows:
        if abs(float(row['tau']) - tau) < 1e-9:
            return row
    return None


def hypotheses_from(measurements):
    """Map measurements through the thresholds. No ablation result is an input."""
    cmd = measurements['command_variation']
    test = measurements['test']
    part3 = measurements['part3']
    mse_h = test['mse_horizon']
    horizon_ratio = float(mse_h[-1] / max(mse_h[0], 1e-12))
    turn, drive = test.get('mse_turn'), test.get('mse_drive')
    history_ratio = None if turn is None or drive is None or drive == 0 else float(turn / drive)
    miss_1 = _tau(part3, 1.0)
    miss_05 = _tau(part3, 0.5)
    execute_ratio = None
    if miss_1 is not None and miss_05 is not None and miss_05['mean_miss_mm'] > 0:
        execute_ratio = float(miss_1['mean_miss_mm'] / miss_05['mean_miss_mm'])

    factors = {}
    rate_indicated = cmd['wheel_rms_100ms_over_20ms'] > RATE_RATIO
    factors['rate'] = dict(
        prediction='higher_rate_lowers_miss' if rate_indicated else 'not_indicated',
        indicated=rate_indicated,
        rule=f'wheel RMS over 100 ms > {RATE_RATIO} x wheel RMS over 20 ms',
        value=cmd['wheel_rms_100ms_over_20ms'],
    )
    history_indicated = history_ratio is not None and history_ratio > HISTORY_RATIO
    factors['history'] = dict(
        prediction='stack_0.5s_lowers_miss' if history_indicated else 'not_indicated',
        indicated=history_indicated,
        rule=f'turn MSE > {HISTORY_RATIO} x drive MSE',
        value=history_ratio,
    )
    horizon_indicated = horizon_ratio > HORIZON_RATIO
    factors['predict_horizon'] = dict(
        prediction='shorter_predict_lowers_miss' if horizon_indicated else 'not_indicated',
        indicated=horizon_indicated,
        rule=f'open-loop MSE at the last step > {HORIZON_RATIO} x MSE at the first step',
        value=horizon_ratio,
    )
    execute_indicated = execute_ratio is not None and execute_ratio > EXECUTE_RATIO
    factors['execute_window'] = dict(
        prediction='shorter_execute_lowers_miss' if execute_indicated else 'not_indicated',
        indicated=execute_indicated,
        rule=f'hand-off miss at 1.0 s > {EXECUTE_RATIO} x hand-off miss at 0.5 s',
        value=execute_ratio,
    )
    short_indicated = cmd['short_chunk_median_ratio'] > SHORT_RATIO
    factors['short_chunk'] = dict(
        prediction='short_chunk_increases_miss' if short_indicated else 'not_indicated',
        indicated=short_indicated,
        rule=f'median expert wheel-command ratio of 0.5 s ahead vs 0.1 s ahead > {SHORT_RATIO}',
        value=cmd['short_chunk_median_ratio'],
    )
    return dict(
        thresholds=dict(rate=RATE_RATIO, horizon=HORIZON_RATIO, execute=EXECUTE_RATIO,
                        history=HISTORY_RATIO, short_chunk=SHORT_RATIO),
        factors=factors,
    )


def run(weights, handoff_seeds=50, test_n=200, workers=16):
    STUDY.mkdir(parents=True, exist_ok=True)
    analysis_path = STUDY / 'baseline_analysis.json'
    hypothesis_path = STUDY / 'hypotheses.json'
    weights = Path(weights)
    if analysis_path.exists():
        saved = json.loads(analysis_path.read_text())
        if saved.get('weights') == str(weights) and 'part3' in saved and 'command_variation' in saved:
            print(f'reusing {analysis_path}', flush=True)
            hypo = hypotheses_from(saved)
            hypothesis_path.write_text(json.dumps(hypo, indent=2))
            _print(hypo)
            return hypo

    model = NumpyPolicy(weights)
    if model.cfg.encoder != 'flat' or model.cfg.features != 'rel_add':
        raise SystemExit(f'{weights.name} is {model.cfg.encoder} / {model.cfg.features}, expected flat rel_add')
    print(f'command variation on expert demos, policy {model.cfg.name}', flush=True)
    t0 = time.time()
    variation = command_variation()
    print(f"  100 ms / 20 ms wheel RMS {variation['wheel_rms_100ms_over_20ms']:.2f}  "
          f"short-chunk ratio {variation['short_chunk_median_ratio']:.2f}", flush=True)

    seeds = json.loads((DATA / 'seeds.json').read_text())
    test_seeds = seeds['test'][:test_n]
    handoff = seeds['test'][:handoff_seeds]
    print('Part 1 train…', flush=True)
    train = diagnose.part1_train(model, 300)
    print(f"  train MSE {train['all']['mse']:.4f}", flush=True)

    print('Part 1 test…', flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    expert_ee, expert_meta, test_rows = {}, {}, []
    with ProcessPoolExecutor(workers, initializer=diagnose.init_worker, initargs=(str(weights),)) as pool:
        futs = {pool.submit(diagnose.part1_test_job, int(seed)): seed for seed in test_seeds}
        for n, fut in enumerate(as_completed(futs), 1):
            seed = futs[fut]
            err, meta = fut.result()
            test_rows.append(err)
            expert_ee[seed] = meta.pop('ee')
            expert_meta[seed] = meta
            if n % 50 == 0:
                print(f'  {n}/{len(test_seeds)}', flush=True)
    test = diagnose.reduce_errors([r for r in test_rows if r])
    print(f"  test MSE {test['mse']:.4f}  turn {test['mse_turn']}  drive {test['mse_drive']}", flush=True)

    print('Parts 2–3…', flush=True)
    p23 = []
    with ProcessPoolExecutor(workers, initializer=diagnose.init_worker, initargs=(str(weights),)) as pool:
        futs = []
        for seed in handoff:
            ct = expert_meta[seed]['catch_time']
            if ct is None:
                ct = expert_meta[seed]['duration']
            futs.append(pool.submit(diagnose.part23_job, int(seed), ct, expert_ee[seed], diagnose.TAUS))
        for n, fut in enumerate(as_completed(futs), 1):
            p23.append(fut.result())
            if n % 10 == 0:
                print(f'  {n}/{len(futs)}', flush=True)
    part2 = diagnose.summarize_part2(p23)
    part3 = diagnose.summarize_part3(p23)
    report = dict(
        weights=str(weights), config=model.cfg.to_dict(),
        command_variation=variation, train=train, test=test, part2=part2, part3=part3,
        n_test=test_n, n_handoff=handoff_seeds, elapsed_s=time.time() - t0,
    )
    analysis_path.write_text(json.dumps(report, indent=2))
    hypo = hypotheses_from(report)
    hypothesis_path.write_text(json.dumps(hypo, indent=2))
    print(f'wrote {analysis_path} and {hypothesis_path} in {report["elapsed_s"]:.0f}s', flush=True)
    _print(hypo)
    return hypo


def _print(hypo):
    for name, row in hypo['factors'].items():
        print(f"  {name}: {row['prediction']}  (value={row['value']})", flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    run(args.weights, workers=args.workers)
