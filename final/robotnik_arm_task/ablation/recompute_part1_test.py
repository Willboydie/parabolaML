"""Recompute Part 1 test errors with catch-truncated trajectories, matching training."""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def main():
    from diagnose import WEIGHTS, DATA, RESULTS, init_worker, part1_test_job, reduce_errors
    seeds = json.loads((DATA / 'seeds.json').read_text())['test'][:200]
    rows = []
    with ProcessPoolExecutor(16, initializer=init_worker, initargs=(str(WEIGHTS),)) as pool:
        futs = {pool.submit(part1_test_job, s): s for s in seeds}
        for n, fut in enumerate(as_completed(futs), 1):
            err, _meta = fut.result()
            rows.append(err)
            if n % 50 == 0:
                print(f'{n}/200', flush=True)
    test = reduce_errors(rows)
    report = json.loads((RESULTS / 'diagnose.json').read_text())
    report['test'] = test
    report['test_note'] = 'truncated at catch+0.1s to match training episode ends'
    (RESULTS / 'diagnose.json').write_text(json.dumps(report, indent=2))
    print(f"test MSE {test['mse']:.4f}  wheel {test['wheel_mean']:.3f}  joint {test['joint_mean']:.4f}")
    for b in test['before']:
        print(f"  {b['name']}: mse={b['mse']:.4f} wheel={b['wheel']:.3f} joint={b['joint']:.4f} n={b['n']}")


if __name__ == '__main__':
    main()

