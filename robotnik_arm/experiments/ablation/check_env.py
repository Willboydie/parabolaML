"""Compare the lean environment's 1 kHz expert with the recorded native episodes, and time it."""
import json
import time
from pathlib import Path

from catch_env import CatchEnv, ExpertDriver, rollout, TASK_DIR

manifest = json.loads((TASK_DIR / 'datasets' / 'one_ball' / 'manifest.json').read_text())
rows = [r for r in manifest if r['split'] == 'test'][:3]
t0 = time.time()
env = CatchEnv()
print(f'env init {time.time() - t0:.2f}s')
for rate in (1000, 10):
    for r in rows:
        t0 = time.time()
        out = rollout(env, ExpertDriver(), r['seed'], rate)
        wall = time.time() - t0
        print(f"rate {rate:4d} seed {r['seed']}: native t={r['final_time']:.3f} err={r['final_ee_error']:.4f} | "
              f"lean success={out['success']} t={out['catch_time']} min={out['min_dist']:.4f} "
              f"wall={wall:.2f}s ({1000 * wall / out['steps']:.3f} ms/step)")
