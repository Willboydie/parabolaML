"""Print base position and wheel commands over one test episode for given models and the expert."""
import json
import sys
from pathlib import Path

from catch_env import CatchEnv, ExpertDriver, OBS_NAMES, rollout
from np_policy import NumpyPolicy, PolicyDriver

ROOT = Path(__file__).resolve().parent
BASE = OBS_NAMES.index('base_pos_x')
WHEEL_QD = OBS_NAMES.index('wheel_qd_0')
env = CatchEnv()
seed = json.loads((ROOT / 'data' / 'seeds.json').read_text())['test'][int(sys.argv[1])]
drivers = [('expert', ExpertDriver(), 10)]
for name in sys.argv[2:]:
    model = NumpyPolicy(ROOT / 'results' / 'models' / f'{name}.npz')
    drivers.append((name, PolicyDriver(model), model.cfg.rate_hz))
for name, driver, rate in drivers:
    out = rollout(env, driver, seed, rate, log=True)
    obs, cmd = out['arrays']['obs'], out['arrays']['cmd']
    for t in (0.5, 2, 5, 10):
        i = min(int(t * 1000), len(obs) - 1)
        print(f'{name} t={t:>4}s base_xy={obs[i, BASE:BASE + 2].round(2)} wheel_cmd={cmd[i, :4].round(2)} '
              f'wheel_qd={obs[i, WHEEL_QD:WHEEL_QD + 4].round(2)}')
    print(f"{name}: success={out['success']} min_dist={out['min_dist'] * 1000:.0f} mm")
