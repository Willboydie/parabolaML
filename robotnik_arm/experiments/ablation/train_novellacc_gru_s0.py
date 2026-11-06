"""Train seed 0 of the position-only backbone with a GRU over the ball frames."""
import json
import os
import subprocess
import sys
from pathlib import Path

from config import Config

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

cfg = Config(
    rate_hz=50,
    predict_s=1.0,
    execute_frac=0.5,
    history='gru',
    history_s=0.24,
    demos=2000,
    encoder='deepsets',
    pool='sum',
    ball_vel_acc=False,
)
assert (cfg.horizon, cfg.execute, cfg.frames, cfg.name) == (
    50, 25, 13, 'r50_p1_e0.5_hgru0.24s_d2000_deepsets_novellacc',
), (cfg.horizon, cfg.execute, cfg.frames, cfg.name)

out = Path('results/models') / f'{cfg.name}_s0.pt'
out.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, 'train.py',
    '--config', json.dumps(cfg.to_dict()),
    '--seed', '0',
    '--out', str(out),
    '--steps', '20000',
    '--data', 'data/grid50_offset.npz',
]
raise SystemExit(subprocess.call(cmd))
