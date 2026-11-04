"""Check the NumPy forward pass against torch on real observations."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from np_policy import NumpyPolicy
from policy import export_numpy, load_policy

ROOT = Path(__file__).resolve().parent
with np.load(ROOT / 'data' / 'grid50.npz') as z:
    obs = z['obs'][:5000]
rng = np.random.default_rng(0)
for pt in sys.argv[1:]:
    ref = load_policy(pt)
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / 'p.npz'
        export_numpy(pt, npz)
        mine = NumpyPolicy(npz)
    worst = 0.0
    for _ in range(20):
        i = int(rng.integers(ref.cfg.frames, len(obs)))
        frames = obs[i - ref.cfg.frames + 1:i + 1]
        a = ref.predict(torch.from_numpy(frames[None]))[0].numpy()
        b = mine.predict(frames)
        worst = max(worst, float(np.max(np.abs(a - b) / (np.abs(a) + 1e-3))))
    print(f'{Path(pt).name}: max relative diff {worst:.2e}')
