import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

"""Permutation invariance of the DeepSets encoder, and agreement with the numpy rollout.

Also hashes the three current one-ball checkpoints. Those hashes are the check that
the earlier three seeds really are different networks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ballset import BALL_NAMES, ROBOT_NAMES, ball_features, robot_features
from config import Config
from np_policy import NumpyPolicy
from policy import DeepSetsPolicy, export_numpy

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results' / 'stage2'
CFG = Config(rate_hz=25, predict_s=2.0, execute_frac=0.5, history='stack', history_s=0.5,
             demos=2000, encoder='deepsets', pool='sum')


def feature_check():
    obs = np.zeros((1, 42), np.float32)
    obs[0, 1:4] = (1.0, 2.0, 0.5)     # ball
    obs[0, 4:7] = (0.3, -0.1, 0.2)    # ball vel
    obs[0, 10:13] = (0.1, 0.2, 0.3)   # base
    obs[0, 14] = 1.0                  # yaw 0, forward is +x
    obs[0, 39:42] = (0.4, 0.5, 0.6)   # ee
    ball = ball_features(obs)[0]
    robot = robot_features(obs)[0]
    assert ball.shape == (22,) and robot.shape == (33,)
    assert abs(ball[0] - 1.0) < 1e-6 and abs(ball[9] - 0.9) < 1e-5
    assert abs(ball[10] - 1.8) < 1e-5 and abs(ball[11] - 0.2) < 1e-5
    assert abs(ball[18] - 0.6) < 1e-5 and abs(ball[19] - 1.5) < 1e-5
    assert abs(ball[21]) < 1e-8
    assert abs(robot[1] - 0.1) < 1e-6 and abs(robot[30] - 0.4) < 1e-6
    flagged = ball_features(obs, oracle=1.0)[0]
    assert abs(flagged[21] - 1.0) < 1e-6


def permutation_check(model):
    model.eval()
    torch.manual_seed(0)
    robot = torch.randn(8, CFG.frames, 33)
    balls = torch.randn(8, 2, CFG.frames, 22)
    with torch.no_grad():
        a = model.forward_set(robot, balls)
        b = model.forward_set(robot, balls.flip(1))
    diff = float((a - b).abs().max())
    return diff


def numpy_check(model):
    obs = np.random.default_rng(1).normal(size=(CFG.frames, 42)).astype(np.float32)
    obs[:, 14] = 1.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    pt = RESULTS / 'permutation_init.pt'
    npz = RESULTS / 'permutation_init.npz'
    torch.save(dict(config=CFG.to_dict(), state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    meta={}), pt)
    export_numpy(pt, npz)
    with torch.no_grad():
        torch_out = model.predict(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
    numpy_out = NumpyPolicy(npz).predict(obs)
    diff = float(np.max(np.abs(torch_out - numpy_out)))
    return diff


def weight_hash(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    digest = hashlib.sha256()
    for key, value in sorted(blob['state_dict'].items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.numpy()).tobytes())
    return digest.hexdigest()


def main():
    feature_check()
    model = DeepSetsPolicy(CFG)
    # Identity normalisation, so the test is about the sum and not the statistics.
    model.ball_std.fill_(1.0)
    model.robot_std.fill_(1.0)
    perm = permutation_check(model)
    numpy_diff = numpy_check(model)
    current = {}
    for seed in (0, 1, 2):
        path = ROOT / 'results' / 'models' / f'r25_p2_e0.5_hstack0.5s_d2000_rel_add_s{seed}.pt'
        current[f's{seed}'] = weight_hash(path)
    distinct = len(set(current.values())) == 3
    report = dict(
        phi='Linear(286, 256), ReLU, Linear(256, 256), ReLU, Linear(256, 128)',
        pool='sum', embed_dim=128, frames=CFG.frames, ball_dim=22, robot_dim=33,
        ball_names=list(BALL_NAMES), robot_names=list(ROBOT_NAMES),
        permutation_max_abs_diff=perm,
        numpy_max_abs_diff=numpy_diff,
        current_oneball_hashes=current,
        current_oneball_hashes_distinct=distinct,
    )
    out = RESULTS / 'deepsets_unit.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'permutation max abs diff {perm:.3e}', flush=True)
    print(f'numpy max abs diff {numpy_diff:.3e}', flush=True)
    print(f'current checkpoints distinct: {distinct}', flush=True)
    for key, value in current.items():
        print(f'  {key} {value[:16]}', flush=True)
    if perm > 1e-4 or numpy_diff > 1e-4 or not distinct:
        raise SystemExit('deepsets check failed')
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
