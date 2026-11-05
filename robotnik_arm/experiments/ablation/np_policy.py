"""Torch-free forward pass and rollout driver for exported ChunkPolicy weights.

Rollout workers use this instead of torch: the CUDA torch build commits about 4 GB
of memory per process on Windows, which 16 workers cannot afford.
"""
import json

import numpy as np

from ballset import ball_features, robot_features
from config import Config, policy_frames

INPUT_CLIP = 10.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.cfg = Config(**json.loads(str(z['config'])))
            w = {k: z[k] for k in z.files if k != 'config'}
        self.act_mean, self.act_std = w['act_mean'], w['act_std']
        idx = sorted({int(k.split('__')[1]) for k in w if k.startswith('trunk__') and k.endswith('__weight')})
        self.trunk = [(w[f'trunk__{i}__weight'].T.copy(), w[f'trunk__{i}__bias']) for i in idx]
        self.head = (w['head__weight'].T.copy(), w['head__bias'])
        self.phi = None
        if self.cfg.encoder == 'deepsets':
            self.ball_mean, self.ball_std = w['ball_mean'], w['ball_std']
            self.robot_mean, self.robot_std = w['robot_mean'], w['robot_std']
            pidx = sorted(int(k.split('__')[1]) for k in w if k.startswith('phi__') and k.endswith('__weight'))
            self.phi = [(w[f'phi__{i}__weight'].T.copy(), w[f'phi__{i}__bias']) for i in pidx]
        else:
            self.obs_mean, self.obs_std = w['obs_mean'], w['obs_std']
        self.gru = None
        if 'gru__weight_ih_l0' in w:
            self.gru = (w['gru__weight_ih_l0'].T.copy(), w['gru__weight_hh_l0'].T.copy(),
                        w['gru__bias_ih_l0'], w['gru__bias_hh_l0'])

    def _mlp(self, x, layers, relu_last):
        for i, (weight, bias) in enumerate(layers):
            x = x @ weight + bias
            if relu_last or i < len(layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def _predict_deepsets(self, frames):
        """frames: (F, 42) one-ball observations, oldest first."""
        ball = ball_features(frames)
        robot = robot_features(frames)
        ball = np.clip((ball - self.ball_mean) / self.ball_std, -INPUT_CLIP, INPUT_CLIP)
        robot = np.clip((robot - self.robot_mean) / self.robot_std, -INPUT_CLIP, INPUT_CLIP)
        emb = self._mlp(ball.reshape(1, -1), self.phi, relu_last=False)[0]
        z = np.concatenate((robot.reshape(-1), emb))
        z = self._mlp(z, self.trunk, relu_last=True)
        out = (z @ self.head[0] + self.head[1]).reshape(self.cfg.horizon, -1)
        return out * self.act_std + self.act_mean

    def predict(self, frames):
        """frames: (F, obs_dim) raw observations, oldest first. Returns (H, act_dim) commands."""
        if self.phi is not None:
            return self._predict_deepsets(frames)
        frames = policy_frames(frames, self.cfg)
        x = np.clip((frames.astype(np.float32) - self.obs_mean) / self.obs_std, -INPUT_CLIP, INPUT_CLIP)
        if self.gru is not None:
            w_ih, w_hh, b_ih, b_hh = self.gru
            n_h = w_hh.shape[0]
            h = np.zeros(n_h, dtype=np.float32)
            gi_all = x @ w_ih + b_ih
            for gi in gi_all:
                gh = h @ w_hh + b_hh
                r = _sigmoid(gi[:n_h] + gh[:n_h])
                u = _sigmoid(gi[n_h:2 * n_h] + gh[n_h:2 * n_h])
                n = np.tanh(gi[2 * n_h:] + r * gh[2 * n_h:])
                h = (1.0 - u) * n + u * h
            z = np.concatenate([x[-1], h])
        else:
            z = x.reshape(-1)
        for weight, bias in self.trunk:
            z = np.maximum(z @ weight + bias, 0.0)
        out = (z @ self.head[0] + self.head[1]).reshape(self.cfg.horizon, -1)
        return out * self.act_std + self.act_mean

    def predict_batch(self, frames):
        """frames: (B, F, obs_dim). Returns (B, H, act_dim). Stacked-history path only."""
        frames = policy_frames(frames, self.cfg)
        x = np.clip((frames.astype(np.float32) - self.obs_mean) / self.obs_std, -INPUT_CLIP, INPUT_CLIP)
        if self.gru is not None:
            return np.stack([self.predict(f) for f in x])
        z = x.reshape(len(x), -1)
        for weight, bias in self.trunk:
            z = np.maximum(z @ weight + bias, 0.0)
        out = (z @ self.head[0] + self.head[1]).reshape(len(x), self.cfg.horizon, -1)
        return out * self.act_std + self.act_mean


class PolicyDriver:
    """Queries the policy when the executed part of the chunk is used up.

    History frames are the observations at previous policy ticks, padded with the first
    tick's observation at episode start, exactly as in training.
    """

    def __init__(self, model):
        self.model, self.cfg = model, model.cfg

    def reset(self, env):
        self.frames, self.chunk, self.index, self.q_ref = [], None, 0, None

    def act(self, env, t, dt_ctrl):
        self.frames.append(env.obs(t))
        cfg = self.cfg
        if self.chunk is None or self.index >= cfg.execute:
            n = cfg.frames
            hist = self.frames[-n:]
            hist = [hist[0]] * (n - len(hist)) + hist
            self.chunk = self.model.predict(np.stack(hist)).astype(np.float64)
            self.q_ref = env.q[4:11].copy()
            self.index = 0
        c = self.chunk[self.index]
        self.index += 1
        if len(self.frames) > 64:
            del self.frames[:-cfg.frames]
        return np.concatenate((c[:4], self.q_ref + c[4:11], c[11:18])), True
