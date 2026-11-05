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
        self.head = None
        self.mixture = None
        if self.cfg.head == 'mdn':
            self.mixture = dict(
                pi=(w['pi_head__weight'].T.copy(), w['pi_head__bias']),
                mu=(w['mu_head__weight'].T.copy(), w['mu_head__bias']),
                sigma=(w['sigma_head__weight'].T.copy(), w['sigma_head__bias']))
        else:
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

    def _mixture_from_embedding(self, z):
        """Trunk features to raw logits, softmax weights, and normalized means.

        ``mu`` is (K, H, act). The softmax subtracts the max logit, which does not
        change the weights. The raw logits are the values before that shift.
        """
        k = self.cfg.n_components
        horizon = self.cfg.horizon
        act = self.act_std.shape[-1]
        pi_w, pi_b = self.mixture['pi']
        logits = z @ pi_w + pi_b
        raw = np.array(logits, copy=True)
        logits = logits - logits.max()
        pi = np.exp(logits)
        pi = pi / pi.sum()
        mu_w, mu_b = self.mixture['mu']
        mu = (z @ mu_w + mu_b).reshape(k, horizon, act)
        return raw.astype(np.float32), pi.astype(np.float32), mu.astype(np.float32)

    def _encode_set(self, frames, oracle_slot):
        """frames (N, F, 42) to trunk features."""
        balls = ball_features(frames)
        if self.cfg.oracle:
            balls = np.array(balls, copy=True)
            balls[..., 21] = 0.0
            if oracle_slot is not None:
                balls[int(oracle_slot), :, 21] = 1.0
        balls = np.clip((balls - self.ball_mean) / self.ball_std, -INPUT_CLIP, INPUT_CLIP)
        robot = robot_features(frames[0])
        robot = np.clip((robot - self.robot_mean) / self.robot_std, -INPUT_CLIP, INPUT_CLIP)
        emb = np.zeros(self.cfg.embed_dim, dtype=np.float32)
        for ball in balls:
            emb = emb + self._mlp(ball.reshape(1, -1), self.phi, relu_last=False)[0]
        z = np.concatenate((robot.reshape(-1), emb))
        return self._mlp(z, self.trunk, relu_last=True)

    def mdn_components(self, frames, oracle_slot=None):
        """frames (N, F, 42). Returns ``pi`` (K,) and normalized means (K, H, act)."""
        if self.mixture is None:
            raise RuntimeError('mdn_components is the mixture head')
        _raw, pi, mu = self._mixture_from_embedding(self._encode_set(frames, oracle_slot))
        return pi, mu

    def mdn_stages(self, frames, oracle_slot=None):
        """frames (N, F, 42). Logits, weights, normalized means, un-standardised chunks.

        ``chunk`` is (K, H, act), the action ``MDNDriver`` would execute for every
        component: normalized mean times ``act_std``, plus ``act_mean``.
        """
        if self.mixture is None:
            raise RuntimeError('mdn_stages is the mixture head')
        raw, pi, mu = self._mixture_from_embedding(self._encode_set(frames, oracle_slot))
        chunk = (mu * self.act_std + self.act_mean).astype(np.float32)
        return raw, pi, mu, chunk

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

    def predict_set(self, frames, oracle_slot=None):
        """frames: (N, F, 42) raw per-ball observations, oldest frame first.

        Returns (H, act_dim). Sums one embedding per ball. When the checkpoint was
        trained with the oracle flag, ``oracle_slot`` is the ball whose flag is 1.
        """
        if self.phi is None:
            raise RuntimeError('predict_set is the DeepSets path')
        if self.mixture is not None:
            raise RuntimeError('a mixture checkpoint selects a component in MDNDriver')
        balls = ball_features(frames)
        if self.cfg.oracle:
            balls = np.array(balls, copy=True)
            balls[..., 21] = 0.0
            if oracle_slot is not None:
                balls[int(oracle_slot), :, 21] = 1.0
        balls = np.clip((balls - self.ball_mean) / self.ball_std, -INPUT_CLIP, INPUT_CLIP)
        robot = robot_features(frames[0])
        robot = np.clip((robot - self.robot_mean) / self.robot_std, -INPUT_CLIP, INPUT_CLIP)
        emb = np.zeros(self.cfg.embed_dim, dtype=np.float32)
        for ball in balls:
            emb = emb + self._mlp(ball.reshape(1, -1), self.phi, relu_last=False)[0]
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


class SetPolicyDriver:
    """PolicyDriver for a DeepSets checkpoint. Each stored frame is ``(N, 42)``.

    ``oracle_slot`` is ignored unless the checkpoint was trained with the oracle flag.
    History padding matches PolicyDriver: the first frame is repeated until the stack is full.
    """

    def __init__(self, model, oracle_slot=None):
        self.model, self.cfg, self.oracle_slot = model, model.cfg, oracle_slot

    def reset(self, env):
        self.frames, self.chunk, self.index, self.q_ref = [], None, 0, None

    def act(self, env, t, dt_ctrl):
        self.frames.append(env.obs_set(t))
        cfg = self.cfg
        if self.chunk is None or self.index >= cfg.execute:
            n = cfg.frames
            hist = self.frames[-n:]
            hist = [hist[0]] * (n - len(hist)) + hist
            stacked = np.stack(hist, axis=1)
            slot = self.oracle_slot if cfg.oracle else None
            self.chunk = self.model.predict_set(stacked, slot).astype(np.float64)
            self.q_ref = env.q[4:11].copy()
            self.index = 0
        c = self.chunk[self.index]
        self.index += 1
        if len(self.frames) > 64:
            del self.frames[:-cfg.frames]
        return np.concatenate((c[:4], self.q_ref + c[4:11], c[11:18])), True


def choose_component(pi, mu_norm, rule, previous, execute):
    """Pick one component. ``previous`` is the normalized chunk still being executed.

    ``weight`` takes the highest mixture weight. ``continue`` takes the component
    whose upcoming steps best match the unexecuted part of the chunk in progress,
    compared in normalized action units. The first query of an episode has no
    previous chunk, so both rules take the highest weight. Component indices are
    not kept: the same index need not mean the same ball at the next replan.
    """
    if rule == 'weight' or previous is None:
        return int(np.argmax(pi))
    if rule != 'continue':
        raise ValueError(f'unknown mixture rule {rule}')
    remain = previous[execute:]
    prefix = mu_norm[:, :len(remain)]
    dist = ((prefix - remain) ** 2).mean(axis=(1, 2))
    order = np.lexsort((-pi, dist))
    return int(order[0])


class MDNDriver:
    """Executes one mixture component for the same 1 s as the squared-error driver.

    Replans when that second is used up. No average across components, and no
    carry of a component index from one replan to the next.
    """

    def __init__(self, model, rule='weight', oracle_slot=None):
        self.model, self.cfg, self.rule, self.oracle_slot = model, model.cfg, rule, oracle_slot

    def reset(self, env):
        self.frames, self.chunk, self.prev_norm = [], None, None
        self.index, self.q_ref = 0, None
        self.choices, self.max_pi = [], []

    def act(self, env, t, dt_ctrl):
        self.frames.append(env.obs_set(t))
        cfg = self.cfg
        if self.chunk is None or self.index >= cfg.execute:
            n = cfg.frames
            hist = self.frames[-n:]
            hist = [hist[0]] * (n - len(hist)) + hist
            stacked = np.stack(hist, axis=1)
            slot = self.oracle_slot if cfg.oracle else None
            pi, mu_norm = self.model.mdn_components(stacked, slot)
            k = choose_component(pi, mu_norm, self.rule, self.prev_norm, cfg.execute)
            self.choices.append(int(k))
            self.max_pi.append(float(pi.max()))
            self.prev_norm = mu_norm[k]
            self.chunk = (mu_norm[k] * self.model.act_std + self.model.act_mean).astype(np.float64)
            self.q_ref = env.q[4:11].copy()
            self.index = 0
        c = self.chunk[self.index]
        self.index += 1
        if len(self.frames) > 64:
            del self.frames[:-cfg.frames]
        return np.concatenate((c[:4], self.q_ref + c[4:11], c[11:18])), True
