"""Ablation config. Kept free of torch so rollout workers can import it cheaply."""
import math
from dataclasses import dataclass, asdict

import numpy as np

GRID_HZ = 50
OBS_DIM = 42


def _cat(parts):
    if isinstance(parts[0], np.ndarray):
        return np.concatenate(parts, axis=-1)
    import torch
    return torch.cat(parts, dim=-1)


def _stack(parts):
    if isinstance(parts[0], np.ndarray):
        return np.stack(parts, axis=-1)
    import torch
    return torch.stack(parts, dim=-1)


def _rotate_yaw(obs, vec):
    """Express a world vector in the base heading frame: forward, left, up."""
    sin_y, cos_y = obs[..., 13], obs[..., 14]
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return _stack((cos_y * x + sin_y * y, -sin_y * x + cos_y * y, z))


def relative_ball(obs):
    """Ball position, velocity and acceleration in the base frame, then ball minus end effector.

    Position is relative to the base origin. Velocity and acceleration are the world
    vectors written in the base axes. All 12 values are forward, left, up.
    """
    pos = _rotate_yaw(obs, obs[..., 1:4] - obs[..., 10:13])
    vel = _rotate_yaw(obs, obs[..., 4:7])
    acc = _rotate_yaw(obs, obs[..., 7:10])
    rel_ee = _rotate_yaw(obs, obs[..., 1:4] - obs[..., 39:42])
    return _cat((pos, vel, acc, rel_ee))


def policy_frames(frames, cfg):
    """Map a raw 42-D observation to the vector the network sees.

    ``world`` keeps the logged observation, dropping elapsed time when asked.
    ``rel_add`` appends the 12 base-frame ball features and keeps everything else.
    ``rel_only`` drops time, world ball kinematics, world base position and world
    end-effector position. It keeps heading, base velocity, arm state and wheel
    state, and uses the base-frame ball features for the geometry.
    """
    if frames.shape[-1] != OBS_DIM:
        return frames
    if cfg.features == 'world':
        return frames[..., 1:] if cfg.drop_time else frames
    rel = relative_ball(frames)
    if cfg.features == 'rel_add':
        base = frames[..., 1:] if cfg.drop_time else frames
        return _cat((base, rel))
    # rel_only: bearing through wheel velocity (cols 13:39), then the relative ball.
    return _cat((frames[..., 13:39], rel))


@dataclass(frozen=True)
class Config:
    rate_hz: int = 10
    predict_s: float = 2.0
    execute_frac: float = 0.5
    history: str = 'none'        # none | stack | gru
    history_s: float = 0.0
    demos: int = 1000
    drop_time: bool = False      # drop elapsed-time from each observation frame
    features: str = 'world'      # world | rel_add | rel_only
    encoder: str = 'flat'        # flat | deepsets
    pool: str = 'sum'            # deepsets pooling; sum is order-invariant
    oracle: bool = False         # deepsets oracle flag is 1 on the chosen ball
    phi_width: int = 256
    phi_layers: int = 2
    embed_dim: int = 128

    @property
    def stride(self):
        """Policy period in 50 Hz grid steps."""
        s = GRID_HZ / self.rate_hz
        assert s == int(s), 'rate must divide 50 Hz'
        return int(s)

    @property
    def horizon(self):
        return max(1, int(round(self.predict_s * self.rate_hz)))

    @property
    def execute(self):
        return max(1, min(self.horizon, int(math.floor(self.execute_frac * self.horizon + 0.5))))

    @property
    def frames(self):
        return 1 if self.history == 'none' else 1 + max(1, int(round(self.history_s * self.rate_hz)))

    @property
    def obs_dim(self):
        if self.features == 'rel_add':
            return (41 if self.drop_time else 42) + 12
        if self.features == 'rel_only':
            return 38
        return 41 if self.drop_time else 42

    @property
    def name(self):
        hist = 'none' if self.history == 'none' else f'{self.history}{self.history_s:g}s'
        extra = '_notime' if self.drop_time else ''
        if self.features != 'world':
            extra += f'_{self.features}'
        if self.encoder != 'flat':
            extra += f'_{self.encoder}'
        if self.oracle:
            extra += '_oracle'
        return f'r{self.rate_hz}_p{self.predict_s:g}_e{self.execute_frac:g}_h{hist}_d{self.demos}{extra}'

    def to_dict(self):
        return asdict(self)


BASELINE = Config()
