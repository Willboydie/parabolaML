"""The MLP-trunk policy with an MSE chunk head (torch, for training)."""
import json

import numpy as np
import torch
from torch import nn

from ballset import BALL_FEAT_DIM, ROBOT_FEAT_DIM, ball_features, robot_features
from config import BASELINE, Config, GRID_HZ, policy_frames  # noqa: F401

TRUNK_WIDTH = 512
TRUNK_DEPTH = 3
GRU_HIDDEN = 128
INPUT_CLIP = 10.0


class ChunkPolicy(nn.Module):
    def __init__(self, cfg, obs_dim=None, act_dim=18):
        super().__init__()
        obs_dim = cfg.obs_dim if obs_dim is None else obs_dim
        self.cfg, self.obs_dim, self.act_dim = cfg, obs_dim, act_dim
        if cfg.history == 'gru':
            self.gru = nn.GRU(obs_dim, GRU_HIDDEN, batch_first=True)
            trunk_in = obs_dim + GRU_HIDDEN
        else:
            self.gru = None
            trunk_in = obs_dim * cfg.frames
        layers, width = [], trunk_in
        for _ in range(TRUNK_DEPTH):
            layers += [nn.Linear(width, TRUNK_WIDTH), nn.ReLU()]
            width = TRUNK_WIDTH
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(TRUNK_WIDTH, cfg.horizon * act_dim)
        for name in ('obs_mean', 'obs_std'):
            self.register_buffer(name, torch.zeros(obs_dim))
        for name in ('act_mean', 'act_std'):
            self.register_buffer(name, torch.zeros(cfg.horizon, act_dim))

    def forward(self, frames):
        """frames: (B, F, obs_dim) raw observations, oldest first. Returns normalized (B, H, act_dim)."""
        frames = policy_frames(frames, self.cfg)
        x = ((frames - self.obs_mean) / self.obs_std).clamp(-INPUT_CLIP, INPUT_CLIP)
        if self.gru is not None:
            _, h = self.gru(x)
            z = torch.cat([x[:, -1], h[-1]], dim=1)
        else:
            z = x.flatten(1)
        return self.head(self.trunk(z)).view(-1, self.cfg.horizon, self.act_dim)

    @torch.no_grad()
    def predict(self, frames):
        return self(frames) * self.act_std + self.act_mean


class DeepSetsPolicy(nn.Module):
    """Shared per-ball MLP, summed, then the same 3 x 512 trunk as the flat policy.

    ``phi`` is two ReLU layers of width ``phi_width`` and a linear map to ``embed_dim``.
    The ball stack is flattened (oldest frame first) before ``phi``, so a 13-frame
    history is 286 inputs. Summing the embeddings makes the output invariant to ball
    order. The oracle flag is one input channel and stays 0 unless the config asks
    for it.
    """

    def __init__(self, cfg, act_dim=18):
        super().__init__()
        if cfg.encoder != 'deepsets' or cfg.pool != 'sum' or cfg.phi_layers != 2:
            raise ValueError('DeepSetsPolicy is the sum-pooled two-layer phi')
        self.cfg, self.act_dim = cfg, act_dim
        ball_in = cfg.frames * BALL_FEAT_DIM
        width = cfg.phi_width
        self.phi = nn.Sequential(
            nn.Linear(ball_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, cfg.embed_dim),
        )
        trunk_in = cfg.frames * ROBOT_FEAT_DIM + cfg.embed_dim
        layers, size = [], trunk_in
        for _ in range(TRUNK_DEPTH):
            layers += [nn.Linear(size, TRUNK_WIDTH), nn.ReLU()]
            size = TRUNK_WIDTH
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(TRUNK_WIDTH, cfg.horizon * act_dim)
        self.register_buffer('ball_mean', torch.zeros(BALL_FEAT_DIM))
        self.register_buffer('ball_std', torch.ones(BALL_FEAT_DIM))
        self.register_buffer('robot_mean', torch.zeros(ROBOT_FEAT_DIM))
        self.register_buffer('robot_std', torch.ones(ROBOT_FEAT_DIM))
        self.register_buffer('act_mean', torch.zeros(cfg.horizon, act_dim))
        self.register_buffer('act_std', torch.ones(cfg.horizon, act_dim))

    def forward_set(self, robot, balls):
        """robot (B, F, 33), balls (B, N, F, 22). Returns normalized (B, H, act)."""
        balls = ((balls - self.ball_mean) / self.ball_std).clamp(-INPUT_CLIP, INPUT_CLIP)
        robot = ((robot - self.robot_mean) / self.robot_std).clamp(-INPUT_CLIP, INPUT_CLIP)
        emb = self.phi(balls.reshape(balls.shape[0], balls.shape[1], -1)).sum(dim=1)
        z = torch.cat((robot.reshape(robot.shape[0], -1), emb), dim=1)
        return self.head(self.trunk(z)).view(-1, self.cfg.horizon, self.act_dim)

    def forward(self, frames):
        """frames: (B, F, 42) raw one-ball observations, oldest first."""
        return self.forward_set(robot_features(frames), ball_features(frames).unsqueeze(1))

    @torch.no_grad()
    def predict(self, frames):
        return self(frames) * self.act_std + self.act_mean


def load_policy(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    cfg = Config(**blob['config'])
    model = DeepSetsPolicy(cfg) if cfg.encoder == 'deepsets' else ChunkPolicy(cfg)
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model


def export_numpy(pt_path, npz_path):
    """Write the weights the torch-free rollout policy needs."""
    blob = torch.load(pt_path, map_location='cpu', weights_only=False)
    arrays = {k.replace('.', '__'): v.numpy().astype(np.float32) for k, v in blob['state_dict'].items()}
    np.savez(npz_path, config=np.array(json.dumps(blob['config'])), **arrays)
    return blob['meta']
