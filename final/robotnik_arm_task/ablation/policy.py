"""The MLP-trunk policy with an MSE chunk head (torch, for training)."""
import json

import numpy as np
import torch
from torch import nn

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
        frames = policy_frames(frames, self.cfg.drop_time)
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


def load_policy(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    model = ChunkPolicy(Config(**blob['config']))
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model


def export_numpy(pt_path, npz_path):
    """Write the weights the torch-free rollout policy needs."""
    blob = torch.load(pt_path, map_location='cpu', weights_only=False)
    arrays = {k.replace('.', '__'): v.numpy().astype(np.float32) for k, v in blob['state_dict'].items()}
    np.savez(npz_path, config=np.array(json.dumps(blob['config'])), **arrays)
    return blob['meta']
