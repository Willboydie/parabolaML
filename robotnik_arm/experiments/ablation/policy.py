"""The MLP-trunk policy with an MSE chunk head, and the mixture-density head (torch, for training)."""
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ballset import ROBOT_FEAT_DIM, ball_dim, ball_features, robot_features
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
    With stacked history the ball frames are flattened before ``phi``. A 13-frame
    history is 286 inputs with velocity and acceleration, and 130 without them.
    With GRU history a 128-unit GRU reads each ball's frames in order. ``phi`` then
    sees that ball's latest frame concatenated with the final GRU state.
    Summing the embeddings makes the output invariant to ball order. The oracle
    flag is the last ball channel and stays 0 unless the config asks for it.
    The robot frames stay a flat stack in either case.
    """

    def __init__(self, cfg, act_dim=18):
        super().__init__()
        if cfg.encoder != 'deepsets' or cfg.pool != 'sum' or cfg.phi_layers != 2:
            raise ValueError('DeepSetsPolicy is the sum-pooled two-layer phi')
        self.cfg, self.act_dim = cfg, act_dim
        self.ball_dim = ball_dim(cfg.ball_vel_acc)
        if cfg.history == 'gru':
            self.ball_gru = nn.GRU(self.ball_dim, GRU_HIDDEN, batch_first=True)
            ball_in = self.ball_dim + GRU_HIDDEN
        elif cfg.history in ('stack', 'none'):
            self.ball_gru = None
            ball_in = cfg.frames * self.ball_dim
        else:
            raise ValueError(f'unknown history {cfg.history}')
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
        if cfg.head == 'mdn':
            if cfg.n_components < 2:
                raise ValueError('the mixture head needs at least two components')
            chunk = cfg.horizon * act_dim
            k = cfg.n_components
            self.pi_head = nn.Linear(TRUNK_WIDTH, k)
            self.mu_head = nn.Linear(TRUNK_WIDTH, k * chunk)
            self.sigma_head = nn.Linear(TRUNK_WIDTH, k * chunk)
            self._init_mixture()
        elif cfg.head == 'mse':
            self.head = nn.Linear(TRUNK_WIDTH, cfg.horizon * act_dim)
        else:
            raise ValueError(f'unknown head {cfg.head}')
        self.register_buffer('ball_mean', torch.zeros(self.ball_dim))
        self.register_buffer('ball_std', torch.ones(self.ball_dim))
        self.register_buffer('robot_mean', torch.zeros(ROBOT_FEAT_DIM))
        self.register_buffer('robot_std', torch.ones(ROBOT_FEAT_DIM))
        self.register_buffer('act_mean', torch.zeros(cfg.horizon, act_dim))
        self.register_buffer('act_std', torch.ones(cfg.horizon, act_dim))

    def _init_mixture(self):
        """Start from equal weights, unit normalized variance, and separated means.

        Identical components get identical gradients and never split. The mean bias
        is a different draw per component so the mixture begins multimodal.
        """
        width = self.pi_head.in_features
        scale = 0.01 / math.sqrt(width)
        nn.init.normal_(self.pi_head.weight, std=scale)
        nn.init.zeros_(self.pi_head.bias)
        nn.init.normal_(self.mu_head.weight, std=scale)
        nn.init.normal_(self.mu_head.bias, std=0.5)
        nn.init.zeros_(self.sigma_head.weight)
        # softplus(bias) + floor = 1, in the same normalized units as the MSE target.
        target = 1.0 - float(self.cfg.sigma_floor)
        nn.init.constant_(self.sigma_head.bias, math.log(math.expm1(target)))

    def _ball_embed(self, balls):
        """Normalized balls (B, N, F, D) to a summed embedding (B, embed)."""
        if self.ball_gru is None:
            return self.phi(balls.reshape(balls.shape[0], balls.shape[1], -1)).sum(dim=1)
        batch, n_balls, frames, dim = balls.shape
        seq = balls.reshape(batch * n_balls, frames, dim)
        _out, hidden = self.ball_gru(seq)
        latest = torch.cat((seq[:, -1], hidden[-1]), dim=-1)
        return self.phi(latest).view(batch, n_balls, -1).sum(dim=1)

    def _encode(self, robot, balls):
        """robot (B, F, 33), balls (B, N, F, D). Returns the trunk features (B, 512)."""
        balls = ((balls - self.ball_mean) / self.ball_std).clamp(-INPUT_CLIP, INPUT_CLIP)
        robot = ((robot - self.robot_mean) / self.robot_std).clamp(-INPUT_CLIP, INPUT_CLIP)
        emb = self._ball_embed(balls)
        z = torch.cat((robot.reshape(robot.shape[0], -1), emb), dim=1)
        return self.trunk(z)

    def forward_set(self, robot, balls):
        """robot (B, F, 33), balls (B, N, F, 22). Returns normalized (B, H, act)."""
        return self.head(self._encode(robot, balls)).view(-1, self.cfg.horizon, self.act_dim)

    def mixture(self, robot, balls):
        """Normalized mixture: log_pi (B, K), mu and sigma (B, K, H, act).

        ``sigma`` is softplus of the raw head plus ``sigma_floor``. The floor is in
        normalized action units, where a typical target has standard deviation 1.
        """
        encoded = self._encode(robot, balls)
        log_pi = torch.log_softmax(self.pi_head(encoded), dim=-1)
        k, horizon, act = self.cfg.n_components, self.cfg.horizon, self.act_dim
        mu = self.mu_head(encoded).view(-1, k, horizon, act)
        sigma = F.softplus(self.sigma_head(encoded)).view(-1, k, horizon, act) + self.cfg.sigma_floor
        return log_pi, mu, sigma

    def forward(self, frames):
        """frames: (B, F, 42) raw one-ball observations, oldest first."""
        balls = ball_features(frames, include_vel_acc=self.cfg.ball_vel_acc).unsqueeze(1)
        return self.forward_set(robot_features(frames), balls)

    @torch.no_grad()
    def predict(self, frames):
        return self(frames) * self.act_std + self.act_mean


def gaussian_mixture_nll(log_pi, mu, sigma, target):
    """Mean negative log-likelihood of a diagonal Gaussian mixture.

    ``log_pi`` is (B, K), ``mu`` and ``sigma`` are (B, K, H, act), ``target`` is
    (B, H, act). The sum inside each component is over the whole chunk, and the
    components are combined with logsumexp.
    """
    log_prob = -0.5 * (((target[:, None] - mu) / sigma) ** 2 + 2.0 * sigma.log() + math.log(2.0 * math.pi))
    log_prob = log_prob.sum(dim=(-1, -2))
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()


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
