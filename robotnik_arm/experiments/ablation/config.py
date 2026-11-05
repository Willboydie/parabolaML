"""Ablation config. Kept free of torch so rollout workers can import it cheaply."""
import math
from dataclasses import dataclass, asdict

GRID_HZ = 50
OBS_DIM = 42


def policy_frames(frames, drop_time):
    """Keep env/dataset observations 42-D; drop elapsed time only at the policy input."""
    if drop_time and frames.shape[-1] == OBS_DIM:
        return frames[..., 1:]
    return frames


@dataclass(frozen=True)
class Config:
    rate_hz: int = 10
    predict_s: float = 2.0
    execute_frac: float = 0.5
    history: str = 'none'        # none | stack | gru
    history_s: float = 0.0
    demos: int = 1000
    drop_time: bool = False      # drop elapsed-time from each observation frame

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
        return 41 if self.drop_time else 42

    @property
    def name(self):
        hist = 'none' if self.history == 'none' else f'{self.history}{self.history_s:g}s'
        extra = '_notime' if self.drop_time else ''
        return f'r{self.rate_hz}_p{self.predict_s:g}_e{self.execute_frac:g}_h{hist}_d{self.demos}{extra}'

    def to_dict(self):
        return asdict(self)


BASELINE = Config()
