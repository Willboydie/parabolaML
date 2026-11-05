"""Per-ball features for the DeepSets encoder. Torch-free unless the caller passes tensors.

One ball's features at one frame are 22 numbers:

    0:3   world position
    3:6   world velocity
    6:9   world acceleration
    9:12  base-frame position, forward / left / up
    12:15 base-frame velocity
    15:18 base-frame acceleration
    18:21 ball minus end effector, base frame
    21    oracle flag, 0 or 1

The robot state at one frame is the other 33 observation columns: time, base pose
and velocity, arm joints and joint velocities, wheel velocities, end effector.
Velocity and acceleration in the base frame are world vectors written in the
base axes. The base's own velocity is not subtracted.
"""
from config import _cat, relative_ball

BALL_FEAT_DIM = 22
ROBOT_FEAT_DIM = 33

BALL_NAMES = (
    [f'ball_world_{k}{a}' for k in ('pos', 'vel', 'acc') for a in 'xyz']
    + [f'ball_base_{k}_{a}' for k in ('pos', 'vel', 'acc') for a in ('forward', 'left', 'up')]
    + [f'ball_ee_{a}' for a in ('forward', 'left', 'up')]
    + ['oracle']
)
ROBOT_NAMES = (
    ['time', 'base_x', 'base_y', 'base_z', 'bearing_sin', 'bearing_cos']
    + [f'base_linvel_{a}' for a in 'xyz'] + [f'base_angvel_{a}' for a in 'xyz']
    + [f'arm_q_{i}' for i in range(7)] + [f'arm_qd_{i}' for i in range(7)]
    + [f'wheel_qd_{i}' for i in range(4)] + [f'ee_{a}' for a in 'xyz']
)
assert len(BALL_NAMES) == BALL_FEAT_DIM
assert len(ROBOT_NAMES) == ROBOT_FEAT_DIM


def ball_features(obs, oracle=0.0):
    """Ball features from a raw 42-D observation. ``oracle`` is 0, 1, or a matching tensor."""
    rel = relative_ball(obs)
    world = obs[..., 1:10]
    flag = world[..., :1] * 0
    if isinstance(oracle, (int, float)):
        if oracle:
            flag = flag + float(oracle)
    else:
        flag = flag + oracle
    return _cat((world, rel, flag))


def robot_features(obs):
    """Robot state from a raw 42-D observation, with the ball columns removed."""
    return _cat((obs[..., 0:1], obs[..., 10:42]))
